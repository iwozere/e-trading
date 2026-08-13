"""
Integration-style tests for `runner.run_once`.

Holdings go through the real `pnl_alert` XML pipeline (a temp Flex Query
export, same fixture shape as `pnl_alert`'s own loader tests) — but the Flex
*download* step is patched out (see `_no_live_flex_download` below): this
dev environment has real `IBKR_FLEX_TOKEN`/`IBKR_FLEX_QUERY_ID` configured,
and an un-mocked download would silently overwrite the tmp_path fixture XML
with real account data before `load_ibkr_xml` reads it back. The earnings
source, open-orders feed, and notification client are fakes injected via
`run_once`'s parameters, and `now` is injected too so trigger-matching is
deterministic instead of racing the real clock.
"""

import asyncio
import textwrap
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List

import pytest

from src.portfolio.management.config import ManagementConfig
from src.portfolio.management.earnings_window import EarningsEvent, resolve_anchor_utc
from src.portfolio.management.runner import run_once


@pytest.fixture(autouse=True)
def _no_live_flex_download(monkeypatch):
    """Prevent every test in this module from hitting the real Flex Web Service."""
    monkeypatch.setattr("src.portfolio.management.runner.download_open_positions_xml", lambda *a, **k: None)

_XML = textwrap.dedent("""\
    <FlexQueryResponse queryName="Open Positions" type="AF">
    <FlexStatements count="1">
    <FlexStatement accountId="U123" fromDate="2026-08-19" toDate="2026-08-19">
    <OpenPositions>
    <OpenPosition symbol="AAA" position="100" markPrice="10.00"
        costBasisPrice="8.00" costBasisMoney="800.00" />
    <OpenPosition symbol="BBB" position="50" markPrice="20.00"
        costBasisPrice="18.00" costBasisMoney="900.00" />
    </OpenPositions>
    </FlexStatement>
    </FlexStatements>
    </FlexQueryResponse>
""")

_EMPTY_XML = textwrap.dedent("""\
    <FlexQueryResponse queryName="Open Positions" type="AF">
    <FlexStatements count="1">
    <FlexStatement accountId="U123">
    <OpenPositions>
    </OpenPositions>
    </FlexStatement>
    </FlexStatements>
    </FlexQueryResponse>
""")


class FakeEarningsSource:
    def __init__(self, events: List[EarningsEvent]):
        self._events = events

    def get_upcoming_events(self, tickers, as_of_date, window_days):
        del as_of_date, window_days
        wanted = set(tickers)
        return [e for e in self._events if e.ticker in wanted]


class FakeOpenOrdersFeed:
    """Stands in for `IBKROpenOrdersFeed`: same connect/fetch/disconnect shape."""

    def __init__(self, protective_qty: Dict[str, float], connect_ok: bool = True):
        self._protective_qty = protective_qty
        self._connect_ok = connect_ok
        self.connected = False
        self.disconnected = False

    def connect(self, attempts=2, backoff_seconds=3.0):
        del attempts, backoff_seconds
        self.connected = self._connect_ok
        return self._connect_ok

    def protective_order_qty(self, symbols):
        wanted = {s.upper() for s in symbols}
        return {k: v for k, v in self._protective_qty.items() if k in wanted}

    def disconnect(self):
        self.disconnected = True


class FakeNotificationClient:
    def __init__(self):
        self.calls = []

    async def send_notification(self, **kwargs):
        self.calls.append(kwargs)
        return True


def _cfg(xml_path: str) -> ManagementConfig:
    return ManagementConfig(ibkr_xml_path=xml_path, trigger_window_minutes=15, recipient_id=2)


def _write_xml(tmp_path, content: str = _XML) -> str:
    path = tmp_path / "Open_Positions.xml"
    path.write_text(content)
    return str(path)


async def _run(cfg, earnings_source, feed, client, now=None, as_of_date=None):
    return await run_once(
        cfg,
        as_of_date=as_of_date,
        now=now,
        earnings_source=earnings_source,
        open_orders_feed=feed,
        client=client,
    )


def test_no_triggers_sends_nothing(tmp_path):
    cfg = _cfg(_write_xml(tmp_path))
    # T-1day/T-1hour of this anchor are nowhere near `now` below.
    events = [EarningsEvent(ticker="AAA", earnings_date=date(2026, 8, 29), session="bmo")]
    now = datetime(2026, 8, 19, 13, 30, tzinfo=timezone.utc)
    client = FakeNotificationClient()

    summary = asyncio.run(_run(cfg, FakeEarningsSource(events), FakeOpenOrdersFeed({}), client, now=now))

    assert summary.holdings_count == 2
    assert summary.triggered_count == 0
    assert summary.notification_sent is False
    assert client.calls == []


def test_t_minus_1_day_trigger_uncovered_position_sends_notification(tmp_path):
    """AAA (held, uncovered) hits its T-1day trigger -> live orders checked, notification sent."""
    cfg = _cfg(_write_xml(tmp_path))
    event = EarningsEvent(ticker="AAA", earnings_date=date(2026, 8, 20), session="bmo")
    now = resolve_anchor_utc(event).replace(day=19)  # exactly T-1day: 2026-08-19 13:30 UTC
    feed = FakeOpenOrdersFeed({})  # AAA has no protective orders -> uncovered
    client = FakeNotificationClient()

    summary = asyncio.run(_run(cfg, FakeEarningsSource([event]), feed, client, now=now))

    assert summary.holdings_count == 2
    assert summary.earnings_events_count == 1
    assert summary.triggered_count == 1
    assert feed.connected is True
    assert feed.disconnected is True
    assert summary.notification_sent is True
    assert len(client.calls) == 1
    assert client.calls[0]["source"] == "portfolio.management"
    assert "AAA" in client.calls[0]["message"]
    assert "UNCOVERED" in client.calls[0]["message"]


def test_t_minus_1_hour_trigger_covered_position_still_notifies(tmp_path):
    """Coverage is fully protected, but the reminder still fires — it's informational, not filtered by status."""
    cfg = _cfg(_write_xml(tmp_path))
    event = EarningsEvent(ticker="BBB", earnings_date=date(2026, 8, 20), session="amc")
    now = resolve_anchor_utc(event) - timedelta(hours=1)
    feed = FakeOpenOrdersFeed({"BBB": 50.0})  # fully covers the 50-share BBB position
    client = FakeNotificationClient()

    summary = asyncio.run(_run(cfg, FakeEarningsSource([event]), feed, client, now=now))

    assert summary.triggered_count == 1
    assert summary.notification_sent is True
    assert "covered" in client.calls[0]["message"]


def test_ticker_not_held_is_ignored_even_with_earnings_event(tmp_path):
    """An earnings event for a ticker not currently held must not trigger."""
    cfg = _cfg(_write_xml(tmp_path))
    event = EarningsEvent(ticker="ZZZ", earnings_date=date(2026, 8, 20), session="bmo")
    now = resolve_anchor_utc(event).replace(day=19)
    client = FakeNotificationClient()

    summary = asyncio.run(_run(cfg, FakeEarningsSource([event]), FakeOpenOrdersFeed({}), client, now=now))

    assert summary.triggered_count == 0
    assert client.calls == []


def test_no_holdings_exits_early_without_earnings_lookup(tmp_path):
    cfg = _cfg(_write_xml(tmp_path, content=_EMPTY_XML))
    earnings_source = FakeEarningsSource([EarningsEvent(ticker="AAA", earnings_date=date(2026, 8, 20))])

    summary = asyncio.run(_run(cfg, earnings_source, FakeOpenOrdersFeed({}), FakeNotificationClient()))

    assert summary.holdings_count == 0
    assert summary.earnings_events_count == 0


def test_live_ibkr_unreachable_still_notifies_with_unknown_coverage(tmp_path):
    """When the live Gateway is unreachable, coverage defaults to uncovered (0 protective qty) and an error is recorded."""
    cfg = _cfg(_write_xml(tmp_path))
    event = EarningsEvent(ticker="AAA", earnings_date=date(2026, 8, 20), session="bmo")
    now = resolve_anchor_utc(event).replace(day=19)
    feed = FakeOpenOrdersFeed({"AAA": 100.0}, connect_ok=False)  # would be fully covered if reachable
    client = FakeNotificationClient()

    summary = asyncio.run(_run(cfg, FakeEarningsSource([event]), feed, client, now=now))

    assert "live_ibkr_unreachable" in summary.errors
    assert summary.notification_sent is True
    assert "UNCOVERED" in client.calls[0]["message"]
