"""Unit tests for `runner.run_once`."""

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd

from src.portfolio.pnl_alert.config import PnLAlertConfig
from src.portfolio.pnl_alert.runner import run_once


def _fake_broker(symbol: str = "NVDA", qty: float = 10.0, avg_cost: float = 120.0) -> SimpleNamespace:
    position = SimpleNamespace(
        contract=SimpleNamespace(symbol=symbol, secType="STK", localSymbol=symbol),
        position=qty,
        avgCost=avg_cost,
    )
    return SimpleNamespace(ib=SimpleNamespace(positions=lambda: [position]))


class _FakeDataManager:
    def __init__(self, close: float = 130.0):
        self._close = close

    def get_ohlcv(self, _symbol: str, _interval: str, _start: Any, _end: Any) -> pd.DataFrame:
        return pd.DataFrame({"close": [self._close]})


def _cfg(threshold_pct: float = 0.10) -> PnLAlertConfig:
    return PnLAlertConfig(
        threshold_pct=threshold_pct,
        channels=["telegram"],
        ibkr_xml_path="",
        include_ibkr=True,
        ibkr_stk_only=True,
        recipient_id=1,
    )


def _fake_client(ok: bool = True) -> MagicMock:
    client = MagicMock()
    client.send_notification = AsyncMock(return_value=ok)
    return client


def test_digest_sent_even_when_nothing_flagged():
    """The digest goes out even when no position clears the threshold."""
    edgar = MagicMock()
    edgar.download_form4_filings.return_value = pd.DataFrame()
    client = _fake_client()

    summary = asyncio.run(
        run_once(
            _cfg(threshold_pct=0.50),  # NVDA's +8.3% below this threshold
            broker=_fake_broker(),
            data_manager=_FakeDataManager(close=130.0),
            client=client,
            edgar=edgar,
        )
    )

    assert summary.digest_row_count == 1
    assert summary.flagged_row_count == 0
    assert summary.notification_sent is True
    assert summary.errors == []


def test_insider_activity_failure_does_not_block_digest():
    """
    An unexpected failure in the insider-activity lookup itself (as opposed to
    a single day's cache read, which `load_insider_activity` already degrades
    gracefully from — see test_insider_activity.py) is caught by runner.py's
    own best-effort wrapper; the PnL digest still goes out.
    """
    client = _fake_client()

    with patch(
        "src.portfolio.pnl_alert.runner.load_insider_activity",
        side_effect=Exception("unexpected failure"),
    ):
        summary = asyncio.run(
            run_once(
                _cfg(threshold_pct=0.05),
                broker=_fake_broker(),
                data_manager=_FakeDataManager(close=130.0),
                client=client,
            )
        )

    assert summary.notification_sent is True
    assert "insider_activity_failed" in summary.errors


def test_no_holdings_exits_early_without_sending():
    """No IBKR positions and no XML source means nothing to evaluate."""
    cfg = PnLAlertConfig(
        threshold_pct=0.10,
        channels=["telegram"],
        ibkr_xml_path="",
        include_ibkr=False,
        recipient_id=1,
    )
    client = _fake_client()

    summary = asyncio.run(run_once(cfg, client=client))

    assert summary.holdings_count == 0
    assert summary.notification_sent is False
    client.send_notification.assert_not_called()


def test_flagged_row_counted_and_sent():
    """A position clearing the threshold is included and counted as flagged."""
    edgar = MagicMock()
    edgar.download_form4_filings.return_value = pd.DataFrame()
    client = _fake_client()

    summary = asyncio.run(
        run_once(
            _cfg(threshold_pct=0.05),  # NVDA's +8.3% clears this
            broker=_fake_broker(),
            data_manager=_FakeDataManager(close=130.0),
            client=client,
            edgar=edgar,
        )
    )

    assert summary.digest_row_count == 1
    assert summary.flagged_row_count == 1
    assert summary.notification_sent is True
