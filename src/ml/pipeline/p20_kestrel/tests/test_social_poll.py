"""Tests for the P20 social poll (StockTwits volume estimate, ApeWisdom paging, failure handling)."""

from datetime import date, datetime, timedelta, timezone
from typing import Any, cast

import pytest
import requests
from curl_cffi import requests as curl_requests

from src.ml.pipeline.p20_kestrel.sentiment import social_poll as poll

_NOW = datetime(2026, 10, 14, 20, 0, tzinfo=timezone.utc)


class _FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json_data = json_data if json_data is not None else {}

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class _FakeSession:
    """Routes GETs to handler(url) → _FakeResponse; records the URLs."""

    def __init__(self, handler):
        self._handler = handler
        self.urls = []
        self.headers = {}

    def get(self, url, timeout=None):
        self.urls.append(url)
        return self._handler(url)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _as_session(fake: _FakeSession) -> curl_requests.Session:
    return cast(curl_requests.Session, cast(Any, fake))


def _messages(*hours_ago, sentiment=None):
    return [
        {"created_at": (_NOW - timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ"),
         "entities": {"sentiment": {"basic": sentiment} if sentiment else None}}
        for h in hours_ago
    ]


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(poll.time, "sleep", lambda s: None)


# ---------------------------------------------------------------------------
# StockTwits volume estimate
# ---------------------------------------------------------------------------


def test_daily_mentions_counts_last_24h_when_page_reaches_past_window():
    assert poll._daily_mentions(_messages(1, 5, 23, 30, 200), _NOW) == 3.0


def test_daily_mentions_quiet_ticker_is_zero_not_page_size():
    """LXEO shape: 30 messages spanning two months, none today — the old code reported 30."""
    assert poll._daily_mentions(_messages(*range(26, 26 + 30 * 48, 48)), _NOW) == 0.0


def test_daily_mentions_extrapolates_saturated_page():
    """AAPL shape: 30 messages in the last 2 hours → ~360/day, not 30."""
    assert poll._daily_mentions(_messages(*[i / 15 for i in range(30)]), _NOW) == pytest.approx(30 * 24 / (29 / 15), 0.01)
    assert poll._daily_mentions([], _NOW) == 0.0


def test_fetch_stocktwits_computes_ratio(monkeypatch):
    monkeypatch.setattr(poll, "datetime", type("_DT", (), {"now": staticmethod(lambda tz=None: _NOW),
                                                           "strptime": datetime.strptime}))
    msgs = (_messages(1, 2, sentiment="Bullish") + _messages(3, sentiment="Bearish")
            + _messages(30, 40, sentiment="Bearish"))  # outside the 24h window: ignored for the ratio too
    session = _FakeSession(lambda url: _FakeResponse(200, {"messages": msgs}))

    assert poll._fetch_stocktwits(_as_session(session), "AAPL") == {"mentions": 3.0, "bullish_ratio": 2 / 3}

    stale = _FakeSession(lambda url: _FakeResponse(200, {"messages": _messages(30, 40, sentiment="Bullish")}))
    assert poll._fetch_stocktwits(_as_session(stale), "LXEO") == {"mentions": 0.0, "bullish_ratio": None}


@pytest.mark.parametrize("status", [403, 429, 503])
def test_fetch_stocktwits_raises_when_blocked(status):
    with pytest.raises(poll._StockTwitsUnavailable):
        poll._fetch_stocktwits(_as_session(_FakeSession(lambda url: _FakeResponse(status))), "AAPL")


def test_fetch_stocktwits_unknown_symbol_is_per_ticker_miss():
    assert poll._fetch_stocktwits(_as_session(_FakeSession(lambda url: _FakeResponse(404))), "ZZZZ") is None


# ---------------------------------------------------------------------------
# ApeWisdom
# ---------------------------------------------------------------------------


def test_fetch_apewisdom_reads_all_pages(monkeypatch):
    pages = {
        1: {"pages": 2, "results": [{"ticker": "spy", "mentions": 310}]},
        2: {"pages": 2, "results": [{"ticker": "BETR", "mentions": 1}]},
    }
    monkeypatch.setattr(poll.requests, "get", lambda url, timeout: _FakeResponse(200, pages[int(url.rsplit("/", 1)[1])]))

    assert poll._fetch_apewisdom() == {"SPY": 310, "BETR": 1}


def test_fetch_apewisdom_partial_table_is_unavailable(monkeypatch):
    responses = iter([_FakeResponse(200, {"pages": 2, "results": [{"ticker": "SPY", "mentions": 3}]}), _FakeResponse(502)])
    monkeypatch.setattr(poll.requests, "get", lambda url, timeout: next(responses))

    assert poll._fetch_apewisdom() is None


# ---------------------------------------------------------------------------
# Reddit credentials
# ---------------------------------------------------------------------------


def test_reddit_placeholder_credentials_skip_auth(monkeypatch):
    monkeypatch.setenv("REDDIT_API_KEY", "xxx")
    monkeypatch.setenv("REDDIT_API_SECRET", "xxx")
    monkeypatch.setattr(poll.requests, "post", lambda *a, **kw: pytest.fail("auth attempted with placeholder creds"))

    assert poll._get_reddit_headers() is None


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


@pytest.fixture
def _db(monkeypatch):
    state = {"upserted": [], "finished": []}
    monkeypatch.setattr(poll, "start_job_run", lambda job, d: None)
    monkeypatch.setattr(poll, "finish_job_run", lambda job, d, **kw: state["finished"].append(kw))
    monkeypatch.setattr(poll, "get_watchlist_tickers", lambda: ["AAPL", "LXEO", "QUIET"])
    monkeypatch.setattr(poll, "upsert_sentiment", lambda rows: state["upserted"].extend(rows) or len(rows))
    monkeypatch.setattr(poll, "_get_reddit_headers", lambda: None)
    return state


def test_run_writes_zero_apewisdom_rows_for_unlisted_tickers(_db, monkeypatch):
    monkeypatch.setattr(poll, "_fetch_apewisdom", lambda: {"AAPL": 12})
    monkeypatch.setattr(poll.curl_requests, "Session", lambda impersonate: _FakeSession(lambda url: _FakeResponse(200, {"messages": []})))

    result = poll.run(date(2026, 10, 14))

    ape = {r["ticker"]: r["mentions"] for r in _db["upserted"] if r["source"] == "apewisdom"}
    assert ape == {"AAPL": 12, "LXEO": 0, "QUIET": 0}
    assert result["stocktwits_ok"] == 3 and result["apewisdom_mentioned"] == 1
    assert _db["finished"][-1]["status"] == "ok"


def test_run_skips_apewisdom_rows_when_table_unavailable(_db, monkeypatch):
    monkeypatch.setattr(poll, "_fetch_apewisdom", lambda: None)
    monkeypatch.setattr(poll.curl_requests, "Session", lambda impersonate: _FakeSession(lambda url: _FakeResponse(200, {"messages": []})))

    poll.run(date(2026, 10, 14))

    assert not [r for r in _db["upserted"] if r["source"] == "apewisdom"]


def test_run_stops_stocktwits_when_blocked(_db, monkeypatch):
    monkeypatch.setattr(poll, "_STOCKTWITS_BLOCK_THRESHOLD", 2)
    monkeypatch.setattr(poll, "_fetch_apewisdom", lambda: {})
    session = _FakeSession(lambda url: _FakeResponse(403))
    monkeypatch.setattr(poll.curl_requests, "Session", lambda impersonate: session)

    result = poll.run(date(2026, 10, 14))

    assert len(session.urls) == 2  # third ticker never requested
    assert result["stocktwits_ok"] == 0
    assert len([r for r in _db["upserted"] if r["source"] == "apewisdom"]) == 3  # other sources unaffected


def test_run_stops_stocktwits_on_first_rate_limit(_db, monkeypatch):
    monkeypatch.setattr(poll, "_fetch_apewisdom", lambda: {})
    session = _FakeSession(lambda url: _FakeResponse(429))
    monkeypatch.setattr(poll.curl_requests, "Session", lambda impersonate: session)

    poll.run(date(2026, 10, 14))

    assert len(session.urls) == 1


def test_rotated_start_moves_with_date():
    tickers = ["A", "B", "C"]
    first = poll._rotated(tickers, date(2026, 10, 14))
    second = poll._rotated(tickers, date(2026, 10, 15))

    assert sorted(first) == tickers and first != second
    assert poll._rotated([], date(2026, 10, 14)) == []
