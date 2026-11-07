"""Tests for the Sleeve B1 catalyst calendar sync -- combined countdown digest."""

from datetime import date

import pytest

from src.ml.pipeline.p20_kestrel.ingest import calendar_sync, finnhub_client
from src.ml.pipeline.p20_kestrel.ingest.calendar_sync import run

_TODAY = date(2026, 10, 13)


@pytest.fixture
def run_env(monkeypatch):
    monkeypatch.setattr(calendar_sync, "start_job_run", lambda *_a, **_kw: None)
    monkeypatch.setattr(calendar_sync, "finish_job_run", lambda *_a, **_kw: None)
    monkeypatch.setattr(calendar_sync, "get_watchlist_tickers", lambda: [])
    monkeypatch.setattr(calendar_sync, "log_alert", lambda **_kw: None)
    monkeypatch.setattr(calendar_sync, "mark_past_catalysts_passed", lambda _today: 0)

    stamped: list[tuple] = []
    monkeypatch.setattr(calendar_sync, "stamp_catalyst_alert", lambda cid, col: stamped.append((cid, col)))

    sent: list[dict] = []

    def _fake_send_push(title, message):
        sent.append({"title": title, "message": message})
        return True

    monkeypatch.setattr(calendar_sync, "send_push", _fake_send_push)

    return stamped, sent


def _catalyst(cid, ticker, event_type, event_date, t10=None, t3=None):
    return {
        "id": cid,
        "ticker": ticker,
        "event_type": event_type,
        "event_date": event_date,
        "state": "upcoming",
        "t10_alerted_at": t10,
        "t3_alerted_at": t3,
    }


def test_run_combines_multiple_catalyst_alerts_into_one_digest(monkeypatch, run_env):
    """Each catalyst used to fire its own push/email; now one run == at most one send_push call."""
    _, sent = run_env
    upcoming = [
        _catalyst(1, "INCY", "clinical_readout", "2026-10-15"),  # T-10 and T-3 both due
        _catalyst(2, "SRPT", "pdufa", "2026-10-16"),  # T-10 and T-3 both due
        _catalyst(3, "ABBV", "adcom", "2026-10-20"),  # only T-10 due (7 days out)
    ]
    monkeypatch.setattr(calendar_sync, "get_all_upcoming_catalysts", lambda days_ahead: upcoming)

    result = run(as_of_date=_TODAY)

    assert result["countdown_checks"] == 5
    assert len(sent) == 1
    body = sent[0]["message"]
    assert "INCY clinical_readout on 2026-10-15" in body
    assert "SRPT pdufa on 2026-10-16" in body
    assert "ABBV adcom on 2026-10-20" in body


def test_digest_includes_legend_only_for_event_types_present(monkeypatch, run_env):
    _, sent = run_env
    upcoming = [_catalyst(1, "INCY", "clinical_readout", "2026-10-15")]
    monkeypatch.setattr(calendar_sync, "get_all_upcoming_catalysts", lambda days_ahead: upcoming)

    run(as_of_date=_TODAY)

    body = sent[0]["message"]
    assert "clinical_readout:" in body
    assert "pdufa:" not in body


def test_no_digest_sent_when_nothing_fires(monkeypatch, run_env):
    _, sent = run_env
    already_alerted = [_catalyst(1, "INCY", "clinical_readout", "2026-10-15", t10="2026-10-05", t3="2026-10-10")]
    monkeypatch.setattr(calendar_sync, "get_all_upcoming_catalysts", lambda days_ahead: already_alerted)

    result = run(as_of_date=_TODAY)

    assert result["countdown_checks"] == 0
    assert sent == []


# ── Retiring past and moved catalysts ─────────────────────────────────────────


def test_run_marks_past_catalysts_passed(monkeypatch, run_env):
    seen: list = []
    monkeypatch.setattr(calendar_sync, "mark_past_catalysts_passed", lambda today: seen.append(today) or 7)
    monkeypatch.setattr(calendar_sync, "get_all_upcoming_catalysts", lambda days_ahead: [])
    assert run(as_of_date=_TODAY)["catalysts_passed"] == 7
    assert seen == [_TODAY]


def test_sync_ticker_cancels_moved_earnings_dates(monkeypatch):
    upserts: list = []
    cancels: list = []
    monkeypatch.setattr(calendar_sync, "_fetch_finnhub_earnings",
                        lambda t, k, th=None: [{"date": "2026-11-11", "epsEstimate": 0.1}, {"date": "bad"}, {"date": ""}])
    monkeypatch.setattr(calendar_sync, "upsert_catalyst", lambda row: upserts.append(row))
    monkeypatch.setattr(calendar_sync, "cancel_superseded_catalysts", lambda *a: cancels.append(a) or 1)

    assert calendar_sync._sync_ticker_earnings("ADMA", "key", _TODAY) == (1, 1)
    assert upserts[0]["event_date"] == date(2026, 11, 11) and upserts[0]["state"] == "upcoming"
    ticker, event_type, source, start, end, keep = cancels[0]
    assert (ticker, event_type, source, keep) == ("ADMA", "earnings", "finnhub", [date(2026, 11, 11)])
    assert (end - start).days == 90


def test_sync_ticker_never_cancels_on_empty_fetch(monkeypatch):
    monkeypatch.setattr(calendar_sync, "_fetch_finnhub_earnings", lambda t, k, th=None: [])
    monkeypatch.setattr(calendar_sync, "cancel_superseded_catalysts", lambda *a: pytest.fail("must not cancel"))
    assert calendar_sync._sync_ticker_earnings("ADMA", "key", _TODAY) == (0, 0)


def test_sync_ticker_does_not_reopen_past_dates(monkeypatch):
    upserts: list = []
    monkeypatch.setattr(calendar_sync, "_fetch_finnhub_earnings", lambda t, k, th=None: [{"date": "2026-10-01"}])
    monkeypatch.setattr(calendar_sync, "upsert_catalyst", lambda row: upserts.append(row))
    monkeypatch.setattr(calendar_sync, "cancel_superseded_catalysts", lambda *a: 0)
    calendar_sync._sync_ticker_earnings("ADMA", "key", _TODAY)
    assert "state" not in upserts[0]


# ── Finnhub pacing / failures ──────────────────────────────────────────────────


class _Resp:
    def __init__(self, status: int, body: object) -> None:
        self.status_code = status
        self._body = body

    def json(self) -> object:
        return self._body


def test_fetch_earnings_failure_is_none_not_empty(monkeypatch):
    monkeypatch.setattr(finnhub_client.requests, "get", lambda *a, **kw: _Resp(500, {}))
    assert calendar_sync._fetch_finnhub_earnings("ADMA", "k") is None


def test_fetch_earnings_429_retried_once_with_throttle(monkeypatch):
    responses = iter([_Resp(429, {}), _Resp(200, {"earningsCalendar": [{"date": "2026-11-11"}]})])
    monkeypatch.setattr(finnhub_client.requests, "get", lambda *a, **kw: next(responses))
    monkeypatch.setattr(finnhub_client.time, "sleep", lambda s: None)
    assert calendar_sync._fetch_finnhub_earnings("ADMA", "k", finnhub_client.FinnhubThrottle(0)) == [{"date": "2026-11-11"}]


def test_sync_ticker_skips_on_failed_fetch(monkeypatch):
    monkeypatch.setattr(calendar_sync, "_fetch_finnhub_earnings", lambda t, k, th=None: None)
    monkeypatch.setattr(calendar_sync, "upsert_catalyst", lambda row: pytest.fail("must not upsert"))
    monkeypatch.setattr(calendar_sync, "cancel_superseded_catalysts", lambda *a: pytest.fail("must not cancel"))
    assert calendar_sync._sync_ticker_earnings("ADMA", "k", _TODAY) is None


def test_run_counts_fetch_failures(monkeypatch, run_env):
    monkeypatch.setenv("FINNHUB_API_KEY", "k")
    monkeypatch.setattr(calendar_sync, "get_watchlist_tickers", lambda: ["AAA", "BBB"])
    monkeypatch.setattr(calendar_sync, "get_all_upcoming_catalysts", lambda days_ahead: [])
    monkeypatch.setattr(calendar_sync, "_sync_ticker_earnings", lambda t, k, d, th: None if t == "AAA" else (2, 1))
    result = run(as_of_date=_TODAY)
    assert (result["finnhub_fetch_failures"], result["catalysts_upserted"], result["catalysts_cancelled"]) == (1, 2, 1)
