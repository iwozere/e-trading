"""Tests for the Sleeve B1 catalyst calendar sync -- combined countdown digest."""

from datetime import date

import pytest

from src.ml.pipeline.p20_kestrel.ingest import calendar_sync
from src.ml.pipeline.p20_kestrel.ingest.calendar_sync import run

_TODAY = date(2026, 10, 13)


@pytest.fixture
def run_env(monkeypatch):
    monkeypatch.setattr(calendar_sync, "start_job_run", lambda *_a, **_kw: None)
    monkeypatch.setattr(calendar_sync, "finish_job_run", lambda *_a, **_kw: None)
    monkeypatch.setattr(calendar_sync, "get_watchlist_tickers", lambda: [])
    monkeypatch.setattr(calendar_sync, "log_alert", lambda **_kw: None)

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
