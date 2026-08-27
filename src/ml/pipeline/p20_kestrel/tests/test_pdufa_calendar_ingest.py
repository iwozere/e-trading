"""Tests for the Sleeve B1 PDUFA/AdCom/clinical-readout calendar ingest (gap 10.2)."""

from datetime import date

import pytest

from src.ml.pipeline.p20_kestrel.ingest import pdufa_calendar_ingest
from src.ml.pipeline.p20_kestrel.ingest.pdufa_calendar_ingest import _fetch_pdufa_bio_index, run

_TODAY = date(2026, 8, 27)


class _FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json_data = json_data if json_data is not None else []

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


# ---------------------------------------------------------------------------
# _fetch_pdufa_bio_index — circuit breaker
# ---------------------------------------------------------------------------


def test_fetch_index_returns_list_on_success(monkeypatch):
    monkeypatch.setattr(pdufa_calendar_ingest.requests, "get", lambda *a, **kw: _FakeResponse(200, [{"t": "AAPL"}]))

    assert _fetch_pdufa_bio_index() == [{"t": "AAPL"}]


def test_fetch_index_returns_none_on_http_error(monkeypatch):
    monkeypatch.setattr(pdufa_calendar_ingest.requests, "get", lambda *a, **kw: _FakeResponse(503, {}))

    assert _fetch_pdufa_bio_index() is None


def test_fetch_index_returns_none_on_connection_error(monkeypatch):
    def _raise(*a, **kw):
        raise ConnectionError("boom")

    monkeypatch.setattr(pdufa_calendar_ingest.requests, "get", _raise)

    assert _fetch_pdufa_bio_index() is None


def test_fetch_index_returns_none_on_unexpected_shape(monkeypatch):
    """A dict (not a list) means the site's response shape changed — fail safe, not loud."""
    monkeypatch.setattr(pdufa_calendar_ingest.requests, "get", lambda *a, **kw: _FakeResponse(200, {"oops": True}))

    assert _fetch_pdufa_bio_index() is None


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


@pytest.fixture
def run_env(monkeypatch):
    monkeypatch.setattr(pdufa_calendar_ingest, "start_job_run", lambda *a, **kw: None)
    monkeypatch.setattr(pdufa_calendar_ingest, "finish_job_run", lambda *a, **kw: None)
    monkeypatch.setattr(pdufa_calendar_ingest, "get_active_tickers", lambda: ["NVDA", "AAPL"])

    upserted: list[dict] = []
    monkeypatch.setattr(pdufa_calendar_ingest, "upsert_catalyst", lambda row: upserted.append(row))
    return upserted


def test_run_maps_categories_to_event_types(monkeypatch, run_env):
    entries = [
        {"t": "NVDA", "n": "Some PDUFA drug", "d": "2026-09-15", "y": "PDUFA", "p": "day"},
        {"t": "AAPL", "n": "Some AdComm meeting", "d": "2026-09-20", "y": "AdComm", "p": "day"},
        {"t": "NVDA", "n": "Some trial readout", "d": "2026-10-01", "y": "Readout", "p": "month"},
    ]
    monkeypatch.setattr(pdufa_calendar_ingest, "_fetch_pdufa_bio_index", lambda: entries)

    result = run(as_of_date=_TODAY)

    assert result["catalysts_upserted"] == 3
    types = {row["event_type"] for row in run_env}
    assert types == {"pdufa", "adcom", "clinical_readout"}


def test_run_skips_non_event_category_rows(monkeypatch, run_env):
    entries = [
        {"t": "NVDA", "y": "Ticker", "d": "2026-09-15", "p": "day"},  # metadata row, not an event
        {"t": "NVDA", "n": "Real PDUFA", "d": "2026-09-15", "y": "PDUFA", "p": "day"},
    ]
    monkeypatch.setattr(pdufa_calendar_ingest, "_fetch_pdufa_bio_index", lambda: entries)

    result = run(as_of_date=_TODAY)

    assert result["catalysts_upserted"] == 1


def test_run_skips_tickers_outside_tracked_universe(monkeypatch, run_env):
    entries = [{"t": "UNTRACKED", "n": "x", "d": "2026-09-15", "y": "PDUFA", "p": "day"}]
    monkeypatch.setattr(pdufa_calendar_ingest, "_fetch_pdufa_bio_index", lambda: entries)

    result = run(as_of_date=_TODAY)

    assert result["catalysts_upserted"] == 0


def test_run_skips_past_events(monkeypatch, run_env):
    entries = [{"t": "NVDA", "n": "x", "d": "2026-01-01", "y": "PDUFA", "p": "day"}]  # before _TODAY
    monkeypatch.setattr(pdufa_calendar_ingest, "_fetch_pdufa_bio_index", lambda: entries)

    result = run(as_of_date=_TODAY)

    assert result["catalysts_upserted"] == 0


def test_run_maps_precision_to_confidence(monkeypatch, run_env):
    entries = [
        {"t": "NVDA", "n": "x", "d": "2026-09-15", "y": "PDUFA", "p": "day"},
        {"t": "AAPL", "n": "y", "d": "2026-09-15", "y": "Readout", "p": "quarter"},
    ]
    monkeypatch.setattr(pdufa_calendar_ingest, "_fetch_pdufa_bio_index", lambda: entries)

    run(as_of_date=_TODAY)

    by_ticker = {row["ticker"]: row["confidence"] for row in run_env}
    assert by_ticker["NVDA"] == "confirmed"
    assert by_ticker["AAPL"] == "estimated"


def test_run_circuit_breaker_on_fetch_failure(monkeypatch, run_env):
    """A failed fetch must not touch existing catalyst rows — just log and skip."""
    monkeypatch.setattr(pdufa_calendar_ingest, "_fetch_pdufa_bio_index", lambda: None)

    result = run(as_of_date=_TODAY)

    assert result["catalysts_upserted"] == 0
    assert result["status"] == "skipped"
    assert run_env == []
