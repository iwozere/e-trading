"""Tests for the Sleeve A revisions feed ingest (gap 10.1, §4.2.1)."""

from datetime import date, timedelta

import pytest

from src.ml.pipeline.p20_kestrel.ingest import revisions_ingest
from src.ml.pipeline.p20_kestrel.ingest.revisions_ingest import (
    _compute_ticker_signals,
    _eps_delta_points,
    _fetch_finnhub_momentum,
    _fetch_fmp_eps_avg_next_fy,
    _fetch_fmp_grades_net_60d,
    _get_target_tickers,
    _net_bullish_score,
    run,
)

_TODAY = date(2026, 8, 13)


# ---------------------------------------------------------------------------
# _get_target_tickers
# ---------------------------------------------------------------------------


def test_get_target_tickers_unions_watchlist_and_positions(monkeypatch):
    monkeypatch.setattr(revisions_ingest, "get_watchlist_tickers", lambda: ["NVDA", "AAPL"])
    monkeypatch.setattr(revisions_ingest, "get_open_positions", lambda: [{"ticker": "AAPL"}, {"ticker": "MSFT"}])

    assert _get_target_tickers() == ["AAPL", "MSFT", "NVDA"]


# ---------------------------------------------------------------------------
# _net_bullish_score
# ---------------------------------------------------------------------------


def test_net_bullish_score_all_strong_buy():
    row = {"strongBuy": 10, "buy": 0, "hold": 0, "sell": 0, "strongSell": 0}
    assert _net_bullish_score(row) == pytest.approx(2.0)


def test_net_bullish_score_all_strong_sell():
    row = {"strongBuy": 0, "buy": 0, "hold": 0, "sell": 0, "strongSell": 10}
    assert _net_bullish_score(row) == pytest.approx(-2.0)


def test_net_bullish_score_zero_analysts_returns_none():
    row = {"strongBuy": 0, "buy": 0, "hold": 0, "sell": 0, "strongSell": 0}
    assert _net_bullish_score(row) is None


# ---------------------------------------------------------------------------
# _fetch_fmp_eps_avg_next_fy
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json_data = json_data if json_data is not None else []

    def json(self):
        return self._json_data


def test_fetch_eps_avg_picks_nearest_future_row(monkeypatch):
    """API returns rows out of order; the nearest *future* fiscal year must win."""
    rows = [
        {"date": "2031-01-25", "epsAvg": 20.0},
        {"date": "2027-01-25", "epsAvg": 8.5},  # nearest future row
        {"date": "2025-01-25", "epsAvg": 5.0},  # past — must be excluded
    ]
    monkeypatch.setattr(revisions_ingest.requests, "get", lambda *a, **kw: _FakeResponse(200, rows))

    result = _fetch_fmp_eps_avg_next_fy("NVDA", "fake-key", _TODAY)
    assert result == pytest.approx(8.5)


def test_fetch_eps_avg_no_future_rows_returns_none(monkeypatch):
    rows = [{"date": "2020-01-25", "epsAvg": 5.0}]
    monkeypatch.setattr(revisions_ingest.requests, "get", lambda *a, **kw: _FakeResponse(200, rows))

    assert _fetch_fmp_eps_avg_next_fy("NVDA", "fake-key", _TODAY) is None


def test_fetch_eps_avg_non_200_returns_none(monkeypatch):
    monkeypatch.setattr(revisions_ingest.requests, "get", lambda *a, **kw: _FakeResponse(402, {}))

    assert _fetch_fmp_eps_avg_next_fy("NVDA", "fake-key", _TODAY) is None


# ---------------------------------------------------------------------------
# _fetch_fmp_grades_net_60d
# ---------------------------------------------------------------------------


def test_grades_net_counts_upgrades_and_downgrades_in_window(monkeypatch):
    rows = [
        {"date": "2026-08-01", "action": "upgrade"},
        {"date": "2026-07-20", "action": "upgrade"},
        {"date": "2026-07-01", "action": "downgrade"},
        {"date": "2026-06-01", "action": "maintain"},  # ignored
        {"date": "2025-01-01", "action": "upgrade"},  # outside 60d window
    ]
    monkeypatch.setattr(revisions_ingest.requests, "get", lambda *a, **kw: _FakeResponse(200, rows))

    assert _fetch_fmp_grades_net_60d("NVDA", "fake-key", _TODAY) == 1


def test_grades_net_non_list_returns_none(monkeypatch):
    monkeypatch.setattr(revisions_ingest.requests, "get", lambda *a, **kw: _FakeResponse(200, {"error": "x"}))

    assert _fetch_fmp_grades_net_60d("NVDA", "fake-key", _TODAY) is None


# ---------------------------------------------------------------------------
# _fetch_finnhub_momentum
# ---------------------------------------------------------------------------


def test_finnhub_momentum_positive_when_bullish_shift(monkeypatch):
    rows = [
        {"period": "2026-08-01", "strongBuy": 20, "buy": 10, "hold": 0, "sell": 0, "strongSell": 0},
        {"period": "2026-07-01", "strongBuy": 15, "buy": 10, "hold": 5, "sell": 0, "strongSell": 0},
        {"period": "2026-06-01", "strongBuy": 10, "buy": 10, "hold": 10, "sell": 0, "strongSell": 0},
        {"period": "2026-05-01", "strongBuy": 0, "buy": 10, "hold": 20, "sell": 0, "strongSell": 0},
    ]
    monkeypatch.setattr(revisions_ingest.requests, "get", lambda *a, **kw: _FakeResponse(200, rows))

    momentum = _fetch_finnhub_momentum("NVDA", "fake-key")
    assert momentum is not None
    assert momentum > 0


def test_finnhub_momentum_insufficient_history_returns_none(monkeypatch):
    rows = [{"period": "2026-08-01", "strongBuy": 1, "buy": 0, "hold": 0, "sell": 0, "strongSell": 0}]
    monkeypatch.setattr(revisions_ingest.requests, "get", lambda *a, **kw: _FakeResponse(200, rows))

    assert _fetch_finnhub_momentum("NVDA", "fake-key") is None


# ---------------------------------------------------------------------------
# _eps_delta_points
# ---------------------------------------------------------------------------


def test_eps_delta_points_zero_without_warmup(monkeypatch):
    monkeypatch.setattr(revisions_ingest, "get_signals", lambda *a, **kw: [])

    assert _eps_delta_points("NVDA", 10.0, _TODAY) == 0.0


def test_eps_delta_points_zero_when_current_missing(monkeypatch):
    assert _eps_delta_points("NVDA", None, _TODAY) == 0.0


def test_eps_delta_points_full_credit_at_threshold(monkeypatch):
    past_date = _TODAY - timedelta(days=60)
    monkeypatch.setattr(
        revisions_ingest,
        "get_signals",
        lambda *a, **kw: [{"date": past_date, "value": 10.0, "sleeve": "A"}],
    )

    # +5% over 60d == full credit per REVISIONS_EPS_DELTA_FULL_CREDIT_PCT
    points = _eps_delta_points("NVDA", 10.5, _TODAY)
    assert points == pytest.approx(revisions_ingest.REVISIONS_EPS_DELTA_MAX_PTS)


def test_eps_delta_points_negative_change_clips_to_zero(monkeypatch):
    past_date = _TODAY - timedelta(days=60)
    monkeypatch.setattr(
        revisions_ingest,
        "get_signals",
        lambda *a, **kw: [{"date": past_date, "value": 10.0, "sleeve": "A"}],
    )

    assert _eps_delta_points("NVDA", 9.0, _TODAY) == 0.0


# ---------------------------------------------------------------------------
# _compute_ticker_signals
# ---------------------------------------------------------------------------


def test_compute_ticker_signals_blends_all_three(monkeypatch):
    monkeypatch.setattr(revisions_ingest, "_fetch_fmp_eps_avg_next_fy", lambda *a, **kw: 10.0)
    monkeypatch.setattr(revisions_ingest, "_fetch_fmp_grades_net_60d", lambda *a, **kw: 4)
    monkeypatch.setattr(revisions_ingest, "_fetch_finnhub_momentum", lambda *a, **kw: 1.0)
    monkeypatch.setattr(revisions_ingest, "_eps_delta_points", lambda *a, **kw: 3.0)

    signals = _compute_ticker_signals("NVDA", "fmp-key", "finnhub-key", _TODAY)

    assert signals["fmp_eps_avg_next_fy"] == 10.0
    assert signals["fmp_grade_net_60d"] == 4.0
    assert signals["finnhub_rec_momentum"] == 1.0
    # full finnhub (12) + full grades (8) + 3.0 eps == 23.0
    assert signals["revisions_score"] == pytest.approx(23.0)


def test_compute_ticker_signals_all_sources_fail_returns_empty(monkeypatch):
    monkeypatch.setattr(revisions_ingest, "_fetch_fmp_eps_avg_next_fy", lambda *a, **kw: None)
    monkeypatch.setattr(revisions_ingest, "_fetch_fmp_grades_net_60d", lambda *a, **kw: None)
    monkeypatch.setattr(revisions_ingest, "_fetch_finnhub_momentum", lambda *a, **kw: None)

    assert _compute_ticker_signals("NVDA", "fmp-key", "finnhub-key", _TODAY) == {}


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


@pytest.fixture
def run_env(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "fake-fmp-key")
    monkeypatch.setenv("FINNHUB_API_KEY", "fake-finnhub-key")
    monkeypatch.setattr(revisions_ingest, "start_job_run", lambda *a, **kw: None)
    monkeypatch.setattr(revisions_ingest, "finish_job_run", lambda *a, **kw: None)
    monkeypatch.setattr(revisions_ingest, "_get_target_tickers", lambda: ["NVDA", "AAPL"])

    upserted: list[dict] = []

    def _fake_upsert(rows):
        upserted.extend(rows)
        return len(rows)

    monkeypatch.setattr(revisions_ingest, "upsert_signals", _fake_upsert)
    return upserted


def test_run_upserts_signals_for_each_ticker(monkeypatch, run_env):
    monkeypatch.setattr(
        revisions_ingest,
        "_compute_ticker_signals",
        lambda ticker, *a, **kw: {"revisions_score": 15.0, "fmp_eps_avg_next_fy": 10.0},
    )

    result = run(as_of_date=_TODAY)

    assert result["tickers_processed"] == 2
    assert result["tickers_total"] == 2
    assert result["rows_upserted"] == 4  # 2 signals x 2 tickers
    assert all(row["sleeve"] == "A" for row in run_env)
    assert all(row["date"] == _TODAY for row in run_env)


def test_run_skips_missing_signals_for_a_ticker(monkeypatch, run_env):
    def _fake_compute(ticker, *a, **kw):
        return {} if ticker == "AAPL" else {"revisions_score": 10.0}

    monkeypatch.setattr(revisions_ingest, "_compute_ticker_signals", _fake_compute)

    result = run(as_of_date=_TODAY)

    assert result["tickers_processed"] == 1
    assert result["rows_upserted"] == 1


def test_run_skipped_without_api_keys(monkeypatch):
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    monkeypatch.setattr(revisions_ingest, "start_job_run", lambda *a, **kw: None)
    monkeypatch.setattr(revisions_ingest, "finish_job_run", lambda *a, **kw: None)

    result = run(as_of_date=_TODAY)

    assert result == {"tickers_processed": 0, "tickers_total": 0, "rows_upserted": 0}
