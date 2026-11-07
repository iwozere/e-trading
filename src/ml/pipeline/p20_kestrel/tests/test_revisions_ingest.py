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
    monkeypatch.setattr(revisions_ingest, "get_signals", lambda *a, **kw: [])

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
    monkeypatch.setattr(revisions_ingest, "get_signals", lambda *a, **kw: [])

    assert _compute_ticker_signals("NVDA", "fmp-key", "finnhub-key", _TODAY) == {}


def test_compute_ticker_signals_outside_fmp_slice_scores_with_stored_values(monkeypatch):
    def _no_fmp(*a, **kw):
        raise AssertionError("FMP must not be called outside today's budget slice")

    stored = {
        "fmp_grade_net_60d": [{"date": _TODAY - timedelta(days=6), "value": 1.0}, {"date": _TODAY - timedelta(days=1), "value": 4.0}],
        "fmp_eps_avg_next_fy": [{"date": _TODAY - timedelta(days=1), "value": 10.0}],
    }
    eps_seen: list[float | None] = []

    def _fake_eps_points(ticker, current, today):
        eps_seen.append(current)
        return 0.0

    monkeypatch.setattr(revisions_ingest, "_fetch_fmp_eps_avg_next_fy", _no_fmp)
    monkeypatch.setattr(revisions_ingest, "_fetch_fmp_grades_net_60d", _no_fmp)
    monkeypatch.setattr(revisions_ingest, "_fetch_finnhub_momentum", lambda *a, **kw: 0.0)
    monkeypatch.setattr(revisions_ingest, "_eps_delta_points", _fake_eps_points)
    monkeypatch.setattr(revisions_ingest, "get_signals", lambda ticker, signal_type, **kw: stored.get(signal_type, []))

    signals = _compute_ticker_signals("NVDA", "fmp-key", "finnhub-key", _TODAY, fetch_fmp=False)

    # carried values feed the score (latest grades row = 4 -> full 8 pts) but are not re-written for today
    assert signals == {"finnhub_rec_momentum": 0.0, "revisions_score": 8.0}
    assert eps_seen == [10.0]


# ---------------------------------------------------------------------------
# FMP daily budget rotation
# ---------------------------------------------------------------------------


def test_weekday_number_skips_weekends():
    friday, saturday, sunday, monday = date(2026, 11, 6), date(2026, 11, 7), date(2026, 11, 8), date(2026, 11, 9)
    n = revisions_ingest._weekday_number(friday)
    assert revisions_ingest._weekday_number(saturday) == n + 1
    assert revisions_ingest._weekday_number(sunday) == n + 1
    assert revisions_ingest._weekday_number(monday) == n + 1


def test_fmp_slice_covers_universe_over_consecutive_weekdays():
    tickers = [f"T{i:03d}" for i in range(375)]
    day = date(2026, 11, 16)  # Monday
    covered: set[str] = set()
    for offset in range(5):  # Mon-Fri
        todays = revisions_ingest._fmp_slice(tickers, day + timedelta(days=offset), call_budget=150)
        assert len(todays) == 75
        covered |= todays
    assert covered == set(tickers)


def test_fmp_slice_all_tickers_when_budget_suffices():
    tickers = ["A", "B", "C"]
    assert revisions_ingest._fmp_slice(tickers, _TODAY, call_budget=6) == {"A", "B", "C"}
    assert revisions_ingest._fmp_slice(tickers, _TODAY, call_budget=1) == set()


def test_run_fetches_fmp_only_for_todays_slice(monkeypatch, run_env):
    tickers = [f"T{i}" for i in range(10)]
    monkeypatch.setattr(revisions_ingest, "_get_target_tickers", lambda: tickers)
    monkeypatch.setattr(revisions_ingest, "REVISIONS_FMP_DAILY_CALL_BUDGET", 6)
    flags: dict[str, bool] = {}

    def _fake_compute(ticker, *a, fetch_fmp=True, **kw):
        flags[ticker] = fetch_fmp
        return {"revisions_score": 1.0}

    monkeypatch.setattr(revisions_ingest, "_compute_ticker_signals", _fake_compute)

    result = run(as_of_date=_TODAY)

    assert result["fmp_tickers"] == 3
    assert sum(flags.values()) == 3


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


# ---------------------------------------------------------------------------
# FMP circuit breaker + chunked upserts
# ---------------------------------------------------------------------------


def test_fmp_gate_skips_all_fmp_calls_after_first_429(monkeypatch):
    calls: list[str] = []

    def _fake_get(url, *a, **kw):
        calls.append(url)
        return _FakeResponse(429, {"Error Message": "Bandwidth Limit Reach"})

    monkeypatch.setattr(revisions_ingest.requests, "get", _fake_get)
    gate = revisions_ingest._FmpGate()

    assert _fetch_fmp_eps_avg_next_fy("NVDA", "k", _TODAY, gate) is None
    assert gate.closed
    assert _fetch_fmp_grades_net_60d("NVDA", "k", _TODAY, gate) is None
    assert _fetch_fmp_eps_avg_next_fy("AAPL", "k", _TODAY, gate) is None
    assert len(calls) == 1


def test_fmp_gate_stays_open_on_non_429_errors(monkeypatch):
    monkeypatch.setattr(revisions_ingest.requests, "get", lambda *a, **kw: _FakeResponse(404, {}))
    gate = revisions_ingest._FmpGate()

    assert _fetch_fmp_grades_net_60d("NVDA", "k", _TODAY, gate) is None
    assert not gate.closed


def test_run_upserts_in_chunks_and_reports_fmp_availability(monkeypatch, run_env):
    monkeypatch.setattr(revisions_ingest, "_get_target_tickers", lambda: ["A", "B", "C", "D", "E"])
    monkeypatch.setattr(revisions_ingest, "_UPSERT_CHUNK_TICKERS", 2)
    batches: list[int] = []

    def _fake_upsert(rows):
        batches.append(len(rows))
        return len(rows)

    monkeypatch.setattr(revisions_ingest, "upsert_signals", _fake_upsert)

    def _fake_compute(ticker, fmp_key, finnhub_key, today, gate, **kw):
        gate.closed = True  # simulate FMP quota exhausted mid-run
        return {"revisions_score": 1.0}

    monkeypatch.setattr(revisions_ingest, "_compute_ticker_signals", _fake_compute)

    result = run(as_of_date=_TODAY)

    assert batches == [2, 2, 1]
    assert result["rows_upserted"] == 5
    assert result["fmp_available"] is False


def test_throttle_spaces_calls(monkeypatch):
    clock = [100.0]
    sleeps: list[float] = []
    monkeypatch.setattr(revisions_ingest.time, "monotonic", lambda: clock[0])

    def _sleep(s):
        sleeps.append(s)
        clock[0] += s

    monkeypatch.setattr(revisions_ingest.time, "sleep", _sleep)
    throttle = revisions_ingest._Throttle(1.1)

    throttle.wait()
    clock[0] += 0.3
    throttle.wait()

    assert sleeps == [pytest.approx(0.8)]


def test_finnhub_429_retried_once_when_throttled(monkeypatch):
    rows = [{"period": f"2026-0{m}-01", "strongBuy": 5 if m == 8 else 0, "buy": 0, "hold": 5, "sell": 0, "strongSell": 0}
            for m in (5, 6, 7, 8)]
    responses = iter([_FakeResponse(429, {}), _FakeResponse(200, rows)])
    monkeypatch.setattr(revisions_ingest.requests, "get", lambda *a, **kw: next(responses))
    monkeypatch.setattr(revisions_ingest.time, "sleep", lambda s: None)

    result = _fetch_finnhub_momentum("NVDA", "k", revisions_ingest._Throttle(0))

    assert result is not None and result > 0
