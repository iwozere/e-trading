"""Tests for the Sleeve A revisions feed ingest (gap 10.1, §4.2.1)."""

from datetime import date, datetime

import pandas as pd
import pytest
from yfinance.exceptions import YFRateLimitError

from src.ml.pipeline.p20_kestrel.ingest import finnhub_client, revisions_ingest
from src.ml.pipeline.p20_kestrel.ingest.revisions_ingest import (
    _compute_ticker_signals,
    _eps_delta_points,
    _fetch_finnhub_momentum,
    _get_target_tickers,
    _net_bullish_score,
    _parse_eps_trend,
    _parse_grades_net,
    _yahoo_frame,
    run,
)

_TODAY = date(2026, 8, 13)


class _FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json_data = json_data if json_data is not None else []

    def json(self):
        return self._json_data


def _eps_trend(rows: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Build an eps_trend-shaped frame: period -> (current, 60daysAgo)."""
    return pd.DataFrame(
        {"current": [v[0] for v in rows.values()], "60daysAgo": [v[1] for v in rows.values()]},
        index=pd.Index(list(rows.keys()), name="period"),
    )


def _grades(rows: list[tuple[str, str]]) -> pd.DataFrame:
    """Build an upgrades_downgrades-shaped frame: (timestamp, action)."""
    return pd.DataFrame(
        {"Firm": ["X"] * len(rows), "Action": [a for _, a in rows]},
        index=pd.DatetimeIndex([datetime.fromisoformat(d) for d, _ in rows], name="GradeDate"),
    )


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
# _parse_eps_trend
# ---------------------------------------------------------------------------


def test_parse_eps_trend_uses_next_fiscal_year():
    frame = _eps_trend({"0q": (0.24, 0.24), "0y": (0.81, 0.80), "+1y": (1.05, 1.00)})

    current, delta = _parse_eps_trend(frame)

    assert current == pytest.approx(1.05)
    assert delta == pytest.approx(0.05)


def test_parse_eps_trend_falls_back_to_current_year_when_next_missing():
    frame = _eps_trend({"0y": (2.0, 2.5), "+1y": (float("nan"), float("nan"))})

    current, delta = _parse_eps_trend(frame)

    assert current == pytest.approx(2.0)
    assert delta == pytest.approx(-0.2)


def test_parse_eps_trend_zero_past_gives_no_delta():
    current, delta = _parse_eps_trend(_eps_trend({"+1y": (0.1, 0.0)}))

    assert current == pytest.approx(0.1)
    assert delta is None


def test_parse_eps_trend_negative_base_uses_absolute_value():
    # -1.00 -> -0.50 is an improvement: +50%
    _, delta = _parse_eps_trend(_eps_trend({"+1y": (-0.5, -1.0)}))

    assert delta == pytest.approx(0.5)


@pytest.mark.parametrize("frame", [None, pd.DataFrame()])
def test_parse_eps_trend_empty_returns_none(frame):
    assert _parse_eps_trend(frame) == (None, None)


# ---------------------------------------------------------------------------
# _parse_grades_net
# ---------------------------------------------------------------------------


def test_grades_net_counts_upgrades_and_downgrades_in_window():
    frame = _grades(
        [
            ("2026-08-01 09:30:00", "up"),
            ("2026-07-20 14:00:00", "up"),
            ("2026-07-01 12:00:00", "down"),
            ("2026-06-20 12:00:00", "main"),  # ignored
            ("2026-06-20 12:00:00", "init"),  # ignored
            ("2025-01-01 12:00:00", "up"),  # outside the 60d window
        ]
    )

    assert _parse_grades_net(frame, _TODAY) == 1


def test_grades_net_empty_frame_is_zero_and_none_is_unknown():
    assert _parse_grades_net(pd.DataFrame(), _TODAY) == 0
    assert _parse_grades_net(None, _TODAY) is None


def test_grades_net_missing_action_column_returns_none():
    frame = pd.DataFrame({"Firm": ["X"]}, index=pd.DatetimeIndex([datetime(2026, 8, 1)]))

    assert _parse_grades_net(frame, _TODAY) is None


# ---------------------------------------------------------------------------
# _yahoo_frame + Yahoo circuit breaker
# ---------------------------------------------------------------------------


class _FakeTicker:
    """Stand-in for yf.Ticker whose properties return or raise what the test configures."""

    behaviour: object = None
    calls = 0

    def __init__(self, symbol, session=None):
        self.symbol = symbol

    def _get(self):
        type(self).calls += 1
        if isinstance(self.behaviour, Exception):
            raise self.behaviour
        return self.behaviour

    eps_trend = property(_get)
    upgrades_downgrades = property(_get)


@pytest.fixture
def fake_yf(monkeypatch):
    _FakeTicker.behaviour = None
    _FakeTicker.calls = 0
    monkeypatch.setattr(revisions_ingest.yf, "Ticker", _FakeTicker)
    monkeypatch.setattr(revisions_ingest, "get_shared_session", lambda: None)
    return _FakeTicker


def test_yahoo_frame_returns_dataframe(fake_yf):
    fake_yf.behaviour = _eps_trend({"+1y": (1.0, 1.0)})

    frame = _yahoo_frame("NVDA", "eps_trend", revisions_ingest._YahooGate())

    assert isinstance(frame, pd.DataFrame)


def test_yahoo_gate_trips_on_rate_limit(fake_yf):
    fake_yf.behaviour = YFRateLimitError()
    gate = revisions_ingest._YahooGate()

    assert _yahoo_frame("NVDA", "eps_trend", gate) is None
    assert gate.closed
    assert _yahoo_frame("AAPL", "upgrades_downgrades", gate) is None
    assert fake_yf.calls == 1


def test_yahoo_gate_trips_after_consecutive_errors(fake_yf, monkeypatch):
    monkeypatch.setattr(revisions_ingest, "_YAHOO_MAX_CONSECUTIVE_ERRORS", 3)
    fake_yf.behaviour = RuntimeError("blocked")
    gate = revisions_ingest._YahooGate()

    for _ in range(2):
        _yahoo_frame("NVDA", "eps_trend", gate)
    assert not gate.closed
    _yahoo_frame("NVDA", "eps_trend", gate)
    assert gate.closed


def test_yahoo_gate_success_resets_error_count(fake_yf, monkeypatch):
    monkeypatch.setattr(revisions_ingest, "_YAHOO_MAX_CONSECUTIVE_ERRORS", 2)
    gate = revisions_ingest._YahooGate()

    fake_yf.behaviour = RuntimeError("blip")
    _yahoo_frame("A", "eps_trend", gate)
    fake_yf.behaviour = pd.DataFrame()
    _yahoo_frame("B", "eps_trend", gate)
    fake_yf.behaviour = RuntimeError("blip")
    _yahoo_frame("C", "eps_trend", gate)

    assert not gate.closed


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
    monkeypatch.setattr(finnhub_client.requests, "get", lambda *a, **kw: _FakeResponse(200, rows))

    momentum = _fetch_finnhub_momentum("NVDA", "fake-key")
    assert momentum is not None
    assert momentum > 0


def test_finnhub_momentum_insufficient_history_returns_none(monkeypatch):
    rows = [{"period": "2026-08-01", "strongBuy": 1, "buy": 0, "hold": 0, "sell": 0, "strongSell": 0}]
    monkeypatch.setattr(finnhub_client.requests, "get", lambda *a, **kw: _FakeResponse(200, rows))

    assert _fetch_finnhub_momentum("NVDA", "fake-key") is None


# ---------------------------------------------------------------------------
# _eps_delta_points
# ---------------------------------------------------------------------------


def test_eps_delta_points_full_credit_at_threshold():
    # +5% over 60d == full credit per REVISIONS_EPS_DELTA_FULL_CREDIT_PCT
    assert _eps_delta_points(0.05) == pytest.approx(revisions_ingest.REVISIONS_EPS_DELTA_MAX_PTS)


def test_eps_delta_points_scales_linearly_and_clips():
    assert _eps_delta_points(0.025) == pytest.approx(revisions_ingest.REVISIONS_EPS_DELTA_MAX_PTS / 2)
    assert _eps_delta_points(0.5) == pytest.approx(revisions_ingest.REVISIONS_EPS_DELTA_MAX_PTS)


@pytest.mark.parametrize("delta", [None, 0.0, -0.1])
def test_eps_delta_points_zero_without_positive_revision(delta):
    assert _eps_delta_points(delta) == 0.0


# ---------------------------------------------------------------------------
# _compute_ticker_signals
# ---------------------------------------------------------------------------


def _patch_sources(monkeypatch, eps_frame, grades_frame, momentum):
    frames = {"eps_trend": eps_frame, "upgrades_downgrades": grades_frame}
    monkeypatch.setattr(revisions_ingest, "_yahoo_frame", lambda ticker, attribute, gate: frames[attribute])
    monkeypatch.setattr(revisions_ingest, "_fetch_finnhub_momentum", lambda *a, **kw: momentum)


def test_compute_ticker_signals_blends_all_three(monkeypatch):
    grades = _grades([(f"2026-08-0{d} 10:00:00", "up") for d in range(1, 5)])
    # +1.5% EPS revision -> 3.0 of 10 pts
    _patch_sources(monkeypatch, _eps_trend({"+1y": (10.15, 10.0)}), grades, 1.0)

    signals = _compute_ticker_signals("NVDA", "finnhub-key", _TODAY)

    assert signals["yahoo_eps_next_fy"] == pytest.approx(10.15)
    assert signals["yahoo_eps_delta_60d_pct"] == pytest.approx(0.015)
    assert signals["yahoo_grade_net_60d"] == 4.0
    assert signals["finnhub_rec_momentum"] == 1.0
    # full finnhub (12) + full grades (8) + 3.0 eps == 23.0
    assert signals["revisions_score"] == pytest.approx(23.0)


def test_compute_ticker_signals_finnhub_only_when_yahoo_down(monkeypatch):
    _patch_sources(monkeypatch, None, None, 0.5)

    signals = _compute_ticker_signals("NVDA", "finnhub-key", _TODAY)

    assert signals == {"finnhub_rec_momentum": 0.5, "revisions_score": 6.0}


def test_compute_ticker_signals_all_sources_fail_returns_empty(monkeypatch):
    _patch_sources(monkeypatch, None, None, None)

    assert _compute_ticker_signals("NVDA", "finnhub-key", _TODAY) == {}


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


@pytest.fixture
def run_env(monkeypatch):
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
        lambda ticker, *a, **kw: {"revisions_score": 15.0, "yahoo_eps_delta_60d_pct": 0.01},
    )

    result = run(as_of_date=_TODAY)

    assert result["tickers_processed"] == 2
    assert result["tickers_total"] == 2
    assert result["rows_upserted"] == 4  # 2 signals x 2 tickers
    assert result["yahoo_eps_tickers"] == 2
    assert result["yahoo_available"] is True
    assert all(row["sleeve"] == "A" for row in run_env)
    assert all(row["date"] == _TODAY for row in run_env)


def test_run_skips_missing_signals_for_a_ticker(monkeypatch, run_env):
    def _fake_compute(ticker, *a, **kw):
        return {} if ticker == "AAPL" else {"revisions_score": 10.0}

    monkeypatch.setattr(revisions_ingest, "_compute_ticker_signals", _fake_compute)

    result = run(as_of_date=_TODAY)

    assert result["tickers_processed"] == 1
    assert result["rows_upserted"] == 1


def test_run_skipped_without_finnhub_key(monkeypatch):
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    monkeypatch.setattr(revisions_ingest, "start_job_run", lambda *a, **kw: None)
    monkeypatch.setattr(revisions_ingest, "finish_job_run", lambda *a, **kw: None)

    result = run(as_of_date=_TODAY)

    assert result == {"tickers_processed": 0, "tickers_total": 0, "rows_upserted": 0}


def test_run_upserts_in_chunks_and_reports_yahoo_availability(monkeypatch, run_env):
    monkeypatch.setattr(revisions_ingest, "_get_target_tickers", lambda: ["A", "B", "C", "D", "E"])
    monkeypatch.setattr(revisions_ingest, "_UPSERT_CHUNK_TICKERS", 2)
    batches: list[int] = []

    def _fake_upsert(rows):
        batches.append(len(rows))
        return len(rows)

    monkeypatch.setattr(revisions_ingest, "upsert_signals", _fake_upsert)

    def _fake_compute(ticker, finnhub_key, today, yahoo_gate, **kw):
        yahoo_gate.closed = True  # simulate Yahoo blocking mid-run
        return {"revisions_score": 1.0}

    monkeypatch.setattr(revisions_ingest, "_compute_ticker_signals", _fake_compute)

    result = run(as_of_date=_TODAY)

    assert batches == [2, 2, 1]
    assert result["rows_upserted"] == 5
    assert result["yahoo_available"] is False


# ---------------------------------------------------------------------------
# Finnhub pacing
# ---------------------------------------------------------------------------


def test_throttle_spaces_calls(monkeypatch):
    clock = [100.0]
    sleeps: list[float] = []
    monkeypatch.setattr(finnhub_client.time, "monotonic", lambda: clock[0])

    def _sleep(s):
        sleeps.append(s)
        clock[0] += s

    monkeypatch.setattr(finnhub_client.time, "sleep", _sleep)
    throttle = finnhub_client.FinnhubThrottle(1.1)

    throttle.wait()
    clock[0] += 0.3
    throttle.wait()

    assert sleeps == [pytest.approx(0.8)]


def test_finnhub_429_retried_once_when_throttled(monkeypatch):
    rows = [{"period": f"2026-0{m}-01", "strongBuy": 5 if m == 8 else 0, "buy": 0, "hold": 5, "sell": 0, "strongSell": 0}
            for m in (5, 6, 7, 8)]
    responses = iter([_FakeResponse(429, {}), _FakeResponse(200, rows)])
    monkeypatch.setattr(finnhub_client.requests, "get", lambda *a, **kw: next(responses))
    monkeypatch.setattr(finnhub_client.time, "sleep", lambda s: None)

    result = _fetch_finnhub_momentum("NVDA", "k", finnhub_client.FinnhubThrottle(0))

    assert result is not None and result > 0
