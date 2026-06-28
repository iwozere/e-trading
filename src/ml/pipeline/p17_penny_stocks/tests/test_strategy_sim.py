"""Tests for the P17 strategy simulator core (pure functions)."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.ml.pipeline.p17_penny_stocks import strategy_sim as ss
from src.ml.pipeline.p17_penny_stocks.strategy_sim import StrategyParams


def _hold(bars):
    """bars: list of (high, low, close)."""
    return pd.DataFrame(bars, columns=["high", "low", "close"])


def _ohlc(bars):
    """bars: list of (open, high, low, close) — full path incl. detection day."""
    return pd.DataFrame(bars, columns=["open", "high", "low", "close"])


# ── simulate_trade ──────────────────────────────────────────────────────────

def test_initial_stop_hit():
    # buy 10, stop 20% -> 8.0; first bar trades down to 7.9
    p = StrategyParams(stop=0.20, trail=0.05, activate=0.20)
    exit_price, reason = ss.simulate_trade(10.0, _hold([(10.1, 7.9, 8.0)]), p, atr=0.1)
    assert reason == "stop"
    assert exit_price == 8.0


def test_trailing_stop_after_activation():
    # arms at +20% (12). day1 high 13 -> run_high 13, trail 5% -> tstop 12.35
    p = StrategyParams(stop=0.20, trail=0.05, activate=0.20)
    bars = _hold([(13.0, 12.5, 12.8), (13.0, 12.0, 12.1)])  # day2 low 12 <= 12.35
    exit_price, reason = ss.simulate_trade(10.0, bars, p, atr=0.1)
    assert reason == "trail"
    assert round(exit_price, 4) == 12.35


def test_activation_and_trail_same_day():
    # day1 high 15 arms (run_high 15, tstop 14.25); same-day low 14 <= 14.25
    p = StrategyParams(stop=0.20, trail=0.05, activate=0.20)
    exit_price, reason = ss.simulate_trade(10.0, _hold([(15.0, 14.0, 14.2)]), p, atr=0.1)
    assert reason == "trail"
    assert round(exit_price, 4) == 14.25


def test_never_exits_marks_open_at_last_close():
    # stays between stop (8) and arm (12) the whole time
    p = StrategyParams(stop=0.20, trail=0.05, activate=0.20)
    exit_price, reason = ss.simulate_trade(10.0, _hold([(11.0, 9.0, 10.5), (11.5, 9.5, 11.0)]), p, atr=0.1)
    assert reason == "open"
    assert exit_price == 11.0


def test_stop_checked_before_activation_when_both_in_first_bar():
    # a wild bar that both spikes to +20% and craters below the stop -> stop wins
    p = StrategyParams(stop=0.20, trail=0.05, activate=0.20)
    exit_price, reason = ss.simulate_trade(10.0, _hold([(13.0, 7.0, 9.0)]), p, atr=0.1)
    assert reason == "stop"


# ── _resolve_frac (atr scaling + clamp) ─────────────────────────────────────

def test_resolve_frac_pct_and_atr():
    assert ss._resolve_frac(0.20, "pct", atr=0.13) == 0.20
    assert abs(ss._resolve_frac(2.0, "atr", atr=0.13) - 0.26) < 1e-9
    assert ss._resolve_frac(2.0, "atr", atr=0.80) == 0.90    # clamped to 90%


# ── evaluate ────────────────────────────────────────────────────────────────

def test_evaluate_aggregates_per_tier_and_total():
    records = [
        {"ticker": "WIN", "tier": "B", "atr": 0.1, "detection_date": "2026-06-01", "buy_price": 10.0},
        {"ticker": "LOSE", "tier": "C", "atr": 0.1, "detection_date": "2026-06-01", "buy_price": 10.0},
    ]
    paths = {
        # buy=close=10; day1 high 13 arms (tstop 12.35), day2 low 12 <= 12.35 -> trail 12.35 (+23.5%)
        "WIN": _ohlc([(10, 10, 10, 10), (12, 13, 12.5, 12.8), (12.5, 13, 12.0, 12.1)]),
        # buy=close=10; day1 low 7.9 <= stop 8 -> stop exit 8 (-20%)
        "LOSE": _ohlc([(10, 10, 10, 10), (10, 10.1, 7.9, 8.0)]),
    }
    sizing = {"A": 1000.0, "B": 500.0, "C": 100.0}
    res = ss.evaluate(records, paths, StrategyParams(), sizing, entry="close")
    pt = res["per_tier"]
    assert pt["TOTAL"]["n"] == 2
    assert pt["B"]["n"] == 1 and pt["C"]["n"] == 1
    # B winner: 500 * 0.235 = 117.5 ; C loser: 100 * -0.20 = -20
    assert round(pt["B"]["pnl"], 2) == 117.5
    assert round(pt["C"]["pnl"], 2) == -20.0
    assert round(pt["TOTAL"]["pnl"], 2) == 97.5
    assert pt["TOTAL"]["invested"] == 600.0
    assert pt["C"]["stop_rate_pct"] == 100.0


def test_evaluate_skips_paths_without_holding_data():
    records = [{"ticker": "X", "tier": "B", "atr": 0.1, "detection_date": "2026-06-01", "buy_price": 10.0}]
    paths = {"X": _ohlc([(10, 10, 10, 10)])}   # only detection day, no holding bars
    res = ss.evaluate(records, paths, StrategyParams(), {"B": 500.0}, entry="close")
    assert res["per_tier"]["TOTAL"]["n"] == 0


# ── objective_value ─────────────────────────────────────────────────────────

def test_objective_value_variants():
    result = {
        "per_tier": {"TOTAL": {"pnl": 250.0, "roi_pct": 12.5}},
        "trades": [{"return_pct": 20.0}, {"return_pct": -10.0}, {"return_pct": 5.0}],
    }
    assert ss.objective_value(result, "total_pnl") == 250.0
    assert ss.objective_value(result, "roi") == 12.5
    assert ss.objective_value(result, "sharpe") > 0
