"""Tests for the P17 point-in-time replay helpers."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p17_penny_stocks.replay import (
    candidate_from_row,
    dedupe_episodes,
    forward_outcomes,
    summarise,
)


def test_candidate_from_row_restores_point_in_time_fields():
    row = {
        "ticker": "ABC",
        "company_name": float("nan"),
        "price": 3.5,
        "dilution_penalty": 20.0,
        "catalyst_score": 55.0,
        "short_interest_pct_float": 0.22,
        "revenue_growth_yoy": float("nan"),
        "catalyst_signals": "catalyst_fda_2026-07-01|catalyst_pr_2026-07-02",
        "dilution_signals": float("nan"),
        "run_date": "2026-07-03",
        "relative_volume": 9.9,
    }
    c = candidate_from_row(row)
    assert c.ticker == "ABC" and c.company_name == ""
    assert c.price == 3.5 and c.dilution_penalty == 20.0 and c.catalyst_score == 55.0
    assert c.short_interest_pct_float == 0.22 and c.revenue_growth_yoy is None
    assert c.catalyst_signals == ["catalyst_fda_2026-07-01", "catalyst_pr_2026-07-02"]
    assert c.dilution_signals == []
    assert c.relative_volume == 0.0  # technicals are recomputed, never restored


def test_dedupe_episodes_cooldown():
    df = pd.DataFrame(
        {
            "ticker": ["A", "A", "A", "B"],
            "run_date": pd.to_datetime(["2026-07-01", "2026-07-03", "2026-08-01", "2026-07-02"]),
        }
    )
    out = dedupe_episodes(df, cooldown_days=14)
    assert sorted(zip(out["ticker"], out["run_date"].dt.strftime("%m-%d"))) == [
        ("A", "07-01"),
        ("A", "08-01"),
        ("B", "07-02"),
    ]


def _px(n: int = 30) -> pd.DataFrame:
    idx = pd.bdate_range("2026-07-01", periods=n)
    close = np.linspace(10.0, 12.9, n)
    return pd.DataFrame({"Open": close, "High": close + 0.5, "Low": close - 0.5, "Close": close}, index=idx)


def test_forward_outcomes_enter_next_open():
    px = _px()
    out = forward_outcomes(px, px.index[4])
    assert out is not None
    entry = px["Open"].iloc[5]
    assert out["fwd1"] == px["Close"].iloc[5] / entry - 1
    assert out["fwd5"] == px["Close"].iloc[9] / entry - 1
    assert out["max_dd_10d"] == px["Low"].iloc[5:15].min() / entry - 1


def test_forward_outcomes_none_after_last_bar():
    px = _px()
    assert forward_outcomes(px, px.index[-1]) is None


def test_forward_outcomes_short_horizon_is_nan():
    px = _px()
    out = forward_outcomes(px, px.index[-3])
    assert out is not None and np.isnan(out["fwd20"])


def test_summarise_win_rate():
    ep = pd.DataFrame(
        {
            "price_5d_return": [0.1, 0.2],
            "fwd5": [0.1, -0.1],
            "fwd10": [0.2, 0.1],
            "fwd20": [np.nan, -0.3],
            "max_up_10d": [0.3, 0.1],
            "max_dd_10d": [-0.1, -0.2],
        }
    )
    s = summarise(ep, "x")
    assert s["n"] == 2 and s["fwd5_win"] == 0.5 and s["fwd10_win"] == 1.0 and s["fwd20_win"] == 0.0
