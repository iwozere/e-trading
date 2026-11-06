"""Tests for P17 TechnicalAgent (indicator computation + per-candidate fault isolation)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p17_penny_stocks.agents.technical_agent import TechnicalAgent
from src.ml.pipeline.p17_penny_stocks.config import P17TechnicalConfig
from src.ml.pipeline.p17_penny_stocks.models.candidate import Candidate


def _ohlcv(n: int = 60, start_price: float = 5.0) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    close = start_price + np.arange(n, dtype=float) * 0.05
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.02,
            "Low": close * 0.98,
            "Close": close,
            "Volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )


def _agent() -> TechnicalAgent:
    return TechnicalAgent(P17TechnicalConfig())


def test_enrich_populates_technical_fields():
    agent = _agent()
    c = Candidate(ticker="GOOD", price=6.0)
    agent.run([c], {"GOOD": _ohlcv()})
    assert c.relative_volume > 0
    assert c.sma20 > 0


def test_bad_ticker_does_not_poison_other_candidates():
    """
    Regression test: a single candidate whose OHLCV blows up _enrich() must not
    abort TechnicalAgent.run() for the whole batch — every other candidate should
    still get enriched normally.
    """
    agent = _agent()
    good = Candidate(ticker="GOOD", price=6.0)
    bad = Candidate(ticker="BAD", price=1.0)

    # BAD has enough rows to reach the accumulation-days step but is missing the
    # "Open" column that step requires — _enrich() raises KeyError partway through.
    bad_df = _ohlcv(30).drop(columns=["Open"])

    result = agent.run([good, bad], {"GOOD": _ohlcv(), "BAD": bad_df})

    assert good.relative_volume > 0
    assert good.sma20 > 0
    assert len(result) == 2


def test_setup_features_for_flat_base_below_high():
    """Price drifts back below an earlier spike high: negative distance and a tight 10d range."""
    df = _ohlcv(60)
    df.loc[df.index[30], "High"] = 20.0  # old spike high inside the 50d window
    df.loc[df.index[-10:], ["Open", "High", "Low", "Close"]] = [[8.0, 8.2, 7.9, 8.0]] * 10
    c = Candidate(ticker="BASE", price=8.0)
    _agent().run([c], {"BASE": df})
    assert c.dist_to_50d_high == 8.0 / 20.0 - 1.0
    assert abs(c.range_10d_pct - (8.2 - 7.9) / 8.0) < 1e-9
    assert c.extension_atr != 0.0


def test_setup_features_steady_uptrend_sits_at_the_high():
    c = Candidate(ticker="UP", price=8.0)
    _agent().run([c], {"UP": _ohlcv(60)})
    assert c.dist_to_50d_high > -0.03
    assert c.extension_atr > 0
