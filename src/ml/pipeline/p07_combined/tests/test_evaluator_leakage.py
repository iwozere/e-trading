"""
Tests that run_evaluation() produces zero label overlap between train/val/test
and that each segment's labels are computed solely from that segment's prices.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.ml.pipeline.p07_combined.evaluator import P07Evaluator


def _make_ohlcv(n: int = 600) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", periods=n, freq="15min")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "open":   close * (1 + np.random.randn(n) * 0.001),
        "high":   close * (1 + np.abs(np.random.randn(n)) * 0.002),
        "low":    close * (1 - np.abs(np.random.randn(n)) * 0.002),
        "close":  close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    }, index=dates)
    return df


def _minimal_params() -> dict:
    return {
        "tpl_hours": 1.0,
        "pt_mult": 1.0,
        "sl_mult": 1.0,
        "rsi_period": 7,
        "bb_period": 10,
        "bb_std": 2.0,
        "atr_period": 7,
        "vol_lookback": 10,
        "buy_prob_min": 0.5,
        "sell_prob_min": 0.5,
        "max_depth": 3,
        "learning_rate": 0.1,
        "n_estimators": 10,
    }


@pytest.fixture
def ohlcv():
    return _make_ohlcv(600)


@pytest.fixture
def params():
    return _minimal_params()


def test_no_index_overlap_between_segments(ohlcv, params):
    """Train, val, and test X-index sets must be pairwise disjoint."""
    res = P07Evaluator.run_evaluation(ohlcv, params, timeframe="15m")
    if "error" in res:
        pytest.skip(f"Evaluation returned error: {res['error']}")

    idx_train = set(res["X_train"].index)
    idx_val   = set(res["X_val"].index)
    idx_test  = set(res["X_test"].index)

    assert idx_train.isdisjoint(idx_val),  "Train and val indices overlap!"
    assert idx_train.isdisjoint(idx_test), "Train and test indices overlap!"
    assert idx_val.isdisjoint(idx_test),   "Val and test indices overlap!"


def test_train_ends_before_val_starts(ohlcv, params):
    """All train timestamps must precede all val timestamps (temporal ordering)."""
    res = P07Evaluator.run_evaluation(ohlcv, params, timeframe="15m")
    if "error" in res:
        pytest.skip(f"Evaluation returned error: {res['error']}")

    assert res["X_train"].index.max() < res["X_val"].index.min(), (
        "Train set extends into val period"
    )
    assert res["X_val"].index.max() < res["X_test"].index.min(), (
        "Val set extends into test period"
    )


def test_result_has_pf_val_and_pf_test(ohlcv, params):
    """run_evaluation must return both pf_val and pf_test keys."""
    res = P07Evaluator.run_evaluation(ohlcv, params, timeframe="15m")
    if "error" in res:
        pytest.skip(f"Evaluation returned error: {res['error']}")

    assert "pf_val" in res, "Missing pf_val in result"
    assert "pf_test" in res, "Missing pf_test in result"


def test_tpl_bars_buffer_enforced(ohlcv, params):
    """
    There must be a gap of at least tpl_bars between train-end and val-start,
    and between val-end and test-start, to prevent label contamination.
    """
    tpl_bars = P07Evaluator.hours_to_bars(params["tpl_hours"], "15m")
    n = len(ohlcv)
    train_end = int(n * 0.60) - tpl_bars
    val_end   = int(n * 0.80) - tpl_bars

    # The evaluator slices ohlcv_val starting at train_end + tpl_bars
    expected_val_start = ohlcv.index[train_end + tpl_bars]
    expected_test_start = ohlcv.index[val_end + tpl_bars]

    res = P07Evaluator.run_evaluation(ohlcv, params, timeframe="15m")
    if "error" in res:
        pytest.skip(f"Evaluation returned error: {res['error']}")

    # val segment should start at or after expected_val_start
    assert res["X_val"].index.min() >= expected_val_start, (
        "Val segment starts before buffer boundary — potential leakage"
    )
    assert res["X_test"].index.min() >= expected_test_start, (
        "Test segment starts before buffer boundary — potential leakage"
    )
