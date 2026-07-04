"""
Tests for P09 walk-forward cointegration backtest mode.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _make_cointegrated_pair(n: int = 500, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    b = np.cumsum(rng.standard_normal(n))
    noise = rng.standard_normal(n) * 0.3
    a = 1.5 * b + 10.0 + noise
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(a, index=idx), pd.Series(b, index=idx)


class TestWalkForwardBacktest:
    def test_returns_correct_n_folds(self):
        from src.ml.pipeline.p09_arbitrage.cointegration_analyzer import CointegrationAnalyzer

        a, b = _make_cointegrated_pair()
        result = CointegrationAnalyzer().walk_forward_backtest(a, b, n_splits=4)
        assert result["n_folds"] == 4

    def test_avg_sharpe_is_finite(self):
        from src.ml.pipeline.p09_arbitrage.cointegration_analyzer import CointegrationAnalyzer

        a, b = _make_cointegrated_pair()
        result = CointegrationAnalyzer().walk_forward_backtest(a, b, n_splits=4)
        assert np.isfinite(result["avg_sharpe"])

    def test_avg_half_life_positive(self):
        from src.ml.pipeline.p09_arbitrage.cointegration_analyzer import CointegrationAnalyzer

        a, b = _make_cointegrated_pair()
        result = CointegrationAnalyzer().walk_forward_backtest(a, b, n_splits=4)
        assert result["avg_half_life"] > 0

    def test_fold_metrics_have_required_keys(self):
        from src.ml.pipeline.p09_arbitrage.cointegration_analyzer import CointegrationAnalyzer

        a, b = _make_cointegrated_pair()
        result = CointegrationAnalyzer().walk_forward_backtest(a, b, n_splits=3)
        required = {"fold", "beta", "half_life", "sharpe", "total_return", "n_train", "n_test"}
        for fold in result["fold_metrics"]:
            assert required.issubset(fold.keys()), f"Missing keys in fold: {fold.keys()}"

    def test_short_series_returns_zero_folds(self):
        from src.ml.pipeline.p09_arbitrage.cointegration_analyzer import CointegrationAnalyzer

        rng = np.random.default_rng(42)
        a = pd.Series(rng.standard_normal(50))
        b = pd.Series(rng.standard_normal(50))
        result = CointegrationAnalyzer().walk_forward_backtest(a, b, n_splits=3)
        assert result["n_folds"] == 0
        assert np.isnan(result["avg_sharpe"])
