"""Unit tests for src.ml.pipeline.p21_momentum.strategy.signal."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.ml.pipeline.p21_momentum.config import MIN_HISTORY
from src.ml.pipeline.p21_momentum.strategy.signal import compute_signal


def _make_series(n_days: int, daily_return: float = 0.001, annualized_vol: float = 0.20, seed: int = 0) -> pd.Series:
    """Deterministic synthetic adjusted-close series with a drift + noise."""
    rng = np.random.default_rng(seed)
    daily_vol = annualized_vol / np.sqrt(252)
    rets = rng.normal(loc=daily_return, scale=daily_vol, size=n_days)
    prices = 100.0 * np.cumprod(1 + rets)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.Series(prices, index=idx)


class TestComputeSignal(unittest.TestCase):
    def test_insufficient_history_returns_none(self):
        series = _make_series(MIN_HISTORY - 1)
        self.assertIsNone(compute_signal(series))

    def test_exact_min_history_boundary_computes(self):
        series = _make_series(MIN_HISTORY)
        result = compute_signal(series)
        self.assertIsNotNone(result)

    def test_low_vol_guard_returns_none(self):
        # Perfectly flat series -> vol == 0, must not raise ZeroDivisionError
        idx = pd.bdate_range("2020-01-01", periods=400)
        series = pd.Series([100.0] * 400, index=idx)
        self.assertIsNone(compute_signal(series))

    def test_ranks_by_signal_not_raw_return(self):
        """
        Regression guard for the spec's flagged critical bug: two tickers with
        similar raw_return but very different vol must rank by risk-adjusted
        signal, not by raw_return alone.
        """
        low_vol_high_signal = _make_series(400, daily_return=0.0015, annualized_vol=0.10, seed=1)
        high_vol_similar_return = _make_series(400, daily_return=0.0016, annualized_vol=0.60, seed=2)

        r1 = compute_signal(low_vol_high_signal)
        r2 = compute_signal(high_vol_similar_return)
        self.assertIsNotNone(r1)
        self.assertIsNotNone(r2)
        assert r1 is not None and r2 is not None
        # Similar raw_return, but r1's much lower vol must give it the higher signal.
        self.assertGreater(r1.signal, r2.signal)
        self.assertLess(r1.vol, r2.vol)

    def test_window_excludes_most_recent_month(self):
        """
        Appending an extreme recent-month move must NOT move the signal,
        since the window ends at -SKIP_RECENT, excluding the most recent
        ~21 trading days (short-term reversal exclusion, spec §4).
        """
        base = _make_series(400, seed=3)
        r_base = compute_signal(base)

        shocked = base.copy()
        shocked.iloc[-5:] = shocked.iloc[-5:] * 3.0  # huge recent spike, within SKIP_RECENT window
        r_shocked = compute_signal(shocked)

        self.assertIsNotNone(r_base)
        self.assertIsNotNone(r_shocked)
        assert r_base is not None and r_shocked is not None
        self.assertAlmostEqual(r_base.raw_return, r_shocked.raw_return, places=9)
        self.assertAlmostEqual(r_base.signal, r_shocked.signal, places=9)


if __name__ == "__main__":
    unittest.main()
