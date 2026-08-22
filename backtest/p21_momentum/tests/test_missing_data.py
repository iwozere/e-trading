"""Unit tests for backtest.p21_momentum.missing_data."""

from __future__ import annotations

import unittest
from datetime import date

import numpy as np
import pandas as pd

from backtest.p21_momentum.missing_data import detect_and_liquidate_delisting, forward_fill_gaps


def _series_with_gap(gap_days: int, total_days: int = 20) -> pd.Series:
    idx = pd.bdate_range("2020-01-01", periods=total_days)
    values = np.linspace(100.0, 120.0, total_days)
    s = pd.Series(values, index=idx)
    # Punch a gap of gap_days starting at index 5
    s.iloc[5 : 5 + gap_days] = np.nan
    return s


class TestForwardFillGaps(unittest.TestCase):
    def test_short_gap_fully_filled(self):
        s = _series_with_gap(gap_days=2)
        result = forward_fill_gaps(s, max_days=3)
        self.assertEqual(len(result.untradeable_dates), 0)
        self.assertFalse(result.series.isna().any())

    def test_gap_exactly_at_limit_filled(self):
        s = _series_with_gap(gap_days=3)
        result = forward_fill_gaps(s, max_days=3)
        self.assertEqual(len(result.untradeable_dates), 0)

    def test_gap_beyond_limit_leaves_untradeable_dates(self):
        s = _series_with_gap(gap_days=5)
        result = forward_fill_gaps(s, max_days=3)
        self.assertEqual(len(result.untradeable_dates), 2)  # 5 - 3 = 2 days remain NaN

    def test_never_back_fills(self):
        idx = pd.bdate_range("2020-01-01", periods=10)
        s = pd.Series([np.nan] * 3 + [100.0] * 7, index=idx)
        result = forward_fill_gaps(s, max_days=3)
        # Leading NaNs have nothing to forward-fill from -> remain NaN
        self.assertTrue(result.series.iloc[:3].isna().all())


class TestDetectAndLiquidateDelisting(unittest.TestCase):
    def test_still_trading_returns_none(self):
        idx = pd.bdate_range("2020-01-01", periods=10)
        s = pd.Series(np.linspace(100, 110, 10), index=idx)
        result = detect_and_liquidate_delisting("AAPL", s, as_of_date=date(2020, 1, 10))
        self.assertIsNone(result)

    def test_disappeared_performance_related_applies_haircut(self):
        idx = pd.bdate_range("2020-01-01", periods=5)
        s = pd.Series([100.0, 101.0, 95.0, 90.0, 80.0], index=idx)
        result = detect_and_liquidate_delisting("XYZ", s, as_of_date=date(2020, 2, 1), performance_related=True)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result.last_price, 80.0)
        self.assertAlmostEqual(result.liquidation_price, 80.0 * 0.70)

    def test_disappeared_non_performance_no_haircut(self):
        idx = pd.bdate_range("2020-01-01", periods=5)
        s = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=idx)
        result = detect_and_liquidate_delisting("ACQ", s, as_of_date=date(2020, 2, 1), performance_related=False)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result.liquidation_price, 104.0)

    def test_empty_series_returns_none(self):
        result = detect_and_liquidate_delisting("EMPTY", pd.Series(dtype=float), as_of_date=date(2020, 1, 1))
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
