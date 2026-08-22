"""Unit tests for src.ml.pipeline.p21_momentum.strategy.filters."""

from __future__ import annotations

import unittest
from datetime import date
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.ml.pipeline.p21_momentum.strategy.filters import (
    f1_history,
    f2_liquidity,
    f3_gap,
    f4_quality,
    f5_exclusions,
    f6_earnings,
    run_all,
    tally_f4_missing_pct,
)


class TestF1History(unittest.TestCase):
    def test_short_history_fails(self):
        result = f1_history(pd.Series(range(100)))
        self.assertFalse(result.passed)
        self.assertEqual(result.flag, "F1_INSUFFICIENT_HISTORY")

    def test_sufficient_history_passes(self):
        result = f1_history(pd.Series(range(300)))
        self.assertTrue(result.passed)


class TestF2Liquidity(unittest.TestCase):
    def test_illiquid_fails(self):
        close = pd.Series([10.0] * 60)
        volume = pd.Series([1000] * 60)  # $10k/day, far below $50M
        result = f2_liquidity(close, volume)
        self.assertFalse(result.passed)
        self.assertEqual(result.flag, "F2_ILLIQUID")

    def test_liquid_passes(self):
        close = pd.Series([100.0] * 60)
        volume = pd.Series([1_000_000] * 60)  # $100M/day
        result = f2_liquidity(close, volume)
        self.assertTrue(result.passed)


class TestF3Gap(unittest.TestCase):
    def test_total_le_zero_short_circuits_to_pass(self):
        # Declining series -> total log return <= 0 -> always passes F3
        window = pd.Series([100.0, 90.0, 80.0, 70.0, 60.0])
        result = f3_gap(window)
        self.assertTrue(result.passed)

    def test_gap_dominated_fails(self):
        # One huge single-day jump dominates an otherwise-flat positive series
        prices = [100.0] * 10 + [500.0] + [505.0] * 5
        window = pd.Series(prices)
        result = f3_gap(window)
        self.assertFalse(result.passed)
        self.assertEqual(result.flag, "F3_GAP_DOMINATED")

    def test_smooth_uptrend_passes(self):
        window = pd.Series(np.linspace(100, 150, 50))
        result = f3_gap(window)
        self.assertTrue(result.passed)


class TestF4Quality(unittest.TestCase):
    def test_missing_data_passes_with_flag(self):
        result = f4_quality(None, None)
        self.assertTrue(result.passed)
        self.assertEqual(result.flag, "F4_DATA_MISSING")

    def test_loss_making_on_both_fails(self):
        result = f4_quality(-10.0, -5.0)
        self.assertFalse(result.passed)
        self.assertEqual(result.flag, "F4_LOSS_MAKING")

    def test_loss_on_one_metric_only_passes(self):
        result = f4_quality(-10.0, 5.0)
        self.assertTrue(result.passed)


class TestF5Exclusions(unittest.TestCase):
    def test_excluded_ticker_fails(self):
        result = f5_exclusions("XYZ", {"XYZ"})
        self.assertFalse(result.passed)

    def test_non_excluded_ticker_passes(self):
        result = f5_exclusions("AAPL", {"XYZ"})
        self.assertTrue(result.passed)


class TestF6Earnings(unittest.TestCase):
    def test_existing_holding_never_excluded(self):
        result = f6_earnings(date(2026, 9, 2), date(2026, 9, 1), is_new_entry=False)
        self.assertTrue(result.passed)

    def test_new_entry_within_blackout_fails(self):
        result = f6_earnings(date(2026, 9, 3), date(2026, 9, 1), is_new_entry=True)
        self.assertFalse(result.passed)

    def test_new_entry_outside_blackout_passes(self):
        result = f6_earnings(date(2026, 9, 20), date(2026, 9, 1), is_new_entry=True)
        self.assertTrue(result.passed)

    def test_new_entry_no_earnings_date_passes(self):
        result = f6_earnings(None, date(2026, 9, 1), is_new_entry=True)
        self.assertTrue(result.passed)


class TestRunAll(unittest.TestCase):
    def _good_candidate_kwargs(self) -> Dict[str, Any]:
        return dict(
            ticker="AAPL",
            adj_close=pd.Series(np.linspace(100, 150, 300)),
            volume=pd.Series([2_000_000] * 300),
            signal_window=pd.Series(np.linspace(100, 150, 250)),
            fcf_ttm=100.0,
            net_income_ttm=50.0,
            excluded=set(),
            next_earnings=None,
            execution_date=date(2026, 9, 1),
            is_new_entry=True,
        )

    def test_all_pass(self):
        outcome = run_all(**self._good_candidate_kwargs())
        self.assertTrue(outcome.passed)
        self.assertIn("F4", outcome.filters)  # F4 always recorded

    def test_f1_failure_short_circuits_remaining_filters(self):
        kwargs = self._good_candidate_kwargs()
        kwargs["adj_close"] = pd.Series(range(50))  # too short
        outcome = run_all(**kwargs)
        self.assertFalse(outcome.passed)
        self.assertIn("F1", outcome.filters)
        self.assertNotIn("F2", outcome.filters)  # short-circuited

    def test_f4_always_recorded_even_when_other_filters_fail(self):
        kwargs = self._good_candidate_kwargs()
        kwargs["excluded"] = {"AAPL"}
        outcome = run_all(**kwargs)
        self.assertFalse(outcome.passed)
        self.assertIn("F4", outcome.filters)
        self.assertIn("F5", outcome.filters)
        self.assertFalse(outcome.filters["F5"].passed)


class TestTallyF4MissingPct(unittest.TestCase):
    def test_empty_list_returns_zero(self):
        self.assertEqual(tally_f4_missing_pct([]), 0.0)

    def test_computes_fraction(self):
        kwargs = TestRunAll()._good_candidate_kwargs()
        outcome_missing = run_all(**{**kwargs, "fcf_ttm": None, "net_income_ttm": None})
        outcome_present = run_all(**kwargs)
        pct = tally_f4_missing_pct([outcome_missing, outcome_present])
        self.assertAlmostEqual(pct, 0.5)


if __name__ == "__main__":
    unittest.main()
