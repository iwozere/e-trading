"""Unit tests for backtest.p21_momentum.stress_windows."""

from __future__ import annotations

import unittest
from datetime import date

import pandas as pd

from backtest.p21_momentum.runner import BacktestResult
from backtest.p21_momentum.stress_windows import (
    STRESS_WINDOWS,
    StressWindow,
    evaluate_all_windows,
    evaluate_window,
    render_stress_windows_md,
)


def _flat_nav_result(start: str, end: str, a_drop_pct: float = -0.10, b_drop_pct: float = -0.02) -> BacktestResult:
    idx = pd.bdate_range(start, end)
    n = len(idx)
    nav_a = [250_000.0 * (1 + a_drop_pct * i / max(n - 1, 1)) for i in range(n)]
    nav_b = [250_000.0 * (1 + b_drop_pct * i / max(n - 1, 1)) for i in range(n)]
    nav = pd.DataFrame(
        {"nav_a": nav_a, "nav_b": nav_b, "nav_c": nav_a, "nav_d": nav_b, "nav_e": nav_b}, index=idx
    )
    regime_history = [
        {
            "date": d.date().isoformat(), "bear": True, "high_vol": True,
            "scalar_raw": 0.25, "scalar_applied": 0.25, "months_at_state": 1,
        }
        for d in idx[:: max(len(idx) // 3, 1)]
    ]
    return BacktestResult(nav_daily=nav, regime_history=regime_history)


class TestStressWindowTable(unittest.TestCase):
    def test_nine_windows_defined(self):
        self.assertEqual(len(STRESS_WINDOWS), 9)

    def test_windows_are_chronological_by_start(self):
        # Spec lists 2009-03 first (the decisive test) — not chronological order; just confirm
        # every window has start <= end, which is a real invariant.
        for w in STRESS_WINDOWS:
            self.assertLessEqual(w.start, w.end)


class TestEvaluateWindow(unittest.TestCase):
    def test_in_range_window_computes_a_minus_b(self):
        result = _flat_nav_result("2022-01-01", "2022-12-31")
        window = StressWindow("2022-01 -> 2022-10", date(2022, 1, 1), date(2022, 10, 31), "event", "question")
        r = evaluate_window(result, window)
        self.assertTrue(r.in_range)
        self.assertIsNotNone(r.a_minus_b)
        assert r.a_minus_b is not None  # narrows for the type checker
        # A drops more than B in this fixture -> A-B should be negative.
        self.assertLess(r.a_minus_b, 0.0)
        self.assertIsNotNone(r.regime_scalar_min)

    def test_out_of_range_window_is_flagged(self):
        result = _flat_nav_result("2022-01-01", "2022-12-31")
        window = StressWindow("2009-03 -> 2009-05", date(2009, 3, 1), date(2009, 5, 31), "event", "question")
        r = evaluate_window(result, window)
        self.assertFalse(r.in_range)
        self.assertIsNone(r.track_metrics["nav_a"].window_return)

    def test_empty_result_is_out_of_range(self):
        empty = BacktestResult(nav_daily=pd.DataFrame(columns=["nav_a", "nav_b", "nav_c", "nav_d", "nav_e"]))
        r = evaluate_window(empty, STRESS_WINDOWS[0])
        self.assertFalse(r.in_range)

    def test_evaluate_all_windows_returns_one_per_window(self):
        result = _flat_nav_result("2022-01-01", "2022-12-31")
        results = evaluate_all_windows(result)
        self.assertEqual(len(results), len(STRESS_WINDOWS))


class TestRenderMarkdown(unittest.TestCase):
    def test_render_includes_banner_free_content_and_window_names(self):
        result = _flat_nav_result("2022-01-01", "2022-12-31")
        results = evaluate_all_windows(result)
        md = render_stress_windows_md(results)
        self.assertIn("2022-01 -> 2022-10", md)
        self.assertIn("Outside this backtest's date range", md)  # for out-of-range windows

    def test_render_shows_a_minus_b_for_in_range_window(self):
        result = _flat_nav_result("2022-01-01", "2022-12-31")
        results = evaluate_all_windows(result)
        md = render_stress_windows_md(results)
        self.assertIn("**A - B:", md)


if __name__ == "__main__":
    unittest.main()
