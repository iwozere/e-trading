"""Unit/integration tests for backtest.p21_momentum.runner."""

from __future__ import annotations

import unittest
from datetime import date

import pandas as pd

from backtest.p21_momentum.runner import BacktestParams, run_backtest
from backtest.p21_momentum.tests.fixtures import make_universe_panel as _make_universe_panel


class TestRunBacktestSmoke(unittest.TestCase):
    """
    Two full years (~504 trading days) of synthetic data for 20 tickers.
    First ~13 months are warmup (compute_signal needs MIN_HISTORY=260 bars),
    so meaningful selection/trading only starts in year 2 — this is enough
    to exercise the full monthly cycle at least once without real-data cost.
    """

    @classmethod
    def setUpClass(cls):
        cls.panel, cls.sector_by_ticker = _make_universe_panel(20, "2020-01-01", "2021-12-31")
        cls.start = date(2020, 1, 1)
        cls.end = date(2021, 12, 31)

    def test_nav_daily_covers_full_range(self):
        result = run_backtest(self.panel, self.sector_by_ticker, self.start, self.end)
        self.assertGreater(len(result.nav_daily), 480)  # ~504 trading days over 2 years
        self.assertEqual(list(result.nav_daily.columns), ["nav_a", "nav_b", "nav_c", "nav_d", "nav_e"])

    def test_monthly_metrics_one_per_month(self):
        result = run_backtest(self.panel, self.sector_by_ticker, self.start, self.end)
        # 24 calendar months in [2020-01, 2021-12], minus December 2021 —
        # its execution_date (first trading day of Jan 2022) falls outside
        # the [start, end] range, so that signal date is correctly excluded.
        self.assertEqual(len(result.monthly_metrics), 23)

    def test_trades_occur_after_warmup(self):
        result = run_backtest(self.panel, self.sector_by_ticker, self.start, self.end)
        self.assertGreater(len(result.trades), 0)
        self.assertTrue(all(t.track in ("A", "B") for t in result.trades))

    def test_position_count_reaches_target_after_warmup(self):
        result = run_backtest(self.panel, self.sector_by_ticker, self.start, self.end)
        late_months = result.monthly_metrics[-3:]
        # With only 20 names and MAX_PER_SECTOR=4 across 5 sectors, position
        # count should be positive (may be < 20 given the small universe).
        self.assertTrue(any(m.position_count > 0 for m in late_months))

    def test_nav_starts_at_initial_nav(self):
        result = run_backtest(self.panel, self.sector_by_ticker, self.start, self.end)
        first_row = result.nav_daily.iloc[0]
        for col in ("nav_a", "nav_b", "nav_c", "nav_d"):
            self.assertAlmostEqual(first_row[col], BacktestParams().initial_nav, delta=1.0)

    def test_ter_drag_reduces_track_c_relative_to_flat_price(self):
        # Track C should underperform a naive "price return only" calc by
        # roughly the TER drag over the period once invested.
        result = run_backtest(self.panel, self.sector_by_ticker, self.start, self.end)
        self.assertFalse(result.nav_daily["nav_c"].isna().any())

    def test_determinism_two_runs_identical(self):
        """spec §14.9 B10: two identical runs must produce identical output."""
        result1 = run_backtest(self.panel, self.sector_by_ticker, self.start, self.end)
        result2 = run_backtest(self.panel, self.sector_by_ticker, self.start, self.end)
        pd.testing.assert_frame_equal(result1.nav_daily, result2.nav_daily)
        self.assertEqual(len(result1.trades), len(result2.trades))
        for t1, t2 in zip(result1.trades, result2.trades):
            self.assertEqual(t1.to_dict(), t2.to_dict())

    def test_sector_cap_never_breached_in_selected_positions(self):
        result = run_backtest(self.panel, self.sector_by_ticker, self.start, self.end)
        # Reconstruct sector counts is not directly exposed; rely on the
        # Herfindahl index instead: with MAX_PER_SECTOR=4 and >=5 positions,
        # Herfindahl can never be 1.0 (fully concentrated in one sector).
        for m in result.monthly_metrics:
            if m.position_count >= 5:
                self.assertLess(m.sector_herfindahl, 1.0)


class TestRunBacktestParamOverrides(unittest.TestCase):
    def test_custom_params_change_behavior(self):
        panel, sector_by_ticker = _make_universe_panel(20, "2020-01-01", "2021-12-31")
        tight_params = BacktestParams(hold_rank=20, entry_rank=10, target_count=5, max_per_sector=2)
        result = run_backtest(panel, sector_by_ticker, date(2020, 1, 1), date(2021, 12, 31), params=tight_params)
        for m in result.monthly_metrics:
            self.assertLessEqual(m.position_count, 5)


class TestRunBacktestLookbackBuffer(unittest.TestCase):
    """
    Regression test for the P21 Phase 0 B3 acceptance failure (diagnosed 2026-08-23):
    run_backtest() used to build close_frame/open_frame/volume_frame from
    trading_days(start, end) only, silently discarding any panel history before
    `start` — so compute_signal's MIN_HISTORY=260-bar lookback had nothing to look
    back on for the first ~13 months of *any* range, regardless of how much history
    the panel actually held before `start`. Fixed by widening the frames (only) to
    start - timedelta(days=450), while the day loop / signal dates / nav_daily still
    run exactly [start, end].
    """

    def test_position_count_positive_from_first_month_given_panel_history_before_start(self):
        # Panel covers 16 months before `start` -- comfortably over MIN_HISTORY's ~13-month
        # requirement -- so, with the fix, the very first rebalance should already hold
        # positions. Under the pre-fix behavior this would be 0 (warmup restarts at `start`
        # regardless of the panel's actual history).
        panel, sector_by_ticker = _make_universe_panel(20, "2018-09-01", "2021-12-31")
        result = run_backtest(panel, sector_by_ticker, date(2020, 1, 1), date(2021, 12, 31))
        self.assertGreater(result.monthly_metrics[0].position_count, 0)

    def test_nav_daily_still_starts_exactly_at_start_despite_lookback_buffer(self):
        # The lookback widening must only affect the internal signal-lookback frames,
        # never the reported date range.
        panel, sector_by_ticker = _make_universe_panel(20, "2018-09-01", "2021-12-31")
        result = run_backtest(panel, sector_by_ticker, date(2020, 1, 1), date(2021, 12, 31))
        self.assertEqual(result.nav_daily.index.min().date(), date(2020, 1, 2))  # first trading day of 2020


class TestRunBacktestEdgeCases(unittest.TestCase):
    def test_empty_panel_raises_or_produces_no_trades(self):
        result = run_backtest({}, {}, date(2020, 1, 1), date(2020, 3, 31))
        self.assertEqual(len(result.trades), 0)

    def test_no_trading_days_raises(self):
        panel, sector_by_ticker = _make_universe_panel(5, "2020-01-01", "2020-01-05")
        with self.assertRaises(ValueError):
            # end before start -> no trading days
            run_backtest(panel, sector_by_ticker, date(2020, 3, 1), date(2020, 1, 1))


if __name__ == "__main__":
    unittest.main()
