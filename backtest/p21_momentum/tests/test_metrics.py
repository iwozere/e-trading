"""Unit tests for backtest.p21_momentum.metrics."""

from __future__ import annotations

import unittest
from datetime import date

import numpy as np
import pandas as pd

from backtest.p21_momentum.metrics import (
    compute_holding_periods_days,
    compute_mechanical_metrics,
    compute_return_metrics,
    compute_risk_metrics,
    compute_rolling_beta_corr,
    compute_rolling_tracking_error,
)
from backtest.p21_momentum.runner import BacktestResult, MonthlyMetrics, run_backtest
from backtest.p21_momentum.tests.fixtures import make_universe_panel
from src.ml.pipeline.p21_momentum.schemas import LedgerEntry


def _nav_series(values: list, start: str = "2020-01-01") -> pd.Series:
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=idx)


class TestRiskMetrics(unittest.TestCase):
    def test_degenerate_series_returns_zeros(self):
        m = compute_risk_metrics(_nav_series([100.0]))
        self.assertEqual(m.annualized_vol, 0.0)
        self.assertIsNone(m.time_to_recovery_days)

    def test_max_drawdown_and_recovery(self):
        # Up to 110, down to 88 (-20% from peak), recovers to 112.
        nav = _nav_series([100, 105, 110, 99, 88, 95, 105, 112])
        m = compute_risk_metrics(nav)
        self.assertAlmostEqual(m.max_drawdown, 88 / 110 - 1.0, places=6)
        self.assertGreater(m.max_drawdown_duration_days, 0)
        self.assertIsNotNone(m.time_to_recovery_days)

    def test_never_recovers_time_to_recovery_is_none(self):
        nav = _nav_series([100, 110, 90, 85, 80])
        m = compute_risk_metrics(nav)
        self.assertIsNone(m.time_to_recovery_days)

    def test_monotonic_series_has_zero_drawdown(self):
        nav = _nav_series([100 + i for i in range(30)])
        m = compute_risk_metrics(nav)
        self.assertAlmostEqual(m.max_drawdown, 0.0, places=9)
        self.assertEqual(m.downside_deviation, 0.0)


class TestReturnMetrics(unittest.TestCase):
    def test_degenerate_series_returns_zeros(self):
        m = compute_return_metrics(_nav_series([100.0]), _nav_series([100.0]))
        self.assertEqual(m.cagr, 0.0)
        self.assertEqual(m.sharpe, 0.0)

    def test_monotonic_growth_has_positive_cagr_and_sharpe(self):
        rng = np.random.default_rng(1)
        n = 500
        rets = rng.normal(0.0006, 0.008, n)
        nav = pd.Series(100.0 * np.cumprod(1 + rets), index=pd.bdate_range("2020-01-01", periods=n))
        nav_c = pd.Series(100.0 * np.cumprod(1 + rets * 0.5), index=nav.index)
        m = compute_return_metrics(nav, nav_c)
        self.assertGreater(m.cagr, 0.0)
        self.assertGreater(m.sharpe, 0.0)
        self.assertGreaterEqual(m.hit_rate_vs_c_monthly, 0.0)
        self.assertLessEqual(m.hit_rate_vs_c_monthly, 1.0)

    def test_information_ratio_zero_when_tracks_identical(self):
        idx = pd.bdate_range("2020-01-01", periods=300)
        rng = np.random.default_rng(2)
        rets = rng.normal(0.0003, 0.01, 300)
        nav = pd.Series(100.0 * np.cumprod(1 + rets), index=idx)
        m = compute_return_metrics(nav, nav.copy())
        self.assertAlmostEqual(m.information_ratio_vs_c, 0.0, places=6)


class TestRollingMetrics(unittest.TestCase):
    def test_rolling_tracking_error_shape(self):
        idx = pd.bdate_range("2020-01-01", periods=400)
        nav = pd.Series(np.linspace(100, 150, 400), index=idx)
        nav_c = pd.Series(np.linspace(100, 140, 400), index=idx)
        te = compute_rolling_tracking_error(nav, nav_c, window_days=60)
        self.assertEqual(len(te), 400)
        self.assertTrue(te.iloc[60:].notna().any())

    def test_rolling_beta_corr_shape(self):
        idx = pd.bdate_range("2020-01-01", periods=400)
        rng = np.random.default_rng(3)
        market_rets = rng.normal(0.0003, 0.01, 400)
        nav_market = pd.Series(100.0 * np.cumprod(1 + market_rets), index=idx)
        nav = pd.Series(100.0 * np.cumprod(1 + market_rets * 1.5), index=idx)
        df = compute_rolling_beta_corr(nav, nav_market, window_days=60)
        self.assertIn("beta", df.columns)
        self.assertIn("corr", df.columns)
        # A 1.5x-levered clone of the market should show beta near 1.5 once the window fills.
        self.assertAlmostEqual(float(df["beta"].iloc[-1]), 1.5, delta=0.2)


class TestHoldingPeriods(unittest.TestCase):
    def _entry(self, ts: str, ticker: str, reason: str) -> LedgerEntry:
        return LedgerEntry(
            ts=ts, track="A", ticker=ticker, side="BUY", shares=10, ref_open=100, fill_price=100,
            slippage_bps=3.0, commission_usd=1.0, gross_usd=1000, net_usd=1001, reason=reason,
        )

    def test_matches_entry_to_next_exit(self):
        trades = [
            self._entry("2020-01-02", "AAA", "ENTRY_RANK_1"),
            self._entry("2020-02-03", "AAA", "EXIT_RANK_DROP"),
        ]
        periods = compute_holding_periods_days(trades, track="A")
        self.assertEqual(periods, [(date(2020, 2, 3) - date(2020, 1, 2)).days])

    def test_rebal_add_does_not_close_or_reopen(self):
        trades = [
            self._entry("2020-01-02", "AAA", "ENTRY_RANK_1"),
            self._entry("2020-01-15", "AAA", "REBAL_ADD"),
            self._entry("2020-03-01", "AAA", "EXIT_RANK_DROP"),
        ]
        periods = compute_holding_periods_days(trades, track="A")
        self.assertEqual(periods, [(date(2020, 3, 1) - date(2020, 1, 2)).days])

    def test_still_held_at_end_contributes_no_observation(self):
        trades = [self._entry("2020-01-02", "AAA", "ENTRY_RANK_1")]
        self.assertEqual(compute_holding_periods_days(trades, track="A"), [])

    def test_other_track_ignored(self):
        trades = [
            self._entry("2020-01-02", "AAA", "ENTRY_RANK_1"),
            self._entry("2020-02-01", "AAA", "EXIT_RANK_DROP"),
        ]
        self.assertEqual(compute_holding_periods_days(trades, track="B"), [])


class TestMechanicalMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        panel, sector_by_ticker = make_universe_panel(20, "2020-01-01", "2021-12-31")
        cls.result = run_backtest(panel, sector_by_ticker, date(2020, 1, 1), date(2021, 12, 31))

    def test_basic_fields_populated(self):
        m = compute_mechanical_metrics(self.result)
        self.assertGreaterEqual(m.position_count_mean, 0.0)
        self.assertGreaterEqual(m.turnover_annualized_median_pct, 0.0)
        self.assertEqual(m.warn_underfilled_count, sum(1 for mm in self.result.monthly_metrics if mm.warn_underfilled))
        self.assertGreaterEqual(m.max_sector_count_ever, 0)

    def test_universe_size_produces_percentages(self):
        m = compute_mechanical_metrics(self.result, universe_size=20)
        self.assertIsNotNone(m.f1_removed_pct_of_universe)

    def test_no_universe_size_omits_percentages(self):
        m = compute_mechanical_metrics(self.result)
        self.assertIsNone(m.f1_removed_pct_of_universe)

    def test_regime_histogram_sums_to_regime_history_length(self):
        m = compute_mechanical_metrics(self.result)
        self.assertEqual(sum(m.regime_scalar_histogram.values()), len(self.result.regime_history))

    def test_empty_result_does_not_raise(self):
        empty = BacktestResult(nav_daily=pd.DataFrame(columns=["nav_a", "nav_b", "nav_c", "nav_d", "nav_e"]))
        m = compute_mechanical_metrics(empty)
        self.assertEqual(m.turnover_annualized_median_pct, 0.0)
        self.assertEqual(m.position_count_mean, 0.0)

    def test_single_month_metrics_direct(self):
        row = MonthlyMetrics(
            month="2020-01", turnover_two_way_usd=1000.0, position_count=15, sector_herfindahl=0.2,
            f1_removed=3, f2_removed=1, f3_removed=0, regime_scalar=1.0, warn_underfilled=True,
            manual_review_count=2, exit_delisted_count=1, max_sector_count=4,
        )
        nav_daily = pd.DataFrame(
            {
                "nav_a": [250_000.0], "nav_b": [250_000.0], "nav_c": [250_000.0],
                "nav_d": [250_000.0], "nav_e": [250_000.0],
            },
            index=pd.DatetimeIndex([pd.Timestamp("2020-01-31")]),
        )
        result = BacktestResult(nav_daily=nav_daily, monthly_metrics=[row])
        m = compute_mechanical_metrics(result)
        self.assertEqual(m.warn_underfilled_count, 1)
        self.assertEqual(m.manual_review_count_total, 2)
        self.assertEqual(m.exit_delisted_count_total, 1)
        self.assertEqual(m.max_sector_count_ever, 4)


if __name__ == "__main__":
    unittest.main()
