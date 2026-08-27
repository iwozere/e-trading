"""Unit tests for backtest.p21_momentum.run_oos_check."""

from __future__ import annotations

import unittest
from datetime import date

from backtest.p21_momentum.run_oos_check import (
    GridSummary,
    OosPointEstimate,
    render_oos_report_md,
    run_oos_point_estimate,
    summarize_grid,
)
from backtest.p21_momentum.tests.fixtures import make_universe_panel


class TestSummarizeGrid(unittest.TestCase):
    def test_counts_and_band_scale_with_inputs(self):
        summary = summarize_grid([0.5, 0.6, 0.7, 5.0], date(2017, 1, 1), date(2026, 6, 30))
        self.assertEqual(summary.n_trials, 4)
        self.assertEqual(summary.n_observations, 113)  # (2026-2017)*12 + (6-1)
        self.assertEqual(summary.best_sharpe, 5.0)
        self.assertGreater(summary.band_low, 0.0)
        self.assertLessEqual(summary.band_low, summary.band_high)

    def test_empty_grid_is_degenerate_not_a_crash(self):
        summary = summarize_grid([], date(2017, 1, 1), date(2026, 6, 30))
        self.assertEqual(summary.n_trials, 0)
        self.assertEqual(summary.best_sharpe, 0.0)
        self.assertFalse(summary.separated)

    def test_clear_outlier_is_separated(self):
        summary = summarize_grid([0.5] * 19 + [5.0], date(2017, 1, 1), date(2026, 6, 30))
        self.assertTrue(summary.separated)

    def test_flat_surface_not_separated(self):
        summary = summarize_grid([1.0] * 20, date(2017, 1, 1), date(2026, 6, 30))
        self.assertFalse(summary.separated)


class TestRunOosPointEstimate(unittest.TestCase):
    def test_returns_populated_estimate_over_given_window(self):
        panel, sector_by_ticker = make_universe_panel(10, "2020-01-01", "2021-06-30")
        point = run_oos_point_estimate(panel, sector_by_ticker, start=date(2020, 1, 1), end=date(2021, 6, 30))
        self.assertIsInstance(point, OosPointEstimate)
        self.assertIsInstance(point.edge_survives_10bps, bool)
        self.assertGreaterEqual(point.turnover_annualized_median_pct, 0.0)


class TestRenderOosReportMd(unittest.TestCase):
    def _summary(self, separated: bool, best: float = 0.9) -> GridSummary:
        return GridSummary(n_trials=729, n_observations=113, band_low=0.8, band_high=1.0, best_sharpe=best, separated=separated)

    def _point(self, edge_survives) -> OosPointEstimate:
        return OosPointEstimate(
            cagr_a=0.10, sharpe_a=0.9, cagr_c=0.08, sharpe_c=0.7,
            edge_cagr_a_minus_c=0.02, edge_survives_10bps=edge_survives,
            turnover_annualized_median_pct=150.0,
        )

    def test_header_and_discipline_note_present(self):
        md = render_oos_report_md(self._summary(False), self._summary(False), self._point(True))
        self.assertIn("# P21 Momentum — Out-of-Sample Check", md)
        self.assertIn("## Discipline note", md)
        self.assertIn("must not be evaluated again", md)

    def test_matching_separation_reports_not_contradicted(self):
        md = render_oos_report_md(self._summary(False), self._summary(False), self._point(True))
        self.assertIn("not contradicted by this out-of-sample look", md)
        self.assertNotIn("disagrees", md)

    def test_disagreeing_separation_reports_disagreement(self):
        md = render_oos_report_md(self._summary(True), self._summary(False), self._point(True))
        self.assertIn("**disagrees**", md)
        self.assertIn("not a reason to retune", md)

    def test_edge_survives_true_reports_survives(self):
        md = render_oos_report_md(self._summary(False), self._summary(False), self._point(True))
        self.assertIn("survives realistic costs", md)

    def test_edge_survives_false_reports_caveat(self):
        md = render_oos_report_md(self._summary(False), self._summary(False), self._point(False))
        self.assertIn("does not survive realistic costs", md)

    def test_edge_survives_none_omits_edge_lines(self):
        md = render_oos_report_md(self._summary(False), self._summary(False), self._point(None))
        self.assertIn("N/A", md)
        self.assertNotIn("survives realistic costs", md)
        self.assertNotIn("does not survive realistic costs", md)


if __name__ == "__main__":
    unittest.main()
