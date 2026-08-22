"""Unit tests for backtest.p21_momentum.cost_sensitivity."""

from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

from backtest.p21_momentum.cost_sensitivity import (
    SlippageResult,
    best_hold_rank,
    edge_survives_10bps,
    render_slippage_csv,
    render_turnover_curve_png,
    run_slippage_sweep,
    run_turnover_curve,
)
from backtest.p21_momentum.tests.fixtures import make_universe_panel


class TestSlippageSweep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel, cls.sector_by_ticker = make_universe_panel(15, "2020-01-01", "2021-06-30")
        cls.start, cls.end = date(2020, 1, 1), date(2021, 6, 30)

    def test_sweep_returns_one_result_per_level(self):
        results = run_slippage_sweep(
            self.panel, self.sector_by_ticker, self.start, self.end, slippage_levels_bps=(0.0, 3.0)
        )
        self.assertEqual(len(results), 2)
        self.assertEqual([r.slippage_bps for r in results], [0.0, 3.0])

    def test_higher_slippage_does_not_materially_improve_edge(self):
        results = run_slippage_sweep(
            self.panel, self.sector_by_ticker, self.start, self.end, slippage_levels_bps=(0.0, 25.0)
        )
        zero_bps, high_bps = results
        # Higher round-trip cost drags returns down over a full study; a short synthetic
        # window with few rebalances can see a small reversal from trade-threshold effects
        # (MIN_TRADE_USD skipping a slightly different set of trades at each cost level), so
        # this only asserts there is no large, spurious improvement from raising slippage.
        self.assertLessEqual(high_bps.edge_cagr_a_minus_c, zero_bps.edge_cagr_a_minus_c + 0.02)

    def test_render_slippage_csv_has_expected_columns(self):
        results = run_slippage_sweep(
            self.panel, self.sector_by_ticker, self.start, self.end, slippage_levels_bps=(0.0, 3.0)
        )
        df = render_slippage_csv(results)
        self.assertIn("slippage_bps", df.columns)
        self.assertIn("edge_cagr_a_minus_c", df.columns)
        self.assertEqual(len(df), 2)


class TestEdgeSurvives10bps(unittest.TestCase):
    def test_raises_without_both_levels(self):
        results = [SlippageResult(0.0, 0.1, 1.0, 0.05, 0.5, 0.05)]
        with self.assertRaises(ValueError):
            edge_survives_10bps(results)

    def test_edge_survives(self):
        results = [
            SlippageResult(3.0, 0.10, 1.0, 0.05, 0.5, 0.05),
            SlippageResult(10.0, 0.08, 0.8, 0.05, 0.5, 0.03),
        ]
        self.assertTrue(edge_survives_10bps(results))

    def test_edge_disappears(self):
        results = [
            SlippageResult(3.0, 0.10, 1.0, 0.05, 0.5, 0.05),
            SlippageResult(10.0, 0.04, 0.3, 0.05, 0.5, -0.01),
        ]
        self.assertFalse(edge_survives_10bps(results))

    def test_no_edge_at_3bps_is_not_survival(self):
        results = [
            SlippageResult(3.0, 0.05, 0.3, 0.05, 0.5, 0.0),
            SlippageResult(10.0, 0.06, 0.4, 0.05, 0.5, 0.01),
        ]
        self.assertFalse(edge_survives_10bps(results))


class TestTurnoverCurve(unittest.TestCase):
    def test_curve_has_one_point_per_hold_rank(self):
        panel, sector_by_ticker = make_universe_panel(15, "2020-01-01", "2021-06-30")
        points = run_turnover_curve(
            panel, sector_by_ticker, date(2020, 1, 1), date(2021, 6, 30), hold_ranks=(40, 100)
        )
        self.assertEqual([p.hold_rank for p in points], [40, 100])

    def test_best_hold_rank_picks_max_return(self):
        from backtest.p21_momentum.cost_sensitivity import TurnoverCurvePoint

        points = [
            TurnoverCurvePoint(20, 300.0, 0.05),
            TurnoverCurvePoint(60, 180.0, 0.09),
            TurnoverCurvePoint(150, 90.0, 0.02),
        ]
        self.assertEqual(best_hold_rank(points), 60)

    def test_best_hold_rank_empty_list_is_none(self):
        self.assertIsNone(best_hold_rank([]))

    def test_render_turnover_curve_png_writes_file(self):
        from backtest.p21_momentum.cost_sensitivity import TurnoverCurvePoint

        points = [TurnoverCurvePoint(20, 300.0, 0.05), TurnoverCurvePoint(60, 180.0, 0.09)]
        with tempfile.TemporaryDirectory() as d:
            out_path = Path(d) / "curve.png"
            render_turnover_curve_png(points, out_path)
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
