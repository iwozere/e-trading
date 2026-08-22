"""Unit tests for src.ml.pipeline.p21_momentum.strategy.tracks."""

from __future__ import annotations

import unittest

from src.ml.pipeline.p21_momentum.strategy.tracks import (
    apply_ter_drag,
    build_nav_series,
    compute_attribution,
)


class TestApplyTerDrag(unittest.TestCase):
    def test_drag_reduces_every_return(self):
        rets = {"2026-01-01": 0.01, "2026-01-02": -0.005}
        adjusted = apply_ter_drag(rets, annual_ter=0.0005)
        daily_drag = 0.0005 / 252
        self.assertAlmostEqual(adjusted["2026-01-01"], 0.01 - daily_drag)
        self.assertAlmostEqual(adjusted["2026-01-02"], -0.005 - daily_drag)


class TestBuildNavSeries(unittest.TestCase):
    def test_compounds_correctly(self):
        rets = {"d1": 0.10, "d2": -0.05}
        nav = build_nav_series(rets, initial_nav=1000.0)
        self.assertAlmostEqual(nav["d1"], 1100.0)
        self.assertAlmostEqual(nav["d2"], 1100.0 * 0.95)

    def test_empty_returns_empty_nav(self):
        self.assertEqual(build_nav_series({}, 1000.0), {})


class TestComputeAttribution(unittest.TestCase):
    def test_decomposition_matches_spec_formulas(self):
        initial = 250_000.0
        result = compute_attribution(
            nav_a=260_000, nav_b=255_000, nav_c=252_000, nav_d=257_000, nav_e=258_000,
            as_of="2026-08-31", initial_nav=initial,
        )
        ra = 260_000 / initial - 1.0
        rb = 255_000 / initial - 1.0
        rc = 252_000 / initial - 1.0
        rd = 257_000 / initial - 1.0
        self.assertAlmostEqual(result.stock_selection_effect, rb - rc)
        self.assertAlmostEqual(result.overlay_effect_on_stocks, ra - rb)
        self.assertAlmostEqual(result.overlay_effect_on_etf, rd - rc)
        self.assertAlmostEqual(result.total_diy_benefit, ra - rd)

    def test_all_tracks_equal_gives_zero_everywhere(self):
        result = compute_attribution(
            nav_a=250_000, nav_b=250_000, nav_c=250_000, nav_d=250_000, nav_e=250_000,
            as_of="2026-08-31", initial_nav=250_000,
        )
        self.assertAlmostEqual(result.stock_selection_effect, 0.0)
        self.assertAlmostEqual(result.overlay_effect_on_stocks, 0.0)
        self.assertAlmostEqual(result.overlay_effect_on_etf, 0.0)
        self.assertAlmostEqual(result.total_diy_benefit, 0.0)


if __name__ == "__main__":
    unittest.main()
