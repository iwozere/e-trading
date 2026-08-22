"""Unit tests for src.ml.pipeline.p21_momentum.strategy.sizing."""

from __future__ import annotations

import unittest

from src.ml.pipeline.p21_momentum.strategy.sizing import shares_from_allocation, size_positions


class TestSizePositions(unittest.TestCase):
    def test_empty_input_returns_empty(self):
        self.assertEqual(size_positions({}), {})

    def test_sums_to_sleeve_usd_when_uncapped(self):
        vols = {f"T{i}": 0.20 for i in range(20)}  # equal vol -> equal weight
        result = size_positions(vols, nav_total=250_000, sleeve_pct=0.20, max_pos_pct=0.10)  # generous cap
        self.assertAlmostEqual(sum(result.values()), 250_000 * 0.20, delta=1.0)
        # Equal vol -> equal allocation
        values = list(result.values())
        self.assertAlmostEqual(max(values), min(values), delta=0.01)

    def test_cap_is_off_total_nav_not_sleeve(self):
        # 5 names, wildly unequal vol so one dominates inverse-vol weighting
        vols = {"LOW_VOL": 0.05, "A": 0.30, "B": 0.30, "C": 0.30, "D": 0.30}
        nav_total = 250_000
        max_pos_pct = 0.01  # $2,500 cap, off NAV not sleeve ($50,000)
        result = size_positions(vols, nav_total=nav_total, sleeve_pct=0.20, max_pos_pct=max_pos_pct)
        cap_usd = nav_total * max_pos_pct
        for v in result.values():
            self.assertLessEqual(v, cap_usd + 1e-6)

    def test_regime_scalar_scales_down_sleeve(self):
        vols = {f"T{i}": 0.20 for i in range(10)}
        full = size_positions(vols, nav_total=250_000, sleeve_pct=0.20, max_pos_pct=0.10, regime_scalar=1.0)
        scaled = size_positions(vols, nav_total=250_000, sleeve_pct=0.20, max_pos_pct=0.10, regime_scalar=0.25)
        self.assertAlmostEqual(sum(scaled.values()), sum(full.values()) * 0.25, delta=1.0)

    def test_converges_within_max_iterations(self):
        # Many tightly-capped names should still converge to a stable allocation
        vols = {f"T{i}": 0.01 + i * 0.001 for i in range(50)}
        result = size_positions(vols, nav_total=250_000, sleeve_pct=0.20, max_pos_pct=0.01, max_iterations=10)
        self.assertEqual(len(result), 50)
        self.assertLessEqual(sum(result.values()), 250_000 * 0.20 + 1.0)


class TestSharesFromAllocation(unittest.TestCase):
    def test_basic_conversion_rounds_to_4_decimals(self):
        shares = shares_from_allocation({"AAPL": 2500.0}, {"AAPL": 175.35})
        self.assertEqual(shares["AAPL"], round(2500.0 / 175.35, 4))

    def test_missing_price_is_omitted(self):
        shares = shares_from_allocation({"AAPL": 2500.0, "MISSING": 1000.0}, {"AAPL": 100.0})
        self.assertIn("AAPL", shares)
        self.assertNotIn("MISSING", shares)

    def test_zero_or_negative_price_is_omitted(self):
        shares = shares_from_allocation({"AAPL": 2500.0}, {"AAPL": 0.0})
        self.assertNotIn("AAPL", shares)


if __name__ == "__main__":
    unittest.main()
