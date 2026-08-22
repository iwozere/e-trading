"""Unit tests for backtest.p21_momentum.phase0_report."""

from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

import pandas as pd

from backtest.p21_momentum.cost_sensitivity import SlippageResult
from backtest.p21_momentum.phase0_report import (
    UNIVERSE_BANNER,
    AcceptanceRow,
    _write_base_case_deliverables,
    evaluate_acceptance_table,
    render_phase0_report_md,
    verify_determinism,
)
from backtest.p21_momentum.runner import run_backtest
from backtest.p21_momentum.stress_windows import STRESS_WINDOWS, evaluate_all_windows
from backtest.p21_momentum.tests.fixtures import make_universe_panel


class TestAcceptanceTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        panel, sector_by_ticker = make_universe_panel(20, "2020-01-01", "2021-12-31")
        cls.result = run_backtest(panel, sector_by_ticker, date(2020, 1, 1), date(2021, 12, 31))
        cls.stress_results = evaluate_all_windows(cls.result)

    def test_ten_rows_in_bxx_order(self):
        rows = evaluate_acceptance_table(self.result, self.stress_results, [], 60.0, None)
        self.assertEqual([r.id for r in rows], [f"B{i}" for i in range(1, 11)])

    def test_windows_outside_range_are_not_evaluated(self):
        # This fixture's range (2020-2021) doesn't cover 2009 or 2022 — B5/B6/B7 must be N/A,
        # not silently scored as pass/fail against no data.
        rows = evaluate_acceptance_table(self.result, self.stress_results, [], 60.0, None)
        by_id = {r.id: r for r in rows}
        self.assertIsNone(by_id["B5"].passed)
        self.assertIsNone(by_id["B6"].passed)

    def test_b9_runtime_threshold(self):
        rows_fast = evaluate_acceptance_table(self.result, self.stress_results, [], 60.0, None)
        rows_slow = evaluate_acceptance_table(self.result, self.stress_results, [], 2000.0, None)
        self.assertTrue(next(r for r in rows_fast if r.id == "B9").passed)
        self.assertFalse(next(r for r in rows_slow if r.id == "B9").passed)

    def test_b10_determinism_flag_passthrough(self):
        rows_true = evaluate_acceptance_table(self.result, self.stress_results, [], 60.0, True)
        rows_false = evaluate_acceptance_table(self.result, self.stress_results, [], 60.0, False)
        rows_none = evaluate_acceptance_table(self.result, self.stress_results, [], 60.0, None)
        self.assertTrue(next(r for r in rows_true if r.id == "B10").passed)
        self.assertFalse(next(r for r in rows_false if r.id == "B10").passed)
        self.assertIsNone(next(r for r in rows_none if r.id == "B10").passed)

    def test_b8_edge_survives_from_slippage_results(self):
        slippage_results = [
            SlippageResult(3.0, 0.10, 1.0, 0.05, 0.5, 0.05),
            SlippageResult(10.0, 0.08, 0.8, 0.05, 0.5, 0.03),
        ]
        rows = evaluate_acceptance_table(self.result, self.stress_results, slippage_results, 60.0, None)
        self.assertTrue(next(r for r in rows if r.id == "B8").passed)


class TestRenderReport(unittest.TestCase):
    def test_banner_and_table_present(self):
        rows = [
            AcceptanceRow("B1", "Median annualized turnover", "140-210%", "150%", True, "resp"),
            AcceptanceRow("B10", "Determinism", "Bit-identical", "verified", True, "resp"),
        ]
        panel, sector_by_ticker = make_universe_panel(5, "2020-01-01", "2020-06-30")
        result = run_backtest(panel, sector_by_ticker, date(2020, 1, 1), date(2020, 6, 30))
        md = render_phase0_report_md(rows, result, date(2020, 1, 1), date(2020, 6, 30))
        self.assertIn(UNIVERSE_BANNER, md)
        self.assertIn("## Acceptance Table", md)
        self.assertIn("B1", md)
        self.assertIn("PASS", md)
        # Banner + acceptance table must lead the document, per spec §14.10 ("does not lead with performance").
        self.assertLess(md.index("Acceptance Table"), md.index("Mechanical Summary"))

    def test_failing_row_renders_fail_and_overall_verdict(self):
        rows = [AcceptanceRow("B1", "x", "y", "z", False, "fix it")]
        panel, sector_by_ticker = make_universe_panel(5, "2020-01-01", "2020-06-30")
        result = run_backtest(panel, sector_by_ticker, date(2020, 1, 1), date(2020, 6, 30))
        md = render_phase0_report_md(rows, result, date(2020, 1, 1), date(2020, 6, 30))
        self.assertIn("**FAIL**", md)
        self.assertIn("DOES NOT PASS", md)


class TestVerifyDeterminism(unittest.TestCase):
    def test_identical_runs_verified_true(self):
        panel, sector_by_ticker = make_universe_panel(10, "2020-01-01", "2020-12-31")
        ok = verify_determinism(panel, sector_by_ticker, date(2020, 1, 1), date(2020, 12, 31))
        self.assertTrue(ok)

    def test_detects_nav_mismatch(self):
        panel, sector_by_ticker = make_universe_panel(10, "2020-01-01", "2020-12-31")
        real_run = run_backtest
        call_count = {"n": 0}

        def _flaky_run_backtest(*args, **kwargs):
            result = real_run(*args, **kwargs)
            call_count["n"] += 1
            if call_count["n"] == 2:
                result.nav_daily = result.nav_daily.copy()
                # Corrupt the second run's first nav_a value.
                first_index = result.nav_daily.index[0]
                result.nav_daily.loc[first_index, "nav_a"] = float(result.nav_daily["nav_a"].iloc[0]) + 5000.0
            return result

        with mock.patch("backtest.p21_momentum.phase0_report.run_backtest", side_effect=_flaky_run_backtest):
            ok = verify_determinism(panel, sector_by_ticker, date(2020, 1, 1), date(2020, 12, 31))
        self.assertFalse(ok)


class TestWriteBaseCaseDeliverables(unittest.TestCase):
    def test_writes_all_four_files(self):
        panel, sector_by_ticker = make_universe_panel(10, "2020-01-01", "2020-12-31")
        result = run_backtest(panel, sector_by_ticker, date(2020, 1, 1), date(2020, 12, 31))
        with tempfile.TemporaryDirectory() as d:
            base_case_dir = Path(d)
            with mock.patch("backtest.p21_momentum.phase0_report.BASE_CASE_DIR", base_case_dir):
                _write_base_case_deliverables(result)
            self.assertTrue((base_case_dir / "nav_daily.csv").exists())
            self.assertTrue((base_case_dir / "trades.jsonl").exists())
            self.assertTrue((base_case_dir / "monthly_metrics.csv").exists())
            self.assertTrue((base_case_dir / "stress_windows.md").exists())
            nav_read_back = pd.read_csv(base_case_dir / "nav_daily.csv")
            self.assertIn("nav_a", nav_read_back.columns)


class TestStressWindowCountMatchesSpec(unittest.TestCase):
    def test_nine_windows(self):
        self.assertEqual(len(STRESS_WINDOWS), 9)


if __name__ == "__main__":
    unittest.main()
