"""Unit tests for backtest.p21_momentum.robustness."""

from __future__ import annotations

import unittest
from datetime import date

import backtest.p21_momentum.robustness as robustness
from backtest.p21_momentum.robustness import (
    OutOfSampleReaccessError,
    deflated_sharpe_band,
    is_top_quartile_separated,
    log_oos_access,
    run_grid,
)
from backtest.p21_momentum.tests.fixtures import make_universe_panel

# A tiny 1-combination grid so tests stay fast — the real 729-grid is exercised only by
# operators running phase0_report.py against the frozen 2005-2026 panel, not by CI.
_TINY_GRID = {
    "lookback_start": (252,),
    "skip_recent": (21,),
    "entry_rank": (20,),
    "hold_rank": (60,),
    "max_per_sector": (4,),
    "vix_threshold": (28.0,),
}


class TestDeflatedSharpeBand(unittest.TestCase):
    def test_degenerate_inputs_return_zero_band(self):
        self.assertEqual(deflated_sharpe_band(1, 100), (0.0, 0.0))
        self.assertEqual(deflated_sharpe_band(100, 1), (0.0, 0.0))

    def test_band_is_ordered_and_positive(self):
        low, high = deflated_sharpe_band(729, 245)
        self.assertGreater(low, 0.0)
        self.assertLessEqual(low, high)

    def test_more_trials_raises_expected_max(self):
        low_10, _ = deflated_sharpe_band(10, 245)
        low_729, _ = deflated_sharpe_band(729, 245)
        self.assertGreater(low_729, low_10)


class TestTopQuartileSeparation(unittest.TestCase):
    def test_too_few_points_not_separated(self):
        self.assertFalse(is_top_quartile_separated([1.0, 2.0]))

    def test_flat_surface_not_separated(self):
        self.assertFalse(is_top_quartile_separated([1.0] * 20))

    def test_clear_outlier_is_separated(self):
        sharpes = [0.5] * 19 + [5.0]
        self.assertTrue(is_top_quartile_separated(sharpes))


class TestOosAccessLog(unittest.TestCase):
    def test_log_creates_file_with_header_and_appends(self, tmp_path=None):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            log_path = Path(d) / "oos_access_log.md"
            log_oos_access("2005-01-01..2016-12-31", "test entry one", log_path=log_path)
            log_oos_access("2017-01-01..2026-06-30", "test entry two", log_path=log_path)
            content = log_path.read_text(encoding="utf-8")
            self.assertIn("Out-of-Sample Access Log", content)
            self.assertIn("test entry one", content)
            self.assertIn("test entry two", content)


class TestRunGrid(unittest.TestCase):
    def setUp(self):
        # Rule 4's guard is process-lifetime state; reset it so each test starts clean.
        robustness._OOS_TOUCHED["value"] = False
        self.panel, self.sector_by_ticker = make_universe_panel(10, "2020-01-01", "2021-06-30")

    def test_in_sample_grid_runs_once_without_flag(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            log_path = Path(d) / "oos_access_log.md"
            rows = run_grid(
                self.panel, self.sector_by_ticker, date(2020, 1, 1), date(2021, 6, 30),
                grid=_TINY_GRID, oos_log_path=log_path,
            )
            self.assertEqual(len(rows), 1)
            self.assertTrue(log_path.exists())

    def test_out_of_sample_touch_twice_without_ack_raises(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            log_path = Path(d) / "oos_access_log.md"
            run_grid(
                self.panel, self.sector_by_ticker, date(2020, 1, 1), date(2026, 6, 30),
                grid=_TINY_GRID, oos_log_path=log_path,
            )
            with self.assertRaises(OutOfSampleReaccessError):
                run_grid(
                    self.panel, self.sector_by_ticker, date(2020, 1, 1), date(2026, 6, 30),
                    grid=_TINY_GRID, oos_log_path=log_path,
                )

    def test_out_of_sample_touch_twice_with_ack_succeeds(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            log_path = Path(d) / "oos_access_log.md"
            run_grid(
                self.panel, self.sector_by_ticker, date(2020, 1, 1), date(2026, 6, 30),
                grid=_TINY_GRID, oos_log_path=log_path,
            )
            rows = run_grid(
                self.panel, self.sector_by_ticker, date(2020, 1, 1), date(2026, 6, 30),
                grid=_TINY_GRID, oos_log_path=log_path, acknowledge_oos_reaccess=True,
            )
            self.assertEqual(len(rows), 1)

    def test_grid_row_has_expected_fields(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            log_path = Path(d) / "oos_access_log.md"
            rows = run_grid(
                self.panel, self.sector_by_ticker, date(2020, 1, 1), date(2021, 6, 30),
                grid=_TINY_GRID, oos_log_path=log_path,
            )
            row = rows[0]
            self.assertEqual(row.lookback_start, 252)
            self.assertIn("sharpe_a", row.to_dict())


if __name__ == "__main__":
    unittest.main()
