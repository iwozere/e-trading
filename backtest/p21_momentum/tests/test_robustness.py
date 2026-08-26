"""Unit tests for backtest.p21_momentum.robustness."""

from __future__ import annotations

import itertools
import tempfile
import unittest
from datetime import date
from pathlib import Path

import backtest.p21_momentum.robustness as robustness
from backtest.p21_momentum.robustness import (
    GridRow,
    OutOfSampleReaccessError,
    deflated_sharpe_band,
    is_top_quartile_separated,
    log_oos_access,
    render_deflated_sharpe_md,
    render_marginal_surfaces_png,
    run_grid,
    run_grid_parallel,
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


def _rows_with_sharpes(sharpes) -> list:
    """Build minimal GridRow objects carrying only the sharpe_a values under test."""
    return [GridRow(252, 21, 20, 60, 4, 28.0, s, 0.5, 150.0) for s in sharpes]


class TestRenderDeflatedSharpeMd(unittest.TestCase):
    """
    Regression tests for the wording bug fixed 2026-08-25: the report used to claim the
    best configuration was "above the by-chance band" whenever it was merely separated
    from the top-quartile median, even when `best` actually fell *inside* [low, high].
    Four cases must each get their own, factually accurate sentence.
    """

    N_TRIALS = 50
    N_OBSERVATIONS = 200

    def _band(self):
        return deflated_sharpe_band(self.N_TRIALS, self.N_OBSERVATIONS)

    def test_below_band_reports_no_skill(self):
        low, _ = self._band()
        md = render_deflated_sharpe_md(_rows_with_sharpes([low - 10.0] * self.N_TRIALS), self.N_OBSERVATIONS)
        self.assertIn("falls below the by-chance band", md)

    def test_flat_surface_not_separated(self):
        low, high = self._band()
        constant = low + 0.1 * (high - low)
        md = render_deflated_sharpe_md(_rows_with_sharpes([constant] * self.N_TRIALS), self.N_OBSERVATIONS)
        self.assertIn("not clearly separated from the top quartile", md)

    def test_separated_but_within_band_is_not_reported_as_above_it(self):
        low, high = self._band()
        width = high - low
        v = low + 0.1 * width
        best = low + 0.6 * width  # separated from v (see docstring math), still <= high
        sharpes = [v] * (self.N_TRIALS - 1) + [best]
        md = render_deflated_sharpe_md(_rows_with_sharpes(sharpes), self.N_OBSERVATIONS)
        self.assertIn("falls **within** the expected-by-chance band, not above it", md)
        self.assertNotIn("above the by-chance band", md)

    def test_separated_and_above_band_reports_above(self):
        low, high = self._band()
        v = low + 0.1 * (high - low)
        best = high + 1.0
        sharpes = [v] * (self.N_TRIALS - 1) + [best]
        md = render_deflated_sharpe_md(_rows_with_sharpes(sharpes), self.N_OBSERVATIONS)
        self.assertIn("separated from the top-quartile median and above the by-chance band", md)


class TestRenderMarginalSurfacesPng(unittest.TestCase):
    """spec §14.10 deliverable — one subplot per ROBUSTNESS_GRID parameter."""

    def test_writes_a_nonempty_png_covering_every_grid_parameter(self):
        keys = list(robustness.ROBUSTNESS_GRID.keys())
        combos = list(itertools.product(*(robustness.ROBUSTNESS_GRID[k] for k in keys)))
        rows = [
            GridRow(
                lookback_start=combo[0],
                skip_recent=combo[1],
                entry_rank=combo[2],
                hold_rank=combo[3],
                max_per_sector=combo[4],
                vix_threshold=combo[5],
                sharpe_a=float(i % 5),
                sharpe_c=0.5,
                turnover_annualized_median_pct=150.0,
            )
            for i, combo in enumerate(combos)
        ]

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "marginal_surfaces.png"
            render_marginal_surfaces_png(rows, out_path)
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)

    def test_handles_a_single_combination(self):
        """The grid can be overridden down to one combo (as tests do) — must not crash on that."""
        rows = [GridRow(252, 21, 20, 60, 4, 28.0, 0.7, 0.5, 150.0)]
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "marginal_surfaces.png"
            render_marginal_surfaces_png(rows, out_path)
            self.assertTrue(out_path.exists())


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


class TestRunGridParallel(unittest.TestCase):
    def setUp(self):
        robustness._OOS_TOUCHED["value"] = False
        # Strictly in-sample (pre-2017) so a test can call both run_grid() and
        # run_grid_parallel() without tripping Rule 4's single-touch guard against itself.
        self.panel, self.sector_by_ticker = make_universe_panel(10, "2010-01-01", "2011-06-30")

    def test_matches_sequential_grid_row_content(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            seq_rows = run_grid(
                self.panel, self.sector_by_ticker, date(2010, 1, 1), date(2011, 6, 30),
                grid=_TINY_GRID, oos_log_path=Path(d) / "seq_log.md",
            )
            par_rows = run_grid_parallel(
                self.panel, self.sector_by_ticker, date(2010, 1, 1), date(2011, 6, 30),
                grid=_TINY_GRID, max_workers=2, oos_log_path=Path(d) / "par_log.md",
            )
            self.assertEqual(len(par_rows), len(seq_rows))
            self.assertEqual(par_rows[0].to_dict(), seq_rows[0].to_dict())

    def test_multi_combo_grid_runs_all_combinations(self):
        import tempfile
        from pathlib import Path

        grid = {**_TINY_GRID, "hold_rank": (40, 60, 100)}
        with tempfile.TemporaryDirectory() as d:
            rows = run_grid_parallel(
                self.panel, self.sector_by_ticker, date(2010, 1, 1), date(2011, 6, 30),
                grid=grid, max_workers=2, oos_log_path=Path(d) / "log.md",
            )
            self.assertEqual(len(rows), 3)
            self.assertEqual(sorted(r.hold_rank for r in rows), [40, 60, 100])

    def test_out_of_sample_touch_twice_without_ack_raises(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            log_path = Path(d) / "oos_access_log.md"
            run_grid_parallel(
                self.panel, self.sector_by_ticker, date(2020, 1, 1), date(2026, 6, 30),
                grid=_TINY_GRID, max_workers=2, oos_log_path=log_path,
            )
            with self.assertRaises(OutOfSampleReaccessError):
                run_grid_parallel(
                    self.panel, self.sector_by_ticker, date(2020, 1, 1), date(2026, 6, 30),
                    grid=_TINY_GRID, max_workers=2, oos_log_path=log_path,
                )


if __name__ == "__main__":
    unittest.main()
