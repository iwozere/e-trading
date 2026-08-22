"""Unit tests for src.ml.pipeline.p21_momentum.results.run_io."""

from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

from src.ml.pipeline.p21_momentum.results.run_io import (
    already_processed,
    append_nav_row,
    append_regime_history,
    read_regime_history,
    read_targets,
    run_dir_for,
    write_daily_mark,
    write_positions,
    write_report,
    write_signals,
    write_targets,
    write_universe,
)
from src.ml.pipeline.p21_momentum.schemas import DailyMarkSnapshot, Position, SignalRow, TargetPosition


class TestRunDirFor(unittest.TestCase):
    def test_creates_dated_folder(self):
        with TemporaryDirectory() as tmp:
            d = run_dir_for(date(2026, 9, 1), results_dir=Path(tmp))
            self.assertTrue(d.exists())
            self.assertEqual(d.name, "2026-09-01")


class TestAlreadyProcessed(unittest.TestCase):
    def test_false_when_missing(self):
        with TemporaryDirectory() as tmp:
            self.assertFalse(already_processed(date(2026, 9, 1), "targets.json", results_dir=Path(tmp)))

    def test_true_after_write(self):
        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            write_targets(date(2026, 9, 1), [], results_dir=results_dir)
            self.assertTrue(already_processed(date(2026, 9, 1), "targets.json", results_dir=results_dir))


class TestTypedWriteRead(unittest.TestCase):
    def test_universe_roundtrip(self):
        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            write_universe(
                date(2026, 9, 1), {"as_of": "2026-09-01", "count": 1, "constituents": []}, results_dir=results_dir
            )
            path = results_dir / "2026-09-01" / "universe.json"
            self.assertTrue(path.exists())

    def test_signals_roundtrip(self):
        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            rows = [SignalRow(ticker="AAPL", raw_return=0.1, vol=0.2, signal=0.5, rank=1)]
            write_signals(date(2026, 9, 1), rows, results_dir=results_dir)
            path = results_dir / "2026-09-01" / "signals.json"
            self.assertTrue(path.exists())

    def test_targets_write_then_read(self):
        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            targets = [TargetPosition(ticker="AAPL", target_weight_pct=0.05, target_usd=2500.0, rank=1)]
            write_targets(date(2026, 9, 1), targets, results_dir=results_dir)
            result = read_targets(date(2026, 9, 1), results_dir=results_dir)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].ticker, "AAPL")

    def test_read_targets_missing_returns_empty(self):
        with TemporaryDirectory() as tmp:
            result = read_targets(date(2026, 9, 1), results_dir=Path(tmp))
            self.assertEqual(result, [])

    def test_positions_write(self):
        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            positions = [Position("AAPL", 10.0, 100.0, "2026-09-01", 1, 1, "Tech", 0.05, 105.0)]
            write_positions(date(2026, 9, 1), positions, results_dir=results_dir)
            path = results_dir / "2026-09-01" / "positions.json"
            self.assertTrue(path.exists())

    def test_daily_mark_write(self):
        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            snapshot = DailyMarkSnapshot(as_of="2026-09-01", nav={"A": 250_000.0})
            write_daily_mark(date(2026, 9, 1), snapshot, results_dir=results_dir)
            path = results_dir / "2026-09-01" / "daily_mark.json"
            self.assertTrue(path.exists())

    def test_report_write(self):
        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            path = write_report(date(2026, 9, 1), "# Report\n", results_dir=results_dir)
            self.assertEqual(path.read_text(encoding="utf-8"), "# Report\n")


class TestRegimeHistory(unittest.TestCase):
    def test_append_and_read(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "regime_history.json"
            append_regime_history({"date": "2026-08-31", "scalar_applied": 1.0}, path=path)
            append_regime_history({"date": "2026-09-30", "scalar_applied": 0.6}, path=path)
            history = read_regime_history(path)
            self.assertEqual(len(history), 2)
            self.assertEqual(history[1]["scalar_applied"], 0.6)

    def test_read_missing_returns_empty(self):
        result = read_regime_history(Path("does/not/exist.json"))
        self.assertEqual(result, [])


class TestNavDailyCsv(unittest.TestCase):
    def test_append_writes_header_once(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "nav_daily.csv"
            row1 = {"date": "2026-09-01", "nav_a": 1, "nav_b": 2, "nav_c": 3, "nav_d": 4, "nav_e": 5}
            row2 = {"date": "2026-09-02", "nav_a": 1, "nav_b": 2, "nav_c": 3, "nav_d": 4, "nav_e": 5}
            append_nav_row(row1, path=path)
            append_nav_row(row2, path=path)
            content = path.read_text(encoding="utf-8")
            self.assertEqual(content.count("date,nav_a"), 1)  # header appears exactly once
            self.assertEqual(len(content.strip().splitlines()), 3)  # header + 2 rows


if __name__ == "__main__":
    unittest.main()
