"""Unit tests for src.ml.pipeline.p21_momentum.execution.ledger."""

from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

from src.ml.pipeline.p21_momentum.execution.ledger import (
    append_ledger_entries,
    read_all_ledger_entries,
    read_current_positions,
    read_ledger_entries_for_month,
    write_current_positions,
)
from src.ml.pipeline.p21_momentum.schemas import LedgerEntry, Position


def _entry(ts: str, ticker: str = "AAPL") -> LedgerEntry:
    return LedgerEntry(
        ts=ts,
        track="A",
        ticker=ticker,
        side="BUY",
        shares=10.0,
        ref_open=100.0,
        fill_price=100.03,
        slippage_bps=3.0,
        commission_usd=0.35,
        gross_usd=1000.3,
        net_usd=1000.65,
        reason="ENTRY_RANK_3",
    )


class TestAppendAndReadLedger(unittest.TestCase):
    def test_append_then_read_all_roundtrips(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.jsonl"
            entries = [_entry("2026-09-01T09:45:00-04:00"), _entry("2026-09-01T09:45:01-04:00", ticker="MSFT")]
            append_ledger_entries(entries, path=path)
            result = read_all_ledger_entries(path=path)
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0].ticker, "AAPL")
            self.assertEqual(result[1].ticker, "MSFT")

    def test_append_is_additive_across_calls(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.jsonl"
            append_ledger_entries([_entry("2026-09-01T09:45:00-04:00")], path=path)
            append_ledger_entries([_entry("2026-10-01T09:45:00-04:00")], path=path)
            result = read_all_ledger_entries(path=path)
            self.assertEqual(len(result), 2)

    def test_read_nonexistent_file_returns_empty(self):
        result = read_all_ledger_entries(path=Path("does/not/exist.jsonl"))
        self.assertEqual(result, [])

    def test_read_for_month_filters_correctly(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.jsonl"
            append_ledger_entries(
                [
                    _entry("2026-09-01T09:45:00-04:00", ticker="SEP"),
                    _entry("2026-10-01T09:45:00-04:00", ticker="OCT"),
                ],
                path=path,
            )
            sept = read_ledger_entries_for_month(2026, 9, path=path)
            self.assertEqual([e.ticker for e in sept], ["SEP"])

    def test_empty_entries_noop(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.jsonl"
            append_ledger_entries([], path=path)
            self.assertFalse(path.exists())


class TestCurrentPositions(unittest.TestCase):
    def test_write_then_read_roundtrips(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "current_positions.json"
            positions = [
                Position(
                    ticker="NVDA",
                    shares=14.2371,
                    avg_cost=175.40,
                    entry_date="2026-06-01",
                    entry_rank=3,
                    current_rank=7,
                    sector="Information Technology",
                    target_weight_pct=0.95,
                    high_water_price=198.20,
                )
            ]
            write_current_positions(
                positions,
                as_of=date(2026, 9, 1),
                track="A",
                nav_total=250_000.0,
                cash=201430.50,
                sleeve_market_value=48569.50,
                regime_scalar=1.0,
                path=path,
            )
            result = read_current_positions(path=path)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].ticker, "NVDA")

    def test_read_nonexistent_returns_empty(self):
        result = read_current_positions(path=Path("does/not/exist.json"))
        self.assertEqual(result, [])

    def test_write_overwrites_previous_content(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "current_positions.json"
            pos1 = [
                Position("A", 1.0, 10.0, "2026-01-01", 1, 1, "Tech", 1.0, 10.0),
            ]
            pos2 = [
                Position("B", 2.0, 20.0, "2026-02-01", 2, 2, "Tech", 1.0, 20.0),
            ]
            write_current_positions(pos1, date(2026, 1, 1), "A", 250000, 0, 0, 1.0, path=path)
            write_current_positions(pos2, date(2026, 2, 1), "A", 250000, 0, 0, 1.0, path=path)
            result = read_current_positions(path=path)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].ticker, "B")


if __name__ == "__main__":
    unittest.main()
