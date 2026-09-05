"""Integration test for src.ml.pipeline.p21_momentum.jobs.run_stop_execute."""

from __future__ import annotations

import unittest
from datetime import date
from unittest.mock import patch

import pandas as pd

from src.ml.pipeline.p21_momentum.jobs import run_stop_execute as job
from src.ml.pipeline.p21_momentum.schemas import PendingStop, Position


def _open_df(today: date, open_price: float) -> pd.DataFrame:
    idx = pd.bdate_range(end=pd.Timestamp(today), periods=5)
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": [open_price] * 5,
            "high": [open_price * 1.01] * 5,
            "low": [open_price * 0.99] * 5,
            "close": [open_price] * 5,
            "volume": [1_000_000] * 5,
        }
    )


class TestRunStopExecuteGuards(unittest.TestCase):
    def test_skips_on_weekend(self):
        result = job.run(run_date=date(2026, 8, 22))  # Saturday
        self.assertTrue(result.get("skipped"))
        self.assertEqual(result.get("reason"), "not_trading_day")

    @patch("src.ml.pipeline.p21_momentum.jobs.run_stop_execute.already_processed", return_value=True)
    def test_skips_when_already_processed(self, _mock):
        del _mock
        result = job.run(run_date=date(2026, 8, 24))  # Monday
        self.assertTrue(result.get("skipped"))
        self.assertEqual(result.get("reason"), "already_processed")


class TestRunStopExecuteEmptyQueue(unittest.TestCase):
    @patch("src.ml.pipeline.p21_momentum.jobs.run_stop_execute.write_stop_exits")
    @patch("src.ml.pipeline.p21_momentum.jobs.run_stop_execute.read_pending_stops", return_value=[])
    @patch("src.ml.pipeline.p21_momentum.jobs.run_stop_execute.already_processed", return_value=False)
    def test_noop_when_queue_empty(self, _mock_processed, _mock_read_pending, mock_write_exits):
        del _mock_processed, _mock_read_pending
        result = job.run(run_date=date(2026, 8, 24))
        self.assertFalse(result.get("skipped"))
        self.assertEqual(result["exits_count"], 0)
        mock_write_exits.assert_called_once()
        (_, exits), _ = mock_write_exits.call_args
        self.assertEqual(exits, [])


class TestRunStopExecuteHappyPath(unittest.TestCase):
    @patch("src.ml.pipeline.p21_momentum.jobs.run_stop_execute._read_nav_total", return_value=250_000.0)
    @patch("src.ml.pipeline.p21_momentum.jobs.run_stop_execute._read_prior_state", return_value=(200_000.0, 1.0))
    @patch("src.ml.pipeline.p21_momentum.jobs.run_stop_execute.write_stop_exits")
    @patch("src.ml.pipeline.p21_momentum.jobs.run_stop_execute.write_pending_stops")
    @patch("src.ml.pipeline.p21_momentum.jobs.run_stop_execute.write_current_positions")
    @patch("src.ml.pipeline.p21_momentum.jobs.run_stop_execute.append_ledger_entries")
    @patch("src.ml.pipeline.p21_momentum.jobs.run_stop_execute.fetch_price_panel")
    @patch("src.ml.pipeline.p21_momentum.jobs.run_stop_execute.read_current_positions")
    @patch("src.ml.pipeline.p21_momentum.jobs.run_stop_execute.read_pending_stops")
    @patch("src.ml.pipeline.p21_momentum.jobs.run_stop_execute.already_processed", return_value=False)
    def test_executes_queued_stop(
        self,
        _mock_processed,
        mock_read_pending,
        mock_read_positions,
        mock_fetch_panel,
        mock_append_ledger,
        mock_write_current,
        mock_write_pending,
        mock_write_exits,
        _mock_prior_state,
        _mock_nav_total,
    ):
        del _mock_processed, _mock_prior_state, _mock_nav_total
        today = date(2026, 8, 25)
        mock_read_pending.return_value = [
            PendingStop(ticker="AAPL", flagged_date="2026-08-24", price_at_flag=60.0, avg_cost=100.0)
        ]
        mock_read_positions.return_value = [
            Position("AAPL", 10.0, 100.0, "2026-06-01", 1, 1, "Tech", 0.01, 100.0),
            Position("MSFT", 5.0, 300.0, "2026-06-01", 2, 2, "Tech", 0.01, 300.0),
        ]
        mock_fetch_panel.return_value = {"AAPL": _open_df(today, 58.0)}

        result = job.run(run_date=today)

        self.assertFalse(result.get("skipped"))
        self.assertFalse(result.get("aborted"))
        self.assertEqual(result["exits_count"], 1)
        self.assertEqual(result["still_queued"], 0)

        # Full position sold, slippage against the trader (sell fills below open).
        mock_append_ledger.assert_called_once()
        (entries,), _ = mock_append_ledger.call_args
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry.ticker, "AAPL")
        self.assertEqual(entry.side, "SELL")
        self.assertEqual(entry.shares, 10.0)
        self.assertEqual(entry.reason, "EXIT_CATASTROPHIC_STOP")
        self.assertLess(entry.fill_price, 58.0)  # slippage works against a SELL

        # MSFT (not stopped) survives; AAPL is gone.
        mock_write_current.assert_called_once()
        written_positions = mock_write_current.call_args[0][0]
        self.assertEqual([p.ticker for p in written_positions], ["MSFT"])

        # Queue cleared once executed.
        mock_write_pending.assert_called_once()
        (still_queued,), _ = mock_write_pending.call_args
        self.assertEqual(still_queued, [])

        # Audit trail written too.
        mock_write_exits.assert_called_once()

    @patch("src.ml.pipeline.p21_momentum.jobs.run_stop_execute.write_stop_exits")
    @patch("src.ml.pipeline.p21_momentum.jobs.run_stop_execute.write_pending_stops")
    @patch("src.ml.pipeline.p21_momentum.jobs.run_stop_execute.fetch_price_panel", return_value={})
    @patch("src.ml.pipeline.p21_momentum.jobs.run_stop_execute.read_current_positions")
    @patch("src.ml.pipeline.p21_momentum.jobs.run_stop_execute.read_pending_stops")
    @patch("src.ml.pipeline.p21_momentum.jobs.run_stop_execute.already_processed", return_value=False)
    def test_leaves_queued_when_no_open_price(
        self, _mock_processed, mock_read_pending, mock_read_positions, _mock_fetch_panel, mock_write_pending, _mock_write_exits
    ):
        del _mock_processed, _mock_fetch_panel, _mock_write_exits
        mock_read_pending.return_value = [
            PendingStop(ticker="AAPL", flagged_date="2026-08-24", price_at_flag=60.0, avg_cost=100.0)
        ]
        mock_read_positions.return_value = [
            Position("AAPL", 10.0, 100.0, "2026-06-01", 1, 1, "Tech", 0.01, 100.0),
        ]

        result = job.run(run_date=date(2026, 8, 25))

        self.assertEqual(result["exits_count"], 0)
        self.assertEqual(result["still_queued"], 1)
        # Re-queued unchanged, to retry next trading day.
        mock_write_pending.assert_called_once()
        (requeued,), _ = mock_write_pending.call_args
        self.assertEqual([s.ticker for s in requeued], ["AAPL"])

    @patch("src.ml.pipeline.p21_momentum.jobs.run_stop_execute.write_stop_exits")
    @patch("src.ml.pipeline.p21_momentum.jobs.run_stop_execute.write_pending_stops")
    @patch("src.ml.pipeline.p21_momentum.jobs.run_stop_execute.read_current_positions", return_value=[])
    @patch("src.ml.pipeline.p21_momentum.jobs.run_stop_execute.read_pending_stops")
    @patch("src.ml.pipeline.p21_momentum.jobs.run_stop_execute.already_processed", return_value=False)
    def test_drops_entry_when_no_longer_held(
        self, _mock_processed, mock_read_pending, _mock_read_positions, mock_write_pending, _mock_write_exits
    ):
        del _mock_processed, _mock_read_positions, _mock_write_exits
        # AAPL was already exited some other way (e.g. a forced rebalance exit) before
        # this job got a chance to run -- the queue entry is stale.
        mock_read_pending.return_value = [
            PendingStop(ticker="AAPL", flagged_date="2026-08-24", price_at_flag=60.0, avg_cost=100.0)
        ]

        result = job.run(run_date=date(2026, 8, 25))

        self.assertEqual(result["exits_count"], 0)
        self.assertEqual(result.get("dropped_not_held"), 1)
        mock_write_pending.assert_called_once()
        (cleared,), _ = mock_write_pending.call_args
        self.assertEqual(cleared, [])


if __name__ == "__main__":
    unittest.main()
