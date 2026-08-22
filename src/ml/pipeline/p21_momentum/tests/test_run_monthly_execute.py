"""Integration test for src.ml.pipeline.p21_momentum.jobs.run_monthly_execute."""

from __future__ import annotations

import unittest
from datetime import date
from unittest.mock import patch

import pandas as pd

from src.ml.pipeline.p21_momentum.jobs import run_monthly_execute as job
from src.ml.pipeline.p21_momentum.schemas import TargetPosition


def _open_df(execution_date: date, open_price: float) -> pd.DataFrame:
    idx = pd.bdate_range(end=pd.Timestamp(execution_date), periods=5)
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


class TestRunMonthlyExecuteGuards(unittest.TestCase):
    def test_skips_when_not_first_trading_day(self):
        result = job.run(run_date=date(2026, 8, 15))
        self.assertTrue(result.get("skipped"))
        self.assertEqual(result.get("reason"), "not_first_trading_day")

    @patch("src.ml.pipeline.p21_momentum.jobs.run_monthly_execute.already_processed", return_value=True)
    def test_skips_when_already_processed(self, _mock):
        del _mock
        result = job.run(run_date=date(2026, 9, 1))  # first trading day of Sept 2026
        self.assertTrue(result.get("skipped"))
        self.assertEqual(result.get("reason"), "already_processed")

    @patch("src.ml.pipeline.p21_momentum.jobs.run_monthly_execute.already_processed", return_value=False)
    @patch("src.ml.pipeline.p21_momentum.jobs.run_monthly_execute.read_targets", return_value=[])
    def test_skips_when_no_targets(self, _mock_targets, _mock_processed):
        del _mock_targets, _mock_processed
        result = job.run(run_date=date(2026, 9, 1))
        self.assertTrue(result.get("skipped"))
        self.assertEqual(result.get("reason"), "no_targets")


class TestRunMonthlyExecuteHappyPath(unittest.TestCase):
    @patch("src.ml.pipeline.p21_momentum.jobs.run_monthly_execute.write_report")
    @patch("src.ml.pipeline.p21_momentum.jobs.run_monthly_execute.write_positions")
    @patch("src.ml.pipeline.p21_momentum.jobs.run_monthly_execute.write_current_positions")
    @patch("src.ml.pipeline.p21_momentum.jobs.run_monthly_execute.append_ledger_entries")
    @patch("src.ml.pipeline.p21_momentum.jobs.run_monthly_execute._read_prior_cash", return_value=200_000.0)
    @patch("src.ml.pipeline.p21_momentum.jobs.run_monthly_execute.fetch_price_panel")
    @patch("src.ml.pipeline.p21_momentum.jobs.run_monthly_execute.read_current_positions", return_value=[])
    @patch("src.ml.pipeline.p21_momentum.jobs.run_monthly_execute.read_targets")
    @patch("src.ml.pipeline.p21_momentum.jobs.run_monthly_execute.already_processed", return_value=False)
    def test_first_ever_run_buys_all_targets(
        self,
        _mock_processed,
        mock_read_targets,
        _mock_read_positions,
        mock_fetch_panel,
        _mock_prior_cash,
        mock_append_ledger,
        mock_write_current,
        mock_write_positions,
        mock_write_report,
    ):
        del _mock_processed, _mock_read_positions, _mock_prior_cash
        execution_date = date(2026, 9, 1)
        targets = [
            TargetPosition(ticker="AAPL", target_weight_pct=0.01, target_usd=2500.0, rank=1, sector="Tech"),
            TargetPosition(ticker="MSFT", target_weight_pct=0.01, target_usd=2500.0, rank=2, sector="Tech"),
        ]
        mock_read_targets.return_value = targets

        panel = {
            "AAPL": _open_df(execution_date, 175.0),
            "MSFT": _open_df(execution_date, 410.0),
            "MTUM": _open_df(execution_date, 200.0),
            "SPY": _open_df(execution_date, 550.0),
        }
        mock_fetch_panel.return_value = panel

        result = job.run(run_date=execution_date)

        self.assertFalse(result.get("skipped"))
        self.assertFalse(result.get("aborted"))
        self.assertEqual(result["trades_count"], 2)
        self.assertEqual(result["positions_count"], 2)
        mock_append_ledger.assert_called_once()
        mock_write_current.assert_called_once()
        mock_write_positions.assert_called_once()
        mock_write_report.assert_called_once()

        ledger_entries = mock_append_ledger.call_args[0][0]
        self.assertTrue(all(e.side == "BUY" for e in ledger_entries))
        self.assertTrue(all(e.reason.startswith("ENTRY_RANK_") for e in ledger_entries))


if __name__ == "__main__":
    unittest.main()
