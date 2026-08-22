"""Integration test for src.ml.pipeline.p21_momentum.jobs.run_daily_mark."""

from __future__ import annotations

import unittest
from datetime import date
from unittest.mock import patch

import pandas as pd

from src.ml.pipeline.p21_momentum.jobs import run_daily_mark as job
from src.ml.pipeline.p21_momentum.schemas import Position


def _close_series_df(closes: list) -> pd.DataFrame:
    idx = pd.bdate_range(end=pd.Timestamp("2026-09-02"), periods=len(closes))
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": [1_000_000] * len(closes),
        }
    )


class TestRunDailyMarkGuards(unittest.TestCase):
    def test_skips_on_weekend(self):
        result = job.run(run_date=date(2026, 8, 22))  # Saturday
        self.assertTrue(result.get("skipped"))
        self.assertEqual(result.get("reason"), "not_trading_day")

    @patch("src.ml.pipeline.p21_momentum.jobs.run_daily_mark.already_processed", return_value=True)
    def test_skips_when_already_processed(self, _mock):
        del _mock
        result = job.run(run_date=date(2026, 8, 24))  # Monday
        self.assertTrue(result.get("skipped"))
        self.assertEqual(result.get("reason"), "already_processed")


class TestRunDailyMarkHappyPath(unittest.TestCase):
    @patch("src.ml.pipeline.p21_momentum.jobs.run_daily_mark._read_prior_cash", return_value=200_000.0)
    @patch("src.ml.pipeline.p21_momentum.jobs.run_daily_mark.write_daily_mark")
    @patch("src.ml.pipeline.p21_momentum.jobs.run_daily_mark.append_nav_row")
    @patch("src.ml.pipeline.p21_momentum.jobs.run_daily_mark.write_current_positions")
    @patch("src.ml.pipeline.p21_momentum.jobs.run_daily_mark.fetch_price_panel")
    @patch("src.ml.pipeline.p21_momentum.jobs.run_daily_mark.read_current_positions")
    @patch("src.ml.pipeline.p21_momentum.jobs.run_daily_mark.already_processed", return_value=False)
    def test_catastrophic_stop_flagged(
        self,
        _mock_processed,
        mock_read_positions,
        mock_fetch_panel,
        mock_write_current,
        mock_append_nav,
        mock_write_daily_mark,
        _mock_prior_cash,
    ):
        del _mock_processed, _mock_prior_cash
        today = date(2026, 8, 24)
        # avg_cost 100, price now 60 -> -40% < -35% catastrophic stop threshold
        mock_read_positions.return_value = [
            Position("AAPL", 10.0, 100.0, "2026-06-01", 1, 1, "Tech", 0.01, 100.0),
        ]
        mock_fetch_panel.return_value = {
            "AAPL": _close_series_df([100.0, 60.0]),
            "MTUM": _close_series_df([200.0, 201.0]),
            "SPY": _close_series_df([550.0, 551.0]),
            "^GSPC": _close_series_df([5000.0, 5010.0]),
            "^VIX": _close_series_df([15.0, 15.5]),
        }

        result = job.run(run_date=today)

        self.assertFalse(result.get("skipped"))
        self.assertFalse(result.get("aborted"))
        self.assertEqual(result["catastrophic_stops_count"], 1)
        mock_write_current.assert_called_once()
        mock_append_nav.assert_called_once()
        mock_write_daily_mark.assert_called_once()

    @patch("src.ml.pipeline.p21_momentum.jobs.run_daily_mark._read_prior_cash", return_value=200_000.0)
    @patch("src.ml.pipeline.p21_momentum.jobs.run_daily_mark.write_daily_mark")
    @patch("src.ml.pipeline.p21_momentum.jobs.run_daily_mark.append_nav_row")
    @patch("src.ml.pipeline.p21_momentum.jobs.run_daily_mark.write_current_positions")
    @patch("src.ml.pipeline.p21_momentum.jobs.run_daily_mark.fetch_price_panel")
    @patch("src.ml.pipeline.p21_momentum.jobs.run_daily_mark.read_current_positions", return_value=[])
    @patch("src.ml.pipeline.p21_momentum.jobs.run_daily_mark.already_processed", return_value=False)
    def test_anomaly_flagged_without_position(
        self,
        _mock_processed,
        _mock_read_positions,
        mock_fetch_panel,
        mock_write_current,
        mock_append_nav,
        mock_write_daily_mark,
        _mock_prior_cash,
    ):
        del _mock_processed, _mock_read_positions, _mock_prior_cash
        del mock_write_current, mock_append_nav, mock_write_daily_mark
        today = date(2026, 8, 24)
        mock_fetch_panel.return_value = {
            "MTUM": _close_series_df([200.0, 400.0]),  # +100% -> anomaly
            "SPY": _close_series_df([550.0, 551.0]),
            "^GSPC": _close_series_df([5000.0, 5010.0]),
            "^VIX": _close_series_df([15.0, 15.5]),
        }

        result = job.run(run_date=today)
        self.assertEqual(result["anomalies_count"], 1)


if __name__ == "__main__":
    unittest.main()
