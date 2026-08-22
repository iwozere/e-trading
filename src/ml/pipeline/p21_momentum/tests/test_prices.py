"""Unit tests for src.ml.pipeline.p21_momentum.data.prices."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pandas as pd

from src.ml.pipeline.p21_momentum.data.prices import (
    _is_stale,
    fetch_fundamentals_cached,
    fetch_price_panel,
    fetch_sectors_cached,
    non_empty_symbols,
)
from src.ml.pipeline.p21_momentum.quality.gates import PipelineAbort


def _df(rows: int = 5) -> pd.DataFrame:
    return pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=rows), "close": [1.0] * rows})


class TestFetchPricePanel(unittest.TestCase):
    @patch("src.ml.pipeline.p21_momentum.data.prices._get_yahoo_downloader")
    def test_full_coverage_returns_panel(self, mock_get_dl):
        mock_dl = MagicMock()
        mock_dl.get_ohlcv_batch.return_value = {"AAPL": _df(), "MSFT": _df()}
        mock_get_dl.return_value = mock_dl

        panel = fetch_price_panel(["AAPL", "MSFT"], datetime(2026, 1, 1), datetime(2026, 2, 1))
        self.assertEqual(set(panel.keys()), {"AAPL", "MSFT"})

    @patch("src.ml.pipeline.p21_momentum.data.prices._get_yahoo_downloader")
    def test_low_coverage_raises_pipeline_abort(self, mock_get_dl):
        mock_dl = MagicMock()
        # 1 of 10 tickers returns data -> 10% coverage, below 95% threshold
        panel = {f"T{i}": pd.DataFrame() for i in range(10)}
        panel["T0"] = _df()
        mock_dl.get_ohlcv_batch.return_value = panel
        mock_get_dl.return_value = mock_dl

        with self.assertRaises(PipelineAbort):
            fetch_price_panel(list(panel.keys()), datetime(2026, 1, 1), datetime(2026, 2, 1))

    def test_non_empty_symbols(self):
        panel = {"A": _df(), "B": pd.DataFrame(), "C": _df()}
        self.assertEqual(non_empty_symbols(panel), ["A", "C"])


class TestIsStale(unittest.TestCase):
    def test_fresh_timestamp_not_stale(self):
        now = datetime.now(timezone.utc).isoformat()
        self.assertFalse(_is_stale(now, ttl_days=90))

    def test_old_timestamp_is_stale(self):
        old = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
        self.assertTrue(_is_stale(old, ttl_days=90))

    def test_malformed_timestamp_is_stale(self):
        self.assertTrue(_is_stale("not-a-date", ttl_days=90))


class TestFetchFundamentalsCached(unittest.TestCase):
    @patch("src.ml.pipeline.p21_momentum.data.prices._get_yahoo_downloader")
    def test_cache_miss_fetches_and_writes(self, mock_get_dl):
        mock_dl = MagicMock()
        fake_fund = MagicMock(free_cash_flow=100.0, net_income=50.0)
        mock_dl.get_fundamentals_batch.return_value = {"AAPL": fake_fund}
        mock_get_dl.return_value = mock_dl

        with patch("src.ml.pipeline.p21_momentum.data.prices._read_cache", return_value={}), patch(
            "src.ml.pipeline.p21_momentum.data.prices._write_cache"
        ) as mock_write:
            result = fetch_fundamentals_cached(["AAPL"])

        self.assertEqual(result["AAPL"]["fcf_ttm"], 100.0)
        self.assertEqual(result["AAPL"]["net_income_ttm"], 50.0)
        mock_write.assert_called_once()

    @patch("src.ml.pipeline.p21_momentum.data.prices._get_yahoo_downloader")
    def test_cache_hit_skips_fetch(self, mock_get_dl):
        fresh = datetime.now(timezone.utc).isoformat()
        cached = {"AAPL": {"fcf_ttm": 1.0, "net_income_ttm": 2.0, "fetched_at": fresh}}
        with patch("src.ml.pipeline.p21_momentum.data.prices._read_cache", return_value=cached):
            result = fetch_fundamentals_cached(["AAPL"])

        mock_get_dl.assert_not_called()
        self.assertEqual(result["AAPL"]["fcf_ttm"], 1.0)


class TestFetchSectorsCached(unittest.TestCase):
    def test_universe_sectors_refreshes_cache(self):
        with patch("src.ml.pipeline.p21_momentum.data.prices._read_cache", return_value={}), patch(
            "src.ml.pipeline.p21_momentum.data.prices._write_cache"
        ) as mock_write:
            result = fetch_sectors_cached(["AAPL"], universe_sectors={"AAPL": "Information Technology"})

        self.assertEqual(result["AAPL"], "Information Technology")
        mock_write.assert_called_once()

    def test_stale_cache_entry_excluded(self):
        old = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        cached = {"AAPL": {"sector": "Financials", "fetched_at": old}}
        with patch("src.ml.pipeline.p21_momentum.data.prices._read_cache", return_value=cached):
            result = fetch_sectors_cached(["AAPL"])
        self.assertNotIn("AAPL", result)


if __name__ == "__main__":
    unittest.main()
