"""Tests for Yahoo average volume extraction from yfinance info dict."""

import unittest

from src.data.downloader.yahoo_data_downloader import YahooDataDownloader


class TestYahooAvgVolumeFromInfo(unittest.TestCase):
    """Ensure _avg_volume_from_yf_info picks the first positive volume field."""

    def test_prefers_average_volume(self) -> None:
        info = {"averageVolume": 1_500_000, "averageVolume10days": 2_000_000}
        self.assertEqual(YahooDataDownloader._avg_volume_from_yf_info(info), 1_500_000.0)

    def test_falls_back_to_ten_day(self) -> None:
        info = {"averageVolume10days": 750_000}
        self.assertEqual(YahooDataDownloader._avg_volume_from_yf_info(info), 750_000.0)

    def test_returns_none_when_missing(self) -> None:
        self.assertIsNone(YahooDataDownloader._avg_volume_from_yf_info({}))
        self.assertIsNone(YahooDataDownloader._avg_volume_from_yf_info({"averageVolume": 0}))


if __name__ == "__main__":
    unittest.main()
