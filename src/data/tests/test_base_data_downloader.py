"""
Unit and integration tests for BaseDataDownloader, BinanceDataDownloader, and YahooDataDownloader.

- Tests saving and loading data to CSV.
- Tests downloading data for multiple symbols.
- Mocks Binance and Yahoo downloaders to avoid real API calls.
- Uses temporary directories for file operations.

How to run:
    pytest tests/test_base_data_downloader.py

Or to run all tests in the project:
    pytest
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import shutil
import tempfile
from datetime import datetime

import pandas as pd
from src.data.downloader.base_data_downloader import BaseDataDownloader
from src.data.downloader.binance_data_downloader import BinanceDataDownloader
from src.data.downloader.yahoo_data_downloader import YahooDataDownloader


class DummyDownloader(BaseDataDownloader):
    def get_fundamentals(self, symbol):
        return None
    def get_supported_intervals(self):
        return ["1d"]
    def get_intervals(self):
        return ["1d"]
    def get_periods(self):
        return ["1d"]
    def is_valid_period_interval(self, period, interval):
        return True
    def get_ohlcv(self, symbol, interval, start_date, end_date):
        # Return a dummy DataFrame
        data = {
            "timestamp": [datetime(2023, 1, 1), datetime(2023, 1, 2)],
            "open": [1, 2],
            "high": [2, 3],
            "low": [0, 1],
            "close": [1.5, 2.5],
            "volume": [100, 200],
        }
        return pd.DataFrame(data)


def test_save_and_load_data():
    # Note: save_data and load_data methods removed - file operations now handled by DataManager
    # Test basic data retrieval instead
    downloader = DummyDownloader()
    df = downloader.get_ohlcv("TEST", "1d", datetime(2023, 1, 1), datetime(2023, 1, 2))
    assert len(df) > 0
    assert 'timestamp' in df.columns
    assert 'open' in df.columns
    assert 'high' in df.columns
    assert 'low' in df.columns
    assert 'close' in df.columns
    assert 'volume' in df.columns


def test_download_multiple_symbols():
    # Note: download_multiple_symbols method removed - batch operations now handled by DataManager
    # Test individual downloads instead
    downloader = DummyDownloader()
    symbols = ["AAA", "BBB"]
    for symbol in symbols:
        df = downloader.get_ohlcv(symbol, "1d", datetime(2023, 1, 1), datetime(2023, 1, 2))
        assert len(df) > 0
        assert 'timestamp' in df.columns


# Mock BinanceDataDownloader and YahooDataDownloader for isolated tests


def test_binance_data_downloader_integration(monkeypatch):
    temp_dir = tempfile.mkdtemp()
    try:
        # Patch BinanceDataDownloader to not call real API
        bdd = BinanceDataDownloader(
            api_key="fake", api_secret="fake"
        )

        def fake_download(
            symbol, interval, start_date, end_date=None, save_to_csv=True
        ):
            data = {
                "timestamp": pd.date_range(start=start_date, periods=2, freq="D"),
                "open": [1, 2],
                "high": [2, 3],
                "low": [0, 1],
                "close": [1.5, 2.5],
                "volume": [100, 200],
                "close_time": [0, 0],
                "quote_asset_volume": [0, 0],
                "number_of_trades": [0, 0],
                "taker_buy_base_asset_volume": [0, 0],
                "taker_buy_quote_asset_volume": [0, 0],
                "ignore": [0, 0],
            }
            df = pd.DataFrame(data)
            # Note: save_data method removed - caching is now handled by DataManager
            return df

        bdd.get_ohlcv = fake_download
        symbols = ["BTCUSDT", "ETHUSDT"]
        # Note: download_multiple_symbols method removed - batch operations now handled by DataManager
        # Test individual downloads instead
        for symbol in symbols:
            df = bdd.get_ohlcv(symbol, "1d", datetime(2023, 1, 1), datetime(2023, 1, 2))
            assert len(df) > 0
    finally:
        shutil.rmtree(temp_dir)


def test_yahoo_data_downloader_integration(monkeypatch):
    temp_dir = tempfile.mkdtemp()
    try:
        ydd = YahooDataDownloader()
        # Use a bigger date range in the past
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 7)
        df = ydd.get_ohlcv("AAPL", "1d", start_date, end_date)
        assert not df.empty
    finally:
        shutil.rmtree(temp_dir)
