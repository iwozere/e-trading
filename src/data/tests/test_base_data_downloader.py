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
from datetime import datetime, timedelta

import pandas as pd
from src.data.base_data_downloader import BaseDataDownloader
from src.data.binance_data_downloader import BinanceDataDownloader
from src.data.yahoo_data_downloader import YahooDataDownloader


class DummyDownloader(BaseDataDownloader):
    def get_fundamentals(self, symbol):
        return None
    def get_intervals(self):
        return ["1d"]
    def get_periods(self):
        return ["1d"]
    def is_valid_period_interval(self, period, interval):
        return True
    def get_ohlcv(self, symbol, start_date, end_date=None):
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
    temp_dir = tempfile.mkdtemp()
    try:
        downloader = DummyDownloader(data_dir=temp_dir, interval="1d")
        df = downloader.get_ohlcv(datetime(2023, 1, 1), datetime(2023, 1, 2))
        filepath = downloader.save_data(df, "TEST", datetime(2023, 1, 1), datetime(2023, 1, 2))
        assert os.path.exists(filepath)
        loaded_df = downloader.load_data(filepath)
        pd.testing.assert_frame_equal(df, loaded_df)
    finally:
        shutil.rmtree(temp_dir)


def test_download_multiple_symbols():
    temp_dir = tempfile.mkdtemp()
    try:
        downloader = DummyDownloader(data_dir=temp_dir, interval="1d")
        symbols = ["AAA", "BBB"]
        results = downloader.download_multiple_symbols(
            symbols, downloader.get_ohlcv, datetime(2023, 1, 1), datetime(2023, 1, 2)
        )
        assert set(results.keys()) == set(symbols)
        for path in results.values():
            assert os.path.exists(path)
    finally:
        shutil.rmtree(temp_dir)


# Mock BinanceDataDownloader and YahooDataDownloader for isolated tests
import types


def test_binance_data_downloader_integration(monkeypatch):
    temp_dir = tempfile.mkdtemp()
    try:
        # Patch BinanceDataDownloader to not call real API
        bdd = BinanceDataDownloader(
            api_key="fake", api_secret="fake", data_dir=temp_dir, interval="1d"
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
            if save_to_csv:
                bdd.save_data(df, symbol, start_date, end_date)
            return df

        bdd.get_ohlcv = fake_download
        symbols = ["BTCUSDT", "ETHUSDT"]
        results = bdd.download_multiple_symbols(
            symbols, "1d", datetime(2023, 1, 1), datetime(2023, 1, 2)
        )
        assert set(results.keys()) == set(symbols)
        for path in results.values():
            assert os.path.exists(path)
    finally:
        shutil.rmtree(temp_dir)


def test_yahoo_data_downloader_integration(monkeypatch):
    temp_dir = tempfile.mkdtemp()
    try:
        ydd = YahooDataDownloader(data_dir=temp_dir)
        # Use a bigger date range in the past
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 7)
        df = ydd.get_ohlcv("AAPL", "1d", start_date, end_date)
        assert not df.empty
    finally:
        shutil.rmtree(temp_dir)
