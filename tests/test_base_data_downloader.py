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

import os
import shutil
import tempfile

import pandas as pd
import pytest
from src.data.base_data_downloader import BaseDataDownloader
from src.data.binance_data_downloader import BinanceDataDownloader
from src.data.yahoo_data_downloader import YahooDataDownloader


class DummyDownloader(BaseDataDownloader):
    def download_data(self, symbol, start_date, end_date=None):
        # Return a simple DataFrame
        data = {
            "timestamp": pd.date_range(start=start_date, periods=2, freq="D"),
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
        df = downloader.download_data("TEST", "2023-01-01", "2023-01-02")
        filepath = downloader.save_data(df, "TEST", "2023-01-01", "2023-01-02")
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
            symbols, downloader.download_data, "2023-01-01", "2023-01-02"
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

        bdd.download_historical_data = fake_download
        symbols = ["BTCUSDT", "ETHUSDT"]
        results = bdd.download_multiple_symbols(
            symbols, "1d", "2023-01-01", "2023-01-02"
        )
        assert set(results.keys()) == set(symbols)
        for path in results.values():
            assert os.path.exists(path)
    finally:
        shutil.rmtree(temp_dir)


def test_yahoo_data_downloader_integration(monkeypatch):
    temp_dir = tempfile.mkdtemp()
    try:
        ydd = YahooDataDownloader(data_dir=temp_dir, interval="1d")

        def fake_download(symbol, start_date, end_date=None, interval=None):
            data = {
                "timestamp": pd.date_range(start=start_date, periods=2, freq="D"),
                "open": [1, 2],
                "high": [2, 3],
                "low": [0, 1],
                "close": [1.5, 2.5],
                "volume": [100, 200],
            }
            return pd.DataFrame(data)

        ydd.download_data = fake_download
        symbols = ["AAPL", "MSFT"]
        results = ydd.download_multiple_symbols(symbols, "2023-01-01", "2023-01-02")
        assert set(results.keys()) == set(symbols)
        for path in results.values():
            assert os.path.exists(path)
    finally:
        shutil.rmtree(temp_dir)
