"""
Binance Data Downloader Module
-----------------------------

This module provides the BinanceDataDownloader class for downloading historical OHLCV (Open, High, Low, Close, Volume) data from the Binance exchange. It supports fetching data for single or multiple symbols and saving the results as CSV files for use in backtesting and analysis workflows.

Main Features:
- Download historical candlestick data for cryptocurrencies
- Save data to CSV files
- Download data for multiple symbols in batch
- Inherits common logic from BaseDataDownloader for file management

Valid values:
- interval: '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'
- period: Any string like '1d', '7d', '1w', '1mo', '3mo', '6mo', '1y', '2y', etc. (used to calculate start_date/end_date)

Classes:
- BinanceDataDownloader: Main class for interacting with the Binance API and managing data downloads
"""

from datetime import datetime
from typing import List, Optional, Union

import pandas as pd
from binance.client import Client
from src.notification.logger import setup_logger
import logging
import os
from src.model.telegram_bot import Fundamentals

from .base_data_downloader import BaseDataDownloader

_logger = setup_logger(__name__)


class BinanceDataDownloader(BaseDataDownloader):
    """
    A class to download historical data from Binance.

    This class provides methods to:
    1. Download historical OHLCV data for cryptocurrencies
    2. Save data to CSV files
    3. Load data from CSV files
    4. Update existing data files with new data
    5. Get fundamental data (NotImplementedError - Binance doesn't provide stock fundamentals)

    **Fundamental Data Capabilities:**
    - ❌ PE Ratio (cryptocurrency exchange)
    - ❌ Financial Ratios (cryptocurrency exchange)
    - ❌ Growth Metrics (cryptocurrency exchange)
    - ❌ Company Information (cryptocurrency exchange)
    - ❌ Market Data (only cryptocurrency data)
    - ❌ Profitability Metrics (cryptocurrency exchange)
    - ❌ Valuation Metrics (cryptocurrency exchange)

    **Data Quality:** N/A - Binance is for cryptocurrencies, not stocks
    **Rate Limits:** 1200 requests per minute (free tier)
    **Coverage:** Cryptocurrencies only

    Parameters:
    -----------
    api_key : str
        Binance API key
    secret_key : str
        Binance secret key
    data_dir : str
        Directory to store downloaded data files

    Example:
    --------
    >>> from datetime import datetime
    >>> downloader = BinanceDataDownloader("YOUR_API_KEY", "YOUR_SECRET_KEY")
    >>> # Get OHLCV data for cryptocurrency
    >>> df = downloader.get_ohlcv("BTCUSDT", "1d", datetime(2023, 1, 1), datetime(2023, 12, 31))
    >>> # Get fundamental data (will raise NotImplementedError)
    >>> try:
    >>>     fundamentals = downloader.get_fundamentals("AAPL")
    >>> except NotImplementedError as e:
    >>>     print(f"Not supported: {e}")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        data_dir: Optional[str] = None,
        interval: Optional[str] = None,
    ):
        super().__init__(data_dir=data_dir, interval=interval)
        self.client = Client(api_key, api_secret)

    def download_multiple_symbols(
        self, symbols: List[str], interval: str, start_date: datetime, end_date: datetime
    ):
        """Download historical data for multiple symbols."""

        def download_func(symbol, interval, start_date, end_date):
            return self.download_historical_data(
                symbol, interval, start_date, end_date, save_to_csv=False
            )

        return super().download_multiple_symbols(
            symbols, download_func, interval, start_date, end_date
        )

    def get_periods(self) -> list:
        return ['1d', '7d', '1w', '1mo', '3mo', '6mo', '1y', '2y']

    def get_intervals(self) -> list:
        return ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']

    def is_valid_period_interval(self, period, interval) -> bool:
        return interval in self.get_intervals() and period in self.get_periods()

    def get_ohlcv(self, symbol, interval, start_date: datetime, end_date: datetime):
        # Wrapper for unified interface
        """
        Download historical klines/candlestick data from Binance.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Kline interval (e.g., '1h', '4h', '1d')
            start_date: Start date as datetime.datetime
            end_date: End date as datetime.datetime
            save_to_csv: Whether to save the data to a CSV file

        Returns:
            DataFrame containing the historical data
        """
        try:
            # Convert dates to timestamps
            start_timestamp = int(start_date.timestamp() * 1000)
            end_timestamp = int(end_date.timestamp() * 1000)

            # Get klines data
            klines = self.client.get_historical_klines(
                symbol, interval, start_timestamp, end_timestamp
            )

            # Convert to DataFrame
            df = pd.DataFrame(
                klines,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_asset_volume",
                    "number_of_trades",
                    "taker_buy_base_asset_volume",
                    "taker_buy_quote_asset_volume",
                    "ignore",
                ],
            )

            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

            # Convert string values to float
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)

            return df
        except Exception as e:
            _logger.error("Error downloading Binance data for %s: %s", symbol, e, exc_info=True)
            raise

    def get_fundamentals(self, symbol: str) -> Fundamentals:
        """
        Get fundamental data for a given symbol.

        Note: Binance doesn't provide fundamental data for stocks, so this method raises NotImplementedError.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')

        Returns:
            Fundamentals: Fundamental data for the stock

        Raises:
            NotImplementedError: Binance doesn't provide fundamental data for stocks
        """
        raise NotImplementedError("Binance doesn't provide fundamental data for stocks")
