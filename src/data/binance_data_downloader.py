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

import os
from datetime import datetime, timedelta
from typing import List, Optional, Union

import pandas as pd
from binance.client import Client
from src.notification.logger import setup_logger
import logging
import time

from .base_data_downloader import BaseDataDownloader
from src.model.telegram_bot import Fundamentals

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
    **Bar Limits:** Maximum 1000 bars per request

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
        # Rate limiting: 1200 requests per minute = 1 request per 0.05 seconds
        self.min_request_interval = 0.05
        self.last_request_time = 0

    def _rate_limit(self):
        """Ensure minimum time between requests to respect rate limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def _calculate_batch_dates(self, start_date: datetime, end_date: datetime, interval: str) -> List[tuple]:
        """
        Calculate batch dates to respect the 1000 bar limit.

        Args:
            start_date: Start date
            end_date: End date
            interval: Time interval

        Returns:
            List of (batch_start, batch_end) tuples
        """
        # Convert interval to minutes for calculation
        interval_minutes = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720,
            '1d': 1440, '3d': 4320, '1w': 10080, '1M': 43200
        }

        minutes_per_interval = interval_minutes.get(interval, 1440)  # default to 1d
        max_bars = 1000

        # Calculate maximum time span for 1000 bars
        max_minutes = minutes_per_interval * max_bars
        max_timedelta = timedelta(minutes=max_minutes)

        batches = []
        current_start = start_date

        while current_start < end_date:
            current_end = min(current_start + max_timedelta, end_date)
            batches.append((current_start, current_end))
            current_start = current_end

        return batches

    def download_multiple_symbols(
        self, symbols: List[str], interval: str, start_date: datetime, end_date: datetime
    ):
        """Download historical data for multiple symbols with rate limiting."""

        def download_func(symbol, interval, start_date, end_date):
            return self.get_ohlcv(symbol, interval, start_date, end_date)

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
        """
        Download historical klines/candlestick data from Binance with batching to respect 1000 bar limit.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Kline interval (e.g., '1h', '4h', '1d')
            start_date: Start date as datetime.datetime
            end_date: End date as datetime.datetime

        Returns:
            DataFrame containing the historical data
        """
        try:
            # Calculate batches to respect 1000 bar limit
            batches = self._calculate_batch_dates(start_date, end_date, interval)

            all_klines = []

            for batch_start, batch_end in batches:
                # Apply rate limiting
                self._rate_limit()

                # Convert dates to timestamps
                start_timestamp = int(batch_start.timestamp() * 1000)
                end_timestamp = int(batch_end.timestamp() * 1000)

                _logger.debug("Downloading batch for %s %s: %s to %s",
                             symbol, interval, batch_start, batch_end)

                # Get klines data for this batch
                klines = self.client.get_historical_klines(
                    symbol, interval, start_timestamp, end_timestamp
                )

                all_klines.extend(klines)

                _logger.debug("Retrieved %d bars for batch", len(klines))

            if not all_klines:
                _logger.warning("No data returned for %s %s", symbol, interval)
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(
                all_klines,
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

            # Remove duplicates and sort by timestamp
            df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

            _logger.info("Successfully downloaded %d bars for %s %s", len(df), symbol, interval)
            return df

        except Exception as e:
            _logger.exception("Error downloading Binance data for %s: %s", symbol, str(e))
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
