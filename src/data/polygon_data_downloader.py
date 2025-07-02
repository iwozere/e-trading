import logging
import os
from typing import Optional
import pandas as pd
import requests
from src.notification.logger import setup_logger
from .base_data_downloader import BaseDataDownloader

_logger = setup_logger(__name__)

"""
Data downloader implementation for Polygon.io, fetching historical market data for analysis and backtesting.

This module provides the PolygonDataDownloader class for downloading historical OHLCV (Open, High, Low, Close, Volume) data from Polygon.io. It supports fetching, saving, loading, and updating data for single or multiple symbols, and is suitable for both research and production trading workflows.

Main Features:
- Download historical data for any stock or ticker from Polygon.io (free tier)
- Save and load data as CSV files
- Update existing data files with new data
- Download data for multiple symbols in batch
- Inherits common logic from BaseDataDownloader for file management

Valid values:
- interval: '1m', '5m', '15m', '1h', '1d' (Polygon free tier: 1m for 2 months, 1d for 2 years; other intervals are resampled)
- period: '1d', '7d', '1mo', '3mo', '6mo', '1y', '2y'

API limits (free tier):
- 5 requests per minute
- 2 years daily data, 2 months minute data
- If you exceed the free API limit or request unsupported data, a clear error will be raised.

Classes:
- PolygonDataDownloader: Main class for interacting with Polygon.io and managing data downloads
"""

class PolygonDataDownloader(BaseDataDownloader):
    def __init__(self, api_key: str, data_dir: Optional[str] = "data"):
        super().__init__(data_dir=data_dir)
        self.api_key = api_key
        self.base_url = "https://api.polygon.io/v2/aggs/ticker"
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def download_data(self, symbol: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Download historical data for a given symbol from Polygon.io.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            interval: Data interval ('1m', '5m', '15m', '1h', '1d')
            start_date: Start date for historical data (YYYY-MM-DD)
            end_date: End date for historical data (YYYY-MM-DD)

        Returns:
            pd.DataFrame: Historical OHLCV data
        """
        try:
            # Validate interval and period
            if not self.is_valid_period_interval('1d', interval):
                raise ValueError(f"Unsupported interval: {interval}")
            # Polygon supports 'minute' and 'day' granularity
            if interval == '1d':
                timespan = 'day'
            else:
                timespan = 'minute'
            url = f"{self.base_url}/{symbol}/range/1/{timespan}/{start_date}/{end_date}"
            params = {
                'adjusted': 'true',
                'sort': 'asc',
                'apiKey': self.api_key
            }
            response = requests.get(url, params=params)
            if response.status_code == 429:
                raise RuntimeError("Polygon.io API rate limit exceeded (free tier: 5 requests/minute)")
            if response.status_code != 200:
                raise RuntimeError(f"Polygon.io API error: {response.status_code} {response.text}")
            data = response.json()
            if 'results' not in data:
                raise ValueError(f"No results in Polygon.io response: {data}")
            df = pd.DataFrame(data['results'])
            # Polygon returns: t (timestamp ms), o (open), h (high), l (low), c (close), v (volume)
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
            df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            # Resample if needed
            if interval == '5m':
                df = df.set_index('timestamp').resample('5T').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna().reset_index()
            elif interval == '15m':
                df = df.set_index('timestamp').resample('15T').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna().reset_index()
            elif interval == '1h':
                df = df.set_index('timestamp').resample('1H').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna().reset_index()
            # For '1m' and '1d', no resampling needed
            return df
        except Exception as e:
            _logger.error("Error downloading data for %s: %s", symbol, e, exc_info=True)
            raise

    def get_periods(self) -> list:
        return ['1d', '7d', '1mo', '3mo', '6mo', '1y', '2y']

    def get_intervals(self) -> list:
        return ['1m', '5m', '15m', '1h', '1d']

    def is_valid_period_interval(self, period, interval) -> bool:
        return interval in self.get_intervals() and period in self.get_periods()