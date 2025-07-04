import logging
import os
from typing import Optional
import pandas as pd
import requests
from src.notification.logger import setup_logger
from .base_data_downloader import BaseDataDownloader

_logger = setup_logger(__name__)

"""
Data downloader implementation for Twelve Data, fetching historical market data for analysis and backtesting.

This module provides the TwelveDataDataDownloader class for downloading historical OHLCV (Open, High, Low, Close, Volume) data from Twelve Data. It supports fetching, saving, loading, and updating data for single or multiple symbols, and is suitable for both research and production trading workflows.

Main Features:
- Download historical data for any stock or ticker from Twelve Data (free tier)
- Save and load data as CSV files
- Update existing data files with new data
- Download data for multiple symbols in batch
- Inherits common logic from BaseDataDownloader for file management

Valid values:
- interval: '1m', '5m', '15m', '1h', '1d' (Twelve Data free tier: 1m for 1 month, 1d for 10 years; other intervals are resampled)
- period: '1d', '7d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y'

API limits (free tier):
- 8 requests per minute, 800 per day
- 1 month 1m data, 10 years 1d data
- If you exceed the free API limit or request unsupported data, a clear error will be raised.

Classes:
- TwelveDataDataDownloader: Main class for interacting with Twelve Data and managing data downloads
"""

class TwelveDataDataDownloader(BaseDataDownloader):
    def __init__(self, api_key: str, data_dir: Optional[str] = "data"):
        super().__init__(data_dir=data_dir)
        self.api_key = api_key
        self.base_url = "https://api.twelvedata.com/time_series"
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def get_ohlcv(self, symbol: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Download historical data for a given symbol from Twelve Data.

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
            # Twelve Data supports: 1min, 5min, 15min, 30min, 1h, 4h, 1d, 1wk, 1mo
            interval_map = {
                '1m': '1min', '5m': '5min', '15m': '15min', '1h': '1h', '1d': '1day'
            }
            td_interval = interval_map.get(interval, '1day')
            params = {
                'symbol': symbol,
                'interval': td_interval,
                'start_date': start_date,
                'end_date': end_date,
                'apikey': self.api_key,
                'format': 'JSON',
                'outputsize': 5000  # max per request
            }
            response = requests.get(self.base_url, params=params)
            if response.status_code == 429:
                raise RuntimeError("Twelve Data API rate limit exceeded (free tier: 8 requests/minute, 800/day)")
            if response.status_code != 200:
                raise RuntimeError(f"Twelve Data API error: {response.status_code} {response.text}")
            data = response.json()
            if 'values' not in data:
                raise ValueError(f"No results in Twelve Data response: {data}")
            df = pd.DataFrame(data['values'])
            # Twelve Data returns: datetime, open, high, low, close, volume (all as strings)
            df['timestamp'] = pd.to_datetime(df['datetime'])
            df = df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'})
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            # Convert numeric columns
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
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
        return ['1d', '7d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y']

    def get_intervals(self) -> list:
        return ['1m', '5m', '15m', '1h', '1d']

    def is_valid_period_interval(self, period, interval) -> bool:
        return interval in self.get_intervals() and period in self.get_periods()