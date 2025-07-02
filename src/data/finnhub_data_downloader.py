import logging
import os
from typing import Optional
import pandas as pd
import requests
from src.notification.logger import setup_logger
from .base_data_downloader import BaseDataDownloader

_logger = setup_logger(__name__)

"""
Data downloader implementation for Finnhub, fetching historical market data for analysis and backtesting.

This module provides the FinnhubDataDownloader class for downloading historical OHLCV (Open, High, Low, Close, Volume) data from Finnhub. It supports fetching, saving, loading, and updating data for single or multiple symbols, and is suitable for both research and production trading workflows.

Main Features:
- Download historical data for any stock or ticker from Finnhub (free tier)
- Save and load data as CSV files
- Update existing data files with new data
- Download data for multiple symbols in batch
- Inherits common logic from BaseDataDownloader for file management

Valid values:
- interval: '1m', '5m', '15m', '1h', '1d' (Finnhub free tier: 1m for 30 days, 1d for 5 years; other intervals are resampled)
- period: '1d', '7d', '1mo', '3mo', '6mo', '1y', '2y', '5y'

API limits (free tier):
- 60 requests per minute
- 30 days 1m data, 5 years 1d data
- If you exceed the free API limit or request unsupported data, a clear error will be raised.

Classes:
- FinnhubDataDownloader: Main class for interacting with Finnhub and managing data downloads
"""

class FinnhubDataDownloader(BaseDataDownloader):
    def __init__(self, api_key: str, data_dir: Optional[str] = "data"):
        super().__init__(data_dir=data_dir)
        self.api_key = api_key
        self.base_url = "https://finnhub.io/api/v1/stock/candle"
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def download_data(self, symbol: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Download historical data for a given symbol from Finnhub.

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
            # Finnhub supports: 1, 5, 15, 30, 60 minute, daily, weekly, monthly
            interval_map = {
                '1m': '1', '5m': '5', '15m': '15', '30m': '30', '1h': '60', '1d': 'D'
            }
            finnhub_interval = interval_map.get(interval, 'D')
            # Convert dates to UNIX timestamps (seconds)
            from datetime import datetime
            start_unix = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
            end_unix = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
            params = {
                'symbol': symbol,
                'resolution': finnhub_interval,
                'from': start_unix,
                'to': end_unix,
                'token': self.api_key
            }
            response = requests.get(self.base_url, params=params)
            if response.status_code == 429:
                raise RuntimeError("Finnhub API rate limit exceeded (free tier: 60 requests/minute)")
            if response.status_code != 200:
                raise RuntimeError(f"Finnhub API error: {response.status_code} {response.text}")
            data = response.json()
            if data.get('s') != 'ok':
                raise ValueError(f"No results in Finnhub response: {data}")
            # Finnhub returns: t (timestamp), o (open), h (high), l (low), c (close), v (volume)
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(data['t'], unit='s'),
                'open': data['o'],
                'high': data['h'],
                'low': data['l'],
                'close': data['c'],
                'volume': data['v']
            })
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
        return ['1d', '7d', '1mo', '3mo', '6mo', '1y', '2y', '5y']

    def get_intervals(self) -> list:
        return ['1m', '5m', '15m', '1h', '1d']

    def is_valid_period_interval(self, period, interval) -> bool:
        return interval in self.get_intervals() and period in self.get_periods()