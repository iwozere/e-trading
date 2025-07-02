import logging
import os
from typing import Dict, List, Optional
import pandas as pd
import requests
from src.notification.logger import setup_logger
_logger = setup_logger(__name__)
from .base_data_downloader import BaseDataDownloader

"""
Data downloader implementation for Alpha Vantage, fetching historical market data for analysis and backtesting.

This module provides the AlphaVantageDataDownloader class for downloading historical OHLCV (Open, High, Low, Close, Volume) data from Alpha Vantage. It supports fetching, saving, loading, and updating data for single or multiple symbols, and is suitable for both research and production trading workflows.

Main Features:
- Download historical data for any stock or ticker from Alpha Vantage
- Save and load data as CSV files
- Update existing data files with new data
- Download data for multiple symbols in batch
- Inherits common logic from BaseDataDownloader for file management

Valid values:
- interval: '1m', '5m', '15m', '30m', '60m', '1d', '1wk', '1mo' (Alpha Vantage supports: 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly)
- period: Any string like '1d', '7d', '1w', '1mo', '3mo', '6mo', '1y', '2y', etc. (used to calculate start_date/end_date)

Classes:
- AlphaVantageDataDownloader: Main class for interacting with Alpha Vantage and managing data downloads
"""

class AlphaVantageDataDownloader(BaseDataDownloader):
    """
    A class to download historical data from Alpha Vantage.

    Parameters:
    -----------
    api_key : str
        Alpha Vantage API key
    data_dir : str
        Directory to store downloaded data files
    """

    def __init__(self, api_key: str, data_dir: Optional[str] = "data"):
        super().__init__(data_dir=data_dir)
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def download_data(
        self, symbol: str, interval: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Download historical data for a given symbol from Alpha Vantage.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            interval: Data interval (e.g., '1d', '1h', '15m')
            start_date: Start date for historical data (YYYY-MM-DD)
            end_date: End date for historical data (YYYY-MM-DD)

        Returns:
            pd.DataFrame: Historical OHLCV data
        """
        try:
            # Alpha Vantage supports: 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly
            interval_map = {
                '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min', '60m': '60min',
                '1d': 'daily', '1h': '60min', '1wk': 'weekly', '1mo': 'monthly'
            }
            av_interval = interval_map.get(interval, 'daily')
            if av_interval in ['daily', 'weekly', 'monthly']:
                function = f'TIME_SERIES_{av_interval.upper()}'
                params = {
                    'function': function,
                    'symbol': symbol,
                    'apikey': self.api_key,
                    'outputsize': 'full',
                    'datatype': 'json',
                }
            else:
                function = 'TIME_SERIES_INTRADAY'
                params = {
                    'function': function,
                    'symbol': symbol,
                    'interval': av_interval,
                    'apikey': self.api_key,
                    'outputsize': 'full',
                    'datatype': 'json',
                }
            response = requests.get(self.base_url, params=params)
            data = response.json()

            # Find the key for the time series data
            ts_key = None
            for k in data.keys():
                if 'Time Series' in k:
                    ts_key = k
                    break
            if not ts_key:
                raise ValueError(f"No time series data found in Alpha Vantage response: {data}")
            ts_data = data[ts_key]

            # Convert to DataFrame
            df = pd.DataFrame.from_dict(ts_data, orient='index')
            df = df.rename(columns=lambda x: x.lower().replace(' ', '').replace('close', 'close').replace('open', 'open').replace('high', 'high').replace('low', 'low').replace('volume', 'volume'))
            # Standardize column names
            col_map = {
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume',
            }
            df = df.rename(columns=col_map)
            # Convert index to datetime and filter by date range
            df['timestamp'] = pd.to_datetime(df.index)
            df = df.reset_index(drop=True)
            df = df.sort_values('timestamp')
            df = df[(df['timestamp'] >= pd.to_datetime(start_date)) & (df['timestamp'] <= pd.to_datetime(end_date))]
            # Ensure all required columns are present
            required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")
            # Convert numeric columns
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        except Exception as e:
            _logger.error("Error downloading data for %s: %s", symbol, e, exc_info=True)
            raise

    def save_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        start_date: str = None,
        end_date: str = None,
    ) -> str:
        try:
            if start_date is None:
                start_date = df["timestamp"].min().strftime("%Y-%m-%d")
            if end_date is None:
                end_date = df["timestamp"].max().strftime("%Y-%m-%d")
            return super().save_data(df, symbol, start_date, end_date)
        except Exception as e:
            _logger.error("Error saving data for %s: %s", symbol, e, exc_info=True)
            raise

    def load_data(self, filepath: str) -> pd.DataFrame:
        try:
            return super().load_data(filepath)
        except Exception as e:
            _logger.error("Error loading data from %s: %s", filepath, e, exc_info=True)
            raise

    def update_data(self, symbol: str, interval: str) -> str:
        try:
            existing_files = [
                f
                for f in os.listdir(self.data_dir)
                if f.startswith(f"{symbol}_{interval}_")
            ]
            if not existing_files:
                df = self.download_data(
                    symbol,
                    interval,
                    "2000-01-01",
                    pd.Timestamp.today().strftime("%Y-%m-%d"),
                )
                return self.save_data(df, symbol)
            latest_file = max(existing_files)
            filepath = os.path.join(self.data_dir, latest_file)
            existing_df = self.load_data(filepath)
            last_date = existing_df["timestamp"].max()
            new_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            new_end = pd.Timestamp.today().strftime("%Y-%m-%d")
            new_df = self.download_data(symbol, interval, new_start, new_end)
            if new_df.empty:
                _logger.info("No new data available for %s", symbol)
                return filepath
            combined_df = pd.concat([existing_df, new_df])
            combined_df = combined_df.drop_duplicates(subset=["timestamp"])
            combined_df = combined_df.sort_values("timestamp")
            return self.save_data(combined_df, symbol)
        except Exception as e:
            _logger.error("Error updating data for %s: %s", symbol, e, exc_info=True)
            raise

    def get_periods(self) -> list:
        return ['1d', '7d', '1w', '1mo', '3mo', '6mo', '1y', '2y']

    def get_intervals(self) -> list:
        return ['1m', '5m', '15m', '30m', '60m', '1d', '1wk', '1mo']

    def is_valid_period_interval(self, period, interval) -> bool:
        return interval in self.get_intervals() and period in self.get_periods()