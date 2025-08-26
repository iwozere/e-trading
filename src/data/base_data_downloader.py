import time
from typing import Dict, List, Optional
import pandas as pd
from abc import ABC, abstractmethod
from src.notification.logger import setup_logger
from src.model.telegram_bot import Fundamentals
from datetime import datetime
from pathlib import Path

"""
Abstract base class for data downloaders, defining the interface for downloading historical market data from various sources.

Base Data Downloader Module
--------------------------

This module provides the BaseDataDownloader class, which implements common logic for saving, loading, and managing historical market data files. It is designed to be inherited by specific data downloader classes (e.g., BinanceDataDownloader, YahooDataDownloader) to ensure consistent file handling and batch operations.

Main Features:
- Save pandas DataFrames to CSV files with standardized naming
- Load data from CSV files and parse timestamps
- Download and save data for multiple symbols using a provided download function
- Rate limiting support for batch operations

Classes:
- BaseDataDownloader: Abstract base class for data downloaders
"""

_logger = setup_logger(__name__)


class BaseDataDownloader(ABC):
    """
    Abstract base class for data downloaders. Provides methods for saving, loading, and managing historical market data files.
    """

    def __init__(self, data_dir: Optional[str] = None, interval: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parents[2] / "dataset"
        self.interval = interval or "1d"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Default rate limiting (can be overridden by subclasses)
        self.min_request_interval = 0.1  # 100ms default
        self.last_request_time = 0

    def _rate_limit(self):
        """Ensure minimum time between requests to respect rate limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def save_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
        start_date: datetime = None,
        end_date: datetime = None,
        directory: str = None,
    ) -> str:
        """
        Save downloaded data to a CSV file.
        start_date and end_date should be datetime.datetime objects (or None).
        directory: Optional directory to save the file. If not provided, uses self.data_dir.
        """
        if start_date is None:
            start_date = df["timestamp"].min()
        if end_date is None:
            end_date = df["timestamp"].max()
        # Convert to string for filename
        start_date_str = start_date.strftime("%Y-%m-%d") if isinstance(start_date, datetime) else str(start_date)
        end_date_str = end_date.strftime("%Y-%m-%d") if isinstance(end_date, datetime) else str(end_date)
        filename = f"{symbol}_{interval}_{start_date_str.replace('-', '')}"
        if end_date_str:
            filename += f"_{end_date_str.replace('-', '')}"
        filename += ".csv"
        # Use provided directory or default
        target_dir = Path(directory) if directory else self.data_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        filepath = target_dir / filename
        df.to_csv(filepath, index=False)
        return str(filepath)

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data from a CSV file.
        """
        df = pd.read_csv(filepath)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def download_multiple_symbols(
        self, symbols: List[str], download_func, *args, **kwargs
    ) -> Dict[str, str]:
        """
        Download data for multiple symbols using the provided download_func.
        start_date and end_date should be datetime.datetime objects (or None).
        Includes rate limiting between symbol processing.
        """
        results = {}
        total_symbols = len(symbols)

        for i, symbol in enumerate(symbols):
            try:
                _logger.info("Processing symbol %s (%d/%d)", symbol, i + 1, total_symbols)

                df = download_func(symbol, *args, **kwargs)
                # Assume start_date and end_date are in kwargs or args
                start_date = kwargs.get("start_date") or (args[0] if args else None)
                end_date = kwargs.get("end_date") or (args[1] if len(args) > 1 else None)
                filepath = self.save_data(
                    df, symbol, self.interval, start_date, end_date
                )
                results[symbol] = filepath

                # Rate limiting between symbols (don't sleep after the last symbol)
                if i < total_symbols - 1:
                    self._rate_limit()

            except Exception as e:
                _logger.exception("Error processing %s: %s", symbol, str(e))
                continue
        return results

    @abstractmethod
    def get_periods(self) -> list:
        """Return the list of valid periods for this data downloader."""
        pass

    @abstractmethod
    def get_intervals(self) -> list:
        """Return the list of valid intervals for this data downloader."""
        pass

    @abstractmethod
    def is_valid_period_interval(self, period, interval) -> bool:
        """Return True if the provided period/interval combination is valid for this data downloader."""
        pass

    @abstractmethod
    def get_ohlcv(self, symbol, interval, start_date: datetime, end_date: datetime, **kwargs):
        """Download historical data for a given symbol. start_date and end_date must be datetime.datetime."""
        pass

    @abstractmethod
    def get_fundamentals(self, symbol: str) -> Fundamentals:
        """Return the fundamentals for a given symbol. Must be implemented by subclasses."""
        pass
