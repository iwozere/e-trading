import os
from typing import Dict, List, Optional
import pandas as pd
from abc import ABC, abstractmethod
from src.notification.logger import setup_logger

"""
Abstract base class for data downloaders, defining the interface for downloading historical market data from various sources.

Base Data Downloader Module
--------------------------

This module provides the BaseDataDownloader class, which implements common logic for saving, loading, and managing historical market data files. It is designed to be inherited by specific data downloader classes (e.g., BinanceDataDownloader, YahooDataDownloader) to ensure consistent file handling and batch operations.

Main Features:
- Save pandas DataFrames to CSV files with standardized naming
- Load data from CSV files and parse timestamps
- Download and save data for multiple symbols using a provided download function

Classes:
- BaseDataDownloader: Abstract base class for data downloaders
"""

_logger = setup_logger(__name__)

class BaseDataDownloader(ABC):
    """
    Abstract base class for data downloaders. Provides methods for saving, loading, and managing historical market data files.
    """

    def __init__(self, data_dir: Optional[str] = None, interval: Optional[str] = None):
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "dataset"
        )
        self.interval = interval or "1d"
        os.makedirs(self.data_dir, exist_ok=True)

    def save_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        start_date: str = None,
        end_date: str = None,
    ) -> str:
        """
        Save downloaded data to a CSV file.
        """
        if start_date is None:
            start_date = df["timestamp"].min().strftime("%Y-%m-%d")
        if end_date is None:
            end_date = df["timestamp"].max().strftime("%Y-%m-%d")
        filename = f"{symbol}_{self.interval}_{start_date.replace('-', '')}"
        if end_date:
            filename += f"_{end_date.replace('-', '')}"
        filename += ".csv"
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False)
        return filepath

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
        """
        results = {}
        for symbol in symbols:
            try:
                df = download_func(symbol, *args, **kwargs)
                # Assume start_date and end_date are in kwargs or args
                start_date = kwargs.get("start_date") or args[0] if args else ""
                end_date = kwargs.get("end_date") or (
                    args[1] if len(args) > 1 else None
                )
                filepath = self.save_data(
                    df, symbol, str(start_date), str(end_date) if end_date else None
                )
                results[symbol] = filepath
            except Exception as e:
                _logger.error(f"Error processing {symbol}: {str(e)}", exc_info=True)
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
    def get_ohlcv(self, symbol, interval, start_date, end_date, **kwargs):
        """Download historical data for a given symbol. Must be implemented by subclasses."""
        pass
