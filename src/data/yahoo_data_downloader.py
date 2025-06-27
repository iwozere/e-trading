import datetime
import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf
from src.notification.logger import _logger

from .base_data_downloader import BaseDataDownloader

"""
Yahoo Data Downloader Module
---------------------------

This module provides the YahooDataDownloader class for downloading historical OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance. It supports fetching, saving, loading, and updating data for single or multiple symbols, and is suitable for both research and production trading workflows.

Main Features:
- Download historical data for any stock or ticker from Yahoo Finance
- Save and load data as CSV files
- Update existing data files with new data
- Download data for multiple symbols in batch
- Inherits common logic from BaseDataDownloader for file management

Classes:
- YahooDataDownloader: Main class for interacting with Yahoo Finance and managing data downloads
"""


class YahooDataDownloader(BaseDataDownloader):
    """
    A class to download historical data from Yahoo Finance.

    This class provides methods to:
    1. Download historical OHLCV data for a given symbol
    2. Save data to CSV files
    3. Load data from CSV files
    4. Update existing data files with new data

    Parameters:
    -----------
    data_dir : str
        Directory to store downloaded data files
    """

    def __init__(self, data_dir: str = "data"):
        super().__init__(data_dir=data_dir)

        # Set up logging
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def download_data(
        self, symbol: str, interval: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Download historical data for a given symbol.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            interval: Data interval
            start_date: Start date for historical data
            end_date: End date for historical data

        Returns:
            pd.DataFrame: Historical OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)

            # Rename columns to match standard format
            df = df.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )

            # Add timestamp column
            df["timestamp"] = df.index

            # Reset index to make timestamp a regular column
            df = df.reset_index(drop=True)

            # Ensure all required columns are present
            required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")

            return df

        except Exception as e:
            _logger.error(f"Error downloading data for {symbol}: {str(e)}")
            raise

    def save_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        start_date: str = None,
        end_date: str = None,
    ) -> str:
        """
        Save downloaded data to a CSV file.

        Args:
            df: DataFrame containing historical data
            symbol: Stock symbol
            start_date: Start date for historical data
            end_date: End date for historical data

        Returns:
            str: Path to the saved file
        """
        try:
            # If start_date or end_date are not provided, extract from df
            if start_date is None:
                start_date = df["timestamp"].min().strftime("%Y-%m-%d")
            if end_date is None:
                end_date = df["timestamp"].max().strftime("%Y-%m-%d")
            return super().save_data(df, symbol, start_date, end_date)

        except Exception as e:
            _logger.error(f"Error saving data for {symbol}: {str(e)}")
            raise

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data from a CSV file.

        Args:
            filepath: Path to the CSV file

        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            return super().load_data(filepath)

        except Exception as e:
            _logger.error(f"Error loading data from {filepath}: {str(e)}")
            raise

    def update_data(self, symbol: str, interval: str) -> str:
        """
        Update existing data file with new data.

        Args:
            symbol: Stock symbol
            interval: Data interval

        Returns:
            str: Path to the updated file
        """
        try:
            # Find existing data file
            existing_files = [
                f
                for f in os.listdir(self.data_dir)
                if f.startswith(f"{symbol}_{interval}_")
            ]
            if not existing_files:
                # If no existing file, download new data
                df = self.download_data(
                    symbol,
                    interval,
                    "2000-01-01",
                    pd.Timestamp.today().strftime("%Y-%m-%d"),
                )
                return self.save_data(df, symbol)

            # Load existing data
            latest_file = max(existing_files)
            filepath = os.path.join(self.data_dir, latest_file)
            existing_df = self.load_data(filepath)

            # Get last date in existing data
            last_date = existing_df["timestamp"].max()

            # Download new data from last date
            new_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            new_end = pd.Timestamp.today().strftime("%Y-%m-%d")
            new_df = self.download_data(symbol, interval, new_start, new_end)

            if new_df.empty:
                _logger.info(f"No new data available for {symbol}")
                return filepath

            # Combine existing and new data
            combined_df = pd.concat([existing_df, new_df])
            combined_df = combined_df.drop_duplicates(subset=["timestamp"])
            combined_df = combined_df.sort_values("timestamp")

            # Save updated data
            return self.save_data(combined_df, symbol)

        except Exception as e:
            _logger.error(f"Error updating data for {symbol}: {str(e)}")
            raise

    def download_multiple_symbols(
        self, symbols: List[str], interval: str, start_date: str, end_date: str
    ) -> Dict[str, str]:
        """
        Download data for multiple symbols.

        Args:
            symbols: List of stock symbols
            interval: Data interval
            start_date: Start date for historical data
            end_date: End date for historical data

        Returns:
            Dict[str, str]: Dictionary mapping symbols to file paths
        """

        def download_func(symbol, interval, start_date, end_date):
            return self.download_data(symbol, interval, start_date, end_date)

        return super().download_multiple_symbols(
            symbols, download_func, interval, start_date, end_date
        )


if __name__ == "__main__":
    # Example usage
    downloader = YahooDataDownloader(data_dir="data")

    # Download data for a single symbol
    symbol = "AAPL"
    interval = "1d"
    start_date = "2020-01-01"
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    df = downloader.download_data(symbol, interval, start_date, end_date)
    filepath = downloader.save_data(df, symbol)
    print(f"Data saved to {filepath}")

    # Download data for multiple symbols
    symbols = ["AAPL", "MSFT", "GOOGL"]
    interval = "1d"
    start_date = "2020-01-01"
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    results = downloader.download_multiple_symbols(
        symbols, interval, start_date, end_date
    )
    print("Downloaded files:", results)
