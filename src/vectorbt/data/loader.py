import pandas as pd
import numpy as np
import os
import glob
from typing import List, Optional
import logging

_logger = logging.getLogger(__name__)

class DataLoader:
    """
    Handles loading and merging of Binance OHLCV data for multiple assets.
    """

    def __init__(self, data_dir: str = "data", symbols: Optional[List[str]] = None):
        """
        Initialize the DataLoader.

        Args:
            data_dir: Path to the directory containing CSV files.
            symbols: List of symbols to load (e.g. ['BTC', 'ETH']).
                     If None, uses default major pairs.
        """
        self.data_dir = data_dir
        self.symbols = symbols or ["BTC", "ETH", "XRP", "LTC"]

    def _find_file(self, symbol: str, interval: str) -> Optional[str]:
        """
        Find the CSV file for a given symbol and interval.
        Handles mapping between pandas 'min' and filename 'm'.
        """
        # Try primary pattern
        pattern = os.path.join(self.data_dir, f"{symbol}USDT_{interval}_*.csv")
        files = glob.glob(pattern)

        # If not found, try mapping 'min' to 'm' (legacy filename format)
        if not files and 'min' in interval:
            legacy_interval = interval.replace('min', 'm')
            pattern = os.path.join(self.data_dir, f"{symbol}USDT_{legacy_interval}_*.csv")
            files = glob.glob(pattern)

        if not files:
            _logger.warning(f"No file found for {symbol} at interval {interval}")
            return None
        return files[0]

    def load_symbol_data(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """
        Load data for a single symbol.
        """
        file_path = self._find_file(symbol, interval)
        if not file_path:
            return None

        try:
            df = pd.read_csv(file_path)

            if 'timestamp' in df.columns:
                # Prefer the new formatted timestamp column
                df['datetime'] = pd.to_datetime(df['timestamp'])
            elif 'close_time' in df.columns:
                # Fallback to Binance close_time (ms)
                df['datetime'] = pd.to_datetime(df['close_time'], unit='ms')
            else:
                _logger.warning(f"No timestamp or close_time found in {file_path}")
                # Try to use the first column if it looks like a date?
                # For now, let's just fail or assume the index is already datetime if we can
                return None

            # Set datetime as index and remove name
            df.set_index('datetime', inplace=True)
            df.index.name = None
            df.sort_index(inplace=True)

            # Select only OHLCV and map to capitalized
            # Handle both lowercase and TitleCase from CSV
            col_map = {
                'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume',
                'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'
            }

            # Find which columns exist
            existing_cols = [c for c in df.columns if c in col_map]
            df = df[existing_cols].rename(columns=col_map)  # type: ignore

            # Ensure we have all 5
            required = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing = [r for r in required if r not in df.columns]
            if missing:
                _logger.warning(f"Missing required columns in {file_path}: {missing}")
                return None

            return df[required]  # type: ignore
        except Exception as e:
            _logger.error(f"Error loading {file_path}: {e}")
            return None

    def load_all_symbols(self, interval: str, start_date: str = "2020-01-01", end_date: str = "2025-12-31") -> pd.DataFrame:
        """
        Load and merge all requested symbols into a MultiIndex DataFrame.
        """
        data_frames = {}
        for symbol in self.symbols:
            df = self.load_symbol_data(symbol, interval)
            if df is not None:
                # Slice by date range (datetime-indexed label slice)
                df = df.loc[pd.Timestamp(start_date) : pd.Timestamp(end_date)]
                data_frames[symbol] = df

        if not data_frames:
            raise ValueError("No data could be loaded for any symbol.")

        # Create MultiIndex DataFrame
        # columns: (symbol, column)
        merged_df = pd.concat(data_frames, axis=1)
        merged_df.columns.names = ['symbol', 'column']

        # Forward fill and drop NaNs that might occur if symbols have different start dates
        # though Binance data is usually well-aligned for these major pairs.
        merged_df = merged_df.ffill().dropna()

        _logger.info(f"Loaded and merged data for {list(data_frames.keys())} at interval {interval}")
        return merged_df

if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    loader = DataLoader(data_dir="data")
    try:
        data = loader.load_all_symbols("1h", start_date="2020-01-01", end_date="2020-01-05")
        print(data.head())
        print(data.columns)
    except Exception as e:
        print(f"Test failed: {e}")
