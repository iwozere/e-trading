"""
Standardized data handling utilities.

This module provides consistent data processing, validation, and transformation
across all data sources in the system.
"""

import pandas as pd
from typing import Optional, Dict, Any
from datetime import datetime, timezone
from pathlib import Path
import logging

from src.data.utils.validation import validate_ohlcv_data, get_data_quality_score
from src.data.utils.caching import get_cache

_logger = logging.getLogger(__name__)


class DataHandler:
    """
    Standardized data handler for consistent data processing across data sources.

    Provides methods for:
    - Data validation and quality assessment
    - Standardized data format conversion
    - Caching and persistence
    - Data transformation and cleaning
    """

    def __init__(self, provider: str, cache_enabled: bool = True):
        """
        Initialize data handler.

        Args:
            provider: Data provider name (e.g., 'binance', 'yahoo')
            cache_enabled: Whether to enable caching
        """
        self.provider = provider
        self.cache_enabled = cache_enabled
        self.cache = get_cache() if cache_enabled else None

    def standardize_ohlcv_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
        timestamp_col: str = "timestamp",
        timezone_aware: bool = False
    ) -> pd.DataFrame:
        """
        Standardize OHLCV data to consistent format.

        Args:
            df: Input DataFrame with OHLCV data
            symbol: Trading symbol
            interval: Data interval
            timestamp_col: Name of timestamp column
            timezone_aware: Whether timestamps should be timezone-aware

        Returns:
            Standardized DataFrame with consistent column names and format
        """
        if df.empty:
            return df

        # Create a copy to avoid modifying original
        df_std = df.copy()

        # Standardize column names
        column_mapping = {
            'open': ['open', 'Open', 'OPEN', 'o', 'O'],
            'high': ['high', 'High', 'HIGH', 'h', 'H'],
            'low': ['low', 'Low', 'LOW', 'l', 'L'],
            'close': ['close', 'Close', 'CLOSE', 'c', 'C'],
            'volume': ['volume', 'Volume', 'VOLUME', 'vol', 'Vol', 'VOL', 'v', 'V'],
            'timestamp': [timestamp_col, 'datetime', 'date', 'time', 'Date', 'Time']
        }

        # Map columns to standard names
        for std_name, possible_names in column_mapping.items():
            for col_name in possible_names:
                if col_name in df_std.columns:
                    if std_name != 'timestamp':
                        df_std.rename(columns={col_name: std_name}, inplace=True)
                    break

        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df_std.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Handle timestamp column
        if 'timestamp' not in df_std.columns:
            # Use index if it's datetime
            if isinstance(df_std.index, pd.DatetimeIndex):
                df_std = df_std.reset_index()
                df_std.rename(columns={'index': 'timestamp'}, inplace=True)
            else:
                raise ValueError("No timestamp column found and index is not datetime")

        # Standardize timestamp format
        df_std['timestamp'] = pd.to_datetime(df_std['timestamp'], utc=True)

        # Convert to timezone-naive if required
        if not timezone_aware and df_std['timestamp'].dt.tz is not None:
            df_std['timestamp'] = df_std['timestamp'].dt.tz_localize(None)

        # Ensure numeric columns are float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df_std[col] = pd.to_numeric(df_std[col], errors='coerce')

        # Remove rows with invalid data
        initial_rows = len(df_std)
        df_std.dropna(inplace=True)
        if len(df_std) < initial_rows:
            _logger.warning("Removed %d rows with invalid data for %s", initial_rows - len(df_std), symbol)

        # Sort by timestamp
        df_std.sort_values('timestamp', inplace=True)
        df_std.reset_index(drop=True, inplace=True)

        # Add metadata
        df_std.attrs['symbol'] = symbol
        df_std.attrs['interval'] = interval
        df_std.attrs['provider'] = self.provider
        df_std.attrs['standardized_at'] = datetime.now(timezone.utc)

        return df_std

    def validate_and_score_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        strict_validation: bool = False
    ) -> Dict[str, Any]:
        """
        Validate data and calculate quality score.

        Args:
            df: DataFrame to validate
            symbol: Trading symbol for logging
            strict_validation: Whether to raise errors on validation failures

        Returns:
            Dictionary with validation results and quality score
        """
        # Validate data
        is_valid, errors = validate_ohlcv_data(df)
        quality_score = get_data_quality_score(df)

        # Log results
        if not is_valid:
            _logger.warning("Data validation failed for %s: %s", symbol, errors)
            if strict_validation:
                raise ValueError(f"Data validation failed for {symbol}: {errors}")
        else:
            _logger.info("Data validation passed for %s", symbol)

        _logger.info("Data quality score for %s: %.2f", symbol, quality_score['quality_score'])

        return {
            'is_valid': is_valid,
            'errors': errors,
            'quality_score': quality_score
        }

    def cache_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        file_format: str = "parquet"
    ) -> bool:
        """
        Cache data using the data caching system.

        Args:
            df: DataFrame to cache
            symbol: Trading symbol
            interval: Data interval
            start_date: Start date for data range
            end_date: End date for data range
            file_format: File format (csv, parquet)

        Returns:
            True if caching successful, False otherwise
        """
        if not self.cache_enabled or self.cache is None:
            return False

        try:
            return self.cache.put(
                df, self.provider, symbol, interval, start_date, end_date, file_format
            )
        except Exception:
            _logger.exception("Failed to cache data for %s:", symbol)
            return False

    def get_cached_data(
        self,
        symbol: str,
        interval: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        file_format: str = "parquet"
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve cached data.

        Args:
            symbol: Trading symbol
            interval: Data interval
            start_date: Start date for data range
            end_date: End date for data range
            file_format: File format (csv, parquet)

        Returns:
            Cached DataFrame if found, None otherwise
        """
        if not self.cache_enabled or self.cache is None:
            return None

        try:
            return self.cache.get(
                self.provider, symbol, interval, start_date, end_date, file_format
            )
        except Exception:
            _logger.exception("Failed to retrieve cached data for %s:", symbol)
            return None

    def save_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
        directory: Path,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        file_format: str = "parquet"
    ) -> Path:
        """
        Save data to file with standardized naming.

        Args:
            df: DataFrame to save
            symbol: Trading symbol
            interval: Data interval
            directory: Directory to save file
            start_date: Start date for data range
            end_date: End date for data range
            file_format: File format (csv, parquet)

        Returns:
            Path to saved file
        """
        # Ensure directory exists
        directory.mkdir(parents=True, exist_ok=True)

        # Generate filename
        if start_date and end_date:
            start_str = start_date.strftime("%Y%m%d")
            end_str = end_date.strftime("%Y%m%d")
            filename = f"{symbol}_{interval}_{start_str}_{end_str}.{file_format}"
        else:
            filename = f"{symbol}_{interval}.{file_format}"

        filepath = directory / filename

        # Save data
        try:
            if file_format == "parquet":
                df.to_parquet(filepath, index=False, compression='snappy')
            elif file_format == "csv":
                df.to_csv(filepath, index=False)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")

            _logger.info("Saved data to %s", filepath)
            return filepath

        except Exception:
            _logger.exception("Failed to save data to %s:", filepath)
            raise

    def load_data(self, filepath: Path) -> pd.DataFrame:
        """
        Load data from file.

        Args:
            filepath: Path to data file

        Returns:
            Loaded DataFrame
        """
        try:
            if filepath.suffix.lower() == '.parquet':
                df = pd.read_parquet(filepath)
            else:
                df = pd.read_csv(filepath)

            # Parse timestamp if present
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])

            _logger.info("Loaded data from %s", filepath)
            return df

        except Exception:
            _logger.exception("Failed to load data from %s:", filepath)
            raise

    def merge_data(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        symbol: str
    ) -> pd.DataFrame:
        """
        Merge two DataFrames, handling duplicates and maintaining data integrity.

        Args:
            df1: First DataFrame
            df2: Second DataFrame
            symbol: Trading symbol for logging

        Returns:
            Merged DataFrame
        """
        if df1.empty:
            return df2
        if df2.empty:
            return df1

        # Ensure both have timestamp column
        if 'timestamp' not in df1.columns or 'timestamp' not in df2.columns:
            raise ValueError("Both DataFrames must have 'timestamp' column")

        # Combine DataFrames
        combined = pd.concat([df1, df2], ignore_index=True)

        # Sort by timestamp
        combined.sort_values('timestamp', inplace=True)

        # Remove duplicates based on timestamp
        initial_rows = len(combined)
        combined.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)

        if len(combined) < initial_rows:
            _logger.info("Removed %d duplicate rows for %s", initial_rows - len(combined), symbol)

        # Reset index
        combined.reset_index(drop=True, inplace=True)

        # Validate merged data
        self.validate_and_score_data(combined, symbol)

        return combined

    def resample_data(
        self,
        df: pd.DataFrame,
        target_interval: str,
        symbol: str
    ) -> pd.DataFrame:
        """
        Resample data to a different interval.

        Args:
            df: Input DataFrame
            target_interval: Target interval (e.g., '1h', '1d')
            symbol: Trading symbol for logging

        Returns:
            Resampled DataFrame
        """
        if df.empty or 'timestamp' not in df.columns:
            return df

        # Set timestamp as index for resampling
        df_resampled = df.set_index('timestamp').copy()

        # Define resampling rules
        resampling_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }

        try:
            # Resample data
            df_resampled = df_resampled.resample(target_interval).agg(resampling_rules)

            # Remove rows with all NaN values
            df_resampled.dropna(inplace=True)

            # Reset index
            df_resampled.reset_index(inplace=True)

            _logger.info("Resampled %s data to %s interval", symbol, target_interval)

            # Validate resampled data
            self.validate_and_score_data(df_resampled, symbol)

            return df_resampled

        except Exception:
            _logger.exception("Failed to resample data for %s:", symbol)
            return df


# Global data handler instances
_data_handlers = {}


def get_data_handler(provider: str, cache_enabled: bool = True) -> DataHandler:
    """
    Get or create a data handler for a specific provider.

    Args:
        provider: Data provider name
        cache_enabled: Whether to enable caching

    Returns:
        DataHandler instance for the provider
    """
    if provider not in _data_handlers:
        _data_handlers[provider] = DataHandler(provider, cache_enabled)

    return _data_handlers[provider]
