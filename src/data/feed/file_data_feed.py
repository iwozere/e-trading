"""
File Data Feed Module
--------------------

This module provides a data feed that reads historical data from CSV files.
Perfect for backtesting and testing trading strategies with custom datasets.

Features:
- Load OHLCV data from CSV files
- Support for various CSV formats
- Configurable column mapping
- Data validation and cleaning
- Backtrader integration
- Optional real-time simulation mode

Classes:
- FileDataFeed: Data feed for CSV files with optional real-time simulation
"""

import os
import sys
import time
import threading
from typing import Dict, Any
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import backtrader as bt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class FileDataFeed(bt.feed.DataBase):
    """
    Data feed that reads OHLCV data from CSV files.

    Supports both static backtesting and simulated real-time mode for testing.
    """

    lines = ('open', 'high', 'low', 'close', 'volume', 'openinterest')

    params = (
        ('dataname', None),  # Path to CSV file
        ('symbol', None),    # Symbol name for logging
        ('datetime_col', 'datetime'),  # Name of datetime column
        ('open_col', 'open'),
        ('high_col', 'high'),
        ('low_col', 'low'),
        ('close_col', 'close'),
        ('volume_col', 'volume'),
        ('openinterest_col', None),  # Optional
        ('separator', ','),
        ('datetime_format', None),  # Auto-detect if None
        ('timezone', None),  # Timezone for datetime parsing
        ('reverse', False),  # Reverse data order
        ('fromdate', None),  # Filter start date
        ('todate', None),    # Filter end date
        ('simulate_realtime', False),  # Simulate real-time data delivery
        ('realtime_interval', 60),     # Seconds between real-time updates
        ('on_new_bar', None),          # Callback for new bars in real-time mode
    )

    def __init__(self, **kwargs):
        """
        Initialize file data feed.

        Args:
            **kwargs: Parameters including file path and column mappings
        """
        super().__init__(**kwargs)

        self.df = None
        self.current_index = 0
        self.is_realtime = self.p.simulate_realtime
        self.realtime_thread = None
        self.should_stop = False
        self.data_queue = []
        self.last_delivered = None

        # Load and prepare data
        self._load_csv_data()

        if self.is_realtime:
            self._start_realtime_simulation()

    def _load_csv_data(self):
        """Load and validate CSV data."""
        try:
            file_path = Path(self.p.dataname)
            if not file_path.exists():
                raise FileNotFoundError(f"CSV file not found: {file_path}")

            _logger.info("Loading CSV data from: %s", file_path)

            # Read CSV file
            df = pd.read_csv(
                file_path,
                sep=self.p.separator
            )

            # Validate required columns
            self._validate_columns(df)

            # Prepare DataFrame
            df = self._prepare_dataframe(df)

            # Apply date filters
            if self.p.fromdate or self.p.todate:
                df = self._apply_date_filters(df)

            # Reverse if requested
            if self.p.reverse:
                df = df.iloc[::-1].reset_index(drop=True)

            # Validate data quality
            self._validate_data_quality(df)

            self.df = df
            _logger.info("Loaded %d rows of data for %s", len(df), self.p.symbol or 'Unknown')

        except Exception as e:
            _logger.exception("Error loading CSV data: %s", e)
            raise

    def _parse_datetime(self, date_str):
        """Parse datetime string using specified format."""
        try:
            return pd.to_datetime(date_str, format=self.p.datetime_format)
        except Exception as e:
            _logger.warning("Error parsing datetime '%s': %s", date_str, e)
            return pd.to_datetime(date_str)

    def _validate_columns(self, df):
        """Validate that required columns exist in the DataFrame."""
        required_cols = {
            'datetime': self.p.datetime_col,
            'open': self.p.open_col,
            'high': self.p.high_col,
            'low': self.p.low_col,
            'close': self.p.close_col,
            'volume': self.p.volume_col,
        }

        missing_cols = []
        for col_type, col_name in required_cols.items():
            if col_name and col_name not in df.columns:
                missing_cols.append(f"{col_type} ({col_name})")

        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

    def _prepare_dataframe(self, df):
        """Prepare DataFrame with proper column names and types."""
        # Create a copy to avoid modifying original
        prepared_df = df.copy()

        # Rename columns to standard names
        column_mapping = {}
        if self.p.datetime_col and self.p.datetime_col != 'datetime':
            column_mapping[self.p.datetime_col] = 'datetime'
        if self.p.open_col and self.p.open_col != 'open':
            column_mapping[self.p.open_col] = 'open'
        if self.p.high_col and self.p.high_col != 'high':
            column_mapping[self.p.high_col] = 'high'
        if self.p.low_col and self.p.low_col != 'low':
            column_mapping[self.p.low_col] = 'low'
        if self.p.close_col and self.p.close_col != 'close':
            column_mapping[self.p.close_col] = 'close'
        if self.p.volume_col and self.p.volume_col != 'volume':
            column_mapping[self.p.volume_col] = 'volume'

        if column_mapping:
            prepared_df = prepared_df.rename(columns=column_mapping)

        # Ensure datetime column is properly parsed
        if 'datetime' in prepared_df.columns:
            if not pd.api.types.is_datetime64_any_dtype(prepared_df['datetime']):
                prepared_df['datetime'] = pd.to_datetime(prepared_df['datetime'])

            # Apply timezone if specified
            if self.p.timezone:
                if prepared_df['datetime'].dt.tz is None:
                    prepared_df['datetime'] = prepared_df['datetime'].dt.tz_localize(self.p.timezone)
                else:
                    prepared_df['datetime'] = prepared_df['datetime'].dt.tz_convert(self.p.timezone)

        # Convert price columns to float
        price_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in price_cols:
            if col in prepared_df.columns:
                prepared_df[col] = pd.to_numeric(prepared_df[col], errors='coerce')

        # Add openinterest column if not present
        if 'openinterest' not in prepared_df.columns:
            if self.p.openinterest_col and self.p.openinterest_col in df.columns:
                prepared_df['openinterest'] = pd.to_numeric(df[self.p.openinterest_col], errors='coerce')
            else:
                prepared_df['openinterest'] = 0.0

        # Sort by datetime
        if 'datetime' in prepared_df.columns:
            prepared_df = prepared_df.sort_values('datetime').reset_index(drop=True)

        return prepared_df

    def _apply_date_filters(self, df):
        """Apply date range filters to the DataFrame."""
        if 'datetime' not in df.columns:
            return df

        mask = pd.Series(True, index=df.index)

        if self.p.fromdate:
            if isinstance(self.p.fromdate, str):
                fromdate = pd.to_datetime(self.p.fromdate)
            else:
                fromdate = self.p.fromdate
            mask &= df['datetime'] >= fromdate

        if self.p.todate:
            if isinstance(self.p.todate, str):
                todate = pd.to_datetime(self.p.todate)
            else:
                todate = self.p.todate
            mask &= df['datetime'] <= todate

        filtered_df = df[mask].reset_index(drop=True)
        _logger.info("Applied date filters: %d -> %d rows", len(df), len(filtered_df))

        return filtered_df

    def _validate_data_quality(self, df):
        """Validate data quality and log warnings for issues."""
        if df.empty:
            raise ValueError("DataFrame is empty after processing")

        # Check for missing values
        missing_counts = df[['open', 'high', 'low', 'close', 'volume']].isnull().sum()
        if missing_counts.any():
            _logger.warning("Missing values found: %s", missing_counts.to_dict())

        # Check for invalid OHLC relationships
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        ).sum()

        if invalid_ohlc > 0:
            _logger.warning("Found %d rows with invalid OHLC relationships", invalid_ohlc)

        # Check for negative values
        negative_values = (df[['open', 'high', 'low', 'close', 'volume']] < 0).sum().sum()
        if negative_values > 0:
            _logger.warning("Found %d negative price/volume values", negative_values)

    def _start_realtime_simulation(self):
        """Start real-time simulation thread."""
        if not self.is_realtime:
            return

        self.realtime_thread = threading.Thread(
            target=self._realtime_simulation_loop,
            daemon=True
        )
        self.realtime_thread.start()
        _logger.info("Started real-time simulation for %s", self.p.symbol or 'file data')

    def _realtime_simulation_loop(self):
        """Main loop for real-time simulation."""
        while not self.should_stop and self.current_index < len(self.df):
            try:
                # Get next row
                if self.current_index < len(self.df):
                    row = self.df.iloc[self.current_index]

                    # Add to queue for delivery
                    self.data_queue.append({
                        'datetime': row['datetime'],
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                        'volume': row['volume'],
                        'openinterest': row.get('openinterest', 0.0)
                    })

                    # Call callback if provided
                    if self.p.on_new_bar:
                        try:
                            self.p.on_new_bar(
                                self.p.symbol or 'file_data',
                                row['datetime'],
                                row.to_dict()
                            )
                        except Exception as e:
                            _logger.exception("Error in on_new_bar callback: %s", e)

                    self.current_index += 1
                    self.last_delivered = row['datetime']

                    _logger.debug("Delivered bar %d/%d for %s at %s",
                                self.current_index, len(self.df),
                                self.p.symbol or 'file_data', row['datetime'])

                # Wait for next interval
                time.sleep(self.p.realtime_interval)

            except Exception as e:
                _logger.exception("Error in real-time simulation loop: %s", e)
                time.sleep(self.p.realtime_interval)

    def _load(self):
        """
        Backtrader's _load method - called when Backtrader needs more data.
        """
        if self.df is None or self.df.empty:
            return False

        if self.is_realtime:
            # Real-time mode: deliver from queue
            if self.data_queue:
                data_point = self.data_queue.pop(0)
                self.lines.datetime[0] = bt.date2num(data_point['datetime'])
                self.lines.open[0] = data_point['open']
                self.lines.high[0] = data_point['high']
                self.lines.low[0] = data_point['low']
                self.lines.close[0] = data_point['close']
                self.lines.volume[0] = data_point['volume']
                self.lines.openinterest[0] = data_point['openinterest']
                return True
            return False
        else:
            # Static mode: deliver sequentially
            if self.current_index >= len(self.df):
                return False

            row = self.df.iloc[self.current_index]
            self.lines.datetime[0] = bt.date2num(row['datetime'])
            self.lines.open[0] = row['open']
            self.lines.high[0] = row['high']
            self.lines.low[0] = row['low']
            self.lines.close[0] = row['close']
            self.lines.volume[0] = row['volume']
            self.lines.openinterest[0] = row.get('openinterest', 0.0)

            self.current_index += 1
            return True

    def stop(self):
        """Stop the data feed and clean up resources."""
        self.should_stop = True
        if self.realtime_thread and self.realtime_thread.is_alive():
            self.realtime_thread.join(timeout=5)
        _logger.info("Stopped file data feed for %s", self.p.symbol or 'file_data')

    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the file data feed.

        Returns:
            Dictionary with status information
        """
        return {
            'symbol': self.p.symbol or 'file_data',
            'file_path': str(self.p.dataname),
            'total_rows': len(self.df) if self.df is not None else 0,
            'current_index': self.current_index,
            'is_realtime': self.is_realtime,
            'last_delivered': self.last_delivered,
            'queue_size': len(self.data_queue) if self.is_realtime else 0,
            'should_stop': self.should_stop,
            'data_source': 'CSV File'
        }

    @classmethod
    def create_sample_csv(cls, file_path: str, symbol: str = "TEST",
                         days: int = 30, interval_minutes: int = 60):
        """
        Create a sample CSV file for testing purposes.

        Args:
            file_path: Path where to save the CSV file
            symbol: Symbol name for the data
            days: Number of days of data to generate
            interval_minutes: Interval between bars in minutes
        """
        import numpy as np

        # Generate datetime index
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        date_range = pd.date_range(
            start=start_time,
            end=end_time,
            freq=f'{interval_minutes}min'
        )

        # Generate realistic OHLCV data
        np.random.seed(42)  # For reproducible data

        n_bars = len(date_range)
        base_price = 100.0

        # Generate price movements
        returns = np.random.normal(0, 0.02, n_bars)  # 2% volatility
        prices = base_price * np.exp(np.cumsum(returns))

        # Generate OHLCV data
        data = []
        for i, (timestamp, close_price) in enumerate(zip(date_range, prices)):
            # Generate realistic OHLC from close price
            volatility = abs(np.random.normal(0, 0.01))  # Intrabar volatility

            high = close_price * (1 + volatility)
            low = close_price * (1 - volatility)

            # Open is previous close (with some gap)
            if i == 0:
                open_price = close_price
            else:
                gap = np.random.normal(0, 0.005)  # Small gap
                open_price = prices[i-1] * (1 + gap)

            # Ensure OHLC relationships are valid
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)

            # Generate volume
            volume = np.random.randint(1000, 10000)

            data.append({
                'datetime': timestamp,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close_price, 2),
                'volume': volume
            })

        # Create DataFrame and save to CSV
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)

        _logger.info("Created sample CSV file: %s (%d bars)", file_path, len(df))
        return file_path


# Example usage and testing
if __name__ == "__main__":
    import tempfile
    import os

    # Create sample data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_file = f.name

    FileDataFeed.create_sample_csv(sample_file, symbol="TESTBTC", days=7, interval_minutes=60)

    try:
        # Test static mode
        print("Testing static mode...")
        data_feed = FileDataFeed(
            dataname=sample_file,
            symbol="TESTBTC",
            simulate_realtime=False
        )

        print(f"Status: {data_feed.get_status()}")

        # Test real-time simulation mode
        print("\nTesting real-time simulation mode...")

        def on_new_bar(symbol, timestamp, bar_data):
            print(f"New bar for {symbol} at {timestamp}: Close={bar_data['close']}")

        realtime_feed = FileDataFeed(
            dataname=sample_file,
            symbol="TESTBTC",
            simulate_realtime=True,
            realtime_interval=1,  # 1 second for testing
            on_new_bar=on_new_bar
        )

        print(f"Real-time status: {realtime_feed.get_status()}")

        # Let it run for a few seconds
        time.sleep(5)

        realtime_feed.stop()

    finally:
        # Clean up
        os.unlink(sample_file)
        print("Test completed!")