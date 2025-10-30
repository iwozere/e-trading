"""
Base Live Data Feed Module
-------------------------

This module provides the base class for live data feeds that integrate with Backtrader.
All specific data feed implementations (Binance, Yahoo, IBKR) inherit from this class.

Main Features:
- Common interface for all live data feeds
- Historical data loading with configurable lookback
- Real-time data updates via WebSocket/polling
- Automatic error handling and reconnection
- Backtrader integration

Classes:
- BaseLiveDataFeed: Abstract base class for live data feeds
"""

import time
import threading
from abc import abstractmethod
from typing import Optional, Callable, Dict, Any
from datetime import datetime, timedelta

import backtrader as bt
import pandas as pd
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

class BaseLiveDataFeed(bt.feed.DataBase):
    """
    Base class for live data feeds that provide real-time market data to Backtrader.

    This class handles:
    - Historical data loading
    - Real-time data updates
    - Error handling and reconnection
    - Backtrader integration
    """

    params = (
        ("datetime", None),
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", None),
    )

    def __init__(self,
                 symbol: str,
                 interval: str,
                 lookback_bars: int = 1000,
                 retry_interval: int = 60,
                 on_new_bar: Optional[Callable] = None,
                 data_manager: Optional[Any] = None,
                 **kwargs):
        """
        Initialize the live data feed.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT', 'AAPL')
            interval: Data interval (e.g., '1m', '1h', '1d')
            lookback_bars: Number of historical bars to load initially
            retry_interval: Seconds to wait before retrying on connection failure
            on_new_bar: Optional callback function when new data arrives
            data_manager: DataManager instance for historical data loading
            **kwargs: Additional arguments passed to PandasData
        """
        # Call parent constructor first
        super().__init__(**kwargs)
        self.symbol = symbol
        self.interval = interval
        self.lookback_bars = lookback_bars
        self.retry_interval = retry_interval
        self.on_new_bar = on_new_bar

        # Initialize DataManager if not provided
        if data_manager is None:
            # Import DataManager dynamically to avoid circular imports
            from src.data.data_manager import DataManager
            self.data_manager = DataManager()
        else:
            self.data_manager = data_manager

        # Data storage is now handled by the PandasData composition pattern
        self.last_update = None
        self.is_connected = False
        self.should_stop = False

        # Load historical data first
        _logger.info("Loading %d historical bars for %s %s", lookback_bars, symbol, interval)
        historical_data = self._load_historical_data()

        if historical_data is None or historical_data.empty:
            raise ValueError("Failed to load historical data for %s", symbol)

        # Prepare data for Backtrader PandasData
        # Reset index to make datetime a column instead of index
        df_copy = historical_data.copy(deep=True)
        df_copy = df_copy.reset_index()
        df_copy = df_copy.rename(columns={'datetime': 'timestamp'})

        # Ensure timestamp column is datetime
        df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])

        # Ensure the index is timezone-naive for Backtrader compatibility
        if df_copy['timestamp'].dt.tz is not None:
            df_copy['timestamp'] = df_copy['timestamp'].dt.tz_localize(None)

        # Filter out parameters that are not valid for PandasData
        pandas_kwargs = {}
        for key, value in kwargs.items():
            if key not in ['api_key', 'api_secret', 'testnet', 'data_manager']:
                pandas_kwargs[key] = value

        # Create PandasData instance with the prepared data
        self.pandas_data = bt.feeds.PandasData(
            dataname=df_copy,
            datetime=0,  # 0 indicates datetime is in column 0 (timestamp)
            open=1,      # open is now column 1
            high=2,      # high is now column 2
            low=3,       # low is now column 3
            close=4,     # close is now column 4
            volume=5,    # volume is now column 5
            openinterest=None,
            name=symbol,
            **pandas_kwargs
        )

        # Delegate to PandasData for Backtrader compatibility
        self.p = self.pandas_data.p

        # Start real-time updates
        self._start_realtime_updates()

    def __getattr__(self, name):
        """Delegate attribute access to PandasData instance for Backtrader compatibility."""
        return getattr(self.pandas_data, name)

    @property
    def df(self):
        """Get the DataFrame for backward compatibility with tests."""
        return self.pandas_data.p.dataname

    def _load_historical_data(self) -> Optional[pd.DataFrame]:
        """
        Load historical data using DataManager.

        This method provides a default implementation that uses DataManager
        to load historical data. Subclasses can override this method if they
        need custom historical data loading logic.

        Returns:
            DataFrame with columns: datetime, open, high, low, close, volume
        """
        try:
            # Calculate date range based on lookback_bars and interval
            end_date = datetime.now()
            start_date = self._calculate_start_date(end_date)

            _logger.info("Loading historical data for %s %s from %s to %s",
                        self.symbol, self.interval, start_date, end_date)

            # Use DataManager to get historical data
            df = self.data_manager.get_ohlcv(
                symbol=self.symbol,
                timeframe=self.interval,
                start_date=start_date,
                end_date=end_date
            )

            if df is not None and not df.empty:
                _logger.info("Loaded %d historical bars for %s %s", len(df), self.symbol, self.interval)
                return df
            else:
                _logger.warning("No historical data returned for %s %s", self.symbol, self.interval)
                return None

        except Exception as e:
            _logger.error("Error loading historical data for %s %s: %s", self.symbol, self.interval, e)
            return None

    def _calculate_start_date(self, end_date: datetime) -> datetime:
        """
        Calculate start date based on lookback_bars and interval.

        Args:
            end_date: End date for the data range

        Returns:
            Calculated start date
        """
        # Convert interval to minutes
        interval_minutes = self._parse_interval_to_minutes(self.interval)
        if interval_minutes == 0:
            # Default to 1 day if interval is unknown
            interval_minutes = 1440

        # Calculate total minutes for lookback_bars
        total_minutes = self.lookback_bars * interval_minutes

        # Calculate start date
        start_date = end_date - timedelta(minutes=total_minutes)

        return start_date

    def _parse_interval_to_minutes(self, interval: str) -> int:
        """
        Parse interval string to minutes.

        Args:
            interval: Interval string (e.g., '1m', '1h', '1d')

        Returns:
            Interval in minutes
        """
        interval = interval.lower()

        if interval.endswith('m'):
            return int(interval[:-1])
        elif interval.endswith('h'):
            return int(interval[:-1]) * 60
        elif interval.endswith('d'):
            return int(interval[:-1]) * 24 * 60
        elif interval.endswith('w'):
            return int(interval[:-1]) * 7 * 24 * 60
        else:
            _logger.warning("Unknown interval format: %s", interval)
            return 0

    @abstractmethod
    def _connect_realtime(self) -> bool:
        """
        Connect to real-time data source.

        Returns:
            True if connection successful, False otherwise
        """

    @abstractmethod
    def _disconnect_realtime(self):
        """Disconnect from real-time data source."""

    @abstractmethod
    def _get_latest_data(self) -> Optional[pd.DataFrame]:
        """
        Get the latest data from the source.

        Returns:
            DataFrame with latest bar(s), or None if no new data
        """

    def _start_realtime_updates(self):
        """Start the real-time update thread."""
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        _logger.info("Started real-time updates for %s", self.symbol)

    def _update_loop(self):
        """Main loop for real-time updates."""
        while not self.should_stop:
            try:
                if not self.is_connected:
                    _logger.info("Connecting to real-time data for %s", self.symbol)
                    if self._connect_realtime():
                        self.is_connected = True
                        _logger.info("Connected to real-time data for %s", self.symbol)
                    else:
                        _logger.warning("Failed to connect to real-time data for %s, retrying in %d seconds", self.symbol, self.retry_interval)
                        time.sleep(self.retry_interval)
                        continue

                # Get latest data
                latest_data = self._get_latest_data()
                if latest_data is not None and not latest_data.empty:
                    self._process_new_data(latest_data)

                # Sleep before next update
                time.sleep(self._get_update_interval())

            except Exception as e:
                _logger.exception("Error in update loop for %s: %s")
                self.is_connected = False
                time.sleep(self.retry_interval)

    def _process_new_data(self, new_data: pd.DataFrame):
        """
        Process new data and update Backtrader lines.

        Args:
            new_data: DataFrame with new bar(s)
        """
        try:
            # Check if we have new data
            if self.df is None or new_data.index[-1] not in self.df.index:
                # Add new data to DataFrame
                if self.df is None:
                    # Create new pandas_data with the new data
                    self.pandas_data = bt.feeds.PandasData(
                        dataname=new_data,
                        datetime=0,
                        open=1,
                        high=2,
                        low=3,
                        close=4,
                        volume=5,
                        openinterest=None,
                        name=self.symbol
                    )
                    self.p = self.pandas_data.p
                else:
                    # Concatenate new data with existing data
                    combined_data = pd.concat([self.df, new_data])
                    # Update the pandas_data with combined data
                    self.pandas_data = bt.feeds.PandasData(
                        dataname=combined_data,
                        datetime=0,
                        open=1,
                        high=2,
                        low=3,
                        close=4,
                        volume=5,
                        openinterest=None,
                        name=self.symbol
                    )
                    self.p = self.pandas_data.p

                # Update Backtrader lines
                latest = self.df.iloc[-1]
                try:
                    # Ensure lines are properly initialized before accessing
                    if hasattr(self.lines, 'datetime') and len(self.lines.datetime) > 0:
                        self.lines.datetime[0] = bt.date2num(self.df.index[-1])
                        self.lines.open[0] = latest["open"]
                        self.lines.high[0] = latest["high"]
                        self.lines.low[0] = latest["low"]
                        self.lines.close[0] = latest["close"]
                        self.lines.volume[0] = latest["volume"]
                        self.lines.openinterest[0] = 0
                    else:
                        _logger.warning("Lines not properly initialized for %s, skipping line update", self.symbol)
                except (IndexError, AttributeError) as e:
                    _logger.warning("Failed to update lines for %s: %s", self.symbol, str(e))

                self.last_update = datetime.now()

                # Call callback if provided
                if self.on_new_bar:
                    try:
                        self.on_new_bar(self.symbol, self.df.index[-1], latest.to_dict())
                    except Exception as e:
                        _logger.exception("Error in on_new_bar callback: %s")

                _logger.debug("Updated %s with new bar at %s", self.symbol, self.df.index[-1])

        except Exception as e:
            _logger.exception("Error processing new data for %s: %s")

    def _get_update_interval(self) -> int:
        """
        Get the update interval in seconds based on the data interval.

        Returns:
            Update interval in seconds
        """
        # Default to 1 minute for most intervals
        interval_map = {
            '1m': 60,
            '5m': 60,
            '15m': 60,
            '30m': 60,
            '1h': 60,
            '4h': 300,  # 5 minutes
            '1d': 3600,  # 1 hour
        }
        return interval_map.get(self.interval, 60)

    def _load(self):
        """
        Backtrader's _load method - called when Backtrader needs more data.
        For live feeds, this is typically not used as data is pushed via real-time updates.
        """
        # For live feeds, data is pushed via real-time updates
        # This method is kept for compatibility but should not be called
        return None

    def stop(self):
        """Stop the real-time updates and disconnect."""
        _logger.info("Stopping real-time updates for %s", self.symbol)
        self.should_stop = True
        self._disconnect_realtime()
        if hasattr(self, 'update_thread'):
            self.update_thread.join(timeout=5)

    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the data feed.

        Returns:
            Dictionary with status information
        """
        return {
            'symbol': self.symbol,
            'interval': self.interval,
            'is_connected': self.is_connected,
            'last_update': self.last_update,
            'data_points': len(self.df) if self.df is not None else 0,
            'should_stop': self.should_stop,
            'data_source': self.__class__.__name__.replace('LiveDataFeed', '').lower()
        }
