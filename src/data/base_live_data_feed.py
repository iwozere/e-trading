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
from abc import ABC, abstractmethod
from typing import Optional, Callable, Dict, Any
from datetime import datetime, timedelta

import backtrader as bt
import pandas as pd
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class BaseLiveDataFeed(bt.feeds.PandasData):
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
                 **kwargs):
        """
        Initialize the live data feed.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT', 'AAPL')
            interval: Data interval (e.g., '1m', '1h', '1d')
            lookback_bars: Number of historical bars to load initially
            retry_interval: Seconds to wait before retrying on connection failure
            on_new_bar: Optional callback function when new data arrives
            **kwargs: Additional arguments passed to PandasData
        """
        self.symbol = symbol
        self.interval = interval
        self.lookback_bars = lookback_bars
        self.retry_interval = retry_interval
        self.on_new_bar = on_new_bar
        
        # Initialize data storage
        self.df = None
        self.last_update = None
        self.is_connected = False
        self.should_stop = False
        
        # Load historical data first
        _logger.info(f"Loading {lookback_bars} historical bars for {symbol} {interval}")
        historical_data = self._load_historical_data()
        
        if historical_data is None or historical_data.empty:
            raise ValueError(f"Failed to load historical data for {symbol}")
        
        # Initialize PandasData with historical data
        super().__init__(dataname=historical_data, **kwargs)
        
        # Start real-time updates
        self._start_realtime_updates()
    
    @abstractmethod
    def _load_historical_data(self) -> Optional[pd.DataFrame]:
        """
        Load historical data from the data source.
        
        Returns:
            DataFrame with columns: datetime, open, high, low, close, volume
        """
        pass
    
    @abstractmethod
    def _connect_realtime(self) -> bool:
        """
        Connect to real-time data source.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def _disconnect_realtime(self):
        """Disconnect from real-time data source."""
        pass
    
    @abstractmethod
    def _get_latest_data(self) -> Optional[pd.DataFrame]:
        """
        Get the latest data from the source.
        
        Returns:
            DataFrame with latest bar(s), or None if no new data
        """
        pass
    
    def _start_realtime_updates(self):
        """Start the real-time update thread."""
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        _logger.info(f"Started real-time updates for {self.symbol}")
    
    def _update_loop(self):
        """Main loop for real-time updates."""
        while not self.should_stop:
            try:
                if not self.is_connected:
                    _logger.info(f"Connecting to real-time data for {self.symbol}")
                    if self._connect_realtime():
                        self.is_connected = True
                        _logger.info(f"Connected to real-time data for {self.symbol}")
                    else:
                        _logger.warning(f"Failed to connect to real-time data for {self.symbol}, retrying in {self.retry_interval}s")
                        time.sleep(self.retry_interval)
                        continue
                
                # Get latest data
                latest_data = self._get_latest_data()
                if latest_data is not None and not latest_data.empty:
                    self._process_new_data(latest_data)
                
                # Sleep before next update
                time.sleep(self._get_update_interval())
                
            except Exception as e:
                _logger.error(f"Error in update loop for {self.symbol}: {str(e)}")
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
                    self.df = new_data
                else:
                    self.df = pd.concat([self.df, new_data])
                
                # Update Backtrader lines
                latest = self.df.iloc[-1]
                self.lines.datetime[0] = bt.date2num(self.df.index[-1])
                self.lines.open[0] = latest["open"]
                self.lines.high[0] = latest["high"]
                self.lines.low[0] = latest["low"]
                self.lines.close[0] = latest["close"]
                self.lines.volume[0] = latest["volume"]
                self.lines.openinterest[0] = 0
                
                self.last_update = datetime.now()
                
                # Call callback if provided
                if self.on_new_bar:
                    try:
                        self.on_new_bar(self.symbol, self.df.index[-1], latest.to_dict())
                    except Exception as e:
                        _logger.error(f"Error in on_new_bar callback: {str(e)}")
                
                _logger.debug(f"Updated {self.symbol} with new bar at {self.df.index[-1]}")
        
        except Exception as e:
            _logger.error(f"Error processing new data for {self.symbol}: {str(e)}")
    
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
        _logger.info(f"Stopping real-time updates for {self.symbol}")
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
            'should_stop': self.should_stop
        } 