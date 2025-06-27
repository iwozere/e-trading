"""
Yahoo Finance Live Data Feed Module
---------------------------------

This module provides a live data feed for Yahoo Finance using polling.
Since Yahoo Finance doesn't provide WebSocket streams, this implementation
uses periodic polling to simulate real-time updates.

Features:
- Historical data loading via yfinance
- Real-time updates via polling
- Configurable polling intervals
- Error handling and rate limiting
- Backtrader integration

Classes:
- YahooLiveDataFeed: Live data feed for Yahoo Finance
"""

import time
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from src.data.base_live_data_feed import BaseLiveDataFeed
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class YahooLiveDataFeed(BaseLiveDataFeed):
    """
    Live data feed for Yahoo Finance using polling for real-time updates.
    
    Features:
    - Loads historical data via yfinance
    - Real-time updates via periodic polling
    - Configurable polling intervals
    - Error handling and rate limiting
    """
    
    def __init__(self, 
                 symbol: str,
                 interval: str,
                 polling_interval: int = 60,
                 **kwargs):
        """
        Initialize Yahoo Finance live data feed.
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'MSFT')
            interval: Data interval (e.g., '1m', '1h', '1d')
            polling_interval: Seconds between polling attempts
            **kwargs: Additional arguments passed to BaseLiveDataFeed
        """
        self.polling_interval = polling_interval
        self.ticker = None
        self.last_poll_time = None
        
        # Convert interval to yfinance format
        self.yahoo_interval = self._convert_interval(interval)
        
        super().__init__(symbol=symbol, interval=interval, **kwargs)
    
    def _convert_interval(self, interval: str) -> str:
        """
        Convert standard interval format to yfinance format.
        
        Args:
            interval: Standard interval (e.g., '1m', '1h', '1d')
            
        Returns:
            yfinance interval format
        """
        interval_map = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d',
        }
        return interval_map.get(interval, '1d')
    
    def _load_historical_data(self) -> Optional[pd.DataFrame]:
        """
        Load historical data from Yahoo Finance.
        
        Returns:
            DataFrame with historical OHLCV data
        """
        try:
            _logger.info(f"Loading {self.lookback_bars} historical bars for {self.symbol}")
            
            # Create ticker object
            self.ticker = yf.Ticker(self.symbol)
            
            # Calculate period based on lookback_bars and interval
            period = self._calculate_period()
            
            # Get historical data
            df = self.ticker.history(
                period=period,
                interval=self.yahoo_interval,
                prepost=True
            )
            
            if df.empty:
                _logger.warning(f"No historical data found for {self.symbol}")
                return None
            
            # Ensure we have the right number of bars
            if len(df) > self.lookback_bars:
                df = df.tail(self.lookback_bars)
            
            # Rename columns to match expected format
            df.columns = [col.lower() for col in df.columns]
            
            # Select only required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            df = df[required_columns]
            
            _logger.info(f"Loaded {len(df)} historical bars for {self.symbol}")
            return df
            
        except Exception as e:
            _logger.error(f"Error loading historical data for {self.symbol}: {str(e)}")
            return None
    
    def _calculate_period(self) -> str:
        """
        Calculate the appropriate period string for yfinance based on lookback_bars and interval.
        
        Returns:
            Period string for yfinance (e.g., '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        """
        interval_minutes = self._get_interval_minutes()
        total_minutes = self.lookback_bars * interval_minutes
        
        # Convert to days
        total_days = total_minutes / (24 * 60)
        
        if total_days <= 1:
            return '1d'
        elif total_days <= 5:
            return '5d'
        elif total_days <= 30:
            return '1mo'
        elif total_days <= 90:
            return '3mo'
        elif total_days <= 180:
            return '6mo'
        elif total_days <= 365:
            return '1y'
        elif total_days <= 730:
            return '2y'
        elif total_days <= 1825:
            return '5y'
        else:
            return 'max'
    
    def _get_interval_minutes(self) -> int:
        """
        Get the interval duration in minutes.
        
        Returns:
            Interval duration in minutes
        """
        interval_map = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '4h': 240,
            '1d': 1440,
        }
        return interval_map.get(self.interval, 1440)
    
    def _connect_realtime(self) -> bool:
        """
        Connect to Yahoo Finance for real-time data.
        For Yahoo Finance, this just validates the ticker exists.
        
        Returns:
            True if ticker is valid, False otherwise
        """
        try:
            if self.ticker is None:
                self.ticker = yf.Ticker(self.symbol)
            
            # Test if ticker is valid by getting basic info
            info = self.ticker.info
            if 'regularMarketPrice' not in info:
                _logger.error(f"Invalid ticker symbol: {self.symbol}")
                return False
            
            _logger.info(f"Connected to Yahoo Finance for {self.symbol}")
            return True
            
        except Exception as e:
            _logger.error(f"Error connecting to Yahoo Finance for {self.symbol}: {str(e)}")
            return False
    
    def _disconnect_realtime(self):
        """Disconnect from Yahoo Finance."""
        self.ticker = None
        _logger.info(f"Disconnected from Yahoo Finance for {self.symbol}")
    
    def _get_latest_data(self) -> Optional[pd.DataFrame]:
        """
        Get latest data from Yahoo Finance via polling.
        
        Returns:
            DataFrame with latest bar(s), or None if no new data
        """
        try:
            if self.ticker is None:
                return None
            
            # Get recent data (last few bars to ensure we have the latest)
            recent_data = self.ticker.history(
                period='1d',
                interval=self.yahoo_interval,
                prepost=True
            )
            
            if recent_data.empty:
                return None
            
            # Rename columns to match expected format
            recent_data.columns = [col.lower() for col in recent_data.columns]
            
            # Select only required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            recent_data = recent_data[required_columns]
            
            # Check if we have new data
            if self.df is not None and not self.df.empty:
                last_known_time = self.df.index[-1]
                new_data = recent_data[recent_data.index > last_known_time]
                
                if not new_data.empty:
                    _logger.debug(f"Found {len(new_data)} new bars for {self.symbol}")
                    return new_data
                else:
                    return None
            else:
                # First time getting data
                return recent_data.tail(1)
            
        except Exception as e:
            _logger.error(f"Error getting latest data for {self.symbol}: {str(e)}")
            return None
    
    def _get_update_interval(self) -> int:
        """
        Get the update interval in seconds.
        For Yahoo Finance, use the configured polling interval.
        
        Returns:
            Update interval in seconds
        """
        return self.polling_interval
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the Yahoo Finance data feed.
        
        Returns:
            Dictionary with status information
        """
        status = super().get_status()
        status.update({
            'polling_interval': self.polling_interval,
            'yahoo_interval': self.yahoo_interval,
            'last_poll_time': self.last_poll_time,
            'ticker_valid': self.ticker is not None
        })
        return status 