"""
IBKR Live Data Feed Module
-------------------------

This module provides a live data feed for Interactive Brokers using ib_insync.
It connects to IBKR TWS or IB Gateway for real-time market data.

Features:
- Historical data loading via IBKR API
- Real-time updates via IBKR data streams
- Automatic reconnection on connection loss
- Error handling and rate limiting
- Backtrader integration

Classes:
- IBKRLiveDataFeed: Live data feed for Interactive Brokers
"""

import time
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

import pandas as pd
from ib_insync import *

from src.data.base_live_data_feed import BaseLiveDataFeed
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class IBKRLiveDataFeed(BaseLiveDataFeed):
    """
    Live data feed for Interactive Brokers using ib_insync.
    
    Features:
    - Loads historical data via IBKR API
    - Real-time updates via IBKR data streams
    - Automatic reconnection on connection loss
    - Error handling and rate limiting
    """
    
    def __init__(self, 
                 symbol: str,
                 interval: str,
                 host: str = '127.0.0.1',
                 port: int = 7497,
                 client_id: int = 1,
                 **kwargs):
        """
        Initialize IBKR live data feed.
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'SPY')
            interval: Data interval (e.g., '1m', '1h', '1d')
            host: IBKR TWS/Gateway host
            port: IBKR TWS/Gateway port (7497 for TWS, 4001 for Gateway)
            client_id: IBKR client ID
            **kwargs: Additional arguments passed to BaseLiveDataFeed
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        
        # IBKR connection
        self.ib = IB()
        
        # Contract and data subscription
        self.contract = None
        self.data_subscription = None
        
        # Convert interval to IBKR format
        self.ibkr_interval = self._convert_interval(interval)
        
        super().__init__(symbol=symbol, interval=interval, **kwargs)
    
    def _convert_interval(self, interval: str) -> str:
        """
        Convert standard interval format to IBKR format.
        
        Args:
            interval: Standard interval (e.g., '1m', '1h', '1d')
            
        Returns:
            IBKR interval format
        """
        interval_map = {
            '1m': '1 min',
            '5m': '5 mins',
            '15m': '15 mins',
            '30m': '30 mins',
            '1h': '1 hour',
            '4h': '4 hours',
            '1d': '1 day',
        }
        return interval_map.get(interval, '1 min')
    
    def _create_contract(self) -> Optional[Contract]:
        """
        Create IBKR contract for the symbol.
        
        Returns:
            IBKR Contract object, or None if creation fails
        """
        try:
            # Try to create a stock contract first
            contract = Stock(self.symbol, 'SMART', 'USD')
            
            # Request contract details to validate
            self.ib.reqContractDetails(contract)
            self.ib.sleep(1)  # Wait for response
            
            # Check if we got contract details
            if not self.ib.reqContractDetails(contract):
                # Try forex contract
                contract = Forex(self.symbol)
                self.ib.reqContractDetails(contract)
                self.ib.sleep(1)
            
            return contract
            
        except Exception as e:
            _logger.error(f"Error creating contract for {self.symbol}: {str(e)}")
            return None
    
    def _load_historical_data(self) -> Optional[pd.DataFrame]:
        """
        Load historical data from IBKR.
        
        Returns:
            DataFrame with historical OHLCV data
        """
        try:
            _logger.info(f"Loading {self.lookback_bars} historical bars for {self.symbol}")
            
            # Create contract
            self.contract = self._create_contract()
            if self.contract is None:
                return None
            
            # Calculate duration based on lookback_bars and interval
            duration = self._calculate_duration()
            
            # Request historical data
            bars = self.ib.reqHistoricalData(
                self.contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=self.ibkr_interval,
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1,
                keepUpToDate=False,
                chartOptions=[]
            )
            
            if not bars:
                _logger.warning(f"No historical data found for {self.symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame([{
                'datetime': bar.date,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume
            } for bar in bars])
            
            # Set datetime as index
            df.set_index('datetime', inplace=True)
            
            # Ensure we have the right number of bars
            if len(df) > self.lookback_bars:
                df = df.tail(self.lookback_bars)
            
            _logger.info(f"Loaded {len(df)} historical bars for {self.symbol}")
            return df
            
        except Exception as e:
            _logger.error(f"Error loading historical data for {self.symbol}: {str(e)}")
            return None
    
    def _calculate_duration(self) -> str:
        """
        Calculate the appropriate duration string for IBKR based on lookback_bars and interval.
        
        Returns:
            Duration string for IBKR (e.g., '1 D', '1 W', '1 M', '3 M', '6 M', '1 Y')
        """
        interval_minutes = self._get_interval_minutes()
        total_minutes = self.lookback_bars * interval_minutes
        
        # Convert to days
        total_days = total_minutes / (24 * 60)
        
        if total_days <= 1:
            return '1 D'
        elif total_days <= 7:
            return '1 W'
        elif total_days <= 30:
            return '1 M'
        elif total_days <= 90:
            return '3 M'
        elif total_days <= 180:
            return '6 M'
        else:
            return '1 Y'
    
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
        return interval_map.get(self.interval, 1)
    
    def _connect_realtime(self) -> bool:
        """
        Connect to IBKR TWS/Gateway for real-time data.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Connect to IBKR
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            
            # Wait for connection
            self.ib.sleep(2)
            
            if not self.ib.isConnected():
                _logger.error(f"Failed to connect to IBKR at {self.host}:{self.port}")
                return False
            
            _logger.info(f"Connected to IBKR for {self.symbol}")
            
            # Subscribe to real-time data
            self._subscribe_realtime_data()
            
            return True
            
        except Exception as e:
            _logger.error(f"Error connecting to IBKR: {str(e)}")
            return False
    
    def _subscribe_realtime_data(self):
        """Subscribe to real-time data for the contract."""
        try:
            if self.contract is None:
                self.contract = self._create_contract()
            
            if self.contract is not None:
                # Request real-time bars
                self.data_subscription = self.ib.reqRealTimeBars(
                    self.contract,
                    barSize=5,  # 5-second bars
                    whatToShow='TRADES',
                    useRTH=True
                )
                
                # Set up callback for real-time data
                self.ib.barUpdateEvent += self._on_bar_update
                
                _logger.info(f"Subscribed to real-time data for {self.symbol}")
            
        except Exception as e:
            _logger.error(f"Error subscribing to real-time data: {str(e)}")
    
    def _disconnect_realtime(self):
        """Disconnect from IBKR."""
        try:
            if self.data_subscription:
                self.ib.cancelRealTimeBars(self.data_subscription)
                self.data_subscription = None
            
            if self.ib.isConnected():
                self.ib.disconnect()
            
            _logger.info(f"Disconnected from IBKR for {self.symbol}")
            
        except Exception as e:
            _logger.error(f"Error disconnecting from IBKR: {str(e)}")
    
    def _on_bar_update(self, bars):
        """Callback for real-time bar updates."""
        try:
            if not bars:
                return
            
            # Get the latest bar
            latest_bar = bars[-1]
            
            # Create DataFrame with the new bar
            new_data = pd.DataFrame([{
                'open': latest_bar.open,
                'high': latest_bar.high,
                'low': latest_bar.low,
                'close': latest_bar.close,
                'volume': latest_bar.volume
            }], index=[latest_bar.time])
            
            self._process_new_data(new_data)
            
        except Exception as e:
            _logger.error(f"Error processing real-time bar update: {str(e)}")
    
    def _get_latest_data(self) -> Optional[pd.DataFrame]:
        """
        Get latest data from IBKR.
        For real-time feeds, this is handled by the bar update callback.
        
        Returns:
            None (data is processed via callbacks)
        """
        # For real-time feeds, data is processed via _on_bar_update
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the IBKR data feed.
        
        Returns:
            Dictionary with status information
        """
        status = super().get_status()
        status.update({
            'ibkr_connected': self.ib.isConnected() if self.ib else False,
            'host': self.host,
            'port': self.port,
            'client_id': self.client_id,
            'ibkr_interval': self.ibkr_interval,
            'contract_valid': self.contract is not None,
            'data_subscribed': self.data_subscription is not None
        })
        return status
