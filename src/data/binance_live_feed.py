"""
Binance Live Data Feed Module
----------------------------

This module provides a live data feed for Binance using WebSocket connections.
It loads historical data via REST API and provides real-time updates via WebSocket.

Features:
- Historical data loading via Binance REST API
- Real-time updates via WebSocket streams
- Automatic reconnection on connection loss
- Error handling and rate limiting
- Backtrader integration

Classes:
- BinanceLiveDataFeed: Live data feed for Binance
"""

import time
import json
import websocket
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException

from src.data.base_live_data_feed import BaseLiveDataFeed
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class BinanceLiveDataFeed(BaseLiveDataFeed):
    """
    Live data feed for Binance using WebSocket for real-time updates.
    
    Features:
    - Loads historical data via REST API
    - Real-time updates via WebSocket kline streams
    - Automatic reconnection on connection loss
    - Error handling and rate limiting
    """
    
    def __init__(self, 
                 symbol: str,
                 interval: str,
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 testnet: bool = False,
                 **kwargs):
        """
        Initialize Binance live data feed.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Data interval (e.g., '1m', '1h', '1d')
            api_key: Binance API key (optional for public data)
            api_secret: Binance API secret (optional for public data)
            testnet: Use Binance testnet
            **kwargs: Additional arguments passed to BaseLiveDataFeed
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # Initialize Binance client
        self.client = Client(api_key, api_secret, testnet=testnet)
        
        # WebSocket connection
        self.ws = None
        self.ws_url = None
        
        # Convert interval to Binance format
        self.binance_interval = self._convert_interval(interval)
        
        super().__init__(symbol=symbol, interval=interval, **kwargs)
    
    def _convert_interval(self, interval: str) -> str:
        """
        Convert standard interval format to Binance format.
        
        Args:
            interval: Standard interval (e.g., '1m', '1h', '1d')
            
        Returns:
            Binance interval format
        """
        interval_map = {
            '1m': Client.KLINE_INTERVAL_1MINUTE,
            '5m': Client.KLINE_INTERVAL_5MINUTE,
            '15m': Client.KLINE_INTERVAL_15MINUTE,
            '30m': Client.KLINE_INTERVAL_30MINUTE,
            '1h': Client.KLINE_INTERVAL_1HOUR,
            '4h': Client.KLINE_INTERVAL_4HOUR,
            '1d': Client.KLINE_INTERVAL_1DAY,
        }
        return interval_map.get(interval, Client.KLINE_INTERVAL_1MINUTE)
    
    def _load_historical_data(self) -> Optional[pd.DataFrame]:
        """
        Load historical data from Binance REST API.
        
        Returns:
            DataFrame with historical OHLCV data
        """
        try:
            _logger.info(f"Loading {self.lookback_bars} historical bars for {self.symbol}")
            
            # Calculate start time based on lookback_bars
            # Approximate: each bar represents the interval duration
            interval_minutes = self._get_interval_minutes()
            total_minutes = self.lookback_bars * interval_minutes
            start_time = datetime.now() - timedelta(minutes=total_minutes)
            
            # Get historical klines
            klines = self.client.get_historical_klines(
                symbol=self.symbol,
                interval=self.binance_interval,
                start_str=start_time.strftime('%Y-%m-%d %H:%M:%S'),
                limit=self.lookback_bars
            )
            
            if not klines:
                _logger.warning(f"No historical data found for {self.symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            
            # Convert price columns to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            # Select only required columns
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            _logger.info(f"Loaded {len(df)} historical bars for {self.symbol}")
            return df
            
        except BinanceAPIException as e:
            _logger.error(f"Binance API error loading historical data: {str(e)}")
            return None
        except Exception as e:
            _logger.error(f"Error loading historical data for {self.symbol}: {str(e)}")
            return None
    
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
        Connect to Binance WebSocket for real-time data.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Create WebSocket URL
            stream_name = f"{self.symbol.lower()}@kline_{self.binance_interval}"
            self.ws_url = f"wss://stream.binance.com:9443/ws/{stream_name}"
            
            # Create WebSocket connection
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_message=self._on_ws_message,
                on_error=self._on_ws_error,
                on_close=self._on_ws_close,
                on_open=self._on_ws_open
            )
            
            # Start WebSocket in a separate thread
            import threading
            self.ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
            self.ws_thread.start()
            
            # Wait for connection to establish
            time.sleep(2)
            
            return self.ws.sock is not None and self.ws.sock.connected
            
        except Exception as e:
            _logger.error(f"Error connecting to Binance WebSocket: {str(e)}")
            return False
    
    def _disconnect_realtime(self):
        """Disconnect from Binance WebSocket."""
        if self.ws:
            self.ws.close()
            self.ws = None
    
    def _on_ws_open(self, ws):
        """WebSocket connection opened."""
        _logger.info(f"Binance WebSocket connected for {self.symbol}")
    
    def _on_ws_close(self, ws, close_status_code, close_msg):
        """WebSocket connection closed."""
        _logger.warning(f"Binance WebSocket disconnected for {self.symbol}: {close_msg}")
        self.is_connected = False
    
    def _on_ws_error(self, ws, error):
        """WebSocket error occurred."""
        _logger.error(f"Binance WebSocket error for {self.symbol}: {str(error)}")
        self.is_connected = False
    
    def _on_ws_message(self, ws, message):
        """WebSocket message received."""
        try:
            data = json.loads(message)
            
            # Extract kline data
            if 'k' in data:
                kline = data['k']
                
                # Check if this is a completed kline
                if kline['x']:  # kline is closed
                    new_data = pd.DataFrame([{
                        'open': float(kline['o']),
                        'high': float(kline['h']),
                        'low': float(kline['l']),
                        'close': float(kline['c']),
                        'volume': float(kline['v'])
                    }], index=[pd.to_datetime(kline['t'], unit='ms')])
                    
                    self._process_new_data(new_data)
            
        except json.JSONDecodeError as e:
            _logger.error(f"Error decoding WebSocket message: {str(e)}")
        except Exception as e:
            _logger.error(f"Error processing WebSocket message: {str(e)}")
    
    def _get_latest_data(self) -> Optional[pd.DataFrame]:
        """
        Get latest data from Binance.
        For WebSocket feeds, this is handled by the WebSocket callback.
        
        Returns:
            None (data is processed via WebSocket callbacks)
        """
        # For WebSocket feeds, data is processed via _on_ws_message
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the Binance data feed.
        
        Returns:
            Dictionary with status information
        """
        status = super().get_status()
        status.update({
            'ws_connected': self.ws is not None and self.ws.sock is not None and self.ws.sock.connected,
            'ws_url': self.ws_url,
            'binance_interval': self.binance_interval,
            'testnet': self.testnet
        })
        return status
