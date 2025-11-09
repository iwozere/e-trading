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
import asyncio
import websockets
from typing import Optional, Dict, Any
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.data.feed.base_live_data_feed import BaseLiveDataFeed
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

        # Initialize Binance client (lazy initialization)
        self.client = None
        self._client_initialized = False

        # WebSocket connection
        self.ws = None
        self.ws_url = None
        self.ws_task = None
        self.loop = None

        # Convert interval to Binance format
        self.binance_interval = self._convert_interval(interval)

        super().__init__(symbol=symbol, interval=interval, **kwargs)

    def _get_client(self):
        """Get or create Binance client with lazy initialization."""
        if not self._client_initialized:
            try:
                # Lazy import to avoid loading binance package at module import time
                from binance.client import Client
                self.client = Client(self.api_key, self.api_secret, testnet=self.testnet)
                self._client_initialized = True
            except Exception as e:
                _logger.warning("Failed to initialize Binance client: %s", e)
                self.client = None
        return self.client

    def _convert_interval(self, interval: str) -> str:
        """
        Convert standard interval format to Binance format.

        Args:
            interval: Standard interval (e.g., '1m', '1h', '1d')

        Returns:
            Binance interval format
        """
        # Use string constants instead of Client constants to avoid early import
        interval_map = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d',
        }
        return interval_map.get(interval, '1m')

    # Note: _load_historical_data() is now inherited from BaseLiveDataFeed
    # which uses DataManager for historical data loading. This ensures
    # consistent data access and caching across all live feeds.
    def _load_historical_data_old(self) -> Optional[pd.DataFrame]:
        """
        Load historical data from Binance REST API with proper pagination.

        Returns:
            DataFrame with historical OHLCV data
        """
        try:
            _logger.info("Loading %d historical bars for %s", self.lookback_bars, self.symbol)

            per_call_limit = 1000
            remaining = int(self.lookback_bars)
            interval_minutes = self._get_interval_minutes()

            # Use UTC time
            end_dt = datetime.now(timezone.utc).replace(tzinfo=timezone.utc)
            start_dt = end_dt - timedelta(minutes=remaining * interval_minutes)

            all_rows = []
            start_str = int(start_dt.timestamp() * 1000)  # ms

            while remaining > 0:
                batch_limit = min(per_call_limit, remaining)
                client = self._get_client()
                if client is None:
                    _logger.error("Binance client not available")
                    return None
                klines = client.get_historical_klines(
                    symbol=self.symbol,
                    interval=self.binance_interval,
                    start_str=start_str,
                    limit=batch_limit,
                )

                if not klines:
                    _logger.warning("No historical data returned for %s", self.symbol)
                    break

                all_rows.extend(klines)
                remaining -= len(klines)

                # advance start_str using last kline close time + 1ms to avoid duplicates
                last_close_ms = klines[-1][6]  # close_time
                start_str = last_close_ms + 1

                if len(klines) < batch_limit:  # nothing more to fetch
                    break

            if not all_rows:
                return None

            # Convert to DataFrame
            df = pd.DataFrame(all_rows, columns=[
                'open_time','open','high','low','close','volume',
                'close_time','quote_asset_volume','number_of_trades',
                'taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore'
            ])

            # Use open_time for index (consistent with WS) OR switch to close_time consciously
            df['datetime'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
            df.set_index('datetime', inplace=True)

            # Convert price columns to float
            for col in ['open','high','low','close','volume']:
                df[col] = df[col].astype(float)

            # Select only required columns
            df = df[['open','high','low','close','volume']]

            # Validate data quality
            from src.data.utils.validation import validate_ohlcv_data, get_data_quality_score
            is_valid, errors = validate_ohlcv_data(df)
            if not is_valid:
                _logger.warning("Historical data validation failed for %s: %s", self.symbol, errors)
                quality_score = get_data_quality_score(df)
                _logger.info("Historical data quality score: %.2f", quality_score['quality_score'])

            _logger.info("Loaded %d historical bars for %s", len(df), self.symbol)
            return df

        except Exception as e:
            # Check if it's a Binance API exception (lazy import)
            try:
                from binance.exceptions import BinanceAPIException
                if isinstance(e, BinanceAPIException):
                    _logger.exception("Binance API error loading historical data: %s", e)
                else:
                    _logger.exception("Unexpected error loading historical data for %s: %s", self.symbol, e)
            except ImportError:
                _logger.exception("Error loading historical data for %s: %s", self.symbol, e)
            return None
        except Exception as e:
            _logger.exception("Unexpected error loading historical data for %s: %s", self.symbol, e)
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

            # Start async event loop in a separate thread
            import threading
            self.loop = asyncio.new_event_loop()
            self.ws_thread = threading.Thread(
                target=self._run_websocket_loop,
                daemon=True
            )
            self.ws_thread.start()

            # Wait briefly but also check repeatedly for connection
            for _ in range(20):
                if self.is_connected:
                    return True
                time.sleep(0.1)
            return False

        except Exception as e:
            _logger.exception("Error connecting to Binance WebSocket: %s", e)
            return False

    def _run_websocket_loop(self):
        """Run the WebSocket event loop in a separate thread."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._websocket_handler())

    async def _websocket_handler(self):
        """Handle WebSocket connection and messages."""
        try:
            async with websockets.connect(
                self.ws_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            ) as websocket:
                self.ws = websocket
                self.is_connected = True
                _logger.info("Binance WebSocket connected for %s", self.symbol)

                async for message in websocket:
                    await self._on_ws_message(message)

        except websockets.exceptions.ConnectionClosed:
            _logger.warning("Binance WebSocket disconnected for %s", self.symbol)
            self.is_connected = False
        except Exception as e:
            _logger.exception("Binance WebSocket error for %s: %s", self.symbol, e)
            self.is_connected = False

    def _disconnect_realtime(self):
        """Disconnect from Binance WebSocket."""
        if self.loop and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.ws:
            self.ws = None
        self.is_connected = False

    async def _on_ws_message(self, message):
        """WebSocket message received."""
        try:
            data = json.loads(message)

            # Extract kline data
            if 'k' in data:
                kline = data['k']

                # Check if this is a completed kline
                if kline['x']:  # kline is closed
                    # Use UTC time for consistency
                    ts = pd.to_datetime(kline['t'], unit='ms', utc=True)
                    new_data = pd.DataFrame([{
                        'open': float(kline['o']),
                        'high': float(kline['h']),
                        'low': float(kline['l']),
                        'close': float(kline['c']),
                        'volume': float(kline['v'])
                    }], index=[ts])

                    # Validate new data before processing
                    from src.data.utils.validation import validate_ohlcv_data
                    is_valid, errors = validate_ohlcv_data(new_data)
                    if not is_valid:
                        _logger.warning("WebSocket data validation failed: %s", errors)
                    else:
                        self._process_new_data(new_data)

        except json.JSONDecodeError as e:
            _logger.exception("Error decoding WebSocket message: %s", e)
        except Exception as e:
            _logger.exception("Error processing WebSocket message: %s", e)

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
            'testnet': self.testnet,
            'data_source': 'Binance'  # Override the base class data_source
        })
        return status
