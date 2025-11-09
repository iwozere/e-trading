"""
CoinGecko Live Data Feed Module
------------------------------

This module provides a live data feed for CoinGecko using polling-based updates.
Since CoinGecko doesn't provide WebSocket API, this implementation uses periodic
REST API calls to simulate real-time data updates.

Features:
- Historical data loading via CoinGecko REST API
- Real-time updates via periodic polling
- Automatic reconnection on connection loss
- Error handling and rate limiting
- Backtrader integration

Classes:
- CoinGeckoLiveDataFeed: Live data feed for CoinGecko
"""

import time
import requests
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

import pandas as pd

from src.data.feed.base_live_data_feed import BaseLiveDataFeed
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class CoinGeckoLiveDataFeed(BaseLiveDataFeed):
    """
    Live data feed for CoinGecko using polling for real-time updates.

    Features:
    - Loads historical data via REST API
    - Real-time updates via periodic polling (since no WebSocket available)
    - Automatic reconnection on connection loss
    - Error handling and rate limiting (50 calls/minute)
    - Backtrader integration

    Note: CoinGecko doesn't provide WebSocket API, so this implementation
    uses polling to simulate real-time updates. The polling interval is
    adjusted to respect CoinGecko's rate limits.
    """

    def __init__(self,
                 symbol: str,
                 interval: str,
                 polling_interval: int = 60,  # Poll every 60 seconds to respect rate limits
                 **kwargs):
        """
        Initialize CoinGecko live data feed.

        Args:
            symbol: Trading symbol (e.g., 'bitcoin', 'ethereum')
            interval: Data interval (e.g., '1m', '1h', '1d')
            polling_interval: Seconds between API calls (default: 60 to respect rate limits)
            **kwargs: Additional arguments passed to BaseLiveDataFeed
        """
        self.base_url = "https://api.coingecko.com/api/v3"
        self.polling_interval = polling_interval
        self.last_poll_time = None
        self.session = requests.Session()

        # Rate limiting: CoinGecko allows 50 calls per minute
        self.max_calls_per_minute = 50
        self.call_times = []

        super().__init__(symbol=symbol, interval=interval, **kwargs)

    def _convert_interval(self, interval: str) -> str:
        """
        Convert standard interval format to CoinGecko format.

        Args:
            interval: Standard interval (e.g., '1m', '1h', '1d')

        Returns:
            CoinGecko interval format
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
        Load historical data from CoinGecko REST API.

        Returns:
            DataFrame with historical OHLCV data
        """
        try:
            _logger.info("Loading %d historical bars for %s", self.lookback_bars, self.symbol)

            # Calculate start time based on lookback_bars
            interval_minutes = self._get_interval_minutes()
            total_minutes = self.lookback_bars * interval_minutes
            start_time = datetime.now() - timedelta(minutes=total_minutes)

            # CoinGecko free API limit: 365 days maximum
            max_start_time = datetime.now() - timedelta(days=365)
            if start_time < max_start_time:
                start_time = max_start_time
                _logger.warning("CoinGecko API limited to 365 days, adjusting start time to %s", start_time)

            # Convert to UNIX timestamps
            start_timestamp = int(start_time.timestamp())
            end_timestamp = int(datetime.now().timestamp())

            # Get historical data
            url = f"{self.base_url}/coins/{self.symbol}/market_chart/range"
            params = {
                "vs_currency": "usd",
                "from": start_timestamp,
                "to": end_timestamp,
            }

            response = self._make_api_call(url, params)
            if response is None:
                return None

            data = response.json()

            # Process the data
            df = self._process_coin_data(data, start_timestamp, end_timestamp)

            if df is None or df.empty:
                _logger.warning("No historical data found for %s", self.symbol)
                return None

            _logger.info("Loaded %d historical bars for %s", len(df), self.symbol)
            return df

        except Exception:
            _logger.exception("Error loading historical data for %s: %s")
            return None

    def _process_coin_data(self, data: Dict, start_timestamp: int, end_timestamp: int) -> Optional[pd.DataFrame]:
        """
        Process CoinGecko API response into OHLCV DataFrame.

        Args:
            data: API response data
            start_timestamp: Start timestamp
            end_timestamp: End timestamp

        Returns:
            DataFrame with OHLCV data
        """
        try:
            prices = data.get("prices", [])
            volumes = {v[0]: v[1] for v in data.get("total_volumes", [])}

            if not prices:
                return None

            # Convert to DataFrame
            df = pd.DataFrame(prices, columns=["timestamp", "close"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["volume"] = df["timestamp"].map(lambda ts: volumes.get(int(ts.timestamp() * 1000), 0))

            # Resample to requested interval
            rule = self._get_resample_rule()
            ohlcv = df.resample(rule, on="timestamp").agg({
                "close": ["first", "max", "min", "last"],
                "volume": "sum"
            })
            ohlcv.columns = ["open", "high", "low", "close", "volume"]
            ohlcv = ohlcv.reset_index()
            ohlcv = ohlcv.rename(columns={"timestamp": "datetime"})
            ohlcv = ohlcv[["datetime", "open", "high", "low", "close", "volume"]]
            ohlcv.set_index("datetime", inplace=True)

            return ohlcv

        except Exception:
            _logger.exception("Error processing CoinGecko data: %s")
            return None

    def _get_resample_rule(self) -> str:
        """
        Get pandas resample rule for the interval.

        Returns:
            Resample rule string
        """
        interval_map = {
            '1m': 'T',
            '5m': '5T',
            '15m': '15T',
            '30m': '30T',
            '1h': 'H',
            '4h': '4H',
            '1d': 'D',
        }
        return interval_map.get(self.interval, 'D')

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

    def _make_api_call(self, url: str, params: Dict) -> Optional[requests.Response]:
        """
        Make API call with rate limiting.

        Args:
            url: API URL
            params: Request parameters

        Returns:
            Response object or None if failed
        """
        try:
            # Check rate limits
            current_time = time.time()
            self.call_times = [t for t in self.call_times if current_time - t < 60]

            if len(self.call_times) >= self.max_calls_per_minute:
                wait_time = 60 - (current_time - self.call_times[0])
                if wait_time > 0:
                    _logger.warning("Rate limit reached, waiting %d seconds", wait_time)
                    time.sleep(wait_time)

            # Make the API call
            response = self.session.get(url, params=params, timeout=30)
            self.call_times.append(current_time)

            if response.status_code == 429:
                _logger.warning("Rate limit exceeded, waiting 60 seconds")
                time.sleep(60)
                return self._make_api_call(url, params)  # Retry once
            elif response.status_code != 200:
                _logger.error("CoinGecko API error: %d %s", response.status_code, response.text)
                return None

            return response

        except requests.exceptions.RequestException:
            _logger.exception("Request error: %s")
            return None
        except Exception:
            _logger.exception("Unexpected error in API call: %s")
            return None

    def _connect_realtime(self) -> bool:
        """
        Connect to CoinGecko for real-time data (polling-based).

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Test connection by making a simple API call
            url = f"{self.base_url}/ping"
            response = self._make_api_call(url, {})

            if response is not None:
                _logger.info("Successfully connected to CoinGecko API")
                return True
            else:
                _logger.error("Failed to connect to CoinGecko API")
                return False

        except Exception:
            _logger.exception("Error connecting to CoinGecko: %s")
            return False

    def _disconnect_realtime(self):
        """Disconnect from CoinGecko (no persistent connection to close)."""
        _logger.info("Disconnected from CoinGecko API")

    def _get_latest_data(self) -> Optional[pd.DataFrame]:
        """
        Get the latest data from CoinGecko via polling.

        Returns:
            DataFrame with latest bar(s), or None if no new data
        """
        try:
            current_time = time.time()

            # Check if enough time has passed since last poll
            if (self.last_poll_time is not None and
                current_time - self.last_poll_time < self.polling_interval):
                return None

            # Get current price and volume
            url = f"{self.base_url}/simple/price"
            params = {
                "ids": self.symbol,
                "vs_currencies": "usd",
                "include_24hr_vol": "true",
                "include_24hr_change": "true",
                "include_last_updated_at": "true"
            }

            response = self._make_api_call(url, params)
            if response is None:
                return None

            data = response.json()

            if self.symbol not in data:
                _logger.warning("Symbol %s not found in CoinGecko response", self.symbol)
                return None

            coin_data = data[self.symbol]

            # Create a new bar
            timestamp = datetime.fromtimestamp(coin_data.get("last_updated_at", current_time))
            price = coin_data.get("usd", 0)
            volume_24h = coin_data.get("usd_24h_vol", 0)

            # Create DataFrame with current data
            new_bar = pd.DataFrame({
                "open": [price],
                "high": [price],
                "low": [price],
                "close": [price],
                "volume": [volume_24h / 24]  # Approximate hourly volume
            }, index=[timestamp])

            self.last_poll_time = current_time
            return new_bar

        except Exception:
            _logger.exception("Error getting latest data for %s: %s")
            return None

    def _get_update_interval(self) -> int:
        """
        Get the update interval in seconds.

        Returns:
            Update interval in seconds
        """
        return self.polling_interval

    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the live feed.

        Returns:
            Dictionary with status information
        """
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "is_connected": self.is_connected,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "last_poll_time": datetime.fromtimestamp(self.last_poll_time).isoformat() if self.last_poll_time else None,
            "polling_interval": self.polling_interval,
            "api_calls_last_minute": len(self.call_times),
            "max_api_calls_per_minute": self.max_calls_per_minute,
            "data_source": "CoinGecko"
        }
