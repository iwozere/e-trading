"""
Binance Data Downloader Module
-----------------------------

This module provides the BinanceDataDownloader class for downloading historical OHLCV (Open, High, Low, Close, Volume) data from the Binance exchange. It supports fetching data for single or multiple symbols and saving the results as CSV files for use in backtesting and analysis workflows.

Main Features:
- Download historical candlestick data for cryptocurrencies
- Save data to CSV files
- Download data for multiple symbols in batch
- Inherits common logic from BaseDataDownloader for file management

Valid values:
- interval: '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'
- period: Any string like '1d', '7d', '1w', '1mo', '3mo', '6mo', '1y', '2y', etc. (used to calculate start_date/end_date)

Classes:
- BinanceDataDownloader: Main class for interacting with the Binance API and managing data downloads
"""

from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd

from src.data.downloader.base_data_downloader import BaseDataDownloader
from src.notification.logger import setup_logger
from src.model.schemas import Fundamentals

_logger = setup_logger(__name__)


class BinanceDataDownloader(BaseDataDownloader):
    """
    A class to download historical data from Binance.

    This class provides methods to:
    1. Download historical OHLCV data for cryptocurrencies
    2. Save data to CSV files
    3. Load data from CSV files
    4. Update existing data files with new data
    5. Get fundamental data (NotImplementedError - Binance doesn't provide stock fundamentals)

    **Fundamental Data Capabilities:**
    - ❌ PE Ratio (cryptocurrency exchange)
    - ❌ Financial Ratios (cryptocurrency exchange)
    - ❌ Growth Metrics (cryptocurrency exchange)
    - ❌ Company Information (cryptocurrency exchange)
    - ❌ Market Data (only cryptocurrency data)
    - ❌ Profitability Metrics (cryptocurrency exchange)
    - ❌ Valuation Metrics (cryptocurrency exchange)

    **Data Quality:** N/A - Binance is for cryptocurrencies, not stocks
    **Rate Limits:** 1200 requests per minute (free tier)
    **Coverage:** Cryptocurrencies only
    **Bar Limits:** Maximum 1000 bars per request

    Parameters:
    -----------
    api_key : str
        Binance API key
    secret_key : str
        Binance secret key

    Example:
    --------
    >>> from datetime import datetime
    >>> downloader = BinanceDataDownloader("YOUR_API_KEY", "YOUR_SECRET_KEY")
    >>> # Get OHLCV data for cryptocurrency
    >>> df = downloader.get_ohlcv("BTCUSDT", "1d", datetime(2023, 1, 1), datetime(2023, 12, 31))
    >>> # Get fundamental data (will raise NotImplementedError)
    >>> try:
    >>>     fundamentals = downloader.get_fundamentals("AAPL")
    >>> except NotImplementedError as e:
    >>>     print(f"Not supported: {e}")
    """

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """
        Initialize Binance data downloader.

        Args:
            api_key: Binance API key (optional for public data)
            api_secret: Binance API secret (optional for public data)
        """
        super().__init__()
        # Get API credentials from parameter or config
        from config.donotshare.donotshare import BINANCE_KEY, BINANCE_SECRET

        self.api_key = api_key or BINANCE_KEY
        self.api_secret = api_secret or BINANCE_SECRET
        self.client = None  # Lazy initialization

    def _get_client(self):
        """Get or create Binance client with lazy initialization."""
        if self.client is None:
            try:
                # Lazy import to avoid loading binance package at module import time
                from binance.client import Client
                self.client = Client(self.api_key, self.api_secret)
            except Exception as e:
                _logger.warning("Failed to initialize Binance client: %s", e)
                # Create a mock client for testing
                self.client = None
        return self.client

    def _calculate_batch_dates(self, start_date: datetime, end_date: datetime, interval: str) -> List[tuple]:
        """
        Calculate batch dates to respect the 1000 bar limit.

        Args:
            start_date: Start date
            end_date: End date
            interval: Time interval

        Returns:
            List of (batch_start, batch_end) tuples
        """
        # Convert interval to minutes for calculation
        interval_minutes = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720,
            '1d': 1440, '3d': 4320, '1w': 10080, '1M': 43200
        }

        minutes_per_interval = interval_minutes.get(interval, 1440)  # default to 1d
        max_bars = 1000

        # Calculate maximum time span for 1000 bars
        max_minutes = minutes_per_interval * max_bars
        max_timedelta = timedelta(minutes=max_minutes)

        batches = []
        current_start = start_date

        while current_start < end_date:
            current_end = min(current_start + max_timedelta, end_date)
            batches.append((current_start, current_end))
            current_start = current_end

        return batches

    def get_supported_intervals(self) -> List[str]:
        """Return list of supported intervals for Binance."""
        return ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']

    def get_periods(self) -> List[str]:
        """Return list of supported periods for Binance."""
        return ['1d', '7d', '1mo', '3mo', '6mo', '1y', '2y']

    def get_intervals(self) -> List[str]:
        """Return list of supported intervals for Binance."""
        return self.get_supported_intervals()

    def is_valid_period_interval(self, period: str, interval: str) -> bool:
        """Check if the given period and interval combination is valid."""
        return interval in self.get_supported_intervals() and period in self.get_periods()

    def get_ohlcv(self, symbol: str, interval: str, start_date: datetime, end_date: datetime, **kwargs) -> pd.DataFrame:
        """
        Download historical klines/candlestick data from Binance with batching to respect 1000 bar limit.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Kline interval (e.g., '1h', '4h', '1d')
            start_date: Start date as datetime.datetime
            end_date: End date as datetime.datetime

        Returns:
            DataFrame containing the historical data
        """
        try:
            # Calculate batches to respect 1000 bar limit
            batches = self._calculate_batch_dates(start_date, end_date, interval)

            all_klines = []

            for batch_start, batch_end in batches:
                # Convert dates to timestamps
                start_timestamp = int(batch_start.timestamp() * 1000)
                end_timestamp = int(batch_end.timestamp() * 1000)

                _logger.debug("Downloading batch for %s %s: %s to %s",
                             symbol, interval, batch_start, batch_end)

                # Get klines data for this batch
                client = self._get_client()
                if client is None:
                    _logger.error("Binance client not available")
                    return pd.DataFrame()

                klines = client.get_historical_klines(
                    symbol, interval, start_timestamp, end_timestamp
                )

                all_klines.extend(klines)

                _logger.debug("Retrieved %d bars for batch", len(klines))

            if not all_klines:
                _logger.warning("No data returned for %s %s", symbol, interval)
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(
                all_klines,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_asset_volume",
                    "number_of_trades",
                    "taker_buy_base_asset_volume",
                    "taker_buy_quote_asset_volume",
                    "ignore",
                ],
            )

            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

            # Convert string values to float
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)

            # Remove duplicates and sort by timestamp
            df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

            # Keep timestamp as a column for consistency with other downloaders
            # The timestamp will be used as index by the data manager if needed

            _logger.info("Successfully downloaded %d bars for %s %s", len(df), symbol, interval)
            return df

        except Exception as e:
            _logger.exception("Error downloading Binance data for %s: %s", symbol, str(e))
            raise

    def get_fundamentals(self, symbol: str) -> Fundamentals:
        """
        Get fundamental data for a given symbol.

        Note: Binance doesn't provide fundamental data for stocks, so this method raises NotImplementedError.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')

        Returns:
            Fundamentals: Fundamental data for the stock

        Raises:
            NotImplementedError: Binance doesn't provide fundamental data for stocks
        """
        raise NotImplementedError("Binance doesn't provide fundamental data for stocks")
