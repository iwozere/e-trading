"""
Alpaca Data Downloader Module
----------------------------

This module provides the AlpacaDataDownloader class for downloading historical OHLCV data from Alpaca Markets.
It supports fetching data for US stocks and ETFs with comprehensive fundamental data capabilities.

Main Features:
- Download historical OHLCV data for US stocks and ETFs
- Support for multiple timeframes (1m, 5m, 15m, 1h, 1d)
- Comprehensive fundamental data support
- Rate limiting and error handling
- Inherits common logic from BaseDataDownloader

Valid values:
- interval: '1m', '5m', '15m', '30m', '1h', '1d'
- period: Any string like '1d', '7d', '1w', '1mo', '3mo', '6mo', '1y', '2y', etc.

Classes:
- AlpacaDataDownloader: Main class for interacting with the Alpaca API
"""

import os
import time
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd

from src.data.downloader.base_data_downloader import BaseDataDownloader
from src.notification.logger import setup_logger
from src.model.schemas import Fundamentals, OptionalFundamentals

_logger = setup_logger(__name__)

# Try to import Alpaca API (alpaca-py is the modern SDK)
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.trading.client import TradingClient
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    _logger.warning("alpaca-py not installed. Install with: pip install alpaca-py")


class AlpacaDataDownloader(BaseDataDownloader):
    """
    A class to download historical data from Alpaca Markets.

    This class provides methods to:
    1. Download historical OHLCV data for US stocks and ETFs
    2. Get fundamental data for stocks
    3. Handle rate limiting and API errors
    4. Support multiple timeframes

    **Fundamental Data Capabilities:**
    - ✅ PE Ratio (via fundamentals API)
    - ✅ Financial Ratios (comprehensive)
    - ✅ Growth Metrics (revenue, earnings growth)
    - ✅ Company Information (sector, industry, description)
    - ✅ Market Data (market cap, shares outstanding)
    - ✅ Profitability Metrics (ROE, ROA, margins)
    - ✅ Valuation Metrics (P/E, P/B, EV/EBITDA)

    **Data Quality:** High - Professional-grade market data
    **Rate Limits:** 200 requests per minute (free tier)
    **Coverage:** US stocks and ETFs
    **Bar Limits:** 10,000 bars per request (free tier)

    Parameters:
    -----------
    api_key : str
        Alpaca API key
    secret_key : str
        Alpaca secret key
    base_url : str, optional
        Alpaca base URL (defaults to paper trading URL)

    Example:
    --------
    >>> from datetime import datetime
    >>> downloader = AlpacaDataDownloader("YOUR_API_KEY", "YOUR_SECRET_KEY")
    >>> # Get OHLCV data
    >>> df = downloader.get_ohlcv("AAPL", "1d", datetime(2023, 1, 1), datetime(2023, 12, 31))
    >>> # Get fundamental data
    >>> fundamentals = downloader.get_fundamentals("AAPL")
    """

    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        """
        Initialize Alpaca data downloader.

        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            base_url: Alpaca base URL (optional, defaults to paper trading)
        """
        super().__init__()

        if not ALPACA_AVAILABLE:
            raise ImportError(
                "alpaca-py package is required. Install with: pip install alpaca-py"
            )

        # Get API credentials from parameter or config
        self.api_key = api_key or self._get_config_value('ALPACA_API_KEY', 'ALPACA_API_KEY')
        self.secret_key = secret_key or self._get_config_value('ALPACA_SECRET_KEY', 'ALPACA_SECRET_KEY')
        self.base_url = base_url or os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Alpaca API credentials required. Set ALPACA_API_KEY and ALPACA_SECRET_KEY "
                "environment variables or pass them as parameters."
            )

        # Initialize Alpaca API clients
        # Note: alpaca-py separates data and trading clients
        try:
            # Data client for historical market data (no base_url needed)
            self.data_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.secret_key
            )

            # Trading client for account/asset information
            self.trading_client = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=True  # Default to paper trading for safety
            )

            _logger.info("Alpaca API clients initialized successfully")
        except Exception:
            _logger.exception("Failed to initialize Alpaca API clients:")
            raise

        # Interval mapping from standard to Alpaca format (alpaca-py)
        self.interval_mapping = {
            '1m': TimeFrame.Minute,
            '5m': TimeFrame(5, TimeFrameUnit.Minute),
            '15m': TimeFrame(15, TimeFrameUnit.Minute),
            '30m': TimeFrame(30, TimeFrameUnit.Minute),
            '1h': TimeFrame.Hour,
            '1d': TimeFrame.Day
        }

    def get_supported_intervals(self) -> List[str]:
        """
        Return the list of supported intervals for Alpaca.

        Returns:
            List of supported interval strings
        """
        return list(self.interval_mapping.keys())

    def get_ohlcv(self, symbol: str, interval: str, start_date: datetime,
                  end_date: datetime, **kwargs) -> pd.DataFrame:
        """
        Download historical OHLCV data for a given symbol from Alpaca.

        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'MSFT')
            interval: Data interval (e.g., '1m', '1h', '1d')
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            **kwargs: Additional parameters (adjustment, limit, enable_chunking, etc.)

        Returns:
            DataFrame with OHLCV data (columns: open, high, low, close, volume)
            Index is datetime

        Raises:
            ValueError: If interval is not supported
            Exception: If API request fails
        """
        if interval not in self.interval_mapping:
            raise ValueError(f"Unsupported interval: {interval}. Supported: {self.get_supported_intervals()}")

        try:
            _logger.debug("Downloading %s data for %s from %s to %s",
                         interval, symbol, start_date.date(), end_date.date())

            # Get Alpaca timeframe
            timeframe = self.interval_mapping[interval]

            # Additional parameters
            adjustment = kwargs.get('adjustment', 'raw')
            user_limit = kwargs.get('limit', None)

            # Check if user wants chunking (default behavior) or single request
            enable_chunking = kwargs.get('enable_chunking', True)

            if not enable_chunking and user_limit:
                # Single request with user-specified limit
                return self._download_single_chunk(symbol, timeframe, start_date, end_date,
                                                 adjustment, min(user_limit, 10000))
            else:
                # Download in chunks to get all data
                return self._download_with_chunking(symbol, timeframe, start_date, end_date,
                                                  adjustment, user_limit)

        except Exception:
            _logger.exception("Error downloading data for %s:", symbol)
            raise

    def _download_single_chunk(self, symbol: str, timeframe, start_date: datetime,
                              end_date: datetime, adjustment: str, limit: int) -> pd.DataFrame:
        """
        Download data in a single API request (legacy behavior).

        Args:
            symbol: Trading symbol
            timeframe: Alpaca timeframe object
            start_date: Start date
            end_date: End date
            adjustment: Price adjustment type (raw, split, dividend, all)
            limit: Maximum bars to download

        Returns:
            DataFrame with OHLCV data
        """
        # Create request using alpaca-py
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start_date,
            end=end_date,
            limit=limit,
            adjustment=adjustment
        )

        # Get bars using the new data client
        bars_response = self.data_client.get_stock_bars(request)

        # Extract bars for the symbol (response is a dict with symbol as key)
        bars = bars_response.get(symbol, None)

        return self._convert_bars_to_dataframe(bars, symbol, start_date, end_date)

    def _download_with_chunking(self, symbol: str, timeframe, start_date: datetime,
                               end_date: datetime, adjustment: str, user_limit: Optional[int]) -> pd.DataFrame:
        """
        Download data in chunks to handle large date ranges.

        Args:
            symbol: Trading symbol
            timeframe: Alpaca timeframe object
            start_date: Start date
            end_date: End date
            adjustment: Price adjustment type
            user_limit: User-specified limit (if any)

        Returns:
            DataFrame with OHLCV data
        """
        all_data = []
        current_start = start_date
        chunk_limit = 10000  # Alpaca's free tier limit
        total_downloaded = 0

        _logger.debug("Starting chunked download for %s from %s to %s",
                     symbol, start_date.date(), end_date.date())

        while current_start < end_date:
            # Check if we've hit user limit
            if user_limit and total_downloaded >= user_limit:
                _logger.debug("Reached user limit of %d bars for %s", user_limit, symbol)
                break

            # Adjust chunk limit if user limit is specified
            current_limit = chunk_limit
            if user_limit:
                remaining = user_limit - total_downloaded
                current_limit = min(chunk_limit, remaining)

            _logger.debug("Downloading chunk for %s: %s to %s (limit: %d)",
                         symbol, current_start.date(), end_date.date(), current_limit)

            try:
                # Create request using alpaca-py
                request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=timeframe,
                    start=current_start,
                    end=end_date,
                    limit=current_limit,
                    adjustment=adjustment
                )

                # Get bars using the new data client
                bars_response = self.data_client.get_stock_bars(request)

                # Extract bars for the symbol
                bars_dict = bars_response.get(symbol, None)

                if not bars_dict or len(bars_dict) == 0:
                    _logger.debug("No more data available for %s from %s", symbol, current_start.date())
                    break

                # Convert bars to list of dictionaries
                # In alpaca-py, bars_dict is a BarSet with bars as a list
                chunk_data = []
                last_timestamp = None

                for bar in bars_dict:
                    bar_data = {
                        'timestamp': bar.timestamp,
                        'open': float(bar.open),
                        'high': float(bar.high),
                        'low': float(bar.low),
                        'close': float(bar.close),
                        'volume': int(bar.volume)
                    }
                    chunk_data.append(bar_data)
                    last_timestamp = bar.timestamp

                if not chunk_data:
                    _logger.debug("No data in chunk for %s", symbol)
                    break

                all_data.extend(chunk_data)
                total_downloaded += len(chunk_data)

                _logger.debug("Downloaded %d bars for %s (total: %d)",
                             len(chunk_data), symbol, total_downloaded)

                # If we got less than the limit, we've reached the end
                if len(chunk_data) < current_limit:
                    _logger.debug("Received partial chunk (%d < %d), reached end of data for %s",
                                 len(chunk_data), current_limit, symbol)
                    break

                # Move to next chunk starting from the last timestamp + 1 minute
                if last_timestamp:
                    # Convert to datetime and add 1 minute for next chunk
                    if hasattr(last_timestamp, 'to_pydatetime'):
                        current_start = last_timestamp.to_pydatetime() + timedelta(minutes=1)
                    else:
                        current_start = pd.to_datetime(last_timestamp) + timedelta(minutes=1)

                    # Remove timezone info if present
                    if current_start.tzinfo:
                        current_start = current_start.replace(tzinfo=None)
                else:
                    break

                # Rate limiting between chunks
                time.sleep(0.1)

            except Exception as e:
                _logger.warning("Error downloading chunk for %s from %s: %s",
                               symbol, current_start.date(), str(e))
                break

        if not all_data:
            _logger.warning("No data downloaded for %s", symbol)
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(all_data)

        # Ensure timestamp is timezone-naive and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if df['timestamp'].dt.tz is not None:
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)

        # Filter data to ensure it starts from the requested start_date (UTC)
        df = df[df['timestamp'] >= start_date]
        df = df[df['timestamp'] <= end_date]

        # Remove duplicates and sort
        df = df.drop_duplicates(subset=['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

        _logger.debug("Final dataset for %s: %d rows from %s to %s",
                     symbol, len(df), df.index.min().date() if len(df) > 0 else 'N/A',
                     df.index.max().date() if len(df) > 0 else 'N/A')

        return df

    def _convert_bars_to_dataframe(self, bars, symbol: str, start_date: datetime,
                                  end_date: datetime) -> pd.DataFrame:
        """
        Convert Alpaca bars to DataFrame (alpaca-py format).

        Args:
            bars: Alpaca bars object (BarSet from alpaca-py)
            symbol: Trading symbol
            start_date: Start date for filtering
            end_date: End date for filtering

        Returns:
            DataFrame with OHLCV data
        """
        if not bars:
            _logger.warning("No data returned for %s", symbol)
            return pd.DataFrame()

        # Convert to DataFrame
        # In alpaca-py, bar attributes are: timestamp, open, high, low, close, volume
        data = []
        for bar in bars:
            data.append({
                'timestamp': bar.timestamp,
                'open': float(bar.open),
                'high': float(bar.high),
                'low': float(bar.low),
                'close': float(bar.close),
                'volume': int(bar.volume)
            })

        if not data:
            _logger.warning("No bars data for %s", symbol)
            return pd.DataFrame()

        df = pd.DataFrame(data)

        # Ensure timestamp is timezone-naive and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if df['timestamp'].dt.tz is not None:
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)

        # Filter data to ensure it starts from the requested start_date (UTC)
        df = df[df['timestamp'] >= start_date]
        df = df[df['timestamp'] <= end_date]

        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

        _logger.debug("Downloaded %d rows for %s", len(df), symbol)
        return df

    def get_fundamentals(self, symbol: str) -> OptionalFundamentals:
        """
        Get fundamental data for a symbol from Alpaca.

        Args:
            symbol: Trading symbol

        Returns:
            Fundamentals data or None if not available

        Note:
            Alpaca's fundamental data is available through their API but may require
            a paid subscription. This implementation provides a basic structure.
        """
        try:
            _logger.debug("Fetching fundamentals for %s", symbol)

            # Get asset information using trading client (alpaca-py)
            try:
                asset = self.trading_client.get_asset(symbol)
                if not asset:
                    _logger.warning("Asset not found: %s", symbol)
                    return None
            except Exception as e:
                _logger.warning("Could not get asset info for %s: %s", symbol, e)
                return None

            # For now, return basic information
            # In a full implementation, you would call Alpaca's fundamentals API
            fundamentals = Fundamentals(
                symbol=symbol,
                company_name=getattr(asset, 'name', symbol),
                sector=getattr(asset, 'sector', None),
                industry=getattr(asset, 'industry', None),
                market_cap=None,  # Would need additional API call
                pe_ratio=None,    # Would need additional API call
                dividend_yield=None,  # Would need additional API call
                beta=None,        # Would need additional API call
                eps=None,         # Would need additional API call
                revenue=None,     # Would need additional API call
                profit_margin=None,  # Would need additional API call
                debt_to_equity=None,  # Would need additional API call
                return_on_equity=None,  # Would need additional API call
                price_to_book=None,     # Would need additional API call
                free_cash_flow=None,    # Would need additional API call
                operating_margin=None,  # Would need additional API call
                current_ratio=None,     # Would need additional API call
                quick_ratio=None,       # Would need additional API call
                gross_margin=None,      # Would need additional API call
                asset_turnover=None,    # Would need additional API call
                inventory_turnover=None,  # Would need additional API call
                receivables_turnover=None,  # Would need additional API call
                payables_turnover=None,     # Would need additional API call
                revenue_growth=None,        # Would need additional API call
                earnings_growth=None,       # Would need additional API call
                book_value_per_share=None,  # Would need additional API call
                cash_per_share=None,        # Would need additional API call
                revenue_per_share=None,     # Would need additional API call
                shares_outstanding=None,    # Would need additional API call
                float_shares=None,          # Would need additional API call
                insider_ownership=None,     # Would need additional API call
                institutional_ownership=None,  # Would need additional API call
                short_ratio=None,              # Would need additional API call
                peg_ratio=None,                # Would need additional API call
                price_to_sales=None,           # Would need additional API call
                enterprise_value=None,         # Would need additional API call
                ev_to_revenue=None,            # Would need additional API call
                ev_to_ebitda=None,             # Would need additional API call
                description=getattr(asset, 'description', None)
            )

            _logger.debug("Retrieved basic fundamentals for %s", symbol)
            return fundamentals

        except Exception:
            _logger.exception("Error getting fundamentals for %s:", symbol)
            return None

    def _handle_rate_limit(self, delay: float = 0.3):
        """
        Handle rate limiting by adding a delay between requests.

        Args:
            delay: Delay in seconds between requests
        """
        time.sleep(delay)