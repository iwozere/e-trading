from typing import Optional, List, Dict
from datetime import datetime
import pandas as pd
import requests
from src.notification.logger import setup_logger
from src.model.schemas import OptionalFundamentals, Fundamentals
from src.data.downloader.base_data_downloader import BaseDataDownloader

_logger = setup_logger(__name__)

"""
Data downloader implementation for Polygon.io, fetching historical market data for analysis and backtesting.

This module provides the PolygonDataDownloader class for downloading historical OHLCV (Open, High, Low, Close, Volume) data from Polygon.io. It supports fetching, saving, loading, and updating data for single or multiple symbols, and is suitable for both research and production trading workflows.

Main Features:
- Download historical data for any stock or ticker from Polygon.io (free tier)
- Save and load data as CSV files
- Update existing data files with new data
- Download data for multiple symbols in batch
- Inherits common logic from BaseDataDownloader for file management

Valid values:
- interval: '1m', '5m', '15m', '1h', '1d' (Polygon free tier: 1m for 2 months, 1d for 2 years; other intervals are resampled)
- period: '1d', '7d', '1mo', '3mo', '6mo', '1y', '2y'

API limits (free tier):
- 5 requests per minute
- 2 years daily data, 2 months minute data
- If you exceed the free API limit or request unsupported data, a clear error will be raised.

Classes:
- PolygonDataDownloader: Main class for interacting with Polygon.io and managing data downloads
"""

class PolygonDataDownloader(BaseDataDownloader):
    """
    A class to download historical data from Polygon.io.

    This class provides methods to:
    1. Download historical OHLCV data for a given symbol
    2. Save data to CSV files
    3. Load data from CSV files
    4. Update existing data files with new data
    5. Get basic fundamental data for stocks (limited by free tier)

    **Fundamental Data Capabilities:**
    - ❌ PE Ratio (requires paid tier)
    - ❌ Financial Ratios (requires paid tier)
    - ❌ Growth Metrics (requires paid tier)
    - ✅ Company Information (name, sector, industry, country, exchange)
    - ✅ Market Data (market cap, current price, shares outstanding)
    - ❌ Profitability Metrics (requires paid tier)
    - ❌ Valuation Metrics (requires paid tier)

    **Data Quality:** Basic (free tier) - Limited fundamental data available
    **Rate Limits:** 5 API calls per minute (free tier)
    **Coverage:** US stocks and ETFs (free tier)

    Parameters:
    -----------
    api_key : str
        Polygon.io API key
    data_dir : str
        Directory to store downloaded data files

    Example:
    --------
    >>> from datetime import datetime
    >>> downloader = PolygonDataDownloader("YOUR_API_KEY")
    >>> df = downloader.get_ohlcv("AAPL", "1d", datetime(2023, 1, 1), datetime(2023, 12, 31))
    >>> # Get fundamental data (limited in free tier)
    >>> fundamentals = downloader.get_fundamentals("AAPL")
    >>> print(f"Company: {fundamentals.company_name}")
    >>> print(f"Market Cap: ${fundamentals.market_cap:,.0f}")
    """

    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        # Get API key from parameter or config
        self.api_key = api_key or self._get_config_value('POLYGON_API_KEY', 'POLYGON_API_KEY')
        self.base_url = "https://api.polygon.io/v2/aggs/ticker"

        if not self.api_key:
            raise ValueError("Polygon API key is required. Get one at: https://polygon.io/")

    def get_ohlcv(self, symbol: str, interval: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Download historical data for a given symbol from Polygon.io.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            interval: Data interval ('1m', '5m', '15m', '1h', '1d')
            start_date: Start date as datetime.datetime
            end_date: End date as datetime.datetime

        Returns:
            pd.DataFrame: Historical OHLCV data
        """
        try:
            # Validate interval and period
            if not self.is_valid_period_interval('1d', interval):
                raise ValueError(f"Unsupported interval: {interval}")
            # Polygon supports 'minute' and 'day' granularity
            if interval == '1d':
                timespan = 'day'
            else:
                timespan = 'minute'
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            url = f"{self.base_url}/{symbol}/range/1/{timespan}/{start_str}/{end_str}"
            params = {
                'adjusted': 'true',
                'sort': 'asc',
                'apiKey': self.api_key
            }
            response = requests.get(url, params=params)
            if response.status_code == 429:
                raise RuntimeError("Polygon.io API rate limit exceeded (free tier: 5 requests/minute)")
            if response.status_code != 200:
                raise RuntimeError(f"Polygon.io API error: {response.status_code} {response.text}")
            data = response.json()
            if 'results' not in data:
                raise ValueError(f"No results in Polygon.io response: {data}")
            df = pd.DataFrame(data['results'])
            # Polygon returns: t (timestamp ms), o (open), h (high), l (low), c (close), v (volume)
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
            df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            # Resample if needed
            if interval == '5m':
                df = df.set_index('timestamp').resample('5T').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna().reset_index()
            elif interval == '15m':
                df = df.set_index('timestamp').resample('15T').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna().reset_index()
            elif interval == '1h':
                df = df.set_index('timestamp').resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna().reset_index()
            # For '1m' and '1d', no resampling needed
            return df
        except Exception as e:
            _logger.exception("Error downloading data for %s: %s", symbol, str(e))
            raise

    def get_periods(self) -> list:
        return ['1d', '7d', '1mo', '3mo', '6mo', '1y', '2y']

    def get_intervals(self) -> list:
        return ['1m', '5m', '15m', '1h', '1d']

    def is_valid_period_interval(self, period, interval) -> bool:
        return interval in self.get_intervals() and period in self.get_periods()

    def get_fundamentals(self, symbol: str) -> OptionalFundamentals:
        """
        Get comprehensive fundamental data for a given stock using Polygon.io.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')

        Returns:
            Fundamentals: Comprehensive fundamental data for the stock
        """
        try:
            # Get ticker details
            ticker_url = f"https://api.polygon.io/v3/reference/tickers/{symbol}"
            ticker_params = {'apiKey': self.api_key}

            ticker_response = requests.get(ticker_url, params=ticker_params)
            if ticker_response.status_code == 429:
                raise RuntimeError("Polygon.io API rate limit exceeded")
            if ticker_response.status_code != 200:
                raise RuntimeError(f"Polygon.io API error: {ticker_response.status_code}")

            ticker_data = ticker_response.json()

            if not ticker_data or 'results' not in ticker_data:
                _logger.error("No data returned from Polygon.io for ticker %s", symbol)
                return Fundamentals(
                    ticker=symbol.upper(),
                    company_name="Unknown",
                    current_price=0.0,
                    market_cap=0.0,
                    pe_ratio=0.0,
                    forward_pe=0.0,
                    dividend_yield=0.0,
                    earnings_per_share=0.0,
                    data_source="Polygon.io",
                    last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )

            ticker_info = ticker_data['results']

            # Get current price
            quote_url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}/quote"
            quote_params = {'apiKey': self.api_key}

            quote_response = requests.get(quote_url, params=quote_params)
            quote_data = quote_response.json() if quote_response.status_code == 200 else {}
            current_price = quote_data.get('results', {}).get('p', 0.0) if quote_data else 0.0

            # Get financials (basic info)
            financials_url = f"https://api.polygon.io/v2/reference/financials/{symbol}"
            financials_params = {'apiKey': self.api_key}

            financials_response = requests.get(financials_url, params=financials_params)
            financials_data = financials_response.json() if financials_response.status_code == 200 else {}

            _logger.debug("Retrieved fundamentals for %s: %s", symbol, ticker_info.get('name', 'Unknown'))

            return Fundamentals(
                ticker=symbol.upper(),
                company_name=ticker_info.get("name", "Unknown"),
                current_price=current_price,
                market_cap=float(ticker_info.get("market_cap", 0)) if ticker_info.get("market_cap") else 0.0,
                pe_ratio=0.0,  # Polygon.io doesn't provide PE ratio in basic tier
                forward_pe=0.0,
                dividend_yield=0.0,  # Polygon.io doesn't provide dividend yield in basic tier
                earnings_per_share=0.0,  # Polygon.io doesn't provide EPS in basic tier
                # Additional fields
                price_to_book=None,
                return_on_equity=None,
                return_on_assets=None,
                debt_to_equity=None,
                current_ratio=None,
                quick_ratio=None,
                revenue=None,
                revenue_growth=None,
                net_income=None,
                net_income_growth=None,
                free_cash_flow=None,
                operating_margin=None,
                profit_margin=None,
                beta=None,
                sector=ticker_info.get("sic_description", None),
                industry=ticker_info.get("sic_description", None),
                country=ticker_info.get("locale", None),
                exchange=ticker_info.get("primary_exchange", None),
                currency=ticker_info.get("currency_name", None),
                shares_outstanding=float(ticker_info.get("share_class_shares_outstanding", 0)) if ticker_info.get("share_class_shares_outstanding") else None,
                float_shares=None,
                short_ratio=None,
                payout_ratio=None,
                peg_ratio=None,
                price_to_sales=None,
                enterprise_value=None,
                enterprise_value_to_ebitda=None,
                data_source="Polygon.io",
                last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )

        except Exception as e:
            _logger.exception("Failed to get fundamentals for %s: %s", symbol, str(e))
            return Fundamentals(
                ticker=symbol.upper(),
                company_name="Unknown",
                current_price=0.0,
                market_cap=0.0,
                pe_ratio=0.0,
                forward_pe=0.0,
                dividend_yield=0.0,
                earnings_per_share=0.0,
                data_source="Polygon.io",
                last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )

    def get_supported_intervals(self) -> List[str]:
        """Return list of supported intervals for Polygon.io."""
        return ['1m', '5m', '15m', '1h', '1d']

    def download_multiple_symbols(
        self, symbols: List[str], interval: str, start_date: datetime, end_date: datetime
    ) -> Dict[str, str]:
        def download_func(symbol, interval, start_date, end_date):
            return self.get_ohlcv(symbol, interval, start_date, end_date)
        return super().download_multiple_symbols(
            symbols, download_func, interval, start_date, end_date
        )
