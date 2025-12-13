from typing import Optional, List, Dict
from datetime import datetime
import pandas as pd
import requests
from src.notification.logger import setup_logger
from src.model.schemas import Fundamentals
from src.data.downloader.base_data_downloader import BaseDataDownloader

_logger = setup_logger(__name__)

"""
Data downloader implementation for Twelve Data, fetching historical market data for analysis and backtesting.

This module provides the TwelveDataDataDownloader class for downloading historical OHLCV (Open, High, Low, Close, Volume) data from Twelve Data. It supports fetching, saving, loading, and updating data for single or multiple symbols, and is suitable for both research and production trading workflows.

Main Features:
- Download historical data for any stock or ticker from Twelve Data (free tier)
- Save and load data as CSV files
- Update existing data files with new data
- Download data for multiple symbols in batch
- Inherits common logic from BaseDataDownloader for file management

Valid values:
- interval: '1m', '5m', '15m', '1h', '1d' (Twelve Data free tier: 1m for 1 month, 1d for 10 years; other intervals are resampled)
- period: '1d', '7d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y'

API limits (free tier):
- 8 requests per minute, 800 per day
- 1 month 1m data, 10 years 1d data
- If you exceed the free API limit or request unsupported data, a clear error will be raised.

Classes:
- TwelveDataDataDownloader: Main class for interacting with Twelve Data and managing data downloads
"""

class TwelveDataDataDownloader(BaseDataDownloader):
    """
    A class to download historical data from Twelve Data.

    This class provides methods to:
    1. Download historical OHLCV data for a given symbol
    2. Save data to CSV files
    3. Load data from CSV files
    4. Update existing data files with new data
    5. Get basic fundamental data for stocks

    **Fundamental Data Capabilities:**
    - ✅ PE Ratio (basic)
    - ✅ Price-to-Book Ratio
    - ❌ Financial Ratios (ROE, ROA, debt/equity, etc.)
    - ❌ Growth Metrics (revenue growth, net income growth)
    - ✅ Company Information (name, sector, industry, country, exchange)
    - ✅ Market Data (market cap, current price)
    - ✅ Earnings Data (EPS, revenue from earnings reports)
    - ✅ Beta (volatility measure)

    **Data Quality:** Basic - Limited fundamental data available
    **Rate Limits:** 8 API calls per minute (free tier), 800 per day (free tier)
    **Coverage:** Global stocks and ETFs

    Parameters:
    -----------
    api_key : str
        Twelve Data API key
    data_dir : str
        Directory to store downloaded data files

    Example:
    --------
    >>> from datetime import datetime
    >>> downloader = TwelveDataDataDownloader("YOUR_API_KEY")
    >>> df = downloader.get_ohlcv("AAPL", "1d", datetime(2023, 1, 1), datetime(2023, 12, 31))
    >>> # Get fundamental data
    >>> fundamentals = downloader.get_fundamentals("AAPL")
    >>> print(f"PE Ratio: {fundamentals.pe_ratio}")
    >>> print(f"Earnings Per Share: {fundamentals.earnings_per_share}")
    """

    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        # Get API key from parameter or config
        self.api_key = api_key or self._get_config_value('TWELVE_DATA_KEY', 'TWELVE_DATA_KEY')
        self.base_url = "https://api.twelvedata.com/time_series"

        if not self.api_key:
            raise ValueError("Twelve Data API key is required. Get one at: https://twelvedata.com/")

    def get_ohlcv(self, symbol: str, interval: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Download historical data for a given symbol from Twelve Data.

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
            # Twelve Data supports: 1min, 5min, 15min, 30min, 1h, 4h, 1d, 1wk, 1mo
            interval_map = {
                '1m': '1min', '5m': '5min', '15m': '15min', '1h': '1h', '1d': '1day'
            }
            td_interval = interval_map.get(interval, '1day')
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            params = {
                'symbol': symbol,
                'interval': td_interval,
                'start_date': start_str,
                'end_date': end_str,
                'apikey': self.api_key,
                'format': 'JSON',
                'outputsize': 5000  # max per request
            }
            response = requests.get(self.base_url, params=params)
            if response.status_code == 429:
                raise RuntimeError("Twelve Data API rate limit exceeded (free tier: 8 requests/minute, 800/day)")
            if response.status_code != 200:
                raise RuntimeError(f"Twelve Data API error: {response.status_code} {response.text}")
            data = response.json()
            if 'values' not in data:
                raise ValueError(f"No results in Twelve Data response: {data}")
            df = pd.DataFrame(data['values'])
            # Twelve Data returns: datetime, open, high, low, close, volume (all as strings)
            df['timestamp'] = pd.to_datetime(df['datetime'])
            df = df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'})
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            # Convert numeric columns
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
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
        return ['1d', '7d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y']

    def get_intervals(self) -> list:
        return ['1m', '5m', '15m', '1h', '1d']

    def is_valid_period_interval(self, period, interval) -> bool:
        return interval in self.get_intervals() and period in self.get_periods()

    def get_fundamentals(self, symbol: str) -> Fundamentals:
        """
        Get comprehensive fundamental data for a given stock using Twelve Data.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')

        Returns:
            Fundamentals: Comprehensive fundamental data for the stock
        """
        try:
            # Get company profile
            profile_url = "https://api.twelvedata.com/profile"
            profile_params = {
                'symbol': symbol,
                'apikey': self.api_key
            }

            profile_response = requests.get(profile_url, params=profile_params)
            if profile_response.status_code == 429:
                raise RuntimeError("Twelve Data API rate limit exceeded")
            if profile_response.status_code != 200:
                raise RuntimeError(f"Twelve Data API error: {profile_response.status_code}")

            profile_data = profile_response.json()

            if not profile_data or 'status' in profile_data and profile_data['status'] == 'error':
                _logger.error("No data returned from Twelve Data for ticker %s", symbol)
                return Fundamentals(
                    ticker=symbol.upper(),
                    company_name="Unknown",
                    current_price=0.0,
                    market_cap=0.0,
                    pe_ratio=0.0,
                    forward_pe=0.0,
                    dividend_yield=0.0,
                    earnings_per_share=0.0,
                    data_source="Twelve Data",
                    last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )

            # Get current price
            quote_url = "https://api.twelvedata.com/quote"
            quote_params = {
                'symbol': symbol,
                'apikey': self.api_key
            }

            quote_response = requests.get(quote_url, params=quote_params)
            quote_data = quote_response.json() if quote_response.status_code == 200 else {}
            current_price = float(quote_data.get('close', 0)) if quote_data and 'close' in quote_data else 0.0

            # Get earnings data
            earnings_url = "https://api.twelvedata.com/earnings"
            earnings_params = {
                'symbol': symbol,
                'apikey': self.api_key
            }

            earnings_response = requests.get(earnings_url, params=earnings_params)
            earnings_data = earnings_response.json() if earnings_response.status_code == 200 else {}
            latest_earnings = earnings_data.get('earnings', [{}])[0] if earnings_data and 'earnings' in earnings_data else {}

            _logger.debug("Retrieved fundamentals for %s: %s", symbol, profile_data.get('name', 'Unknown'))

            return Fundamentals(
                ticker=symbol.upper(),
                company_name=profile_data.get("name", "Unknown"),
                current_price=current_price,
                market_cap=float(profile_data.get("market_cap", 0)) if profile_data.get("market_cap") else 0.0,
                pe_ratio=float(profile_data.get("pe_ratio", 0)) if profile_data.get("pe_ratio") else 0.0,
                forward_pe=0.0,  # Twelve Data doesn't provide forward PE
                dividend_yield=float(profile_data.get("dividend_yield", 0)) if profile_data.get("dividend_yield") else 0.0,
                earnings_per_share=float(latest_earnings.get("eps", 0)) if latest_earnings.get("eps") else 0.0,
                # Additional fields
                price_to_book=float(profile_data.get("pb_ratio", 0)) if profile_data.get("pb_ratio") else None,
                return_on_equity=None,  # Twelve Data doesn't provide ROE
                return_on_assets=None,  # Twelve Data doesn't provide ROA
                debt_to_equity=None,  # Twelve Data doesn't provide debt/equity
                current_ratio=None,  # Twelve Data doesn't provide current ratio
                quick_ratio=None,  # Twelve Data doesn't provide quick ratio
                revenue=float(latest_earnings.get("revenue", 0)) if latest_earnings.get("revenue") else None,
                revenue_growth=None,  # Twelve Data doesn't provide revenue growth
                net_income=None,  # Twelve Data doesn't provide net income
                net_income_growth=None,  # Twelve Data doesn't provide net income growth
                free_cash_flow=None,  # Twelve Data doesn't provide FCF
                operating_margin=None,  # Twelve Data doesn't provide operating margin
                profit_margin=None,  # Twelve Data doesn't provide profit margin
                beta=float(profile_data.get("beta", 0)) if profile_data.get("beta") else None,
                sector=profile_data.get("sector", None),
                industry=profile_data.get("industry", None),
                country=profile_data.get("country", None),
                exchange=profile_data.get("exchange", None),
                currency=profile_data.get("currency", None),
                shares_outstanding=None,  # Twelve Data doesn't provide shares outstanding
                float_shares=None,  # Twelve Data doesn't provide float shares
                short_ratio=None,  # Twelve Data doesn't provide short ratio
                payout_ratio=None,  # Twelve Data doesn't provide payout ratio
                peg_ratio=None,  # Twelve Data doesn't provide PEG ratio
                price_to_sales=None,  # Twelve Data doesn't provide P/S ratio
                enterprise_value=None,  # Twelve Data doesn't provide enterprise value
                enterprise_value_to_ebitda=None,  # Twelve Data doesn't provide EV/EBITDA
                data_source="Twelve Data",
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
                data_source="Twelve Data",
                last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )

    def download_multiple_symbols(
        self, symbols: List[str], interval: str, start_date: datetime, end_date: datetime
    ) -> Dict[str, str]:
        def download_func(symbol, interval, start_date, end_date):
            return self.get_ohlcv(symbol, interval, start_date, end_date)
        return super().download_multiple_symbols(
            symbols, download_func, interval, start_date, end_date
        )

    def get_supported_intervals(self) -> List[str]:
        """Return list of supported intervals for Twelve Data."""
        return ['1m', '5m', '15m', '1h', '1d']
