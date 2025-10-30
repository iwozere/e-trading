import os
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import requests
from src.notification.logger import setup_logger
from src.model.schemas import OptionalFundamentals, Fundamentals
_logger = setup_logger(__name__)
from src.data.downloader.base_data_downloader import BaseDataDownloader

"""
Data downloader implementation for Alpha Vantage, fetching historical market data for analysis and backtesting.

This module provides the AlphaVantageDataDownloader class for downloading historical OHLCV (Open, High, Low, Close, Volume) data from Alpha Vantage. It supports fetching, saving, loading, and updating data for single or multiple symbols, and is suitable for both research and production trading workflows.

Main Features:
- Download historical data for any stock or ticker from Alpha Vantage
- Save and load data as CSV files
- Update existing data files with new data
- Download data for multiple symbols in batch
- Inherits common logic from BaseDataDownloader for file management

Valid values:
- interval: '1m', '5m', '15m', '30m', '60m', '1d', '1wk', '1mo' (Alpha Vantage supports: 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly)
- period: Any string like '1d', '7d', '1w', '1mo', '3mo', '6mo', '1y', '2y', etc. (used to calculate start_date/end_date)

Classes:
- AlphaVantageDataDownloader: Main class for interacting with Alpha Vantage and managing data downloads
"""

def safe_float(val, default=0.0):
    try:
        if val is None or val == "" or val == "None":
            return default
        return float(val)
    except Exception:
        return default

class AlphaVantageDataDownloader(BaseDataDownloader):
    """
    A class to download historical data from Alpha Vantage.

    This class provides methods to:
    1. Download historical OHLCV data for a given symbol
    2. Save data to CSV files
    3. Load data from CSV files
    4. Update existing data files with new data
    5. Get comprehensive fundamental data for stocks

    **Fundamental Data Capabilities:**
    - ✅ PE Ratio (trailing and forward)
    - ✅ Financial Ratios (P/B, ROE, ROA, debt/equity, current ratio, quick ratio)
    - ✅ Growth Metrics (revenue growth, net income growth)
    - ✅ Company Information (name, sector, industry, country, exchange)
    - ✅ Market Data (market cap, current price, shares outstanding)
    - ✅ Profitability Metrics (operating margin, profit margin, free cash flow)
    - ✅ Valuation Metrics (beta, PEG ratio, price-to-sales, enterprise value)

    **Data Quality:** High - Alpha Vantage provides comprehensive fundamental data
    **Rate Limits:** 5 API calls per minute (free tier), 500 per day (free tier)
    **Coverage:** Global stocks and ETFs

    Parameters:
    -----------
    api_key : str
        Alpha Vantage API key

    Example:
    --------
    >>> from datetime import datetime
    >>> downloader = AlphaVantageDataDownloader(api_key="YOUR_API_KEY")
    >>> df = downloader.get_ohlcv("AAPL", "1d", datetime(2023, 1, 1), datetime(2023, 12, 31))
    >>> # Get fundamental data
    >>> fundamentals = downloader.get_fundamentals("AAPL")
    >>> print(f"PE Ratio: {fundamentals.pe_ratio}")
    >>> print(f"Market Cap: ${fundamentals.market_cap:,.0f}")
    """

    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        # Get API key from parameter or config
        from config.donotshare.donotshare import ALPHA_VANTAGE_KEY

        self.api_key = api_key or ALPHA_VANTAGE_KEY
        self.base_url = "https://www.alphavantage.co/query"

        if not self.api_key:
            raise ValueError("Alpha Vantage API key is required. Get one at: https://www.alphavantage.co/support/#api-key")

    def get_ohlcv(
        self, symbol: str, interval: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        Download historical data for a given symbol from Alpha Vantage.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            interval: Data interval (e.g., '1d', '1h', '15m')
            start_date: Start date as datetime.datetime
            end_date: End date as datetime.datetime

        Returns:
            pd.DataFrame: Historical OHLCV data
        """
        try:
            # Alpha Vantage supports: 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly
            interval_map = {
                '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min', '60m': '60min',
                '1d': 'daily', '1h': '60min', '1wk': 'weekly', '1mo': 'monthly'
            }
            av_interval = interval_map.get(interval, 'daily')
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            if av_interval in ['daily', 'weekly', 'monthly']:
                function = f'TIME_SERIES_{av_interval.upper()}'
                params = {
                    'function': function,
                    'symbol': symbol,
                    'apikey': self.api_key,
                    'outputsize': 'full',
                    'datatype': 'json',
                }
            else:
                function = 'TIME_SERIES_INTRADAY'
                params = {
                    'function': function,
                    'symbol': symbol,
                    'interval': av_interval,
                    'apikey': self.api_key,
                    'outputsize': 'full',
                    'datatype': 'json',
                }
            response = requests.get(self.base_url, params=params)
            data = response.json()

            # Find the key for the time series data
            ts_key = None
            for k in data.keys():
                if 'Time Series' in k:
                    ts_key = k
                    break
            if not ts_key:
                raise ValueError(f"No time series data found in Alpha Vantage response: {data}")
            ts_data = data[ts_key]

            # Convert to DataFrame
            df = pd.DataFrame.from_dict(ts_data, orient='index')
            df = df.rename(columns=lambda x: x.lower().replace(' ', '').replace('close', 'close').replace('open', 'open').replace('high', 'high').replace('low', 'low').replace('volume', 'volume'))
            # Standardize column names (after lambda function removes spaces)
            col_map = {
                '1.open': 'open',
                '2.high': 'high',
                '3.low': 'low',
                '4.close': 'close',
                '5.volume': 'volume',
            }
            df = df.rename(columns=col_map)
            # Convert index to datetime and filter by date range
            df['timestamp'] = pd.to_datetime(df.index)
            df = df.reset_index(drop=True)
            df = df.sort_values('timestamp')
            df = df[(df['timestamp'] >= pd.to_datetime(start_str)) & (df['timestamp'] <= pd.to_datetime(end_str))]
            # Ensure all required columns are present
            required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")
            # Convert numeric columns
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        except Exception as e:
            _logger.exception("Error downloading data for %s: %s", symbol, str(e))
            raise


    def get_periods(self) -> list:
        return ['1d', '7d', '1w', '1mo', '3mo', '6mo', '1y', '2y']

    def get_intervals(self) -> list:
        return ['1m', '5m', '15m', '30m', '60m', '1d', '1wk', '1mo']

    def is_valid_period_interval(self, period, interval) -> bool:
        return interval in self.get_intervals() and period in self.get_periods()

    def get_fundamentals(self, symbol: str) -> OptionalFundamentals:
        """
        Get comprehensive fundamental data for a given stock using Alpha Vantage.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')

        Returns:
            Fundamentals: Comprehensive fundamental data for the stock
        """
        try:
            # Get company overview
            overview_url = f"{self.base_url}"
            overview_params = {
                'function': 'OVERVIEW',
                'symbol': symbol,
                'apikey': self.api_key
            }

            overview_response = requests.get(overview_url, params=overview_params)
            if overview_response.status_code != 200:
                raise RuntimeError(f"Alpha Vantage API error: {overview_response.status_code}")

            overview_data = overview_response.json()

            if not overview_data or 'Error Message' in overview_data:
                _logger.error("No data returned from Alpha Vantage for ticker %s", symbol)
                return Fundamentals(
                    ticker=symbol.upper(),
                    company_name="Unknown",
                    current_price=0.0,
                    market_cap=0.0,
                    pe_ratio=0.0,
                    forward_pe=0.0,
                    dividend_yield=0.0,
                    earnings_per_share=0.0,
                    data_source="Alpha Vantage",
                    last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )

            # Get current price
            quote_url = f"{self.base_url}"
            quote_params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.api_key
            }

            quote_response = requests.get(quote_url, params=quote_params)
            quote_data = quote_response.json() if quote_response.status_code == 200 else {}
            current_price = safe_float(quote_data.get('Global Quote', {}).get('05. price', 0)) if quote_data else 0.0

            _logger.debug("Retrieved fundamentals for %s: %s", symbol, overview_data.get('Name', 'Unknown'))

            return Fundamentals(
                ticker=symbol.upper(),
                company_name=overview_data.get("Name", "Unknown"),
                current_price=safe_float(current_price),
                market_cap=safe_float(overview_data.get("MarketCapitalization")),
                pe_ratio=safe_float(overview_data.get("PERatio")),
                forward_pe=safe_float(overview_data.get("ForwardPE")),
                dividend_yield=safe_float(overview_data.get("DividendYield")),
                earnings_per_share=safe_float(overview_data.get("EPS")),
                # Additional fields
                price_to_book=safe_float(overview_data.get("PriceToBookRatio")),
                return_on_equity=safe_float(overview_data.get("ReturnOnEquityTTM")),
                return_on_assets=safe_float(overview_data.get("ReturnOnAssetsTTM")),
                debt_to_equity=safe_float(overview_data.get("DebtToEquityRatio")),
                current_ratio=safe_float(overview_data.get("CurrentRatio")),
                quick_ratio=safe_float(overview_data.get("QuickRatio")),
                revenue=safe_float(overview_data.get("RevenueTTM")),
                revenue_growth=safe_float(overview_data.get("RevenueGrowthTTM")),
                net_income=safe_float(overview_data.get("NetIncomeTTM")),
                net_income_growth=safe_float(overview_data.get("NetIncomeGrowthTTM")),
                free_cash_flow=safe_float(overview_data.get("FreeCashFlowTTM")),
                operating_margin=safe_float(overview_data.get("OperatingMarginTTM")),
                profit_margin=safe_float(overview_data.get("ProfitMarginTTM")),
                beta=safe_float(overview_data.get("Beta")),
                sector=overview_data.get("Sector"),
                industry=overview_data.get("Industry"),
                country=overview_data.get("Country"),
                exchange=overview_data.get("Exchange"),
                currency=overview_data.get("Currency"),
                shares_outstanding=safe_float(overview_data.get("SharesOutstanding")),
                float_shares=safe_float(overview_data.get("FloatShares")),
                short_ratio=safe_float(overview_data.get("ShortRatio")),
                payout_ratio=safe_float(overview_data.get("PayoutRatio")),
                peg_ratio=safe_float(overview_data.get("PEGRatio")),
                price_to_sales=safe_float(overview_data.get("PriceToSalesRatioTTM")),
                enterprise_value=safe_float(overview_data.get("MarketCapitalization")),
                enterprise_value_to_ebitda=safe_float(overview_data.get("EVToEBITDA")),
                data_source="Alpha Vantage",
                last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                sources=None
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
                data_source="Alpha Vantage",
                last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )

    def get_supported_intervals(self) -> List[str]:
        """Return list of supported intervals for Alpha Vantage."""
        return ['1m', '5m', '15m', '30m', '60m', '1d', '1wk', '1mo']

