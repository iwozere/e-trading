#!/usr/bin/env python3
"""
Financial Modeling Prep (FMP) Data Downloader

This module provides data downloading capabilities from Financial Modeling Prep (FMP) API.
FMP offers generous free tier limits (3,000 calls/minute) and comprehensive historical data.

API Documentation: https://site.financialmodelingprep.com/developer/docs
Free API Key: https://site.financialmodelingprep.com/developer/docs
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import logging
from pathlib import Path
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.notification.logger import setup_logger
from config.donotshare.donotshare import FMP_API_KEY

_logger = setup_logger(__name__)


class FMPDataDownloader:
    """Financial Modeling Prep (FMP) data downloader."""

    def __init__(self, api_key: Optional[str] = None, rate_limit_delay: float = 0.1):
        """
        Initialize FMP data downloader.

        Args:
            api_key: FMP API key. If None, uses FMP_API_KEY from donotshare.py
            rate_limit_delay: Delay between requests in seconds
        """
        self.api_key = api_key or FMP_API_KEY
        self.rate_limit_delay = rate_limit_delay
        self.base_url = "https://financialmodelingprep.com/api/v3"

        if not self.api_key:
            raise ValueError("FMP API key is required. Get one at: https://site.financialmodelingprep.com/developer/docs")

        _logger.info("FMP Data Downloader initialized with API key")

    def get_ohlcv(self, symbol: str, interval: str, start_date: datetime,
                   end_date: datetime, limit: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Get OHLCV data from FMP.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            interval: Time interval ('1m', '5m', '15m', '30m', '1h', '4h', '1d')
            start_date: Start date for data
            end_date: End date for data
            limit: Maximum number of data points (optional)

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            # Convert interval to FMP format
            fmp_interval = self._convert_interval(interval)
            if not fmp_interval:
                _logger.error(f"Unsupported interval: {interval}")
                return None

            # Format dates
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')

            # Build URL
            if fmp_interval == '1day':
                # Daily data endpoint
                url = f"{self.base_url}/historical-price-full/{symbol}"
                params = {
                    'apikey': self.api_key,
                    'from': start_str,
                    'to': end_str
                }
            else:
                # Intraday data endpoint
                url = f"{self.base_url}/historical-chart/{fmp_interval}/{symbol}"
                params = {
                    'apikey': self.api_key,
                    'from': start_str,
                    'to': end_str
                }

            # Add limit if specified
            if limit:
                params['limit'] = min(limit, 1000)  # FMP max limit is 1000

            _logger.info(f"Fetching {symbol} {interval} data from {start_str} to {end_str}")

            # Make request with rate limiting
            time.sleep(self.rate_limit_delay)
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Parse response based on endpoint
            if fmp_interval == '1day':
                # Daily data structure: {"historical": [...]}
                if 'historical' not in data:
                    _logger.warning(f"No historical data found for {symbol}")
                    return None
                historical_data = data['historical']
            else:
                # Intraday data structure: [...] (direct array)
                if not isinstance(data, list):
                    _logger.warning(f"No intraday data found for {symbol}")
                    return None
                historical_data = data

            if not historical_data:
                _logger.warning(f"No data returned for {symbol} {interval}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(historical_data)

            # Standardize column names
            df = self._standardize_dataframe(df, symbol, interval)

            _logger.info(f"Successfully fetched {len(df)} data points for {symbol} {interval}")
            return df

        except requests.exceptions.RequestException as e:
            _logger.error(f"Request failed for {symbol} {interval}: {e}")
            return None
        except Exception as e:
            _logger.error(f"Error fetching data for {symbol} {interval}: {e}")
            return None

    def _convert_interval(self, interval: str) -> Optional[str]:
        """Convert standard interval to FMP format."""
        interval_map = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1h': '1hour',
            '4h': '4hour',
            '1d': '1day'
        }
        return interval_map.get(interval)

    def _standardize_dataframe(self, df: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
        """Standardize DataFrame to common OHLCV format."""
        try:
            # FMP column mapping
            column_mapping = {
                'date': 'timestamp',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }

            # Rename columns
            df = df.rename(columns=column_mapping)

            # Ensure required columns exist
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    _logger.error(f"Missing required column: {col}")
                    return pd.DataFrame()

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Convert numeric columns
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Remove rows with NaN values
            df = df.dropna()

            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)

            # Set timestamp as index
            df = df.set_index('timestamp')
            df.index.name = 'timestamp'

            return df

        except Exception as e:
            _logger.error(f"Error standardizing DataFrame for {symbol} {interval}: {e}")
            return pd.DataFrame()

    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols (not implemented for FMP)."""
        # FMP doesn't provide a simple symbol list endpoint
        # In practice, you would use common stock symbols
        return []

    def get_supported_intervals(self) -> List[str]:
        """Get list of supported intervals."""
        return ['1m', '5m', '15m', '30m', '1h', '4h', '1d']

    def get_stock_screener(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get stock screener results from FMP.

        Args:
            criteria: Screening criteria (e.g., {'marketCapMoreThan': 1000000000, 'peRatioLessThan': 20})

        Returns:
            List of stock data dictionaries
        """
        try:
            url = f"{self.base_url}/stock-screener"
            params = {
                'apikey': self.api_key,
                **criteria
            }

            _logger.info(f"Running FMP stock screener with criteria: {criteria}")

            time.sleep(self.rate_limit_delay)
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if isinstance(data, list):
                _logger.info(f"FMP screener returned {len(data)} stocks")
                return data
            else:
                _logger.warning("FMP screener returned unexpected data format")
                return []

        except Exception as e:
            _logger.error(f"Error in FMP stock screener: {e}")
            return []

    def get_fundamentals_batch(self, tickers: List[str]) -> Dict[str, Any]:
        """
        Get fundamental data for multiple tickers.

        Args:
            tickers: List of ticker symbols

        Returns:
            Dictionary mapping ticker symbols to fundamental data
        """
        try:
            fundamentals = {}

            for ticker in tickers:
                ticker_fundamentals = self.get_fundamentals(ticker)
                if ticker_fundamentals:
                    fundamentals[ticker] = ticker_fundamentals

            _logger.info(f"Retrieved fundamentals for {len(fundamentals)} tickers")
            return fundamentals

        except Exception as e:
            _logger.error(f"Error getting batch fundamentals: {e}")
            return {}

    def get_fundamentals(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get fundamental data for a single ticker.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')

        Returns:
            Dictionary with fundamental data or None if failed
        """
        try:
            # Get company profile
            profile = self.get_company_profile(symbol)
            if not profile:
                return None

            # Get key metrics
            metrics = self.get_key_metrics(symbol)

            # Get financial ratios
            ratios = self.get_financial_ratios(symbol)

            # Combine all data
            fundamentals = {
                'symbol': symbol,
                'profile': profile,
                'metrics': metrics or {},
                'ratios': ratios or {}
            }

            _logger.info(f"Retrieved fundamentals for {symbol}")
            return fundamentals

        except Exception as e:
            _logger.error(f"Error getting fundamentals for {symbol}: {e}")
            return None

    def get_company_profile(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get company profile information."""
        try:
            url = f"{self.base_url}/profile/{symbol}"
            params = {'apikey': self.api_key}

            time.sleep(self.rate_limit_delay)
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if isinstance(data, list) and len(data) > 0:
                return data[0]
            elif isinstance(data, dict):
                return data
            else:
                return None

        except Exception as e:
            _logger.error(f"Error getting company profile for {symbol}: {e}")
            return None

    def get_key_metrics(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get key financial metrics."""
        try:
            url = f"{self.base_url}/key-metrics/{symbol}"
            params = {'apikey': self.api_key, 'limit': 1}

            time.sleep(self.rate_limit_delay)
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if isinstance(data, list) and len(data) > 0:
                return data[0]
            else:
                return None

        except Exception as e:
            _logger.error(f"Error getting key metrics for {symbol}: {e}")
            return None

    def get_financial_ratios(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get financial ratios."""
        try:
            url = f"{self.base_url}/ratios/{symbol}"
            params = {'apikey': self.api_key, 'limit': 1}

            time.sleep(self.rate_limit_delay)
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if isinstance(data, list) and len(data) > 0:
                return data[0]
            else:
                return None

        except Exception as e:
            _logger.error(f"Error getting financial ratios for {symbol}: {e}")
            return None

    def get_income_statement(self, symbol: str, limit: int = 1) -> Optional[List[Dict[str, Any]]]:
        """Get income statement data."""
        try:
            url = f"{self.base_url}/income-statement/{symbol}"
            params = {'apikey': self.api_key, 'limit': limit}

            time.sleep(self.rate_limit_delay)
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            return data if isinstance(data, list) else None

        except Exception as e:
            _logger.error(f"Error getting income statement for {symbol}: {e}")
            return None

    def get_balance_sheet(self, symbol: str, limit: int = 1) -> Optional[List[Dict[str, Any]]]:
        """Get balance sheet data."""
        try:
            url = f"{self.base_url}/balance-sheet-statement/{symbol}"
            params = {'apikey': self.api_key, 'limit': limit}

            time.sleep(self.rate_limit_delay)
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            return data if isinstance(data, list) else None

        except Exception as e:
            _logger.error(f"Error getting balance sheet for {symbol}: {e}")
            return None

    def get_cash_flow(self, symbol: str, limit: int = 1) -> Optional[List[Dict[str, Any]]]:
        """Get cash flow statement data."""
        try:
            url = f"{self.base_url}/cash-flow-statement/{symbol}"
            params = {'apikey': self.api_key, 'limit': limit}

            time.sleep(self.rate_limit_delay)
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            return data if isinstance(data, list) else None

        except Exception as e:
            _logger.error(f"Error getting cash flow for {symbol}: {e}")
            return None

    def get_enterprise_value(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get enterprise value data."""
        try:
            url = f"{self.base_url}/enterprise-values/{symbol}"
            params = {'apikey': self.api_key, 'limit': 1}

            time.sleep(self.rate_limit_delay)
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if isinstance(data, list) and len(data) > 0:
                return data[0]
            else:
                return None

        except Exception as e:
            _logger.error(f"Error getting enterprise value for {symbol}: {e}")
            return None

    def get_dcf_valuation(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get DCF valuation data."""
        try:
            url = f"{self.base_url}/dcf/{symbol}"
            params = {'apikey': self.api_key, 'limit': 1}

            time.sleep(self.rate_limit_delay)
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if isinstance(data, list) and len(data) > 0:
                return data[0]
            else:
                return None

        except Exception as e:
            _logger.error(f"Error getting DCF valuation for {symbol}: {e}")
            return None

    def test_connection(self) -> bool:
        """Test API connection."""
        try:
            # Test with a simple request
            url = f"{self.base_url}/profile/AAPL"
            params = {'apikey': self.api_key}

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            _logger.info("FMP API connection test successful")
            return True

        except Exception as e:
            _logger.error(f"FMP API connection test failed: {e}")
            return False


def create_fmp_downloader(**kwargs) -> FMPDataDownloader:
    """Factory function to create FMP downloader."""
    return FMPDataDownloader(**kwargs)


# Example usage
if __name__ == "__main__":
    # Test the downloader
    downloader = FMPDataDownloader()

    # Test connection
    if downloader.test_connection():
        print("✅ FMP API connection successful")

        # Test OHLCV data download
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        df = downloader.get_ohlcv('AAPL', '1d', start_date, end_date)
        if df is not None:
            print(f"✅ Downloaded {len(df)} OHLCV data points for AAPL")
            print(df.head())
        else:
            print("❌ Failed to download OHLCV data")

        # Test fundamental data download
        fundamentals = downloader.get_fundamentals('AAPL')
        if fundamentals:
            print(f"✅ Downloaded fundamental data for AAPL")
            print(f"Company: {fundamentals.get('profile', {}).get('companyName', 'N/A')}")
            print(f"Market Cap: {fundamentals.get('metrics', {}).get('marketCapitalization', 'N/A')}")
        else:
            print("❌ Failed to download fundamental data")

        # Test stock screener
        screener_results = downloader.get_stock_screener({
            'marketCapMoreThan': 1000000000,  # > $1B market cap
            'peRatioLessThan': 20,            # P/E < 20
            'limit': 10
        })
        if screener_results:
            print(f"✅ Stock screener returned {len(screener_results)} results")
            for stock in screener_results[:3]:  # Show first 3
                print(f"  - {stock.get('symbol', 'N/A')}: {stock.get('companyName', 'N/A')}")
        else:
            print("❌ Stock screener returned no results")

    else:
        print("❌ FMP API connection failed")