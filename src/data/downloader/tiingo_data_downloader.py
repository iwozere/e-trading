#!/usr/bin/env python3
"""
Tiingo Data Downloader

This module provides data downloading capabilities from Tiingo API.
Tiingo offers comprehensive historical data back to 1962 for active tickers.

API Documentation: https://api.tiingo.com/documentation
Free API Key: https://www.tiingo.com/account/api/token
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.notification.logger import setup_logger
from src.data.downloader.base_data_downloader import BaseDataDownloader

_logger = setup_logger(__name__)


class TiingoDataDownloader(BaseDataDownloader):
    """Tiingo data downloader."""

    def __init__(self, api_key: Optional[str] = None, rate_limit_delay: float = 0.1):
        """
        Initialize Tiingo data downloader.

        Args:
            api_key: Tiingo API key. If None, uses TIINGO_API_KEY from donotshare.py
            rate_limit_delay: Delay between requests in seconds
        """
        super().__init__()
        self.api_key = api_key or self._get_config_value('TIINGO_API_KEY', 'TIINGO_API_KEY')
        self.rate_limit_delay = rate_limit_delay
        self.base_url = "https://api.tiingo.com/tiingo"

        if not self.api_key:
            raise ValueError("Tiingo API key is required. Get one at: https://www.tiingo.com/account/api/token")

        _logger.info("Tiingo Data Downloader initialized with API key")

    def get_ohlcv(self, symbol: str, interval: str, start_date: datetime,
                   end_date: datetime, limit: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Get OHLCV data from Tiingo.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            interval: Time interval ('1d', '1w', '1m') - Tiingo supports daily, weekly, monthly
            start_date: Start date for data
            end_date: End date for data
            limit: Maximum number of data points (optional)

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            # Convert interval to Tiingo format
            tiingo_interval = self._convert_interval(interval)
            if not tiingo_interval:
                _logger.error("Unsupported interval: %s. Tiingo supports: 1d, 1w, 1m", interval)
                return None

            # Format dates
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')

            # Build URL
            url = f"{self.base_url}/daily/{symbol}/prices"
            params = {
                'startDate': start_str,
                'endDate': end_str,
                'resampleFreq': tiingo_interval,
                'format': 'json'
            }

            # Add limit if specified
            if limit:
                params['limit'] = min(limit, 10000)  # Tiingo max limit is 10,000

            _logger.info("Fetching %s %s data from%s to %s", symbol, interval, start_str, end_str)

            # Make request with rate limiting
            time.sleep(self.rate_limit_delay)
            headers = {
                'Authorization': f'Token {self.api_key}',
                'Content-Type': 'application/json'
            }
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()

            data = response.json()

            if not data:
                _logger.warning("No data returned for %s %s", symbol, interval)
                return None

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Standardize column names
            df = self._standardize_dataframe(df, symbol, interval)

            _logger.info("Successfully fetched %d data points for %s %s", len(df), symbol, interval)
            return df

        except requests.exceptions.RequestException:
            _logger.exception("Request failed for %s %s:", symbol, interval)
            return None
        except Exception:
            _logger.exception("Error fetching data for %s %s:", symbol, interval)
            return None

    def _convert_interval(self, interval: str) -> Optional[str]:
        """Convert standard interval to Tiingo format."""
        interval_map = {
            '1d': 'daily',
            '1w': 'weekly',
            '1m': 'monthly'
        }
        return interval_map.get(interval)

    def _standardize_dataframe(self, df: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
        """Standardize DataFrame to common OHLCV format."""
        try:
            # Tiingo column mapping
            column_mapping = {
                'date': 'timestamp',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
                'adjOpen': 'adj_open',
                'adjHigh': 'adj_high',
                'adjLow': 'adj_low',
                'adjClose': 'adj_close',
                'adjVolume': 'adj_volume'
            }

            # Rename columns
            df = df.rename(columns=column_mapping)

            # Ensure required columns exist
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    _logger.error("Missing required column: %s", col)
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

        except Exception:
            _logger.exception("Error standardizing DataFrame for %s %s:", symbol, interval)
            return pd.DataFrame()

    def get_company_metadata(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get company metadata from Tiingo.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')

        Returns:
            Dictionary with company metadata or None if failed
        """
        try:
            url = f"{self.base_url}/daily/{symbol}"
            headers = {
                'Authorization': f'Token {self.api_key}',
                'Content-Type': 'application/json'
            }

            time.sleep(self.rate_limit_delay)
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            data = response.json()

            _logger.info("Retrieved metadata for %s", symbol)
            return data

        except Exception:
            _logger.exception("Error getting metadata for %s:", symbol)
            return None

    def get_fundamentals(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get fundamental data for a ticker from Tiingo.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')

        Returns:
            Dictionary with fundamental data or None if failed
        """
        try:
            # Get company metadata (includes some fundamental data)
            metadata = self.get_company_metadata(symbol)
            if not metadata:
                return None

            # Get additional fundamental data if available
            fundamentals = {
                'symbol': symbol,
                'name': metadata.get('name'),
                'description': metadata.get('description'),
                'startDate': metadata.get('startDate'),
                'endDate': metadata.get('endDate'),
                'exchangeCode': metadata.get('exchangeCode'),
                'ticker': metadata.get('ticker')
            }

            _logger.info("Retrieved fundamentals for %s", symbol)
            return fundamentals

        except Exception:
            _logger.exception("Error getting fundamentals for %s:", symbol)
            return None

    def get_income_statement(self, symbol: str, start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        """
        Get income statement data from Tiingo.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date for data (optional)
            end_date: End date for data (optional)

        Returns:
            Dictionary with income statement data or None if failed
        """
        try:
            url = f"{self.base_url}/fundamentals/{symbol}/daily"
            params = {'format': 'json'}

            if start_date:
                params['startDate'] = start_date.strftime('%Y-%m-%d')
            if end_date:
                params['endDate'] = end_date.strftime('%Y-%m-%d')

            headers = {
                'Authorization': f'Token {self.api_key}',
                'Content-Type': 'application/json'
            }

            time.sleep(self.rate_limit_delay)
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()

            data = response.json()

            _logger.info("Retrieved income statement for %s", symbol)
            return data

        except Exception:
            _logger.exception("Error getting income statement for %s:", symbol)
            return None

    def get_balance_sheet(self, symbol: str, start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        """
        Get balance sheet data from Tiingo.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date for data (optional)
            end_date: End date for data (optional)

        Returns:
            Dictionary with balance sheet data or None if failed
        """
        try:
            url = f"{self.base_url}/fundamentals/{symbol}/daily"
            params = {'format': 'json'}

            if start_date:
                params['startDate'] = start_date.strftime('%Y-%m-%d')
            if end_date:
                params['endDate'] = end_date.strftime('%Y-%m-%d')

            headers = {
                'Authorization': f'Token {self.api_key}',
                'Content-Type': 'application/json'
            }

            time.sleep(self.rate_limit_delay)
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()

            data = response.json()

            _logger.info("Retrieved balance sheet for %s", symbol)
            return data

        except Exception:
            _logger.exception("Error getting balance sheet for %s:", symbol)
            return None

    def get_cash_flow(self, symbol: str, start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        """
        Get cash flow statement data from Tiingo.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date for data (optional)
            end_date: End date for data (optional)

        Returns:
            Dictionary with cash flow data or None if failed
        """
        try:
            url = f"{self.base_url}/fundamentals/{symbol}/daily"
            params = {'format': 'json'}

            if start_date:
                params['startDate'] = start_date.strftime('%Y-%m-%d')
            if end_date:
                params['endDate'] = end_date.strftime('%Y-%m-%d')

            headers = {
                'Authorization': f'Token {self.api_key}',
                'Content-Type': 'application/json'
            }

            time.sleep(self.rate_limit_delay)
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()

            data = response.json()

            _logger.info("Retrieved cash flow for %s", symbol)
            return data

        except Exception:
            _logger.exception("Error getting cash flow for %s:", symbol)
            return None

    def get_financial_ratios(self, symbol: str, start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        """
        Get financial ratios data from Tiingo.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date for data (optional)
            end_date: End date for data (optional)

        Returns:
            Dictionary with financial ratios or None if failed
        """
        try:
            url = f"{self.base_url}/fundamentals/{symbol}/daily"
            params = {'format': 'json'}

            if start_date:
                params['startDate'] = start_date.strftime('%Y-%m-%d')
            if end_date:
                params['endDate'] = end_date.strftime('%Y-%m-%d')

            headers = {
                'Authorization': f'Token {self.api_key}',
                'Content-Type': 'application/json'
            }

            time.sleep(self.rate_limit_delay)
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()

            data = response.json()

            _logger.info("Retrieved financial ratios for %s", symbol)
            return data

        except Exception:
            _logger.exception("Error getting financial ratios for %s:", symbol)
            return None

    def get_fundamentals_batch(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Get fundamental data for multiple tickers from Tiingo.

        Args:
            symbols: List of stock symbols (e.g., ['AAPL', 'GOOGL'])

        Returns:
            Dictionary mapping symbols to fundamental data
        """
        results = {}

        for symbol in symbols:
            try:
                fundamentals = self.get_fundamentals(symbol)
                if fundamentals:
                    results[symbol] = fundamentals
                else:
                    _logger.warning("No fundamental data available for %s", symbol)

            except Exception:
                _logger.exception("Error getting fundamentals for %s:", symbol)
                continue

        _logger.info("Retrieved fundamentals for %d out of %d symbols", len(results), len(symbols))
        return results

    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols from Tiingo."""
        try:
            url = f"{self.base_url}/daily"
            headers = {
                'Authorization': f'Token {self.api_key}',
                'Content-Type': 'application/json'
            }

            time.sleep(self.rate_limit_delay)
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            data = response.json()

            if isinstance(data, list):
                symbols = [item.get('ticker', '') for item in data if item.get('ticker')]
                _logger.info("Retrieved %d available symbols", len(symbols))
                return symbols
            else:
                return []

        except Exception:
            _logger.exception("Error getting available symbols:")
            return []

    def get_supported_intervals(self) -> List[str]:
        """Get list of supported intervals."""
        return ['1d', '1w', '1m']  # Tiingo supports daily, weekly, monthly

    def get_periods(self) -> List[str]:
        """Return list of supported periods for Tiingo."""
        return ['1d', '7d', '1mo', '3mo', '6mo', '1y', '2y']

    def get_intervals(self) -> List[str]:
        """Return list of supported intervals for Tiingo."""
        return self.get_supported_intervals()

    def is_valid_period_interval(self, period: str, interval: str) -> bool:
        """Check if the given period and interval combination is valid."""
        return interval in self.get_supported_intervals() and period in self.get_periods()

    def test_connection(self) -> bool:
        """Test API connection."""
        try:
            # Test with a simple request
            url = f"{self.base_url}/daily/AAPL"
            headers = {
                'Authorization': f'Token {self.api_key}',
                'Content-Type': 'application/json'
            }

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            _logger.info("Tiingo API connection test successful")
            return True

        except Exception:
            _logger.exception("Tiingo API connection test failed:")
            return False


def create_tiingo_downloader(**kwargs) -> TiingoDataDownloader:
    """Factory function to create Tiingo downloader."""
    return TiingoDataDownloader(**kwargs)


# Example usage
if __name__ == "__main__":
    # Test the downloader
    downloader = TiingoDataDownloader()

    # Test connection
    if downloader.test_connection():
        print("✅ Tiingo API connection successful")

        # Test data download
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1 year of data

        df = downloader.get_ohlcv('AAPL', '1d', start_date, end_date)
        if df is not None:
            print(f"✅ Downloaded {len(df)} data points for AAPL")
            print(df.head())
        else:
            print("❌ Failed to download data")

        # Test fundamental data
        fundamentals = downloader.get_fundamentals('AAPL')
        if fundamentals:
            print("✅ Downloaded fundamental data for AAPL")
            print(f"Company: {fundamentals.get('name', 'N/A')}")
            print(f"Exchange: {fundamentals.get('exchangeCode', 'N/A')}")
        else:
            print("❌ Failed to download fundamental data")

        # Test financial statements
        print("\n--- Testing Financial Statements ---")

        # Income statement
        income_stmt = downloader.get_income_statement('AAPL')
        if income_stmt:
            print("✅ Downloaded income statement")
        else:
            print("❌ Failed to download income statement")

        # Balance sheet
        balance_sheet = downloader.get_balance_sheet('AAPL')
        if balance_sheet:
            print("✅ Downloaded balance sheet")
        else:
            print("❌ Failed to download balance sheet")

        # Cash flow
        cash_flow = downloader.get_cash_flow('AAPL')
        if cash_flow:
            print("✅ Downloaded cash flow statement")
        else:
            print("❌ Failed to download cash flow statement")

        # Financial ratios
        ratios = downloader.get_financial_ratios('AAPL')
        if ratios:
            print("✅ Downloaded financial ratios")
        else:
            print("❌ Failed to download financial ratios")

    else:
        print("❌ Tiingo API connection failed")
