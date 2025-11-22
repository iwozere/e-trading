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
from pathlib import Path
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.notification.logger import setup_logger
from src.data.downloader.base_data_downloader import BaseDataDownloader
from config.donotshare.donotshare import FMP_API_KEY

_logger = setup_logger(__name__)


class FMPDataDownloader(BaseDataDownloader):
    """Financial Modeling Prep (FMP) data downloader."""

    def __init__(self, api_key: Optional[str] = None, rate_limit_delay: float = 0.1):
        """
        Initialize FMP data downloader.

        Args:
            api_key: FMP API key. If None, uses FMP_API_KEY from donotshare.py
            rate_limit_delay: Delay between requests in seconds
        """
        super().__init__()
        self.api_key = api_key or FMP_API_KEY
        self.rate_limit_delay = rate_limit_delay
        #self.base_url = "https://financialmodelingprep.com/api/v3" -- Deprecated v3 endpoints end of life August 31, 2025
        self.stable_url = "https://financialmodelingprep.com/stable"

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
                _logger.error("Unsupported interval: %s", interval)
                return None

            # Format dates
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')

            # Build URL
            if fmp_interval == '1day':
                # Daily data endpoint
                url = f"{self.stable_url}/historical-price-full/{symbol}"
                params = {
                    'apikey': self.api_key,
                    'from': start_str,
                    'to': end_str
                }
            else:
                # Intraday data endpoint
                url = f"{self.stable_url}/historical-chart/{fmp_interval}/{symbol}"
                params = {
                    'apikey': self.api_key,
                    'from': start_str,
                    'to': end_str
                }

            # Add limit if specified
            if limit:
                params['limit'] = min(limit, 1000)  # FMP max limit is 1000

            _logger.info("Fetching %s %s data from %s to %s", symbol, interval, start_str, end_str)

            # Make request with rate limiting
            time.sleep(self.rate_limit_delay)
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Parse response based on endpoint
            if fmp_interval == '1day':
                # Daily data structure: {"historical": [...]}
                if 'historical' not in data:
                    _logger.warning("No historical data found for %s", symbol)
                    return None
                historical_data = data['historical']
            else:
                # Intraday data structure: [...] (direct array)
                if not isinstance(data, list):
                    _logger.warning("No intraday data found for %s", symbol)
                    return None
                historical_data = data

            if not historical_data:
                _logger.warning("No data returned for %s %s", symbol, interval)
                return None

            # Convert to DataFrame
            df = pd.DataFrame(historical_data)

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

    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols (not implemented for FMP)."""
        # FMP doesn't provide a simple symbol list endpoint
        # In practice, you would use common stock symbols
        return []

    def get_supported_intervals(self) -> List[str]:
        """Get list of supported intervals."""
        return ['1m', '5m', '15m', '30m', '1h', '4h', '1d']

    def get_periods(self) -> List[str]:
        """Return list of supported periods for FMP."""
        return ['1d', '7d', '1mo', '3mo', '6mo', '1y', '2y']

    def get_intervals(self) -> List[str]:
        """Return list of supported intervals for FMP."""
        return self.get_supported_intervals()

    def is_valid_period_interval(self, period: str, interval: str) -> bool:
        """Check if the given period and interval combination is valid."""
        return interval in self.get_supported_intervals() and period in self.get_periods()

    def get_stock_screener(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get stock screener results from FMP.

        Args:
            criteria: Screening criteria (e.g., {'marketCapMoreThan': 1000000000, 'peRatioLessThan': 20})

        Returns:
            List of stock data dictionaries
        """
        try:
            url = f"{self.stable_url}/stock-screener"
            params = {
                'apikey': self.api_key,
                **criteria
            }

            _logger.info("Running FMP stock screener with criteria: %s", criteria)

            time.sleep(self.rate_limit_delay)
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if isinstance(data, list):
                _logger.info("FMP screener returned %d stocks", len(data))
                return data
            else:
                _logger.warning("FMP screener returned unexpected data format")
                return []

        except Exception:
            _logger.exception("Error in FMP stock screener:")
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

            _logger.info("Retrieved fundamentals for %d tickers", len(fundamentals))
            return fundamentals

        except Exception:
            _logger.exception("Error getting batch fundamentals:")
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

            _logger.info("Retrieved fundamentals for %s", symbol)
            return fundamentals

        except Exception:
            _logger.exception("Error getting fundamentals for %s:", symbol)
            return None

    def get_company_profile(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get company profile information using the /stable endpoint."""
        try:
            # Use /stable endpoint (v3 endpoints deprecated as of August 31, 2025)
            url = f"{self.stable_url}/profile"
            params = {
                'symbol': symbol,
                'apikey': self.api_key
            }

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

        except Exception:
            _logger.exception("Error getting company profile for %s:", symbol)
            return None

    def get_key_metrics(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get key financial metrics using the /stable endpoint (requires paid subscription)."""
        try:
            # Use /stable endpoint (v3 endpoints deprecated as of August 31, 2025)
            # Note: /stable/key-metrics requires paid subscription
            url = f"{self.stable_url}/key-metrics"
            params = {
                'symbol': symbol,
                'apikey': self.api_key
            }

            time.sleep(self.rate_limit_delay)
            response = requests.get(url, params=params, timeout=30)

            # Gracefully handle 402 Payment Required (premium feature)
            if response.status_code == 402:
                _logger.debug("Key metrics for %s requires premium subscription (402 Payment Required)", symbol)
                return None

            response.raise_for_status()

            data = response.json()

            if isinstance(data, list) and len(data) > 0:
                return data[0]
            elif isinstance(data, dict):
                return data
            else:
                return None

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 402:
                _logger.debug("Key metrics for %s requires premium subscription", symbol)
            else:
                _logger.exception("HTTP error getting key metrics for %s:", symbol)
            return None
        except Exception:
            _logger.exception("Error getting key metrics for %s:", symbol)
            return None

    def get_financial_ratios(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get financial ratios using the /stable endpoint (requires paid subscription)."""
        try:
            # Use /stable endpoint (v3 endpoints deprecated as of August 31, 2025)
            # Note: /stable/ratios requires paid subscription
            url = f"{self.stable_url}/ratios"
            params = {
                'symbol': symbol,
                'apikey': self.api_key
            }

            time.sleep(self.rate_limit_delay)
            response = requests.get(url, params=params, timeout=30)

            # Gracefully handle 402 Payment Required (premium feature)
            if response.status_code == 402:
                _logger.debug("Financial ratios for %s requires premium subscription (402 Payment Required)", symbol)
                return None

            response.raise_for_status()

            data = response.json()

            if isinstance(data, list) and len(data) > 0:
                return data[0]
            elif isinstance(data, dict):
                return data
            else:
                return None

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 402:
                _logger.debug("Financial ratios for %s requires premium subscription", symbol)
            else:
                _logger.exception("HTTP error getting financial ratios for %s:", symbol)
            return None
        except Exception:
            _logger.exception("Error getting financial ratios for %s:", symbol)
            return None

    def get_income_statement(self, symbol: str, limit: int = 1) -> Optional[List[Dict[str, Any]]]:
        """Get income statement data."""
        try:
            url = f"{self.stable_url}/income-statement/{symbol}"
            params = {'apikey': self.api_key, 'limit': limit}

            time.sleep(self.rate_limit_delay)
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            return data if isinstance(data, list) else None

        except Exception:
            _logger.exception("Error getting income statement for %s:", symbol)
            return None

    def get_balance_sheet(self, symbol: str, limit: int = 1) -> Optional[List[Dict[str, Any]]]:
        """Get balance sheet data."""
        try:
            url = f"{self.stable_url}/balance-sheet-statement/{symbol}"
            params = {'apikey': self.api_key, 'limit': limit}

            time.sleep(self.rate_limit_delay)
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            return data if isinstance(data, list) else None

        except Exception:
            _logger.exception("Error getting balance sheet for %s:", symbol)
            return None

    def get_cash_flow(self, symbol: str, limit: int = 1) -> Optional[List[Dict[str, Any]]]:
        """Get cash flow statement data."""
        try:
            url = f"{self.stable_url}/cash-flow-statement/{symbol}"
            params = {'apikey': self.api_key, 'limit': limit}

            time.sleep(self.rate_limit_delay)
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            return data if isinstance(data, list) else None

        except Exception:
            _logger.exception("Error getting cash flow for %s:", symbol)
            return None

    def get_enterprise_value(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get enterprise value data."""
        try:
            url = f"{self.stable_url}/enterprise-values/{symbol}"
            params = {'apikey': self.api_key, 'limit': 1}

            time.sleep(self.rate_limit_delay)
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if isinstance(data, list) and len(data) > 0:
                return data[0]
            else:
                return None

        except Exception:
            _logger.exception("Error getting enterprise value for %s:", symbol)
            return None

    def get_dcf_valuation(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get DCF valuation data."""
        try:
            url = f"{self.stable_url}/dcf/{symbol}"
            params = {'apikey': self.api_key, 'limit': 1}

            time.sleep(self.rate_limit_delay)
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if isinstance(data, list) and len(data) > 0:
                return data[0]
            else:
                return None

        except Exception:
            _logger.exception("Error getting DCF valuation for %s:", symbol)
            return None

    # Short Squeeze Detection Pipeline Extensions

    def get_short_interest_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get short interest data for a symbol.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')

        Returns:
            Dictionary with short interest data or None if failed
        """
        try:
            url = f"{self.stable_url}/short-interest/{symbol}"
            params = {'apikey': self.api_key}

            _logger.debug("Fetching short interest data for %s", symbol)

            time.sleep(self.rate_limit_delay)
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if isinstance(data, list) and len(data) > 0:
                # Return the most recent short interest data
                short_data = data[0]
                _logger.debug("Retrieved short interest data for %s: %s shares short",
                            symbol, short_data.get('shortInterest', 'N/A'))
                return short_data
            else:
                _logger.warning("No short interest data found for %s", symbol)
                return None

        except Exception:
            _logger.exception("Error getting short interest data for %s:", symbol)
            return None

    def get_float_shares_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get float shares data for a symbol from company profile.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')

        Returns:
            Dictionary with float shares data or None if failed
        """
        try:
            # Float shares are typically available in the company profile
            profile = self.get_company_profile(symbol)
            if not profile:
                return None

            # Extract float-related data
            float_data = {
                'symbol': symbol,
                'sharesOutstanding': profile.get('sharesOutstanding'),
                'floatShares': profile.get('floatShares'),
                'marketCap': profile.get('mktCap'),
                'lastUpdated': profile.get('lastUpdated', datetime.now().isoformat())
            }

            _logger.debug("Retrieved float shares data for %s: %s float shares",
                        symbol, float_data.get('floatShares', 'N/A'))
            return float_data

        except Exception:
            _logger.exception("Error getting float shares data for %s:", symbol)
            return None

    def get_market_cap_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get market capitalization data for a symbol.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')

        Returns:
            Dictionary with market cap data or None if failed
        """
        try:
            profile = self.get_company_profile(symbol)
            if not profile:
                return None

            market_cap_data = {
                'symbol': symbol,
                'marketCap': profile.get('mktCap'),
                'price': profile.get('price'),
                'sharesOutstanding': profile.get('sharesOutstanding'),
                'lastUpdated': profile.get('lastUpdated', datetime.now().isoformat())
            }

            _logger.debug("Retrieved market cap data for %s: $%s",
                        symbol, market_cap_data.get('marketCap', 'N/A'))
            return market_cap_data

        except Exception:
            _logger.exception("Error getting market cap data for %s:", symbol)
            return None

    def load_universe_from_screener(self, criteria: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Load universe of stocks from FMP stock screener for short squeeze analysis.

        Args:
            criteria: Optional screening criteria. If None, uses default criteria for short squeeze detection

        Returns:
            List of ticker symbols that meet the criteria
        """
        try:
            # Default criteria for short squeeze detection
            default_criteria = {
                'marketCapMoreThan': 100_000_000,  # > $100M market cap
                'marketCapLowerThan': 10_000_000_000,  # < $10B market cap
                'volumeMoreThan': 200_000,  # > 200k average volume
                'exchange': 'NYSE,NASDAQ',  # Major US exchanges
                'limit': 1000  # Maximum results
            }

            # Use provided criteria or defaults
            screening_criteria = criteria or default_criteria

            _logger.info("Loading universe with criteria: %s", screening_criteria)

            # Get screener results
            screener_results = self.get_stock_screener(screening_criteria)

            if not screener_results:
                _logger.warning("No stocks returned from screener")
                return []

            # Extract ticker symbols
            tickers = []
            for stock in screener_results:
                symbol = stock.get('symbol')
                if symbol:
                    tickers.append(symbol)

            _logger.info("Loaded universe of %d stocks from FMP screener", len(tickers))
            return tickers

        except Exception:
            _logger.exception("Error loading universe from screener:")
            return []

    def get_short_squeeze_batch_data(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get short squeeze relevant data for multiple tickers in batch.

        Args:
            tickers: List of ticker symbols

        Returns:
            Dictionary mapping ticker symbols to their short squeeze data
        """
        try:
            batch_data = {}
            total_tickers = len(tickers)

            _logger.info("Fetching short squeeze data for %d tickers", total_tickers)

            for i, ticker in enumerate(tickers, 1):
                try:
                    # Get company profile (includes market cap, float shares)
                    profile = self.get_company_profile(ticker)

                    # Get short interest data
                    short_interest = self.get_short_interest_data(ticker)

                    # Get key metrics for additional data
                    metrics = self.get_key_metrics(ticker)

                    # Combine all data
                    ticker_data = {
                        'symbol': ticker,
                        'profile': profile,
                        'shortInterest': short_interest,
                        'metrics': metrics,
                        'timestamp': datetime.now().isoformat()
                    }

                    batch_data[ticker] = ticker_data

                    if i % 50 == 0:  # Log progress every 50 tickers
                        _logger.info("Processed %d/%d tickers", i, total_tickers)

                except Exception as e:
                    _logger.warning("Failed to get data for ticker %s: %s", ticker, e)
                    continue

            _logger.info("Successfully retrieved data for %d/%d tickers", len(batch_data), total_tickers)
            return batch_data

        except Exception:
            _logger.exception("Error in batch data retrieval:")
            return {}

    def test_connection(self) -> bool:
        """Test API connection."""
        try:
            # Test with a simple request
            url = f"{self.stable_url}/profile/AAPL"
            params = {'apikey': self.api_key}

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            _logger.info("FMP API connection test successful")
            return True

        except Exception:
            _logger.exception("FMP API connection test failed:")
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
            print("✅ Downloaded fundamental data for AAPL")
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