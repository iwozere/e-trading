"""
Financial Modeling Prep (FMP) Data Downloader

This module provides integration with the Financial Modeling Prep API for:
- Stock screening with professional criteria
- Fundamental data retrieval
- Company information
- Financial statements
"""

import os
import requests
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import pandas as pd

from src.data.base_data_downloader import BaseDataDownloader
from src.common.fundamentals import Fundamentals
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class FMPDataDownloader(BaseDataDownloader):
    """
    Financial Modeling Prep (FMP) Data Downloader

    Provides access to FMP's professional stock screening and fundamental data.
    Supports single API calls for stock screening with sophisticated criteria.
    """

    def __init__(self, api_key: Optional[str] = None, data_dir: str = "data"):
        """
        Initialize FMP Data Downloader.

        Args:
            api_key: FMP API key. If not provided, will try to get from environment variable FMP_API_KEY
            data_dir: Directory to store downloaded data
        """
        super().__init__(data_dir)

        # Get API key from parameter or environment variable
        self.api_key = api_key or os.getenv('FMP_API_KEY')
        if not self.api_key:
            raise ValueError("FMP API key is required. Set FMP_API_KEY environment variable or pass api_key parameter.")

        self.base_url = "https://financialmodelingprep.com/api/v3"
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests (FMP rate limit)

        _logger.info("FMP Data Downloader initialized")

    def _rate_limit(self):
        """Apply rate limiting to respect FMP API limits."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Make a request to the FMP API with rate limiting and error handling.

        Args:
            endpoint: API endpoint (e.g., '/stock-screener')
            params: Query parameters

        Returns:
            API response as dictionary

        Raises:
            Exception: If API request fails
        """
        self._rate_limit()

        url = f"{self.base_url}{endpoint}"
        params = params or {}
        params['apikey'] = self.api_key

        try:
            _logger.debug("Making FMP API request to: %s", url)
            response = requests.get(url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                _logger.debug("FMP API request successful, received %d items", len(data) if isinstance(data, list) else 1)
                return data
            elif response.status_code == 429:
                _logger.error("FMP API rate limit exceeded")
                raise Exception("FMP API rate limit exceeded. Please wait before making another request.")
            elif response.status_code == 401:
                _logger.error("FMP API authentication failed. Check your API key.")
                raise Exception("FMP API authentication failed. Please check your API key.")
            else:
                _logger.error("FMP API request failed with status %d: %s", response.status_code, response.text)
                raise Exception(f"FMP API request failed with status {response.status_code}: {response.text}")

        except requests.exceptions.RequestException as e:
            _logger.error("Network error during FMP API request: %s", str(e))
            raise Exception(f"Network error during FMP API request: {str(e)}")

    def get_stock_screener(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get undervalued stocks using FMP's professional stock screener.

        This is a single API call that returns pre-filtered stocks based on sophisticated criteria.

        Args:
            criteria: Dictionary containing screening criteria
                     Supported criteria:
                     - marketCapMoreThan/marketCapLowerThan: Market capitalization filters
                     - peRatioLessThan/peRatioMoreThan: P/E ratio filters
                     - priceToBookRatioLessThan/priceToBookRatioMoreThan: P/B ratio filters
                     - priceToSalesRatioLessThan/priceToSalesRatioMoreThan: P/S ratio filters
                     - debtToEquityLessThan: Debt to equity ratio filter
                     - currentRatioMoreThan: Current ratio filter
                     - quickRatioMoreThan: Quick ratio filter
                     - returnOnEquityMoreThan: ROE filter
                     - returnOnAssetsMoreThan: ROA filter
                     - returnOnCapitalEmployedMoreThan: ROCE filter
                     - dividendYieldMoreThan/dividendYieldLessThan: Dividend yield filters
                     - betaLessThan/betaMoreThan: Beta filters
                     - exchange: Exchange filter (NYSE, NASDAQ, etc.)
                     - limit: Maximum number of results

        Returns:
            List of dictionaries containing stock data that meets the criteria

        Example:
            criteria = {
                "marketCapMoreThan": 1000000000,  # $1B+ market cap
                "peRatioLessThan": 15,
                "priceToBookRatioLessThan": 1.5,
                "debtToEquityLessThan": 0.5,
                "returnOnEquityMoreThan": 0.12,
                "limit": 50
            }
        """
        try:
            _logger.info("Running FMP stock screener with criteria: %s", criteria)

            # Validate criteria
            self._validate_screener_criteria(criteria)

            # Make API request
            results = self._make_request('/stock-screener', criteria)

            if not isinstance(results, list):
                _logger.error("Unexpected response format from FMP screener")
                return []

            _logger.info("FMP screener returned %d stocks", len(results))
            return results

        except Exception as e:
            _logger.error("Error in FMP stock screener: %s", str(e))
            raise

    def _validate_screener_criteria(self, criteria: Dict[str, Any]):
        """Validate screener criteria to ensure they are supported by FMP API."""
        supported_criteria = {
            # Market Cap
            "marketCapMoreThan", "marketCapLowerThan",

            # Valuation
            "peRatioLessThan", "peRatioMoreThan",
            "priceToBookRatioLessThan", "priceToBookRatioMoreThan",
            "priceToSalesRatioLessThan", "priceToSalesRatioMoreThan",
            "priceToCashFlowRatioLessThan", "priceToCashFlowRatioMoreThan",
            "priceToFreeCashFlowRatioLessThan", "priceToFreeCashFlowRatioMoreThan",

            # Financial Health
            "debtToEquityLessThan", "debtToEquityMoreThan",
            "currentRatioMoreThan", "currentRatioLessThan",
            "quickRatioMoreThan", "quickRatioLessThan",
            "cashRatioMoreThan", "cashRatioLessThan",

            # Profitability
            "returnOnEquityMoreThan", "returnOnEquityLessThan",
            "returnOnAssetsMoreThan", "returnOnAssetsLessThan",
            "returnOnCapitalEmployedMoreThan", "returnOnCapitalEmployedLessThan",
            "returnOnTangibleAssetsMoreThan", "returnOnTangibleAssetsLessThan",

            # Dividends
            "dividendYieldMoreThan", "dividendYieldLessThan",
            "payoutRatioLessThan", "payoutRatioMoreThan",

            # Risk
            "betaLessThan", "betaMoreThan",

            # Exchange
            "exchange",

            # Limit
            "limit"
        }

        invalid_criteria = set(criteria.keys()) - supported_criteria
        if invalid_criteria:
            _logger.warning("Unsupported criteria provided: %s", invalid_criteria)

    def get_fundamentals(self, symbol: str) -> Fundamentals:
        """
        Get comprehensive fundamental data for a single stock.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')

        Returns:
            Fundamentals object with comprehensive data
        """
        try:
            _logger.debug("Getting fundamentals for %s", symbol)

            # Get company profile
            profile_data = self._make_request(f'/profile/{symbol}')
            if not profile_data or not isinstance(profile_data, list) or len(profile_data) == 0:
                _logger.warning("No profile data found for %s", symbol)
                return self._create_default_fundamentals(symbol)

            profile = profile_data[0]

            # Get key metrics
            metrics_data = self._make_request(f'/key-metrics/{symbol}')
            metrics = metrics_data[0] if metrics_data and len(metrics_data) > 0 else {}

            # Get financial ratios
            ratios_data = self._make_request(f'/ratios/{symbol}')
            ratios = ratios_data[0] if ratios_data and len(ratios_data) > 0 else {}

            # Create Fundamentals object
            fundamentals = Fundamentals(
                ticker=symbol.upper(),
                company_name=profile.get('companyName', 'Unknown'),
                current_price=profile.get('price', 0.0),
                market_cap=profile.get('mktCap', 0.0),
                pe_ratio=ratios.get('peRatio', 0.0),
                forward_pe=ratios.get('forwardPeRatio', 0.0),
                dividend_yield=profile.get('lastDiv', 0.0) / profile.get('price', 1.0) if profile.get('price', 0) > 0 else 0.0,
                earnings_per_share=ratios.get('eps', 0.0),
                price_to_book=ratios.get('priceToBookRatio', 0.0),
                return_on_equity=ratios.get('returnOnEquity', 0.0),
                return_on_assets=ratios.get('returnOnAssets', 0.0),
                debt_to_equity=ratios.get('debtEquityRatio', 0.0),
                current_ratio=ratios.get('currentRatio', 0.0),
                quick_ratio=ratios.get('quickRatio', 0.0),
                revenue=profile.get('revenue', 0.0),
                revenue_growth=metrics.get('revenueGrowth', 0.0),
                net_income=profile.get('netIncome', 0.0),
                net_income_growth=metrics.get('netIncomeGrowth', 0.0),
                free_cash_flow=profile.get('freeCashFlow', 0.0),
                operating_margin=ratios.get('operatingMargin', 0.0),
                profit_margin=ratios.get('netProfitMargin', 0.0),
                beta=profile.get('beta', 0.0),
                sector=profile.get('sector', 'Unknown'),
                industry=profile.get('industry', 'Unknown'),
                country=profile.get('country', 'Unknown'),
                exchange=profile.get('exchange', 'Unknown'),
                currency=profile.get('currency', 'USD'),
                shares_outstanding=profile.get('sharesOutstanding', 0.0),
                float_shares=profile.get('sharesFloat', 0.0),
                short_ratio=profile.get('sharesShort', 0.0) / profile.get('sharesFloat', 1.0) if profile.get('sharesFloat', 0) > 0 else 0.0,
                payout_ratio=ratios.get('dividendPayoutRatio', 0.0),
                peg_ratio=ratios.get('pegRatio', 0.0),
                price_to_sales=ratios.get('priceToSalesRatio', 0.0),
                enterprise_value=profile.get('enterpriseValue', 0.0),
                enterprise_value_to_ebitda=ratios.get('enterpriseValueMultiple', 0.0),
                data_source="Financial Modeling Prep",
                last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )

            _logger.debug("Retrieved fundamentals for %s: %s", symbol, fundamentals.company_name)
            return fundamentals

        except Exception as e:
            _logger.error("Error getting fundamentals for %s: %s", symbol, str(e))
            return self._create_default_fundamentals(symbol)

    def get_fundamentals_batch(self, symbols: List[str]) -> Dict[str, Fundamentals]:
        """
        Get fundamental data for multiple symbols using batch processing.

        Args:
            symbols: List of stock symbols

        Returns:
            Dictionary mapping symbols to Fundamentals objects
        """
        results = {}

        for symbol in symbols:
            try:
                fundamentals = self.get_fundamentals(symbol)
                results[symbol] = fundamentals
            except Exception as e:
                _logger.error("Error getting fundamentals for %s: %s", symbol, str(e))
                results[symbol] = self._create_default_fundamentals(symbol)

        return results

    def get_ohlcv(self, symbol: str, interval: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Get OHLCV data for a stock.

        Args:
            symbol: Stock symbol
            interval: Data interval ('1min', '5min', '15min', '30min', '1hour', '4hour', '1day')
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Convert interval to FMP format
            fmp_interval = self._convert_interval_to_fmp(interval)

            # Format dates
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')

            # Make API request
            endpoint = f'/historical-price-full/{symbol}'
            params = {
                'from': start_str,
                'to': end_str,
                'timeseries': fmp_interval
            }

            data = self._make_request(endpoint, params)

            if not data or 'historical' not in data:
                _logger.warning("No OHLCV data found for %s", symbol)
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # Convert to DataFrame
            df = pd.DataFrame(data['historical'])
            df['timestamp'] = pd.to_datetime(df['date'])
            df = df.rename(columns={
                'date': 'timestamp',
                'adjClose': 'close'
            })

            # Select and reorder columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df = df.sort_values('timestamp')

            return df

        except Exception as e:
            _logger.error("Error getting OHLCV data for %s: %s", symbol, str(e))
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    def _convert_interval_to_fmp(self, interval: str) -> str:
        """Convert standard interval format to FMP format."""
        interval_mapping = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1h': '1hour',
            '4h': '4hour',
            '1d': '1day'
        }
        return interval_mapping.get(interval, '1day')

    def get_company_profile(self, symbol: str) -> Dict[str, Any]:
        """
        Get detailed company profile information.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with company profile data
        """
        try:
            data = self._make_request(f'/profile/{symbol}')
            return data[0] if data and len(data) > 0 else {}
        except Exception as e:
            _logger.error("Error getting company profile for %s: %s", symbol, str(e))
            return {}

    def get_financial_ratios(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive financial ratios.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with financial ratios
        """
        try:
            data = self._make_request(f'/ratios/{symbol}')
            return data[0] if data and len(data) > 0 else {}
        except Exception as e:
            _logger.error("Error getting financial ratios for %s: %s", symbol, str(e))
            return {}

    def get_key_metrics(self, symbol: str) -> Dict[str, Any]:
        """
        Get key financial metrics.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with key metrics
        """
        try:
            data = self._make_request(f'/key-metrics/{symbol}')
            return data[0] if data and len(data) > 0 else {}
        except Exception as e:
            _logger.error("Error getting key metrics for %s: %s", symbol, str(e))
            return {}

    def _create_default_fundamentals(self, symbol: str) -> Fundamentals:
        """Create a default Fundamentals object for failed downloads."""
        return Fundamentals(
            ticker=symbol.upper(),
            company_name="Unknown",
            current_price=0.0,
            market_cap=0.0,
            pe_ratio=0.0,
            forward_pe=0.0,
            dividend_yield=0.0,
            earnings_per_share=0.0,
            data_source="Financial Modeling Prep",
            last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

    def get_periods(self) -> List[str]:
        """Get supported time periods."""
        return ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]

    def get_intervals(self) -> List[str]:
        """Get supported time intervals."""
        return ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]

    def is_valid_period_interval(self, period: str, interval: str) -> bool:
        """Check if the period and interval combination is valid."""
        return period in self.get_periods() and interval in self.get_intervals()
