"""
Data Manager Module
-------------------

This module provides the main DataManager class that serves as the unified facade
for all data operations in the E-Trading system. It orchestrates data retrieval,
caching, provider selection, and live feed management.

The DataManager implements the architecture described in REFACTOR.md:
- Single entry point for all data requests
- Provider-agnostic caching with UnifiedCache
- Intelligent provider selection with failover
- Integration with live data feeds
- Centralized error handling and retry logic

Classes:
- DataManager: Main facade for all data operations
"""

import os
import re
import yaml
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd

from src.data.cache.fundamentals_cache import get_fundamentals_cache
from src.data.cache.fundamentals_combiner import get_fundamentals_combiner


# Custom exception classes for enhanced error handling
class RateLimitException(Exception):
    """Exception raised when rate limits are exceeded."""
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class TimeoutException(Exception):
    """Exception raised when requests timeout."""
    pass


class NetworkException(Exception):
    """Exception raised for network-related errors."""
    pass
from src.notification.logger import setup_logger

# Initialize logger
_logger = setup_logger(__name__)

# Import API keys from donotshare configuration
try:
    from config.donotshare.donotshare import (
        ALPHA_VANTAGE_API_KEY,
        FMP_API_KEY,
        POLYGON_API_KEY,
        TWELVE_DATA_API_KEY,
        FINNHUB_API_KEY,
        TIINGO_API_KEY,
        ALPACA_API_KEY,
        ALPACA_SECRET_KEY,
        DATA_CACHE_DIR
    )
except ImportError:
    # Fallback to environment variables if donotshare is not available
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
    FMP_API_KEY = os.getenv('FMP_API_KEY')
    POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
    TWELVE_DATA_API_KEY = os.getenv('TWELVE_DATA_API_KEY')
    FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')
    TIINGO_API_KEY = os.getenv('TIINGO_API_KEY')
    ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
    ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
    DATA_CACHE_DIR = os.getenv('DATA_CACHE_DIR', 'c:/data-cache')  # Fallback if import fails

# Import cache and utilities
from src.data.cache.unified_cache import UnifiedCache
from src.data.utils.rate_limiting import RateLimiter
from src.data.utils.retry import retry_on_exception
from src.data.utils.validation import validate_ohlcv_data

# Import downloaders
from src.data.downloader.base_data_downloader import BaseDataDownloader
from src.data.downloader.binance_data_downloader import BinanceDataDownloader
from src.data.downloader.yahoo_data_downloader import YahooDataDownloader
from src.data.downloader.alpha_vantage_data_downloader import AlphaVantageDataDownloader
from src.data.downloader.fmp_data_downloader import FMPDataDownloader
from src.data.downloader.tiingo_data_downloader import TiingoDataDownloader
from src.data.downloader.polygon_data_downloader import PolygonDataDownloader
from src.data.downloader.twelvedata_data_downloader import TwelveDataDataDownloader
from src.data.downloader.finnhub_data_downloader import FinnhubDataDownloader
from src.data.downloader.coingecko_data_downloader import CoinGeckoDataDownloader
from src.data.downloader.alpaca_data_downloader import AlpacaDataDownloader


# Import live feeds
from src.data.feed.base_live_data_feed import BaseLiveDataFeed
from src.data.feed.binance_live_feed import BinanceLiveDataFeed
from src.data.feed.yahoo_live_feed import YahooLiveDataFeed
from src.data.feed.coingecko_live_feed import CoinGeckoLiveDataFeed

_logger = setup_logger(__name__)


class ProviderSelector:
    """
    Provider selection logic based on symbol type, timeframe, and data quality.

    This class encapsulates the provider selection rules defined in PROVIDER_COMPARISON.md
    and allows for configuration-driven provider selection.
    """

    def __init__(self, config_path: Optional[str] = None, cache_dir: Optional[str] = None):
        """
        Initialize provider selector.

        Args:
            config_path: Path to YAML configuration file for provider rules
            cache_dir: Cache directory for downloaders that need it
        """
        self.config_path = config_path or "config/data/provider_rules.yaml"
        self.cache_dir = cache_dir
        self.rules = self._load_provider_rules()
        self._initialize_downloaders()

    def _load_provider_rules(self) -> Dict[str, Any]:
        """Load provider selection rules from configuration file."""
        try:
            # Try multiple possible paths for the config file
            possible_paths = [
                self.config_path,
                os.path.join(os.getcwd(), self.config_path),
                os.path.join(Path(__file__).parent.parent.parent, self.config_path),
                os.path.join(Path(__file__).parent.parent.parent.parent, self.config_path)
            ]

            config_file_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    config_file_path = path
                    break

            if config_file_path:
                with open(config_file_path, 'r') as f:
                    rules = yaml.safe_load(f)
                    _logger.debug("Loaded provider rules from %s", config_file_path)
                    return rules
            else:
                _logger.warning("Provider rules config file not found at any of: %s", possible_paths)
                # Default rules if config file doesn't exist
                return self._get_default_rules()
        except Exception as e:
            _logger.warning("Failed to load provider rules from %s: %s", self.config_path, e)
            return self._get_default_rules()

    def _get_default_rules(self) -> Dict[str, Any]:
        """Get default provider selection rules."""
        return {
            'crypto': {
                'primary': 'binance',
                'backup': ['coingecko', 'alpha_vantage'],
                'timeframes': ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M']
            },
            'stock_intraday': {
                'primary': 'fmp',
                'backup': ['alpaca', 'alpha_vantage', 'polygon'],
                'timeframes': ['1m', '5m', '15m', '30m', '1h', '4h']
            },
            'stock_daily': {
                'primary': 'yahoo',
                'backup': ['alpaca', 'tiingo', 'fmp'],
                'timeframes': ['1d', '1w', '1M']
            },
            'stock_weekly_monthly': {
                'primary': 'tiingo',
                'backup': ['yahoo', 'fmp'],
                'timeframes': ['1w', '1M']
            }
        }

    def _initialize_downloaders(self):
        """Initialize available downloaders."""
        self.downloaders = {}

        # Initialize downloaders with error handling
        downloader_classes = {
            'binance': BinanceDataDownloader,
            'yahoo': YahooDataDownloader,
            'alpha_vantage': AlphaVantageDataDownloader,
            'fmp': FMPDataDownloader,
            'tiingo': TiingoDataDownloader,
            'polygon': PolygonDataDownloader,
            'twelvedata': TwelveDataDataDownloader,
            'finnhub': FinnhubDataDownloader,
            'coingecko': CoinGeckoDataDownloader,
            'alpaca': AlpacaDataDownloader,
        }

        for name, downloader_class in downloader_classes.items():
            try:
                # Check for required API keys and initialize with appropriate parameters
                if name == 'binance':
                    # Binance doesn't require API keys for public data
                    self.downloaders[name] = downloader_class()
                elif name == 'yahoo':
                    # Yahoo Finance doesn't require API keys
                    self.downloaders[name] = downloader_class()
                elif name == 'alpha_vantage':
                    if not ALPHA_VANTAGE_API_KEY:
                        _logger.warning("Skipping %s downloader: No API key found", name)
                        continue
                    self.downloaders[name] = downloader_class(api_key=ALPHA_VANTAGE_API_KEY)
                elif name == 'fmp':
                    if not FMP_API_KEY:
                        _logger.warning("Skipping %s downloader: No API key found", name)
                        continue
                    self.downloaders[name] = downloader_class(api_key=FMP_API_KEY)
                elif name == 'polygon':
                    if not POLYGON_API_KEY:
                        _logger.warning("Skipping %s downloader: No API key found", name)
                        continue
                    self.downloaders[name] = downloader_class(api_key=POLYGON_API_KEY)
                elif name == 'coingecko':
                    # CoinGecko doesn't require API key or data_dir
                    self.downloaders[name] = downloader_class()
                elif name == 'twelvedata':
                    if not TWELVE_DATA_API_KEY:
                        _logger.warning("Skipping %s downloader: No API key found", name)
                        continue
                    self.downloaders[name] = downloader_class(api_key=TWELVE_DATA_API_KEY)
                elif name == 'finnhub':
                    if not FINNHUB_API_KEY:
                        _logger.warning("Skipping %s downloader: No API key found", name)
                        continue
                    self.downloaders[name] = downloader_class(api_key=FINNHUB_API_KEY)
                elif name == 'tiingo':
                    if not TIINGO_API_KEY:
                        _logger.warning("Skipping %s downloader: No API key found", name)
                        continue
                    self.downloaders[name] = downloader_class(api_key=TIINGO_API_KEY)
                elif name == 'alpaca':
                    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
                        _logger.warning("Skipping %s downloader: No API key or secret key found", name)
                        continue
                    self.downloaders[name] = downloader_class(api_key=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY)
                else:
                    # For other downloaders, try to initialize without parameters
                    self.downloaders[name] = downloader_class()

                _logger.info("Initialized %s downloader", name)
            except Exception as e:
                _logger.warning("Failed to initialize %s downloader: %s", name, e)

    def _classify_symbol(self, symbol: str) -> str:
        """
        Classify symbol type (crypto, stock, etc.) using configuration-driven rules.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT', 'AAPL')

        Returns:
            Symbol classification
        """
        symbol_upper = symbol.upper()

        # Get classification rules from config
        classification_rules = self.rules.get('symbol_classification', {})

        # Check crypto patterns first
        crypto_rules = classification_rules.get('crypto', {})
        crypto_patterns = crypto_rules.get('patterns', [])
        crypto_suffixes = crypto_rules.get('suffixes', [])
        crypto_assets = crypto_rules.get('known_assets', [])

        # Check against crypto patterns
        for pattern in crypto_patterns:
            if re.match(pattern, symbol_upper):
                return 'crypto'

        # Check against crypto suffixes
        if any(symbol_upper.endswith(suffix) for suffix in crypto_suffixes):
            return 'crypto'

        # Check if starts with known crypto asset
        for asset in crypto_assets:
            if symbol_upper.startswith(asset) and len(symbol_upper) > len(asset):
                return 'crypto'

        # Check stock patterns
        stock_rules = classification_rules.get('stock', {})
        stock_patterns = stock_rules.get('patterns', [])

        for pattern in stock_patterns:
            if re.match(pattern, symbol_upper):
                return 'stock'

        # Check for exchange suffixes
        exchange_suffixes = stock_rules.get('exchange_suffixes', {})
        for suffix in exchange_suffixes.keys():
            if symbol_upper.endswith(suffix):
                return 'stock'

        # Default to unknown for unrecognized symbols
        return 'unknown'

    def classify_symbol(self, symbol: str) -> str:
        """
        Public method to classify symbol type (crypto, stock, etc.).

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT', 'AAPL')

        Returns:
            Symbol classification ('crypto', 'stock', 'unknown')
        """
        return self._classify_symbol(symbol)

    def classify_symbol_for_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """
        Enhanced symbol classification specifically for fundamentals data retrieval.

        This method provides detailed symbol information needed for optimal
        fundamentals provider selection, including market detection, exchange
        identification, and international symbol handling.

        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'GOOGL', 'ASML.AS')

        Returns:
            Dictionary containing detailed symbol classification:
            {
                'symbol': str,           # Original symbol
                'normalized': str,       # Normalized symbol
                'symbol_type': str,      # 'stock', 'etf', 'reit', 'crypto', 'unknown'
                'market': str,           # 'US', 'UK', 'EU', 'ASIA', 'unknown'
                'exchange': str,         # 'NASDAQ', 'NYSE', 'LSE', 'AMS', etc.
                'country': str,          # 'US', 'GB', 'NL', etc.
                'international': bool,   # True if non-US symbol
                'currency': str,         # 'USD', 'EUR', 'GBP', etc.
                'fundamentals_support': str  # 'full', 'limited', 'none'
            }
        """
        try:
            # Handle None or invalid input
            if not symbol or not isinstance(symbol, str):
                return {
                    'symbol': symbol,
                    'normalized': '',
                    'symbol_type': 'unknown',
                    'market': 'unknown',
                    'exchange': 'unknown',
                    'country': 'unknown',
                    'international': True,
                    'currency': 'USD',
                    'fundamentals_support': 'none'
                }

            symbol_upper = symbol.upper().strip()

            # Handle empty string after stripping
            if not symbol_upper:
                return {
                    'symbol': symbol,
                    'normalized': '',
                    'symbol_type': 'unknown',
                    'market': 'unknown',
                    'exchange': 'unknown',
                    'country': 'unknown',
                    'international': True,
                    'currency': 'USD',
                    'fundamentals_support': 'none'
                }

            # Initialize classification result
            classification = {
                'symbol': symbol,
                'normalized': symbol_upper,
                'symbol_type': 'unknown',
                'market': 'unknown',
                'exchange': 'unknown',
                'country': 'unknown',
                'international': False,
                'currency': 'USD',  # Default to USD
                'fundamentals_support': 'none'
            }

            # 1. Detect exchange from symbol suffix
            exchange_info = self._detect_exchange_from_symbol(symbol_upper)
            if exchange_info:
                classification.update(exchange_info)

            # 2. Classify symbol type
            symbol_type = self._classify_symbol_type_detailed(symbol_upper, classification)
            classification['symbol_type'] = symbol_type

            # 3. Determine market and country
            market_info = self._determine_market_and_country(symbol_upper, classification)
            classification.update(market_info)

            # 4. Assess fundamentals support
            fundamentals_support = self._assess_fundamentals_support(classification)
            classification['fundamentals_support'] = fundamentals_support

            # 5. Set international flag
            classification['international'] = classification['country'] != 'US'

            return classification

        except Exception:
            _logger.exception("Error classifying symbol %s for fundamentals:", symbol)
            return {
                'symbol': symbol,
                'normalized': symbol.upper(),
                'symbol_type': 'unknown',
                'market': 'unknown',
                'exchange': 'unknown',
                'country': 'unknown',
                'international': True,
                'currency': 'USD',
                'fundamentals_support': 'none'
            }

    def _detect_exchange_from_symbol(self, symbol: str) -> Optional[Dict[str, str]]:
        """
        Detect exchange information from symbol suffix.

        Args:
            symbol: Uppercase symbol

        Returns:
            Dictionary with exchange information or None
        """
        # Exchange suffix mappings
        exchange_mappings = {
            '.L': {'exchange': 'LSE', 'country': 'GB', 'market': 'UK', 'currency': 'GBP'},
            '.TO': {'exchange': 'TSX', 'country': 'CA', 'market': 'CANADA', 'currency': 'CAD'},
            '.SW': {'exchange': 'SWX', 'country': 'CH', 'market': 'SWISS', 'currency': 'CHF'},
            '.DE': {'exchange': 'XETRA', 'country': 'DE', 'market': 'EU', 'currency': 'EUR'},
            '.PA': {'exchange': 'EPA', 'country': 'FR', 'market': 'EU', 'currency': 'EUR'},
            '.AS': {'exchange': 'AMS', 'country': 'NL', 'market': 'EU', 'currency': 'EUR'},
            '.MI': {'exchange': 'BIT', 'country': 'IT', 'market': 'EU', 'currency': 'EUR'},
            '.MC': {'exchange': 'BME', 'country': 'ES', 'market': 'EU', 'currency': 'EUR'},
            '.BR': {'exchange': 'EURONEXT', 'country': 'BE', 'market': 'EU', 'currency': 'EUR'},
            '.VI': {'exchange': 'WBAG', 'country': 'AT', 'market': 'EU', 'currency': 'EUR'},
            '.HK': {'exchange': 'HKEX', 'country': 'HK', 'market': 'ASIA', 'currency': 'HKD'},
            '.T': {'exchange': 'TSE', 'country': 'JP', 'market': 'ASIA', 'currency': 'JPY'},
            '.SS': {'exchange': 'SSE', 'country': 'CN', 'market': 'ASIA', 'currency': 'CNY'},
            '.SZ': {'exchange': 'SZSE', 'country': 'CN', 'market': 'ASIA', 'currency': 'CNY'},
            '.AX': {'exchange': 'ASX', 'country': 'AU', 'market': 'OCEANIA', 'currency': 'AUD'},
            '.NZ': {'exchange': 'NZX', 'country': 'NZ', 'market': 'OCEANIA', 'currency': 'NZD'},
        }

        for suffix, info in exchange_mappings.items():
            if symbol.endswith(suffix):
                # Remove suffix from normalized symbol
                normalized = symbol[:-len(suffix)]
                result = info.copy()
                result['normalized'] = normalized
                return result

        return None

    def _classify_symbol_type_detailed(self, symbol: str, classification: Dict[str, Any]) -> str:
        """
        Classify symbol type with detailed analysis.

        Args:
            symbol: Uppercase symbol
            classification: Current classification info

        Returns:
            Symbol type ('stock', 'etf', 'reit', 'crypto', 'unknown')
        """
        # First check if it's crypto using the original symbol (before suffix removal)
        original_symbol = classification.get('symbol', symbol).upper()
        if self._classify_symbol(original_symbol) == 'crypto':
            return 'crypto'

        # ETF patterns (common ETF suffixes and patterns)
        etf_patterns = [
            r'.*ETF$',      # Ends with ETF
            r'^SPY$',       # SPDR S&P 500
            r'^QQQ$',       # Invesco QQQ
            r'^IWM$',       # iShares Russell 2000
            r'^VTI$',       # Vanguard Total Stock Market
            r'^VOO$',       # Vanguard S&P 500
            r'^VEA$',       # Vanguard FTSE Developed Markets
            r'^VWO$',       # Vanguard FTSE Emerging Markets
            r'^BND$',       # Vanguard Total Bond Market
            r'^GLD$',       # SPDR Gold Shares
            r'^SLV$',       # iShares Silver Trust
        ]

        for pattern in etf_patterns:
            if re.match(pattern, symbol):
                return 'etf'

        # REIT patterns
        reit_patterns = [
            r'.*REIT$',     # Ends with REIT
            r'^REI[T]?$',   # REI or REIT
        ]

        for pattern in reit_patterns:
            if re.match(pattern, symbol):
                return 'reit'

        # Check for common stock patterns
        # Most symbols without special suffixes are stocks
        if re.match(r'^[A-Z]{1,5}$', symbol):  # 1-5 letter symbols
            return 'stock'

        # Symbols with numbers might be preferred shares or special classes
        if re.match(r'^[A-Z]{1,4}[0-9]$', symbol):
            return 'stock'

        # Class shares (e.g., BRK-A, BRK-B, GOOGL, GOOG)
        if re.match(r'^[A-Z]{1,4}[-.]?[A-Z]$', symbol):
            return 'stock'

        return 'stock'  # Default to stock for most cases

    def _determine_market_and_country(self, symbol: str, classification: Dict[str, Any]) -> Dict[str, str]:
        """
        Determine market and country information.

        Args:
            symbol: Uppercase symbol
            classification: Current classification info

        Returns:
            Dictionary with market and country info
        """
        # If exchange info already determined market/country, use it
        if classification.get('country') != 'unknown':
            return {}

        # For symbols without exchange suffix, assume US market
        market_info = {
            'market': 'US',
            'country': 'US',
            'exchange': self._determine_us_exchange(symbol),
            'currency': 'USD'
        }

        return market_info

    def _determine_us_exchange(self, symbol: str) -> str:
        """
        Determine likely US exchange for a symbol.

        Args:
            symbol: Uppercase symbol

        Returns:
            Exchange name ('NASDAQ', 'NYSE', 'AMEX', 'OTC')
        """
        # Common NASDAQ patterns (tech companies, biotech, etc.)
        nasdaq_patterns = [
            r'^[A-Z]{4,5}$',  # 4-5 letter symbols often NASDAQ
            r'^Q[A-Z]{3}$',   # Q-prefixed symbols
        ]

        # Known NASDAQ symbols (partial list)
        nasdaq_symbols = {
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA',
            'NFLX', 'ADBE', 'CRM', 'ORCL', 'CSCO', 'INTC', 'AMD', 'QCOM'
        }

        # Known NYSE symbols (partial list)
        nyse_symbols = {
            'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'V', 'MA', 'JNJ', 'PG',
            'KO', 'PEP', 'WMT', 'HD', 'UNH', 'CVX', 'XOM', 'T', 'VZ'
        }

        if symbol in nasdaq_symbols:
            return 'NASDAQ'
        elif symbol in nyse_symbols:
            return 'NYSE'
        else:
            # Use patterns as heuristic
            for pattern in nasdaq_patterns:
                if re.match(pattern, symbol):
                    return 'NASDAQ'

            # Default to NYSE for shorter symbols
            if len(symbol) <= 3:
                return 'NYSE'
            else:
                return 'NASDAQ'

    def _assess_fundamentals_support(self, classification: Dict[str, Any]) -> str:
        """
        Assess level of fundamentals support for this symbol.

        Args:
            classification: Symbol classification info

        Returns:
            Support level ('full', 'limited', 'none')
        """
        symbol_type = classification.get('symbol_type', 'unknown')
        country = classification.get('country', 'unknown')
        market = classification.get('market', 'unknown')

        # Crypto symbols have no fundamentals
        if symbol_type == 'crypto':
            return 'none'

        # US stocks and ETFs have full support
        if country == 'US' and symbol_type in ['stock', 'etf', 'reit']:
            return 'full'

        # Major international markets have good support
        major_markets = ['UK', 'EU', 'CANADA', 'ASIA']
        if market in major_markets and symbol_type in ['stock', 'etf']:
            return 'limited'

        # Other international markets have limited support
        if country != 'unknown' and symbol_type in ['stock', 'etf']:
            return 'limited'

        return 'none'

    def _get_rule_name(self, symbol_type: str, timeframe: str) -> Optional[str]:
        """
        Map symbol type and timeframe to the appropriate rule name.

        Args:
            symbol_type: Type of symbol (crypto, stock, etc.)
            timeframe: Data timeframe (e.g., '1m', '1h', '1d')

        Returns:
            Rule name or None if no mapping found
        """
        if symbol_type == 'crypto':
            return 'crypto'
        elif symbol_type == 'stock':
            # Map stock timeframes to appropriate rules
            if timeframe in ['1m', '5m', '15m', '30m', '1h', '4h']:
                return 'stock_intraday'
            elif timeframe in ['1d']:
                return 'stock_daily'
            elif timeframe in ['1w', '1M']:
                return 'stock_weekly_monthly'

        return None

    def get_best_provider(self, symbol: str, timeframe: str) -> Optional[str]:
        """
        Get the best provider for a given symbol and timeframe.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe (e.g., '1m', '1h', '1d')

        Returns:
            Best provider name, or None if no suitable provider found
        """
        symbol_type = self._classify_symbol(symbol)

        # Map symbol type + timeframe to rule name
        rule_name = self._get_rule_name(symbol_type, timeframe)
        if not rule_name:
            _logger.warning("No rule found for %s %s", symbol_type, timeframe)
            return None

        # Get rules for this rule name
        if rule_name not in self.rules:
            _logger.warning("No rules found for rule: %s", rule_name)
            return None

        rules = self.rules[rule_name]

        # Check if timeframe is supported
        if timeframe not in rules.get('timeframes', []):
            _logger.warning("Timeframe %s not supported for %s", timeframe, rule_name)
            return None

        # Return primary provider if available
        primary = rules.get('primary')
        if primary and primary in self.downloaders:
            return primary

        # Try backup providers
        backup = rules.get('backup', [])
        for provider in backup:
            if provider in self.downloaders:
                return provider

        _logger.error("No suitable provider found for %s (%s)", symbol, timeframe)
        return None

    def get_best_downloader(self, symbol: str, timeframe: str) -> Optional[BaseDataDownloader]:
        """
        Get the best downloader for a given symbol and timeframe.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe (e.g., '1m', '1h', '1d')

        Returns:
            Best downloader instance, or None if no suitable provider found
        """
        provider_name = self.get_best_provider(symbol, timeframe)
        if provider_name and provider_name in self.downloaders:
            return self.downloaders[provider_name]
        return None

    def get_provider_with_failover(self, symbol: str, timeframe: str) -> List[str]:
        """
        Get ordered list of providers with failover support.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe

        Returns:
            Ordered list of provider names (primary first, then backups)
        """
        symbol_type = self._classify_symbol(symbol)

        # Map symbol type + timeframe to rule name (same logic as get_best_provider)
        rule_name = self._get_rule_name(symbol_type, timeframe)
        if not rule_name:
            _logger.warning("No rule found for %s %s", symbol_type, timeframe)
            return []

        # Get rules for this rule name
        if rule_name not in self.rules:
            _logger.warning("No rules found for rule: %s", rule_name)
            return []

        rules = self.rules[rule_name]
        providers = []

        # Check if timeframe is supported
        if timeframe not in rules.get('timeframes', []):
            _logger.warning("Timeframe %s not supported for %s", timeframe, rule_name)
            return []

        # Add primary provider
        primary = rules.get('primary')
        if primary and primary in self.downloaders:
            providers.append(primary)

        # Add backup providers
        backup = rules.get('backup', [])
        for provider in backup:
            if provider in self.downloaders and provider not in providers:
                providers.append(provider)

        return providers

    def get_ticker_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive ticker information for symbol classification and provider selection.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary with ticker information
        """
        symbol_upper = symbol.upper()
        symbol_type = self._classify_symbol(symbol)

        info = {
            'original_ticker': symbol,
            'symbol_type': symbol_type,
            'formatted_ticker': symbol_upper,
            'exchange': None,
            'base_asset': None,
            'quote_asset': None
        }

        # Get classification rules
        classification_rules = self.rules.get('symbol_classification', {})

        if symbol_type == 'crypto':
            # Parse crypto pair
            crypto_rules = classification_rules.get('crypto', {})
            crypto_suffixes = crypto_rules.get('suffixes', [])

            # Try to parse base/quote assets
            for suffix in crypto_suffixes:
                if symbol_upper.endswith(suffix):
                    base = symbol_upper[:-len(suffix)]
                    if base:
                        info['base_asset'] = base
                        info['quote_asset'] = suffix
                        break
        elif symbol_type == 'stock':
            # Check for exchange suffix
            stock_rules = classification_rules.get('stock', {})
            exchange_suffixes = stock_rules.get('exchange_suffixes', {})

            for suffix, exchange_name in exchange_suffixes.items():
                if symbol_upper.endswith(suffix):
                    info['exchange'] = exchange_name
                    break

            if not info['exchange']:
                info['exchange'] = "US Markets (NASDAQ/NYSE)"

        return info

    def get_data_provider_config(self, symbol: str, interval: str = None) -> Dict[str, Any]:
        """
        Get configuration for data retrieval based on ticker and interval.

        Args:
            symbol: The ticker symbol
            interval: Time interval (1d, 1h, 5m, 15m, etc.)

        Returns:
            Dictionary with provider-specific configuration
        """
        ticker_info = self.get_ticker_info(symbol)
        best_provider = self.get_best_provider(symbol, interval or '1d')

        config = {
            'ticker': ticker_info['original_ticker'],
            'provider': best_provider,
            'formatted_ticker': ticker_info['formatted_ticker'],
            'best_provider': best_provider,
            'symbol_type': ticker_info['symbol_type'],
            'exchange': ticker_info['exchange'],
            'base_asset': ticker_info['base_asset'],
            'quote_asset': ticker_info['quote_asset']
        }

        if best_provider == 'binance':
            config.update({
                'interval': interval or '1d',
                'limit': 1000,
                'reason': 'Crypto symbol - Binance provides best coverage'
            })
        elif best_provider == 'yahoo':
            config.update({
                'period': '1y',
                'interval': interval or '1d',
                'reason': f'Stock symbol with {interval} interval - yfinance for daily data'
            })
        elif best_provider == 'fmp':
            config.update({
                'interval': interval or '1d',
                'reason': f'Stock symbol with {interval} interval - FMP for intraday data'
            })

        return config

    def validate_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Comprehensive ticker validation.

        Args:
            symbol: The ticker to validate

        Returns:
            Dictionary with validation results
        """
        if not symbol:
            return {
                'valid': False,
                'error': 'Empty ticker',
                'suggestions': []
            }

        symbol_upper = symbol.upper().strip()

        # Basic format validation
        if len(symbol_upper) < 1 or len(symbol_upper) > 15:
            return {
                'valid': False,
                'error': 'Invalid ticker length',
                'suggestions': ['Ticker should be 1-15 characters long']
            }

        if not re.match(r'^[A-Z0-9.-]+$', symbol_upper):
            return {
                'valid': False,
                'error': 'Invalid ticker format',
                'suggestions': ['Use only alphanumeric characters, dots, and hyphens']
            }

        # Classify the ticker
        ticker_info = self.get_ticker_info(symbol)

        if ticker_info['symbol_type'] == 'unknown':
            return {
                'valid': False,
                'error': 'Unknown ticker format',
                'suggestions': [
                    'Check if ticker exists on supported exchanges',
                    'Verify ticker symbol is correct',
                    'Consider adding exchange suffix (e.g., .L for London)'
                ]
            }

        return {
            'valid': True,
            'symbol_type': ticker_info['symbol_type'],
            'exchange': ticker_info['exchange'],
            'base_asset': ticker_info['base_asset'],
            'quote_asset': ticker_info['quote_asset'],
            'suggestions': []
        }


class DataManager:
    """
    Main facade for all data operations in the E-Trading system.

    This class implements the unified data access architecture described in REFACTOR.md:
    - Single entry point for all data requests
    - Provider-agnostic caching with UnifiedCache
    - Intelligent provider selection with automatic failover
    - Integration with live data feeds
    - Centralized error handling and retry logic
    """

    def __init__(self, cache_dir: str = DATA_CACHE_DIR, config_path: Optional[str] = None):
        """
        Initialize DataManager.

        Args:
            cache_dir: Cache directory path
            config_path: Path to provider configuration file
        """
        self.cache = UnifiedCache(cache_dir)
        self.provider_selector = ProviderSelector(config_path, cache_dir)
        self.rate_limiters = {}

        # Initialize rate limiters for each provider
        self._initialize_rate_limiters()

        _logger.info("DataManager initialized successfully")

    def _initialize_rate_limiters(self):
        """Initialize rate limiters for each provider."""
        # Rate limits from PROVIDER_COMPARISON.md
        rate_limits = {
            'binance': {'requests_per_minute': 1200},
            'yahoo': {'requests_per_minute': 100},
            'alpha_vantage': {'requests_per_minute': 5},
            'fmp': {'requests_per_minute': 3000},
            'tiingo': {'requests_per_minute': 100},
            'polygon': {'requests_per_minute': 5},
            'coingecko': {'requests_per_minute': 50},
            'alpaca': {'requests_per_minute': 200},
        }

        for provider, limits in rate_limits.items():
            self.rate_limiters[provider] = RateLimiter(**limits)

    @retry_on_exception(max_attempts=3, base_delay=1.0)
    def get_ohlcv(self, symbol: str, timeframe: str,
                  start_date: datetime, end_date: datetime,
                  force_refresh: bool = False) -> pd.DataFrame:
        """
        Retrieve historical OHLCV data with caching and provider selection.

        This method implements the main data retrieval flow:
        1. Check cache for existing data
        2. If cache miss or force_refresh, select best provider
        3. Download data from provider with rate limiting
        4. Validate and cache the data
        5. Return combined data

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT', 'AAPL')
            timeframe: Data timeframe (e.g., '1m', '1h', '1d')
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            force_refresh: Force refresh from provider, bypassing cache

        Returns:
            DataFrame with OHLCV data

        Raises:
            ValueError: If invalid parameters provided
            RuntimeError: If no suitable provider found or data retrieval fails
        """
        # Validate inputs
        if start_date >= end_date:
            raise ValueError("start_date must be before end_date")

        if not symbol or not timeframe:
            raise ValueError("symbol and timeframe are required")

        _logger.info("Requesting data for %s %s from %s to %s", symbol, timeframe, start_date, end_date)

        # Check cache first (unless force refresh)
        if not force_refresh:
            cached_data = self._get_cached_data(symbol, timeframe, start_date, end_date)
            if cached_data is not None and not cached_data.empty:
                _logger.info("Cache hit for %s %s", symbol, timeframe)
                return cached_data

        # Cache miss or force refresh - get from provider
        _logger.info("Cache miss for %s %s, fetching from provider", symbol, timeframe)

        # Get providers with failover support
        providers = self.provider_selector.get_provider_with_failover(symbol, timeframe)
        if not providers:
            raise RuntimeError(f"No suitable provider found for {symbol} {timeframe}")

        # Try providers in order with failover
        last_error = None
        for provider in providers:
            try:
                _logger.info("Trying provider: %s", provider)

                # Apply rate limiting
                if provider in self.rate_limiters:
                    self.rate_limiters[provider].wait_if_needed()

                # Get downloader
                downloader = self.provider_selector.downloaders[provider]

                # Download data
                data = downloader.get_ohlcv(symbol, timeframe, start_date, end_date)

                if data is not None and not data.empty:
                    # Normalize schema: lowercase columns, ensure 'timestamp' column and tz-naive
                    data_copy = data.copy()

                    # If index is datetime and no 'timestamp' column, create it from index
                    if 'timestamp' not in data_copy.columns and isinstance(data_copy.index, pd.DatetimeIndex):
                        ts_index = data_copy.index
                        if ts_index.tz is not None:
                            ts_index = ts_index.tz_localize(None)
                        data_copy.insert(0, 'timestamp', ts_index)

                    # Lowercase known OHLCV columns
                    rename_map = {c: c.lower() for c in data_copy.columns}
                    data_copy = data_copy.rename(columns=rename_map)

                    # Ensure required columns exist
                    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    missing = [c for c in required_cols if c not in data_copy.columns]
                    if missing:
                        _logger.warning("Missing required columns after normalization: %s", missing)

                    # Make timestamp tz-naive and set as index for downstream consumers
                    if 'timestamp' in data_copy.columns:
                        data_copy['timestamp'] = pd.to_datetime(data_copy['timestamp'], errors='coerce')
                        if data_copy['timestamp'].dt.tz is not None:
                            data_copy['timestamp'] = data_copy['timestamp'].dt.tz_localize(None)
                        data_copy = data_copy.set_index('timestamp')

                    # Validate data
                    is_valid, errors = validate_ohlcv_data(data_copy, symbol=symbol, interval=timeframe)
                    if not is_valid:
                        _logger.warning("Data validation failed for %s %s: %s", symbol, timeframe, errors)
                        # Continue with invalid data but log warning

                    # Cache the data
                    self._cache_data(data_copy, symbol, timeframe, start_date, end_date, provider)

                    _logger.info("Successfully retrieved data from %s", provider)
                    return data_copy
                else:
                    _logger.warning("Provider %s returned empty data", provider)

            except Exception as e:
                _logger.warning("Provider %s failed: %s", provider, e)
                last_error = e
                continue

        # All providers failed
        raise RuntimeError(f"All providers failed for {symbol} {timeframe}. Last error: {last_error}")

    def _get_cached_data(self, symbol: str, timeframe: str,
                        start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Get data from cache for the specified date range with staleness check.

        For daily and higher timeframes, if the latest cached data is more than
        24 hours old and we're requesting recent data (end_date is today), the
        cache is considered stale and we return None to force a fresh fetch.
        """
        try:
            # Use UnifiedCache get method with date range
            cached_df = self.cache.get(symbol, timeframe, start_date, end_date)
            if cached_df is not None and not cached_df.empty:
                # Check if cached data is stale
                latest_cached_date = cached_df.index[-1]
                now = datetime.now(timezone.utc)

                # Make latest_cached_date timezone-aware if it isn't
                if latest_cached_date.tzinfo is None:
                    latest_cached_date = latest_cached_date.replace(tzinfo=timezone.utc)

                # Calculate age of the latest data point
                data_age = now - latest_cached_date

                # Define staleness thresholds based on timeframe
                staleness_thresholds = {
                    '1m': timedelta(minutes=5),
                    '5m': timedelta(minutes=15),
                    '15m': timedelta(hours=1),
                    '30m': timedelta(hours=2),
                    '1h': timedelta(hours=4),
                    '4h': timedelta(hours=12),
                    '1d': timedelta(hours=24),
                    '1w': timedelta(days=7),
                    '1M': timedelta(days=30),
                }

                # Get threshold for this timeframe (default to 24 hours)
                threshold = staleness_thresholds.get(timeframe, timedelta(hours=24))

                # If data is stale and we're requesting recent data, reject cache
                end_date_naive = end_date.replace(tzinfo=None) if end_date.tzinfo else end_date
                now_naive = now.replace(tzinfo=None)
                is_requesting_recent_data = (now_naive - end_date_naive) < timedelta(hours=2)

                if data_age > threshold and is_requesting_recent_data:
                    _logger.info(
                        "Cache data for %s %s is stale (age: %s, threshold: %s), fetching fresh data",
                        symbol, timeframe, data_age, threshold
                    )
                    return None

                _logger.info("Cache hit for %s %s: %d rows", symbol, timeframe, len(cached_df))
                return cached_df

        except Exception as e:
            _logger.warning("Error retrieving cached data: %s", e)

        return None

    def _cache_data(self, data: pd.DataFrame, symbol: str, timeframe: str,
                   start_date: datetime, end_date: datetime, provider: str):
        """Cache data using UnifiedCache."""
        try:
            # Use UnifiedCache put method with the full data and date range
            success = self.cache.put(data, symbol, timeframe, start_date, end_date, provider)
            if success:
                _logger.info("Cached %d rows for %s %s from %s", len(data), symbol, timeframe, provider)
            else:
                _logger.warning("Failed to cache data for %s %s", symbol, timeframe)

        except Exception:
            _logger.exception("Error caching data:")

    def get_live_feed(self, symbol: str, timeframe: str,
                     lookback_bars: int = 1000, **kwargs) -> Optional[BaseLiveDataFeed]:
        """
        Create and return a live data feed instance.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            lookback_bars: Number of historical bars to load
            **kwargs: Additional parameters for the live feed

        Returns:
            Live data feed instance, or None if creation fails
        """
        try:
            # Get the best provider for live feeds
            provider = self.provider_selector.get_best_provider(symbol, timeframe)
            if not provider:
                _logger.error("No suitable provider found for live feed: %s %s", symbol, timeframe)
                return None

            # Map provider to live feed class
            feed_classes = {
                'binance': BinanceLiveDataFeed,
                'yahoo': YahooLiveDataFeed,
                'coingecko': CoinGeckoLiveDataFeed,
            }

            if provider not in feed_classes:
                _logger.error("No live feed available for provider: %s", provider)
                return None

            feed_class = feed_classes[provider]

            # Create feed configuration
            config = {
                'symbol': symbol,
                'interval': timeframe,
                'lookback_bars': lookback_bars,
                'data_manager': self,  # Pass self for historical data backfilling
                **kwargs
            }

            # Create and return feed instance
            feed = feed_class(**config)
            _logger.info("Created live feed for %s %s using %s", symbol, timeframe, provider)
            return feed

        except Exception:
            _logger.exception("Failed to create live feed for %s %s:", symbol, timeframe)
            return None

    def get_fundamentals(self, symbol: str, providers: Optional[List[str]] = None,
                        force_refresh: bool = False, combination_strategy: str = "priority_based",
                        data_type: str = "general") -> Dict[str, Any]:
        """
        Retrieve fundamentals data with caching and multi-provider combination.

        This method implements the enhanced fundamentals data retrieval flow:
        1. Input validation and symbol normalization
        2. Check cache for valid data (TTL based on data type)
        3. If cache miss or force_refresh, fetch from multiple providers with retry logic
        4. Combine data from multiple providers using specified strategy
        5. Cache new data and cleanup stale data
        6. Return combined fundamentals data

        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'GOOGL')
            providers: List of specific providers to use (None for auto-selection)
            force_refresh: Force refresh even if cache is valid
            combination_strategy: Strategy for combining data ('priority_based', 'quality_based', 'consensus')
            data_type: Type of data to determine TTL and provider sequence (profiles, ratios, statements, etc.)

        Returns:
            Dictionary containing combined fundamentals data
        """
        try:
            # 1. Input validation and normalization
            normalized_symbol = self._normalize_symbol(symbol)
            if not normalized_symbol:
                _logger.error("Invalid symbol provided: %s", symbol)
                return {}

            # Initialize combiner and cache with configuration
            combiner = get_fundamentals_combiner()
            fundamentals_cache = get_fundamentals_cache(self.cache.cache_dir, combiner)

            # 2. Cache validation with data-type specific TTL
            if not force_refresh:
                cached_data = self._get_cached_fundamentals(normalized_symbol, data_type, fundamentals_cache)
                if cached_data:
                    return cached_data

            # 3. Enhanced provider selection
            selected_providers = self._select_fundamentals_providers(normalized_symbol, providers, data_type, combiner)
            if not selected_providers:
                _logger.error("No suitable providers found for %s", normalized_symbol)
                return {}

            _logger.info("Fetching fundamentals for %s from providers: %s", normalized_symbol, selected_providers)

            # 4. Fetch data from multiple providers with error handling
            provider_data = self._fetch_fundamentals_from_providers(normalized_symbol, selected_providers)

            if not provider_data:
                _logger.error("No fundamentals data available for %s from any provider", normalized_symbol)
                # Try to return cached data as fallback
                cached_fallback = fundamentals_cache.find_latest_json(normalized_symbol, data_type=data_type)
                if cached_fallback:
                    _logger.info("Returning stale cached data as fallback for %s", normalized_symbol)
                    return fundamentals_cache.read_json(cached_fallback.file_path) or {}
                return {}

            # 5. Data combination and validation
            combined_data = self._combine_and_validate_fundamentals(provider_data, combination_strategy, data_type, combiner)

            if not combined_data:
                _logger.error("Failed to combine fundamentals data for %s", normalized_symbol)
                return {}

            # 6. Cache management and cleanup
            self._cache_fundamentals_data(normalized_symbol, provider_data, combined_data, fundamentals_cache)

            _logger.info("Successfully retrieved and combined fundamentals for %s from %d providers",
                        normalized_symbol, len(provider_data))

            return combined_data

        except Exception:
            _logger.exception("Error retrieving fundamentals for %s:", symbol)
            return {}

    def _normalize_symbol(self, symbol: str) -> str:
        """
        Normalize trading symbol for consistent processing.

        Args:
            symbol: Raw trading symbol

        Returns:
            Normalized symbol or empty string if invalid
        """
        if not symbol or not isinstance(symbol, str):
            return ""

        # Remove whitespace and convert to uppercase
        normalized = symbol.strip().upper()

        # Basic validation - symbol should contain only alphanumeric characters, dots, and hyphens
        if not re.match(r'^[A-Z0-9.\-]+$', normalized):
            _logger.warning("Symbol contains invalid characters: %s", symbol)
            return ""

        # Handle common symbol mappings
        symbol_mappings = {
            'BRK.B': 'BRK-B',
            'BRK.A': 'BRK-A'
        }

        return symbol_mappings.get(normalized, normalized)

    def _get_cached_fundamentals(self, symbol: str, data_type: str, fundamentals_cache) -> Optional[Dict[str, Any]]:
        """
        Get cached fundamentals data with data-type specific TTL validation.

        Args:
            symbol: Normalized trading symbol
            data_type: Type of data for TTL determination
            fundamentals_cache: Cache instance

        Returns:
            Cached data if valid, None otherwise
        """
        try:
            cached_metadata = fundamentals_cache.find_latest_json(symbol, data_type=data_type)
            if not cached_metadata:
                _logger.debug("No cached data found for %s %s", symbol, data_type)
                return None

            # Load cached data
            cached_data = fundamentals_cache.read_json(cached_metadata.file_path)
            if not cached_data:
                _logger.warning("Failed to read cached data for %s", symbol)
                return None

            _logger.info("Using cached fundamentals for %s from %s (age: %s)",
                        symbol, cached_metadata.provider,
                        datetime.now() - cached_metadata.timestamp)
            return cached_data

        except Exception:
            _logger.exception("Error accessing cached fundamentals for %s:", symbol)
            return None

    def _select_fundamentals_providers(self, symbol: str, requested_providers: Optional[List[str]],
                                     data_type: str, combiner) -> List[str]:
        """
        Select optimal providers for fundamentals data retrieval with enhanced logic.

        This method implements sophisticated provider selection based on:
        - Symbol classification (US vs international stocks)
        - Data type specific provider sequences from fundamentals.json
        - Provider availability and capability validation
        - Symbol compatibility filtering with international support
        - Fallback logic when preferred providers are unavailable

        Args:
            symbol: Normalized trading symbol
            requested_providers: User-specified providers (optional)
            data_type: Type of data for provider selection
            combiner: Fundamentals combiner instance

        Returns:
            List of provider names in priority order
        """
        try:
            # Get detailed symbol classification for provider selection
            symbol_classification = self.provider_selector.classify_symbol_for_fundamentals(symbol)

            # Check if symbol supports fundamentals at all
            if symbol_classification['fundamentals_support'] == 'none':
                _logger.debug("Symbol %s does not support fundamentals data", symbol)
                return []

            if requested_providers:
                # Validate and filter requested providers
                valid_providers = self._validate_requested_providers(
                    requested_providers, symbol_classification
                )
                if valid_providers:
                    _logger.debug("Using validated requested providers for %s: %s", symbol, valid_providers)
                    return valid_providers

            # Load provider sequences from fundamentals.json configuration
            provider_sequence = self._load_data_type_provider_sequence(data_type, combiner)
            _logger.debug("Provider sequence for %s data type: %s", data_type, provider_sequence)

            # Filter providers by symbol compatibility and availability
            compatible_providers = self._filter_compatible_providers(
                provider_sequence, symbol_classification
            )

            if compatible_providers:
                _logger.debug("Using compatible providers for %s %s: %s",
                            symbol, data_type, compatible_providers)
                return compatible_providers

            # Fallback: try general provider sequence if data-type specific failed
            if data_type != 'general':
                general_sequence = self._load_data_type_provider_sequence('general', combiner)
                general_compatible = self._filter_compatible_providers(
                    general_sequence, symbol_classification
                )
                if general_compatible:
                    _logger.warning("Using general provider sequence for %s: %s", symbol, general_compatible)
                    return general_compatible

            # Enhanced fallback: try international-optimized sequence for international symbols
            if symbol_classification.get('international', False):
                intl_providers = self._get_international_optimized_providers(symbol_classification)
                if intl_providers:
                    _logger.warning("Using international-optimized providers for %s: %s", symbol, intl_providers)
                    return intl_providers

            # Last resort: find any available provider with fundamentals support
            fallback_providers = self._get_fallback_providers(symbol_classification)
            if fallback_providers:
                _logger.warning("Using fallback providers for %s: %s", symbol, fallback_providers)
                return fallback_providers

            _logger.error("No suitable providers found for %s", symbol)
            return []

        except Exception:
            _logger.exception("Error selecting providers for %s:", symbol)
            return []

    def _load_data_type_provider_sequence(self, data_type: str, combiner) -> List[str]:
        """
        Load provider sequence from fundamentals.json configuration for specific data type.

        This method implements enhanced data-type specific provider selection by:
        - Loading provider sequences from fundamentals.json configuration
        - Mapping data types to appropriate provider sequences
        - Providing intelligent fallbacks for unmapped data types
        - Validating provider availability before returning sequences

        Args:
            data_type: Type of data (e.g., 'statements', 'ratios', 'profile')
            combiner: Fundamentals combiner instance

        Returns:
            List of provider names in priority order
        """
        try:
            # Enhanced data type mapping for better provider selection
            data_type_mappings = {
                'general': 'profile',
                'company': 'profile',
                'overview': 'profile',
                'financial_statements': 'statements',
                'income_statement': 'statements',
                'balance_sheet': 'statements',
                'cash_flow': 'statements',
                'financial_ratios': 'ratios',
                'valuation_ratios': 'ratios',
                'profitability_ratios': 'ratios',
                'liquidity_ratios': 'ratios',
                'efficiency_ratios': 'ratios',
                'leverage_ratios': 'ratios',
                'growth_ratios': 'ratios',
                'ttm_metrics': 'ratios',
                'earnings': 'calendar',
                'earnings_calendar': 'calendar',
                'dividend_history': 'dividends',
                'dividend_calendar': 'dividends',
                'stock_splits': 'splits',
                'insider_transactions': 'insider_trading',
                'analyst_recommendations': 'analyst_estimates',
                'price_targets': 'analyst_estimates'
            }

            # Provider name mappings to handle configuration vs implementation differences
            provider_name_mappings = {
                'alphavantage': 'alpha_vantage',
                'alpha_vantage': 'alpha_vantage',
                'yfinance': 'yahoo',  # yfinance uses yahoo downloader
                'yahoo': 'yahoo',
                'fmp': 'fmp',
                'twelvedata': 'twelvedata',
                'tiingo': 'tiingo',
                'polygon': 'polygon',
                'finnhub': 'finnhub',
                'alpaca': 'alpaca',
                'binance': 'binance',
                'coingecko': 'coingecko'
            }

            # Map data type to configuration key
            config_key = data_type_mappings.get(data_type, data_type)

            # Get provider sequence from combiner configuration
            provider_sequence = combiner.get_provider_sequence(config_key)

            if provider_sequence:
                # Normalize provider names to match implementation
                normalized_sequence = self._normalize_provider_names(provider_sequence, provider_name_mappings)

                # Validate provider availability
                available_providers = self._validate_provider_availability(normalized_sequence)
                if available_providers:
                    _logger.debug("Loaded provider sequence for %s (%s): %s",
                                data_type, config_key, available_providers)
                    return available_providers
                else:
                    _logger.warning("No providers available for %s sequence: %s",
                                  config_key, normalized_sequence)

            # Enhanced fallback logic with data-type specific preferences
            fallback_sequence = self._get_data_type_fallback_sequence(data_type, combiner)
            if fallback_sequence:
                _logger.debug("Using fallback provider sequence for %s: %s", data_type, fallback_sequence)
                return fallback_sequence

            # Final fallback to general sequence
            general_sequence = combiner.get_provider_sequence('profile')
            if general_sequence:
                normalized_general = self._normalize_provider_names(general_sequence, provider_name_mappings)
                available_general = self._validate_provider_availability(normalized_general)
                if available_general:
                    _logger.warning("Using general provider sequence for %s: %s", data_type, available_general)
                    return available_general

            # Last resort: hardcoded default sequence (using implementation names)
            default_sequence = ['yahoo', 'fmp', 'alpha_vantage']
            available_default = self._validate_provider_availability(default_sequence)
            if available_default:
                _logger.warning("Using default provider sequence for %s: %s", data_type, available_default)
                return available_default

            # If no providers are available at all
            _logger.error("No providers available for data type %s", data_type)
            return []

        except Exception:
            _logger.exception("Error loading provider sequence for %s:", data_type)
            # Return safe default with availability check (using implementation names)
            safe_default = ['yahoo', 'fmp', 'alpha_vantage']
            return self._validate_provider_availability(safe_default)

    def _validate_provider_availability(self, provider_sequence: List[str]) -> List[str]:
        """
        Validate that providers in the sequence are available and support fundamentals.

        Args:
            provider_sequence: List of provider names to validate

        Returns:
            List of available provider names
        """
        available_providers = []

        # Handle case where provider_sequence might be None or empty
        if not provider_sequence:
            return available_providers

        for provider_name in provider_sequence:
            # Handle case where provider_name might be a list (nested structure)
            if isinstance(provider_name, list):
                _logger.warning("Found nested list in provider sequence: %s", provider_name)
                continue

            # Ensure provider_name is a string
            if not isinstance(provider_name, str):
                _logger.warning("Invalid provider name type: %s (%s)", provider_name, type(provider_name))
                continue

            # Check if provider is initialized and available
            if provider_name not in self.provider_selector.downloaders:
                _logger.debug("Provider %s not available (not initialized)", provider_name)
                continue

            downloader = self.provider_selector.downloaders[provider_name]

            # Check if provider supports fundamentals
            if not hasattr(downloader, 'get_fundamentals'):
                _logger.debug("Provider %s not available (no fundamentals support)", provider_name)
                continue

            # Provider is available
            available_providers.append(provider_name)

        return available_providers

    def _get_data_type_fallback_sequence(self, data_type: str, combiner) -> List[str]:
        """
        Get intelligent fallback provider sequence based on data type characteristics.

        Args:
            data_type: Type of data requested
            combiner: Fundamentals combiner instance

        Returns:
            List of fallback provider names
        """
        # Data type categories for intelligent fallbacks
        statement_types = ['statements', 'financial_statements', 'income_statement',
                          'balance_sheet', 'cash_flow']
        ratio_types = ['ratios', 'financial_ratios', 'valuation_ratios', 'profitability_ratios',
                      'liquidity_ratios', 'efficiency_ratios', 'leverage_ratios', 'growth_ratios',
                      'ttm_metrics']
        profile_types = ['profile', 'company', 'overview', 'general']
        calendar_types = ['calendar', 'earnings', 'earnings_calendar']
        dividend_types = ['dividends', 'dividend_history', 'dividend_calendar']

        # Select fallback based on data type category (using implementation names)
        if data_type in statement_types:
            # For statements, prefer FMP and Alpha Vantage
            fallback_candidates = ['fmp', 'alpha_vantage', 'yahoo', 'twelvedata']
        elif data_type in ratio_types:
            # For ratios, prefer Yahoo Finance and FMP
            fallback_candidates = ['yahoo', 'fmp', 'alpha_vantage', 'twelvedata']
        elif data_type in profile_types:
            # For profiles, prefer FMP and Yahoo Finance
            fallback_candidates = ['fmp', 'yahoo', 'alpha_vantage', 'twelvedata']
        elif data_type in calendar_types:
            # For calendar events, prefer Yahoo Finance
            fallback_candidates = ['yahoo', 'fmp', 'alpha_vantage']
        elif data_type in dividend_types:
            # For dividends, prefer Yahoo Finance and FMP
            fallback_candidates = ['yahoo', 'fmp', 'alpha_vantage']
        else:
            # Default fallback for unknown data types
            fallback_candidates = ['yahoo', 'fmp', 'alpha_vantage', 'twelvedata']

        # Validate availability of fallback candidates
        return self._validate_provider_availability(fallback_candidates)

    def _normalize_provider_names(self, provider_sequence: List[str],
                                 provider_mappings: Dict[str, str]) -> List[str]:
        """
        Normalize provider names from configuration to match implementation names.

        Args:
            provider_sequence: List of provider names from configuration
            provider_mappings: Dictionary mapping config names to implementation names

        Returns:
            List of normalized provider names
        """
        normalized = []

        for provider in provider_sequence:
            if isinstance(provider, str):
                # Map provider name using mappings
                normalized_name = provider_mappings.get(provider, provider)
                normalized.append(normalized_name)
            else:
                _logger.warning("Invalid provider name in sequence: %s (%s)", provider, type(provider))

        return normalized

    def _get_international_optimized_providers(self, symbol_classification: Dict[str, Any]) -> List[str]:
        """
        Get provider sequence optimized for international symbols.

        Args:
            symbol_classification: Symbol classification information

        Returns:
            List of provider names optimized for international coverage
        """
        # Providers with good international coverage, in priority order (using implementation names)
        international_providers = ['yahoo', 'twelvedata', 'alpha_vantage', 'fmp']

        # Filter by availability and compatibility
        available_providers = []
        for provider in international_providers:
            if (provider in self.provider_selector.downloaders and
                self._is_provider_compatible_with_symbol(provider, symbol_classification)):
                available_providers.append(provider)

        return available_providers

    def _validate_requested_providers(self, requested_providers: List[str],
                                    symbol_classification: Dict[str, Any]) -> List[str]:
        """
        Validate user-requested providers for symbol compatibility.

        Args:
            requested_providers: List of provider names requested by user
            symbol_classification: Symbol classification information

        Returns:
            List of valid provider names
        """
        valid_providers = []

        for provider in requested_providers:
            # Check if provider is available
            if provider not in self.provider_selector.downloaders:
                _logger.warning("Provider %s not available", provider)
                continue

            downloader = self.provider_selector.downloaders[provider]

            # Check if provider supports fundamentals
            if not hasattr(downloader, 'get_fundamentals'):
                _logger.warning("Provider %s does not support fundamentals", provider)
                continue

            # Check symbol compatibility
            if self._is_provider_compatible_with_symbol(provider, symbol_classification):
                valid_providers.append(provider)
            else:
                _logger.warning("Provider %s not compatible with symbol %s",
                              provider, symbol_classification['symbol'])

        return valid_providers

    def _filter_compatible_providers(self, provider_sequence: List[str],
                                   symbol_classification: Dict[str, Any]) -> List[str]:
        """
        Filter provider sequence by symbol compatibility and availability with enhanced logic.

        This method implements comprehensive provider filtering based on:
        - Provider availability in the system
        - Fundamentals support capability
        - Symbol compatibility (market, exchange, symbol type)
        - Provider-specific limitations and strengths
        - Provider quality scores and reliability metrics

        Args:
            provider_sequence: Ordered list of providers from configuration
            symbol_classification: Symbol classification information

        Returns:
            List of compatible and available providers, sorted by suitability
        """
        compatible_providers = []

        for provider in provider_sequence:
            # Check availability
            if provider not in self.provider_selector.downloaders:
                _logger.debug("Provider %s not available, skipping", provider)
                continue

            downloader = self.provider_selector.downloaders[provider]

            # Check fundamentals support
            if not hasattr(downloader, 'get_fundamentals'):
                _logger.debug("Provider %s does not support fundamentals, skipping", provider)
                continue

            # Enhanced symbol compatibility check
            compatibility_result = self._check_provider_symbol_compatibility(provider, symbol_classification)
            if compatibility_result['compatible']:
                # Add provider with compatibility metadata
                provider_info = {
                    'provider': provider,
                    'quality_score': compatibility_result.get('quality_score', 3),
                    'strengths': compatibility_result.get('strengths', []),
                    'limitations': compatibility_result.get('limitations', []),
                    'reason': compatibility_result.get('reason', 'Compatible')
                }
                compatible_providers.append(provider_info)
                _logger.debug("Provider %s compatible with %s: %s",
                            provider, symbol_classification['symbol'], compatibility_result['reason'])
            else:
                _logger.debug("Provider %s not compatible with symbol %s: %s",
                            provider, symbol_classification['symbol'], compatibility_result['reason'])

        # Sort providers by suitability for this symbol
        if compatible_providers:
            sorted_providers = self._sort_providers_by_suitability(compatible_providers, symbol_classification)
            provider_names = [p['provider'] for p in sorted_providers]

            # Log the final selection with reasoning
            if len(provider_names) > 1:
                _logger.debug("Sorted providers for %s by suitability: %s",
                            symbol_classification['symbol'], provider_names)

            return provider_names

        return []

    def _sort_providers_by_suitability(self, compatible_providers: List[Dict[str, Any]],
                                     symbol_classification: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Sort compatible providers by suitability for the specific symbol.

        Args:
            compatible_providers: List of compatible provider info dictionaries
            symbol_classification: Symbol classification information

        Returns:
            Sorted list of provider info dictionaries
        """
        def calculate_suitability_score(provider_info: Dict[str, Any]) -> float:
            """Calculate suitability score for provider."""
            base_score = provider_info.get('quality_score', 3)

            # Boost score based on symbol characteristics
            international = symbol_classification.get('international', False)
            market = symbol_classification.get('market', 'unknown')
            symbol_type = symbol_classification.get('symbol_type', 'unknown')

            # International symbol adjustments
            if international:
                if provider_info['provider'] in ['yfinance', 'twelvedata', 'alpha_vantage']:
                    base_score += 1.0  # Boost for international-friendly providers
                elif provider_info['provider'] in ['fmp', 'alpaca', 'tiingo']:
                    base_score -= 0.5  # Penalty for US-only providers
            else:
                # US symbol adjustments
                if provider_info['provider'] in ['fmp', 'alpaca', 'tiingo']:
                    base_score += 0.5  # Boost for US-optimized providers

            # Symbol type adjustments
            if symbol_type == 'etf':
                if provider_info['provider'] in ['yfinance', 'fmp']:
                    base_score += 0.3  # ETFs work well with these providers
            elif symbol_type == 'reit':
                if provider_info['provider'] in ['yfinance', 'fmp']:
                    base_score += 0.3  # REITs work well with these providers

            # Market-specific adjustments
            if market == 'EU':
                if provider_info['provider'] in ['yfinance', 'twelvedata']:
                    base_score += 0.5  # Better EU coverage
            elif market == 'UK':
                if provider_info['provider'] in ['yfinance', 'alpha_vantage']:
                    base_score += 0.5  # Better UK coverage
            elif market == 'ASIA':
                if provider_info['provider'] == 'yfinance':
                    base_score += 0.5  # Yahoo has good Asian coverage

            return base_score

        # Calculate suitability scores and sort
        for provider_info in compatible_providers:
            provider_info['suitability_score'] = calculate_suitability_score(provider_info)

        # Sort by suitability score (descending), then by original order (ascending)
        return sorted(compatible_providers,
                     key=lambda p: (-p['suitability_score'], compatible_providers.index(p)))

    def _check_provider_symbol_compatibility(self, provider: str,
                                           symbol_classification: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced provider compatibility checking with detailed reasoning.

        Args:
            provider: Provider name
            symbol_classification: Symbol classification information

        Returns:
            Dictionary with compatibility result and reasoning
        """
        symbol_type = symbol_classification.get('symbol_type', 'unknown')
        country = symbol_classification.get('country', 'unknown')
        market = symbol_classification.get('market', 'unknown')
        international = symbol_classification.get('international', False)
        exchange = symbol_classification.get('exchange', 'unknown')

        # Crypto symbols don't use fundamentals
        if symbol_type == 'crypto':
            return {
                'compatible': False,
                'reason': 'Crypto symbols do not support fundamentals data'
            }

        # Enhanced provider-specific compatibility rules
        provider_compatibility = {
            'yfinance': {
                'symbol_types': ['stock', 'etf', 'reit'],
                'markets': ['US', 'UK', 'EU', 'CANADA', 'ASIA', 'OCEANIA'],
                'exchanges': ['NYSE', 'NASDAQ', 'LSE', 'TSX', 'AMS', 'EPA', 'XETRA', 'HKEX', 'TSE'],
                'international_support': True,
                'strengths': ['international_coverage', 'calculated_ratios', 'ttm_metrics'],
                'limitations': ['scraping_based', 'occasional_outages'],
                'quality_score': 4
            },
            'fmp': {
                'symbol_types': ['stock', 'etf', 'reit'],
                'markets': ['US'],
                'exchanges': ['NYSE', 'NASDAQ', 'AMEX'],
                'international_support': False,
                'strengths': ['structured_statements', 'comprehensive_ratios', 'historical_data'],
                'limitations': ['us_only', 'api_limits'],
                'quality_score': 5
            },
            'alpha_vantage': {
                'symbol_types': ['stock', 'etf'],
                'markets': ['US', 'UK', 'EU'],
                'exchanges': ['NYSE', 'NASDAQ', 'LSE', 'XETRA', 'EPA'],
                'international_support': True,
                'strengths': ['consistent_json', 'reliable_overview'],
                'limitations': ['strict_rate_limits', 'limited_international'],
                'quality_score': 4
            },
            'alpaca': {
                'symbol_types': ['stock', 'etf'],
                'markets': ['US'],
                'exchanges': ['NYSE', 'NASDAQ'],
                'international_support': False,
                'strengths': ['real_time_data', 'trading_integration'],
                'limitations': ['us_only', 'limited_fundamentals'],
                'quality_score': 3
            },
            'tiingo': {
                'symbol_types': ['stock', 'etf'],
                'markets': ['US'],
                'exchanges': ['NYSE', 'NASDAQ'],
                'international_support': False,
                'strengths': ['historical_data', 'data_quality'],
                'limitations': ['us_only', 'limited_fundamentals'],
                'quality_score': 4
            },
            'polygon': {
                'symbol_types': ['stock', 'etf'],
                'markets': ['US'],
                'exchanges': ['NYSE', 'NASDAQ'],
                'international_support': False,
                'strengths': ['real_time_data', 'comprehensive_market_data'],
                'limitations': ['us_only', 'expensive'],
                'quality_score': 4
            },
            'twelvedata': {
                'symbol_types': ['stock', 'etf'],
                'markets': ['US', 'UK', 'EU', 'ASIA'],
                'exchanges': ['NYSE', 'NASDAQ', 'LSE', 'XETRA', 'EPA', 'AMS', 'HKEX'],
                'international_support': True,
                'strengths': ['good_api_design', 'international_coverage'],
                'limitations': ['limited_free_fundamentals', 'paid_features'],
                'quality_score': 4
            },
            'finnhub': {
                'symbol_types': ['stock', 'etf'],
                'markets': ['US', 'UK', 'EU'],
                'exchanges': ['NYSE', 'NASDAQ', 'LSE', 'XETRA'],
                'international_support': True,
                'strengths': ['real_time_data', 'news_integration'],
                'limitations': ['limited_fundamentals', 'rate_limits'],
                'quality_score': 3
            }
        }

        # Get provider compatibility info
        compat_info = provider_compatibility.get(provider, {
            'symbol_types': ['stock', 'etf'],
            'markets': ['US'],
            'exchanges': ['NYSE', 'NASDAQ'],
            'international_support': False,
            'strengths': [],
            'limitations': ['unknown_provider'],
            'quality_score': 2
        })

        # Check symbol type compatibility
        if symbol_type not in compat_info['symbol_types']:
            return {
                'compatible': False,
                'reason': f'Provider {provider} does not support {symbol_type} symbols'
            }

        # Check market compatibility
        if market not in compat_info['markets']:
            if international and not compat_info['international_support']:
                return {
                    'compatible': False,
                    'reason': f'Provider {provider} does not support international markets ({market})'
                }

        # Check exchange compatibility (if exchange is known)
        if (exchange != 'unknown' and
            'exchanges' in compat_info and
            exchange not in compat_info['exchanges']):
            return {
                'compatible': False,
                'reason': f'Provider {provider} does not support exchange {exchange}'
            }

        # Provider is compatible
        strengths = ', '.join(compat_info.get('strengths', []))
        return {
            'compatible': True,
            'reason': f'Compatible - strengths: {strengths}',
            'quality_score': compat_info.get('quality_score', 3),
            'strengths': compat_info.get('strengths', []),
            'limitations': compat_info.get('limitations', [])
        }

    def _is_provider_compatible_with_symbol(self, provider: str,
                                          symbol_classification: Dict[str, Any]) -> bool:
        """
        Legacy compatibility method for backward compatibility.

        Args:
            provider: Provider name
            symbol_classification: Symbol classification information

        Returns:
            True if provider is compatible with symbol
        """
        result = self._check_provider_symbol_compatibility(provider, symbol_classification)
        return result['compatible']

    def _get_fallback_providers(self, symbol_classification: Dict[str, Any]) -> List[str]:
        """
        Get intelligent fallback providers when no configured providers are available.

        This method implements smart fallback logic that considers:
        - Provider compatibility with symbol characteristics
        - Provider quality scores and reliability
        - International vs domestic symbol optimization
        - Provider availability and fundamentals support

        Args:
            symbol_classification: Symbol classification information

        Returns:
            List of fallback provider names (limited to 3, ordered by suitability)
        """
        fallback_candidates = []

        # Evaluate all available providers
        for provider_name, downloader in self.provider_selector.downloaders.items():
            if hasattr(downloader, 'get_fundamentals'):
                compatibility_result = self._check_provider_symbol_compatibility(
                    provider_name, symbol_classification
                )
                if compatibility_result['compatible']:
                    fallback_candidates.append({
                        'provider': provider_name,
                        'quality_score': compatibility_result.get('quality_score', 3),
                        'strengths': compatibility_result.get('strengths', []),
                        'limitations': compatibility_result.get('limitations', [])
                    })

        if not fallback_candidates:
            return []

        # Sort by quality score and international support preference
        international = symbol_classification.get('international', False)

        def sort_key(candidate):
            provider = candidate['provider']
            quality = candidate['quality_score']

            # Boost score for international-friendly providers if needed
            if international and provider in ['yfinance', 'twelvedata', 'alpha_vantage']:
                quality += 1

            # Boost score for US-optimized providers for US symbols
            if not international and provider in ['fmp', 'alpaca', 'tiingo']:
                quality += 0.5

            return quality

        # Sort candidates by adjusted quality score (descending)
        sorted_candidates = sorted(fallback_candidates, key=sort_key, reverse=True)

        # Extract provider names and limit to 3
        fallback_providers = [candidate['provider'] for candidate in sorted_candidates[:3]]

        _logger.debug("Selected fallback providers for %s: %s",
                     symbol_classification['symbol'], fallback_providers)

        return fallback_providers

    def _fetch_fundamentals_from_providers(self, symbol: str, providers: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Fetch fundamentals data from multiple providers with enhanced error handling and retry logic.

        This method implements sophisticated retry mechanisms including:
        - Configurable retry attempts with exponential backoff
        - Rate limit detection and handling
        - Provider-specific timeout handling
        - Detailed error classification and logging

        Args:
            symbol: Normalized trading symbol
            providers: List of provider names

        Returns:
            Dictionary mapping provider names to their fundamentals data
        """
        provider_data = {}

        # Configuration for retry logic
        retry_config = {
            'max_retries': 3,
            'base_delay': 1.0,  # Base delay in seconds
            'max_delay': 30.0,  # Maximum delay in seconds
            'exponential_base': 2.0,  # Exponential backoff base
            'jitter': True  # Add random jitter to prevent thundering herd
        }

        for provider_name in providers:
            success = False

            # Validate provider availability first
            if not self._validate_single_provider_availability(provider_name):
                continue

            for attempt in range(retry_config['max_retries']):
                try:
                    _logger.debug("Fetching fundamentals for %s from %s (attempt %d/%d)",
                                symbol, provider_name, attempt + 1, retry_config['max_retries'])

                    # Get downloader with timeout handling
                    downloader = self.provider_selector.downloaders[provider_name]

                    # Fetch fundamentals with timeout
                    fundamentals = self._fetch_with_timeout(downloader, symbol, provider_name)

                    if fundamentals:
                        # Convert and validate data format
                        fundamentals_dict = self._normalize_fundamentals_data(fundamentals)
                        if fundamentals_dict and self._validate_fundamentals_data(fundamentals_dict):
                            provider_data[provider_name] = fundamentals_dict
                            _logger.debug("Successfully fetched fundamentals for %s from %s",
                                        symbol, provider_name)
                            success = True
                            break
                        else:
                            _logger.warning("Invalid fundamentals data from %s for %s", provider_name, symbol)
                    else:
                        _logger.warning("No fundamentals data returned from %s for %s", provider_name, symbol)

                except RateLimitException as e:
                    # Handle rate limiting with longer delays
                    delay = self._calculate_rate_limit_delay(e, attempt)
                    _logger.warning("Rate limit hit for %s %s, waiting %.2f seconds",
                                  provider_name, symbol, delay)
                    self._sleep_with_jitter(delay, retry_config['jitter'])
                    continue

                except TimeoutException as e:
                    # Handle timeouts with exponential backoff
                    delay = self._calculate_exponential_backoff(attempt, retry_config)
                    _logger.warning("Timeout for %s %s (attempt %d), waiting %.2f seconds: %s",
                                  provider_name, symbol, attempt + 1, delay, e)
                    if attempt < retry_config['max_retries'] - 1:
                        self._sleep_with_jitter(delay, retry_config['jitter'])
                    continue

                except NetworkException as e:
                    # Handle network errors with exponential backoff
                    delay = self._calculate_exponential_backoff(attempt, retry_config)
                    _logger.warning("Network error for %s %s (attempt %d), waiting %.2f seconds: %s",
                                  provider_name, symbol, attempt + 1, delay, e)
                    if attempt < retry_config['max_retries'] - 1:
                        self._sleep_with_jitter(delay, retry_config['jitter'])
                    continue

                except Exception as e:
                    # Handle other errors with classification
                    error_type = self._classify_error(e)
                    delay = self._calculate_exponential_backoff(attempt, retry_config)

                    _logger.warning("Error (%s) for %s %s (attempt %d): %s",
                                  error_type, provider_name, symbol, attempt + 1, e)

                    # Don't retry for certain error types
                    if error_type in ['authentication', 'invalid_symbol', 'not_supported']:
                        _logger.error("Non-retryable error for %s %s: %s", provider_name, symbol, e)
                        break

                    if attempt < retry_config['max_retries'] - 1:
                        self._sleep_with_jitter(delay, retry_config['jitter'])
                    continue

            if not success:
                _logger.error("All attempts failed for %s %s after %d retries",
                            provider_name, symbol, retry_config['max_retries'])

        return provider_data

    def _validate_single_provider_availability(self, provider_name: str) -> bool:
        """
        Validate that a single provider is available and supports fundamentals.

        Args:
            provider_name: Name of the provider to validate

        Returns:
            True if provider is available and supports fundamentals
        """
        if provider_name not in self.provider_selector.downloaders:
            _logger.warning("Downloader not available for provider: %s", provider_name)
            return False

        downloader = self.provider_selector.downloaders[provider_name]
        if not hasattr(downloader, 'get_fundamentals'):
            _logger.warning("Provider %s does not support fundamentals", provider_name)
            return False

        return True

    def _fetch_with_timeout(self, downloader, symbol: str, provider_name: str, timeout: float = 30.0):
        """
        Fetch fundamentals data with timeout handling.

        Args:
            downloader: Provider downloader instance
            symbol: Trading symbol
            provider_name: Provider name for logging
            timeout: Timeout in seconds

        Returns:
            Fundamentals data or None

        Raises:
            TimeoutException: If request times out
        """
        import signal
        import time

        def timeout_handler(signum, frame):
            raise TimeoutException(f"Request timed out after {timeout} seconds")

        # Set up timeout handling (Unix-like systems)
        if hasattr(signal, 'SIGALRM'):
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout))

        try:
            start_time = time.time()
            fundamentals = downloader.get_fundamentals(symbol)
            elapsed_time = time.time() - start_time

            _logger.debug("Fetched fundamentals for %s from %s in %.2f seconds",
                        symbol, provider_name, elapsed_time)
            return fundamentals

        finally:
            # Clean up timeout handling
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

    def _calculate_exponential_backoff(self, attempt: int, config: Dict[str, Any]) -> float:
        """
        Calculate exponential backoff delay.

        Args:
            attempt: Current attempt number (0-based)
            config: Retry configuration

        Returns:
            Delay in seconds
        """
        delay = config['base_delay'] * (config['exponential_base'] ** attempt)
        return min(delay, config['max_delay'])

    def _calculate_rate_limit_delay(self, exception: Exception, attempt: int) -> float:
        """
        Calculate delay for rate limit exceptions.

        Args:
            exception: Rate limit exception
            attempt: Current attempt number

        Returns:
            Delay in seconds
        """
        # Try to extract retry-after header if available
        if hasattr(exception, 'retry_after') and exception.retry_after is not None:
            return float(exception.retry_after)

        # Default rate limit backoff (longer than normal exponential backoff)
        base_delay = 60.0  # 1 minute base delay for rate limits
        return base_delay * (2 ** attempt)

    def _sleep_with_jitter(self, delay: float, use_jitter: bool = True) -> None:
        """
        Sleep with optional jitter to prevent thundering herd.

        Args:
            delay: Base delay in seconds
            use_jitter: Whether to add random jitter
        """
        import time
        import random

        if use_jitter:
            # Add up to 25% jitter
            jitter = delay * 0.25 * random.random()
            actual_delay = delay + jitter
        else:
            actual_delay = delay

        time.sleep(actual_delay)

    def _classify_error(self, exception: Exception) -> str:
        """
        Classify error types for appropriate retry handling.

        Args:
            exception: Exception to classify

        Returns:
            Error type string
        """
        error_message = str(exception).lower()

        # Authentication errors
        if any(term in error_message for term in ['unauthorized', 'api key', 'authentication', 'forbidden']):
            return 'authentication'

        # Invalid symbol errors
        if any(term in error_message for term in ['invalid symbol', 'symbol not found', 'not found']):
            return 'invalid_symbol'

        # Not supported errors
        if any(term in error_message for term in ['not supported', 'not available', 'not implemented']):
            return 'not_supported'

        # Rate limit errors
        if any(term in error_message for term in ['rate limit', 'too many requests', 'quota exceeded']):
            return 'rate_limit'

        # Network errors
        if any(term in error_message for term in ['connection', 'network', 'timeout', 'dns']):
            return 'network'

        # Server errors
        if any(term in error_message for term in ['server error', '500', '502', '503', '504']):
            return 'server'

        return 'unknown'

    def _validate_fundamentals_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate fundamentals data structure and content.

        Args:
            data: Fundamentals data dictionary

        Returns:
            True if data is valid
        """
        if not data or not isinstance(data, dict):
            return False

        # Check for minimum required fields
        required_fields = ['symbol']  # At minimum, should have symbol
        for field in required_fields:
            if field not in data:
                return False

        # Check for reasonable data (not all None/empty)
        non_empty_fields = sum(1 for value in data.values() if value is not None and value != '')
        if non_empty_fields < 2:  # Should have at least symbol + one other field
            return False

        return True




    def _normalize_fundamentals_data(self, fundamentals) -> Optional[Dict[str, Any]]:
        """
        Normalize fundamentals data to dictionary format.

        Args:
            fundamentals: Raw fundamentals data from provider

        Returns:
            Normalized dictionary or None if conversion fails
        """
        try:
            if isinstance(fundamentals, dict):
                return fundamentals
            elif hasattr(fundamentals, '__dict__'):
                return fundamentals.__dict__
            elif hasattr(fundamentals, '_asdict'):  # namedtuple
                return fundamentals._asdict()
            else:
                # Try to convert using vars()
                return vars(fundamentals) if hasattr(fundamentals, '__dict__') else None
        except Exception:
            _logger.exception("Failed to normalize fundamentals data:")
            return None

    def _combine_and_validate_fundamentals(self, provider_data: Dict[str, Dict[str, Any]],
                                         combination_strategy: str, data_type: str, combiner) -> Dict[str, Any]:
        """
        Combine and validate fundamentals data from multiple providers.

        Args:
            provider_data: Dictionary mapping provider names to their data
            combination_strategy: Strategy for combining data
            data_type: Type of data for validation
            combiner: Fundamentals combiner instance

        Returns:
            Combined and validated fundamentals data
        """
        try:
            if not provider_data:
                return {}

            # Combine data using specified strategy
            combined_data = combiner.combine_snapshots(provider_data, combination_strategy, data_type)

            if not combined_data:
                _logger.error("Data combination failed")
                return {}

            # Basic validation of combined data
            if not self._validate_combined_fundamentals(combined_data):
                _logger.error("Combined data failed validation")
                return {}

            return combined_data

        except Exception:
            _logger.exception("Error combining fundamentals data:")
            return {}

    def _validate_combined_fundamentals(self, data: Dict[str, Any]) -> bool:
        """
        Validate combined fundamentals data.

        Args:
            data: Combined fundamentals data

        Returns:
            True if data is valid, False otherwise
        """
        try:
            # Check if data is not empty
            if not data:
                return False

            # Check for required fields (basic validation)
            # This could be enhanced with more sophisticated validation
            return True

        except Exception:
            _logger.exception("Error validating fundamentals data:")
            return False

    def _cache_fundamentals_data(self, symbol: str, provider_data: Dict[str, Dict[str, Any]],
                               combined_data: Dict[str, Any], fundamentals_cache) -> None:
        """
        Cache fundamentals data with enhanced management.

        Args:
            symbol: Trading symbol
            provider_data: Individual provider data
            combined_data: Combined data
            fundamentals_cache: Cache instance
        """
        timestamp = datetime.now()

        # Cache individual provider data
        for provider_name, data in provider_data.items():
            try:
                fundamentals_cache.write_json(symbol, provider_name, data, timestamp)

                # Cleanup stale data for this provider
                removed_files = fundamentals_cache.cleanup_stale_data(symbol, provider_name, timestamp)
                if removed_files:
                    _logger.debug("Cleaned up %d stale files for %s %s", len(removed_files), symbol, provider_name)

            except Exception:
                _logger.exception("Failed to cache data for %s %s:", symbol, provider_name)

        # Cache combined data
        try:
            fundamentals_cache.write_json(symbol, 'combined', combined_data, timestamp)
        except Exception:
            _logger.exception("Failed to cache combined data for %s:", symbol)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()

    def clear_cache(self, symbol: Optional[str] = None, timeframe: Optional[str] = None):
        """
        Clear cache data.

        Args:
            symbol: Specific symbol to clear (None for all)
            timeframe: Specific timeframe to clear (None for all)
        """
        if symbol and timeframe:
            self.cache.clear_symbol_timeframe(symbol, timeframe)
        elif symbol:
            self.cache.clear_symbol(symbol)
        else:
            self.cache.clear_all()

        _logger.info("Cache cleared for %s %s", symbol or 'all', timeframe or 'all timeframes')


# Convenience function for easy access
def get_data_manager(cache_dir: str = DATA_CACHE_DIR, config_path: Optional[str] = None) -> DataManager:
    """
    Get a DataManager instance.

    Args:
        cache_dir: Cache directory path
        config_path: Path to provider configuration file

    Returns:
        DataManager instance
    """
    return DataManager(cache_dir, config_path)
