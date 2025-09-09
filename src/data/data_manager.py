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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

import pandas as pd

from src.notification.logger import setup_logger

# Import API keys from donotshare configuration
try:
    from config.donotshare.donotshare import (
        ALPHA_VANTAGE_KEY,
        FMP_API_KEY,
        POLYGON_KEY,
        TWELVE_DATA_KEY,
        FINNHUB_KEY,
        TIINGO_API_KEY,
        DATA_CACHE_DIR
    )
except ImportError:
    # Fallback to environment variables if donotshare is not available
    ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY')
    FMP_API_KEY = os.getenv('FMP_API_KEY')
    POLYGON_KEY = os.getenv('POLYGON_KEY')
    TWELVE_DATA_KEY = os.getenv('TWELVE_DATA_KEY')
    FINNHUB_KEY = os.getenv('FINNHUB_KEY')
    TIINGO_API_KEY = os.getenv('TIINGO_API_KEY')
    DATA_CACHE_DIR = os.getenv('DATA_CACHE_DIR', 'd:/data-cache')

# Import cache and utilities
from src.data.cache.unified_cache import UnifiedCache
from src.data.utils.rate_limiting import RateLimiter
from src.data.utils.retry import retry_on_exception
from src.data.utils.validation import validate_ohlcv_data

# Import downloaders
from src.data.downloader import (
    BaseDataDownloader,
    BinanceDataDownloader,
    YahooDataDownloader,
    AlphaVantageDataDownloader,
    FMPDataDownloader,
    TiingoDataDownloader,
    PolygonDataDownloader,
    TwelveDataDataDownloader,
    FinnhubDataDownloader,
    CoinGeckoDataDownloader,
)

# Import live feeds
from src.data.feed import (
    BaseLiveDataFeed,
    BinanceLiveDataFeed,
    YahooLiveDataFeed,
    IBKRLiveDataFeed,
    CoinGeckoLiveDataFeed,
)

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
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
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
                'backup': ['alpha_vantage', 'polygon'],
                'timeframes': ['1m', '5m', '15m', '30m', '1h']
            },
            'stock_daily': {
                'primary': 'yahoo',
                'backup': ['tiingo', 'fmp'],
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
                    if not ALPHA_VANTAGE_KEY:
                        _logger.warning("Skipping %s downloader: No API key found", name)
                        continue
                    self.downloaders[name] = downloader_class(api_key=ALPHA_VANTAGE_KEY)
                elif name == 'fmp':
                    if not FMP_API_KEY:
                        _logger.warning("Skipping %s downloader: No API key found", name)
                        continue
                    self.downloaders[name] = downloader_class(api_key=FMP_API_KEY)
                elif name == 'polygon':
                    if not POLYGON_KEY:
                        _logger.warning("Skipping %s downloader: No API key found", name)
                        continue
                    self.downloaders[name] = downloader_class(api_key=POLYGON_KEY)
                elif name == 'coingecko':
                    # CoinGecko doesn't require API key or data_dir
                    self.downloaders[name] = downloader_class()
                elif name == 'twelvedata':
                    if not TWELVE_DATA_KEY:
                        _logger.warning("Skipping %s downloader: No API key found", name)
                        continue
                    self.downloaders[name] = downloader_class(api_key=TWELVE_DATA_KEY)
                elif name == 'finnhub':
                    if not FINNHUB_KEY:
                        _logger.warning("Skipping %s downloader: No API key found", name)
                        continue
                    self.downloaders[name] = downloader_class(api_key=FINNHUB_KEY)
                elif name == 'tiingo':
                    if not TIINGO_API_KEY:
                        _logger.warning("Skipping %s downloader: No API key found", name)
                        continue
                    self.downloaders[name] = downloader_class(api_key=TIINGO_API_KEY)
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
            if timeframe in ['1m', '5m', '15m', '30m', '1h']:
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
                    # Convert timezone-aware timestamps to timezone-naive for validation
                    data_copy = data.copy()
                    if 'timestamp' in data_copy.columns and data_copy['timestamp'].dt.tz is not None:
                        data_copy['timestamp'] = data_copy['timestamp'].dt.tz_localize(None)

                    # Validate data
                    is_valid, errors = validate_ohlcv_data(data_copy, symbol=symbol, interval=timeframe)
                    if not is_valid:
                        _logger.warning("Data validation failed for %s %s: %s", symbol, timeframe, errors)
                        # Continue with invalid data but log warning

                    # Cache the data
                    self._cache_data(data, symbol, timeframe, start_date, end_date, provider)

                    _logger.info("Successfully retrieved data from %s", provider)
                    return data
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
        """Get data from cache for the specified date range."""
        try:
            # Use UnifiedCache get method with date range
            cached_df = self.cache.get(symbol, timeframe, start_date, end_date)
            if cached_df is not None and not cached_df.empty:
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

        except Exception as e:
            _logger.error("Error caching data: %s", e)

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

        except Exception as e:
            _logger.error("Failed to create live feed for %s %s: %s", symbol, timeframe, e)
            return None

    def get_fundamentals(self, symbol: str, providers: Optional[List[str]] = None,
                        force_refresh: bool = False, combination_strategy: str = "priority_based") -> Dict[str, Any]:
        """
        Retrieve fundamentals data with caching and multi-provider combination.

        This method implements the fundamentals data retrieval flow:
        1. Check cache for valid data (7-day rule)
        2. If cache miss or force_refresh, fetch from multiple providers
        3. Combine data from multiple providers using specified strategy
        4. Cache new data and cleanup stale data
        5. Return combined fundamentals data

        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'GOOGL')
            providers: List of specific providers to use (None for auto-selection)
            force_refresh: Force refresh even if cache is valid
            combination_strategy: Strategy for combining data ('priority_based', 'quality_based', 'consensus')

        Returns:
            Dictionary containing combined fundamentals data
        """
        try:
            # Import here to avoid circular imports
            from src.data.cache.fundamentals_cache import get_fundamentals_cache
            from src.data.cache.fundamentals_combiner import get_fundamentals_combiner

            fundamentals_cache = get_fundamentals_cache(self.cache.cache_dir)
            combiner = get_fundamentals_combiner()

            # Check cache first (unless force refresh)
            if not force_refresh:
                cached_data = fundamentals_cache.find_latest_json(symbol)
                if cached_data:
                    _logger.info("Using cached fundamentals for %s from %s", symbol, cached_data.provider)
                    return fundamentals_cache.read_json(cached_data.file_path) or {}

            # Determine providers to use
            if providers is None:
                # Auto-select providers based on symbol type
                ticker_info = self.provider_selector.get_ticker_info(symbol)
                if ticker_info['symbol_type'] == 'crypto':
                    providers = ['binance', 'coingecko']
                else:
                    # Stock providers in priority order
                    providers = ['fmp', 'yfinance', 'alpha_vantage', 'ibkr']

            _logger.info("Fetching fundamentals for %s from providers: %s", symbol, providers)

            # Fetch data from multiple providers
            provider_data = {}
            successful_providers = []

            for provider_name in providers:
                try:
                    downloader = self.provider_selector.get_best_downloader(symbol, '1d')  # Use daily timeframe for fundamentals
                    if downloader and hasattr(downloader, 'get_fundamentals'):
                        fundamentals = downloader.get_fundamentals(symbol)
                        if fundamentals:
                            # Convert Fundamentals object to dictionary if needed
                            if hasattr(fundamentals, '__dict__'):
                                # It's a dataclass or object, convert to dict
                                fundamentals_dict = fundamentals.__dict__
                            elif isinstance(fundamentals, dict):
                                # It's already a dictionary
                                fundamentals_dict = fundamentals
                            else:
                                # Try to convert using vars()
                                fundamentals_dict = vars(fundamentals) if hasattr(fundamentals, '__dict__') else {}

                            provider_data[provider_name] = fundamentals_dict
                            successful_providers.append(provider_name)
                            _logger.debug("Successfully fetched fundamentals for %s from %s", symbol, provider_name)
                        else:
                            _logger.warning("No fundamentals data returned from %s for %s", provider_name, symbol)
                    else:
                        _logger.warning("Downloader for %s does not support fundamentals", provider_name)

                except Exception as e:
                    _logger.error("Failed to fetch fundamentals from %s for %s: %s", provider_name, symbol, e)
                    continue

            if not provider_data:
                _logger.error("No fundamentals data available for %s from any provider", symbol)
                return {}

            # Combine data from multiple providers
            combined_data = combiner.combine_snapshots(provider_data, combination_strategy)

            if not combined_data:
                _logger.error("Failed to combine fundamentals data for %s", symbol)
                return {}

            # Cache the combined data for each successful provider
            timestamp = datetime.now()
            for provider_name in successful_providers:
                try:
                    # Cache individual provider data
                    fundamentals_cache.write_json(symbol, provider_name, provider_data[provider_name], timestamp)

                    # Cleanup stale data for this provider
                    removed_files = fundamentals_cache.cleanup_stale_data(symbol, provider_name, timestamp)
                    if removed_files:
                        _logger.info("Cleaned up %d stale cache files for %s %s", len(removed_files), symbol, provider_name)

                except Exception as e:
                    _logger.error("Failed to cache fundamentals for %s %s: %s", symbol, provider_name, e)

            # Cache the combined data as well
            try:
                fundamentals_cache.write_json(symbol, 'combined', combined_data, timestamp)
            except Exception as e:
                _logger.error("Failed to cache combined fundamentals for %s: %s", symbol, e)

            _logger.info("Successfully retrieved and combined fundamentals for %s from %d providers",
                        symbol, len(successful_providers))

            return combined_data

        except Exception as e:
            _logger.error("Error retrieving fundamentals for %s: %s", symbol, e)
            return {}

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
