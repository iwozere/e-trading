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
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

import pandas as pd
from src.notification.logger import setup_logger

# Import cache and utilities
from .cache.unified_cache import UnifiedCache
from .utils.rate_limiting import RateLimiter
from .utils.retry import request_with_backoff
from .utils.validation import validate_ohlcv_data

# Import downloaders
from .downloader import (
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
from .feed import (
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

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize provider selector.

        Args:
            config_path: Path to YAML configuration file for provider rules
        """
        self.config_path = config_path or "config/data/provider_rules.yaml"
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
            _logger.warning(f"Failed to load provider rules from {self.config_path}: {e}")
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
                    if not os.getenv('ALPHA_VANTAGE_API_KEY'):
                        _logger.warning(f"Skipping {name} downloader: No API key found")
                        continue
                    self.downloaders[name] = downloader_class(api_key=os.getenv('ALPHA_VANTAGE_API_KEY'))
                elif name == 'fmp':
                    if not os.getenv('FMP_API_KEY'):
                        _logger.warning(f"Skipping {name} downloader: No API key found")
                        continue
                    self.downloaders[name] = downloader_class(api_key=os.getenv('FMP_API_KEY'))
                elif name == 'polygon':
                    if not os.getenv('POLYGON_API_KEY'):
                        _logger.warning(f"Skipping {name} downloader: No API key found")
                        continue
                    self.downloaders[name] = downloader_class(api_key=os.getenv('POLYGON_API_KEY'))
                else:
                    # For other downloaders, try to initialize without parameters
                    self.downloaders[name] = downloader_class()

                _logger.info(f"Initialized {name} downloader")
            except Exception as e:
                _logger.warning(f"Failed to initialize {name} downloader: {e}")

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

        # Default to stock for unknown symbols
        return 'stock'

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

        # Get rules for this symbol type
        if symbol_type not in self.rules:
            _logger.warning(f"No rules found for symbol type: {symbol_type}")
            return None

        rules = self.rules[symbol_type]

        # Check if timeframe is supported
        if timeframe not in rules.get('timeframes', []):
            _logger.warning(f"Timeframe {timeframe} not supported for {symbol_type}")
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

        _logger.error(f"No suitable provider found for {symbol} ({timeframe})")
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

        if symbol_type not in self.rules:
            return []

        rules = self.rules[symbol_type]
        providers = []

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
        Get comprehensive ticker information similar to TickerClassifier.

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

    def __init__(self, cache_dir: str = "d:/data-cache", config_path: Optional[str] = None):
        """
        Initialize DataManager.

        Args:
            cache_dir: Cache directory path
            config_path: Path to provider configuration file
        """
        self.cache = UnifiedCache(cache_dir)
        self.provider_selector = ProviderSelector(config_path)
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
            'alpha_vantage': {'requests_per_minute': 5, 'requests_per_day': 25},
            'fmp': {'requests_per_minute': 3000},
            'tiingo': {'requests_per_day': 1000},
            'polygon': {'requests_per_minute': 5},
            'coingecko': {'requests_per_minute': 50},
        }

        for provider, limits in rate_limits.items():
            self.rate_limiters[provider] = RateLimiter(**limits)

    @request_with_backoff(max_attempts=3, base_delay=1.0)
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

        _logger.info(f"Requesting data for {symbol} {timeframe} from {start_date} to {end_date}")

        # Check cache first (unless force refresh)
        if not force_refresh:
            cached_data = self._get_cached_data(symbol, timeframe, start_date, end_date)
            if cached_data is not None and not cached_data.empty:
                _logger.info(f"Cache hit for {symbol} {timeframe}")
                return cached_data

        # Cache miss or force refresh - get from provider
        _logger.info(f"Cache miss for {symbol} {timeframe}, fetching from provider")

        # Get providers with failover support
        providers = self.provider_selector.get_provider_with_failover(symbol, timeframe)
        if not providers:
            raise RuntimeError(f"No suitable provider found for {symbol} {timeframe}")

        # Try providers in order with failover
        last_error = None
        for provider in providers:
            try:
                _logger.info(f"Trying provider: {provider}")

                # Apply rate limiting
                if provider in self.rate_limiters:
                    self.rate_limiters[provider].wait_if_needed()

                # Get downloader
                downloader = self.provider_selector.downloaders[provider]

                # Download data
                data = downloader.get_ohlcv(symbol, timeframe, start_date, end_date)

                if data is not None and not data.empty:
                    # Validate data
                    is_valid, errors = validate_ohlcv_data(data, symbol=symbol, interval=timeframe)
                    if not is_valid:
                        _logger.warning(f"Data validation failed for {symbol} {timeframe}: {errors}")
                        # Continue with invalid data but log warning

                    # Cache the data
                    self._cache_data(data, symbol, timeframe, start_date, end_date, provider)

                    _logger.info(f"Successfully retrieved data from {provider}")
                    return data
                else:
                    _logger.warning(f"Provider {provider} returned empty data")

            except Exception as e:
                _logger.warning(f"Provider {provider} failed: {e}")
                last_error = e
                continue

        # All providers failed
        raise RuntimeError(f"All providers failed for {symbol} {timeframe}. Last error: {last_error}")

    def _get_cached_data(self, symbol: str, timeframe: str,
                        start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Get data from cache for the specified date range."""
        try:
            # Get all years in the date range
            years = list(range(start_date.year, end_date.year + 1))
            all_data = []

            for year in years:
                # Determine year start/end dates
                year_start = max(start_date, datetime(year, 1, 1))
                year_end = min(end_date, datetime(year + 1, 1, 1) - timedelta(seconds=1))

                # Try to get data from cache
                cached_df = self.cache.get(symbol, timeframe, year)
                if cached_df is not None and not cached_df.empty:
                    # Filter by date range
                    mask = (cached_df.index >= year_start) & (cached_df.index <= year_end)
                    filtered_data = cached_df[mask]
                    if not filtered_data.empty:
                        all_data.append(filtered_data)

            if all_data:
                return pd.concat(all_data, ignore_index=False).sort_index()

        except Exception as e:
            _logger.warning(f"Error retrieving cached data: {e}")

        return None

    def _cache_data(self, data: pd.DataFrame, symbol: str, timeframe: str,
                   start_date: datetime, end_date: datetime, provider: str):
        """Cache data using UnifiedCache."""
        try:
            # Cache data by year
            years = list(range(start_date.year, end_date.year + 1))

            for year in years:
                year_start = max(start_date, datetime(year, 1, 1))
                year_end = min(end_date, datetime(year + 1, 1, 1) - timedelta(seconds=1))

                # Filter data for this year
                mask = (data.index >= year_start) & (data.index <= year_end)
                year_data = data[mask]

                if not year_data.empty:
                    self.cache.put(year_data, symbol, timeframe, year_start, year_end, provider)

        except Exception as e:
            _logger.error(f"Error caching data: {e}")

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
                _logger.error(f"No suitable provider found for live feed: {symbol} {timeframe}")
                return None

            # Map provider to live feed class
            feed_classes = {
                'binance': BinanceLiveDataFeed,
                'yahoo': YahooLiveDataFeed,
                'coingecko': CoinGeckoLiveDataFeed,
            }

            if provider not in feed_classes:
                _logger.error(f"No live feed available for provider: {provider}")
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
            _logger.info(f"Created live feed for {symbol} {timeframe} using {provider}")
            return feed

        except Exception as e:
            _logger.error(f"Failed to create live feed for {symbol} {timeframe}: {e}")
            return None

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_cache_stats()

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

        _logger.info(f"Cache cleared for {symbol or 'all'} {timeframe or 'all timeframes'}")


# Convenience function for easy access
def get_data_manager(cache_dir: str = "d:/data-cache", config_path: Optional[str] = None) -> DataManager:
    """
    Get a DataManager instance.

    Args:
        cache_dir: Cache directory path
        config_path: Path to provider configuration file

    Returns:
        DataManager instance
    """
    return DataManager(cache_dir, config_path)
