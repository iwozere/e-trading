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

OHLCV contract (see ``src.data.ohlcv_contract``): ``get_ohlcv`` / ``get_ohlcv_batch`` return
time on a **tz-naive DatetimeIndex** with columns ``open``, ``high``, ``low``, ``close``, ``volume``
(lowercase). There is **no** ``timestamp`` column after normalization—the index *is* the time axis.
Some call sites reset_index locally or use ``coerce_ohlcv_timestamp_column`` when they need a column.
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


from src.error_handling.exceptions import RateLimitException, NetworkException
class TimeoutException(Exception):
    """Exception raised when requests timeout."""
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
from src.data.downloader.data_downloader_factory import DataDownloaderFactory



# Import live feeds
from src.data.feed.base_live_data_feed import BaseLiveDataFeed
from src.data.feed.binance_live_feed import BinanceLiveDataFeed
from src.data.feed.yahoo_live_feed import YahooLiveDataFeed
from src.data.feed.coingecko_live_feed import CoinGeckoLiveDataFeed

_logger = setup_logger(__name__)


from src.data.provider_selector import ProviderSelector  # noqa: F401

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
        self._rate_limited_providers = set()  # Session-level blacklist (legacy)
        self._provider_cooldowns = {}         # Temporary cooldowns: {provider_name: expiration_time}


        # Initialize rate limiters for each provider
        self._initialize_rate_limiters()

        _logger.info("DataManager initialized successfully")

    # Conservative default rate limit applied to any provider not listed below.
    # Prevents accidental API bans when new providers are added without an explicit limit.
    _DEFAULT_RATE_LIMIT = {'requests_per_minute': 10}

    # Override via PROVIDER_RATE_LIMITS env var (JSON) for runtime tuning without code changes.
    _BUILTIN_RATE_LIMITS = {
        'binance': {'requests_per_minute': 1200},
        'yahoo': {'requests_per_minute': 100},
        'alpha_vantage': {'requests_per_minute': 5},
        'fmp': {'requests_per_minute': 3000},
        'tiingo': {'requests_per_minute': 100},
        'polygon': {'requests_per_minute': 5},
        'coingecko': {'requests_per_minute': 50},
        'alpaca': {'requests_per_minute': 200},
        'eodhd': {'requests_per_minute': 100},
        'twelvedata': {'requests_per_minute': 55},
        'finnhub': {'requests_per_minute': 60},
    }

    def _initialize_rate_limiters(self):
        """Initialize rate limiters for each provider with config-driven overrides."""
        import json, os
        rate_limits = dict(self._BUILTIN_RATE_LIMITS)

        env_overrides = os.getenv("PROVIDER_RATE_LIMITS")
        if env_overrides:
            try:
                rate_limits.update(json.loads(env_overrides))
                _logger.info("Applied PROVIDER_RATE_LIMITS overrides from env")
            except (json.JSONDecodeError, TypeError):
                _logger.warning("PROVIDER_RATE_LIMITS env var is not valid JSON; using built-in limits")

        for provider, limits in rate_limits.items():
            self.rate_limiters[provider] = RateLimiter(**limits)

    @retry_on_exception(max_attempts=3, base_delay=1.0)
    def get_ohlcv(self, symbol: str, timeframe: str,
                  start_date: datetime, end_date: datetime,
                  force_refresh: bool = False) -> pd.DataFrame:
        """
        Retrieve historical OHLCV data with caching, gap detection, and provider selection.

        This method implements the main data retrieval flow:
        1. Load whatever cache exists for the requested range
        2. Detect prefix, suffix, and intermediate gaps
        3. Fetch only the missing segments from providers
        4. Cache new data (merge-safe) and return the combined result

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT', 'AAPL')
            timeframe: Data timeframe (e.g., '1m', '1h', '1d')
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            force_refresh: Force refresh from provider, bypassing cache

        Returns:
            Normalized OHLCV (``src.data.ohlcv_contract``): tz-naive ``DatetimeIndex`` and
            columns ``open`` … ``volume``. Cache-only, download-only, and merged paths share this shape.

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

        # --- Step 1: Load cache and detect gaps ---
        missing_segments = []  # list of (start, end) tuples
        cached_data = None

        if not force_refresh:
            cached_data = self.cache.get(symbol, timeframe, start_date, end_date)

        if force_refresh or cached_data is None or cached_data.empty:
            # No cache at all — fetch the entire range
            missing_segments.append((start_date, end_date))
            cached_data = None
        else:
            # Cache exists — detect gaps
            bar_duration = self._get_bar_duration(timeframe)
            # Covers overnight/weekend gaps so EOD runs don't trigger suffix
            # re-fetches just because end_date extends past the last trading bar.
            # 1d=8 covers weekends (3d) plus known extended closures (Sandy=5d, 9/11=7d).
            # 1h=18 bars (~overnight+buffer), sub-hour scaled.
            _TOLERANCE = {'1d': 8.0, '1h': 18.0, '30m': 36.0, '15m': 72.0, '5m': 216.0}
            tolerance_factor = _TOLERANCE.get(timeframe, 4.0)

            cache_start = cached_data.index[0]
            cache_end = cached_data.index[-1]

            # Make timezone-aware for comparison
            safe_start = start_date.replace(tzinfo=timezone.utc) if start_date.tzinfo is None else start_date
            safe_end = end_date.replace(tzinfo=timezone.utc) if end_date.tzinfo is None else end_date
            if cache_start.tzinfo is None:
                cache_start = cache_start.replace(tzinfo=timezone.utc)
            if cache_end.tzinfo is None:
                cache_end = cache_end.replace(tzinfo=timezone.utc)

            # Prefix gap
            if (cache_start - safe_start) > (bar_duration * tolerance_factor):
                missing_segments.append((safe_start, cache_start - bar_duration))
                _logger.info("Gap detected: PREFIX for %s (%s to %s)", symbol, safe_start, cache_start)

            # Suffix gap (stale data)
            if (safe_end - cache_end) > (bar_duration * tolerance_factor):
                missing_segments.append((cache_end + bar_duration, safe_end))
                _logger.info("Gap detected: SUFFIX for %s (%s to %s)", symbol, cache_end, safe_end)

            # Intermediate gaps
            diffs = cached_data.index.to_series().diff().dropna()
            large_gaps = diffs[diffs > (bar_duration * tolerance_factor)]
            for gap_end_ts, duration in large_gaps.items():
                gap_start_ts = gap_end_ts - duration + bar_duration
                missing_segments.append((gap_start_ts.to_pydatetime(), gap_end_ts.to_pydatetime() - bar_duration))
                _logger.info("Gap detected: INTERMEDIATE for %s (%s to %s)", symbol, gap_start_ts, gap_end_ts)

        # --- Step 2: If no gaps, return cache ---
        if not missing_segments:
            _logger.info("Cache hit for %s %s (no gaps): %d rows", symbol, timeframe, len(cached_data))
            return cached_data

        # --- Step 3: Fetch missing segments from providers ---
        _logger.info("Fetching %d missing segment(s) for %s %s", len(missing_segments), symbol, timeframe)

        providers = self.provider_selector.get_provider_with_failover(symbol, timeframe)
        if not providers:
            if cached_data is not None and not cached_data.empty:
                _logger.warning("No providers available for %s, returning partial cache", symbol)
                return cached_data
            raise RuntimeError(f"No suitable provider found for {symbol} {timeframe}")

        fetched_frames = []
        for seg_start, seg_end in missing_segments:
            segment_fetched = False
            for provider in providers:
                try:
                    if provider in self.rate_limiters:
                        self.rate_limiters[provider].wait_if_needed()
                    else:
                        # Default-deny: apply a conservative limiter for unknown providers
                        # to prevent accidental API bans until an explicit limit is configured.
                        if not hasattr(self, '_default_rate_limiter'):
                            self._default_rate_limiter = RateLimiter(**self._DEFAULT_RATE_LIMIT)
                        self._default_rate_limiter.wait_if_needed()
                        _logger.debug("Applying default rate limit (10 rpm) for unconfigured provider: %s", provider)

                    downloader = self.provider_selector.downloaders[provider]
                    data = downloader.get_ohlcv(symbol, timeframe, seg_start, seg_end)

                    if data is not None and not data.empty:
                        data_copy = self._normalize_ohlcv(data)
                        # Cache the segment (put now merges safely)
                        self._cache_data(data_copy, symbol, timeframe, seg_start, seg_end, provider)
                        fetched_frames.append(data_copy)
                        segment_fetched = True
                        _logger.info("Filled gap for %s from %s (%s to %s, %d rows)",
                                     symbol, provider, seg_start, seg_end, len(data_copy))
                        break
                    else:
                        _logger.warning("Provider %s returned empty data for gap %s-%s", provider, seg_start, seg_end)
                except Exception as e:
                    _logger.warning("Provider %s failed for gap %s-%s: %s", provider, seg_start, seg_end, e)
                    continue

            if not segment_fetched:
                _logger.warning("Could not fill gap %s-%s for %s from any provider", seg_start, seg_end, symbol)

        # --- Step 4: Merge cache + newly fetched data ---
        all_frames = []
        if cached_data is not None and not cached_data.empty:
            all_frames.append(cached_data)
        all_frames.extend(fetched_frames)

        if not all_frames:
            raise RuntimeError(f"All providers failed for {symbol} {timeframe} and no cache available.")

        result = pd.concat(all_frames)
        result = result[~result.index.duplicated(keep='last')].sort_index()
        _logger.info("Returning %d rows for %s %s (gaps filled)", len(result), symbol, timeframe)
        return result

    def get_ohlcv_batch(self, symbols: List[str], timeframe: str,
                        start_date: datetime, end_date: datetime,
                        force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Retrieve historical OHLCV data for multiple symbols, utilizing cache where possible.
        
        This method optimizes for batch retrieval (like daily pipeline runs). It checks the
        UnifiedCache for each symbol. If a symbol is missing data, it calculates the required
        delta date range, groups all missing symbols, and downloads them in a single batch
        using YahooDataDownloader, then caches the newly fetched deltas.

        Args:
            symbols: List of trading symbols
            timeframe: Data interval (e.g. '1d')
            start_date: Request start date
            end_date: Request end date
            force_refresh: If True, bypasses cache and forces a full re-download.

        Returns:
            Dictionary mapping symbol to OHLCV DataFrames in the **normalized contract**
            (tz-naive ``DatetimeIndex``; columns ``open`` … ``volume``). Cache hits return the
            same shape as freshly merged downloads. See ``src.data.ohlcv_contract``.
        """
        results = {}
        missing_ranges = {}  # {symbol: (missing_start, end_date)}

        _logger.info("Batch request for %d symbols (%s) from %s to %s", len(symbols), timeframe, start_date, end_date)

        # 1. Check Cache for all symbols
        # 1. Check Cache and Identify Gaps for all symbols
        for sym in symbols:
            # Step A: Load whatever we have in cache for the full requested range
            # Note: UnifiedCache.get already handles multiple years
            cached_df = self.cache.get(sym, timeframe, start_date, end_date)
            
            if force_refresh or cached_df is None or cached_df.empty:
                missing_ranges.setdefault(sym, []).append((start_date, end_date))
                results[sym] = pd.DataFrame()
                continue
            
            results[sym] = cached_df
            
            # Step B: Identify Prefix Gap (missing data at start)
            cache_start = cached_df.index[0]
            if cache_start.tzinfo is None:
                cache_start = cache_start.replace(tzinfo=timezone.utc)
            
            safe_start = start_date.replace(tzinfo=timezone.utc) if start_date.tzinfo is None else start_date
            
            # Tolerance: covers overnight/weekend gaps (same table as get_ohlcv).
            # 1d=8 covers weekends (3d) plus known extended closures (Sandy=5d, 9/11=7d).
            bar_duration = self._get_bar_duration(timeframe)
            _TOLERANCE = {'1d': 8.0, '1h': 18.0, '30m': 36.0, '15m': 72.0, '5m': 216.0}
            tolerance_factor = _TOLERANCE.get(timeframe, 4.0)
            
            if (cache_start - safe_start) > (bar_duration * tolerance_factor):
                missing_ranges.setdefault(sym, []).append((safe_start, cache_start - bar_duration))
            
            # Step C: Identify Suffix Gap (missing data at end)
            cache_end = cached_df.index[-1]
            if cache_end.tzinfo is None:
                cache_end = cache_end.replace(tzinfo=timezone.utc)
            
            safe_end = end_date.replace(tzinfo=timezone.utc) if end_date.tzinfo is None else end_date
            
            if (safe_end - cache_end) > (bar_duration * tolerance_factor):
                missing_ranges.setdefault(sym, []).append((cache_end + bar_duration, safe_end))
                
            # Step D: Identify Intermediate Gaps (gaps in the middle)
            # Use diff() to find gaps > tolerance_factor * bar_duration
            diffs = cached_df.index.to_series().diff().dropna()
            large_gaps = diffs[diffs > (bar_duration * tolerance_factor)]
            for gap_end_ts, duration in large_gaps.items():
                gap_start_ts = gap_end_ts - duration + bar_duration
                # Groups these for later batch download
                missing_ranges.setdefault(sym, []).append((gap_start_ts.to_pydatetime(), gap_end_ts.to_pydatetime() - bar_duration))

        if not missing_ranges:
            _logger.info("All %d symbols loaded from cache. No batch download required.", len(symbols))
            return results

        # 2. Group missing ranges by their dates to optimize batching
        ranges_to_symbols = {}
        for sym, segments in missing_ranges.items():
            for m_start, m_end in segments:
                date_key = (m_start.strftime('%Y-%m-%d'), m_end.strftime('%Y-%m-%d'))
                ranges_to_symbols.setdefault(date_key, []).append(sym)

        # 3. Download and cache deltas
        _logger.info("Fetching missing data for %d symbols across %d date ranges...", len(missing_ranges), len(ranges_to_symbols))
        
        # We explicitly use YahooDataDownloader for batching as it supports yf.download
        yahoo_dl = self.provider_selector._initialize_downloader("yahoo")

        for (start_str, end_str), batch_symbols in ranges_to_symbols.items():
            b_start = datetime.strptime(start_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            b_end = datetime.strptime(end_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            
            try:
                # Download batch via Yahoo
                _logger.info("Batch downloading %d symbols from %s to %s", len(batch_symbols), b_start, b_end)
                batch_df = yahoo_dl.get_ohlcv_batch(batch_symbols, timeframe, b_start, b_end)
                
                # Yahoo returns MultiIndex if > 1 symbol, or single index if 1 symbol, but get_ohlcv_batch normalizes
                # it internally and returns a joined dataframe with 'ticker' column, OR a dict depending on implementation.
                # NOTE: We know YahooDataDownloader.get_ohlcv_batch currently returns a combined flat DataFrame 
                # if group_by='ticker' wasn't unpacked. Let's unpack it correctly.
                
                if isinstance(batch_df, dict):
                    raw_dict = batch_df
                else:
                    # If it's a flat df with a generic multi-index
                    raise NotImplementedError("Expected YahooDataDownloader.get_ohlcv_batch to return a dict, need to adapt if returns DataFrame")

                for sym, df_data in raw_dict.items():
                    if df_data is None or df_data.empty:
                        continue
                        
                    # Normalize columns using shared helper
                    df_copy = self._normalize_ohlcv(df_data)

                    # Cache the new delta segment
                    self._cache_data(df_copy, sym, timeframe, b_start, b_end, 'yahoo')
                    
                    # Merge with existing cache if we had a partial hit
                    if sym in results and not results[sym].empty:
                        # Append and deduplicate
                        merged = pd.concat([results[sym], df_copy])
                        merged = merged[~merged.index.duplicated(keep='last')].sort_index()
                        results[sym] = merged
                    else:
                        results[sym] = df_copy
            
            except Exception as e:
                _logger.error("Failed batch download for a group: %s", e)

        return results

    def get_funding_rate(self, symbol: str,
                         start_date: datetime, end_date: datetime,
                         force_refresh: bool = False) -> pd.DataFrame:
        """
        Retrieve historical funding rate data with caching.
        """
        data_type = 'funding_rate'
        timeframe = '8h'  # Funding rates are typically every 8 hours on Binance
        
        # Check cache
        if not force_refresh:
            cached_data = self.cache.get(symbol, timeframe, start_date, end_date, data_type=data_type)
            if cached_data is not None and not cached_data.empty:
                _logger.info("Cache hit for %s %s (%s)", symbol, timeframe, data_type)
                return cached_data

        # Fetch from Binance
        _logger.info("Cache miss for %s %s (%s), fetching from Binance", symbol, timeframe, data_type)
        downloader = self.provider_selector._initialize_downloader('binance')
        if not downloader:
             raise RuntimeError("Binance downloader not available")

        # Download
        data = downloader.get_funding_rate_history(symbol, start_date, end_date)
        
        if data is not None and not data.empty:
            # Normalize index
            if not isinstance(data.index, pd.DatetimeIndex) and 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data = data.set_index('timestamp')
            
            # Cache
            self.cache.put(data, symbol, timeframe, start_date, end_date, 'binance', data_type=data_type)
            return data
            
        return pd.DataFrame()

    def get_open_interest(self, symbol: str, period: str,
                          start_date: datetime, end_date: datetime,
                          force_refresh: bool = False) -> pd.DataFrame:
        """
        Retrieve historical open interest data with caching.
        """
        data_type = 'open_interest'
        
        # Check cache
        if not force_refresh:
            cached_data = self.cache.get(symbol, period, start_date, end_date, data_type=data_type)
            if cached_data is not None and not cached_data.empty:
                _logger.info("Cache hit for %s %s (%s)", symbol, period, data_type)
                return cached_data

        # Fetch from Binance
        _logger.info("Cache miss for %s %s (%s), fetching from Binance", symbol, period, data_type)
        downloader = self.provider_selector._initialize_downloader('binance')
        if not downloader:
             raise RuntimeError("Binance downloader not available")

        # Download
        data = downloader.get_open_interest_history(symbol, period, start_date, end_date)
        
        if data is not None and not data.empty:
            # Normalize index
            if not isinstance(data.index, pd.DatetimeIndex) and 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data = data.set_index('timestamp')
            
            # Cache
            self.cache.put(data, symbol, period, start_date, end_date, 'binance', data_type=data_type)
            return data
            
        return pd.DataFrame()

    def get_long_short_ratio(self, symbol: str, period: str,
                             start_date: datetime, end_date: datetime,
                             force_refresh: bool = False) -> pd.DataFrame:
        """
        Retrieve historical long/short ratio data with caching.
        """
        data_type = 'long_short_ratio'
        
        # Check cache
        if not force_refresh:
            cached_data = self.cache.get(symbol, period, start_date, end_date, data_type=data_type)
            if cached_data is not None and not cached_data.empty:
                _logger.info("Cache hit for %s %s (%s)", symbol, period, data_type)
                return cached_data

        # Fetch from Binance
        _logger.info("Cache miss for %s %s (%s), fetching from Binance", symbol, period, data_type)
        downloader = self.provider_selector._initialize_downloader('binance')
        if not downloader:
             raise RuntimeError("Binance downloader not available")

        # Download
        data = downloader.get_long_short_ratio(symbol, period, start_date, end_date)
        
        if data is not None and not data.empty:
            # Normalize index
            if not isinstance(data.index, pd.DatetimeIndex) and 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data = data.set_index('timestamp')
            
            # Cache
            self.cache.put(data, symbol, period, start_date, end_date, 'binance', data_type=data_type)
            return data
            
        return pd.DataFrame()

    def _get_bar_duration(self, timeframe: str) -> timedelta:
        """Helper to get expected duration of one bar."""
        bar_durations = {
            '1m': timedelta(minutes=1),
            '5m': timedelta(minutes=5),
            '15m': timedelta(minutes=15),
            '30m': timedelta(minutes=30),
            '1h': timedelta(hours=1),
            '4h': timedelta(hours=4),
            '1d': timedelta(days=1),
            '1w': timedelta(days=7),
            '1M': timedelta(days=30),
        }
        return bar_durations.get(timeframe, timedelta(hours=24))

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

                # Coverage check: Does cached data reach the requested end date?
                # We allow for one bar of tolerance based on timeframe
                expected_end = end_date_naive

                # Heuristic for bar duration
                bar_durations = {
                    '1m': timedelta(minutes=1),
                    '5m': timedelta(minutes=5),
                    '15m': timedelta(minutes=15),
                    '30m': timedelta(minutes=30),
                    '1h': timedelta(hours=1),
                    '4h': timedelta(hours=4),
                    '1d': timedelta(days=1),
                }
                tolerance = bar_durations.get(timeframe, timedelta(hours=1))

                # If the gap between latest cached and requested end is larger than one bar,
                # it means the cache is incomplete at the tail.
                if (expected_end - latest_cached_date.replace(tzinfo=None)) > tolerance:
                    # Special case: don't force fetch if the requested end_date is in the future
                    # and we already have data up to very recently
                    if not is_requesting_recent_data:
                        _logger.info(
                            "Cache for %s %s is incomplete (last bar: %s, requested: %s). Forcing fetch.",
                            symbol, timeframe, latest_cached_date, expected_end
                        )
                        return None

                _logger.info("Cache hit for %s %s: %d rows", symbol, timeframe, len(cached_df))
                return cached_df

        except Exception as e:
            _logger.warning("Error retrieving cached data: %s", e)

        return None

    def _normalize_ohlcv(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize raw OHLCV from any provider into the **single in-memory contract** used for
        cache read/write and merged batches (same shape for downloaded vs cached rows).

        Output shape:
            - Index: tz-naive ``DatetimeIndex`` (time axis; not a ``timestamp`` column).
            - Columns: ``open``, ``high``, ``low``, ``close``, ``volume`` (lowercased).

        Callers that need a ``timestamp`` column (e.g. legacy sorting / TRF masks) should use
        ``src.ml.pipeline.shared.ohlcv_timestamp.coerce_ohlcv_timestamp_column`` or
        ``df.reset_index(names="timestamp")`` at the point of use.

        Used by ``get_ohlcv`` gap fills and ``get_ohlcv_batch`` after downloads.

        Args:
            data: Raw DataFrame from a provider (column or index time).

        Returns:
            DataFrame indexed by normalized time; see ``src.data.ohlcv_contract``.
        """
        data_copy = data.copy()

        # If index is datetime and no 'timestamp' column, create it from index
        if 'timestamp' not in data_copy.columns and isinstance(data_copy.index, pd.DatetimeIndex):
            ts_index = data_copy.index
            if ts_index.tz is not None:
                ts_index = ts_index.tz_localize(None)
            data_copy.insert(0, 'timestamp', ts_index)

        # Lowercase all columns
        rename_map = {c: c.lower() for c in data_copy.columns}
        data_copy = data_copy.rename(columns=rename_map)

        # Ensure required columns exist
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required_cols if c not in data_copy.columns]
        if missing:
            _logger.warning("Missing required columns after normalization: %s", missing)

        # Make timestamp tz-naive and set as index
        if 'timestamp' in data_copy.columns:
            data_copy['timestamp'] = pd.to_datetime(data_copy['timestamp'], errors='coerce')
            if data_copy['timestamp'].dt.tz is not None:
                data_copy['timestamp'] = data_copy['timestamp'].dt.tz_localize(None)
            data_copy = data_copy.set_index('timestamp')

        return data_copy

    def _cache_data(self, data: pd.DataFrame, symbol: str, timeframe: str,
                   start_date: datetime, end_date: datetime, provider: str):
        """Cache data using UnifiedCache."""
        try:
            # Use UnifiedCache put method with the full data and date range
            success = self.cache.put(data, symbol, timeframe, start_date, end_date, provider)
            if not success:
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
                        data_type: str = "general", max_age_days: Optional[int] = None) -> Dict[str, Any]:
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

            # 2. Cache validation with data-type specific TTL (or override)
            if not force_refresh:
                cached_data = self._get_cached_fundamentals(normalized_symbol, data_type, fundamentals_cache, max_age_days=max_age_days)
                if cached_data:
                    return self._enrich_combined_fundamentals_output(
                        cached_data,
                        provider_data=None,
                        symbol=normalized_symbol,
                        fundamentals_cache=fundamentals_cache,
                        max_age_days=max_age_days,
                    )

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
                # Try to return cached data as fallback (ignore TTL if we are in fallback mode?)
                # Actually let's use the provided max_age_days if any
                cached_fallback = fundamentals_cache.find_latest_json(normalized_symbol, data_type=data_type, max_age_days=max_age_days)
                if cached_fallback:
                    _logger.info("Returning stale cached data as fallback for %s", normalized_symbol)
                    return fundamentals_cache.read_json(cached_fallback.file_path) or {}
                return {}

            # 5. Data combination and validation
            combined_data = self._combine_and_validate_fundamentals(provider_data, combination_strategy, data_type, combiner)

            if not combined_data:
                _logger.error("Failed to combine fundamentals data for %s", normalized_symbol)
                return {}

            combined_data = self._enrich_combined_fundamentals_output(
                combined_data,
                provider_data=provider_data,
                symbol=normalized_symbol,
                fundamentals_cache=fundamentals_cache,
                max_age_days=max_age_days,
            )

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

    def _get_cached_fundamentals(self, symbol: str, data_type: str, fundamentals_cache, max_age_days: Optional[int] = None) -> Optional[Dict[str, Any]]:
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
            cached_metadata = fundamentals_cache.find_latest_json(symbol, data_type=data_type, max_age_days=max_age_days)
            if not cached_metadata:
                _logger.debug("No cached data found for %s %s (max_age=%s)", symbol, data_type, max_age_days)
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
            if not self.provider_selector._initialize_downloader(provider_name):
                _logger.debug("Provider %s not available (failed initialization or no keys)", provider_name)
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
            # For statements, prefer Alpha Vantage and Yahoo
            fallback_candidates = ['alpha_vantage', 'yahoo', 'twelvedata', 'fmp']
        elif data_type in ratio_types:
            # For ratios, prefer Yahoo Finance and Alpha Vantage
            fallback_candidates = ['yahoo', 'alpha_vantage', 'twelvedata', 'fmp']
        elif data_type in profile_types:
            # For profiles, prefer Yahoo Finance and Finnhub
            fallback_candidates = ['yahoo', 'finnhub', 'alpha_vantage', 'twelvedata', 'fmp']
        elif data_type in calendar_types:
            # For calendar events, prefer Yahoo Finance
            fallback_candidates = ['yahoo', 'finnhub', 'fmp', 'alpha_vantage']
        elif data_type in dividend_types:
            # For dividends, prefer Yahoo Finance
            fallback_candidates = ['yahoo', 'finnhub', 'fmp', 'alpha_vantage']
        else:
            # Default fallback for unknown data types
            fallback_candidates = ['yahoo', 'finnhub', 'alpha_vantage', 'twelvedata', 'fmp']

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
            # Check if provider is available and initialize if needed
            if not self.provider_selector._initialize_downloader(provider):
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

    def _is_provider_on_cooldown(self, provider: str) -> bool:
        """
        Check if a provider is currently on cooldown due to rate limiting.

        Args:
            provider: Provider name

        Returns:
            True if provider is on cooldown, False otherwise
        """
        if provider in self._provider_cooldowns:
            expiration = self._provider_cooldowns[provider]
            if datetime.now() < expiration:
                _logger.debug("Provider %s is on cooldown until %s", provider, expiration)
                return True
            else:
                # Cooldown expired, remove from dictionary
                _logger.info("Cooldown for provider %s has expired", provider)
                del self._provider_cooldowns[provider]

        # Also check the legacy session-level blacklist
        if provider in self._rate_limited_providers:
            _logger.debug("Provider %s is blacklisted for the session", provider)
            return True

        return False

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
            # Skip if provider is on cooldown (replaces permanent session blacklist check)
            if self._is_provider_on_cooldown(provider):
                continue


            # Check availability and initialize if needed
            if not self.provider_selector._initialize_downloader(provider):
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
                _logger.info("Provider %s not compatible with %s: %s",
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
                if provider_info['provider'] in ['yahoo', 'twelvedata', 'alpha_vantage', 'finnhub']:
                    base_score += 1.0  # Boost for international-friendly providers
                elif provider_info['provider'] in ['fmp', 'alpaca', 'tiingo']:
                    base_score -= 0.5  # Penalty for US-only providers
            else:
                # US symbol adjustments
                if provider_info['provider'] in ['fmp', 'alpaca', 'tiingo', 'yahoo', 'finnhub', 'alpha_vantage', 'twelvedata']:
                    base_score += 0.5  # Boost for US-optimized providers

            # Symbol type adjustments
            if symbol_type == 'etf':
                if provider_info['provider'] in ['yahoo', 'fmp']:
                    base_score += 0.3  # ETFs work well with these providers
            elif symbol_type == 'reit':
                if provider_info['provider'] in ['yahoo', 'fmp']:
                    base_score += 0.3  # REITs work well with these providers

            # Market-specific adjustments
            if market == 'EU':
                if provider_info['provider'] in ['yahoo', 'twelvedata']:
                    base_score += 0.5  # Better EU coverage
            elif market == 'UK':
                if provider_info['provider'] in ['yahoo', 'alpha_vantage']:
                    base_score += 0.5  # Better UK coverage
            elif market == 'ASIA':
                if provider_info['provider'] == 'yahoo':
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
            'yahoo': {
                'symbol_types': ['stock', 'etf', 'reit'],
                'markets': ['US', 'UK', 'EU', 'CANADA', 'ASIA', 'OCEANIA'],
                'exchanges': ['NYSE', 'NASDAQ', 'LSE', 'TSX', 'AMS', 'EPA', 'XETRA', 'HKEX', 'TSE'],
                'international_support': True,
                'strengths': ['international_coverage', 'calculated_ratios', 'ttm_metrics'],
                'limitations': ['scraping_based', 'occasional_outages'],
                'quality_score': 5
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
                'quality_score': 5
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
                'quality_score': 5
            },
            'finnhub': {
                'symbol_types': ['stock', 'etf'],
                'markets': ['US', 'UK', 'EU'],
                'exchanges': ['NYSE', 'NASDAQ', 'LSE', 'XETRA'],
                'international_support': True,
                'strengths': ['real_time_data', 'news_integration'],
                'limitations': ['limited_fundamentals', 'rate_limits'],
                'quality_score': 5
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
            # Skip if provider is on cooldown
            if self._is_provider_on_cooldown(provider_name):
                continue

            # Enforce rate limiter check/wait (staggers concurrent requests across threads)
            if provider_name in self.rate_limiters:
                self.rate_limiters[provider_name].wait_if_needed()

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
                    # Handle rate limiting with temporary cooldowns
                    delay = self._calculate_rate_limit_delay(e, attempt)
                    
                    # Set a cooldown for the provider
                    # For strong rate limits (long delay) or persistent failures, set a longer cooldown
                    cooldown_seconds = 3600 if delay > 60 or attempt >= 1 else 60
                    self._set_provider_cooldown(provider_name, cooldown_seconds)
                    
                    _logger.warning("Rate limit hit for %s %s, setting %ds cooldown and moving to next provider",
                                  provider_name, symbol, cooldown_seconds)
                    break # Stop retrying and move to next provider


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

    def _set_provider_cooldown(self, provider: str, seconds: int):
        """
        Set a temporary cooldown for a provider.

        Args:
            provider: Provider name
            seconds: Cooldown duration in seconds
        """
        expiration = datetime.now() + timedelta(seconds=seconds)
        self._provider_cooldowns[provider] = expiration
        _logger.info("Provider %s put on cooldown for %d seconds (until %s)",
                    provider, seconds, expiration.strftime("%H:%M:%S"))


    def _validate_single_provider_availability(self, provider_name: str) -> bool:
        """
        Validate that a single provider is available and supports fundamentals.

        Args:
            provider_name: Name of the provider to validate

        Returns:
            True if provider is available and supports fundamentals
        """
        # Check availability and initialize if needed
        if not self.provider_selector._initialize_downloader(provider_name):
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
        import threading

        def timeout_handler(signum, frame):
            raise TimeoutException(f"Request timed out after {timeout} seconds")

        # Set up timeout handling (Unix-like systems, main thread only)
        sigalrm_set = False
        if hasattr(signal, 'SIGALRM') and threading.current_thread() is threading.main_thread():
            try:
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(timeout))
                sigalrm_set = True
            except ValueError:
                pass

        try:
            start_time = time.time()
            fundamentals = downloader.get_fundamentals(symbol)
            elapsed_time = time.time() - start_time

            _logger.debug("Fetched fundamentals for %s from %s in %.2f seconds",
                        symbol, provider_name, elapsed_time)
            return fundamentals

        finally:
            # Clean up timeout handling
            if sigalrm_set:
                try:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                except ValueError:
                    pass

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

    @staticmethod
    def _is_positive_fundamental_scalar(value: Any) -> bool:
        try:
            return value is not None and float(value) > 0
        except (TypeError, ValueError):
            return False

    def _market_cap_from_fundamentals_payload(self, payload: Dict[str, Any]) -> Optional[float]:
        """Resolve market cap from flat or FMP-style nested fundamentals dict."""
        if not isinstance(payload, dict):
            return None
        for key in ("market_cap", "marketCap"):
            v = payload.get(key)
            if self._is_positive_fundamental_scalar(v):
                return float(v)
        profile = payload.get("profile")
        if isinstance(profile, dict):
            for key in ("marketCap", "market_cap", "mktCap"):
                v = profile.get(key)
                if self._is_positive_fundamental_scalar(v):
                    return float(v)
        metrics = payload.get("metrics")
        if isinstance(metrics, dict):
            v = metrics.get("marketCap")
            if self._is_positive_fundamental_scalar(v):
                return float(v)
        return None

    def _avg_volume_from_fundamentals_payload(self, payload: Dict[str, Any]) -> Optional[float]:
        """Resolve average daily volume from flat or nested profile/metrics."""
        if not isinstance(payload, dict):
            return None
        for key in ("avg_volume", "average_volume", "averageVolume", "avgVolume"):
            v = payload.get(key)
            if v is not None and self._is_positive_fundamental_scalar(v):
                return float(v)
        profile = payload.get("profile")
        if isinstance(profile, dict):
            for key in ("averageVolume", "averageVolume10days", "avgVolume", "volAvg"):
                v = profile.get(key)
                if v is not None and self._is_positive_fundamental_scalar(v):
                    return float(v)
        metrics = payload.get("metrics")
        if isinstance(metrics, dict):
            for key in ("averageVolume", "averageVolume10days"):
                v = metrics.get(key)
                if v is not None and self._is_positive_fundamental_scalar(v):
                    return float(v)
        return None

    def _enrich_combined_fundamentals_output(
        self,
        combined: Dict[str, Any],
        provider_data: Optional[Dict[str, Dict[str, Any]]] = None,
        symbol: Optional[str] = None,
        fundamentals_cache=None,
        max_age_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Fill missing/invalid market_cap or avg_volume after combine or on cache read.

        Yahoo often emits market_cap=0 for thin ETFs; FMP stores cap under profile.marketCap.
        FundamentalsCombiner rejects non-positive market_cap, so combined snapshots can omit cap
        even when a provider payload still contains a valid nested value.
        """
        if not combined or not isinstance(combined, dict):
            return combined

        meta = combined.setdefault("_metadata", {})
        sources = meta.setdefault("field_sources", {})

        if not self._is_positive_fundamental_scalar(combined.get("market_cap")):
            cap_val: Optional[float] = None
            cap_src = None
            if provider_data:
                for pname, pdata in provider_data.items():
                    if not isinstance(pdata, dict):
                        continue
                    c = self._market_cap_from_fundamentals_payload(pdata)
                    if c is not None:
                        cap_val, cap_src = c, pname
                        break
            if cap_val is None and symbol and fundamentals_cache is not None:
                for pname in ("fmp", "yahoo", "finnhub", "alpha_vantage", "twelvedata"):
                    meta_file = fundamentals_cache.find_latest_json(
                        symbol, provider=pname, data_type="general", max_age_days=max_age_days
                    )
                    if not meta_file:
                        continue
                    payload = fundamentals_cache.read_json(meta_file.file_path)
                    c = self._market_cap_from_fundamentals_payload(payload or {})
                    if c is not None:
                        cap_val, cap_src = c, f"{pname}_cache"
                        break
            if cap_val is not None:
                combined["market_cap"] = cap_val
                sources["market_cap"] = cap_src or "enrichment"

        if combined.get("avg_volume") is None or not self._is_positive_fundamental_scalar(combined.get("avg_volume")):
            vol_val: Optional[float] = None
            vol_src = None
            if provider_data:
                for pname, pdata in provider_data.items():
                    if not isinstance(pdata, dict):
                        continue
                    v = self._avg_volume_from_fundamentals_payload(pdata)
                    if v is not None:
                        vol_val, vol_src = v, pname
                        break
            if vol_val is None and symbol and fundamentals_cache is not None:
                for pname in ("yahoo", "fmp", "finnhub", "alpha_vantage", "twelvedata"):
                    meta_file = fundamentals_cache.find_latest_json(
                        symbol, provider=pname, data_type="general", max_age_days=max_age_days
                    )
                    if not meta_file:
                        continue
                    payload = fundamentals_cache.read_json(meta_file.file_path)
                    v = self._avg_volume_from_fundamentals_payload(payload or {})
                    if v is not None:
                        vol_val, vol_src = v, f"{pname}_cache"
                        break
            if vol_val is not None:
                combined["avg_volume"] = vol_val
                sources["avg_volume"] = vol_src or "enrichment"

        return combined

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
            fundamentals_cache.write_json(symbol, "combined", combined_data, timestamp)
            removed_combined = fundamentals_cache.cleanup_stale_data(symbol, "combined", timestamp)
            if removed_combined:
                _logger.debug(
                    "Cleaned up %d stale combined files for %s",
                    len(removed_combined),
                    symbol,
                )
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


# Convenience functions for easy access
_provider_selector_cache = None

def get_provider_selector(config_path: Optional[str] = None, cache_dir: Optional[str] = None) -> ProviderSelector:
    """
    Get a cached ProviderSelector instance.

    Args:
        config_path: Path to provider configuration file
        cache_dir: Cache directory path

    Returns:
        ProviderSelector instance
    """
    global _provider_selector_cache
    if _provider_selector_cache is None:
        _provider_selector_cache = ProviderSelector(config_path=config_path, cache_dir=cache_dir)
    return _provider_selector_cache


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
