"""
Data module for the e-trading system.

This module provides comprehensive data management capabilities including:
- Data sources and providers
- Data validation and quality assessment
- Caching and persistence
- Rate limiting and retry mechanisms
- Data aggregation and synchronization
- Advanced caching with Redis support
- Real-time data streaming
- Performance optimization
"""

# New unified DataManager (main entry point)
from .data_manager import DataManager, get_data_manager, ProviderSelector

# Base classes
from .sources import BaseDataSource
from .downloader import BaseDataDownloader
from .feed import BaseLiveDataFeed

# Legacy components (to be deprecated)
from .sources.data_source_factory import (
    DataSourceFactory,
    get_data_source_factory,
    register_data_source,
    create_data_source,
    get_data_source
)
from .sources.data_aggregator import DataAggregator
from .feed.binance_data_feed import BinanceEnhancedFeed

# Specific implementations (now organized in submodules)
from .downloader import (
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

from .feed import (
    BinanceLiveDataFeed,
    YahooLiveDataFeed,
    IBKRLiveDataFeed,
    CoinGeckoLiveDataFeed,
)
from .utils.advanced_caching import get_advanced_cache, configure_advanced_cache

# Utilities
from .utils import (
    get_data_handler,
    validate_ohlcv_data,
    get_data_quality_score,
    RateLimiter,
    DataCache,
                    # Phase 3: Advanced Features
                FileBasedCache,
                get_file_cache,
                configure_file_cache,
                TimeBasedInvalidation,
                VersionBasedInvalidation,
                FileCacheInvalidationStrategy,
                FileCacheCompressor,
    # StreamMultiplexer,
    # get_stream_multiplexer,
    # create_stream_config,
    LazyDataLoader,
    ParallelProcessor,
    MemoryOptimizer,
    PerformanceMonitor,
    get_performance_monitor,
    get_memory_optimizer,
    get_data_compressor,
    optimize_dataframe_performance,
    compress_dataframe_efficiently,
)

__all__ = [
    # New unified DataManager (main entry point)
    'DataManager',
    'get_data_manager',
    'ProviderSelector',

    # Base classes
    'BaseDataDownloader',
    'BaseLiveDataFeed',
    'BaseDataSource',

    # Data downloaders
    'BinanceDataDownloader',
    'YahooDataDownloader',
    'AlphaVantageDataDownloader',
    'FMPDataDownloader',
    'TiingoDataDownloader',
    'PolygonDataDownloader',
    'TwelveDataDataDownloader',
    'FinnhubDataDownloader',
    'CoinGeckoDataDownloader',

    # Live feeds
    'BinanceLiveDataFeed',
    'YahooLiveDataFeed',
    'IBKRLiveDataFeed',
    'CoinGeckoLiveDataFeed',

    # Legacy components (to be deprecated)
    'DataSourceFactory',
    'get_data_source_factory',
    'register_data_source',
    'create_data_source',
    'get_data_source',
    'DataAggregator',
    'BinanceEnhancedFeed',
    'get_advanced_cache',
    'configure_advanced_cache',

    # Core utilities
    'get_data_handler',
    'validate_ohlcv_data',
    'get_data_quality_score',
    'RateLimiter',
    'DataCache',

                    # Phase 3: Advanced Caching
                'FileBasedCache',
                'get_file_cache',
                'configure_file_cache',
                'TimeBasedInvalidation',
                'VersionBasedInvalidation',
                'FileCacheInvalidationStrategy',
                'FileCacheCompressor',

    # Phase 3: Data Streaming (commented out - module not available)
    # 'StreamMultiplexer',
    # 'get_stream_multiplexer',
    # 'create_stream_config',

    # Phase 3: Performance Optimization
    'LazyDataLoader',
    'ParallelProcessor',
    'MemoryOptimizer',
    'PerformanceMonitor',
    'get_performance_monitor',
    'get_memory_optimizer',
    'get_data_compressor',
    'optimize_dataframe_performance',
    'compress_dataframe_efficiently',
]
