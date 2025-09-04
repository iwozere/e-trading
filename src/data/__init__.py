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

from .base_data_downloader import BaseDataDownloader
from .base_data_source import BaseDataSource
from .data_source_factory import (
    DataSourceFactory,
    get_data_source_factory,
    register_data_source,
    create_data_source,
    get_data_source
)
from .data_aggregator import DataAggregator
from .binance_live_feed import BinanceLiveDataFeed
from .binance_data_feed import BinanceEnhancedFeed
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
    StreamMultiplexer,
    get_stream_multiplexer,
    create_stream_config,
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
    # Base classes
    'BaseDataDownloader',
    'BaseDataSource',

    # Factory and management
    'DataSourceFactory',
    'get_data_source_factory',
    'register_data_source',
    'create_data_source',
    'get_data_source',

    # Data aggregation
    'DataAggregator',

    # Specific implementations
    'BinanceLiveDataFeed',
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

    # Phase 3: Data Streaming
    'StreamMultiplexer',
    'get_stream_multiplexer',
    'create_stream_config',

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
