from .retry import request_with_backoff, exponential_backoff
from .validation import validate_ohlcv_data, validate_timestamps, get_data_quality_score
from .rate_limiting import RateLimiter, get_provider_limiter, configure_provider_limits, get_all_provider_stats
from .caching import DataCache, get_cache, configure_cache
from .data_handler import DataHandler, get_data_handler

# Phase 3: Advanced Features
from .file_based_cache import (
    FileBasedCache, FileCacheCompressor, FileCacheInvalidationStrategy,
    TimeBasedInvalidation, VersionBasedInvalidation,
    get_file_cache, configure_file_cache
)
from .data_streaming import (
    StreamConfig, StreamMetrics, WebSocketConnection, ConnectionPool,
    DataStreamProcessor, StreamMultiplexer, BackpressureHandler,
    get_stream_multiplexer, create_stream_config
)
from .performance_optimization import (
    PerformanceMetrics, DataCompressor, LazyDataLoader, ParallelProcessor,
    MemoryOptimizer, PerformanceMonitor,
    get_performance_monitor, get_memory_optimizer, get_data_compressor,
    optimize_dataframe_performance, compress_dataframe_efficiently
)

__all__ = [
    # Phase 2: Core Utilities
    'request_with_backoff', 'exponential_backoff',
    'validate_ohlcv_data', 'validate_timestamps', 'get_data_quality_score',
    'RateLimiter', 'get_provider_limiter', 'configure_provider_limits', 'get_all_provider_stats',
    'DataCache', 'get_cache', 'configure_cache',
    'DataHandler', 'get_data_handler',

                    # Phase 3: Advanced Caching
                'FileBasedCache', 'FileCacheCompressor', 'FileCacheInvalidationStrategy',
                'TimeBasedInvalidation', 'VersionBasedInvalidation',
                'get_file_cache', 'configure_file_cache',

    # Phase 3: Data Streaming
    'StreamConfig', 'StreamMetrics', 'WebSocketConnection', 'ConnectionPool',
    'DataStreamProcessor', 'StreamMultiplexer', 'BackpressureHandler',
    'get_stream_multiplexer', 'create_stream_config',

    # Phase 3: Performance Optimization
    'PerformanceMetrics', 'DataCompressor', 'LazyDataLoader', 'ParallelProcessor',
    'MemoryOptimizer', 'PerformanceMonitor',
    'get_performance_monitor', 'get_memory_optimizer', 'get_data_compressor',
    'optimize_dataframe_performance', 'compress_dataframe_efficiently',
]
