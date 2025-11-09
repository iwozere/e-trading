"""
Advanced caching system with Redis support and performance metrics.

This module provides enhanced caching capabilities including:
- Redis support for distributed caching
- Cache invalidation strategies
- Performance metrics and monitoring
- Cache compression and optimization
"""

import time
import hashlib
import pickle
import gzip
from typing import Any, Optional, Dict, List, Union
from datetime import datetime
from pathlib import Path
import logging
import threading
from dataclasses import dataclass, asdict

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False

from src.data.utils.caching import DataCache

# Import cache directory setting
try:
    from config.donotshare.donotshare import DATA_CACHE_DIR
except ImportError:
    DATA_CACHE_DIR = "c:/data-cache"

_logger = logging.getLogger(__name__)


@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0
    total_size_bytes: int = 0
    compression_ratio: float = 1.0
    avg_response_time_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def total_operations(self) -> int:
        """Total cache operations."""
        return self.hits + self.misses + self.sets + self.deletes


class CacheCompressor:
    """Handles data compression for caching."""

    def __init__(self, compression_level: int = 3):
        """
        Initialize compressor.

        Args:
            compression_level: Compression level (1-22 for zstd, 1-9 for gzip)
        """
        self.compression_level = compression_level
        self.use_zstd = ZSTD_AVAILABLE

    def compress(self, data: bytes) -> bytes:
        """Compress data."""
        if self.use_zstd:
            return zstd.compress(data, level=self.compression_level)
        else:
            return gzip.compress(data, compresslevel=self.compression_level)

    def decompress(self, data: bytes) -> bytes:
        """Decompress data."""
        if self.use_zstd:
            return zstd.decompress(data)
        else:
            return gzip.decompress(data)

    def get_compression_ratio(self, original: bytes, compressed: bytes) -> float:
        """Calculate compression ratio."""
        return len(compressed) / len(original) if original else 1.0


class RedisCache:
    """Redis-based distributed cache implementation."""

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        max_connections: int = 10,
        compression_enabled: bool = True,
        default_ttl: int = 3600
    ):
        """
        Initialize Redis cache.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            max_connections: Maximum connection pool size
            compression_enabled: Whether to enable compression
            default_ttl: Default TTL in seconds
        """
        if not REDIS_AVAILABLE:
            raise ImportError("Redis is not available. Install with: pip install redis")

        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.compression_enabled = compression_enabled
        self.default_ttl = default_ttl

        # Initialize Redis connection pool
        self.pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            max_connections=max_connections,
            decode_responses=False  # Keep as bytes for compression
        )

        self.compressor = CacheCompressor() if compression_enabled else None
        self.metrics = CacheMetrics()
        self._lock = threading.Lock()

        _logger.info("Redis cache initialized: %s:%s/db%s", host, port, db)

    def _get_redis(self):
        """Get Redis client from pool."""
        if not REDIS_AVAILABLE:
            raise ImportError("Redis package not available")
        return redis.Redis(connection_pool=self.pool)

    def _make_key(self, key: str) -> str:
        """Create a standardized cache key."""
        return f"data_cache:{key}"

    def _serialize(self, data: Any) -> bytes:
        """Serialize data for storage."""
        start_time = time.time()

        # Serialize to pickle
        serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

        # Compress if enabled
        if self.compression_enabled and self.compressor:
            compressed = self.compressor.compress(serialized)
            compression_ratio = self.compressor.get_compression_ratio(serialized, compressed)

            with self._lock:
                self.metrics.compression_ratio = (
                    (self.metrics.compression_ratio + compression_ratio) / 2
                )

            serialized = compressed

        # Update metrics
        with self._lock:
            self.metrics.total_size_bytes += len(serialized)
            self.metrics.avg_response_time_ms = (
                (self.metrics.avg_response_time_ms + (time.time() - start_time) * 1000) / 2
            )

        return serialized

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize data from storage."""
        start_time = time.time()

        # Decompress if needed
        if self.compression_enabled and self.compressor:
            try:
                data = self.compressor.decompress(data)
            except Exception as e:
                _logger.warning("Failed to decompress data: %s", e)
                return None

        # Deserialize from pickle
        try:
            result = pickle.loads(data)
        except Exception:
            _logger.exception("Failed to deserialize data:")
            return None

        # Update metrics
        with self._lock:
            self.metrics.avg_response_time_ms = (
                (self.metrics.avg_response_time_ms + (time.time() - start_time) * 1000) / 2
            )

        return result

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        start_time = time.time()

        try:
            redis_client = self._get_redis()
            cache_key = self._make_key(key)
            data = redis_client.get(cache_key)

            if data is None:
                with self._lock:
                    self.metrics.misses += 1
                return None

            result = self._deserialize(data)

            with self._lock:
                self.metrics.hits += 1

            return result

        except Exception:
            _logger.exception("Redis get error:")
            with self._lock:
                self.metrics.errors += 1
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        try:
            redis_client = self._get_redis()
            cache_key = self._make_key(key)
            serialized = self._serialize(value)

            ttl = ttl or self.default_ttl
            success = redis_client.setex(cache_key, ttl, serialized)

            with self._lock:
                self.metrics.sets += 1

            return success

        except Exception:
            _logger.exception("Redis set error:")
            with self._lock:
                self.metrics.errors += 1
            return False

    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            redis_client = self._get_redis()
            cache_key = self._make_key(key)
            deleted = redis_client.delete(cache_key)

            with self._lock:
                self.metrics.deletes += 1

            return deleted > 0

        except Exception:
            _logger.exception("Redis delete error:")
            with self._lock:
                self.metrics.errors += 1
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            redis_client = self._get_redis()
            cache_key = self._make_key(key)
            return redis_client.exists(cache_key) > 0
        except Exception:
            _logger.exception("Redis exists error:")
            return False

    def get_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics."""
        with self._lock:
            return asdict(self.metrics)

    def clear_metrics(self) -> None:
        """Clear performance metrics."""
        with self._lock:
            self.metrics = CacheMetrics()

    def get_info(self) -> Dict[str, Any]:
        """Get Redis server information."""
        try:
            redis_client = self._get_redis()
            info = redis_client.info()
            return {
                'redis_version': info.get('redis_version'),
                'connected_clients': info.get('connected_clients'),
                'used_memory_human': info.get('used_memory_human'),
                'total_commands_processed': info.get('total_commands_processed'),
                'keyspace_hits': info.get('keyspace_hits'),
                'keyspace_misses': info.get('keyspace_misses'),
            }
        except Exception:
            _logger.exception("Failed to get Redis info:")
            return {}


class CacheInvalidationStrategy:
    """Base class for cache invalidation strategies."""

    def should_invalidate(self, key: str, metadata: Dict[str, Any]) -> bool:
        """Determine if cache entry should be invalidated."""
        raise NotImplementedError


class TimeBasedInvalidation(CacheInvalidationStrategy):
    """Time-based cache invalidation."""

    def __init__(self, max_age_hours: int = 24):
        """
        Initialize time-based invalidation.

        Args:
            max_age_hours: Maximum age in hours before invalidation
        """
        self.max_age_hours = max_age_hours

    def should_invalidate(self, key: str, metadata: Dict[str, Any]) -> bool:
        """Check if entry is too old."""
        created_at = metadata.get('created_at')
        if not created_at:
            return True

        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        age = datetime.now() - created_at
        return age.total_seconds() > (self.max_age_hours * 3600)


class VersionBasedInvalidation(CacheInvalidationStrategy):
    """Version-based cache invalidation."""

    def __init__(self, current_version: str = "1.0.0"):
        """
        Initialize version-based invalidation.

        Args:
            current_version: Current data version
        """
        self.current_version = current_version

    def should_invalidate(self, key: str, metadata: Dict[str, Any]) -> bool:
        """Check if entry version is outdated."""
        cached_version = metadata.get('version')
        return cached_version != self.current_version


class AdvancedDataCache(DataCache):
    """Advanced data cache with Redis support and invalidation strategies."""

    def __init__(
        self,
        cache_dir: Union[str, Path] = DATA_CACHE_DIR,
        max_size_gb: float = 10.0,
        retention_days: int = 30,
        redis_config: Optional[Dict[str, Any]] = None,
        invalidation_strategies: Optional[List[CacheInvalidationStrategy]] = None,
        compression_enabled: bool = True
    ):
        """
        Initialize advanced data cache.

        Args:
            cache_dir: Directory for file-based caching
            max_size_gb: Maximum cache size in GB
            retention_days: Data retention period in days
            redis_config: Redis configuration dictionary
            invalidation_strategies: List of invalidation strategies
            compression_enabled: Whether to enable compression
        """
        super().__init__(cache_dir, max_size_gb, retention_days)

        # Initialize Redis cache if configured
        self.redis_cache = None
        if redis_config and REDIS_AVAILABLE:
            try:
                self.redis_cache = RedisCache(
                    host=redis_config.get('host', 'localhost'),
                    port=redis_config.get('port', 6379),
                    db=redis_config.get('db', 0),
                    password=redis_config.get('password'),
                    max_connections=redis_config.get('max_connections', 10),
                    compression_enabled=compression_enabled,
                    default_ttl=redis_config.get('default_ttl', 3600)
                )
                _logger.info("Redis cache enabled")
            except Exception as e:
                _logger.warning("Failed to initialize Redis cache: %s", e)

        # Initialize invalidation strategies
        self.invalidation_strategies = invalidation_strategies or [
            TimeBasedInvalidation(max_age_hours=24)
        ]

        # Performance tracking
        self.metrics = CacheMetrics()
        self._lock = threading.Lock()

    def _make_cache_key(
        self,
        provider: str,
        symbol: str,
        interval: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> str:
        """Create a standardized cache key."""
        key_parts = [provider, symbol, interval]

        if start_date:
            key_parts.append(start_date.strftime("%Y%m%d"))
        if end_date:
            key_parts.append(end_date.strftime("%Y%m%d"))

        return hashlib.md5(":".join(key_parts).encode()).hexdigest()

    def _get_metadata(self, df: Any) -> Dict[str, Any]:
        """Generate metadata for cache entry."""
        return {
            'created_at': datetime.now().isoformat(),
            'version': '1.0.0',
            'rows': len(df) if hasattr(df, '__len__') else 0,
            'columns': list(df.columns) if hasattr(df, 'columns') else [],
            'size_bytes': len(pickle.dumps(df)) if hasattr(df, '__len__') else 0
        }

    def _should_invalidate(self, metadata: Dict[str, Any]) -> bool:
        """Check if cache entry should be invalidated."""
        for strategy in self.invalidation_strategies:
            if strategy.should_invalidate("", metadata):
                return True
        return False

    def get(
        self,
        provider: str,
        symbol: str,
        interval: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        file_format: str = "parquet"
    ) -> Optional[Any]:
        """Get data from cache (Redis first, then file)."""
        start_time = time.time()
        cache_key = self._make_cache_key(provider, symbol, interval, start_date, end_date)

        # Try Redis first
        if self.redis_cache:
            try:
                result = self.redis_cache.get(cache_key)
                if result is not None:
                    # Check if data should be invalidated
                    metadata = result.get('metadata', {})
                    if not self._should_invalidate(metadata):
                        with self._lock:
                            self.metrics.hits += 1
                        return result['data']
                    else:
                        # Invalidate outdated data
                        self.redis_cache.delete(cache_key)
                        with self._lock:
                            self.metrics.deletes += 1
            except Exception as e:
                _logger.warning("Redis get failed: %s", e)

        # Fall back to file cache
        try:
            result = super().get(provider, symbol, interval, start_date, end_date, file_format)
            if result is not None:
                with self._lock:
                    self.metrics.hits += 1
            else:
                with self._lock:
                    self.metrics.misses += 1
            return result
        except Exception:
            _logger.exception("File cache get failed:")
            with self._lock:
                self.metrics.errors += 1
            return None

    def put(
        self,
        df: Any,
        provider: str,
        symbol: str,
        interval: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        file_format: str = "parquet"
    ) -> bool:
        """Put data in cache (both Redis and file)."""
        cache_key = self._make_cache_key(provider, symbol, interval, start_date, end_date)
        metadata = self._get_metadata(df)

        success = True

        # Store in Redis
        if self.redis_cache:
            try:
                cache_data = {
                    'data': df,
                    'metadata': metadata
                }
                success &= self.redis_cache.set(cache_key, cache_data)
            except Exception as e:
                _logger.warning("Redis put failed: %s", e)
                success = False

        # Store in file cache
        try:
            success &= super().put(df, provider, symbol, interval, start_date, end_date, file_format)
        except Exception:
            _logger.exception("File cache put failed:")
            success = False

        if success:
            with self._lock:
                self.metrics.sets += 1

        return success

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive cache metrics."""
        metrics = {
            'file_cache': super().get_stats(),
            'advanced_metrics': asdict(self.metrics)
        }

        if self.redis_cache:
            metrics['redis_cache'] = self.redis_cache.get_metrics()
            metrics['redis_info'] = self.redis_cache.get_info()

        return metrics

    def clear_metrics(self) -> None:
        """Clear all cache metrics."""
        with self._lock:
            self.metrics = CacheMetrics()

        if self.redis_cache:
            self.redis_cache.clear_metrics()

    def cleanup(self) -> None:
        """Clean up cache resources."""
        super().cleanup()

        if self.redis_cache:
            try:
                self.redis_cache.pool.disconnect()
            except Exception as e:
                _logger.warning("Failed to disconnect Redis pool: %s", e)


# Global advanced cache instance
_advanced_cache_instance: Optional[AdvancedDataCache] = None


def get_advanced_cache(
    cache_dir: Union[str, Path] = DATA_CACHE_DIR,
    redis_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> AdvancedDataCache:
    """
    Get or create global advanced cache instance.

    Args:
        cache_dir: Cache directory
        redis_config: Redis configuration
        **kwargs: Additional cache configuration

    Returns:
        AdvancedDataCache instance
    """
    global _advanced_cache_instance

    if _advanced_cache_instance is None:
        _advanced_cache_instance = AdvancedDataCache(
            cache_dir=cache_dir,
            redis_config=redis_config,
            **kwargs
        )

    return _advanced_cache_instance


def configure_advanced_cache(
    cache_dir: Union[str, Path] = DATA_CACHE_DIR,
    redis_config: Optional[Dict[str, Any]] = None,
    invalidation_strategies: Optional[List[CacheInvalidationStrategy]] = None,
    **kwargs
) -> AdvancedDataCache:
    """
    Configure and return advanced cache instance.

    Args:
        cache_dir: Cache directory
        redis_config: Redis configuration
        invalidation_strategies: Cache invalidation strategies
        **kwargs: Additional configuration

    Returns:
        Configured AdvancedDataCache instance
    """
    global _advanced_cache_instance

    _advanced_cache_instance = AdvancedDataCache(
        cache_dir=cache_dir,
        redis_config=redis_config,
        invalidation_strategies=invalidation_strategies,
        **kwargs
    )

    return _advanced_cache_instance
