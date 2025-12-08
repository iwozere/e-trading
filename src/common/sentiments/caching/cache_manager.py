"""
Multi-tier cache manager with automatic fallback.

Provides unified caching interface with automatic fallback from Redis to
in-memory caching when Redis is unavailable. Includes cache warming,
intelligent key strategies, and comprehensive monitoring.
"""

import hashlib
import asyncio
from typing import Any, Optional, Dict, List, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import os

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from .memory_cache import MemoryCache
from .redis_cache import RedisCache
from .cache_metrics import get_cache_metrics

_logger = setup_logger(__name__)


@dataclass
class CacheConfig:
    """Configuration for cache manager."""
    # Memory cache settings
    memory_max_size: int = 1000
    memory_default_ttl: int = 3600  # 1 hour

    # Redis settings
    redis_enabled: bool = True
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_max_connections: int = 10
    redis_key_prefix: str = "sentiment:"
    redis_default_ttl: int = 7200  # 2 hours

    # Cache warming settings
    warming_enabled: bool = True
    warming_batch_size: int = 50
    warming_interval_seconds: int = 1800  # 30 minutes

    # Performance settings
    fallback_enabled: bool = True
    metrics_enabled: bool = True
    cleanup_interval_seconds: int = 300  # 5 minutes

    @classmethod
    def from_env(cls) -> 'CacheConfig':
        """Create configuration from environment variables."""
        return cls(
            memory_max_size=int(os.getenv("CACHE_MEMORY_MAX_SIZE", "1000")),
            memory_default_ttl=int(os.getenv("CACHE_MEMORY_TTL", "3600")),
            redis_enabled=os.getenv("CACHE_REDIS_ENABLED", "true").lower() == "true",
            redis_host=os.getenv("CACHE_REDIS_HOST", "localhost"),
            redis_port=int(os.getenv("CACHE_REDIS_PORT", "6379")),
            redis_db=int(os.getenv("CACHE_REDIS_DB", "0")),
            redis_password=os.getenv("CACHE_REDIS_PASSWORD"),
            redis_max_connections=int(os.getenv("CACHE_REDIS_MAX_CONN", "10")),
            redis_key_prefix=os.getenv("CACHE_REDIS_PREFIX", "sentiment:"),
            redis_default_ttl=int(os.getenv("CACHE_REDIS_TTL", "7200")),
            warming_enabled=os.getenv("CACHE_WARMING_ENABLED", "true").lower() == "true",
            warming_batch_size=int(os.getenv("CACHE_WARMING_BATCH_SIZE", "50")),
            warming_interval_seconds=int(os.getenv("CACHE_WARMING_INTERVAL", "1800")),
            fallback_enabled=os.getenv("CACHE_FALLBACK_ENABLED", "true").lower() == "true",
            metrics_enabled=os.getenv("CACHE_METRICS_ENABLED", "true").lower() == "true",
            cleanup_interval_seconds=int(os.getenv("CACHE_CLEANUP_INTERVAL", "300"))
        )


class CacheKeyStrategy:
    """Strategies for generating cache keys for different data types."""

    @staticmethod
    def sentiment_summary_key(ticker: str, since_ts: int, adapter: str) -> str:
        """Generate key for sentiment summary data."""
        return f"summary:{adapter}:{ticker}:{since_ts}"

    @staticmethod
    def sentiment_messages_key(ticker: str, since_ts: int, adapter: str, limit: int) -> str:
        """Generate key for sentiment messages data."""
        return f"messages:{adapter}:{ticker}:{since_ts}:{limit}"

    @staticmethod
    def hf_predictions_key(texts: List[str]) -> str:
        """Generate key for HuggingFace predictions based on text hash."""
        # Create hash of all texts to use as key
        text_content = "|".join(sorted(texts))
        text_hash = hashlib.md5(text_content.encode('utf-8')).hexdigest()
        return f"hf_predictions:{text_hash}"

    @staticmethod
    def aggregated_sentiment_key(ticker: str, lookback_hours: int, config_hash: str) -> str:
        """Generate key for aggregated sentiment results."""
        return f"aggregated:{ticker}:{lookback_hours}:{config_hash}"

    @staticmethod
    def config_hash(config: Dict[str, Any]) -> str:
        """Generate hash for configuration to use in cache keys."""
        # Create deterministic hash of relevant config parameters
        relevant_keys = ['providers', 'weights', 'heuristic', 'min_mentions_for_hf']
        config_str = str(sorted((k, v) for k, v in config.items() if k in relevant_keys))
        return hashlib.md5(config_str.encode('utf-8')).hexdigest()[:8]


class CacheManager:
    """
    Multi-tier cache manager with automatic fallback and warming.

    Provides unified caching interface that automatically falls back from
    Redis to in-memory caching when Redis is unavailable. Includes intelligent
    cache warming, key strategies, and comprehensive monitoring.
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize cache manager.

        Args:
            config: Cache configuration (uses defaults if None)
        """
        self.config = config or CacheConfig.from_env()
        self._metrics = get_cache_metrics() if self.config.metrics_enabled else None

        # Initialize memory cache (always available)
        self._memory_cache = MemoryCache(
            max_size=self.config.memory_max_size,
            default_ttl_seconds=self.config.memory_default_ttl
        )

        # Initialize Redis cache (optional)
        self._redis_cache: Optional[RedisCache] = None
        if self.config.redis_enabled:
            try:
                self._redis_cache = RedisCache(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    db=self.config.redis_db,
                    password=self.config.redis_password,
                    max_connections=self.config.redis_max_connections,
                    key_prefix=self.config.redis_key_prefix
                )
                if self._redis_cache.is_available():
                    _logger.info("Redis cache initialized and available")
                else:
                    _logger.warning("Redis cache initialized but not available")
            except Exception as e:
                _logger.warning("Failed to initialize Redis cache: %s", e)
                self._redis_cache = None

        # Cache warming state
        self._warming_enabled = self.config.warming_enabled
        self._warming_data: Dict[str, Any] = {}
        self._last_warming_time: Optional[datetime] = None
        self._last_cleanup_time: Optional[datetime] = None

    def get(self, key: str, use_redis: bool = True) -> Optional[Any]:
        """
        Get value from cache with automatic tier fallback.

        Args:
            key: Cache key
            use_redis: Whether to try Redis first (default: True)

        Returns:
            Cached value or None if not found
        """
        # Try Redis first if enabled and available
        if use_redis and self._redis_cache and self._redis_cache.is_available():
            try:
                value = self._redis_cache.get(key)
                if value is not None:
                    # Also cache in memory for faster subsequent access
                    self._memory_cache.set(key, value, self.config.memory_default_ttl)
                    return value
            except Exception as e:
                _logger.debug("Redis get failed, falling back to memory: %s", e)

        # Fallback to memory cache
        return self._memory_cache.get(key)

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None,
            use_redis: bool = True) -> bool:
        """
        Set value in cache across all available tiers.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: TTL override (uses defaults if None)
            use_redis: Whether to use Redis tier (default: True)

        Returns:
            True if successfully cached in at least one tier
        """
        success = False

        # Set in memory cache (always try)
        memory_ttl = ttl_seconds or self.config.memory_default_ttl
        if self._memory_cache.set(key, value, memory_ttl):
            success = True

        # Set in Redis cache if available
        if use_redis and self._redis_cache and self._redis_cache.is_available():
            try:
                redis_ttl = ttl_seconds or self.config.redis_default_ttl
                if self._redis_cache.set(key, value, redis_ttl):
                    success = True
            except Exception as e:
                _logger.debug("Redis set failed: %s", e)

        return success

    def delete(self, key: str) -> bool:
        """
        Delete key from all cache tiers.

        Args:
            key: Cache key to delete

        Returns:
            True if deleted from at least one tier
        """
        success = False

        # Delete from memory cache
        if self._memory_cache.delete(key):
            success = True

        # Delete from Redis cache
        if self._redis_cache and self._redis_cache.is_available():
            try:
                if self._redis_cache.delete(key):
                    success = True
            except Exception as e:
                _logger.debug("Redis delete failed: %s", e)

        return success

    def exists(self, key: str) -> bool:
        """
        Check if key exists in any cache tier.

        Args:
            key: Cache key to check

        Returns:
            True if key exists in any tier
        """
        # Check Redis first (more authoritative)
        if self._redis_cache and self._redis_cache.is_available():
            try:
                if self._redis_cache.exists(key):
                    return True
            except Exception as e:
                _logger.debug("Redis exists check failed: %s", e)

        # Check memory cache
        return self._memory_cache.exists(key)

    def get_or_set(self, key: str, factory: Callable[[], Any],
                   ttl_seconds: Optional[int] = None) -> Any:
        """
        Get value from cache or set it using factory function.

        Args:
            key: Cache key
            factory: Function to generate value if not cached
            ttl_seconds: TTL for new values

        Returns:
            Cached or newly generated value
        """
        # Try to get existing value
        value = self.get(key)
        if value is not None:
            return value

        # Generate new value
        try:
            value = factory()
            if value is not None:
                self.set(key, value, ttl_seconds)
            return value
        except Exception as e:
            _logger.warning("Cache factory function failed for key %s: %s", key, e)
            return None

    async def get_or_set_async(self, key: str, factory: Callable[[], Any],
                              ttl_seconds: Optional[int] = None) -> Any:
        """
        Async version of get_or_set.

        Args:
            key: Cache key
            factory: Async function to generate value if not cached
            ttl_seconds: TTL for new values

        Returns:
            Cached or newly generated value
        """
        # Try to get existing value
        value = self.get(key)
        if value is not None:
            return value

        # Generate new value
        try:
            if asyncio.iscoroutinefunction(factory):
                value = await factory()
            else:
                value = factory()

            if value is not None:
                self.set(key, value, ttl_seconds)
            return value
        except Exception as e:
            _logger.warning("Async cache factory function failed for key %s: %s", key, e)
            return None

    def warm_cache(self, warming_data: Dict[str, Any]) -> None:
        """
        Warm cache with frequently accessed data.

        Args:
            warming_data: Dictionary of key-value pairs to pre-cache
        """
        if not self._warming_enabled:
            return

        _logger.debug("Warming cache with %d entries", len(warming_data))

        for key, value in warming_data.items():
            try:
                self.set(key, value, self.config.redis_default_ttl)
            except Exception as e:
                _logger.debug("Failed to warm cache key %s: %s", key, e)

        self._last_warming_time = datetime.now(timezone.utc)

    def should_warm_cache(self) -> bool:
        """Check if cache warming should be performed."""
        if not self._warming_enabled:
            return False

        if self._last_warming_time is None:
            return True

        return (datetime.now(timezone.utc) - self._last_warming_time >
                timedelta(seconds=self.config.warming_interval_seconds))

    def cleanup_expired(self) -> Dict[str, int]:
        """
        Clean up expired entries from all cache tiers.

        Returns:
            Dictionary with cleanup counts per tier
        """
        results = {}

        # Cleanup memory cache
        try:
            memory_cleaned = self._memory_cache.cleanup_expired()
            results['memory'] = memory_cleaned
        except Exception as e:
            _logger.debug("Memory cache cleanup failed: %s", e)
            results['memory'] = 0

        # Redis cleanup is handled automatically by Redis TTL
        results['redis'] = 0

        self._last_cleanup_time = datetime.now(timezone.utc)
        return results

    def should_cleanup(self) -> bool:
        """Check if cleanup should be performed."""
        if self._last_cleanup_time is None:
            return True

        return (datetime.now(timezone.utc) - self._last_cleanup_time >
                timedelta(seconds=self.config.cleanup_interval_seconds))

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {
            'config': {
                'redis_enabled': self.config.redis_enabled,
                'redis_available': self._redis_cache.is_available() if self._redis_cache else False,
                'warming_enabled': self._warming_enabled,
                'fallback_enabled': self.config.fallback_enabled
            },
            'memory': self._memory_cache.get_stats()
        }

        # Add Redis stats if available
        if self._redis_cache and self._redis_cache.is_available():
            try:
                stats['redis'] = self._redis_cache.get_info()
            except Exception as e:
                _logger.debug("Failed to get Redis stats: %s", e)
                stats['redis'] = {'error': str(e)}

        # Add metrics if enabled
        if self._metrics:
            stats['metrics'] = self._metrics.get_summary()

        return stats

    def clear_all(self) -> None:
        """Clear all cache tiers."""
        _logger.info("Clearing all cache tiers")

        # Clear memory cache
        self._memory_cache.clear()

        # Clear Redis cache
        if self._redis_cache and self._redis_cache.is_available():
            try:
                self._redis_cache.clear_prefix()
            except Exception as e:
                _logger.debug("Failed to clear Redis cache: %s", e)

    def close(self) -> None:
        """Close all cache connections."""
        if self._redis_cache:
            self._redis_cache.close()


# Global cache manager instance
_global_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager()
    return _global_cache_manager