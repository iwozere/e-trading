"""
Redis cache implementation with connection pooling and failover.

Provides Redis-based caching with automatic connection management,
failover to memory cache, and comprehensive error handling.
"""

import json
import pickle
import time
from typing import Any, Optional, Dict
from pathlib import Path
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from .cache_metrics import get_cache_metrics

_logger = setup_logger(__name__)

# Optional Redis dependency
try:
    import redis
    from redis.connection import ConnectionPool
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    ConnectionPool = None
    REDIS_AVAILABLE = False
    _logger.debug("Redis not available, will use memory-only caching")


class RedisCache:
    """
    Redis-based cache with connection pooling and automatic failover.

    Features:
    - Connection pooling for better performance
    - Automatic reconnection on connection failures
    - JSON and pickle serialization support
    - Comprehensive error handling with fallback
    - TTL support with Redis expiration
    """

    def __init__(self,
                 host: str = "localhost",
                 port: int = 6379,
                 db: int = 0,
                 password: Optional[str] = None,
                 max_connections: int = 10,
                 socket_timeout: float = 5.0,
                 socket_connect_timeout: float = 5.0,
                 key_prefix: str = "sentiment:",
                 serialization: str = "json"):
        """
        Initialize Redis cache.

        Args:
            host: Redis server host
            port: Redis server port
            db: Redis database number
            password: Redis password (optional)
            max_connections: Maximum connections in pool
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Connection timeout in seconds
            key_prefix: Prefix for all cache keys
            serialization: Serialization method ('json' or 'pickle')
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.key_prefix = key_prefix
        self.serialization = serialization
        self._metrics = get_cache_metrics()
        self._pool: Optional[ConnectionPool] = None
        self._client: Optional[redis.Redis] = None
        self._available = False
        self._last_connection_attempt = 0
        self._connection_retry_interval = 30  # seconds

        if not REDIS_AVAILABLE:
            _logger.warning("Redis not available, RedisCache will always fail")
            return

        # Initialize connection pool
        try:
            self._pool = ConnectionPool(
                host=host,
                port=port,
                db=db,
                password=password,
                max_connections=max_connections,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
                retry_on_timeout=True,
                health_check_interval=30
            )
            self._client = redis.Redis(connection_pool=self._pool)
            self._test_connection()
            self._available = True
            _logger.info("Redis cache initialized: %s:%d/%d", host, port, db)

        except Exception as e:
            _logger.warning("Failed to initialize Redis cache: %s", e)
            self._available = False

    def _test_connection(self) -> bool:
        """Test Redis connection."""
        if not self._client:
            return False

        try:
            self._client.ping()
            return True
        except Exception as e:
            _logger.debug("Redis connection test failed: %s", e)
            return False

    def _ensure_connection(self) -> bool:
        """Ensure Redis connection is available."""
        if not REDIS_AVAILABLE or not self._client:
            return False

        current_time = time.time()

        # Don't retry too frequently
        if (not self._available and
            current_time - self._last_connection_attempt < self._connection_retry_interval):
            return False

        self._last_connection_attempt = current_time

        try:
            if self._test_connection():
                if not self._available:
                    _logger.info("Redis connection restored")
                self._available = True
                return True
            else:
                self._available = False
                return False

        except Exception as e:
            _logger.debug("Redis connection check failed: %s", e)
            self._available = False
            return False

    def _make_key(self, key: str) -> str:
        """Create full Redis key with prefix."""
        return f"{self.key_prefix}{key}"

    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage."""
        if self.serialization == "json":
            return json.dumps(value, default=str).encode('utf-8')
        elif self.serialization == "pickle":
            return pickle.dumps(value)
        else:
            raise ValueError(f"Unknown serialization method: {self.serialization}")

    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        if self.serialization == "json":
            return json.loads(data.decode('utf-8'))
        elif self.serialization == "pickle":
            return pickle.loads(data)
        else:
            raise ValueError(f"Unknown serialization method: {self.serialization}")

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from Redis cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/unavailable
        """
        if not self._ensure_connection():
            self._metrics.record_miss("redis")
            return None

        operation_id = f"redis_get_{id(self)}_{time.time()}"
        self._metrics.start_operation(operation_id)

        try:
            redis_key = self._make_key(key)
            data = self._client.get(redis_key)

            if data is None:
                self._metrics.record_miss("redis", operation_id)
                return None

            value = self._deserialize_value(data)
            self._metrics.record_hit("redis", operation_id)
            return value

        except Exception as e:
            _logger.debug("Redis get failed for key %s: %s", key, e)
            self._available = False
            self._metrics.record_miss("redis", operation_id)
            return None

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """
        Set value in Redis cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: TTL in seconds (optional)

        Returns:
            True if successfully cached
        """
        if not self._ensure_connection():
            return False

        operation_id = f"redis_set_{id(self)}_{time.time()}"
        self._metrics.start_operation(operation_id)

        try:
            redis_key = self._make_key(key)
            serialized_value = self._serialize_value(value)

            if ttl_seconds:
                result = self._client.setex(redis_key, ttl_seconds, serialized_value)
            else:
                result = self._client.set(redis_key, serialized_value)

            success = bool(result)
            if success:
                self._metrics.record_set("redis", operation_id)
            return success

        except Exception as e:
            _logger.debug("Redis set failed for key %s: %s", key, e)
            self._available = False
            return False

    def delete(self, key: str) -> bool:
        """
        Delete entry from Redis cache.

        Args:
            key: Cache key to delete

        Returns:
            True if entry was deleted
        """
        if not self._ensure_connection():
            return False

        operation_id = f"redis_del_{id(self)}_{time.time()}"
        self._metrics.start_operation(operation_id)

        try:
            redis_key = self._make_key(key)
            result = self._client.delete(redis_key)

            success = result > 0
            if success:
                self._metrics.record_delete("redis", operation_id)
            return success

        except Exception as e:
            _logger.debug("Redis delete failed for key %s: %s", key, e)
            self._available = False
            return False

    def exists(self, key: str) -> bool:
        """
        Check if key exists in Redis.

        Args:
            key: Cache key to check

        Returns:
            True if key exists
        """
        if not self._ensure_connection():
            return False

        try:
            redis_key = self._make_key(key)
            return bool(self._client.exists(redis_key))

        except Exception as e:
            _logger.debug("Redis exists check failed for key %s: %s", key, e)
            self._available = False
            return False

    def get_ttl(self, key: str) -> Optional[int]:
        """
        Get remaining TTL for a key.

        Args:
            key: Cache key

        Returns:
            Remaining TTL in seconds, or None if key doesn't exist
        """
        if not self._ensure_connection():
            return None

        try:
            redis_key = self._make_key(key)
            ttl = self._client.ttl(redis_key)

            if ttl == -2:  # Key doesn't exist
                return None
            elif ttl == -1:  # Key exists but no TTL
                return None
            else:
                return ttl

        except Exception as e:
            _logger.debug("Redis TTL check failed for key %s: %s", key, e)
            self._available = False
            return None

    def clear_prefix(self, pattern: Optional[str] = None) -> int:
        """
        Clear keys matching a pattern.

        Args:
            pattern: Pattern to match (default: all keys with prefix)

        Returns:
            Number of keys deleted
        """
        if not self._ensure_connection():
            return 0

        try:
            if pattern:
                search_pattern = f"{self.key_prefix}{pattern}"
            else:
                search_pattern = f"{self.key_prefix}*"

            keys = self._client.keys(search_pattern)
            if keys:
                return self._client.delete(*keys)
            return 0

        except Exception as e:
            _logger.debug("Redis clear failed for pattern %s: %s", pattern, e)
            self._available = False
            return 0

    def get_info(self) -> Dict[str, Any]:
        """
        Get Redis server information.

        Returns:
            Redis info dictionary or empty dict if unavailable
        """
        if not self._ensure_connection():
            return {}

        try:
            info = self._client.info()
            return {
                'redis_version': info.get('redis_version'),
                'used_memory': info.get('used_memory'),
                'used_memory_human': info.get('used_memory_human'),
                'connected_clients': info.get('connected_clients'),
                'total_commands_processed': info.get('total_commands_processed'),
                'keyspace_hits': info.get('keyspace_hits'),
                'keyspace_misses': info.get('keyspace_misses')
            }

        except Exception as e:
            _logger.debug("Redis info failed: %s", e)
            self._available = False
            return {}

    def is_available(self) -> bool:
        """Check if Redis cache is currently available."""
        return self._available and self._ensure_connection()

    def close(self) -> None:
        """Close Redis connection pool."""
        if self._pool:
            try:
                self._pool.disconnect()
            except Exception as e:
                _logger.debug("Error closing Redis pool: %s", e)
            finally:
                self._pool = None
                self._client = None
                self._available = False