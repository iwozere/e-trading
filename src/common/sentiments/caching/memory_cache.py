"""
In-memory cache implementation with LRU eviction.

Provides fast in-memory caching with configurable TTL and LRU eviction policy.
Used as the default cache tier and fallback when Redis is unavailable.
"""

import time
import threading
from typing import Any, Optional, Dict, Tuple
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import sys
import json
import pickle

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from .cache_metrics import get_cache_metrics

_logger = setup_logger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with value and expiration time."""
    value: Any
    expires_at: float
    created_at: float
    access_count: int = 0
    last_accessed: float = 0.0

    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return time.time() > self.expires_at

    def touch(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = time.time()


class MemoryCache:
    """
    Thread-safe in-memory cache with LRU eviction and TTL support.

    Features:
    - LRU (Least Recently Used) eviction policy
    - TTL (Time To Live) expiration
    - Thread-safe operations
    - Memory usage tracking
    - Automatic cleanup of expired entries
    """

    def __init__(self, max_size: int = 1000, default_ttl_seconds: int = 3600):
        """
        Initialize memory cache.

        Args:
            max_size: Maximum number of entries to store
            default_ttl_seconds: Default TTL for cache entries
        """
        self.max_size = max_size
        self.default_ttl = default_ttl_seconds
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._metrics = get_cache_metrics()
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # 5 minutes

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        operation_id = f"mem_get_{id(self)}_{time.time()}"
        self._metrics.start_operation(operation_id)

        with self._lock:
            self._maybe_cleanup()

            if key not in self._cache:
                self._metrics.record_miss("memory", operation_id)
                return None

            entry = self._cache[key]

            # Check expiration
            if entry.is_expired():
                del self._cache[key]
                self._metrics.record_miss("memory", operation_id)
                self._metrics.record_eviction("memory")
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()

            self._metrics.record_hit("memory", operation_id)
            self._update_size_metrics()
            return entry.value

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: TTL override, uses default if None

        Returns:
            True if successfully cached
        """
        operation_id = f"mem_set_{id(self)}_{time.time()}"
        self._metrics.start_operation(operation_id)

        try:
            with self._lock:
                ttl = ttl_seconds or self.default_ttl
                expires_at = time.time() + ttl
                created_at = time.time()

                entry = CacheEntry(
                    value=value,
                    expires_at=expires_at,
                    created_at=created_at
                )

                # Remove existing entry if present
                if key in self._cache:
                    del self._cache[key]

                # Add new entry
                self._cache[key] = entry

                # Evict oldest entries if over capacity
                while len(self._cache) > self.max_size:
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                    self._metrics.record_eviction("memory")

                self._metrics.record_set("memory", operation_id)
                self._update_size_metrics()
                return True

        except Exception as e:
            _logger.warning("Failed to set cache entry %s: %s", key, e)
            return False

    def delete(self, key: str) -> bool:
        """
        Delete entry from cache.

        Args:
            key: Cache key to delete

        Returns:
            True if entry was deleted, False if not found
        """
        operation_id = f"mem_del_{id(self)}_{time.time()}"
        self._metrics.start_operation(operation_id)

        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._metrics.record_delete("memory", operation_id)
                self._update_size_metrics()
                return True

            return False

    def clear(self) -> None:
        """Clear all entries from cache."""
        with self._lock:
            self._cache.clear()
            self._update_size_metrics()

    def exists(self, key: str) -> bool:
        """
        Check if key exists and is not expired.

        Args:
            key: Cache key to check

        Returns:
            True if key exists and is valid
        """
        with self._lock:
            if key not in self._cache:
                return False

            entry = self._cache[key]
            if entry.is_expired():
                del self._cache[key]
                self._metrics.record_eviction("memory")
                return False

            return True

    def get_ttl(self, key: str) -> Optional[int]:
        """
        Get remaining TTL for a key.

        Args:
            key: Cache key

        Returns:
            Remaining TTL in seconds, or None if key doesn't exist
        """
        with self._lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]
            if entry.is_expired():
                del self._cache[key]
                self._metrics.record_eviction("memory")
                return None

            return int(entry.expires_at - time.time())

    def get_size(self) -> int:
        """Get current number of entries in cache."""
        with self._lock:
            return len(self._cache)

    def get_memory_usage(self) -> int:
        """
        Estimate memory usage in bytes.

        Returns:
            Estimated memory usage in bytes
        """
        with self._lock:
            try:
                # Rough estimation using pickle serialization
                total_size = 0
                sample_size = min(10, len(self._cache))

                if sample_size == 0:
                    return 0

                # Sample a few entries to estimate average size
                sample_keys = list(self._cache.keys())[:sample_size]
                for key in sample_keys:
                    entry = self._cache[key]
                    try:
                        key_size = len(key.encode('utf-8'))
                        value_size = len(pickle.dumps(entry.value))
                        total_size += key_size + value_size + 64  # overhead estimate
                    except Exception:
                        total_size += 256  # fallback estimate

                # Extrapolate to full cache
                avg_entry_size = total_size / sample_size
                return int(avg_entry_size * len(self._cache))

            except Exception as e:
                _logger.debug("Failed to estimate memory usage: %s", e)
                return len(self._cache) * 256  # rough fallback

    def cleanup_expired(self) -> int:
        """
        Remove expired entries from cache.

        Returns:
            Number of entries removed
        """
        with self._lock:
            expired_keys = []
            current_time = time.time()

            for key, entry in self._cache.items():
                if current_time > entry.expires_at:
                    expired_keys.append(key)

            for key in expired_keys:
                del self._cache[key]
                self._metrics.record_eviction("memory")

            if expired_keys:
                self._update_size_metrics()

            return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            stats = {
                'size': len(self._cache),
                'max_size': self.max_size,
                'memory_usage_bytes': self.get_memory_usage(),
                'default_ttl_seconds': self.default_ttl
            }

            if self._cache:
                entries = list(self._cache.values())
                current_time = time.time()

                stats.update({
                    'avg_age_seconds': sum(current_time - e.created_at for e in entries) / len(entries),
                    'avg_access_count': sum(e.access_count for e in entries) / len(entries),
                    'expired_count': sum(1 for e in entries if e.is_expired())
                })

            return stats

    def _maybe_cleanup(self) -> None:
        """Perform cleanup if interval has passed."""
        current_time = time.time()
        if current_time - self._last_cleanup > self._cleanup_interval:
            expired_count = self.cleanup_expired()
            if expired_count > 0:
                _logger.debug("Cleaned up %d expired entries from memory cache", expired_count)
            self._last_cleanup = current_time

    def _update_size_metrics(self) -> None:
        """Update cache size metrics."""
        self._metrics.update_cache_size(
            "memory",
            len(self._cache),
            self.get_memory_usage()
        )