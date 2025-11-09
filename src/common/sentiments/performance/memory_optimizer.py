"""
Memory-efficient data structures and optimization for sentiment analysis.

Provides memory-efficient data structures, streaming processing capabilities,
and memory usage monitoring to handle large datasets efficiently.
"""

import gc
import sys
import weakref
from typing import Any, Dict, List, Optional, Iterator
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
import psutil
import threading
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_mb: float
    available_mb: float
    used_mb: float
    percent_used: float
    process_rss_mb: float
    process_vms_mb: float
    timestamp: datetime = field(default_factory=timezone.utcnow)


class MemoryMonitor:
    """Monitor memory usage and provide alerts."""

    def __init__(self, warning_threshold_percent: float = 80.0,
                 critical_threshold_percent: float = 90.0):
        """
        Initialize memory monitor.

        Args:
            warning_threshold_percent: Memory usage percentage for warnings
            critical_threshold_percent: Memory usage percentage for critical alerts
        """
        self.warning_threshold = warning_threshold_percent
        self.critical_threshold = critical_threshold_percent
        self._last_warning_time = 0.0
        self._warning_interval = 60.0  # seconds

    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        try:
            # System memory
            memory = psutil.virtual_memory()

            # Process memory
            process = psutil.Process()
            process_memory = process.memory_info()

            return MemoryStats(
                total_mb=memory.total / (1024 * 1024),
                available_mb=memory.available / (1024 * 1024),
                used_mb=memory.used / (1024 * 1024),
                percent_used=memory.percent,
                process_rss_mb=process_memory.rss / (1024 * 1024),
                process_vms_mb=process_memory.vms / (1024 * 1024)
            )
        except Exception as e:
            _logger.debug("Failed to get memory stats: %s", e)
            return MemoryStats(0, 0, 0, 0, 0, 0)

    def check_memory_pressure(self) -> Optional[str]:
        """
        Check for memory pressure and return alert level.

        Returns:
            'critical', 'warning', or None
        """
        stats = self.get_memory_stats()

        if stats.percent_used >= self.critical_threshold:
            return 'critical'
        elif stats.percent_used >= self.warning_threshold:
            return 'warning'

        return None

    def log_memory_warning(self, force: bool = False) -> None:
        """Log memory warning if threshold exceeded."""
        import time
        current_time = time.time()

        if not force and current_time - self._last_warning_time < self._warning_interval:
            return

        alert_level = self.check_memory_pressure()
        if alert_level:
            stats = self.get_memory_stats()
            _logger.warning(
                "Memory pressure [%s]: %.1f%% used (%.1f/%.1f MB), process: %.1f MB RSS",
                alert_level.upper(),
                stats.percent_used,
                stats.used_mb,
                stats.total_mb,
                stats.process_rss_mb
            )
            self._last_warning_time = current_time


class StreamingDataProcessor:
    """Process data in streaming fashion to minimize memory usage."""

    def __init__(self, chunk_size: int = 1000, memory_monitor: Optional[MemoryMonitor] = None):
        """
        Initialize streaming processor.

        Args:
            chunk_size: Size of data chunks to process at once
            memory_monitor: Memory monitor for pressure detection
        """
        self.chunk_size = chunk_size
        self.memory_monitor = memory_monitor or MemoryMonitor()

    def process_in_chunks(self, data: List[Any],
                         processor: callable) -> Iterator[Any]:
        """
        Process data in chunks to minimize memory usage.

        Args:
            data: List of data items to process
            processor: Function to process each chunk

        Yields:
            Processed results for each chunk
        """
        for i in range(0, len(data), self.chunk_size):
            # Check memory pressure
            alert_level = self.memory_monitor.check_memory_pressure()
            if alert_level == 'critical':
                _logger.warning("Critical memory pressure, forcing garbage collection")
                gc.collect()

                # Reduce chunk size temporarily
                current_chunk_size = max(10, self.chunk_size // 2)
            elif alert_level == 'warning':
                current_chunk_size = max(50, int(self.chunk_size * 0.8))
            else:
                current_chunk_size = self.chunk_size

            # Process chunk
            chunk = data[i:i + current_chunk_size]
            try:
                result = processor(chunk)
                yield result
            except Exception as e:
                _logger.error("Chunk processing failed: %s", e)
                continue
            finally:
                # Clean up chunk reference
                del chunk

    def aggregate_streaming_results(self, results_iterator: Iterator[Any],
                                  aggregator: callable) -> Any:
        """
        Aggregate streaming results efficiently.

        Args:
            results_iterator: Iterator of results to aggregate
            aggregator: Function to aggregate results

        Returns:
            Aggregated result
        """
        accumulated = None
        processed_count = 0

        for result in results_iterator:
            if accumulated is None:
                accumulated = result
            else:
                accumulated = aggregator(accumulated, result)

            processed_count += 1

            # Periodic memory check and cleanup
            if processed_count % 10 == 0:
                self.memory_monitor.log_memory_warning()
                if processed_count % 50 == 0:
                    gc.collect()

        return accumulated


class MemoryEfficientDict:
    """Memory-efficient dictionary with automatic cleanup."""

    def __init__(self, max_size: int = 10000, cleanup_ratio: float = 0.2):
        """
        Initialize memory-efficient dictionary.

        Args:
            max_size: Maximum number of items to store
            cleanup_ratio: Ratio of items to remove when max_size is reached
        """
        self.max_size = max_size
        self.cleanup_ratio = cleanup_ratio
        self._data: Dict[Any, Any] = {}
        self._access_order = deque()
        self._lock = threading.RLock()

    def __getitem__(self, key: Any) -> Any:
        """Get item and update access order."""
        with self._lock:
            if key in self._data:
                # Move to end (most recently used)
                self._access_order.remove(key)
                self._access_order.append(key)
                return self._data[key]
            raise KeyError(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        """Set item with automatic cleanup."""
        with self._lock:
            # Update existing item
            if key in self._data:
                self._data[key] = value
                self._access_order.remove(key)
                self._access_order.append(key)
                return

            # Add new item
            self._data[key] = value
            self._access_order.append(key)

            # Cleanup if necessary
            if len(self._data) > self.max_size:
                self._cleanup()

    def __contains__(self, key: Any) -> bool:
        """Check if key exists."""
        return key in self._data

    def get(self, key: Any, default: Any = None) -> Any:
        """Get item with default value."""
        try:
            return self[key]
        except KeyError:
            return default

    def _cleanup(self) -> None:
        """Remove least recently used items."""
        items_to_remove = int(len(self._data) * self.cleanup_ratio)

        for _ in range(items_to_remove):
            if self._access_order:
                oldest_key = self._access_order.popleft()
                if oldest_key in self._data:
                    del self._data[oldest_key]

        _logger.debug("Cleaned up %d items from memory-efficient dict", items_to_remove)

    def clear(self) -> None:
        """Clear all items."""
        with self._lock:
            self._data.clear()
            self._access_order.clear()

    def size(self) -> int:
        """Get current size."""
        return len(self._data)


class MemoryEfficientDataStructures:
    """Collection of memory-efficient data structures for sentiment analysis."""

    @staticmethod
    def create_message_buffer(max_messages: int = 5000) -> MemoryEfficientDict:
        """Create memory-efficient buffer for sentiment messages."""
        return MemoryEfficientDict(max_size=max_messages, cleanup_ratio=0.3)

    @staticmethod
    def create_cache_dict(max_entries: int = 1000) -> MemoryEfficientDict:
        """Create memory-efficient cache dictionary."""
        return MemoryEfficientDict(max_size=max_entries, cleanup_ratio=0.2)

    @staticmethod
    def optimize_message_data(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimize message data structure for memory efficiency.

        Args:
            messages: List of message dictionaries

        Returns:
            Optimized message list with reduced memory footprint
        """
        optimized = []

        for msg in messages:
            # Keep only essential fields
            optimized_msg = {
                'body': msg.get('body', ''),
                'user': msg.get('user', {}).get('username', ''),
                'likes': int(msg.get('likes', 0)),
                'replies': int(msg.get('replies', 0)),
                'retweets': int(msg.get('retweets', 0)),
                'timestamp': msg.get('timestamp')
            }

            # Remove empty/null values
            optimized_msg = {k: v for k, v in optimized_msg.items()
                           if v is not None and v != ''}

            optimized.append(optimized_msg)

        return optimized

    @staticmethod
    def batch_process_with_memory_limit(data: List[Any],
                                      processor: callable,
                                      memory_limit_mb: float = 500.0) -> List[Any]:
        """
        Process data in batches with memory limit enforcement.

        Args:
            data: Data to process
            processor: Processing function
            memory_limit_mb: Memory limit in MB

        Returns:
            Processed results
        """
        monitor = MemoryMonitor()
        streaming_processor = StreamingDataProcessor(chunk_size=100, memory_monitor=monitor)

        results = []

        for chunk_result in streaming_processor.process_in_chunks(data, processor):
            results.append(chunk_result)

            # Check memory usage
            stats = monitor.get_memory_stats()
            if stats.process_rss_mb > memory_limit_mb:
                _logger.warning(
                    "Memory limit exceeded (%.1f MB > %.1f MB), forcing cleanup",
                    stats.process_rss_mb, memory_limit_mb
                )
                gc.collect()

        return results


class MemoryOptimizer:
    """
    Main memory optimization coordinator.

    Provides centralized memory optimization services including monitoring,
    cleanup, and efficient data structure management.
    """

    def __init__(self,
                 warning_threshold: float = 80.0,
                 critical_threshold: float = 90.0,
                 auto_cleanup: bool = True):
        """
        Initialize memory optimizer.

        Args:
            warning_threshold: Memory usage percentage for warnings
            critical_threshold: Memory usage percentage for critical alerts
            auto_cleanup: Enable automatic cleanup when thresholds are exceeded
        """
        self.monitor = MemoryMonitor(warning_threshold, critical_threshold)
        self.auto_cleanup = auto_cleanup
        self._cleanup_callbacks: List[callable] = []
        self._weak_refs: List[weakref.ref] = []

    def register_cleanup_callback(self, callback: callable) -> None:
        """Register a callback to be called during cleanup."""
        self._cleanup_callbacks.append(callback)

    def register_weak_reference(self, obj: Any) -> None:
        """Register a weak reference for automatic cleanup."""
        self._weak_refs.append(weakref.ref(obj))

    def force_cleanup(self) -> Dict[str, Any]:
        """Force memory cleanup and return statistics."""
        _logger.info("Forcing memory cleanup")

        # Get initial stats
        initial_stats = self.monitor.get_memory_stats()

        # Call cleanup callbacks
        for callback in self._cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                _logger.debug("Cleanup callback failed: %s", e)

        # Clean up weak references
        self._weak_refs = [ref for ref in self._weak_refs if ref() is not None]

        # Force garbage collection
        collected = gc.collect()

        # Get final stats
        final_stats = self.monitor.get_memory_stats()

        cleanup_result = {
            'initial_memory_mb': initial_stats.process_rss_mb,
            'final_memory_mb': final_stats.process_rss_mb,
            'memory_freed_mb': initial_stats.process_rss_mb - final_stats.process_rss_mb,
            'gc_objects_collected': collected,
            'callbacks_executed': len(self._cleanup_callbacks),
            'weak_refs_cleaned': len([ref for ref in self._weak_refs if ref() is None])
        }

        _logger.info(
            "Memory cleanup completed: freed %.1f MB, collected %d objects",
            cleanup_result['memory_freed_mb'], collected
        )

        return cleanup_result

    def check_and_cleanup(self) -> Optional[Dict[str, Any]]:
        """Check memory pressure and cleanup if necessary."""
        if not self.auto_cleanup:
            return None

        alert_level = self.monitor.check_memory_pressure()

        if alert_level == 'critical':
            return self.force_cleanup()
        elif alert_level == 'warning':
            self.monitor.log_memory_warning()
            # Light cleanup
            gc.collect()
            return {'light_cleanup': True}

        return None

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get memory optimization statistics."""
        stats = self.monitor.get_memory_stats()

        return {
            'memory_stats': {
                'total_mb': stats.total_mb,
                'used_mb': stats.used_mb,
                'percent_used': stats.percent_used,
                'process_rss_mb': stats.process_rss_mb
            },
            'optimizer_stats': {
                'cleanup_callbacks': len(self._cleanup_callbacks),
                'weak_references': len(self._weak_refs),
                'auto_cleanup_enabled': self.auto_cleanup,
                'warning_threshold': self.monitor.warning_threshold,
                'critical_threshold': self.monitor.critical_threshold
            }
        }


# Global memory optimizer instance
_global_optimizer: Optional[MemoryOptimizer] = None


def get_memory_optimizer() -> MemoryOptimizer:
    """Get the global memory optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = MemoryOptimizer()
    return _global_optimizer