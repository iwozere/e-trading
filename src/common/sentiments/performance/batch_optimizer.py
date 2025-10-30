"""
Batch processing optimization for sentiment analysis.

Provides intelligent batch sizing, parallel processing coordination,
and adaptive performance optimization based on system resources and API constraints.
"""

import asyncio
import time
import math
from typing import List, Dict, Any, Optional, Callable, TypeVar, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import sys
import psutil
import os

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

T = TypeVar('T')


@dataclass
class BatchConfig:
    """Configuration for batch processing optimization."""
    # Batch sizing
    min_batch_size: int = 5
    max_batch_size: int = 100
    initial_batch_size: int = 20

    # Concurrency settings
    max_concurrent_batches: int = 4
    max_concurrent_per_adapter: int = 8

    # Performance thresholds
    target_batch_time_seconds: float = 30.0
    max_batch_time_seconds: float = 120.0
    memory_threshold_mb: int = 512

    # Adaptive optimization
    enable_adaptive_sizing: bool = True
    performance_window_size: int = 10
    adjustment_factor: float = 0.2

    # Resource monitoring
    cpu_threshold_percent: float = 80.0
    memory_threshold_percent: float = 85.0

    @classmethod
    def from_env(cls) -> 'BatchConfig':
        """Create configuration from environment variables."""
        return cls(
            min_batch_size=int(os.getenv("BATCH_MIN_SIZE", "5")),
            max_batch_size=int(os.getenv("BATCH_MAX_SIZE", "100")),
            initial_batch_size=int(os.getenv("BATCH_INITIAL_SIZE", "20")),
            max_concurrent_batches=int(os.getenv("BATCH_MAX_CONCURRENT", "4")),
            max_concurrent_per_adapter=int(os.getenv("BATCH_MAX_PER_ADAPTER", "8")),
            target_batch_time_seconds=float(os.getenv("BATCH_TARGET_TIME", "30.0")),
            max_batch_time_seconds=float(os.getenv("BATCH_MAX_TIME", "120.0")),
            memory_threshold_mb=int(os.getenv("BATCH_MEMORY_THRESHOLD", "512")),
            enable_adaptive_sizing=os.getenv("BATCH_ADAPTIVE", "true").lower() == "true",
            performance_window_size=int(os.getenv("BATCH_PERF_WINDOW", "10")),
            adjustment_factor=float(os.getenv("BATCH_ADJUSTMENT_FACTOR", "0.2")),
            cpu_threshold_percent=float(os.getenv("BATCH_CPU_THRESHOLD", "80.0")),
            memory_threshold_percent=float(os.getenv("BATCH_MEMORY_THRESHOLD", "85.0"))
        )


@dataclass
class BatchPerformanceMetrics:
    """Performance metrics for batch processing."""
    batch_size: int
    processing_time_seconds: float
    memory_usage_mb: float
    cpu_usage_percent: float
    success_rate: float
    throughput_items_per_second: float
    timestamp: datetime


class BatchOptimizer:
    """
    Intelligent batch processing optimizer with adaptive sizing.

    Automatically adjusts batch sizes based on performance metrics,
    system resources, and API constraints to maximize throughput
    while maintaining reliability.
    """

    def __init__(self, config: Optional[BatchConfig] = None):
        """
        Initialize batch optimizer.

        Args:
            config: Batch configuration (uses defaults if None)
        """
        self.config = config or BatchConfig.from_env()
        self._performance_history: List[BatchPerformanceMetrics] = []
        self._current_batch_sizes: Dict[str, int] = {}
        self._adapter_semaphores: Dict[str, asyncio.Semaphore] = {}
        self._global_semaphore = asyncio.Semaphore(self.config.max_concurrent_batches)
        self._last_resource_check = 0.0
        self._resource_check_interval = 5.0  # seconds

    def get_optimal_batch_size(self, adapter_name: str, item_count: int) -> int:
        """
        Get optimal batch size for an adapter based on performance history.

        Args:
            adapter_name: Name of the adapter
            item_count: Total number of items to process

        Returns:
            Optimal batch size for the adapter
        """
        if adapter_name not in self._current_batch_sizes:
            self._current_batch_sizes[adapter_name] = self.config.initial_batch_size

        current_size = self._current_batch_sizes[adapter_name]

        # Don't exceed available items
        if item_count < current_size:
            return min(item_count, self.config.max_batch_size)

        # Check if adaptive sizing is enabled
        if not self.config.enable_adaptive_sizing:
            return min(current_size, item_count, self.config.max_batch_size)

        # Analyze recent performance for this adapter
        recent_metrics = self._get_recent_metrics(adapter_name)
        if len(recent_metrics) < 3:
            return min(current_size, item_count, self.config.max_batch_size)

        # Calculate performance trend
        avg_time = sum(m.processing_time_seconds for m in recent_metrics) / len(recent_metrics)
        avg_throughput = sum(m.throughput_items_per_second for m in recent_metrics) / len(recent_metrics)
        avg_success_rate = sum(m.success_rate for m in recent_metrics) / len(recent_metrics)

        # Adjust batch size based on performance
        new_size = current_size

        if avg_time > self.config.target_batch_time_seconds and avg_success_rate > 0.9:
            # Too slow but reliable - decrease batch size
            new_size = max(
                self.config.min_batch_size,
                int(current_size * (1 - self.config.adjustment_factor))
            )
        elif avg_time < self.config.target_batch_time_seconds * 0.7 and avg_success_rate > 0.95:
            # Fast and reliable - increase batch size
            new_size = min(
                self.config.max_batch_size,
                int(current_size * (1 + self.config.adjustment_factor))
            )
        elif avg_success_rate < 0.8:
            # Poor reliability - decrease batch size significantly
            new_size = max(
                self.config.min_batch_size,
                int(current_size * 0.7)
            )

        # Check system resources before finalizing
        if self._should_reduce_load():
            new_size = max(self.config.min_batch_size, int(new_size * 0.8))

        self._current_batch_sizes[adapter_name] = new_size
        return min(new_size, item_count, self.config.max_batch_size)

    def create_batches(self, items: List[T], adapter_name: str) -> List[List[T]]:
        """
        Create optimally sized batches from a list of items.

        Args:
            items: List of items to batch
            adapter_name: Name of the adapter for optimization

        Returns:
            List of batches
        """
        if not items:
            return []

        batch_size = self.get_optimal_batch_size(adapter_name, len(items))
        batches = []

        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batches.append(batch)

        _logger.debug(
            "Created %d batches of size ~%d for adapter %s (%d items total)",
            len(batches), batch_size, adapter_name, len(items)
        )

        return batches

    async def process_batches_parallel(
        self,
        batches: List[List[T]],
        processor: Callable[[List[T]], Any],
        adapter_name: str
    ) -> List[Any]:
        """
        Process batches in parallel with concurrency control.

        Args:
            batches: List of batches to process
            processor: Async function to process each batch
            adapter_name: Name of the adapter for metrics

        Returns:
            List of processing results
        """
        if not batches:
            return []

        # Ensure adapter semaphore exists
        if adapter_name not in self._adapter_semaphores:
            self._adapter_semaphores[adapter_name] = asyncio.Semaphore(
                self.config.max_concurrent_per_adapter
            )

        adapter_semaphore = self._adapter_semaphores[adapter_name]

        async def process_single_batch(batch: List[T]) -> Any:
            """Process a single batch with performance tracking."""
            start_time = time.time()
            start_memory = self._get_memory_usage_mb()
            start_cpu = self._get_cpu_usage()

            async with self._global_semaphore:
                async with adapter_semaphore:
                    try:
                        result = await processor(batch)
                        success_rate = 1.0
                    except Exception as e:
                        _logger.warning("Batch processing failed for %s: %s", adapter_name, e)
                        result = None
                        success_rate = 0.0

            # Record performance metrics
            end_time = time.time()
            processing_time = end_time - start_time
            end_memory = self._get_memory_usage_mb()
            end_cpu = self._get_cpu_usage()

            metrics = BatchPerformanceMetrics(
                batch_size=len(batch),
                processing_time_seconds=processing_time,
                memory_usage_mb=end_memory - start_memory,
                cpu_usage_percent=(start_cpu + end_cpu) / 2,
                success_rate=success_rate,
                throughput_items_per_second=len(batch) / processing_time if processing_time > 0 else 0,
                timestamp=datetime.now(timezone.utc)
            )

            self._record_performance(adapter_name, metrics)
            return result

        # Process all batches concurrently
        _logger.info("Processing %d batches for adapter %s", len(batches), adapter_name)
        tasks = [process_single_batch(batch) for batch in batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and log them
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                _logger.error("Batch %d failed for adapter %s: %s", i, adapter_name, result)
            else:
                valid_results.append(result)

        return valid_results

    def _get_recent_metrics(self, adapter_name: str) -> List[BatchPerformanceMetrics]:
        """Get recent performance metrics for an adapter."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=30)
        return [
            m for m in self._performance_history[-self.config.performance_window_size:]
            if m.timestamp > cutoff_time and adapter_name in str(m)
        ]

    def _record_performance(self, adapter_name: str, metrics: BatchPerformanceMetrics) -> None:
        """Record performance metrics."""
        # Add adapter name to metrics for filtering
        metrics_dict = {
            'adapter': adapter_name,
            'metrics': metrics
        }

        self._performance_history.append(metrics)

        # Keep only recent history
        max_history = self.config.performance_window_size * 10
        if len(self._performance_history) > max_history:
            self._performance_history = self._performance_history[-max_history:]

        # Log performance if significant
        if metrics.processing_time_seconds > self.config.target_batch_time_seconds:
            _logger.debug(
                "Slow batch for %s: size=%d, time=%.2fs, throughput=%.1f/s",
                adapter_name, metrics.batch_size, metrics.processing_time_seconds,
                metrics.throughput_items_per_second
            )

    def _should_reduce_load(self) -> bool:
        """Check if system load is high and batch sizes should be reduced."""
        current_time = time.time()
        if current_time - self._last_resource_check < self._resource_check_interval:
            return False

        self._last_resource_check = current_time

        try:
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > self.config.cpu_threshold_percent:
                _logger.debug("High CPU usage detected: %.1f%%", cpu_percent)
                return True

            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > self.config.memory_threshold_percent:
                _logger.debug("High memory usage detected: %.1f%%", memory.percent)
                return True

            return False

        except Exception as e:
            _logger.debug("Failed to check system resources: %s", e)
            return False

    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            return psutil.cpu_percent(interval=0.1)
        except Exception:
            return 0.0

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of batch processing performance."""
        if not self._performance_history:
            return {}

        recent_metrics = self._performance_history[-50:]  # Last 50 batches

        return {
            'total_batches_processed': len(self._performance_history),
            'recent_batches': len(recent_metrics),
            'avg_batch_size': sum(m.batch_size for m in recent_metrics) / len(recent_metrics),
            'avg_processing_time': sum(m.processing_time_seconds for m in recent_metrics) / len(recent_metrics),
            'avg_throughput': sum(m.throughput_items_per_second for m in recent_metrics) / len(recent_metrics),
            'avg_success_rate': sum(m.success_rate for m in recent_metrics) / len(recent_metrics),
            'current_batch_sizes': dict(self._current_batch_sizes),
            'system_cpu_percent': self._get_cpu_usage(),
            'system_memory_mb': self._get_memory_usage_mb()
        }

    def reset_performance_history(self) -> None:
        """Reset performance history and batch size optimization."""
        self._performance_history.clear()
        self._current_batch_sizes.clear()
        _logger.info("Reset batch optimizer performance history")


# Global batch optimizer instance
_global_optimizer: Optional[BatchOptimizer] = None


def get_batch_optimizer() -> BatchOptimizer:
    """Get the global batch optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = BatchOptimizer()
    return _global_optimizer