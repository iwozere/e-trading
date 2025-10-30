"""
Cache metrics collection and monitoring.

Provides comprehensive metrics for cache performance monitoring including
hit/miss ratios, response times, and cache size statistics.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Optional
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


@dataclass
class CacheStats:
    """Statistics for cache performance monitoring."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    total_response_time_ms: float = 0.0
    max_response_time_ms: float = 0.0
    min_response_time_ms: float = float('inf')
    cache_size: int = 0
    memory_usage_bytes: int = 0
    last_reset: datetime = field(default_factory=timezone.utcnow)

    @property
    def hit_ratio(self) -> float:
        """Calculate cache hit ratio."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def miss_ratio(self) -> float:
        """Calculate cache miss ratio."""
        return 1.0 - self.hit_ratio

    @property
    def avg_response_time_ms(self) -> float:
        """Calculate average response time."""
        total_ops = self.hits + self.misses + self.sets + self.deletes
        return self.total_response_time_ms / total_ops if total_ops > 0 else 0.0

    def reset(self) -> None:
        """Reset all statistics."""
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.evictions = 0
        self.total_response_time_ms = 0.0
        self.max_response_time_ms = 0.0
        self.min_response_time_ms = float('inf')
        self.last_reset = datetime.now(timezone.utc)


class CacheMetrics:
    """
    Cache metrics collector with automatic reporting and alerting.

    Collects and aggregates cache performance metrics across different
    cache tiers and provides monitoring capabilities.
    """

    def __init__(self, report_interval_seconds: int = 300):
        """
        Initialize cache metrics collector.

        Args:
            report_interval_seconds: Interval for automatic metrics reporting
        """
        self.report_interval = report_interval_seconds
        self.stats_by_tier: Dict[str, CacheStats] = {}
        self.last_report_time: Optional[datetime] = None
        self._operation_start_times: Dict[str, float] = {}

    def get_stats(self, tier: str) -> CacheStats:
        """Get statistics for a specific cache tier."""
        if tier not in self.stats_by_tier:
            self.stats_by_tier[tier] = CacheStats()
        return self.stats_by_tier[tier]

    def start_operation(self, operation_id: str) -> None:
        """Start timing a cache operation."""
        self._operation_start_times[operation_id] = time.time()

    def record_hit(self, tier: str, operation_id: Optional[str] = None) -> None:
        """Record a cache hit."""
        stats = self.get_stats(tier)
        stats.hits += 1
        self._record_operation_time(stats, operation_id)

    def record_miss(self, tier: str, operation_id: Optional[str] = None) -> None:
        """Record a cache miss."""
        stats = self.get_stats(tier)
        stats.misses += 1
        self._record_operation_time(stats, operation_id)

    def record_set(self, tier: str, operation_id: Optional[str] = None) -> None:
        """Record a cache set operation."""
        stats = self.get_stats(tier)
        stats.sets += 1
        self._record_operation_time(stats, operation_id)

    def record_delete(self, tier: str, operation_id: Optional[str] = None) -> None:
        """Record a cache delete operation."""
        stats = self.get_stats(tier)
        stats.deletes += 1
        self._record_operation_time(stats, operation_id)

    def record_eviction(self, tier: str) -> None:
        """Record a cache eviction."""
        stats = self.get_stats(tier)
        stats.evictions += 1

    def update_cache_size(self, tier: str, size: int, memory_bytes: int = 0) -> None:
        """Update cache size metrics."""
        stats = self.get_stats(tier)
        stats.cache_size = size
        stats.memory_usage_bytes = memory_bytes

    def _record_operation_time(self, stats: CacheStats, operation_id: Optional[str]) -> None:
        """Record operation timing if available."""
        if operation_id and operation_id in self._operation_start_times:
            elapsed_ms = (time.time() - self._operation_start_times[operation_id]) * 1000
            stats.total_response_time_ms += elapsed_ms
            stats.max_response_time_ms = max(stats.max_response_time_ms, elapsed_ms)
            stats.min_response_time_ms = min(stats.min_response_time_ms, elapsed_ms)
            del self._operation_start_times[operation_id]

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all cache metrics."""
        summary = {}
        for tier, stats in self.stats_by_tier.items():
            summary[tier] = {
                'hit_ratio': stats.hit_ratio,
                'miss_ratio': stats.miss_ratio,
                'total_operations': stats.hits + stats.misses + stats.sets + stats.deletes,
                'avg_response_time_ms': stats.avg_response_time_ms,
                'max_response_time_ms': stats.max_response_time_ms,
                'cache_size': stats.cache_size,
                'memory_usage_mb': stats.memory_usage_bytes / (1024 * 1024),
                'evictions': stats.evictions
            }
        return summary

    def should_report(self) -> bool:
        """Check if it's time for automatic reporting."""
        if self.last_report_time is None:
            return True
        return datetime.now(timezone.utc) - self.last_report_time > timedelta(seconds=self.report_interval)

    def report_metrics(self) -> None:
        """Log current metrics summary."""
        if not self.should_report():
            return

        summary = self.get_summary()
        for tier, metrics in summary.items():
            _logger.info(
                "Cache metrics [%s]: hit_ratio=%.3f, ops=%d, avg_time=%.2fms, size=%d, evictions=%d",
                tier,
                metrics['hit_ratio'],
                metrics['total_operations'],
                metrics['avg_response_time_ms'],
                metrics['cache_size'],
                metrics['evictions']
            )

        self.last_report_time = datetime.now(timezone.utc)

    def reset_stats(self, tier: Optional[str] = None) -> None:
        """Reset statistics for a specific tier or all tiers."""
        if tier:
            if tier in self.stats_by_tier:
                self.stats_by_tier[tier].reset()
        else:
            for stats in self.stats_by_tier.values():
                stats.reset()

    def check_performance_alerts(self) -> List[str]:
        """Check for performance issues and return alert messages."""
        alerts = []

        for tier, stats in self.stats_by_tier.items():
            # Low hit ratio alert
            if stats.hit_ratio < 0.5 and (stats.hits + stats.misses) > 100:
                alerts.append(f"Low hit ratio for {tier}: {stats.hit_ratio:.3f}")

            # High response time alert
            if stats.avg_response_time_ms > 100:
                alerts.append(f"High response time for {tier}: {stats.avg_response_time_ms:.2f}ms")

            # High eviction rate alert
            total_ops = stats.hits + stats.misses + stats.sets
            if total_ops > 0 and stats.evictions / total_ops > 0.1:
                alerts.append(f"High eviction rate for {tier}: {stats.evictions}/{total_ops}")

        return alerts


# Global metrics instance
_global_metrics: Optional[CacheMetrics] = None


def get_cache_metrics() -> CacheMetrics:
    """Get the global cache metrics instance."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = CacheMetrics()
    return _global_metrics