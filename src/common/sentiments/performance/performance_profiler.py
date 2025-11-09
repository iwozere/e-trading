"""
Performance profiling and monitoring for sentiment analysis.

Provides comprehensive performance profiling, bottleneck detection,
and optimization recommendations for sentiment analysis operations.
"""

import time
import asyncio
import functools
from typing import Dict, List, Any, Optional, Callable, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from pathlib import Path
import sys
import threading
import cProfile
import pstats
import io

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


@dataclass
class ProfilerConfig:
    """Configuration for performance profiler."""
    enable_profiling: bool = True
    enable_detailed_profiling: bool = False
    max_profile_entries: int = 1000
    profile_threshold_seconds: float = 1.0
    report_interval_seconds: int = 300  # 5 minutes
    bottleneck_threshold_seconds: float = 5.0
    memory_profiling: bool = True


class TimingResult(NamedTuple):
    """Result of a timing measurement."""
    duration_seconds: float
    start_time: datetime
    end_time: datetime
    function_name: str
    args_hash: str


@dataclass
class PerformanceMetrics:
    """Performance metrics for a function or operation."""
    function_name: str
    call_count: int = 0
    total_time_seconds: float = 0.0
    min_time_seconds: float = float('inf')
    max_time_seconds: float = 0.0
    avg_time_seconds: float = 0.0
    last_call_time: Optional[datetime] = None
    error_count: int = 0
    recent_timings: deque = field(default_factory=lambda: deque(maxlen=100))

    def add_timing(self, duration: float, error: bool = False) -> None:
        """Add a timing measurement."""
        self.call_count += 1
        if error:
            self.error_count += 1
            return

        self.total_time_seconds += duration
        self.min_time_seconds = min(self.min_time_seconds, duration)
        self.max_time_seconds = max(self.max_time_seconds, duration)
        self.avg_time_seconds = self.total_time_seconds / (self.call_count - self.error_count)
        self.last_call_time = datetime.now(timezone.utc)
        self.recent_timings.append(duration)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.call_count == 0:
            return 0.0
        return (self.call_count - self.error_count) / self.call_count

    @property
    def recent_avg_time(self) -> float:
        """Calculate average time for recent calls."""
        if not self.recent_timings:
            return 0.0
        return sum(self.recent_timings) / len(self.recent_timings)


class PerformanceProfiler:
    """
    Comprehensive performance profiler for sentiment analysis operations.

    Provides function timing, bottleneck detection, and performance optimization
    recommendations with minimal overhead.
    """

    def __init__(self, config: Optional[ProfilerConfig] = None):
        """
        Initialize performance profiler.

        Args:
            config: Profiler configuration
        """
        self.config = config or ProfilerConfig()
        self._metrics: Dict[str, PerformanceMetrics] = {}
        self._active_timers: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._last_report_time = datetime.now(timezone.utc)
        self._profiler: Optional[cProfile.Profile] = None

        if self.config.enable_detailed_profiling:
            self._profiler = cProfile.Profile()

    def time_function(self, func_name: Optional[str] = None):
        """
        Decorator to time function execution.

        Args:
            func_name: Custom name for the function (uses actual name if None)
        """
        def decorator(func: Callable) -> Callable:
            name = func_name or f"{func.__module__}.{func.__name__}"

            if asyncio.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    return await self._time_async_call(name, func, *args, **kwargs)
                return async_wrapper
            else:
                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    return self._time_sync_call(name, func, *args, **kwargs)
                return sync_wrapper

        return decorator

    async def _time_async_call(self, name: str, func: Callable, *args, **kwargs) -> Any:
        """Time an async function call."""
        if not self.config.enable_profiling:
            return await func(*args, **kwargs)

        start_time = time.time()
        error_occurred = False

        try:
            if self._profiler:
                self._profiler.enable()

            result = await func(*args, **kwargs)
            return result

        except Exception:
            error_occurred = True
            raise

        finally:
            if self._profiler:
                self._profiler.disable()

            duration = time.time() - start_time
            self._record_timing(name, duration, error_occurred)

    def _time_sync_call(self, name: str, func: Callable, *args, **kwargs) -> Any:
        """Time a synchronous function call."""
        if not self.config.enable_profiling:
            return func(*args, **kwargs)

        start_time = time.time()
        error_occurred = False

        try:
            if self._profiler:
                self._profiler.enable()

            result = func(*args, **kwargs)
            return result

        except Exception:
            error_occurred = True
            raise

        finally:
            if self._profiler:
                self._profiler.disable()

            duration = time.time() - start_time
            self._record_timing(name, duration, error_occurred)

    def start_timer(self, operation_name: str) -> str:
        """
        Start a manual timer for an operation.

        Args:
            operation_name: Name of the operation

        Returns:
            Timer ID for stopping the timer
        """
        if not self.config.enable_profiling:
            return ""

        timer_id = f"{operation_name}_{time.time()}_{id(self)}"
        self._active_timers[timer_id] = time.time()
        return timer_id

    def stop_timer(self, timer_id: str, operation_name: str, error: bool = False) -> float:
        """
        Stop a manual timer and record the timing.

        Args:
            timer_id: Timer ID from start_timer
            operation_name: Name of the operation
            error: Whether an error occurred

        Returns:
            Duration in seconds
        """
        if not self.config.enable_profiling or not timer_id:
            return 0.0

        if timer_id not in self._active_timers:
            _logger.warning("Timer ID not found: %s", timer_id)
            return 0.0

        start_time = self._active_timers.pop(timer_id)
        duration = time.time() - start_time
        self._record_timing(operation_name, duration, error)
        return duration

    def _record_timing(self, name: str, duration: float, error: bool = False) -> None:
        """Record a timing measurement."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = PerformanceMetrics(function_name=name)

            self._metrics[name].add_timing(duration, error)

            # Log slow operations
            if duration > self.config.profile_threshold_seconds:
                _logger.debug(
                    "Slow operation detected: %s took %.3f seconds",
                    name, duration
                )

            # Check for bottlenecks
            if duration > self.config.bottleneck_threshold_seconds:
                _logger.warning(
                    "Performance bottleneck: %s took %.3f seconds",
                    name, duration
                )

    def get_metrics(self, function_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance metrics.

        Args:
            function_name: Specific function name (returns all if None)

        Returns:
            Performance metrics dictionary
        """
        with self._lock:
            if function_name:
                if function_name in self._metrics:
                    metrics = self._metrics[function_name]
                    return {
                        'function_name': metrics.function_name,
                        'call_count': metrics.call_count,
                        'total_time_seconds': metrics.total_time_seconds,
                        'avg_time_seconds': metrics.avg_time_seconds,
                        'min_time_seconds': metrics.min_time_seconds,
                        'max_time_seconds': metrics.max_time_seconds,
                        'recent_avg_time_seconds': metrics.recent_avg_time,
                        'success_rate': metrics.success_rate,
                        'error_count': metrics.error_count,
                        'last_call_time': metrics.last_call_time
                    }
                return {}

            # Return all metrics
            result = {}
            for name, metrics in self._metrics.items():
                result[name] = {
                    'call_count': metrics.call_count,
                    'avg_time_seconds': metrics.avg_time_seconds,
                    'recent_avg_time_seconds': metrics.recent_avg_time,
                    'success_rate': metrics.success_rate,
                    'total_time_seconds': metrics.total_time_seconds
                }

            return result

    def get_bottlenecks(self, min_calls: int = 5) -> List[Dict[str, Any]]:
        """
        Identify performance bottlenecks.

        Args:
            min_calls: Minimum number of calls to consider

        Returns:
            List of bottlenecks sorted by average time
        """
        bottlenecks = []

        with self._lock:
            for name, metrics in self._metrics.items():
                if (metrics.call_count >= min_calls and
                    metrics.avg_time_seconds > self.config.profile_threshold_seconds):

                    bottlenecks.append({
                        'function_name': name,
                        'avg_time_seconds': metrics.avg_time_seconds,
                        'call_count': metrics.call_count,
                        'total_time_seconds': metrics.total_time_seconds,
                        'success_rate': metrics.success_rate
                    })

        # Sort by average time (worst first)
        bottlenecks.sort(key=lambda x: x['avg_time_seconds'], reverse=True)
        return bottlenecks

    def get_optimization_recommendations(self) -> List[str]:
        """
        Generate optimization recommendations based on profiling data.

        Returns:
            List of optimization recommendations
        """
        recommendations = []
        bottlenecks = self.get_bottlenecks()

        if not bottlenecks:
            return ["No significant performance bottlenecks detected."]

        for bottleneck in bottlenecks[:5]:  # Top 5 bottlenecks
            name = bottleneck['function_name']
            avg_time = bottleneck['avg_time_seconds']
            call_count = bottleneck['call_count']

            if 'fetch' in name.lower() and avg_time > 2.0:
                recommendations.append(
                    f"Consider caching or batch processing for {name} "
                    f"(avg: {avg_time:.2f}s, calls: {call_count})"
                )
            elif 'hf' in name.lower() or 'huggingface' in name.lower():
                recommendations.append(
                    f"Consider batch processing or model optimization for {name} "
                    f"(avg: {avg_time:.2f}s, calls: {call_count})"
                )
            elif avg_time > 5.0:
                recommendations.append(
                    f"Investigate performance issues in {name} "
                    f"(avg: {avg_time:.2f}s, calls: {call_count})"
                )

        if len(bottlenecks) > 3:
            recommendations.append(
                "Multiple bottlenecks detected. Consider implementing "
                "parallel processing or caching strategies."
            )

        return recommendations

    def generate_profile_report(self) -> str:
        """
        Generate a comprehensive performance report.

        Returns:
            Formatted performance report
        """
        report_lines = [
            "=== Performance Profile Report ===",
            f"Generated at: {datetime.now(timezone.utc).isoformat()}",
            ""
        ]

        # Summary statistics
        with self._lock:
            total_functions = len(self._metrics)
            total_calls = sum(m.call_count for m in self._metrics.values())
            total_time = sum(m.total_time_seconds for m in self._metrics.values())

        report_lines.extend([
            f"Total functions profiled: {total_functions}",
            f"Total function calls: {total_calls}",
            f"Total execution time: {total_time:.3f} seconds",
            ""
        ])

        # Top functions by time
        metrics_by_time = sorted(
            self._metrics.items(),
            key=lambda x: x[1].total_time_seconds,
            reverse=True
        )

        report_lines.append("Top functions by total time:")
        for name, metrics in metrics_by_time[:10]:
            report_lines.append(
                f"  {name}: {metrics.total_time_seconds:.3f}s "
                f"({metrics.call_count} calls, avg: {metrics.avg_time_seconds:.3f}s)"
            )

        report_lines.append("")

        # Bottlenecks
        bottlenecks = self.get_bottlenecks()
        if bottlenecks:
            report_lines.append("Performance bottlenecks:")
            for bottleneck in bottlenecks[:5]:
                report_lines.append(
                    f"  {bottleneck['function_name']}: "
                    f"avg {bottleneck['avg_time_seconds']:.3f}s "
                    f"({bottleneck['call_count']} calls)"
                )
        else:
            report_lines.append("No significant bottlenecks detected.")

        report_lines.append("")

        # Recommendations
        recommendations = self.get_optimization_recommendations()
        if recommendations:
            report_lines.append("Optimization recommendations:")
            for rec in recommendations:
                report_lines.append(f"  - {rec}")

        # Detailed profiling data
        if self._profiler and self.config.enable_detailed_profiling:
            report_lines.extend([
                "",
                "=== Detailed Profile Data ===",
                ""
            ])

            s = io.StringIO()
            ps = pstats.Stats(self._profiler, stream=s)
            ps.sort_stats('cumulative').print_stats(20)
            report_lines.append(s.getvalue())

        return "\n".join(report_lines)

    def should_report(self) -> bool:
        """Check if it's time to generate a report."""
        return (datetime.now(timezone.utc) - self._last_report_time >
                timedelta(seconds=self.config.report_interval_seconds))

    def auto_report(self) -> None:
        """Generate and log performance report if interval has passed."""
        if self.should_report():
            report = self.generate_profile_report()
            _logger.info("Performance Report:\n%s", report)
            self._last_report_time = datetime.now(timezone.utc)

    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        with self._lock:
            self._metrics.clear()
            self._active_timers.clear()
            if self._profiler:
                self._profiler.clear()

        _logger.info("Performance metrics reset")

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary performance statistics."""
        with self._lock:
            if not self._metrics:
                return {}

            total_calls = sum(m.call_count for m in self._metrics.values())
            total_time = sum(m.total_time_seconds for m in self._metrics.values())
            avg_success_rate = sum(m.success_rate for m in self._metrics.values()) / len(self._metrics)

            return {
                'total_functions': len(self._metrics),
                'total_calls': total_calls,
                'total_time_seconds': total_time,
                'avg_success_rate': avg_success_rate,
                'bottleneck_count': len(self.get_bottlenecks()),
                'profiling_enabled': self.config.enable_profiling,
                'detailed_profiling_enabled': self.config.enable_detailed_profiling
            }


# Global profiler instance
_global_profiler: Optional[PerformanceProfiler] = None


def get_performance_profiler() -> PerformanceProfiler:
    """Get the global performance profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler


# Convenience decorators
def profile_function(func_name: Optional[str] = None):
    """Convenience decorator for profiling functions."""
    return get_performance_profiler().time_function(func_name)


def profile_async(func_name: Optional[str] = None):
    """Convenience decorator for profiling async functions."""
    return get_performance_profiler().time_function(func_name)