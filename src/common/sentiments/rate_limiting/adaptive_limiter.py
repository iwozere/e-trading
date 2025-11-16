"""
Adaptive rate limiting based on API response times and error rates.

Provides intelligent rate limiting that automatically adjusts based on
API performance, error rates, and response times to optimize throughput
while respecting API constraints.
"""

import statistics
from typing import Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from collections import deque
from pathlib import Path
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from .rate_limiter import RateLimiter, RateLimitConfig

_logger = setup_logger(__name__)


@dataclass
class ResponseMetrics:
    """Metrics for API response analysis."""
    response_time_ms: float
    status_code: Optional[int]
    success: bool
    timestamp: datetime
    rate_limited: bool = False
    error_type: Optional[str] = None


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive rate limiting."""
    # Base rate limiting
    base_requests_per_second: float = 1.0
    min_requests_per_second: float = 0.1
    max_requests_per_second: float = 10.0

    # Adaptation parameters
    adaptation_window_size: int = 50
    adaptation_interval_seconds: int = 30
    response_time_threshold_ms: float = 2000.0
    error_rate_threshold: float = 0.1

    # Adjustment factors
    increase_factor: float = 1.2
    decrease_factor: float = 0.8
    aggressive_decrease_factor: float = 0.5

    # Stability requirements
    min_samples_for_adjustment: int = 10
    stability_window_size: int = 20


class AdaptiveRateLimiter:
    """
    Adaptive rate limiter that adjusts based on API performance.

    Monitors API response times, error rates, and rate limiting signals
    to automatically optimize the request rate for maximum throughput
    while respecting API constraints.
    """

    def __init__(self, config: AdaptiveConfig, name: str = "adaptive"):
        """
        Initialize adaptive rate limiter.

        Args:
            config: Adaptive configuration
            name: Name for this limiter instance
        """
        self.config = config
        self.name = name

        # Initialize base rate limiter
        rate_config = RateLimitConfig(
            requests_per_second=config.base_requests_per_second,
            burst_capacity=max(5, int(config.base_requests_per_second * 2)),
            strict_mode=False
        )
        self.rate_limiter = RateLimiter(rate_config, name)

        # Response tracking
        self.response_history: deque[ResponseMetrics] = deque(
            maxlen=config.adaptation_window_size
        )

        # Adaptation state
        self.current_rate = config.base_requests_per_second
        self.last_adaptation_time = datetime.now(timezone.utc)
        self.consecutive_good_periods = 0
        self.consecutive_bad_periods = 0

        # Statistics
        self.total_adaptations = 0
        self.rate_increases = 0
        self.rate_decreases = 0

    async def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire permission to make a request.

        Args:
            timeout: Maximum time to wait

        Returns:
            True if permission granted
        """
        return await self.rate_limiter.acquire(1, timeout)

    def record_response(self, response_time_ms: float, success: bool,
                       status_code: Optional[int] = None,
                       error_type: Optional[str] = None) -> None:
        """
        Record API response metrics for adaptation.

        Args:
            response_time_ms: Response time in milliseconds
            success: Whether the request was successful
            status_code: HTTP status code (if applicable)
            error_type: Type of error (if any)
        """
        # Detect rate limiting
        rate_limited = (
            status_code in [429, 503] or
            error_type in ['rate_limit', 'throttled'] or
            response_time_ms > self.config.response_time_threshold_ms * 2
        )

        metrics = ResponseMetrics(
            response_time_ms=response_time_ms,
            status_code=status_code,
            success=success,
            timestamp=datetime.now(timezone.utc),
            rate_limited=rate_limited,
            error_type=error_type
        )

        self.response_history.append(metrics)

        # Trigger immediate adaptation if rate limited
        if rate_limited:
            _logger.debug("Rate limiting detected for %s, triggering adaptation", self.name)
            self._adapt_rate(force=True)

        # Check for periodic adaptation
        self._check_adaptation_schedule()

    def _check_adaptation_schedule(self) -> None:
        """Check if it's time for periodic rate adaptation."""
        now = datetime.now(timezone.utc)
        time_since_last = (now - self.last_adaptation_time).total_seconds()

        if (time_since_last >= self.config.adaptation_interval_seconds and
            len(self.response_history) >= self.config.min_samples_for_adjustment):
            self._adapt_rate()

    def _adapt_rate(self, force: bool = False) -> None:
        """
        Adapt the rate limit based on recent performance.

        Args:
            force: Force adaptation even if not enough time has passed
        """
        if not force:
            now = datetime.now(timezone.utc)
            time_since_last = (now - self.last_adaptation_time).total_seconds()
            if time_since_last < self.config.adaptation_interval_seconds:
                return

        if len(self.response_history) < self.config.min_samples_for_adjustment:
            return

        # Analyze recent performance
        recent_responses = list(self.response_history)[-self.config.stability_window_size:]

        # Calculate metrics
        error_rate = sum(1 for r in recent_responses if not r.success) / len(recent_responses)
        rate_limited_rate = sum(1 for r in recent_responses if r.rate_limited) / len(recent_responses)

        successful_responses = [r for r in recent_responses if r.success and not r.rate_limited]
        avg_response_time = 0.0
        if successful_responses:
            avg_response_time = statistics.mean(r.response_time_ms for r in successful_responses)

        # Determine adaptation action
        old_rate = self.current_rate
        adaptation_reason = ""

        if rate_limited_rate > 0.1:  # More than 10% rate limited
            # Aggressive decrease
            self.current_rate *= self.config.aggressive_decrease_factor
            self.consecutive_bad_periods += 1
            self.consecutive_good_periods = 0
            adaptation_reason = f"rate_limited ({rate_limited_rate:.1%})"

        elif error_rate > self.config.error_rate_threshold:
            # Moderate decrease due to errors
            self.current_rate *= self.config.decrease_factor
            self.consecutive_bad_periods += 1
            self.consecutive_good_periods = 0
            adaptation_reason = f"high_error_rate ({error_rate:.1%})"

        elif avg_response_time > self.config.response_time_threshold_ms:
            # Decrease due to slow responses
            self.current_rate *= self.config.decrease_factor
            self.consecutive_bad_periods += 1
            self.consecutive_good_periods = 0
            adaptation_reason = f"slow_response ({avg_response_time:.0f}ms)"

        elif (error_rate < 0.05 and
              avg_response_time < self.config.response_time_threshold_ms * 0.5 and
              rate_limited_rate == 0):
            # Good performance - consider increasing
            self.consecutive_good_periods += 1
            self.consecutive_bad_periods = 0

            # Only increase after multiple good periods
            if self.consecutive_good_periods >= 3:
                self.current_rate *= self.config.increase_factor
                self.consecutive_good_periods = 0
                adaptation_reason = f"good_performance (err:{error_rate:.1%}, rt:{avg_response_time:.0f}ms)"
        else:
            # Stable performance - no change
            adaptation_reason = "stable"

        # Apply bounds
        self.current_rate = max(
            self.config.min_requests_per_second,
            min(self.config.max_requests_per_second, self.current_rate)
        )

        # Update rate limiter if rate changed
        if abs(self.current_rate - old_rate) > 0.01:
            self._update_rate_limiter()
            self.total_adaptations += 1

            if self.current_rate > old_rate:
                self.rate_increases += 1
            else:
                self.rate_decreases += 1

            _logger.info(
                "Adapted rate for %s: %.2f -> %.2f req/s (reason: %s)",
                self.name, old_rate, self.current_rate, adaptation_reason
            )

        self.last_adaptation_time = datetime.now(timezone.utc)

    def _update_rate_limiter(self) -> None:
        """Update the underlying rate limiter with new rate."""
        new_config = RateLimitConfig(
            requests_per_second=self.current_rate,
            burst_capacity=max(5, int(self.current_rate * 2)),
            strict_mode=False
        )
        self.rate_limiter.update_config(new_config)

    def get_current_rate(self) -> float:
        """Get current requests per second rate."""
        return self.current_rate

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance analysis metrics."""
        if not self.response_history:
            return {}

        recent_responses = list(self.response_history)
        successful_responses = [r for r in recent_responses if r.success]

        metrics = {
            'total_responses': len(recent_responses),
            'successful_responses': len(successful_responses),
            'error_rate': (len(recent_responses) - len(successful_responses)) / len(recent_responses),
            'rate_limited_count': sum(1 for r in recent_responses if r.rate_limited),
            'current_rate': self.current_rate,
            'base_rate': self.config.base_requests_per_second
        }

        if successful_responses:
            response_times = [r.response_time_ms for r in successful_responses]
            metrics.update({
                'avg_response_time_ms': statistics.mean(response_times),
                'median_response_time_ms': statistics.median(response_times),
                'p95_response_time_ms': statistics.quantiles(response_times, n=20)[18] if len(response_times) > 10 else max(response_times),
                'min_response_time_ms': min(response_times),
                'max_response_time_ms': max(response_times)
            })

        return metrics

    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get adaptation behavior statistics."""
        return {
            'total_adaptations': self.total_adaptations,
            'rate_increases': self.rate_increases,
            'rate_decreases': self.rate_decreases,
            'consecutive_good_periods': self.consecutive_good_periods,
            'consecutive_bad_periods': self.consecutive_bad_periods,
            'adaptation_efficiency': self.rate_increases / max(1, self.total_adaptations),
            'last_adaptation_time': self.last_adaptation_time,
            'response_history_size': len(self.response_history)
        }

    def reset_adaptation_history(self) -> None:
        """Reset adaptation history and statistics."""
        self.response_history.clear()
        self.current_rate = self.config.base_requests_per_second
        self.consecutive_good_periods = 0
        self.consecutive_bad_periods = 0
        self.total_adaptations = 0
        self.rate_increases = 0
        self.rate_decreases = 0
        self._update_rate_limiter()
        _logger.info("Reset adaptation history for %s", self.name)

    def force_rate_adjustment(self, new_rate: float, reason: str = "manual") -> None:
        """
        Manually set a new rate limit.

        Args:
            new_rate: New requests per second rate
            reason: Reason for the adjustment
        """
        old_rate = self.current_rate
        self.current_rate = max(
            self.config.min_requests_per_second,
            min(self.config.max_requests_per_second, new_rate)
        )

        self._update_rate_limiter()
        self.total_adaptations += 1

        _logger.info(
            "Manual rate adjustment for %s: %.2f -> %.2f req/s (reason: %s)",
            self.name, old_rate, self.current_rate, reason
        )

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics including base limiter stats."""
        stats = {
            'adaptive_metrics': self.get_performance_metrics(),
            'adaptation_stats': self.get_adaptation_statistics(),
            'base_limiter_stats': self.rate_limiter.get_statistics(),
            'config': {
                'base_rate': self.config.base_requests_per_second,
                'min_rate': self.config.min_requests_per_second,
                'max_rate': self.config.max_requests_per_second,
                'response_threshold_ms': self.config.response_time_threshold_ms,
                'error_threshold': self.config.error_rate_threshold
            }
        }
        return stats