"""
Rate limiting utilities for data providers.

This module provides rate limiting functionality to ensure compliance
with API rate limits across different data providers.
"""

import time
import threading
from typing import Dict, Optional, Any
import logging

_logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Thread-safe rate limiter for API requests.

    Supports both per-second and per-minute rate limits with
    configurable burst handling and backoff strategies.
    """

    def __init__(
        self,
        requests_per_second: int = 10,
        requests_per_minute: int = 600,
        burst_size: int = 5,
        backoff_factor: float = 2.0,
        max_backoff: float = 60.0
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_second: Maximum requests per second
            requests_per_minute: Maximum requests per minute
            burst_size: Maximum burst requests allowed
            backoff_factor: Multiplicative factor for backoff
            max_backoff: Maximum backoff delay in seconds
        """
        self.requests_per_second = requests_per_second
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.backoff_factor = backoff_factor
        self.max_backoff = max_backoff

        # Thread safety
        self._lock = threading.RLock()

        # Request tracking
        self._second_requests = []
        self._minute_requests = []
        self._consecutive_failures = 0
        self._last_failure_time = None

        # Statistics
        self._total_requests = 0
        self._total_delays = 0
        self._total_failures = 0

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire permission to make a request.

        Args:
            timeout: Maximum time to wait for permission (None = wait indefinitely)

        Returns:
            True if permission granted, False if timeout
        """
        start_time = time.time()

        while True:
            with self._lock:
                if self._can_make_request():
                    self._record_request()
                    return True

            # Check timeout
            if timeout is not None and (time.time() - start_time) > timeout:
                return False

            # Calculate delay
            delay = self._calculate_delay()
            time.sleep(delay)

    def _can_make_request(self) -> bool:
        """Check if a request can be made now."""
        now = time.time()

        # Clean old requests
        self._cleanup_old_requests(now)

        # Check second limit
        if len(self._second_requests) >= self.requests_per_second:
            return False

        # Check minute limit
        if len(self._minute_requests) >= self.requests_per_minute:
            return False

        # Check burst limit
        if len(self._second_requests) >= self.burst_size:
            return False

        return True

    def _cleanup_old_requests(self, now: float):
        """Remove old request records."""
        # Remove requests older than 1 second
        cutoff_second = now - 1.0
        self._second_requests = [t for t in self._second_requests if t > cutoff_second]

        # Remove requests older than 1 minute
        cutoff_minute = now - 60.0
        self._minute_requests = [t for t in self._minute_requests if t > cutoff_minute]

    def _record_request(self):
        """Record that a request was made."""
        now = time.time()
        self._second_requests.append(now)
        self._minute_requests.append(now)
        self._total_requests += 1

    def _calculate_delay(self) -> float:
        """Calculate delay before next request attempt."""
        base_delay = 1.0 / self.requests_per_second

        # Apply backoff if we've had recent failures
        if self._consecutive_failures > 0:
            backoff_delay = base_delay * (self.backoff_factor ** self._consecutive_failures)
            backoff_delay = min(backoff_delay, self.max_backoff)
            return max(base_delay, backoff_delay)

        return base_delay

    def record_success(self):
        """Record a successful request."""
        with self._lock:
            self._consecutive_failures = 0
            self._last_failure_time = None

    def record_failure(self, error: Optional[Exception] = None):
        """Record a failed request."""
        with self._lock:
            self._consecutive_failures += 1
            self._last_failure_time = time.time()
            self._total_failures += 1

            if error:
                _logger.warning(
                    "Request failed (consecutive failures: %d): %s",
                    self._consecutive_failures, error
                )

    def wait_if_needed(self):
        """Wait if necessary to comply with rate limits."""
        while not self._can_make_request():
            delay = self._calculate_delay()
            time.sleep(delay)
            self._total_delays += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        with self._lock:
            return {
                'total_requests': self._total_requests,
                'total_delays': self._total_delays,
                'total_failures': self._total_failures,
                'consecutive_failures': self._consecutive_failures,
                'current_second_requests': len(self._second_requests),
                'current_minute_requests': len(self._minute_requests),
                'requests_per_second': self.requests_per_second,
                'requests_per_minute': self.requests_per_minute
            }

    def reset_stats(self):
        """Reset statistics."""
        with self._lock:
            self._total_requests = 0
            self._total_delays = 0
            self._total_failures = 0
            self._consecutive_failures = 0


class ProviderRateLimiter:
    """
    Manages rate limits for multiple data providers.

    Each provider can have its own rate limiting rules and
    the system automatically handles provider-specific limits.
    """

    def __init__(self):
        """Initialize provider rate limiter."""
        self._limiters: Dict[str, RateLimiter] = {}
        self._lock = threading.RLock()

    def get_limiter(self, provider: str) -> RateLimiter:
        """
        Get or create a rate limiter for a specific provider.

        Args:
            provider: Provider name (e.g., 'binance', 'yahoo')

        Returns:
            RateLimiter instance for the provider
        """
        with self._lock:
            if provider not in self._limiters:
                # Create default limiter based on provider
                self._limiters[provider] = self._create_default_limiter(provider)

            return self._limiters[provider]

    def _create_default_limiter(self, provider: str) -> RateLimiter:
        """Create a default rate limiter for a provider."""
        # Default limits based on common provider restrictions
        defaults = {
            'binance': (10, 1200),      # 10 req/s, 1200 req/min
            'yahoo': (100, 2000),       # 100 req/s, 2000 req/min
            'alpha_vantage': (1, 5),    # 1 req/s, 5 req/min (free tier)
            'finnhub': (1, 60),         # 1 req/s, 60 req/min (free tier)
            'polygon': (1, 5),          # 1 req/s, 5 req/min (free tier)
            'twelve_data': (8, 800),    # 8 req/s, 800 req/min (free tier)
            'fmp': (5, 300),            # 5 req/s, 300 req/min (free tier)
            'coingecko': (1, 50),       # 1 req/s, 50 req/min (free tier)
            'ibkr': (10, 100),          # 10 req/s, 100 req/min
        }

        req_per_sec, req_per_min = defaults.get(provider, (10, 600))
        return RateLimiter(req_per_sec, req_per_min)

    def configure_provider(
        self,
        provider: str,
        requests_per_second: int,
        requests_per_minute: int,
        **kwargs
    ):
        """
        Configure rate limits for a specific provider.

        Args:
            provider: Provider name
            requests_per_second: Maximum requests per second
            requests_per_minute: Maximum requests per minute
            **kwargs: Additional RateLimiter parameters
        """
        with self._lock:
            self._limiters[provider] = RateLimiter(
                requests_per_second=requests_per_second,
                requests_per_minute=requests_per_minute,
                **kwargs
            )

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all providers."""
        with self._lock:
            return {
                provider: limiter.get_stats()
                for provider, limiter in self._limiters.items()
            }

    def reset_all_stats(self):
        """Reset statistics for all providers."""
        with self._lock:
            for limiter in self._limiters.values():
                limiter.reset_stats()


class AdaptiveRateLimiter(RateLimiter):
    """
    Adaptive rate limiter that adjusts limits based on response patterns.

    Automatically reduces rate limits when errors occur and
    gradually increases them when responses are successful.
    """

    def __init__(
        self,
        requests_per_second: int = 10,
        requests_per_minute: int = 600,
        min_requests_per_second: int = 1,
        adaptation_factor: float = 0.8,
        recovery_factor: float = 1.1,
        **kwargs
    ):
        """
        Initialize adaptive rate limiter.

        Args:
            requests_per_second: Initial requests per second
            requests_per_minute: Initial requests per minute
            min_requests_per_second: Minimum requests per second
            adaptation_factor: Factor to reduce limits on failure
            recovery_factor: Factor to increase limits on success
            **kwargs: Additional RateLimiter parameters
        """
        super().__init__(requests_per_second, requests_per_minute, **kwargs)

        self.min_requests_per_second = min_requests_per_second
        self.adaptation_factor = adaptation_factor
        self.recovery_factor = recovery_factor

        self._original_requests_per_second = requests_per_second
        self._original_requests_per_minute = requests_per_minute

    def record_failure(self, error: Optional[Exception] = None):
        """Record a failed request and adapt rate limits."""
        super().record_failure(error)

        # Reduce rate limits on failure
        self.requests_per_second = max(
            self.min_requests_per_second,
            int(self.requests_per_second * self.adaptation_factor)
        )
        self.requests_per_minute = max(
            self.min_requests_per_second * 60,
            int(self.requests_per_minute * self.adaptation_factor)
        )

        _logger.info(
            "Adapted rate limits after failure: %d req/s, %d req/min",
            self.requests_per_second, self.requests_per_minute
        )

    def record_success(self):
        """Record a successful request and potentially recover rate limits."""
        super().record_success()

        # Gradually recover rate limits on success
        if self._consecutive_failures == 0:
            self.requests_per_second = min(
                self._original_requests_per_second,
                int(self.requests_per_second * self.recovery_factor)
            )
            self.requests_per_minute = min(
                self._original_requests_per_minute,
                int(self.requests_per_minute * self.recovery_factor)
            )

    def reset_to_original_limits(self):
        """Reset rate limits to original values."""
        self.requests_per_second = self._original_requests_per_second
        self.requests_per_minute = self._original_requests_per_minute
        self._consecutive_failures = 0
        _logger.info("Reset rate limits to original values")


# Global provider rate limiter instance
_provider_limiter = ProviderRateLimiter()


def get_provider_limiter(provider: str) -> RateLimiter:
    """Get rate limiter for a specific provider."""
    return _provider_limiter.get_limiter(provider)


def configure_provider_limits(
    provider: str,
    requests_per_second: int,
    requests_per_minute: int,
    **kwargs
):
    """Configure rate limits for a specific provider."""
    _provider_limiter.configure_provider(
        provider, requests_per_second, requests_per_minute, **kwargs
    )


def get_all_provider_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all providers."""
    return _provider_limiter.get_all_stats()
