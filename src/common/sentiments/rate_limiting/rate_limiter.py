"""
Basic rate limiting implementation with token bucket algorithm.

Provides fundamental rate limiting capabilities using token bucket algorithm
with configurable burst capacity and refill rates.
"""

import asyncio
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
import threading

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiter."""
    requests_per_second: float = 1.0
    burst_capacity: int = 5
    window_size_seconds: int = 60
    enable_burst: bool = True
    strict_mode: bool = False  # If True, reject requests when limit exceeded

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'RateLimitConfig':
        """Create configuration from dictionary."""
        return cls(
            requests_per_second=float(config.get('requests_per_second', 1.0)),
            burst_capacity=int(config.get('burst_capacity', 5)),
            window_size_seconds=int(config.get('window_size_seconds', 60)),
            enable_burst=config.get('enable_burst', True),
            strict_mode=config.get('strict_mode', False)
        )


class TokenBucket:
    """
    Token bucket implementation for rate limiting.

    The token bucket algorithm allows for burst traffic while maintaining
    an average rate limit over time.
    """

    def __init__(self, rate: float, capacity: int):
        """
        Initialize token bucket.

        Args:
            rate: Token refill rate (tokens per second)
            capacity: Maximum number of tokens in bucket
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = float(capacity)
        self.last_refill = time.time()
        self._lock = threading.Lock()

    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from the bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed, False if not enough tokens available
        """
        with self._lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            return False

    def wait_time(self, tokens: int = 1) -> float:
        """
        Calculate wait time until enough tokens are available.

        Args:
            tokens: Number of tokens needed

        Returns:
            Wait time in seconds
        """
        with self._lock:
            self._refill()

            if self.tokens >= tokens:
                return 0.0

            tokens_needed = tokens - self.tokens
            return tokens_needed / self.rate

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill

        if elapsed > 0:
            tokens_to_add = elapsed * self.rate
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
            self.last_refill = now

    def get_status(self) -> Dict[str, Any]:
        """Get current bucket status."""
        with self._lock:
            self._refill()
            return {
                'tokens_available': self.tokens,
                'capacity': self.capacity,
                'rate': self.rate,
                'utilization': 1.0 - (self.tokens / self.capacity)
            }


class RateLimiter:
    """
    Rate limiter with token bucket algorithm and monitoring.

    Provides rate limiting with configurable parameters, monitoring,
    and both blocking and non-blocking operation modes.
    """

    def __init__(self, config: RateLimitConfig, name: str = "default"):
        """
        Initialize rate limiter.

        Args:
            config: Rate limiting configuration
            name: Name for this rate limiter instance
        """
        self.config = config
        self.name = name
        self.bucket = TokenBucket(config.requests_per_second, config.burst_capacity)

        # Statistics
        self.total_requests = 0
        self.allowed_requests = 0
        self.rejected_requests = 0
        self.total_wait_time = 0.0
        self.created_at = datetime.now(timezone.utc)
        self._lock = threading.Lock()

    async def acquire(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """
        Acquire tokens from the rate limiter.

        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait (None for no timeout)

        Returns:
            True if tokens were acquired, False if timeout or strict mode rejection
        """
        with self._lock:
            self.total_requests += 1

        # Try immediate acquisition
        if self.bucket.consume(tokens):
            with self._lock:
                self.allowed_requests += 1
            return True

        # Handle strict mode
        if self.config.strict_mode:
            with self._lock:
                self.rejected_requests += 1
            _logger.debug("Rate limit exceeded for %s (strict mode)", self.name)
            return False

        # Calculate wait time
        wait_time = self.bucket.wait_time(tokens)

        # Check timeout
        if timeout is not None and wait_time > timeout:
            with self._lock:
                self.rejected_requests += 1
            _logger.debug("Rate limit timeout for %s (%.2fs > %.2fs)",
                         self.name, wait_time, timeout)
            return False

        # Wait for tokens
        if wait_time > 0:
            _logger.debug("Rate limiting %s: waiting %.2fs", self.name, wait_time)
            await asyncio.sleep(wait_time)

            with self._lock:
                self.total_wait_time += wait_time

        # Try acquisition again
        if self.bucket.consume(tokens):
            with self._lock:
                self.allowed_requests += 1
            return True
        else:
            with self._lock:
                self.rejected_requests += 1
            return False

    def try_acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens without waiting.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens were acquired immediately
        """
        with self._lock:
            self.total_requests += 1

        if self.bucket.consume(tokens):
            with self._lock:
                self.allowed_requests += 1
            return True
        else:
            with self._lock:
                self.rejected_requests += 1
            return False

    def get_wait_time(self, tokens: int = 1) -> float:
        """
        Get estimated wait time for tokens.

        Args:
            tokens: Number of tokens needed

        Returns:
            Estimated wait time in seconds
        """
        return self.bucket.wait_time(tokens)

    def get_statistics(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        with self._lock:
            uptime = (datetime.now(timezone.utc) - self.created_at).total_seconds()

            stats = {
                'name': self.name,
                'config': {
                    'requests_per_second': self.config.requests_per_second,
                    'burst_capacity': self.config.burst_capacity,
                    'strict_mode': self.config.strict_mode
                },
                'statistics': {
                    'total_requests': self.total_requests,
                    'allowed_requests': self.allowed_requests,
                    'rejected_requests': self.rejected_requests,
                    'success_rate': self.allowed_requests / max(1, self.total_requests),
                    'rejection_rate': self.rejected_requests / max(1, self.total_requests),
                    'total_wait_time_seconds': self.total_wait_time,
                    'avg_wait_time_seconds': self.total_wait_time / max(1, self.allowed_requests),
                    'uptime_seconds': uptime,
                    'requests_per_second_actual': self.total_requests / max(1, uptime)
                },
                'bucket_status': self.bucket.get_status()
            }

        return stats

    def reset_statistics(self) -> None:
        """Reset all statistics."""
        with self._lock:
            self.total_requests = 0
            self.allowed_requests = 0
            self.rejected_requests = 0
            self.total_wait_time = 0.0
            self.created_at = datetime.now(timezone.utc)

    def update_config(self, config: RateLimitConfig) -> None:
        """
        Update rate limiter configuration.

        Args:
            config: New configuration
        """
        self.config = config
        self.bucket = TokenBucket(config.requests_per_second, config.burst_capacity)
        _logger.info("Updated rate limiter config for %s: %.2f req/s, burst %d",
                    self.name, config.requests_per_second, config.burst_capacity)

    def is_available(self) -> bool:
        """Check if tokens are immediately available."""
        return self.bucket.tokens >= 1

    def get_utilization(self) -> float:
        """Get current utilization (0.0 = unused, 1.0 = fully utilized)."""
        bucket_status = self.bucket.get_status()
        return bucket_status['utilization']


class RateLimiterPool:
    """
    Pool of rate limiters for different services/adapters.

    Manages multiple rate limiters with different configurations
    and provides unified access and monitoring.
    """

    def __init__(self):
        """Initialize rate limiter pool."""
        self._limiters: Dict[str, RateLimiter] = {}
        self._lock = threading.Lock()

    def add_limiter(self, name: str, config: RateLimitConfig) -> RateLimiter:
        """
        Add a rate limiter to the pool.

        Args:
            name: Unique name for the rate limiter
            config: Rate limiting configuration

        Returns:
            Created rate limiter instance
        """
        with self._lock:
            limiter = RateLimiter(config, name)
            self._limiters[name] = limiter
            _logger.info("Added rate limiter: %s (%.2f req/s)",
                        name, config.requests_per_second)
            return limiter

    def get_limiter(self, name: str) -> Optional[RateLimiter]:
        """
        Get a rate limiter by name.

        Args:
            name: Name of the rate limiter

        Returns:
            Rate limiter instance or None if not found
        """
        return self._limiters.get(name)

    def remove_limiter(self, name: str) -> bool:
        """
        Remove a rate limiter from the pool.

        Args:
            name: Name of the rate limiter to remove

        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if name in self._limiters:
                del self._limiters[name]
                _logger.info("Removed rate limiter: %s", name)
                return True
            return False

    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all rate limiters."""
        stats = {}
        for name, limiter in self._limiters.items():
            stats[name] = limiter.get_statistics()
        return stats

    def get_pool_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the entire pool."""
        with self._lock:
            total_requests = sum(l.total_requests for l in self._limiters.values())
            total_allowed = sum(l.allowed_requests for l in self._limiters.values())
            total_rejected = sum(l.rejected_requests for l in self._limiters.values())

            return {
                'limiter_count': len(self._limiters),
                'total_requests': total_requests,
                'total_allowed': total_allowed,
                'total_rejected': total_rejected,
                'overall_success_rate': total_allowed / max(1, total_requests),
                'limiter_names': list(self._limiters.keys())
            }

    def list_limiters(self) -> List[str]:
        """Get list of all rate limiter names."""
        return list(self._limiters.keys())


# Global rate limiter pool
_global_pool: Optional[RateLimiterPool] = None


def get_rate_limiter_pool() -> RateLimiterPool:
    """Get the global rate limiter pool."""
    global _global_pool
    if _global_pool is None:
        _global_pool = RateLimiterPool()
    return _global_pool