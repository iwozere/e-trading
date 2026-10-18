"""
Circuit breaker shared by all workers (spec §6.4): after N consecutive 429/5xx/network failures, every worker
pauses for a fixed time before sending anything else.

If the breaker opens ``stop_after_opens`` times in a row on nothing but 429s, with no success in between, FMP is
refusing the account rather than pacing it (for example the bandwidth cap with a body that never says "Limit
Reach"): ``rate_limit_exhausted`` then tells the client to stop the run instead of retrying an over-limit account.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class CircuitBreaker:
    """Counts consecutive retryable failures across all workers and opens a shared pause when they pile up."""

    def __init__(
        self,
        threshold: int = 20,
        pause_s: float = 60.0,
        stop_after_opens: int = 3,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """
        Args:
            threshold: Consecutive failures that open the breaker (spec: 20).
            pause_s: Pause length once open (spec: 60 s).
            stop_after_opens: Consecutive all-429 openings, with no success between them, after which
                ``rate_limit_exhausted`` is true; 0 disables the stop.
            clock: Monotonic clock (injectable for tests).
            sleep: Async sleep (injectable for tests).
        """
        self.threshold = threshold
        self.pause_s = pause_s
        self.stop_after_opens = stop_after_opens
        self.consecutive_failures = 0
        self.times_opened = 0
        self.rate_limited_opens = 0
        self._streak_rate_limited = True
        self._open_until = 0.0
        self._clock = clock
        self._sleep = sleep

    def record_failure(self, rate_limited: bool = False) -> None:
        """
        Count a retryable failure; open the breaker when the threshold is reached.

        Args:
            rate_limited: Whether the failure was a 429 (``rate_limited``) rather than a 5xx or network error.
        """
        if self.consecutive_failures == 0:
            self._streak_rate_limited = True
        self._streak_rate_limited = self._streak_rate_limited and rate_limited
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.threshold:
            self._open_until = self._clock() + self.pause_s
            self.times_opened += 1
            self.consecutive_failures = 0
            self.rate_limited_opens = self.rate_limited_opens + 1 if self._streak_rate_limited else 0
            _logger.warning("Circuit breaker open: %d consecutive failures, pausing all workers %.0fs",
                            self.threshold, self.pause_s)

    def record_success(self) -> None:
        """Any non-retryable response resets the count and the run of all-429 openings."""
        self.consecutive_failures = 0
        self.rate_limited_opens = 0

    @property
    def rate_limit_exhausted(self) -> bool:
        """Whether the breaker opened ``stop_after_opens`` times in a row on 429s alone, so the run must stop."""
        return 0 < self.stop_after_opens <= self.rate_limited_opens

    @property
    def is_open(self) -> bool:
        """Whether workers must currently wait."""
        return self._clock() < self._open_until

    async def wait_if_open(self) -> None:
        """Block until the pause is over (no-op when closed)."""
        remaining = self._open_until - self._clock()
        if remaining > 0:
            await self._sleep(remaining)
