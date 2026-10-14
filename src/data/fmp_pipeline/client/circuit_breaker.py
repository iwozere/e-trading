"""
Circuit breaker shared by all workers (spec §6.4): after N consecutive 429/5xx/network failures, every worker
pauses for a fixed time before sending anything else.
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
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """
        Args:
            threshold: Consecutive failures that open the breaker (spec: 20).
            pause_s: Pause length once open (spec: 60 s).
            clock: Monotonic clock (injectable for tests).
            sleep: Async sleep (injectable for tests).
        """
        self.threshold = threshold
        self.pause_s = pause_s
        self.consecutive_failures = 0
        self.times_opened = 0
        self._open_until = 0.0
        self._clock = clock
        self._sleep = sleep

    def record_failure(self) -> None:
        """Count a retryable failure; open the breaker when the threshold is reached."""
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.threshold:
            self._open_until = self._clock() + self.pause_s
            self.times_opened += 1
            self.consecutive_failures = 0
            _logger.warning("Circuit breaker open: %d consecutive failures, pausing all workers %.0fs",
                            self.threshold, self.pause_s)

    def record_success(self) -> None:
        """Any non-retryable response resets the count."""
        self.consecutive_failures = 0

    @property
    def is_open(self) -> bool:
        """Whether workers must currently wait."""
        return self._clock() < self._open_until

    async def wait_if_open(self) -> None:
        """Block until the pause is over (no-op when closed)."""
        remaining = self._open_until - self._clock()
        if remaining > 0:
            await self._sleep(remaining)
