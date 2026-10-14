"""
Shared async request pacer (spec §6.4: "do not exceed the limit even briefly; bursts count").

Grants are spaced exactly ``60 / per_minute`` seconds apart, whichever worker asks. A token bucket with any burst
capacity could exceed the per-minute limit inside a shorter window, so there is deliberately no burst: any
60-second window holds at most ``per_minute`` grants. FMP sends no rate-limit headers (probe F11), so nothing
adapts this from responses; 429 handling is the client's job.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Optional


class AsyncRateLimiter:
    """Evenly spaced request grants shared by every worker of one event loop."""

    def __init__(
        self,
        per_minute: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """
        Args:
            per_minute: Maximum requests per minute (spec default 600, 80% of Premium's 750).
            clock: Monotonic clock in seconds (injectable for tests).
            sleep: Async sleep (injectable for tests).

        Raises:
            ValueError: If ``per_minute`` is not positive.
        """
        if per_minute <= 0:
            raise ValueError("per_minute must be positive")
        self.interval = 60.0 / per_minute
        self._clock = clock
        self._sleep = sleep
        self._next = 0.0
        self._lock: Optional[asyncio.Lock] = None

    async def acquire(self) -> None:
        """Wait until this caller's slot; slots are handed out in call order."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            now = self._clock()
            slot = max(now, self._next)
            self._next = slot + self.interval
        if slot > now:
            await self._sleep(slot - now)
