"""
Base adapter interface for sentiment data providers.

This module defines the abstract base class that all sentiment adapters must implement,
providing a consistent interface for sentiment data collection.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Dict, List


class AdapterStatus(Enum):
    """Adapter health status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    CIRCUIT_OPEN = "circuit_open"


@dataclass
class AdapterHealthInfo:
    """Health information for an adapter."""

    status: AdapterStatus
    last_success: datetime | None = None
    last_failure: datetime | None = None
    failure_count: int = 0
    error_message: str | None = None
    response_time_ms: float | None = None


class BaseSentimentAdapter(ABC):
    """
    Abstract base class for all sentiment data adapters.

    All sentiment adapters must implement this interface to ensure
    consistent behavior and error handling across different data sources.
    """

    def __init__(self, name: str, concurrency: int = 5, rate_limit_delay: float = 0.5):
        """
        Initialize the base adapter.

        Args:
            name: Unique name for this adapter
            concurrency: Maximum concurrent requests
            rate_limit_delay: Delay between requests in seconds
        """
        self.name = name
        self.concurrency = concurrency
        self.rate_limit_delay = rate_limit_delay
        self.semaphore = asyncio.Semaphore(concurrency)
        self._health_info = AdapterHealthInfo(status=AdapterStatus.HEALTHY)

    @abstractmethod
    async def fetch_messages(
        self, ticker: str, since_ts: int | None = None, limit: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Fetch individual messages for a ticker.

        Args:
            ticker: Stock ticker symbol
            since_ts: Unix timestamp to fetch messages since
            limit: Maximum number of messages to fetch

        Returns:
            List of message dictionaries with normalized structure
        """
        pass

    @abstractmethod
    async def fetch_summary(self, ticker: str, since_ts: int | None = None) -> Dict[str, Any]:
        """
        Fetch aggregated sentiment summary for a ticker.

        Args:
            ticker: Stock ticker symbol
            since_ts: Unix timestamp to fetch data since

        Returns:
            Dictionary containing sentiment metrics and counts
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up adapter resources."""
        pass

    async def health_check(self) -> AdapterHealthInfo:
        """
        Perform a health check on the adapter.

        Returns:
            Current health information for the adapter
        """
        return self._health_info

    def _update_health_success(self, response_time_ms: float) -> None:
        """Update health info after successful operation."""
        self._health_info.status = AdapterStatus.HEALTHY
        self._health_info.last_success = datetime.now(UTC)
        self._health_info.response_time_ms = response_time_ms
        self._health_info.failure_count = 0
        self._health_info.error_message = None

    def _update_health_failure(self, error: Exception) -> None:
        """Update health info after failed operation."""
        self._health_info.last_failure = datetime.now(UTC)
        self._health_info.failure_count += 1
        self._health_info.error_message = str(error)

        # Update status based on failure count.
        # CIRCUIT_OPEN means the circuit breaker has tripped: callers should stop
        # forwarding requests until a manual reset or timed half-open recovery is
        # implemented (see _update_health_success which resets to HEALTHY).
        if self._health_info.failure_count >= 5:
            self._health_info.status = AdapterStatus.CIRCUIT_OPEN
        elif self._health_info.failure_count >= 3:
            self._health_info.status = AdapterStatus.DEGRADED

    def is_healthy(self) -> bool:
        """Check if adapter is in a healthy state (not CIRCUIT_OPEN or FAILED)."""
        return self._health_info.status in (AdapterStatus.HEALTHY, AdapterStatus.DEGRADED)
