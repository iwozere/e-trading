"""
Priority request queue for urgent sentiment analysis requests.

Provides priority-based request queuing to ensure urgent requests
are processed before normal requests while maintaining fairness.
"""

import asyncio
import heapq
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import IntEnum
from pathlib import Path
import sys
import uuid

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class RequestPriority(IntEnum):
    """Priority levels for requests (lower values = higher priority)."""
    CRITICAL = 1    # Critical/urgent requests
    HIGH = 2        # High priority requests
    NORMAL = 3      # Normal priority requests
    LOW = 4         # Low priority/background requests


@dataclass
class QueuedRequest:
    """Represents a queued request with priority and metadata."""
    priority: RequestPriority
    request_id: str
    adapter_name: str
    ticker: str
    created_at: datetime
    timeout_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: 'QueuedRequest') -> bool:
        """Compare requests for priority queue ordering."""
        # First by priority (lower = higher priority)
        if self.priority != other.priority:
            return self.priority < other.priority
        # Then by creation time (older = higher priority)
        return self.created_at < other.created_at

    def is_expired(self) -> bool:
        """Check if request has expired."""
        if self.timeout_at is None:
            return False
        return datetime.now(timezone.utc) > self.timeout_at


class PriorityRequestQueue:
    """
    Priority-based request queue with timeout and fairness controls.

    Manages requests with different priority levels, ensuring urgent requests
    are processed first while preventing starvation of lower priority requests.
    """

    def __init__(self,
                 max_queue_size: int = 1000,
                 default_timeout_seconds: int = 300,
                 fairness_window_size: int = 10):
        """
        Initialize priority request queue.

        Args:
            max_queue_size: Maximum number of queued requests
            default_timeout_seconds: Default timeout for requests
            fairness_window_size: Number of requests to consider for fairness
        """
        self.max_queue_size = max_queue_size
        self.default_timeout_seconds = default_timeout_seconds
        self.fairness_window_size = fairness_window_size

        # Priority queue (min-heap)
        self._queue: List[QueuedRequest] = []
        self._queue_lock = asyncio.Lock()

        # Request tracking
        self._pending_requests: Dict[str, QueuedRequest] = {}
        self._processed_count_by_priority: Dict[RequestPriority, int] = {
            priority: 0 for priority in RequestPriority
        }

        # Fairness tracking
        self._recent_processed: List[RequestPriority] = []
        self._fairness_threshold = 0.8  # Max ratio of high priority in recent window

        # Statistics
        self.total_queued = 0
        self.total_processed = 0
        self.total_expired = 0
        self.total_rejected = 0
        self.created_at = datetime.now(timezone.utc)

    async def enqueue(self,
                     adapter_name: str,
                     ticker: str,
                     priority: RequestPriority = RequestPriority.NORMAL,
                     timeout_seconds: Optional[int] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Enqueue a request with specified priority.

        Args:
            adapter_name: Name of the requesting adapter
            ticker: Stock ticker symbol
            priority: Request priority level
            timeout_seconds: Custom timeout (uses default if None)
            metadata: Additional request metadata

        Returns:
            Request ID if queued successfully, None if rejected
        """
        async with self._queue_lock:
            # Check queue capacity
            if len(self._queue) >= self.max_queue_size:
                self.total_rejected += 1
                _logger.warning("Request queue full, rejecting request for %s:%s",
                               adapter_name, ticker)
                return None

            # Create request
            request_id = str(uuid.uuid4())
            timeout = timeout_seconds or self.default_timeout_seconds
            timeout_at = datetime.now(timezone.utc) + timedelta(seconds=timeout)

            request = QueuedRequest(
                priority=priority,
                request_id=request_id,
                adapter_name=adapter_name,
                ticker=ticker,
                created_at=datetime.now(timezone.utc),
                timeout_at=timeout_at,
                metadata=metadata or {}
            )

            # Add to queue and tracking
            heapq.heappush(self._queue, request)
            self._pending_requests[request_id] = request
            self.total_queued += 1

            _logger.debug("Queued request %s for %s:%s (priority: %s, queue size: %d)",
                         request_id[:8], adapter_name, ticker, priority.name, len(self._queue))

            return request_id

    async def dequeue(self, timeout_seconds: Optional[float] = None) -> Optional[QueuedRequest]:
        """
        Dequeue the highest priority request.

        Args:
            timeout_seconds: Maximum time to wait for a request

        Returns:
            Next request to process or None if timeout/empty
        """
        start_time = time.time()

        while True:
            async with self._queue_lock:
                # Clean up expired requests
                await self._cleanup_expired_requests()

                # Check if queue is empty
                if not self._queue:
                    if timeout_seconds is None:
                        return None

                    # Check timeout
                    elapsed = time.time() - start_time
                    if elapsed >= timeout_seconds:
                        return None

                    # Wait a bit and try again
                    await asyncio.sleep(0.1)
                    continue

                # Apply fairness check
                next_request = self._queue[0]
                if self._should_apply_fairness(next_request.priority):
                    # Look for a lower priority request to process instead
                    lower_priority_request = self._find_lower_priority_request()
                    if lower_priority_request:
                        next_request = lower_priority_request
                        # Remove from queue
                        self._queue.remove(next_request)
                        heapq.heapify(self._queue)
                    else:
                        # No lower priority request, process the high priority one
                        next_request = heapq.heappop(self._queue)
                else:
                    # Normal processing
                    next_request = heapq.heappop(self._queue)

                # Remove from tracking
                if next_request.request_id in self._pending_requests:
                    del self._pending_requests[next_request.request_id]

                # Update statistics
                self.total_processed += 1
                self._processed_count_by_priority[next_request.priority] += 1
                self._recent_processed.append(next_request.priority)

                # Keep recent window manageable
                if len(self._recent_processed) > self.fairness_window_size:
                    self._recent_processed.pop(0)

                _logger.debug("Dequeued request %s for %s:%s (priority: %s, wait time: %.2fs)",
                             next_request.request_id[:8],
                             next_request.adapter_name,
                             next_request.ticker,
                             next_request.priority.name,
                             (datetime.now(timezone.utc) - next_request.created_at).total_seconds())

                return next_request

    async def _cleanup_expired_requests(self) -> None:
        """Remove expired requests from the queue."""
        expired_requests = []

        # Find expired requests
        for request in self._queue:
            if request.is_expired():
                expired_requests.append(request)

        # Remove expired requests
        for expired in expired_requests:
            try:
                self._queue.remove(expired)
                if expired.request_id in self._pending_requests:
                    del self._pending_requests[expired.request_id]
                self.total_expired += 1

                _logger.debug("Expired request %s for %s:%s",
                             expired.request_id[:8],
                             expired.adapter_name,
                             expired.ticker)
            except ValueError:
                # Request already removed
                pass

        # Rebuild heap if we removed items
        if expired_requests:
            heapq.heapify(self._queue)

    def _should_apply_fairness(self, priority: RequestPriority) -> bool:
        """Check if fairness should be applied to prevent starvation."""
        if priority in [RequestPriority.CRITICAL, RequestPriority.HIGH]:
            if len(self._recent_processed) < self.fairness_window_size:
                return False

            # Count high priority requests in recent window
            high_priority_count = sum(
                1 for p in self._recent_processed
                if p in [RequestPriority.CRITICAL, RequestPriority.HIGH]
            )

            high_priority_ratio = high_priority_count / len(self._recent_processed)
            return high_priority_ratio > self._fairness_threshold

        return False

    def _find_lower_priority_request(self) -> Optional[QueuedRequest]:
        """Find a lower priority request for fairness processing."""
        for request in self._queue:
            if request.priority in [RequestPriority.NORMAL, RequestPriority.LOW]:
                return request
        return None

    async def cancel_request(self, request_id: str) -> bool:
        """
        Cancel a pending request.

        Args:
            request_id: ID of the request to cancel

        Returns:
            True if request was cancelled
        """
        async with self._queue_lock:
            if request_id not in self._pending_requests:
                return False

            request = self._pending_requests[request_id]

            try:
                self._queue.remove(request)
                heapq.heapify(self._queue)
                del self._pending_requests[request_id]

                _logger.debug("Cancelled request %s for %s:%s",
                             request_id[:8], request.adapter_name, request.ticker)
                return True
            except ValueError:
                # Request not in queue (maybe already processed)
                if request_id in self._pending_requests:
                    del self._pending_requests[request_id]
                return False

    async def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status and statistics."""
        async with self._queue_lock:
            # Count by priority
            priority_counts = {priority.name: 0 for priority in RequestPriority}
            for request in self._queue:
                priority_counts[request.priority.name] += 1

            # Calculate average wait time
            if self._queue:
                now = datetime.now(timezone.utc)
                wait_times = [(now - req.created_at).total_seconds() for req in self._queue]
                avg_wait_time = sum(wait_times) / len(wait_times)
                max_wait_time = max(wait_times)
            else:
                avg_wait_time = 0.0
                max_wait_time = 0.0

            uptime = (datetime.now(timezone.utc) - self.created_at).total_seconds()

            return {
                'queue_size': len(self._queue),
                'max_queue_size': self.max_queue_size,
                'utilization': len(self._queue) / self.max_queue_size,
                'priority_distribution': priority_counts,
                'statistics': {
                    'total_queued': self.total_queued,
                    'total_processed': self.total_processed,
                    'total_expired': self.total_expired,
                    'total_rejected': self.total_rejected,
                    'processing_rate': self.total_processed / max(1, uptime),
                    'expiration_rate': self.total_expired / max(1, self.total_queued),
                    'rejection_rate': self.total_rejected / max(1, self.total_queued + self.total_rejected)
                },
                'performance': {
                    'avg_wait_time_seconds': avg_wait_time,
                    'max_wait_time_seconds': max_wait_time,
                    'fairness_ratio': self._calculate_fairness_ratio()
                }
            }

    def _calculate_fairness_ratio(self) -> float:
        """Calculate fairness ratio (lower = more fair)."""
        if not self._recent_processed:
            return 0.0

        priority_counts = {priority: 0 for priority in RequestPriority}
        for priority in self._recent_processed:
            priority_counts[priority] += 1

        total = len(self._recent_processed)
        high_priority_ratio = (
            priority_counts[RequestPriority.CRITICAL] +
            priority_counts[RequestPriority.HIGH]
        ) / total

        return high_priority_ratio

    async def clear_queue(self) -> int:
        """
        Clear all pending requests from the queue.

        Returns:
            Number of requests cleared
        """
        async with self._queue_lock:
            cleared_count = len(self._queue)
            self._queue.clear()
            self._pending_requests.clear()

            _logger.info("Cleared %d requests from priority queue", cleared_count)
            return cleared_count

    async def get_pending_requests(self,
                                 adapter_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of pending requests.

        Args:
            adapter_name: Filter by adapter name (optional)

        Returns:
            List of pending request information
        """
        async with self._queue_lock:
            requests = []

            for request in self._queue:
                if adapter_name and request.adapter_name != adapter_name:
                    continue

                wait_time = (datetime.now(timezone.utc) - request.created_at).total_seconds()
                time_to_expire = None
                if request.timeout_at:
                    time_to_expire = (request.timeout_at - datetime.now(timezone.utc)).total_seconds()

                requests.append({
                    'request_id': request.request_id,
                    'adapter_name': request.adapter_name,
                    'ticker': request.ticker,
                    'priority': request.priority.name,
                    'wait_time_seconds': wait_time,
                    'time_to_expire_seconds': time_to_expire,
                    'metadata': request.metadata
                })

            return requests

    def get_priority_statistics(self) -> Dict[str, Any]:
        """Get statistics broken down by priority level."""
        total_processed = sum(self._processed_count_by_priority.values())

        stats = {}
        for priority in RequestPriority:
            count = self._processed_count_by_priority[priority]
            stats[priority.name] = {
                'processed_count': count,
                'percentage': (count / max(1, total_processed)) * 100,
                'current_queued': sum(1 for req in self._queue if req.priority == priority)
            }

        return stats


# Global priority queue instance
_global_priority_queue: Optional[PriorityRequestQueue] = None


def get_priority_queue() -> PriorityRequestQueue:
    """Get the global priority request queue."""
    global _global_priority_queue
    if _global_priority_queue is None:
        _global_priority_queue = PriorityRequestQueue()
    return _global_priority_queue