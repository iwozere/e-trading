"""
Priority Message Queue System

Priority-based message queue with immediate processing for high-priority messages.
Ensures 5-second delivery SLA for critical messages.
"""

import asyncio
import heapq
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading

from src.notification.service.message_queue import QueuedMessage, MessagePriority
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class PriorityLevel(Enum):
    """Message priority levels with numeric values for ordering."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


@dataclass
class PriorityMessage:
    """Message with priority information for queue processing."""

    priority: PriorityLevel
    timestamp: float  # When message was enqueued
    message: QueuedMessage
    deadline: Optional[float] = None  # SLA deadline timestamp

    def __post_init__(self):
        """Set deadline for critical messages."""
        if self.priority == PriorityLevel.CRITICAL and self.deadline is None:
            # 5-second SLA for critical messages
            self.deadline = self.timestamp + 5.0

    def __lt__(self, other):
        """Compare messages for priority queue ordering."""
        if not isinstance(other, PriorityMessage):
            return NotImplemented

        # First compare by priority level
        if self.priority != other.priority:
            return self.priority.value < other.priority.value

        # For same priority, compare by timestamp (FIFO)
        return self.timestamp < other.timestamp

    @property
    def is_expired(self) -> bool:
        """Check if message has exceeded its deadline."""
        if self.deadline is None:
            return False
        return time.time() > self.deadline

    @property
    def time_to_deadline(self) -> Optional[float]:
        """Get seconds until deadline (None if no deadline)."""
        if self.deadline is None:
            return None
        return max(0.0, self.deadline - time.time())

    @classmethod
    def from_queued_message(cls, message: QueuedMessage) -> 'PriorityMessage':
        """Create PriorityMessage from QueuedMessage."""
        # Map MessagePriority to PriorityLevel
        priority_mapping = {
            MessagePriority.CRITICAL: PriorityLevel.CRITICAL,
            MessagePriority.HIGH: PriorityLevel.HIGH,
            MessagePriority.NORMAL: PriorityLevel.NORMAL,
            MessagePriority.LOW: PriorityLevel.LOW
        }

        priority_level = priority_mapping.get(message.priority, PriorityLevel.NORMAL)

        return cls(
            priority=priority_level,
            timestamp=time.time(),
            message=message
        )


class PriorityQueue:
    """
    Thread-safe priority queue for messages.

    Supports:
    - Priority-based ordering (CRITICAL > HIGH > NORMAL > LOW)
    - FIFO within same priority level
    - Deadline tracking for critical messages
    - Immediate processing bypass for high-priority messages
    """

    def __init__(self):
        """Initialize priority queue."""
        self._heap: List[PriorityMessage] = []
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._logger = setup_logger(f"{__name__}.PriorityQueue")

        # Statistics
        self._stats = {
            "enqueued": 0,
            "dequeued": 0,
            "expired": 0,
            "by_priority": {level.name: 0 for level in PriorityLevel}
        }

    def enqueue(self, message: PriorityMessage):
        """
        Add message to priority queue.

        Args:
            message: PriorityMessage to enqueue
        """
        with self._condition:
            heapq.heappush(self._heap, message)

            # Update statistics
            self._stats["enqueued"] += 1
            self._stats["by_priority"][message.priority.name] += 1

            # Notify waiting threads
            self._condition.notify_all()

            self._logger.debug(
                "Enqueued %s priority message (queue size: %d)",
                message.priority.name, len(self._heap)
            )

    def dequeue(self, timeout: Optional[float] = None) -> Optional[PriorityMessage]:
        """
        Remove and return highest priority message.

        Args:
            timeout: Maximum time to wait for message (None = no timeout)

        Returns:
            PriorityMessage or None if timeout/empty
        """
        with self._condition:
            # Wait for messages if queue is empty
            if not self._heap and timeout is not None:
                self._condition.wait(timeout)

            if not self._heap:
                return None

            message = heapq.heappop(self._heap)

            # Update statistics
            self._stats["dequeued"] += 1

            # Check if message expired
            if message.is_expired:
                self._stats["expired"] += 1
                self._logger.warning(
                    "Dequeued expired %s priority message (deadline missed by %.2fs)",
                    message.priority.name,
                    time.time() - message.deadline if message.deadline else 0
                )

            self._logger.debug(
                "Dequeued %s priority message (queue size: %d)",
                message.priority.name, len(self._heap)
            )

            return message

    def dequeue_batch(self, max_size: int, max_wait: float = 1.0) -> List[PriorityMessage]:
        """
        Dequeue multiple messages for batch processing.

        Args:
            max_size: Maximum number of messages to dequeue
            max_wait: Maximum time to wait for messages

        Returns:
            List of PriorityMessage objects
        """
        messages = []
        start_time = time.time()

        while len(messages) < max_size and (time.time() - start_time) < max_wait:
            remaining_time = max_wait - (time.time() - start_time)
            message = self.dequeue(timeout=remaining_time)

            if message is None:
                break

            messages.append(message)

            # Don't batch critical messages with others
            if message.priority == PriorityLevel.CRITICAL:
                break

        if messages:
            self._logger.debug("Dequeued batch of %d messages", len(messages))

        return messages

    def peek(self) -> Optional[PriorityMessage]:
        """
        Look at highest priority message without removing it.

        Returns:
            PriorityMessage or None if empty
        """
        with self._lock:
            if not self._heap:
                return None
            return self._heap[0]

    def size(self) -> int:
        """Get current queue size."""
        with self._lock:
            return len(self._heap)

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        with self._lock:
            return len(self._heap) == 0

    def clear(self):
        """Clear all messages from queue."""
        with self._condition:
            self._heap.clear()
            self._condition.notify_all()

        self._logger.info("Priority queue cleared")

    def get_statistics(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self._lock:
            stats = self._stats.copy()
            stats["current_size"] = len(self._heap)

            # Count messages by priority in current queue
            current_by_priority = {level.name: 0 for level in PriorityLevel}
            for message in self._heap:
                current_by_priority[message.priority.name] += 1

            stats["current_by_priority"] = current_by_priority

            return stats

    def get_expired_messages(self) -> List[PriorityMessage]:
        """
        Get all expired messages from queue.

        Returns:
            List of expired messages (removed from queue)
        """
        expired_messages = []

        with self._condition:
            # Rebuild heap without expired messages
            valid_messages = []

            for message in self._heap:
                if message.is_expired:
                    expired_messages.append(message)
                    self._stats["expired"] += 1
                else:
                    valid_messages.append(message)

            # Rebuild heap
            self._heap = valid_messages
            heapq.heapify(self._heap)

            if expired_messages:
                self._logger.warning("Removed %d expired messages", len(expired_messages))

        return expired_messages


class PriorityMessageHandler:
    """
    High-level priority message handling system.

    Manages multiple priority queues and processing strategies.
    """

    def __init__(self):
        """Initialize priority message handler."""
        self._critical_queue = PriorityQueue()  # Immediate processing
        self._normal_queue = PriorityQueue()    # Batch processing
        self._logger = setup_logger(f"{__name__}.PriorityMessageHandler")

        # Processing configuration
        self.critical_sla_seconds = 5.0
        self.high_priority_bypass_batching = True
        self.batch_size_normal = 10
        self.batch_timeout_normal = 30.0

        # Statistics
        self._stats = {
            "messages_processed": 0,
            "sla_violations": 0,
            "bypassed_batching": 0
        }

    def enqueue_message(self, message: QueuedMessage) -> PriorityMessage:
        """
        Enqueue message in appropriate priority queue.

        Args:
            message: QueuedMessage to enqueue

        Returns:
            PriorityMessage that was enqueued
        """
        priority_message = PriorityMessage.from_queued_message(message)

        # Route to appropriate queue
        if priority_message.priority in [PriorityLevel.CRITICAL, PriorityLevel.HIGH]:
            # Critical and high priority messages go to immediate processing queue
            self._critical_queue.enqueue(priority_message)

            if priority_message.priority == PriorityLevel.HIGH:
                self._stats["bypassed_batching"] += 1

            self._logger.info(
                "Enqueued %s priority message for immediate processing",
                priority_message.priority.name
            )
        else:
            # Normal and low priority messages go to batch processing queue
            self._normal_queue.enqueue(priority_message)

            self._logger.debug(
                "Enqueued %s priority message for batch processing",
                priority_message.priority.name
            )

        return priority_message

    def get_immediate_messages(self, timeout: Optional[float] = None) -> List[PriorityMessage]:
        """
        Get messages that need immediate processing.

        Args:
            timeout: Maximum time to wait for messages

        Returns:
            List of high-priority messages
        """
        messages = []

        # Get critical messages first
        while True:
            message = self._critical_queue.dequeue(timeout=0.1 if not messages else 0)
            if message is None:
                break

            messages.append(message)

            # Check SLA for critical messages
            if message.priority == PriorityLevel.CRITICAL and message.is_expired:
                self._stats["sla_violations"] += 1
                self._logger.error(
                    "Critical message SLA violation: deadline missed by %.2fs",
                    time.time() - message.deadline if message.deadline else 0
                )

        return messages

    def get_batch_messages(self, max_size: Optional[int] = None, max_wait: Optional[float] = None) -> List[PriorityMessage]:
        """
        Get messages for batch processing.

        Args:
            max_size: Maximum batch size (defaults to configured batch size)
            max_wait: Maximum wait time (defaults to configured timeout)

        Returns:
            List of messages for batch processing
        """
        batch_size = max_size or self.batch_size_normal
        batch_timeout = max_wait or self.batch_timeout_normal

        messages = self._normal_queue.dequeue_batch(batch_size, batch_timeout)

        if messages:
            self._logger.debug("Retrieved batch of %d messages for processing", len(messages))

        return messages

    def cleanup_expired_messages(self) -> int:
        """
        Remove expired messages from all queues.

        Returns:
            Number of expired messages removed
        """
        critical_expired = self._critical_queue.get_expired_messages()
        normal_expired = self._normal_queue.get_expired_messages()

        total_expired = len(critical_expired) + len(normal_expired)

        if total_expired > 0:
            self._logger.warning("Cleaned up %d expired messages", total_expired)

            # Count SLA violations for critical messages
            critical_violations = len([m for m in critical_expired if m.priority == PriorityLevel.CRITICAL])
            self._stats["sla_violations"] += critical_violations

        return total_expired

    def get_queue_sizes(self) -> Dict[str, int]:
        """Get sizes of all queues."""
        return {
            "critical_high": self._critical_queue.size(),
            "normal_low": self._normal_queue.size()
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        stats = self._stats.copy()

        # Add queue statistics
        stats["queue_stats"] = {
            "critical_high": self._critical_queue.get_statistics(),
            "normal_low": self._normal_queue.get_statistics()
        }

        # Add queue sizes
        stats["queue_sizes"] = self.get_queue_sizes()

        # Add configuration
        stats["configuration"] = {
            "critical_sla_seconds": self.critical_sla_seconds,
            "high_priority_bypass_batching": self.high_priority_bypass_batching,
            "batch_size_normal": self.batch_size_normal,
            "batch_timeout_normal": self.batch_timeout_normal
        }

        return stats

    def configure(
        self,
        critical_sla_seconds: Optional[float] = None,
        high_priority_bypass_batching: Optional[bool] = None,
        batch_size_normal: Optional[int] = None,
        batch_timeout_normal: Optional[float] = None
    ):
        """
        Configure priority handling parameters.

        Args:
            critical_sla_seconds: SLA for critical messages
            high_priority_bypass_batching: Whether high priority bypasses batching
            batch_size_normal: Batch size for normal priority messages
            batch_timeout_normal: Batch timeout for normal priority messages
        """
        if critical_sla_seconds is not None:
            self.critical_sla_seconds = critical_sla_seconds

        if high_priority_bypass_batching is not None:
            self.high_priority_bypass_batching = high_priority_bypass_batching

        if batch_size_normal is not None:
            self.batch_size_normal = batch_size_normal

        if batch_timeout_normal is not None:
            self.batch_timeout_normal = batch_timeout_normal

        self._logger.info("Priority handler configuration updated")


# Global priority message handler
priority_handler = PriorityMessageHandler()