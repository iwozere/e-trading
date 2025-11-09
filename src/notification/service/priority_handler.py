"""
Priority Message Handling System

Priority queue implementation with CRITICAL, HIGH, NORMAL, LOW levels.
Ensures immediate processing for high-priority messages with SLA guarantees.
"""

import asyncio
import heapq
import time
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import threading

from src.data.db.models.model_notification import MessagePriority
from src.notification.service.message_queue import QueuedMessage
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class PriorityLevel(Enum):
    """Priority levels with numeric values for ordering."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


@dataclass
class PriorityQueueItem:
    """Item in priority queue with ordering support."""

    priority: int
    timestamp: float
    message: QueuedMessage
    sla_deadline: Optional[float] = None

    def __lt__(self, other):
        """Compare items for priority queue ordering."""
        # Lower priority number = higher priority
        if self.priority != other.priority:
            return self.priority < other.priority

        # Same priority: earlier timestamp first
        return self.timestamp < other.timestamp

    def __eq__(self, other):
        """Check equality."""
        return (self.priority == other.priority and
                self.timestamp == other.timestamp and
                self.message.id == other.message.id)

    @property
    def is_sla_violated(self) -> bool:
        """Check if SLA deadline has been violated."""
        if self.sla_deadline is None:
            return False
        return time.time() > self.sla_deadline

    @property
    def sla_remaining_seconds(self) -> Optional[float]:
        """Get remaining seconds until SLA violation."""
        if self.sla_deadline is None:
            return None
        return max(0, self.sla_deadline - time.time())


@dataclass
class PriorityStats:
    """Statistics for priority handling."""

    total_processed: int = 0
    critical_processed: int = 0
    high_processed: int = 0
    normal_processed: int = 0
    low_processed: int = 0

    sla_violations: int = 0
    avg_processing_time_ms: float = 0.0

    critical_avg_time_ms: float = 0.0
    high_avg_time_ms: float = 0.0
    normal_avg_time_ms: float = 0.0
    low_avg_time_ms: float = 0.0

    def add_processing_time(self, priority: MessagePriority, processing_time_ms: float):
        """Add processing time for statistics."""
        self.total_processed += 1

        # Update total average
        if self.total_processed == 1:
            self.avg_processing_time_ms = processing_time_ms
        else:
            self.avg_processing_time_ms = (
                (self.avg_processing_time_ms * (self.total_processed - 1) + processing_time_ms) /
                self.total_processed
            )

        # Update priority-specific stats
        if priority == MessagePriority.CRITICAL:
            self.critical_processed += 1
            if self.critical_processed == 1:
                self.critical_avg_time_ms = processing_time_ms
            else:
                self.critical_avg_time_ms = (
                    (self.critical_avg_time_ms * (self.critical_processed - 1) + processing_time_ms) /
                    self.critical_processed
                )
        elif priority == MessagePriority.HIGH:
            self.high_processed += 1
            if self.high_processed == 1:
                self.high_avg_time_ms = processing_time_ms
            else:
                self.high_avg_time_ms = (
                    (self.high_avg_time_ms * (self.high_processed - 1) + processing_time_ms) /
                    self.high_processed
                )
        elif priority == MessagePriority.NORMAL:
            self.normal_processed += 1
            if self.normal_processed == 1:
                self.normal_avg_time_ms = processing_time_ms
            else:
                self.normal_avg_time_ms = (
                    (self.normal_avg_time_ms * (self.normal_processed - 1) + processing_time_ms) /
                    self.normal_processed
                )
        elif priority == MessagePriority.LOW:
            self.low_processed += 1
            if self.low_processed == 1:
                self.low_avg_time_ms = processing_time_ms
            else:
                self.low_avg_time_ms = (
                    (self.low_avg_time_ms * (self.low_processed - 1) + processing_time_ms) /
                    self.low_processed
                )


class PriorityMessageHandler:
    """
    Priority message handling system with SLA guarantees.

    Features:
    - Priority queue with CRITICAL, HIGH, NORMAL, LOW levels
    - Immediate processing for high-priority messages
    - 5-second SLA for CRITICAL messages
    - Rate limit and batching bypass for critical messages
    - SLA violation tracking and alerting
    - Priority-based processing statistics
    """

    def __init__(self):
        """Initialize priority handler."""
        self._priority_queue: List[PriorityQueueItem] = []
        self._processing_queue: asyncio.Queue = asyncio.Queue()
        self._lock = threading.RLock()
        self._stats = PriorityStats()
        self._logger = setup_logger(f"{__name__}.PriorityMessageHandler")

        # SLA configuration (in seconds)
        self._sla_config = {
            MessagePriority.CRITICAL: 5.0,   # 5 seconds for critical
            MessagePriority.HIGH: 30.0,      # 30 seconds for high
            MessagePriority.NORMAL: 300.0,   # 5 minutes for normal
            MessagePriority.LOW: 1800.0      # 30 minutes for low
        }

        # Priority mapping
        self._priority_mapping = {
            MessagePriority.CRITICAL: PriorityLevel.CRITICAL.value,
            MessagePriority.HIGH: PriorityLevel.HIGH.value,
            MessagePriority.NORMAL: PriorityLevel.NORMAL.value,
            MessagePriority.LOW: PriorityLevel.LOW.value
        }

        # Processing state
        self._processing_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
        self._running = False

    async def enqueue_message(self, message: QueuedMessage) -> bool:
        """
        Enqueue a message with priority handling.

        Args:
            message: Message to enqueue

        Returns:
            True if message was enqueued successfully
        """
        try:
            current_time = time.time()
            priority_value = self._priority_mapping.get(message.priority, PriorityLevel.NORMAL.value)

            # Calculate SLA deadline
            sla_seconds = self._sla_config.get(message.priority)
            sla_deadline = current_time + sla_seconds if sla_seconds else None

            # Create priority queue item
            queue_item = PriorityQueueItem(
                priority=priority_value,
                timestamp=current_time,
                message=message,
                sla_deadline=sla_deadline
            )

            with self._lock:
                heapq.heappush(self._priority_queue, queue_item)

            # For critical messages, trigger immediate processing
            if message.priority == MessagePriority.CRITICAL:
                await self._trigger_immediate_processing()

            self._logger.debug(
                "Enqueued message %s with priority %s (SLA: %.1fs)",
                message.id, message.priority.value, sla_seconds or 0
            )

            return True

        except Exception as e:
            self._logger.error("Failed to enqueue message %s: %s", message.id, e)
            return False

    async def dequeue_next_message(self, timeout: Optional[float] = None) -> Optional[QueuedMessage]:
        """
        Dequeue the next highest priority message.

        Args:
            timeout: Maximum time to wait for a message

        Returns:
            Next message to process, or None if timeout
        """
        start_time = time.time()

        while True:
            with self._lock:
                if self._priority_queue:
                    # Get highest priority message
                    queue_item = heapq.heappop(self._priority_queue)

                    # Check for SLA violation
                    if queue_item.is_sla_violated:
                        self._stats.sla_violations += 1
                        self._logger.warning(
                            "SLA violation for message %s (priority: %s, overdue: %.1fs)",
                            queue_item.message.id,
                            queue_item.message.priority.value,
                            time.time() - queue_item.sla_deadline
                        )

                    return queue_item.message

            # No messages available, check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return None

            # Wait a bit before checking again
            await asyncio.sleep(0.1)

    async def dequeue_by_priority(
        self,
        priority: MessagePriority,
        max_messages: int = 10
    ) -> List[QueuedMessage]:
        """
        Dequeue messages of specific priority.

        Args:
            priority: Priority level to dequeue
            max_messages: Maximum number of messages to dequeue

        Returns:
            List of messages with specified priority
        """
        target_priority = self._priority_mapping.get(priority, PriorityLevel.NORMAL.value)
        messages = []

        with self._lock:
            # Extract messages with matching priority
            remaining_items = []

            while self._priority_queue and len(messages) < max_messages:
                item = heapq.heappop(self._priority_queue)

                if item.priority == target_priority:
                    # Check for SLA violation
                    if item.is_sla_violated:
                        self._stats.sla_violations += 1
                        self._logger.warning(
                            "SLA violation for message %s (priority: %s)",
                            item.message.id, item.message.priority.value
                        )

                    messages.append(item.message)
                else:
                    remaining_items.append(item)

            # Put back non-matching items
            for item in remaining_items:
                heapq.heappush(self._priority_queue, item)

        if messages:
            self._logger.debug(
                "Dequeued %d messages with priority %s",
                len(messages), priority.value
            )

        return messages

    async def get_critical_messages(self) -> List[QueuedMessage]:
        """
        Get all critical messages immediately (bypass normal queue).

        Returns:
            List of critical messages
        """
        return await self.dequeue_by_priority(MessagePriority.CRITICAL, max_messages=100)

    async def get_high_priority_messages(self, max_messages: int = 20) -> List[QueuedMessage]:
        """
        Get high priority messages.

        Args:
            max_messages: Maximum number of messages to return

        Returns:
            List of high priority messages
        """
        return await self.dequeue_by_priority(MessagePriority.HIGH, max_messages)

    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get current queue status and statistics.

        Returns:
            Dictionary with queue status information
        """
        with self._lock:
            # Count messages by priority
            priority_counts = {
                "CRITICAL": 0,
                "HIGH": 0,
                "NORMAL": 0,
                "LOW": 0
            }

            sla_violations_pending = 0

            for item in self._priority_queue:
                if item.priority == PriorityLevel.CRITICAL.value:
                    priority_counts["CRITICAL"] += 1
                elif item.priority == PriorityLevel.HIGH.value:
                    priority_counts["HIGH"] += 1
                elif item.priority == PriorityLevel.NORMAL.value:
                    priority_counts["NORMAL"] += 1
                elif item.priority == PriorityLevel.LOW.value:
                    priority_counts["LOW"] += 1

                if item.is_sla_violated:
                    sla_violations_pending += 1

            total_queued = len(self._priority_queue)

        return {
            "total_queued": total_queued,
            "priority_counts": priority_counts,
            "sla_violations_pending": sla_violations_pending,
            "processing_stats": {
                "total_processed": self._stats.total_processed,
                "sla_violations_total": self._stats.sla_violations,
                "avg_processing_time_ms": self._stats.avg_processing_time_ms,
                "priority_processing_times": {
                    "CRITICAL": self._stats.critical_avg_time_ms,
                    "HIGH": self._stats.high_avg_time_ms,
                    "NORMAL": self._stats.normal_avg_time_ms,
                    "LOW": self._stats.low_avg_time_ms
                }
            },
            "sla_config": {
                priority.value: seconds for priority, seconds in self._sla_config.items()
            }
        }

    def get_sla_violations(self) -> List[Dict[str, Any]]:
        """
        Get current SLA violations in queue.

        Returns:
            List of SLA violation information
        """
        violations = []

        with self._lock:
            for item in self._priority_queue:
                if item.is_sla_violated:
                    violations.append({
                        "message_id": item.message.id,
                        "priority": item.message.priority.value,
                        "enqueued_at": datetime.fromtimestamp(item.timestamp).isoformat(),
                        "sla_deadline": datetime.fromtimestamp(item.sla_deadline).isoformat(),
                        "overdue_seconds": time.time() - item.sla_deadline,
                        "message_type": item.message.message_type,
                        "channels": item.message.channels
                    })

        return violations

    async def _trigger_immediate_processing(self):
        """Trigger immediate processing for critical messages."""
        # This would integrate with the message processor to interrupt normal processing
        # and handle critical messages immediately
        self._logger.info("Triggering immediate processing for critical message")

        # In a real implementation, this would:
        # 1. Signal the message processor to check for critical messages
        # 2. Potentially interrupt current batch processing
        # 3. Ensure critical messages are processed within SLA

    def record_processing_time(self, message: QueuedMessage, processing_time_ms: float):
        """
        Record processing time for statistics.

        Args:
            message: Processed message
            processing_time_ms: Processing time in milliseconds
        """
        self._stats.add_processing_time(message.priority, processing_time_ms)

        # Check if processing time exceeded SLA
        sla_seconds = self._sla_config.get(message.priority)
        if sla_seconds and processing_time_ms > (sla_seconds * 1000):
            self._logger.warning(
                "Processing time SLA violation for message %s: %.1fms > %.1fms",
                message.id, processing_time_ms, sla_seconds * 1000
            )

    def set_sla_config(self, priority: MessagePriority, sla_seconds: float):
        """
        Set SLA configuration for a priority level.

        Args:
            priority: Message priority
            sla_seconds: SLA time in seconds
        """
        self._sla_config[priority] = sla_seconds
        self._logger.info("Updated SLA for %s priority: %.1fs", priority.value, sla_seconds)

    def clear_queue(self) -> int:
        """
        Clear all messages from queue (admin function).

        Returns:
            Number of messages cleared
        """
        with self._lock:
            count = len(self._priority_queue)
            self._priority_queue.clear()

        self._logger.warning("Cleared priority queue: %d messages removed", count)
        return count

    def get_next_sla_deadline(self) -> Optional[Tuple[datetime, MessagePriority]]:
        """
        Get the next SLA deadline in the queue.

        Returns:
            Tuple of (deadline, priority) or None if queue is empty
        """
        with self._lock:
            if not self._priority_queue:
                return None

            # Find earliest SLA deadline
            earliest_deadline = None
            earliest_priority = None

            for item in self._priority_queue:
                if item.sla_deadline is not None:
                    if earliest_deadline is None or item.sla_deadline < earliest_deadline:
                        earliest_deadline = item.sla_deadline
                        earliest_priority = item.message.priority

            if earliest_deadline is not None:
                return datetime.fromtimestamp(earliest_deadline), earliest_priority

            return None

    async def start_monitoring(self):
        """Start SLA monitoring task."""
        self._running = True
        self._shutdown_event.clear()

        # Start SLA monitoring task
        monitor_task = asyncio.create_task(self._sla_monitor_loop())
        self._processing_tasks.add(monitor_task)

        self._logger.info("Priority message handler monitoring started")

    async def stop_monitoring(self):
        """Stop SLA monitoring task."""
        self._running = False
        self._shutdown_event.set()

        # Cancel all monitoring tasks
        for task in self._processing_tasks:
            task.cancel()

        # Wait for tasks to complete
        if self._processing_tasks:
            await asyncio.gather(*self._processing_tasks, return_exceptions=True)

        self._processing_tasks.clear()
        self._logger.info("Priority message handler monitoring stopped")

    async def _sla_monitor_loop(self):
        """Monitor SLA violations and alert."""
        while self._running:
            try:
                # Check for SLA violations
                violations = self.get_sla_violations()

                if violations:
                    self._logger.warning("Found %d SLA violations in queue", len(violations))

                    # Alert for critical violations
                    critical_violations = [v for v in violations if v["priority"] == "CRITICAL"]
                    if critical_violations:
                        self._logger.error(
                            "CRITICAL SLA violations detected: %d messages overdue",
                            len(critical_violations)
                        )

                # Wait before next check
                await asyncio.sleep(10)  # Check every 10 seconds

            except asyncio.CancelledError:
                break
            except Exception:
                self._logger.exception("Error in SLA monitor loop:")
                await asyncio.sleep(10)


# Global priority handler instance
priority_handler = PriorityMessageHandler()