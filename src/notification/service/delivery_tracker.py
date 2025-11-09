"""
Delivery Status Tracking System

Comprehensive tracking of message delivery attempts, results, and performance metrics.
Supports multi-channel delivery tracking with detailed status information.
"""

import asyncio
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum
import threading
import uuid

from src.data.db.models.model_notification import MessagePriority
from src.notification.service.message_queue import QueuedMessage
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class DeliveryStatus(Enum):
    """Delivery status for individual channel attempts."""
    PENDING = "PENDING"
    SENDING = "SENDING"
    DELIVERED = "DELIVERED"
    FAILED = "FAILED"
    RATE_LIMITED = "RATE_LIMITED"
    RETRYING = "RETRYING"
    PERMANENTLY_FAILED = "PERMANENTLY_FAILED"


class DeliveryResult(Enum):
    """Overall delivery result for a message."""
    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    FAILED = "FAILED"
    PENDING = "PENDING"


@dataclass
class ChannelDeliveryAttempt:
    """Individual delivery attempt for a specific channel."""

    attempt_id: str
    message_id: int
    channel: str
    status: DeliveryStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    response_time_ms: Optional[int] = None
    external_id: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0

    def __post_init__(self):
        """Initialize attempt with unique ID if not provided."""
        if not self.attempt_id:
            self.attempt_id = str(uuid.uuid4())

    @property
    def duration_ms(self) -> Optional[int]:
        """Get attempt duration in milliseconds."""
        if self.completed_at and self.started_at:
            delta = self.completed_at - self.started_at
            return int(delta.total_seconds() * 1000)
        return None

    @property
    def is_completed(self) -> bool:
        """Check if attempt is completed (success or failure)."""
        return self.status in [
            DeliveryStatus.DELIVERED,
            DeliveryStatus.FAILED,
            DeliveryStatus.PERMANENTLY_FAILED
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "attempt_id": self.attempt_id,
            "message_id": self.message_id,
            "channel": self.channel,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "response_time_ms": self.response_time_ms,
            "external_id": self.external_id,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "duration_ms": self.duration_ms
        }


@dataclass
class MessageDeliveryStatus:
    """Complete delivery status for a message across all channels."""

    message_id: int
    message_type: str
    priority: MessagePriority
    channels: List[str]
    recipient_id: str
    created_at: datetime

    # Channel-specific delivery attempts
    channel_attempts: Dict[str, List[ChannelDeliveryAttempt]] = field(default_factory=dict)

    # Overall status
    overall_status: DeliveryResult = DeliveryResult.PENDING
    completed_at: Optional[datetime] = None

    # Callbacks for status updates
    status_callbacks: List[Callable] = field(default_factory=list)

    def add_attempt(self, attempt: ChannelDeliveryAttempt):
        """Add a delivery attempt for a channel."""
        channel = attempt.channel
        if channel not in self.channel_attempts:
            self.channel_attempts[channel] = []

        self.channel_attempts[channel].append(attempt)
        self._update_overall_status()

    def get_latest_attempt(self, channel: str) -> Optional[ChannelDeliveryAttempt]:
        """Get the latest delivery attempt for a channel."""
        attempts = self.channel_attempts.get(channel, [])
        return attempts[-1] if attempts else None

    def get_successful_channels(self) -> List[str]:
        """Get list of channels that were successfully delivered."""
        successful = []
        for channel, attempts in self.channel_attempts.items():
            if attempts and attempts[-1].status == DeliveryStatus.DELIVERED:
                successful.append(channel)
        return successful

    def get_failed_channels(self) -> List[str]:
        """Get list of channels that failed delivery."""
        failed = []
        for channel, attempts in self.channel_attempts.items():
            if attempts and attempts[-1].status in [
                DeliveryStatus.FAILED,
                DeliveryStatus.PERMANENTLY_FAILED
            ]:
                failed.append(channel)
        return failed

    def get_pending_channels(self) -> List[str]:
        """Get list of channels still pending delivery."""
        pending = []
        for channel in self.channels:
            latest = self.get_latest_attempt(channel)
            if not latest or latest.status in [
                DeliveryStatus.PENDING,
                DeliveryStatus.SENDING,
                DeliveryStatus.RATE_LIMITED,
                DeliveryStatus.RETRYING
            ]:
                pending.append(channel)
        return pending

    def _update_overall_status(self):
        """Update overall delivery status based on channel attempts."""
        successful_channels = self.get_successful_channels()
        failed_channels = self.get_failed_channels()
        pending_channels = self.get_pending_channels()

        if len(successful_channels) == len(self.channels):
            # All channels delivered successfully
            self.overall_status = DeliveryResult.SUCCESS
            if not self.completed_at:
                self.completed_at = datetime.now(timezone.utc)
        elif len(successful_channels) > 0 and len(pending_channels) == 0:
            # Some channels succeeded, some failed, none pending
            self.overall_status = DeliveryResult.PARTIAL_SUCCESS
            if not self.completed_at:
                self.completed_at = datetime.now(timezone.utc)
        elif len(pending_channels) == 0 and len(successful_channels) == 0:
            # All channels failed
            self.overall_status = DeliveryResult.FAILED
            if not self.completed_at:
                self.completed_at = datetime.now(timezone.utc)
        else:
            # Still have pending channels
            self.overall_status = DeliveryResult.PENDING

    def get_total_response_time_ms(self) -> Optional[int]:
        """Get total response time across all successful deliveries."""
        total_time = 0
        successful_count = 0

        for attempts in self.channel_attempts.values():
            for attempt in attempts:
                if (attempt.status == DeliveryStatus.DELIVERED and
                    attempt.response_time_ms is not None):
                    total_time += attempt.response_time_ms
                    successful_count += 1

        return total_time if successful_count > 0 else None

    def get_average_response_time_ms(self) -> Optional[float]:
        """Get average response time across all successful deliveries."""
        total_time = 0
        successful_count = 0

        for attempts in self.channel_attempts.values():
            for attempt in attempts:
                if (attempt.status == DeliveryStatus.DELIVERED and
                    attempt.response_time_ms is not None):
                    total_time += attempt.response_time_ms
                    successful_count += 1

        return total_time / successful_count if successful_count > 0 else None

    def add_status_callback(self, callback: Callable):
        """Add callback to be called when status changes."""
        self.status_callbacks.append(callback)

    async def notify_status_change(self):
        """Notify all registered callbacks of status change."""
        for callback in self.status_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self)
                else:
                    callback(self)
            except Exception as e:
                _logger.error("Status callback failed for message %s: %s", self.message_id, e)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type,
            "priority": self.priority.value,
            "channels": self.channels,
            "recipient_id": self.recipient_id,
            "created_at": self.created_at.isoformat(),
            "overall_status": self.overall_status.value,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "successful_channels": self.get_successful_channels(),
            "failed_channels": self.get_failed_channels(),
            "pending_channels": self.get_pending_channels(),
            "total_response_time_ms": self.get_total_response_time_ms(),
            "average_response_time_ms": self.get_average_response_time_ms(),
            "channel_attempts": {
                channel: [attempt.to_dict() for attempt in attempts]
                for channel, attempts in self.channel_attempts.items()
            }
        }


@dataclass
class DeliveryStats:
    """Delivery statistics for monitoring and analytics."""

    total_messages: int = 0
    successful_deliveries: int = 0
    failed_deliveries: int = 0
    partial_deliveries: int = 0
    pending_deliveries: int = 0

    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0

    avg_response_time_ms: float = 0.0
    min_response_time_ms: Optional[int] = None
    max_response_time_ms: Optional[int] = None

    # Per-channel statistics
    channel_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Per-priority statistics
    priority_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def add_delivery(self, delivery_status: MessageDeliveryStatus):
        """Add delivery to statistics."""
        self.total_messages += 1

        # Update overall delivery stats
        if delivery_status.overall_status == DeliveryResult.SUCCESS:
            self.successful_deliveries += 1
        elif delivery_status.overall_status == DeliveryResult.FAILED:
            self.failed_deliveries += 1
        elif delivery_status.overall_status == DeliveryResult.PARTIAL_SUCCESS:
            self.partial_deliveries += 1
        else:
            self.pending_deliveries += 1

        # Update attempt stats
        for attempts in delivery_status.channel_attempts.values():
            for attempt in attempts:
                self.total_attempts += 1

                if attempt.status == DeliveryStatus.DELIVERED:
                    self.successful_attempts += 1

                    # Update response time stats
                    if attempt.response_time_ms is not None:
                        if self.min_response_time_ms is None:
                            self.min_response_time_ms = attempt.response_time_ms
                            self.max_response_time_ms = attempt.response_time_ms
                            self.avg_response_time_ms = attempt.response_time_ms
                        else:
                            self.min_response_time_ms = min(self.min_response_time_ms, attempt.response_time_ms)
                            self.max_response_time_ms = max(self.max_response_time_ms, attempt.response_time_ms)

                            # Update running average
                            total_time = self.avg_response_time_ms * (self.successful_attempts - 1) + attempt.response_time_ms
                            self.avg_response_time_ms = total_time / self.successful_attempts

                elif attempt.is_completed:
                    self.failed_attempts += 1

                # Update channel stats
                channel = attempt.channel
                if channel not in self.channel_stats:
                    self.channel_stats[channel] = {
                        "total_attempts": 0,
                        "successful_attempts": 0,
                        "failed_attempts": 0,
                        "avg_response_time_ms": 0.0
                    }

                ch_stats = self.channel_stats[channel]
                ch_stats["total_attempts"] += 1

                if attempt.status == DeliveryStatus.DELIVERED:
                    ch_stats["successful_attempts"] += 1
                    if attempt.response_time_ms is not None:
                        # Update channel average response time
                        total_time = (ch_stats["avg_response_time_ms"] *
                                    (ch_stats["successful_attempts"] - 1) +
                                    attempt.response_time_ms)
                        ch_stats["avg_response_time_ms"] = total_time / ch_stats["successful_attempts"]
                elif attempt.is_completed:
                    ch_stats["failed_attempts"] += 1

        # Update priority stats
        priority = delivery_status.priority.value
        if priority not in self.priority_stats:
            self.priority_stats[priority] = {
                "total_messages": 0,
                "successful_deliveries": 0,
                "failed_deliveries": 0,
                "partial_deliveries": 0,
                "avg_response_time_ms": 0.0
            }

        pr_stats = self.priority_stats[priority]
        pr_stats["total_messages"] += 1

        if delivery_status.overall_status == DeliveryResult.SUCCESS:
            pr_stats["successful_deliveries"] += 1
        elif delivery_status.overall_status == DeliveryResult.FAILED:
            pr_stats["failed_deliveries"] += 1
        elif delivery_status.overall_status == DeliveryResult.PARTIAL_SUCCESS:
            pr_stats["partial_deliveries"] += 1

        # Update priority average response time
        avg_time = delivery_status.get_average_response_time_ms()
        if avg_time is not None:
            successful_count = (pr_stats["successful_deliveries"] +
                              pr_stats["partial_deliveries"])
            if successful_count == 1:
                pr_stats["avg_response_time_ms"] = avg_time
            else:
                total_time = (pr_stats["avg_response_time_ms"] * (successful_count - 1) + avg_time)
                pr_stats["avg_response_time_ms"] = total_time / successful_count


class DeliveryTracker:
    """
    Comprehensive delivery status tracking system.

    Features:
    - Track delivery attempts and results for each message
    - Support multi-channel delivery tracking
    - Record response times and external message IDs
    - Provide delivery confirmation callbacks
    - Generate delivery statistics and analytics
    - Persistent storage in database
    """

    def __init__(self):
        """Initialize delivery tracker."""
        self._active_deliveries: Dict[int, MessageDeliveryStatus] = {}
        self._lock = threading.RLock()
        self._stats = DeliveryStats()
        self._logger = setup_logger(f"{__name__}.DeliveryTracker")

        # Global status change callbacks
        self._global_callbacks: List[Callable] = []

        # Configuration
        self._max_active_deliveries = 10000  # Limit memory usage
        self._cleanup_interval_hours = 1

        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

    async def start_tracking(self, message: QueuedMessage) -> MessageDeliveryStatus:
        """
        Start tracking delivery for a message.

        Args:
            message: Message to track

        Returns:
            MessageDeliveryStatus object for tracking
        """
        delivery_status = MessageDeliveryStatus(
            message_id=message.id,
            message_type=message.message_type,
            priority=message.priority,
            channels=message.channels.copy(),
            recipient_id=message.recipient_id,
            created_at=message.created_at
        )

        with self._lock:
            self._active_deliveries[message.id] = delivery_status

        # Persist to database
        await self._save_delivery_status(delivery_status)

        self._logger.info(
            "Started tracking delivery for message %s (channels: %s)",
            message.id, message.channels
        )

        return delivery_status

    async def start_channel_attempt(
        self,
        message_id: int,
        channel: str
    ) -> Optional[ChannelDeliveryAttempt]:
        """
        Start a delivery attempt for a specific channel.

        Args:
            message_id: Message ID
            channel: Channel name

        Returns:
            ChannelDeliveryAttempt object or None if message not found
        """
        with self._lock:
            delivery_status = self._active_deliveries.get(message_id)

            if not delivery_status:
                # Try to load from database
                delivery_status = await self._load_delivery_status(message_id)
                if delivery_status:
                    self._active_deliveries[message_id] = delivery_status
                else:
                    self._logger.warning("Cannot start attempt for unknown message %s", message_id)
                    return None

            # Create new attempt
            attempt = ChannelDeliveryAttempt(
                attempt_id=str(uuid.uuid4()),
                message_id=message_id,
                channel=channel,
                status=DeliveryStatus.SENDING,
                started_at=datetime.now(timezone.utc)
            )

            # Get retry count from previous attempts
            previous_attempts = delivery_status.channel_attempts.get(channel, [])
            attempt.retry_count = len(previous_attempts)

            delivery_status.add_attempt(attempt)

        # Persist attempt to database
        await self._save_delivery_attempt(attempt)

        self._logger.debug(
            "Started delivery attempt %s for message %s on channel %s (retry: %d)",
            attempt.attempt_id, message_id, channel, attempt.retry_count
        )

        return attempt

    async def complete_channel_attempt(
        self,
        attempt_id: str,
        status: DeliveryStatus,
        response_time_ms: Optional[int] = None,
        external_id: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Complete a delivery attempt.

        Args:
            attempt_id: Attempt ID
            status: Final delivery status
            response_time_ms: Response time in milliseconds
            external_id: External message ID from channel
            error_message: Error message if failed

        Returns:
            True if attempt was found and updated
        """
        attempt = None
        delivery_status = None

        with self._lock:
            # Find the attempt
            for msg_id, msg_delivery in self._active_deliveries.items():
                for channel_attempts in msg_delivery.channel_attempts.values():
                    for att in channel_attempts:
                        if att.attempt_id == attempt_id:
                            attempt = att
                            delivery_status = msg_delivery
                            break
                    if attempt:
                        break
                if attempt:
                    break

        if not attempt:
            self._logger.warning("Cannot complete unknown attempt %s", attempt_id)
            return False

        # Update attempt
        attempt.status = status
        attempt.completed_at = datetime.now(timezone.utc)
        attempt.response_time_ms = response_time_ms
        attempt.external_id = external_id
        attempt.error_message = error_message

        # Update overall delivery status
        delivery_status._update_overall_status()

        # Persist updates
        await self._save_delivery_attempt(attempt)
        await self._save_delivery_status(delivery_status)

        # Notify callbacks
        await delivery_status.notify_status_change()
        await self._notify_global_callbacks(delivery_status)

        # Update statistics if delivery is complete
        if delivery_status.overall_status != DeliveryResult.PENDING:
            self._stats.add_delivery(delivery_status)

        self._logger.info(
            "Completed delivery attempt %s for message %s: %s (response_time: %sms)",
            attempt_id, attempt.message_id, status.value, response_time_ms
        )

        return True

    async def get_delivery_status(self, message_id: int) -> Optional[MessageDeliveryStatus]:
        """
        Get delivery status for a message.

        Args:
            message_id: Message ID

        Returns:
            MessageDeliveryStatus or None if not found
        """
        with self._lock:
            delivery_status = self._active_deliveries.get(message_id)

        if delivery_status:
            return delivery_status

        # Try to load from database
        return await self._load_delivery_status(message_id)

    async def get_delivery_history(
        self,
        recipient_id: Optional[str] = None,
        channel: Optional[str] = None,
        status: Optional[DeliveryResult] = None,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[MessageDeliveryStatus]:
        """
        Get delivery history with filtering.

        Args:
            recipient_id: Filter by recipient
            channel: Filter by channel
            status: Filter by delivery status
            since: Filter deliveries since this time
            limit: Maximum number of results

        Returns:
            List of MessageDeliveryStatus objects
        """
        # This would query the database for historical delivery data
        # For now, return from active deliveries with filtering

        results = []

        with self._lock:
            for delivery_status in self._active_deliveries.values():
                # Apply filters
                if recipient_id and delivery_status.recipient_id != recipient_id:
                    continue

                if channel and channel not in delivery_status.channels:
                    continue

                if status and delivery_status.overall_status != status:
                    continue

                if since and delivery_status.created_at < since:
                    continue

                results.append(delivery_status)

        # Sort by creation time (newest first) and limit
        results.sort(key=lambda d: d.created_at, reverse=True)
        return results[:limit]

    def get_statistics(
        self,
        since: Optional[datetime] = None,
        channel: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get delivery statistics.

        Args:
            since: Calculate stats since this time
            channel: Filter by specific channel

        Returns:
            Dictionary with delivery statistics
        """
        # For now, return current stats
        # In a full implementation, this would calculate stats from database

        stats_dict = {
            "total_messages": self._stats.total_messages,
            "successful_deliveries": self._stats.successful_deliveries,
            "failed_deliveries": self._stats.failed_deliveries,
            "partial_deliveries": self._stats.partial_deliveries,
            "pending_deliveries": self._stats.pending_deliveries,
            "success_rate": (
                self._stats.successful_deliveries / self._stats.total_messages
                if self._stats.total_messages > 0 else 0
            ),
            "total_attempts": self._stats.total_attempts,
            "successful_attempts": self._stats.successful_attempts,
            "failed_attempts": self._stats.failed_attempts,
            "attempt_success_rate": (
                self._stats.successful_attempts / self._stats.total_attempts
                if self._stats.total_attempts > 0 else 0
            ),
            "avg_response_time_ms": self._stats.avg_response_time_ms,
            "min_response_time_ms": self._stats.min_response_time_ms,
            "max_response_time_ms": self._stats.max_response_time_ms,
            "channel_statistics": self._stats.channel_stats,
            "priority_statistics": self._stats.priority_stats
        }

        # Filter by channel if requested
        if channel and channel in self._stats.channel_stats:
            stats_dict["channel_filter"] = channel
            stats_dict["filtered_channel_stats"] = self._stats.channel_stats[channel]

        return stats_dict

    def add_global_callback(self, callback: Callable):
        """Add global status change callback."""
        self._global_callbacks.append(callback)

    async def _notify_global_callbacks(self, delivery_status: MessageDeliveryStatus):
        """Notify all global callbacks."""
        for callback in self._global_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(delivery_status)
                else:
                    callback(delivery_status)
            except Exception as e:
                self._logger.error("Global callback failed for message %s: %s", delivery_status.message_id, e)

    async def _save_delivery_status(self, delivery_status: MessageDeliveryStatus):
        """Save delivery status to database."""
        try:
            # For now, just log the status change
            # In a full implementation, this would update the database
            self._logger.debug(
                "Delivery status updated for message %s: %s",
                delivery_status.message_id,
                delivery_status.overall_status.value
            )
        except Exception as e:
            self._logger.error("Failed to save delivery status for message %s: %s", delivery_status.message_id, e)

    async def _save_delivery_attempt(self, attempt: ChannelDeliveryAttempt):
        """Save delivery attempt to database."""
        try:
            # For now, just log the attempt
            # In a full implementation, this would save to msg_delivery_status table
            self._logger.debug(
                "Delivery attempt saved: %s for message %s on channel %s: %s",
                attempt.attempt_id, attempt.message_id, attempt.channel, attempt.status.value
            )
        except Exception as e:
            self._logger.error("Failed to save delivery attempt %s: %s", attempt.attempt_id, e)

    async def _load_delivery_status(self, message_id: int) -> Optional[MessageDeliveryStatus]:
        """Load delivery status from database."""
        try:
            # For now, return None since we don't have database persistence
            # In a full implementation, this would load from the database
            self._logger.debug("Attempted to load delivery status for message %s from database", message_id)
            return None

        except Exception as e:
            self._logger.error("Failed to load delivery status for message %s: %s", message_id, e)
            return None

    async def start_cleanup_task(self):
        """Start background cleanup task."""
        if self._cleanup_task and not self._cleanup_task.done():
            return

        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._logger.info("Started delivery tracker cleanup task")

    async def stop_cleanup_task(self):
        """Stop background cleanup task."""
        self._running = False

        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        self._logger.info("Stopped delivery tracker cleanup task")

    async def _cleanup_loop(self):
        """Background cleanup of completed deliveries."""
        while self._running:
            try:
                await self._cleanup_completed_deliveries()
                await asyncio.sleep(self._cleanup_interval_hours * 3600)  # Convert hours to seconds
            except asyncio.CancelledError:
                break
            except Exception:
                self._logger.exception("Error in delivery tracker cleanup:")
                await asyncio.sleep(300)  # Wait 5 minutes before retry

    async def _cleanup_completed_deliveries(self):
        """Clean up completed deliveries from memory."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)  # Keep 24 hours in memory

        with self._lock:
            completed_ids = []

            for msg_id, delivery_status in self._active_deliveries.items():
                if (delivery_status.overall_status != DeliveryResult.PENDING and
                    delivery_status.completed_at and
                    delivery_status.completed_at < cutoff_time):
                    completed_ids.append(msg_id)

            # Remove completed deliveries
            for msg_id in completed_ids:
                del self._active_deliveries[msg_id]

        if completed_ids:
            self._logger.info("Cleaned up %d completed deliveries from memory", len(completed_ids))


# Global delivery tracker instance
delivery_tracker = DeliveryTracker()