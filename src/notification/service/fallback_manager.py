"""
Channel Fallback and Recovery Manager

Manages channel fallback routing, recovery mechanisms, and dead letter queue
for the notification service. Provides automatic failover when primary channels
are unavailable and recovery mechanisms for failed messages.
"""

from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

from src.notification.service.health_monitor import HealthMonitor, HealthStatus
from src.notification.channels.base import (
    NotificationChannel, DeliveryResult, DeliveryStatus, MessageContent
)
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class FallbackStrategy(str, Enum):
    """Fallback strategy options."""
    ROUND_ROBIN = "round_robin"
    PRIORITY_ORDER = "priority_order"
    HEALTH_BASED = "health_based"
    LOAD_BALANCED = "load_balanced"


class MessageFailureReason(str, Enum):
    """Reasons for message failure."""
    CHANNEL_DOWN = "channel_down"
    CHANNEL_UNHEALTHY = "channel_unhealthy"
    RATE_LIMITED = "rate_limited"
    DELIVERY_FAILED = "delivery_failed"
    TIMEOUT = "timeout"
    CONFIGURATION_ERROR = "configuration_error"
    PERMANENT_FAILURE = "permanent_failure"


@dataclass
class FallbackRule:
    """Configuration for channel fallback behavior."""
    primary_channel: str
    fallback_channels: List[str]
    strategy: FallbackStrategy = FallbackStrategy.PRIORITY_ORDER
    max_attempts: int = 3
    retry_delay_seconds: int = 60
    health_threshold: HealthStatus = HealthStatus.DEGRADED
    enabled: bool = True

    def __post_init__(self):
        """Validate fallback rule configuration."""
        if not self.fallback_channels:
            raise ValueError("At least one fallback channel must be specified")

        if self.primary_channel in self.fallback_channels:
            raise ValueError("Primary channel cannot be in fallback channels list")

        if self.max_attempts < 1:
            raise ValueError("Max attempts must be at least 1")


@dataclass
class FailedMessage:
    """Information about a failed message."""
    message_id: int
    original_channels: List[str]
    content: MessageContent
    recipient: str
    priority: str
    failure_reason: MessageFailureReason
    failure_details: str
    failed_at: datetime
    retry_count: int = 0
    last_retry_at: Optional[datetime] = None
    attempted_channels: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "message_id": self.message_id,
            "original_channels": self.original_channels,
            "content": {
                "text": self.content.text,
                "subject": self.content.subject,
                "html": self.content.html,
                "has_attachments": self.content.has_attachments
            },
            "recipient": self.recipient,
            "priority": self.priority,
            "failure_reason": self.failure_reason.value,
            "failure_details": self.failure_details,
            "failed_at": self.failed_at.isoformat(),
            "retry_count": self.retry_count,
            "last_retry_at": self.last_retry_at.isoformat() if self.last_retry_at else None,
            "attempted_channels": list(self.attempted_channels),
            "metadata": self.metadata
        }


@dataclass
class FallbackAttempt:
    """Record of a fallback attempt."""
    message_id: int
    original_channel: str
    fallback_channel: str
    attempted_at: datetime
    result: DeliveryResult
    strategy_used: FallbackStrategy


class FallbackManager:
    """
    Channel fallback and recovery manager.

    Features:
    - Automatic channel fallback based on health status
    - Multiple fallback strategies (priority, round-robin, health-based)
    - Dead letter queue for permanently failed messages
    - Message retry with exponential backoff
    - Manual message reprocessing
    - Fallback attempt tracking and analytics
    """

    def __init__(self, health_monitor: HealthMonitor):
        """
        Initialize fallback manager.

        Args:
            health_monitor: Health monitor instance for channel status
        """
        self.health_monitor = health_monitor
        self._logger = setup_logger(f"{__name__}.FallbackManager")

        # Fallback configuration
        self._fallback_rules: Dict[str, FallbackRule] = {}
        self._global_fallback_channels: List[str] = []
        self._default_strategy = FallbackStrategy.PRIORITY_ORDER

        # Dead letter queue
        self._dead_letter_queue: Dict[int, FailedMessage] = {}
        self._max_dead_letter_size = 10000
        self._dead_letter_retention_days = 30

        # Retry queue
        self._retry_queue: Dict[int, FailedMessage] = {}
        self._max_retry_attempts = 5
        self._retry_backoff_multiplier = 2.0
        self._base_retry_delay = 60  # seconds

        # Fallback attempt tracking
        self._fallback_attempts: deque = deque(maxlen=1000)
        self._channel_usage_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"attempts": 0, "successes": 0, "failures": 0}
        )

        # Statistics
        self._stats = {
            "total_fallback_attempts": 0,
            "successful_fallbacks": 0,
            "failed_fallbacks": 0,
            "dead_letter_messages": 0,
            "retry_attempts": 0,
            "manual_reprocessing": 0
        }

    def configure_fallback_rule(self, rule: FallbackRule):
        """
        Configure fallback rule for a channel.

        Args:
            rule: Fallback rule configuration
        """
        self._fallback_rules[rule.primary_channel] = rule
        self._logger.info(
            "Configured fallback rule for channel %s: %s fallback channels",
            rule.primary_channel, len(rule.fallback_channels)
        )

    def set_global_fallback_channels(self, channels: List[str]):
        """
        Set global fallback channels used when no specific rule exists.

        Args:
            channels: List of channel names to use as global fallbacks
        """
        self._global_fallback_channels = channels
        self._logger.info("Set global fallback channels: %s", channels)

    def remove_fallback_rule(self, channel: str) -> bool:
        """
        Remove fallback rule for a channel.

        Args:
            channel: Channel name

        Returns:
            True if rule was removed, False if not found
        """
        if channel in self._fallback_rules:
            del self._fallback_rules[channel]
            self._logger.info("Removed fallback rule for channel %s", channel)
            return True
        return False

    async def attempt_delivery_with_fallback(
        self,
        message_id: int,
        channels: List[str],
        recipient: str,
        content: MessageContent,
        priority: str = "NORMAL",
        channel_instances: Dict[str, NotificationChannel] = None
    ) -> Tuple[bool, List[DeliveryResult], Optional[FailedMessage]]:
        """
        Attempt message delivery with automatic fallback.

        Args:
            message_id: Message ID for tracking
            channels: List of channels to attempt
            recipient: Message recipient
            content: Message content
            priority: Message priority
            channel_instances: Dictionary of channel instances

        Returns:
            Tuple of (success, delivery_results, failed_message_if_any)
        """
        if not channels:
            failed_msg = FailedMessage(
                message_id=message_id,
                original_channels=[],
                content=content,
                recipient=recipient,
                priority=priority,
                failure_reason=MessageFailureReason.CONFIGURATION_ERROR,
                failure_details="No channels specified",
                failed_at=datetime.now(timezone.utc)
            )
            return False, [], failed_msg

        delivery_results = []
        attempted_channels = set()

        # Try each channel with fallback
        for primary_channel in channels:
            try:
                # Get channels to try (primary + fallbacks)
                channels_to_try = await self._get_channels_to_try(
                    primary_channel, attempted_channels
                )

                # Attempt delivery through available channels
                for channel_name in channels_to_try:
                    if channel_name in attempted_channels:
                        continue

                    attempted_channels.add(channel_name)

                    # Check channel health
                    if not await self._is_channel_available(channel_name):
                        self._logger.warning(
                            "Skipping unhealthy channel %s for message %s",
                            channel_name, message_id
                        )
                        continue

                    # Get channel instance
                    if not channel_instances or channel_name not in channel_instances:
                        self._logger.error(
                            "Channel instance not available: %s", channel_name
                        )
                        # Create failed result for unavailable channel
                        failed_result = DeliveryResult(
                            success=False,
                            status=DeliveryStatus.FAILED,
                            error_message=f"Channel instance not available: {channel_name}"
                        )
                        delivery_results.append(failed_result)
                        self._update_channel_stats(channel_name, False)
                        continue

                    channel_instance = channel_instances[channel_name]

                    # Attempt delivery
                    try:
                        result = await channel_instance.send_message(
                            recipient, content, str(message_id), priority
                        )

                        delivery_results.append(result)
                        self._update_channel_stats(channel_name, result.success)

                        # Record fallback attempt if not primary channel
                        if channel_name != primary_channel:
                            self._record_fallback_attempt(
                                message_id, primary_channel, channel_name, result
                            )

                        if result.success:
                            self._logger.info(
                                "Message %s delivered successfully via channel %s",
                                message_id, channel_name
                            )
                            return True, delivery_results, None

                        self._logger.warning(
                            "Delivery failed for message %s via channel %s: %s",
                            message_id, channel_name, result.error_message
                        )

                    except Exception as e:
                        error_msg = f"Channel delivery exception: {str(e)}"
                        self._logger.error(
                            "Exception during delivery for message %s via channel %s: %s",
                            message_id, channel_name, error_msg
                        )

                        # Create failed result
                        result = DeliveryResult(
                            success=False,
                            status=DeliveryStatus.FAILED,
                            error_message=error_msg
                        )
                        delivery_results.append(result)
                        self._update_channel_stats(channel_name, False)

            except Exception as e:
                self._logger.error(
                    "Error processing fallback for channel %s: %s",
                    primary_channel, e
                )

        # All channels failed - create failed message
        failed_msg = FailedMessage(
            message_id=message_id,
            original_channels=channels,
            content=content,
            recipient=recipient,
            priority=priority,
            failure_reason=MessageFailureReason.DELIVERY_FAILED,
            failure_details=f"All channels failed. Attempted: {list(attempted_channels)}",
            failed_at=datetime.now(timezone.utc),
            attempted_channels=attempted_channels
        )

        # Add to retry queue if retryable
        if self._is_retryable_failure(delivery_results):
            await self._add_to_retry_queue(failed_msg)
        else:
            await self._add_to_dead_letter_queue(failed_msg)

        return False, delivery_results, failed_msg

    async def _get_channels_to_try(
        self,
        primary_channel: str,
        already_attempted: Set[str]
    ) -> List[str]:
        """
        Get ordered list of channels to try for delivery.

        Args:
            primary_channel: Primary channel name
            already_attempted: Set of already attempted channels

        Returns:
            Ordered list of channels to try
        """
        channels_to_try = [primary_channel]

        # Get fallback rule for this channel
        fallback_rule = self._fallback_rules.get(primary_channel)

        if fallback_rule and fallback_rule.enabled:
            # Use configured fallback channels
            fallback_channels = fallback_rule.fallback_channels.copy()

            # Apply fallback strategy
            if fallback_rule.strategy == FallbackStrategy.HEALTH_BASED:
                fallback_channels = await self._sort_channels_by_health(fallback_channels)
            elif fallback_rule.strategy == FallbackStrategy.LOAD_BALANCED:
                fallback_channels = await self._sort_channels_by_load(fallback_channels)
            elif fallback_rule.strategy == FallbackStrategy.ROUND_ROBIN:
                fallback_channels = await self._apply_round_robin(fallback_channels)
            # PRIORITY_ORDER uses the configured order as-is

            channels_to_try.extend(fallback_channels)

        elif self._global_fallback_channels:
            # Use global fallback channels
            global_fallbacks = [
                ch for ch in self._global_fallback_channels
                if ch != primary_channel and ch not in already_attempted
            ]
            channels_to_try.extend(global_fallbacks)

        # Remove duplicates while preserving order
        seen = set()
        unique_channels = []
        for channel in channels_to_try:
            if channel not in seen and channel not in already_attempted:
                seen.add(channel)
                unique_channels.append(channel)

        return unique_channels

    async def _is_channel_available(self, channel_name: str) -> bool:
        """
        Check if a channel is available for delivery.

        Args:
            channel_name: Channel name to check

        Returns:
            True if channel is available
        """
        try:
            channel_status = self.health_monitor.get_channel_status(channel_name)

            if not channel_status:
                self._logger.debug("No channel status found for %s, assuming available", channel_name)
                return True  # Assume available if no status information

            self._logger.debug("Channel %s status: enabled=%s, overall_status=%s",
                             channel_name, channel_status.is_enabled, channel_status.overall_status)

            # Check if channel is enabled and healthy enough
            if not channel_status.is_enabled:
                self._logger.debug("Channel %s is disabled", channel_name)
                return False

            # Allow delivery if channel is healthy or degraded
            is_available = channel_status.overall_status in [
                HealthStatus.HEALTHY,
                HealthStatus.DEGRADED
            ]

            self._logger.debug("Channel %s availability: %s", channel_name, is_available)
            return is_available

        except Exception as e:
            self._logger.error("Error checking channel availability %s: %s", channel_name, e)
            return False

    async def _sort_channels_by_health(self, channels: List[str]) -> List[str]:
        """Sort channels by health status (healthiest first)."""
        channel_health = []

        for channel in channels:
            status = self.health_monitor.get_channel_status(channel)
            if status:
                # Assign numeric scores for sorting
                health_score = {
                    HealthStatus.HEALTHY: 4,
                    HealthStatus.DEGRADED: 3,
                    HealthStatus.UNHEALTHY: 2,
                    HealthStatus.CRITICAL: 1,
                    HealthStatus.DISABLED: 0,
                    HealthStatus.UNKNOWN: 0
                }.get(status.overall_status, 0)

                channel_health.append((channel, health_score, status.uptime_percentage))

        # Sort by health score (desc), then uptime (desc)
        channel_health.sort(key=lambda x: (x[1], x[2]), reverse=True)

        return [ch[0] for ch in channel_health]

    async def _sort_channels_by_load(self, channels: List[str]) -> List[str]:
        """Sort channels by current load (least loaded first)."""
        # This would integrate with rate limiter to get current load
        # For now, return channels as-is
        return channels

    async def _apply_round_robin(self, channels: List[str]) -> List[str]:
        """Apply round-robin selection to channels."""
        # Simple round-robin based on current time
        import time
        current_time = int(time.time())
        start_index = current_time % len(channels)

        return channels[start_index:] + channels[:start_index]

    def _is_retryable_failure(self, delivery_results: List[DeliveryResult]) -> bool:
        """
        Check if failure is retryable based on delivery results.

        Args:
            delivery_results: List of delivery results

        Returns:
            True if failure is retryable
        """
        if not delivery_results:
            return True

        # Check if any result indicates a retryable failure
        for result in delivery_results:
            if result.status == DeliveryStatus.BOUNCED:
                return False  # Bounced messages are not retryable

            if result.error_message:
                error_lower = result.error_message.lower()
                # Non-retryable errors
                if any(term in error_lower for term in [
                    "unauthorized", "forbidden", "invalid token",
                    "authentication failed", "permission denied"
                ]):
                    return False

        return True

    async def _add_to_retry_queue(self, failed_message: FailedMessage):
        """Add message to retry queue."""
        if failed_message.retry_count >= self._max_retry_attempts:
            await self._add_to_dead_letter_queue(failed_message)
            return

        self._retry_queue[failed_message.message_id] = failed_message
        self._stats["retry_attempts"] += 1

        self._logger.info(
            "Added message %s to retry queue (attempt %s/%s)",
            failed_message.message_id, failed_message.retry_count + 1, self._max_retry_attempts
        )

    async def _add_to_dead_letter_queue(self, failed_message: FailedMessage):
        """Add message to dead letter queue."""
        # Check queue size limit
        if len(self._dead_letter_queue) >= self._max_dead_letter_size:
            # Remove oldest messages
            oldest_messages = sorted(
                self._dead_letter_queue.items(),
                key=lambda x: x[1].failed_at
            )

            for msg_id, _ in oldest_messages[:100]:  # Remove 100 oldest
                del self._dead_letter_queue[msg_id]

        self._dead_letter_queue[failed_message.message_id] = failed_message
        self._stats["dead_letter_messages"] += 1

        self._logger.warning(
            "Added message %s to dead letter queue: %s",
            failed_message.message_id, failed_message.failure_details
        )

    def _record_fallback_attempt(
        self,
        message_id: int,
        original_channel: str,
        fallback_channel: str,
        result: DeliveryResult
    ):
        """Record a fallback attempt for analytics."""
        attempt = FallbackAttempt(
            message_id=message_id,
            original_channel=original_channel,
            fallback_channel=fallback_channel,
            attempted_at=datetime.now(timezone.utc),
            result=result,
            strategy_used=self._default_strategy
        )

        self._fallback_attempts.append(attempt)
        self._stats["total_fallback_attempts"] += 1

        if result.success:
            self._stats["successful_fallbacks"] += 1
        else:
            self._stats["failed_fallbacks"] += 1

    def _update_channel_stats(self, channel_name: str, success: bool):
        """Update channel usage statistics."""
        stats = self._channel_usage_stats[channel_name]
        stats["attempts"] += 1

        if success:
            stats["successes"] += 1
        else:
            stats["failures"] += 1

    async def process_retry_queue(
        self,
        channel_instances: Dict[str, NotificationChannel]
    ) -> Dict[str, Any]:
        """
        Process messages in the retry queue.

        Args:
            channel_instances: Dictionary of channel instances

        Returns:
            Processing results summary
        """
        if not self._retry_queue:
            return {"processed": 0, "succeeded": 0, "failed": 0, "requeued": 0}

        current_time = datetime.now(timezone.utc)
        messages_to_retry = []

        # Find messages ready for retry
        for message_id, failed_msg in list(self._retry_queue.items()):
            # Calculate retry delay with exponential backoff
            retry_delay = self._base_retry_delay * (
                self._retry_backoff_multiplier ** failed_msg.retry_count
            )

            next_retry_time = failed_msg.last_retry_at or failed_msg.failed_at
            next_retry_time += timedelta(seconds=retry_delay)

            if current_time >= next_retry_time:
                messages_to_retry.append(failed_msg)

        results = {"processed": 0, "succeeded": 0, "failed": 0, "requeued": 0}

        # Process retry messages
        for failed_msg in messages_to_retry:
            try:
                # Remove from retry queue
                self._retry_queue.pop(failed_msg.message_id, None)

                # Update retry info
                failed_msg.retry_count += 1
                failed_msg.last_retry_at = current_time

                # Attempt delivery with fallback
                success, delivery_results, new_failed_msg = await self.attempt_delivery_with_fallback(
                    failed_msg.message_id,
                    failed_msg.original_channels,
                    failed_msg.recipient,
                    failed_msg.content,
                    failed_msg.priority,
                    channel_instances
                )

                results["processed"] += 1

                if success:
                    results["succeeded"] += 1
                    self._logger.info(
                        "Retry successful for message %s after %s attempts",
                        failed_msg.message_id, failed_msg.retry_count
                    )
                else:
                    results["failed"] += 1

                    # Check if we should retry again or move to dead letter
                    if failed_msg.retry_count < self._max_retry_attempts:
                        await self._add_to_retry_queue(failed_msg)
                        results["requeued"] += 1
                    else:
                        await self._add_to_dead_letter_queue(failed_msg)

            except Exception as e:
                self._logger.error(
                    "Error processing retry for message %s: %s",
                    failed_msg.message_id, e
                )
                results["failed"] += 1

        return results

    async def reprocess_dead_letter_message(
        self,
        message_id: int,
        channel_instances: Dict[str, NotificationChannel],
        force_channels: Optional[List[str]] = None
    ) -> Tuple[bool, str]:
        """
        Manually reprocess a message from the dead letter queue.

        Args:
            message_id: Message ID to reprocess
            channel_instances: Dictionary of channel instances
            force_channels: Optional list of channels to force use

        Returns:
            Tuple of (success, message)
        """
        if message_id not in self._dead_letter_queue:
            return False, f"Message {message_id} not found in dead letter queue"

        failed_msg = self._dead_letter_queue[message_id]

        try:
            # Use forced channels or original channels
            channels_to_use = force_channels or failed_msg.original_channels

            # Reset attempted channels if forcing specific channels
            if force_channels:
                failed_msg.attempted_channels.clear()

            # Attempt delivery
            success, delivery_results, new_failed_msg = await self.attempt_delivery_with_fallback(
                failed_msg.message_id,
                channels_to_use,
                failed_msg.recipient,
                failed_msg.content,
                failed_msg.priority,
                channel_instances
            )

            self._stats["manual_reprocessing"] += 1

            if success:
                # Remove from dead letter queue
                del self._dead_letter_queue[message_id]

                return True, f"Message {message_id} reprocessed successfully"
            else:
                # Update failed message with new attempt info
                if new_failed_msg:
                    failed_msg.attempted_channels.update(new_failed_msg.attempted_channels)
                    failed_msg.failure_details = new_failed_msg.failure_details

                return False, f"Message {message_id} reprocessing failed: {new_failed_msg.failure_details if new_failed_msg else 'Unknown error'}"

        except Exception as e:
            error_msg = f"Error reprocessing message {message_id}: {str(e)}"
            self._logger.error(error_msg)
            return False, error_msg

    def get_dead_letter_messages(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get messages from dead letter queue.

        Args:
            limit: Maximum number of messages to return
            offset: Number of messages to skip

        Returns:
            List of dead letter messages
        """
        # Sort by failed_at timestamp (newest first)
        sorted_messages = sorted(
            self._dead_letter_queue.values(),
            key=lambda x: x.failed_at,
            reverse=True
        )

        # Apply pagination
        start_idx = offset
        end_idx = offset + limit
        paginated_messages = sorted_messages[start_idx:end_idx]

        return [msg.to_dict() for msg in paginated_messages]

    def get_retry_queue_status(self) -> Dict[str, Any]:
        """Get retry queue status information."""
        if not self._retry_queue:
            return {
                "total_messages": 0,
                "ready_for_retry": 0,
                "next_retry_time": None
            }

        current_time = datetime.now(timezone.utc)
        ready_count = 0
        next_retry_times = []

        for failed_msg in self._retry_queue.values():
            retry_delay = self._base_retry_delay * (
                self._retry_backoff_multiplier ** failed_msg.retry_count
            )

            next_retry_time = failed_msg.last_retry_at or failed_msg.failed_at
            next_retry_time += timedelta(seconds=retry_delay)

            if current_time >= next_retry_time:
                ready_count += 1
            else:
                next_retry_times.append(next_retry_time)

        return {
            "total_messages": len(self._retry_queue),
            "ready_for_retry": ready_count,
            "next_retry_time": min(next_retry_times).isoformat() if next_retry_times else None
        }

    def get_fallback_statistics(self) -> Dict[str, Any]:
        """Get fallback and recovery statistics."""
        # Calculate success rates by channel
        channel_success_rates = {}
        for channel, stats in self._channel_usage_stats.items():
            if stats["attempts"] > 0:
                success_rate = (stats["successes"] / stats["attempts"]) * 100
                channel_success_rates[channel] = {
                    "success_rate": round(success_rate, 2),
                    "total_attempts": stats["attempts"],
                    "successes": stats["successes"],
                    "failures": stats["failures"]
                }

        # Recent fallback attempts
        recent_attempts = []
        for attempt in list(self._fallback_attempts)[-10:]:  # Last 10 attempts
            recent_attempts.append({
                "message_id": attempt.message_id,
                "original_channel": attempt.original_channel,
                "fallback_channel": attempt.fallback_channel,
                "attempted_at": attempt.attempted_at.isoformat(),
                "success": attempt.result.success,
                "strategy": attempt.strategy_used.value
            })

        return {
            "statistics": self._stats.copy(),
            "channel_success_rates": channel_success_rates,
            "recent_fallback_attempts": recent_attempts,
            "dead_letter_queue_size": len(self._dead_letter_queue),
            "retry_queue_size": len(self._retry_queue),
            "configured_fallback_rules": len(self._fallback_rules)
        }

    async def cleanup_old_dead_letters(self) -> int:
        """
        Clean up old messages from dead letter queue.

        Returns:
            Number of messages cleaned up
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=self._dead_letter_retention_days)

        messages_to_remove = [
            msg_id for msg_id, failed_msg in self._dead_letter_queue.items()
            if failed_msg.failed_at < cutoff_time
        ]

        for msg_id in messages_to_remove:
            del self._dead_letter_queue[msg_id]

        if messages_to_remove:
            self._logger.info(
                "Cleaned up %s old messages from dead letter queue",
                len(messages_to_remove)
            )

        return len(messages_to_remove)