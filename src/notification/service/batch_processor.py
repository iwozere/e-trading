"""
Batching System for Normal Priority Messages

Time-based and size-based batching with configurable rules per channel.
Optimizes processing efficiency while maintaining individual message tracking.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import threading

from src.data.db.models.model_notification import MessagePriority
from src.notification.service.message_queue import QueuedMessage
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class BatchTrigger(Enum):
    """Batch processing trigger types."""
    SIZE = "SIZE"           # Triggered by batch size
    TIME = "TIME"           # Triggered by time elapsed
    MANUAL = "MANUAL"       # Manually triggered
    SHUTDOWN = "SHUTDOWN"   # Triggered by system shutdown


@dataclass
class BatchConfig:
    """Batch configuration for a channel."""

    max_batch_size: int = 10        # Maximum messages per batch
    max_wait_time_seconds: float = 30.0  # Maximum time to wait for batch
    min_batch_size: int = 1         # Minimum messages to trigger batch
    enabled: bool = True            # Whether batching is enabled

    # Channel-specific optimizations
    prefer_same_recipient: bool = False  # Group by recipient
    prefer_same_type: bool = False       # Group by message type

    def __post_init__(self):
        """Validate configuration."""
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if self.max_wait_time_seconds <= 0:
            raise ValueError("max_wait_time_seconds must be positive")
        if self.min_batch_size <= 0:
            raise ValueError("min_batch_size must be positive")
        if self.min_batch_size > self.max_batch_size:
            raise ValueError("min_batch_size cannot exceed max_batch_size")


@dataclass
class MessageBatch:
    """A batch of messages for processing."""

    batch_id: str
    channel: str
    messages: List[QueuedMessage] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    trigger: Optional[BatchTrigger] = None

    # Grouping criteria
    recipient_id: Optional[str] = None
    message_type: Optional[str] = None

    @property
    def size(self) -> int:
        """Get batch size."""
        return len(self.messages)

    @property
    def age_seconds(self) -> float:
        """Get batch age in seconds."""
        return time.time() - self.created_at

    @property
    def message_ids(self) -> List[int]:
        """Get list of message IDs in batch."""
        return [msg.id for msg in self.messages]

    def add_message(self, message: QueuedMessage) -> bool:
        """
        Add message to batch if compatible.

        Args:
            message: Message to add

        Returns:
            True if message was added, False if incompatible
        """
        # Check compatibility
        if self.recipient_id is not None and message.recipient_id != self.recipient_id:
            return False

        if self.message_type is not None and message.message_type != self.message_type:
            return False

        # Add message
        self.messages.append(message)

        # Set grouping criteria from first message
        if len(self.messages) == 1:
            self.recipient_id = message.recipient_id
            self.message_type = message.message_type

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert batch to dictionary for logging/monitoring."""
        return {
            "batch_id": self.batch_id,
            "channel": self.channel,
            "size": self.size,
            "age_seconds": self.age_seconds,
            "trigger": self.trigger.value if self.trigger else None,
            "recipient_id": self.recipient_id,
            "message_type": self.message_type,
            "message_ids": self.message_ids,
            "created_at": datetime.fromtimestamp(self.created_at).isoformat()
        }


@dataclass
class BatchStats:
    """Statistics for batch processing."""

    total_batches: int = 0
    total_messages: int = 0

    size_triggered: int = 0
    time_triggered: int = 0
    manual_triggered: int = 0
    shutdown_triggered: int = 0

    avg_batch_size: float = 0.0
    avg_wait_time_seconds: float = 0.0

    # Per-channel stats
    channel_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def add_batch(self, batch: MessageBatch):
        """Add batch statistics."""
        self.total_batches += 1
        self.total_messages += batch.size

        # Update trigger counts
        if batch.trigger == BatchTrigger.SIZE:
            self.size_triggered += 1
        elif batch.trigger == BatchTrigger.TIME:
            self.time_triggered += 1
        elif batch.trigger == BatchTrigger.MANUAL:
            self.manual_triggered += 1
        elif batch.trigger == BatchTrigger.SHUTDOWN:
            self.shutdown_triggered += 1

        # Update averages
        if self.total_batches == 1:
            self.avg_batch_size = batch.size
            self.avg_wait_time_seconds = batch.age_seconds
        else:
            self.avg_batch_size = (
                (self.avg_batch_size * (self.total_batches - 1) + batch.size) /
                self.total_batches
            )
            self.avg_wait_time_seconds = (
                (self.avg_wait_time_seconds * (self.total_batches - 1) + batch.age_seconds) /
                self.total_batches
            )

        # Update channel stats
        channel = batch.channel
        if channel not in self.channel_stats:
            self.channel_stats[channel] = {
                "batches": 0,
                "messages": 0,
                "avg_size": 0.0,
                "avg_wait_time": 0.0
            }

        ch_stats = self.channel_stats[channel]
        ch_stats["batches"] += 1
        ch_stats["messages"] += batch.size

        if ch_stats["batches"] == 1:
            ch_stats["avg_size"] = batch.size
            ch_stats["avg_wait_time"] = batch.age_seconds
        else:
            ch_stats["avg_size"] = (
                (ch_stats["avg_size"] * (ch_stats["batches"] - 1) + batch.size) /
                ch_stats["batches"]
            )
            ch_stats["avg_wait_time"] = (
                (ch_stats["avg_wait_time"] * (ch_stats["batches"] - 1) + batch.age_seconds) /
                ch_stats["batches"]
            )


class BatchProcessor:
    """
    Batching system for normal priority messages.

    Features:
    - Time-based and size-based batching
    - Configurable rules per channel
    - Message grouping by recipient or type
    - Individual message tracking within batches
    - Batch processing optimization
    - Statistics and monitoring
    """

    def __init__(self):
        """Initialize batch processor."""
        self._batches: Dict[str, MessageBatch] = {}  # channel -> current batch
        self._pending_messages: Dict[str, List[QueuedMessage]] = {}  # channel -> messages
        self._batch_configs: Dict[str, BatchConfig] = {}
        self._batch_timers: Dict[str, asyncio.Task] = {}

        self._lock = threading.RLock()
        self._stats = BatchStats()
        self._logger = setup_logger(f"{__name__}.BatchProcessor")

        # Processing callback
        self._batch_processor_callback: Optional[Callable] = None

        # Default configurations
        self._default_configs = {
            "telegram_channel": BatchConfig(
                max_batch_size=5,
                max_wait_time_seconds=10.0,
                prefer_same_recipient=True
            ),
            "email_channel": BatchConfig(
                max_batch_size=20,
                max_wait_time_seconds=60.0,
                prefer_same_recipient=True,
                prefer_same_type=True
            ),
            "sms_channel": BatchConfig(
                max_batch_size=10,
                max_wait_time_seconds=30.0,
                prefer_same_recipient=True
            ),
            "default": BatchConfig(
                max_batch_size=10,
                max_wait_time_seconds=30.0
            )
        }

        # State
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Batch ID counter
        self._batch_counter = 0

    def set_batch_processor_callback(self, callback: Callable[[MessageBatch], None]):
        """
        Set callback function for processing completed batches.

        Args:
            callback: Function to call when batch is ready for processing
        """
        self._batch_processor_callback = callback
        self._logger.info("Batch processor callback set")

    def set_channel_config(self, channel: str, config: BatchConfig):
        """
        Set batch configuration for a channel.

        Args:
            channel: Channel name
            config: Batch configuration
        """
        self._batch_configs[channel] = config
        self._logger.info(
            "Updated batch config for %s: max_size=%d, max_wait=%.1fs",
            channel, config.max_batch_size, config.max_wait_time_seconds
        )

    def _get_config(self, channel: str) -> BatchConfig:
        """Get batch configuration for channel."""
        return self._batch_configs.get(
            channel,
            self._default_configs.get(channel, self._default_configs["default"])
        )

    def _generate_batch_id(self, channel: str) -> str:
        """Generate unique batch ID."""
        self._batch_counter += 1
        timestamp = int(time.time())
        return f"batch_{channel}_{timestamp}_{self._batch_counter}"

    async def add_message(self, message: QueuedMessage) -> bool:
        """
        Add message to batching system.

        Args:
            message: Message to add to batch

        Returns:
            True if message was added successfully
        """
        # Only batch normal and low priority messages
        if message.priority in [MessagePriority.CRITICAL, MessagePriority.HIGH]:
            return False

        # Determine target channel (use first channel for batching)
        if not message.channels:
            return False

        channel = message.channels[0]  # Use first channel for batching
        config = self._get_config(channel)

        if not config.enabled:
            return False

        try:
            with self._lock:
                # Initialize channel data if needed
                if channel not in self._pending_messages:
                    self._pending_messages[channel] = []

                # Add to pending messages
                self._pending_messages[channel].append(message)

                # Try to add to current batch or create new one
                await self._process_pending_messages(channel)

            self._logger.debug("Added message %s to batch queue for channel %s", message.id, channel)
            return True

        except Exception as e:
            self._logger.error("Failed to add message %s to batch: %s", message.id, e)
            return False

    async def _process_pending_messages(self, channel: str):
        """Process pending messages for a channel."""
        config = self._get_config(channel)
        pending = self._pending_messages.get(channel, [])

        if not pending:
            return

        # Get or create current batch
        current_batch = self._batches.get(channel)

        if current_batch is None:
            # Create new batch
            batch_id = self._generate_batch_id(channel)
            current_batch = MessageBatch(
                batch_id=batch_id,
                channel=channel
            )
            self._batches[channel] = current_batch

            # Start timer for this batch
            await self._start_batch_timer(channel, config.max_wait_time_seconds)

        # Add compatible messages to batch
        messages_added = []

        for message in pending:
            if current_batch.size >= config.max_batch_size:
                break

            # Check compatibility based on config
            can_add = True

            if config.prefer_same_recipient and current_batch.size > 0:
                if current_batch.recipient_id != message.recipient_id:
                    can_add = False

            if config.prefer_same_type and current_batch.size > 0:
                if current_batch.message_type != message.message_type:
                    can_add = False

            if can_add and current_batch.add_message(message):
                messages_added.append(message)

        # Remove added messages from pending
        for message in messages_added:
            pending.remove(message)

        # Check if batch should be processed
        if current_batch.size >= config.max_batch_size:
            await self._complete_batch(channel, BatchTrigger.SIZE)
        elif current_batch.size >= config.min_batch_size and current_batch.age_seconds > config.max_wait_time_seconds:
            await self._complete_batch(channel, BatchTrigger.TIME)

    async def _start_batch_timer(self, channel: str, wait_time_seconds: float):
        """Start timer for batch completion."""
        # Cancel existing timer
        if channel in self._batch_timers:
            self._batch_timers[channel].cancel()

        # Start new timer
        async def timer_callback():
            try:
                await asyncio.sleep(wait_time_seconds)
                await self._complete_batch(channel, BatchTrigger.TIME)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                self._logger.error("Batch timer error for channel %s: %s", channel, e)

        self._batch_timers[channel] = asyncio.create_task(timer_callback())

    async def _complete_batch(self, channel: str, trigger: BatchTrigger):
        """Complete and process a batch."""
        with self._lock:
            batch = self._batches.get(channel)

            if batch is None or batch.size == 0:
                return

            # Remove batch from active batches
            del self._batches[channel]

            # Cancel timer
            if channel in self._batch_timers:
                self._batch_timers[channel].cancel()
                del self._batch_timers[channel]

            # Set trigger
            batch.trigger = trigger

            # Update statistics
            self._stats.add_batch(batch)

        self._logger.info(
            "Completed batch %s for channel %s: %d messages (trigger: %s, age: %.1fs)",
            batch.batch_id, channel, batch.size, trigger.value, batch.age_seconds
        )

        # Process batch
        if self._batch_processor_callback:
            try:
                await self._batch_processor_callback(batch)
            except Exception as e:
                self._logger.error("Batch processor callback failed for batch %s: %s", batch.batch_id, e)

        # Continue processing pending messages
        await self._process_pending_messages(channel)

    async def flush_channel(self, channel: str) -> Optional[MessageBatch]:
        """
        Flush current batch for a channel immediately.

        Args:
            channel: Channel to flush

        Returns:
            Flushed batch or None if no batch exists
        """
        with self._lock:
            batch = self._batches.get(channel)

            if batch and batch.size > 0:
                await self._complete_batch(channel, BatchTrigger.MANUAL)
                return batch

        return None

    async def flush_all_channels(self) -> List[MessageBatch]:
        """
        Flush all current batches immediately.

        Returns:
            List of flushed batches
        """
        flushed_batches = []

        with self._lock:
            channels = list(self._batches.keys())

        for channel in channels:
            batch = await self.flush_channel(channel)
            if batch:
                flushed_batches.append(batch)

        self._logger.info("Flushed %d batches from all channels", len(flushed_batches))
        return flushed_batches

    def get_batch_status(self) -> Dict[str, Any]:
        """
        Get current batch status.

        Returns:
            Dictionary with batch status information
        """
        with self._lock:
            channel_status = {}

            for channel, batch in self._batches.items():
                config = self._get_config(channel)
                pending_count = len(self._pending_messages.get(channel, []))

                channel_status[channel] = {
                    "current_batch": batch.to_dict() if batch else None,
                    "pending_messages": pending_count,
                    "config": {
                        "max_batch_size": config.max_batch_size,
                        "max_wait_time_seconds": config.max_wait_time_seconds,
                        "min_batch_size": config.min_batch_size,
                        "enabled": config.enabled,
                        "prefer_same_recipient": config.prefer_same_recipient,
                        "prefer_same_type": config.prefer_same_type
                    }
                }

            # Add channels with pending messages but no active batch
            for channel, pending in self._pending_messages.items():
                if channel not in channel_status and pending:
                    config = self._get_config(channel)
                    channel_status[channel] = {
                        "current_batch": None,
                        "pending_messages": len(pending),
                        "config": {
                            "max_batch_size": config.max_batch_size,
                            "max_wait_time_seconds": config.max_wait_time_seconds,
                            "min_batch_size": config.min_batch_size,
                            "enabled": config.enabled,
                            "prefer_same_recipient": config.prefer_same_recipient,
                            "prefer_same_type": config.prefer_same_type
                        }
                    }

        return {
            "channels": channel_status,
            "statistics": {
                "total_batches": self._stats.total_batches,
                "total_messages": self._stats.total_messages,
                "avg_batch_size": self._stats.avg_batch_size,
                "avg_wait_time_seconds": self._stats.avg_wait_time_seconds,
                "trigger_counts": {
                    "size": self._stats.size_triggered,
                    "time": self._stats.time_triggered,
                    "manual": self._stats.manual_triggered,
                    "shutdown": self._stats.shutdown_triggered
                },
                "channel_stats": self._stats.channel_stats
            }
        }

    async def start(self):
        """Start batch processor."""
        self._running = True
        self._shutdown_event.clear()
        self._logger.info("Batch processor started")

    async def stop(self):
        """Stop batch processor and flush all batches."""
        self._running = False
        self._shutdown_event.set()

        # Cancel all timers
        for timer in self._batch_timers.values():
            timer.cancel()

        # Flush all remaining batches
        with self._lock:
            channels = list(self._batches.keys())

        for channel in channels:
            batch = self._batches.get(channel)
            if batch and batch.size > 0:
                await self._complete_batch(channel, BatchTrigger.SHUTDOWN)

        self._logger.info("Batch processor stopped and all batches flushed")

    def clear_statistics(self):
        """Clear batch processing statistics."""
        self._stats = BatchStats()
        self._logger.info("Batch processing statistics cleared")


# Global batch processor instance
batch_processor = BatchProcessor()