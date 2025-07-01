"""
Async Notification Manager
=========================

Provides a unified async notification system with:
- Queued notifications to prevent blocking
- Batching and rate limiting
- Retry mechanisms
- Smart filtering and aggregation
- Multiple notification channels (Telegram, Email, etc.)
"""

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict

from src.notification.logger import setup_logger
from src.notification.telegram_notifier import TelegramNotifier
from src.notification.emailer import EmailNotifier

_logger = setup_logger(__name__)


class NotificationType(Enum):
    """Types of notifications"""
    TRADE_ENTRY = "trade_entry"
    TRADE_EXIT = "trade_exit"
    TRADE_UPDATE = "trade_update"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    SYSTEM = "system"
    PERFORMANCE = "performance"


class NotificationPriority(Enum):
    """Notification priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Notification:
    """Notification data structure"""
    type: NotificationType
    priority: NotificationPriority
    title: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "trading_bot"
    retry_count: int = 0
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert notification to dictionary"""
        return {
            "type": self.type.value,
            "priority": self.priority.value,
            "title": self.title,
            "message": self.message,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries
        }


class NotificationChannel:
    """Base class for notification channels"""

    def __init__(self, name: str):
        self.name = name
        self.enabled = True
        self.logger = setup_logger(f"{__name__}.{name}")

    async def send(self, notification: Notification) -> bool:
        """Send a notification (to be implemented by subclasses)"""
        raise NotImplementedError

    def is_enabled(self) -> bool:
        """Check if channel is enabled"""
        return self.enabled

    def enable(self):
        """Enable the channel"""
        self.enabled = True

    def disable(self):
        """Disable the channel"""
        self.enabled = False


class TelegramChannel(NotificationChannel):
    """Telegram notification channel"""

    def __init__(self, token: str, chat_id: str):
        super().__init__("telegram")
        self.notifier = TelegramNotifier(token, chat_id)

    async def send(self, notification: Notification) -> bool:
        """Send notification via Telegram"""
        try:
            if notification.type == NotificationType.TRADE_ENTRY:
                return await self.notifier.send_trade_notification_async(notification.data)
            elif notification.type == NotificationType.TRADE_EXIT:
                return await self.notifier.send_trade_update_async(notification.data)
            elif notification.type == NotificationType.ERROR:
                return await self.notifier.send_error_notification_async(notification.message)
            else:
                # Generic message
                return await self.notifier.send_message_async(notification.message)
        except Exception as e:
            self.logger.error(f"Failed to send Telegram notification: {e}")
            return False


class EmailChannel(NotificationChannel):
    """Email notification channel"""

    def __init__(self, api_key: str, sender_email: str, receiver_email: str):
        super().__init__("email")
        self.notifier = EmailNotifier()
        self.sender_email = sender_email
        self.receiver_email = receiver_email

    async def send(self, notification: Notification) -> bool:
        """Send notification via email"""
        try:
            loop = asyncio.get_event_loop()
            attachments = None
            if notification.data and "attachments" in notification.data:
                attachments = notification.data["attachments"]
            await loop.run_in_executor(
                None,
                self.notifier.send_email,
                self.receiver_email,
                notification.title,
                notification.message,
                None,  # from_name
                attachments
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")
            return False


class AsyncNotificationManager:
    """
    Async notification manager with queuing, batching, and retry mechanisms.
    """

    def __init__(self,
                 telegram_token: Optional[str] = None,
                 telegram_chat_id: Optional[str] = None,
                 email_api_key: Optional[str] = None,
                 email_sender: Optional[str] = None,
                 email_receiver: Optional[str] = None,
                 batch_size: int = 10,
                 batch_timeout: float = 30.0,
                 max_queue_size: int = 1000):
        """
        Initialize the notification manager.

        Args:
            telegram_token: Telegram bot token
            telegram_chat_id: Telegram chat ID
            email_api_key: SendGrid API key
            email_sender: Sender email address
            email_receiver: Receiver email address
            batch_size: Number of notifications to batch
            batch_timeout: Timeout for batching in seconds
            max_queue_size: Maximum queue size
        """
        self.logger = setup_logger(__name__)

        # Initialize channels
        self.channels: Dict[str, NotificationChannel] = {}

        if telegram_token and telegram_chat_id:
            self.channels["telegram"] = TelegramChannel(telegram_token, telegram_chat_id)

        if email_api_key and email_sender and email_receiver:
            self.channels["email"] = EmailChannel(email_api_key, email_sender, email_receiver)

        # Configuration
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.max_queue_size = max_queue_size

        # Queues and state
        self.notification_queue = asyncio.Queue(maxsize=max_queue_size)
        self.batch_queue = asyncio.Queue(maxsize=max_queue_size)
        self.worker_task: Optional[asyncio.Task] = None
        self.batch_worker_task: Optional[asyncio.Task] = None
        self.running = False

        # Statistics
        self.stats = {
            "sent": 0,
            "failed": 0,
            "queued": 0,
            "batched": 0
        }

        # Rate limiting
        self.rate_limits = {
            "telegram": {"last_sent": 0, "min_interval": 1.0},  # 1 second between messages
            "email": {"last_sent": 0, "min_interval": 5.0}      # 5 seconds between emails
        }

    async def start(self):
        """Start the notification manager"""
        if self.running:
            return

        self.running = True
        self.worker_task = asyncio.create_task(self._notification_worker())
        self.batch_worker_task = asyncio.create_task(self._batch_worker())

        self.logger.info("Async notification manager started")

    async def stop(self):
        """Stop the notification manager"""
        if not self.running:
            return

        self.running = False

        # Cancel worker tasks
        if self.worker_task:
            self.worker_task.cancel()
        if self.batch_worker_task:
            self.batch_worker_task.cancel()

        # Wait for tasks to complete
        try:
            if self.worker_task:
                await self.worker_task
            if self.batch_worker_task:
                await self.batch_worker_task
        except asyncio.CancelledError:
            pass

        self.logger.info("Async notification manager stopped")

    async def send_notification(self,
                              notification_type: NotificationType,
                              title: str,
                              message: str,
                              priority: NotificationPriority = NotificationPriority.NORMAL,
                              data: Optional[Dict[str, Any]] = None,
                              source: str = "trading_bot",
                              channels: Optional[List[str]] = None,
                              attachments: Optional[list] = None) -> bool:
        """
        Send a notification asynchronously.

        Args:
            notification_type: Type of notification
            title: Notification title
            message: Notification message
            priority: Notification priority
            data: Additional data for the notification
            source: Source of the notification
            channels: Specific channels to use (None for all enabled channels)
            attachments: List of attachments to include with the notification

        Returns:
            True if notification was queued successfully
        """
        try:
            # Add attachments to data if provided
            if attachments:
                if data is None:
                    data = {}
                data["attachments"] = attachments
            notification = Notification(
                type=notification_type,
                priority=priority,
                title=title,
                message=message,
                data=data or {},
                source=source
            )

            # Add to queue
            await self.notification_queue.put(notification)
            self.stats["queued"] += 1

            return True

        except asyncio.QueueFull:
            self.logger.warning("Notification queue is full, dropping notification")
            return False
        except Exception as e:
            self.logger.error(f"Error queuing notification: {e}")
            return False

    async def send_trade_notification(self,
                                    symbol: str,
                                    side: str,
                                    price: float,
                                    quantity: float,
                                    entry_price: Optional[float] = None,
                                    pnl: Optional[float] = None,
                                    exit_type: Optional[str] = None) -> bool:
        """
        Send a trade notification.

        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            price: Trade price
            quantity: Trade quantity
            entry_price: Entry price (for exits)
            pnl: Profit/loss percentage
            exit_type: Exit type (TP/SL)

        Returns:
            True if notification was queued successfully
        """
        if side.upper() == "BUY":
            notification_type = NotificationType.TRADE_ENTRY
            title = f"Buy Order: {symbol}"
            message = f"Buy {quantity} {symbol} at {price}"
        else:
            notification_type = NotificationType.TRADE_EXIT
            title = f"Sell Order: {symbol}"
            message = f"Sell {quantity} {symbol} at {price}"
            if pnl is not None:
                message += f" (PnL: {pnl:.2f}%)"
            if exit_type:
                message += f" ({exit_type})"

        data = {
            "symbol": symbol,
            "side": side.upper(),
            "price": price,
            "quantity": quantity,
            "entry_price": entry_price,
            "pnl": pnl,
            "exit_type": exit_type,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        return await self.send_notification(
            notification_type=notification_type,
            title=title,
            message=message,
            priority=NotificationPriority.HIGH,
            data=data,
            source="trading_bot"
        )

    async def send_error_notification(self,
                                    error_message: str,
                                    source: str = "trading_bot") -> bool:
        """
        Send an error notification.

        Args:
            error_message: Error message
            source: Source of the error

        Returns:
            True if notification was queued successfully
        """
        return await self.send_notification(
            notification_type=NotificationType.ERROR,
            title="Error Alert",
            message=error_message,
            priority=NotificationPriority.CRITICAL,
            source=source
        )

    async def _notification_worker(self):
        """Worker task for processing notifications"""
        while self.running:
            try:
                # Get notification from queue
                notification = await asyncio.wait_for(
                    self.notification_queue.get(),
                    timeout=1.0
                )

                # Process notification
                await self._process_notification(notification)

                # Mark as done
                self.notification_queue.task_done()

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in notification worker: {e}")

    async def _batch_worker(self):
        """Worker task for processing batched notifications"""
        batch: List[Notification] = []
        last_batch_time = time.time()

        while self.running:
            try:
                # Try to get notification from batch queue
                try:
                    notification = await asyncio.wait_for(
                        self.batch_queue.get(),
                        timeout=0.1
                    )
                    batch.append(notification)
                    self.batch_queue.task_done()
                except asyncio.TimeoutError:
                    pass

                # Check if we should process the batch
                current_time = time.time()
                should_process = (
                    len(batch) >= self.batch_size or
                    (batch and current_time - last_batch_time >= self.batch_timeout)
                )

                if should_process and batch:
                    await self._process_batch(batch)
                    batch = []
                    last_batch_time = current_time

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in batch worker: {e}")

    async def _process_notification(self, notification: Notification):
        """Process a single notification"""
        # Check if notification should be batched
        if self._should_batch(notification):
            try:
                await self.batch_queue.put(notification)
                self.stats["batched"] += 1
                return
            except asyncio.QueueFull:
                self.logger.warning("Batch queue is full, processing immediately")

        # Send to all enabled channels
        tasks = []
        for channel_name, channel in self.channels.items():
            if channel.is_enabled():
                # Check rate limiting
                if self._check_rate_limit(channel_name):
                    task = asyncio.create_task(self._send_to_channel(channel, notification))
                    tasks.append(task)

        # Wait for all sends to complete
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = sum(1 for result in results if result is True)

            if success_count > 0:
                self.stats["sent"] += 1
            else:
                self.stats["failed"] += 1
                await self._handle_failed_notification(notification)

    async def _process_batch(self, batch: List[Notification]):
        """Process a batch of notifications"""
        # Group notifications by type and priority
        grouped = defaultdict(list)
        for notification in batch:
            key = (notification.type, notification.priority)
            grouped[key].append(notification)

        # Process each group
        for (notification_type, priority), notifications in grouped.items():
            # Create aggregated message
            aggregated = self._aggregate_notifications(notifications)

            # Send aggregated notification
            await self._process_notification(aggregated)

    def _should_batch(self, notification: Notification) -> bool:
        """Check if notification should be batched"""
        # Don't batch critical notifications
        if notification.priority == NotificationPriority.CRITICAL:
            return False

        # Don't batch trade notifications (they're important)
        if notification.type in [NotificationType.TRADE_ENTRY, NotificationType.TRADE_EXIT]:
            return False

        # Batch other notifications
        return True

    def _aggregate_notifications(self, notifications: List[Notification]) -> Notification:
        """Aggregate multiple notifications into one"""
        if not notifications:
            return notifications[0]

        # Use the highest priority
        max_priority = max(n.priority for n in notifications)

        # Create aggregated message
        titles = [n.title for n in notifications]
        messages = [n.message for n in notifications]

        aggregated_title = f"Batch Update ({len(notifications)} notifications)"
        aggregated_message = "\n\n".join([
            f"**{title}**\n{message}"
            for title, message in zip(titles, messages)
        ])

        return Notification(
            type=notifications[0].type,
            priority=max_priority,
            title=aggregated_title,
            message=aggregated_message,
            source="notification_manager"
        )

    def _check_rate_limit(self, channel_name: str) -> bool:
        """Check if channel is rate limited"""
        if channel_name not in self.rate_limits:
            return True

        limit = self.rate_limits[channel_name]
        current_time = time.time()

        if current_time - limit["last_sent"] < limit["min_interval"]:
            return False

        limit["last_sent"] = current_time
        return True

    async def _send_to_channel(self, channel: NotificationChannel, notification: Notification) -> bool:
        """Send notification to a specific channel"""
        try:
            return await channel.send(notification)
        except Exception as e:
            self.logger.error(f"Error sending to channel {channel.name}: {e}")
            return False

    async def _handle_failed_notification(self, notification: Notification):
        """Handle failed notification (retry logic)"""
        if notification.retry_count < notification.max_retries:
            notification.retry_count += 1
            # Exponential backoff
            delay = 2 ** notification.retry_count
            await asyncio.sleep(delay)

            # Re-queue for retry
            try:
                await self.notification_queue.put(notification)
            except asyncio.QueueFull:
                self.logger.error("Queue full, cannot retry notification")
        else:
            self.logger.error(f"Notification failed after {notification.max_retries} retries")

    def get_stats(self) -> Dict[str, Any]:
        """Get notification statistics"""
        return {
            **self.stats,
            "queue_size": self.notification_queue.qsize(),
            "batch_queue_size": self.batch_queue.qsize(),
            "enabled_channels": [name for name, channel in self.channels.items() if channel.is_enabled()]
        }

    def enable_channel(self, channel_name: str):
        """Enable a notification channel"""
        if channel_name in self.channels:
            self.channels[channel_name].enable()

    def disable_channel(self, channel_name: str):
        """Disable a notification channel"""
        if channel_name in self.channels:
            self.channels[channel_name].disable()


# Global notification manager instance
_notification_manager: Optional[AsyncNotificationManager] = None


def get_notification_manager() -> Optional[AsyncNotificationManager]:
    """Get the global notification manager instance"""
    return _notification_manager


async def initialize_notification_manager(**kwargs) -> AsyncNotificationManager:
    """Initialize the global notification manager"""
    global _notification_manager

    if _notification_manager is None:
        _notification_manager = AsyncNotificationManager(**kwargs)
        await _notification_manager.start()

    return _notification_manager


async def send_notification(notification_type: NotificationType,
                          title: str,
                          message: str,
                          **kwargs) -> bool:
    """Send a notification using the global manager"""
    manager = get_notification_manager()
    if manager is None:
        _logger.warning("Notification manager not initialized")
        return False

    return await manager.send_notification(notification_type, title, message, **kwargs)


async def send_trade_notification(**kwargs) -> bool:
    """Send a trade notification using the global manager"""
    manager = get_notification_manager()
    if manager is None:
        _logger.warning("Notification manager not initialized")
        return False

    return await manager.send_trade_notification(**kwargs)


async def send_error_notification(error_message: str, **kwargs) -> bool:
    """Send an error notification using the global manager"""
    manager = get_notification_manager()
    if manager is None:
        _logger.warning("Notification manager not initialized")
        return False

    return await manager.send_error_notification(error_message, **kwargs)
