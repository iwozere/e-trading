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
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from collections import defaultdict
import os
from email.mime.image import MIMEImage
from email.mime.application import MIMEApplication

from src.model.notification import Notification, NotificationPriority, NotificationType
from src.notification.logger import setup_logger
from src.notification.emailer import EmailNotifier
from aiogram import Bot

_logger = setup_logger(__name__)




class NotificationChannel:
    """Base class for notification channels"""

    def __init__(self, name: str):
        self.name = name
        self.enabled = True

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
    """Telegram notification channel using aiogram directly"""

    def __init__(self, token: str, chat_id: str):
        super().__init__("telegram")
        self.bot = Bot(token=token)
        self.chat_id = chat_id

    async def send(self, notification: Notification) -> bool:
        """Send notification via Telegram using aiogram"""
        try:
            # Use dynamic chat_id if provided, otherwise fall back to default
            target_chat_id = notification.data.get('telegram_chat_id') if notification.data else None
            if target_chat_id is None:
                target_chat_id = self.chat_id
                _logger.debug("Using default chat_id: %s", target_chat_id)
            else:
                _logger.debug("Using dynamic chat_id: %s", target_chat_id)

            # If attachments are present, send as photo
            attachments = None
            if notification.data and "attachments" in notification.data:
                attachments = notification.data["attachments"]
            if attachments:
                for filename, value in attachments.items():
                    if isinstance(value, bytes):
                        from aiogram.types import BufferedInputFile
                        # If message is too long for caption, send text first
                        # Get reply_to_message_id, but only use it if it's valid
                        reply_to_message_id = notification.data.get('reply_to_message_id')

                        if len(notification.message) > 1024:
                            # Try to send with reply first, fall back to regular message if reply fails
                            try:
                                await self.bot.send_message(
                                    chat_id=target_chat_id,
                                    text=notification.message,
                                    parse_mode=None,
                                    reply_to_message_id=reply_to_message_id
                                )
                            except Exception as reply_error:
                                _logger.warning("Failed to send message with reply, sending without reply: %s", reply_error)
                                await self.bot.send_message(
                                    chat_id=target_chat_id,
                                    text=notification.message,
                                    parse_mode=None
                                )

                            try:
                                await self.bot.send_photo(
                                    chat_id=target_chat_id,
                                    photo=BufferedInputFile(value, filename=filename),
                                    caption=f"Chart for {filename}",
                                    parse_mode=None,
                                    reply_to_message_id=reply_to_message_id
                                )
                            except Exception as reply_error:
                                _logger.warning("Failed to send photo with reply, sending without reply: %s", reply_error)
                                await self.bot.send_photo(
                                    chat_id=target_chat_id,
                                    photo=BufferedInputFile(value, filename=filename),
                                    caption=f"Chart for {filename}",
                                    parse_mode=None
                                )
                        else:
                            try:
                                await self.bot.send_photo(
                                    chat_id=target_chat_id,
                                    photo=BufferedInputFile(value, filename=filename),
                                    caption=notification.message,
                                    parse_mode=None,
                                    reply_to_message_id=reply_to_message_id
                                )
                            except Exception as reply_error:
                                _logger.warning("Failed to send photo with reply, sending without reply: %s", reply_error)
                                await self.bot.send_photo(
                                    chat_id=target_chat_id,
                                    photo=BufferedInputFile(value, filename=filename),
                                    caption=notification.message,
                                    parse_mode=None
                                )
                        return True
                    elif isinstance(value, str):
                         from aiogram.types import FSInputFile
                         # Get reply_to_message_id, but only use it if it's valid
                         reply_to_message_id = notification.data.get('reply_to_message_id')

                         if len(notification.message) > 1024:
                             # Try to send with reply first, fall back to regular message if reply fails
                             try:
                                 await self.bot.send_message(
                                     chat_id=target_chat_id,
                                     text=notification.message,
                                     parse_mode=None,
                                     reply_to_message_id=reply_to_message_id
                                 )
                             except Exception as reply_error:
                                 _logger.warning("Failed to send message with reply, sending without reply: %s", reply_error)
                                 await self.bot.send_message(
                                     chat_id=self.chat_id,
                                     text=notification.message,
                                     parse_mode=None
                                 )

                             try:
                                 await self.bot.send_photo(
                                     chat_id=target_chat_id,
                                     photo=FSInputFile(value, filename=filename),
                                     caption=f"Chart for {filename}",
                                     parse_mode=None,
                                     reply_to_message_id=reply_to_message_id
                                 )
                             except Exception as reply_error:
                                 _logger.warning("Failed to send photo with reply, sending without reply: %s", reply_error)
                                 await self.bot.send_photo(
                                     chat_id=self.chat_id,
                                     photo=FSInputFile(value, filename=filename),
                                     caption=f"Chart for {filename}",
                                     parse_mode=None
                                 )
                         else:
                             try:
                                 await self.bot.send_photo(
                                     chat_id=target_chat_id,
                                     photo=FSInputFile(value, filename=filename),
                                     caption=notification.message,
                                     parse_mode=None,
                                     reply_to_message_id=reply_to_message_id
                                 )
                             except Exception as reply_error:
                                 _logger.warning("Failed to send photo with reply, sending without reply: %s", reply_error)
                                 await self.bot.send_photo(
                                     chat_id=self.chat_id,
                                     photo=FSInputFile(value, filename=filename),
                                     caption=notification.message,
                                     parse_mode=None
                                 )
                         return True
                # If no valid attachment, fall through to text
            # Get reply_to_message_id, but only use it if it's valid
            reply_to_message_id = notification.data.get('reply_to_message_id')

            # Try to send with reply first, fall back to regular message if reply fails
            try:
                await self.bot.send_message(
                    chat_id=target_chat_id,
                    text=notification.message,
                    parse_mode=None,
                    reply_to_message_id=reply_to_message_id
                )
            except Exception as reply_error:
                # If reply fails, send without reply
                _logger.warning("Failed to send message with reply, sending without reply: %s", reply_error)
                await self.bot.send_message(
                    chat_id=target_chat_id,
                    text=notification.message,
                    parse_mode=None
                )
            return True
        except Exception as e:
            _logger.exception("Failed to send Telegram notification: %s")
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
            channels = notification.data.get("channels") if notification.data else None
            if channels and "email" not in channels:
                return False
            _logger.debug("EmailChannel.send called for %s, subject: %s", self.receiver_email, notification.title)
            loop = asyncio.get_event_loop()
            attachments = notification.data.get("attachments", {}) if notification.data else {}
            # Prepare attachments as (filename, bytes) pairs
            prepared_attachments = []
            for filename, value in attachments.items():
                if isinstance(value, bytes):
                    if filename.lower().endswith('.png'):
                        prepared_attachments.append((filename, MIMEImage(value, name=filename)))
                    else:
                        prepared_attachments.append((filename, MIMEApplication(value, Name=filename)))
                elif isinstance(value, str):
                    try:
                        with open(value, "rb") as f:
                            file_bytes = f.read()
                            if filename.lower().endswith('.png'):
                                prepared_attachments.append((filename, MIMEImage(file_bytes, name=filename)))
                            else:
                                prepared_attachments.append((filename, MIMEApplication(file_bytes, Name=filename)))
                    except Exception as e:
                        _logger.exception("Failed to attach file %s: %s")
            # Format message as HTML
            html_message = notification.message.replace('\n', '<br>') if notification.message else ''
            await loop.run_in_executor(
                None,
                self.notifier.send_email_with_mime,
                self.receiver_email,
                notification.title,
                html_message,
                None,  # from_name
                prepared_attachments
            )
            return True
        except Exception as e:
            _logger.exception("Failed to send email notification: %s")
            return False


class AsyncNotificationManager:
    """
    Async notification manager with queuing, batching, and retry mechanisms.
    Now supports dependency injection for testability.
    """

    def __init__(self,
                 telegram_token: Optional[str] = None,
                 telegram_chat_id: Optional[str] = None,
                 email_api_key: Optional[str] = None,
                 email_sender: Optional[str] = None,
                 email_receiver: Optional[str] = None,
                 batch_size: int = 10,
                 batch_timeout: float = 30.0,
                 max_queue_size: int = 1000,
                 channels: Optional[Dict[str, Any]] = None,
                 notification_queue: Optional[Any] = None,
                 batch_queue: Optional[Any] = None):
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
            channels: Optional dict of channels for testability
            notification_queue, batch_queue: Optional custom queues for testability
        """
        self.channels: Dict[str, NotificationChannel] = channels or {}
        if not channels:
            if telegram_token and telegram_chat_id:
                self.channels["telegram"] = TelegramChannel(telegram_token, telegram_chat_id)
            if email_api_key and email_sender and email_receiver:
                self.channels["email"] = EmailChannel(email_api_key, email_sender, email_receiver)
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.max_queue_size = max_queue_size
        self.notification_queue = notification_queue or asyncio.Queue(maxsize=max_queue_size)
        self.batch_queue = batch_queue or asyncio.Queue(maxsize=max_queue_size)
        self.worker_task: Optional[asyncio.Task] = None
        self.batch_worker_task: Optional[asyncio.Task] = None
        self.running = False
        self.stats = {"sent": 0, "failed": 0, "queued": 0, "batched": 0}
        self.rate_limits = {
            "telegram": {"last_sent": 0, "min_interval": 1.0},
            "email": {"last_sent": 0, "min_interval": 5.0}
        }

    async def start(self):
        """Start the notification manager"""
        if self.running:
            return

        self.running = True
        self.worker_task = asyncio.create_task(self._notification_worker())
        self.batch_worker_task = asyncio.create_task(self._batch_worker())

        _logger.info("Async notification manager started")

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

        _logger.info("Async notification manager stopped")

    async def send_notification(self,
                              notification_type: NotificationType,
                              title: str,
                              message: str,
                              priority: NotificationPriority = NotificationPriority.NORMAL,
                              data: Optional[Dict[str, Any]] = None,
                              source: str = "trading_bot",
                              channels: Optional[List[str]] = None,
                              attachments: Optional[dict] = None,
                              email_receiver: Optional[str] = None,
                              reply_to_message_id: int = None,
                              telegram_chat_id: int = None) -> bool:
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
            email_receiver: Receiver email address for email notifications
            reply_to_message_id: Reply to message ID for Telegram messages
            telegram_chat_id: Telegram chat ID for Telegram messages

        Returns:
            True if notification was queued successfully
        """
        try:
            # Set the receiver email dynamically before sending
            _logger.debug("Start async send_notification %s", title)
            if email_receiver and "email" in self.channels:
                _logger.debug("Set email_receiver: %s", email_receiver)
                self.channels["email"].receiver_email = email_receiver

            # Add attachments to data if provided
            if attachments:
                if data is None:
                    data = {}
                data["attachments"] = attachments

            if reply_to_message_id is not None:
                if data is None:
                    data = {}
                data['reply_to_message_id'] = reply_to_message_id

            if telegram_chat_id is not None:
                if data is None:
                    data = {}
                data['telegram_chat_id'] = telegram_chat_id

            if channels is not None:
                if data is None:
                    data = {}
                data["channels"] = channels

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
            _logger.warning("Notification queue is full, dropping notification")
            return False
        except Exception as e:
            _logger.exception("Error queuing notification: %s")
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
                _logger.error("Error in notification worker: CancelledError")
                break
            except Exception as e:
                _logger.exception("Error in notification worker: %s")

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
                    _logger.debug("Notification added to batch: %s", notification.title)
                except asyncio.TimeoutError:
                    pass

                # Check if we should process the batch
                current_time = time.time()
                should_process = (
                    len(batch) >= self.batch_size or
                    (batch and current_time - last_batch_time >= self.batch_timeout)
                )

                if should_process and batch:
                    _logger.debug("Processing batch of size %d", len(batch))
                    await self._process_batch(batch)
                    batch = []
                    last_batch_time = current_time

            except asyncio.CancelledError:
                break
            except Exception as e:
                _logger.exception("Error in batch worker: %s")

    async def _process_notification(self, notification: Notification):
        """Process a single notification"""
        _logger.debug("Channels: %s", self.channels)
        _logger.debug("Email channel enabled: %s", 'email' in self.channels and self.channels['email'].is_enabled())
        _logger.debug("Notification channels: %s", getattr(notification, 'channels', None))

        # Check if notification should be batched
        if self._should_batch(notification):
            try:
                _logger.debug("Enqueue the message")
                await self.batch_queue.put(notification)
                self.stats["batched"] += 1
                return
            except asyncio.QueueFull:
                _logger.warning("Batch queue is full, processing immediately")

        # Send to all enabled channels
        _logger.debug("Channels: %s", self.channels)
        _logger.debug("Email channel enabled: %s", 'email' in self.channels and self.channels['email'].is_enabled())
        tasks = []
        for channel_name, channel in self.channels.items():
            channels_list = notification.data.get("channels") if notification.data else None
            if channels_list and channel_name not in channels_list:
                continue
            _logger.debug("Considering channel: %s, enabled: %s", channel_name, channel.is_enabled())
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
        _logger.debug("Start _process_batch")
        for notification in batch:
            _logger.debug("_process_batch: _process_notification")
            # Mark as from_batch to prevent re-batching
            if notification.data is not None:
                notification.data["from_batch"] = True
            else:
                notification.data = {"from_batch": True}
            await self._process_notification(notification)

    def _should_batch(self, notification: Notification) -> bool:
        """Check if notification should be batched"""
        # Don't batch if already from batch
        if notification.data and notification.data.get("from_batch"):
            return False
        # Don't batch critical notifications
        if notification.priority == NotificationPriority.CRITICAL:
            return False
        # Don't batch trade notifications (they're important)
        if notification.type in [NotificationType.TRADE_ENTRY, NotificationType.TRADE_EXIT]:
            return False
        # Only batch if 'email' in channels and 'telegram' not in channels
        channels = notification.data.get("channels") if notification.data else None
        if channels:
            if "telegram" in channels:
                return False
            if "email" in channels:
                return True
        return False

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
            _logger.exception("Error sending to channel %s: %s")
            return False

    def _create_notification(self, notification_type: NotificationType, title: str, message: str, data: Optional[Dict[str, Any]] = None) -> Notification:
        """Create a notification object"""
        return Notification(
            type=notification_type,
            priority=NotificationPriority.NORMAL,
            title=title,
            message=message,
            data=data or {},
            source="trading_bot"
        )

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
                _logger.error("Queue full, cannot retry notification")
        else:
            _logger.error("Notification failed after %s retries", notification.max_retries)

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
        _logger.debug("Start initialize_notification_manager")
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

    # Just before sending the email
    if "email" in manager.channels:
        manager.channels["email"].receiver_email = kwargs.get("email_receiver")

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
