"""
Compatibility Layer for AsyncNotificationManager

Provides backward compatibility for existing code that uses AsyncNotificationManager
by wrapping the new NotificationServiceClient with the same interface.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

from src.model.notification import NotificationType, NotificationPriority
from src.notification.client import NotificationServiceClient, NotificationServiceError
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class AsyncNotificationManagerCompat:
    """
    Compatibility wrapper for AsyncNotificationManager.

    Provides the same interface as the original AsyncNotificationManager
    but uses the new NotificationServiceClient under the hood.
    """

    def __init__(
        self,
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
        batch_queue: Optional[Any] = None,
        notification_service_url: str = "http://localhost:8080"
    ):
        """
        Initialize the compatibility wrapper.

        Args:
            telegram_token: Telegram bot token (ignored - configured in service)
            telegram_chat_id: Default Telegram chat ID
            email_api_key: Email API key (ignored - configured in service)
            email_sender: Sender email (ignored - configured in service)
            email_receiver: Default receiver email
            batch_size: Batch size (ignored - handled by service)
            batch_timeout: Batch timeout (ignored - handled by service)
            max_queue_size: Max queue size (ignored - handled by service)
            channels: Channels dict (ignored - configured in service)
            notification_queue: Notification queue (ignored)
            batch_queue: Batch queue (ignored)
            notification_service_url: URL of the notification service
        """
        self.telegram_chat_id = telegram_chat_id
        self.email_receiver = email_receiver
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.max_queue_size = max_queue_size
        self.running = False

        # Initialize notification service client
        self.client = NotificationServiceClient(base_url=notification_service_url)

        # Mock stats for compatibility
        self.stats = {"sent": 0, "failed": 0, "queued": 0, "batched": 0}

        # Mock channels for compatibility
        self.channels = {
            "telegram": MockChannel("telegram", True),
            "email": MockChannel("email", True)
        }

    async def start(self):
        """Start the notification manager (compatibility method)."""
        self.running = True
        _logger.info("AsyncNotificationManager compatibility layer started")

    async def stop(self):
        """Stop the notification manager (compatibility method)."""
        self.running = False
        await self.client.close_async()
        _logger.info("AsyncNotificationManager compatibility layer stopped")

    async def send_notification(
        self,
        notification_type: NotificationType,
        title: str,
        message: str,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        data: Optional[Dict[str, Any]] = None,
        source: str = "trading_bot",
        channels: Optional[List[str]] = None,
        attachments: Optional[dict] = None,
        email_receiver: Optional[str] = None,
        reply_to_message_id: Optional[int] = None,
        telegram_chat_id: Optional[int] = None
    ) -> bool:
        """
        Send a notification asynchronously.

        Args:
            notification_type: Type of notification
            title: Notification title
            message: Notification message
            priority: Notification priority
            data: Additional data for the notification
            source: Source of the notification
            channels: Specific channels to use
            attachments: Attachments to include
            email_receiver: Receiver email address
            reply_to_message_id: Reply to message ID for Telegram
            telegram_chat_id: Telegram chat ID

        Returns:
            True if notification was sent successfully
        """
        try:
            # Prepare metadata
            metadata = data or {}
            metadata["source"] = source

            # Handle Telegram-specific parameters
            if reply_to_message_id is not None:
                metadata["reply_to_message_id"] = reply_to_message_id

            # Determine recipient
            recipient_id = None
            if telegram_chat_id is not None:
                metadata["telegram_chat_id"] = telegram_chat_id
                recipient_id = str(telegram_chat_id)
            elif email_receiver:
                recipient_id = email_receiver
            elif self.telegram_chat_id:
                metadata["telegram_chat_id"] = self.telegram_chat_id
                recipient_id = str(self.telegram_chat_id)
            elif self.email_receiver:
                recipient_id = self.email_receiver

            # Use specified channels or default to both
            if not channels:
                channels = ["telegram", "email"]

            # Send notification
            response = await self.client.send_notification_async(
                notification_type=notification_type,
                title=title,
                message=message,
                priority=priority,
                channels=channels,
                recipient_id=recipient_id,
                attachments=attachments,
                metadata=metadata
            )

            self.stats["sent"] += 1
            return True

        except NotificationServiceError as e:
            _logger.error("Failed to send notification via service: %s", e)
            self.stats["failed"] += 1
            return False
        except Exception as e:
            _logger.exception("Unexpected error sending notification: %s", e)
            self.stats["failed"] += 1
            return False

    async def send_trade_notification(
        self,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        entry_price: Optional[float] = None,
        pnl: Optional[float] = None,
        exit_type: Optional[str] = None
    ) -> bool:
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
            True if notification was sent successfully
        """
        try:
            response = await self.client.send_trade_notification_async(
                symbol=symbol,
                side=side,
                price=price,
                quantity=quantity,
                entry_price=entry_price,
                pnl=pnl,
                exit_type=exit_type,
                recipient_id=self.telegram_chat_id or self.email_receiver
            )

            self.stats["sent"] += 1
            return True

        except NotificationServiceError as e:
            _logger.error("Failed to send trade notification via service: %s", e)
            self.stats["failed"] += 1
            return False
        except Exception as e:
            _logger.exception("Unexpected error sending trade notification: %s", e)
            self.stats["failed"] += 1
            return False

    async def send_error_notification(
        self,
        error_message: str,
        source: str = "trading_bot"
    ) -> bool:
        """
        Send an error notification.

        Args:
            error_message: Error message
            source: Source of the error

        Returns:
            True if notification was sent successfully
        """
        try:
            response = await self.client.send_error_notification_async(
                error_message=error_message,
                source=source,
                recipient_id=self.telegram_chat_id or self.email_receiver
            )

            self.stats["sent"] += 1
            return True

        except NotificationServiceError as e:
            _logger.error("Failed to send error notification via service: %s", e)
            self.stats["failed"] += 1
            return False
        except Exception as e:
            _logger.exception("Unexpected error sending error notification: %s", e)
            self.stats["failed"] += 1
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get notification statistics."""
        return {
            **self.stats,
            "queue_size": 0,  # Always 0 since we don't queue locally
            "batch_queue_size": 0,  # Always 0 since we don't batch locally
            "enabled_channels": [name for name, channel in self.channels.items() if channel.is_enabled()]
        }

    def enable_channel(self, channel_name: str):
        """Enable a notification channel."""
        if channel_name in self.channels:
            self.channels[channel_name].enable()

    def disable_channel(self, channel_name: str):
        """Disable a notification channel."""
        if channel_name in self.channels:
            self.channels[channel_name].disable()


class MockChannel:
    """Mock channel for compatibility."""

    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled

    def is_enabled(self) -> bool:
        return self.enabled

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False


# Global compatibility instance
_notification_manager: Optional[AsyncNotificationManagerCompat] = None


def get_notification_manager() -> Optional[AsyncNotificationManagerCompat]:
    """Get the global notification manager instance."""
    return _notification_manager


async def initialize_notification_manager(**kwargs) -> AsyncNotificationManagerCompat:
    """Initialize the global notification manager."""
    global _notification_manager

    if _notification_manager is None:
        _notification_manager = AsyncNotificationManagerCompat(**kwargs)
        await _notification_manager.start()

    return _notification_manager


async def send_notification(
    notification_type: NotificationType,
    title: str,
    message: str,
    **kwargs
) -> bool:
    """Send a notification using the global manager."""
    manager = get_notification_manager()
    if manager is None:
        _logger.warning("Notification manager not initialized")
        return False

    return await manager.send_notification(notification_type, title, message, **kwargs)


async def send_trade_notification(**kwargs) -> bool:
    """Send a trade notification using the global manager."""
    manager = get_notification_manager()
    if manager is None:
        _logger.warning("Notification manager not initialized")
        return False

    return await manager.send_trade_notification(**kwargs)


async def send_error_notification(error_message: str, **kwargs) -> bool:
    """Send an error notification using the global manager."""
    manager = get_notification_manager()
    if manager is None:
        _logger.warning("Notification manager not initialized")
        return False

    return await manager.send_error_notification(error_message, **kwargs)