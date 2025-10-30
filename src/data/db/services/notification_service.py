# src/data/db/services/notification_service.py
"""
Notification Service

Service layer for notification operations.
Provides high-level business logic for notification management.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.data.db.repos.repo_notification import NotificationRepository
from src.data.db.models.model_notification import (
    Message, MessageDeliveryStatus, MessageStatus
)
from src.notification.channels.base import (
    NotificationChannel, DeliveryResult, ChannelHealth, MessageContent,
    DeliveryStatus, ChannelHealthStatus
)

from src.notification.logger import setup_logger
_logger = setup_logger(__name__)


class NotificationService:
    """
    Service layer for notification operations.

    Provides high-level business logic for creating, managing, and tracking notifications.
    """

    def __init__(self, notification_repo: NotificationRepository):
        """
        Initialize the notification service.

        Args:
            notification_repo: Notification repository instance
        """
        self.repo = notification_repo

    def create_message(self, message_data: Dict[str, Any]) -> Message:
        """
        Create a new notification message.

        Args:
            message_data: Message data dictionary

        Returns:
            Created Message object

        Raises:
            ValueError: If message data is invalid
        """
        try:
            # Validate required fields
            required_fields = ['message_type', 'channels', 'recipient_id', 'content']
            for field in required_fields:
                if field not in message_data:
                    raise ValueError(f"Missing required field: {field}")

            # Set defaults
            if 'priority' not in message_data:
                message_data['priority'] = 'NORMAL'

            if 'status' not in message_data:
                message_data['status'] = MessageStatus.PENDING.value

            if 'scheduled_for' not in message_data:
                message_data['scheduled_for'] = datetime.now(timezone.utc)

            if 'max_retries' not in message_data:
                message_data['max_retries'] = 3

            if 'retry_count' not in message_data:
                message_data['retry_count'] = 0

            # Create message
            message = self.repo.messages.create_message(message_data)

            # Create delivery status records for each channel
            for channel in message_data['channels']:
                delivery_data = {
                    'message_id': message.id,
                    'channel': channel,
                    'status': DeliveryStatus.PENDING.value,
                    'created_at': datetime.now(timezone.utc)
                }
                self.repo.delivery_status.create_delivery_status(delivery_data)

            self.repo.commit()
            _logger.info("Created message %s with %d delivery channels", message.id, len(message_data['channels']))
            return message

        except Exception as e:
            self.repo.rollback()
            _logger.exception("Failed to create message:")
            raise

    def get_message(self, message_id: int) -> Optional[Message]:
        """
        Get a message by ID.

        Args:
            message_id: Message ID

        Returns:
            Message object or None if not found
        """
        return self.repo.messages.get_message(message_id)

    def list_messages(
        self,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        recipient_id: Optional[str] = None,
        message_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Message]:
        """
        List messages with filtering.

        Args:
            status: Filter by status
            priority: Filter by priority
            recipient_id: Filter by recipient ID
            message_type: Filter by message type
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of Message objects
        """
        # Convert string status to enum if provided
        status_enum = None
        if status:
            try:
                status_enum = MessageStatus(status.upper())
            except ValueError:
                _logger.warning("Invalid status filter: %s", status)

        # Convert string priority to enum if provided
        from src.data.db.models.model_notification import MessagePriority
        priority_enum = None
        if priority:
            try:
                priority_enum = MessagePriority(priority.upper())
            except ValueError:
                _logger.warning("Invalid priority filter: %s", priority)

        return self.repo.messages.list_messages(
            status=status_enum,
            priority=priority_enum,
            recipient_id=recipient_id,
            message_type=message_type,
            limit=limit,
            offset=offset
        )

    def update_message_status(self, message_id: int, status: str, error_message: Optional[str] = None) -> Optional[Message]:
        """
        Update message status.

        Args:
            message_id: Message ID
            status: New status
            error_message: Error message if status is FAILED

        Returns:
            Updated Message object or None if not found
        """
        try:
            update_data = {
                'status': status,
                'processed_at': datetime.now(timezone.utc)
            }

            if error_message:
                update_data['last_error'] = error_message

            message = self.repo.messages.update_message(message_id, update_data)
            if message:
                self.repo.commit()
                _logger.info("Updated message %s status to %s", message_id, status)

            return message

        except Exception as e:
            self.repo.rollback()
            _logger.exception("Failed to update message %s status:", message_id)
            raise

    def get_delivery_status(self, message_id: int) -> List[MessageDeliveryStatus]:
        """
        Get delivery status for all channels of a message.

        Args:
            message_id: Message ID

        Returns:
            List of MessageDeliveryStatus objects
        """
        return self.repo.delivery_status.get_delivery_statuses_by_message(message_id)

    def update_delivery_status(
        self,
        delivery_id: int,
        status: str,
        delivered_at: Optional[datetime] = None,
        response_time_ms: Optional[int] = None,
        error_message: Optional[str] = None,
        external_id: Optional[str] = None
    ) -> Optional[MessageDeliveryStatus]:
        """
        Update delivery status.

        Args:
            delivery_id: Delivery status ID
            status: New delivery status
            delivered_at: Delivery timestamp
            response_time_ms: Response time in milliseconds
            error_message: Error message if delivery failed
            external_id: External service message ID

        Returns:
            Updated MessageDeliveryStatus object or None if not found
        """
        try:
            update_data = {'status': status}

            if delivered_at:
                update_data['delivered_at'] = delivered_at
            elif status == DeliveryStatus.DELIVERED.value:
                update_data['delivered_at'] = datetime.now(timezone.utc)

            if response_time_ms is not None:
                update_data['response_time_ms'] = response_time_ms

            if error_message:
                update_data['error_message'] = error_message

            if external_id:
                update_data['external_id'] = external_id

            delivery_status = self.repo.delivery_status.update_delivery_status(delivery_id, update_data)
            if delivery_status:
                self.repo.commit()
                _logger.info("Updated delivery status %s to %s", delivery_id, status)

            return delivery_status

        except Exception as e:
            self.repo.rollback()
            _logger.exception("Failed to update delivery status %s:", delivery_id)
            raise

    def get_channel_health(self) -> List[ChannelHealth]:
        """
        Get health status for all channels.

        Returns:
            List of ChannelHealth objects
        """
        return self.repo.channel_health.list_channel_health()

    def update_channel_health(self, channel: str, status: str, error_message: Optional[str] = None) -> ChannelHealth:
        """
        Update channel health status.

        Args:
            channel: Channel name
            status: Health status
            error_message: Error message if unhealthy

        Returns:
            Updated ChannelHealth object
        """
        try:
            health_data = {
                'channel': channel,
                'status': status,
                'error_message': error_message,
                'checked_at': datetime.now(timezone.utc)
            }

            health = self.repo.channel_health.create_or_update_channel_health(health_data)
            self.repo.commit()
            _logger.info("Updated channel health for %s: %s", channel, status)
            return health

        except Exception as e:
            self.repo.rollback()
            _logger.exception("Failed to update channel health for %s:", channel)
            raise

    def get_delivery_statistics(self, channel: Optional[str] = None, days: int = 30) -> Dict[str, Any]:
        """
        Get delivery statistics.

        Args:
            channel: Filter by channel
            days: Number of days to analyze

        Returns:
            Dictionary with statistics
        """
        return self.repo.delivery_status.get_delivery_statistics(channel=channel, days=days)

    def cleanup_old_messages(self, days_to_keep: int = 30) -> int:
        """
        Clean up old delivered messages.

        Args:
            days_to_keep: Number of days of messages to keep

        Returns:
            Number of messages deleted
        """
        try:
            deleted_count = self.repo.messages.cleanup_old_messages(days_to_keep)
            self.repo.commit()
            _logger.info("Cleaned up %d old messages", deleted_count)
            return deleted_count

        except Exception as e:
            self.repo.rollback()
            _logger.exception("Failed to cleanup old messages:")
            raise

    def get_pending_messages(self, limit: int = 100) -> List[Message]:
        """
        Get pending messages ready for processing.

        Args:
            limit: Maximum number of messages to return

        Returns:
            List of pending Message objects
        """
        current_time = datetime.now(timezone.utc)
        return self.repo.messages.get_pending_messages(current_time, limit=limit)

    def get_failed_messages_for_retry(self, limit: int = 50) -> List[Message]:
        """
        Get failed messages that can be retried.

        Args:
            limit: Maximum number of messages to return

        Returns:
            List of failed Message objects ready for retry
        """
        current_time = datetime.now(timezone.utc)
        return self.repo.messages.get_failed_messages_for_retry(current_time, limit=limit)

    def check_rate_limit(self, user_id: str, channel: str) -> bool:
        """
        Check if user is within rate limits for a channel.

        Args:
            user_id: User ID
            channel: Channel name

        Returns:
            True if within limits, False if rate limited
        """
        # Default rate limit configuration
        default_config = {
            'max_tokens': 60,  # 60 messages per hour
            'refill_rate': 60  # Refill 1 token per minute
        }

        return self.repo.rate_limits.check_and_consume_token(user_id, channel, default_config)