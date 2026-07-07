# src/data/db/services/notification_service.py
"""
Notification Service

Service layer for notification operations.
Provides high-level business logic for notification management.
"""

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

from sqlalchemy import Text, cast, desc, func, or_

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.data.db.models.model_notification import Message, MessageDeliveryStatus, MessageStatus
from src.data.db.services.base_service import BaseDBService, handle_db_error, with_uow
from src.notification.channels.base import ChannelHealth, DeliveryStatus
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class NotificationService(BaseDBService):
    """Service layer for notification operations."""

    def __init__(self, db_service=None):
        """Initialize the service."""
        super().__init__(db_service)

    @with_uow
    @handle_db_error
    def create_message(self, message_data: Dict[str, Any]) -> Message:
        """Create a new notification message."""
        # Validate required fields
        required_fields = ["message_type", "channels", "recipient_id", "content"]
        for field in required_fields:
            if field not in message_data:
                raise ValueError(f"Missing required field: {field}")

        # Set defaults
        if "priority" not in message_data:
            message_data["priority"] = "NORMAL"

        if "status" not in message_data:
            message_data["status"] = MessageStatus.PENDING.value

        if "scheduled_for" not in message_data:
            message_data["scheduled_for"] = datetime.now(UTC)

        if "max_retries" not in message_data:
            message_data["max_retries"] = 3

        if "retry_count" not in message_data:
            message_data["retry_count"] = 0

        # Create message
        message = self.repos.notifications.create_message(message_data)

        # Create delivery status records for each channel
        for channel in message_data["channels"]:
            delivery_data = {
                "message_id": message.id,
                "channel": channel,
                "status": DeliveryStatus.PENDING.value,
                "created_at": datetime.now(UTC),
            }
            self.repos.notifications.create_delivery_status(delivery_data)

        self._logger.info("Created message %s with %d delivery channels", message.id, len(message_data["channels"]))
        return message

    @with_uow
    @handle_db_error
    def get_message(self, message_id: int) -> Message | None:
        """Get a message by ID."""
        return self.repos.notifications.get_message(message_id)

    @with_uow
    @handle_db_error
    def list_messages(
        self,
        status: str | None = None,
        priority: str | None = None,
        recipient_id: str | None = None,
        message_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Message]:
        """List messages with filtering."""
        # Convert string status to enum if provided
        status_enum = None
        if status:
            try:
                status_enum = MessageStatus(status.upper())
            except ValueError:
                self._logger.warning("Invalid status filter: %s", status)

        # Convert string priority to enum if provided
        from src.data.db.models.model_notification import MessagePriority

        priority_enum = None
        if priority:
            try:
                priority_enum = MessagePriority(priority.upper())
            except ValueError:
                self._logger.warning("Invalid priority filter: %s", priority)

        return self.repos.notifications.list_messages(
            status=status_enum,
            priority=priority_enum,
            recipient_id=recipient_id,
            message_type=message_type,
            limit=limit,
            offset=offset,
        )

    @with_uow
    @handle_db_error
    def search_messages(
        self,
        recipient_id: str | None = None,
        search: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        status: str | None = None,
        channel: str | None = None,
        days: int = 30,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        Search sent messages with flexible filtering for the admin UI.

        Args:
            recipient_id: Filter by recipient (case-insensitive substring match).
            search: Case-insensitive substring matched against the message content,
                message type and template name.
            start_date: Only include messages created at or after this time.
                Defaults to ``days`` ago when not provided.
            end_date: Only include messages created at or before this time.
            status: Filter by message status (e.g. DELIVERED, FAILED).
            channel: Filter by delivery channel (e.g. telegram, email).
            days: Look-back window in days used when ``start_date`` is omitted.
            limit: Maximum number of messages to return (capped at 1000).
            offset: Number of messages to skip for pagination.

        Returns:
            Dictionary with ``total`` (matching row count), ``items`` (list of
            message dicts), ``limit`` and ``offset``.
        """
        # Default to the last `days` days when no explicit start is given.
        if start_date is None:
            start_date = datetime.now(UTC) - timedelta(days=days)

        limit = max(1, min(limit, 1000))
        offset = max(0, offset)

        session = self.repos.s
        query = session.query(Message).filter(Message.created_at >= start_date)

        if end_date is not None:
            query = query.filter(Message.created_at <= end_date)

        if recipient_id:
            query = query.filter(Message.recipient_id.ilike(f"%{recipient_id}%"))

        if status:
            query = query.filter(Message.status == status.upper())

        if channel:
            query = query.filter(Message.channels.contains([channel]))

        if search:
            pattern = f"%{search}%"
            query = query.filter(
                or_(
                    cast(Message.content, Text).ilike(pattern),
                    Message.message_type.ilike(pattern),
                    Message.template_name.ilike(pattern),
                )
            )

        total = query.with_entities(func.count(Message.id)).scalar() or 0

        messages = query.order_by(desc(Message.created_at)).offset(offset).limit(limit).all()

        items = [
            {
                "id": msg.id,
                "message_type": msg.message_type,
                "priority": msg.priority,
                "channels": msg.channels,
                "recipient_id": msg.recipient_id,
                "template_name": msg.template_name,
                "content": msg.content,
                "metadata": msg.message_metadata,
                "status": msg.status,
                "created_at": msg.created_at.isoformat() if msg.created_at else None,
                "scheduled_for": msg.scheduled_for.isoformat() if msg.scheduled_for else None,
                "processed_at": msg.processed_at.isoformat() if msg.processed_at else None,
                "retry_count": msg.retry_count,
                "last_error": msg.last_error,
            }
            for msg in messages
        ]

        return {"total": total, "items": items, "limit": limit, "offset": offset}

    @with_uow
    @handle_db_error
    def update_message_status(
        self, message_id: int, status: str, error_message: str | None = None
    ) -> Message | None:
        """Update message status."""
        update_data = {"status": status, "processed_at": datetime.now(UTC)}

        if error_message:
            update_data["last_error"] = error_message

        message = self.repos.notifications.update_message(message_id, update_data)
        if message:
            self._logger.info("Updated message %s status to %s", message_id, status)

        return message

    @with_uow
    @handle_db_error
    def get_delivery_status(self, message_id: int) -> List[MessageDeliveryStatus]:
        """Get delivery status for all channels of a message."""
        return self.repos.notifications.get_delivery_statuses_by_message(message_id)

    @with_uow
    @handle_db_error
    def update_delivery_status(
        self,
        delivery_id: int,
        status: str,
        delivered_at: datetime | None = None,
        response_time_ms: int | None = None,
        error_message: str | None = None,
        external_id: str | None = None,
    ) -> MessageDeliveryStatus | None:
        """Update delivery status."""
        update_data: Dict[str, Any] = {"status": status}

        if delivered_at:
            update_data["delivered_at"] = delivered_at
        elif status == DeliveryStatus.DELIVERED.value:
            update_data["delivered_at"] = datetime.now(UTC)

        if response_time_ms is not None:
            update_data["response_time_ms"] = response_time_ms

        if error_message:
            update_data["error_message"] = error_message

        if external_id:
            update_data["external_id"] = external_id

        delivery_status = self.repos.notifications.update_delivery_status(delivery_id, update_data)
        if delivery_status:
            self._logger.info("Updated delivery status %s to %s", delivery_id, status)

        return delivery_status

    @with_uow
    @handle_db_error
    def get_channel_health(self) -> List[ChannelHealth]:
        """Get health status for all channels."""
        return self.repos.notifications.list_channel_health()

    @with_uow
    @handle_db_error
    def update_channel_health(self, channel: str, status: str, error_message: str | None = None) -> ChannelHealth:
        """Update channel health status."""
        health_data = {
            "channel": channel,
            "status": status,
            "error_message": error_message,
            "checked_at": datetime.now(UTC),
        }

        health = self.repos.notifications.create_or_update_channel_health(health_data)
        self._logger.info("Updated channel health for %s: %s", channel, status)
        return health

    @with_uow
    def get_delivery_statistics(self, channel: str | None = None, days: int = 30) -> Dict[str, Any]:
        """Get delivery statistics."""
        return self.repos.notifications.get_delivery_statistics(channel=channel, days=days)

    @with_uow
    @handle_db_error
    def cleanup_old_messages(self, days_to_keep: int = 30) -> int:
        """Clean up old delivered messages."""
        deleted_count = self.repos.notifications.cleanup_old_messages(days_to_keep)
        self._logger.info("Cleaned up %d old messages", deleted_count)
        return deleted_count

    @with_uow
    @handle_db_error
    def get_pending_messages(self, limit: int = 100) -> List[Message]:
        """Get pending messages ready for processing."""
        current_time = datetime.now(UTC)
        return self.repos.notifications.get_pending_messages(current_time, limit=limit)

    @with_uow
    @handle_db_error
    def get_failed_messages_for_retry(self, limit: int = 50, channels: List[str] | None = None) -> List[Message]:
        """Get failed messages that can be retried."""
        current_time = datetime.now(UTC)
        return self.repos.notifications.get_failed_messages_for_retry(current_time, limit=limit, channels=channels)

    @with_uow
    @handle_db_error
    def check_rate_limit(self, user_id: str, channel: str) -> bool:
        """Check if user is within rate limits for a channel."""
        # Default rate limit configuration
        default_config = {
            "max_tokens": 60,  # 60 messages per hour
            "refill_rate": 60,  # Refill 1 token per minute
        }

        return self.repos.notifications.check_and_consume_token(user_id, channel, default_config)
