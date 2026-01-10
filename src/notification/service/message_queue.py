"""
Notification Service Message Queue

Database-backed message queue with priority handling, validation, and scheduling.
Provides reliable message storage and retrieval for the notification service.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta, timezone
from enum import Enum
from dataclasses import dataclass

from src.data.db.models.model_notification import (
    Message, MessagePriority, MessageStatus
)
from src.data.db.services.database_service import get_database_service
from src.notification.logger import setup_logger
_logger = setup_logger(__name__)


class QueuePriority(Enum):
    """Queue priority levels."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


@dataclass
class QueuedMessage:
    """Queued message data structure."""
    id: int
    message_type: str
    priority: MessagePriority
    channels: List[str]
    recipient_id: Optional[str]
    template_name: Optional[str]
    content: Dict[str, Any]
    metadata: Optional[Dict[str, Any]]
    scheduled_for: datetime
    retry_count: int
    max_retries: int
    created_at: datetime

    @classmethod
    def from_db_message(cls, message: Message) -> 'QueuedMessage':
        """
        Create QueuedMessage from database Message.

        Args:
            message: Database Message object

        Returns:
            QueuedMessage instance
        """
        return cls(
            id=message.id,
            message_type=message.message_type,
            priority=MessagePriority(message.priority),
            channels=message.channels,
            recipient_id=message.recipient_id,
            template_name=message.template_name,
            content=message.content,
            metadata=message.message_metadata,
            scheduled_for=message.scheduled_for,
            retry_count=message.retry_count,
            max_retries=message.max_retries,
            created_at=message.created_at
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "message_type": self.message_type,
            "priority": self.priority.value,
            "channels": self.channels,
            "recipient_id": self.recipient_id,
            "template_name": self.template_name,
            "content": self.content,
            "metadata": self.metadata,
            "scheduled_for": self.scheduled_for.isoformat(),
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "created_at": self.created_at.isoformat()
        }

    @property
    def is_high_priority(self) -> bool:
        """Check if message is high priority."""
        return self.priority in [MessagePriority.HIGH, MessagePriority.CRITICAL]

    @property
    def can_retry(self) -> bool:
        """Check if message can be retried."""
        return self.retry_count < self.max_retries

    @property
    def queue_priority(self) -> int:
        """Get numeric priority for queue ordering."""
        priority_map = {
            MessagePriority.CRITICAL: 1,
            MessagePriority.HIGH: 2,
            MessagePriority.NORMAL: 3,
            MessagePriority.LOW: 4
        }
        return priority_map.get(self.priority, 3)


class MessageQueue:
    """
    Database-backed message queue with priority handling.

    Provides methods for enqueueing, dequeuing, and managing messages
    with support for priorities, scheduling, and retry logic.
    """

    def __init__(self):
        """Initialize the message queue."""
        self._logger = setup_logger(f"{__name__}.MessageQueue")

    def enqueue(
        self,
        message_data: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        scheduled_for: Optional[datetime] = None
    ) -> int:
        """
        Enqueue a message for processing.

        Args:
            message_data: Message data dictionary
            priority: Message priority
            scheduled_for: When to process the message (defaults to now)

        Returns:
            Message ID

        Raises:
            ValueError: If message data is invalid
        """
        try:
            # Validate message data
            self._validate_message_data(message_data)

            # Set default scheduled time
            if scheduled_for is None:
                scheduled_for = datetime.now(timezone.utc)

            # Prepare message for database
            db_message_data = message_data.copy()
            db_message_data.update({
                'priority': priority.value,
                'scheduled_for': scheduled_for,
                'status': MessageStatus.PENDING.value
            })

            # Store in database
            db_service = get_database_service()
            with db_service.uow() as r:
                message = r.notifications.messages.create_message(db_message_data)
                message_id = message.id

            self._logger.info(
                "Message %s enqueued with priority %s, scheduled for %s",
                message_id, priority.value, scheduled_for
            )

            return message_id

        except Exception:
            self._logger.exception("Failed to enqueue message:")
            raise

    def dequeue(
        self,
        limit: int = 10,
        priority_filter: Optional[MessagePriority] = None,
        channels: Optional[List[str]] = None
    ) -> List[QueuedMessage]:
        """
        Dequeue messages ready for processing.

        Args:
            limit: Maximum number of messages to dequeue
            priority_filter: Only dequeue messages with this priority
            channels: Only dequeue messages for these channels

        Returns:
            List of QueuedMessage objects
        """
        try:
            current_time = datetime.now(timezone.utc)

            db_service = get_database_service()
            with db_service.uow() as r:
                # Get pending messages
                messages = r.notifications.messages.get_pending_messages(
                    current_time=current_time,
                    priority=priority_filter,
                    channels=channels,
                    limit=limit
                )

                # Mark messages as processing
                queued_messages = []
                for message in messages:
                    # Update status to processing
                    r.notifications.messages.update_message(message.id, {
                        'status': MessageStatus.PROCESSING.value,
                        'processed_at': current_time
                    })

                    # Convert to QueuedMessage
                    queued_message = QueuedMessage.from_db_message(message)
                    queued_messages.append(queued_message)

                if queued_messages:
                    self._logger.info(
                        "Dequeued %s messages for processing",
                        len(queued_messages)
                    )

                return queued_messages

        except Exception:
            self._logger.exception("Failed to dequeue messages:")
            raise

    def dequeue_high_priority(self, limit: int = 5, channels: Optional[List[str]] = None) -> List[QueuedMessage]:
        """
        Dequeue only high priority messages (HIGH and CRITICAL).

        Args:
            limit: Maximum number of messages to dequeue
            channels: Only dequeue messages for these channels

        Returns:
            List of high priority QueuedMessage objects
        """
        try:
            current_time = datetime.now(timezone.utc)

            db_service = get_database_service()
            with db_service.uow() as r:
                # Get high priority messages first
                high_priority_messages = []

                # Get CRITICAL messages first
                critical_messages = r.notifications.messages.get_pending_messages(
                    current_time=current_time,
                    priority=MessagePriority.CRITICAL,
                    channels=channels,
                    limit=limit
                )
                high_priority_messages.extend(critical_messages)

                # Get HIGH messages if we have room
                remaining_limit = limit - len(critical_messages)
                if remaining_limit > 0:
                    high_messages = r.notifications.messages.get_pending_messages(
                        current_time=current_time,
                        priority=MessagePriority.HIGH,
                        channels=channels,
                        limit=remaining_limit
                    )
                    high_priority_messages.extend(high_messages)

                # Mark messages as processing and convert
                queued_messages = []
                for message in high_priority_messages:
                    r.notifications.messages.update_message(message.id, {
                        'status': MessageStatus.PROCESSING.value,
                        'processed_at': current_time
                    })

                    queued_message = QueuedMessage.from_db_message(message)
                    queued_messages.append(queued_message)

                if queued_messages:
                    self._logger.info(
                        "Dequeued %s high priority messages",
                        len(queued_messages)
                    )

                return queued_messages

        except Exception:
            self._logger.exception("Failed to dequeue high priority messages:")
            raise

    def dequeue_for_retry(
        self,
        limit: int = 20,
        retry_delay_minutes: int = 5
    ) -> List[QueuedMessage]:
        """
        Dequeue failed messages that are ready for retry.

        Args:
            limit: Maximum number of messages to dequeue
            retry_delay_minutes: Minimum delay before retry

        Returns:
            List of QueuedMessage objects ready for retry
        """
        try:
            current_time = datetime.now(timezone.utc)

            db_service = get_database_service()
            with db_service.uow() as r:
                # Get failed messages ready for retry
                messages = r.notifications.messages.get_failed_messages_for_retry(
                    current_time=current_time,
                    retry_delay_minutes=retry_delay_minutes,
                    limit=limit
                )

                # Mark messages as processing and increment retry count
                queued_messages = []
                for message in messages:
                    r.notifications.messages.update_message(message.id, {
                        'status': MessageStatus.PROCESSING.value,
                        'retry_count': message.retry_count + 1,
                        'processed_at': current_time,
                        'last_error': None  # Clear previous error
                    })

                    # Refresh message to get updated retry count
                    updated_message = r.notifications.messages.get_message(message.id)
                    queued_message = QueuedMessage.from_db_message(updated_message)
                    queued_messages.append(queued_message)

                if queued_messages:
                    self._logger.info(
                        "Dequeued %s messages for retry",
                        len(queued_messages)
                    )

                return queued_messages

        except Exception:
            self._logger.exception("Failed to dequeue retry messages:")
            raise

    def mark_delivered(self, message_id: int) -> bool:
        """
        Mark a message as successfully delivered.

        Args:
            message_id: Message ID

        Returns:
            True if successful, False if message not found
        """
        try:
            db_service = get_database_service()
            with db_service.uow() as r:
                message = r.notifications.messages.update_message(message_id, {
                    'status': MessageStatus.DELIVERED.value,
                    'processed_at': datetime.now(timezone.utc)
                })

                if message:
                    self._logger.info("Message %s marked as delivered", message_id)
                    return True
                else:
                    self._logger.warning("Message %s not found for delivery update", message_id)
                    return False

        except Exception as e:
            self._logger.error("Failed to mark message %s as delivered: %s", message_id, e)
            raise

    def mark_failed(self, message_id: int, error_message: str) -> bool:
        """
        Mark a message as failed.

        Args:
            message_id: Message ID
            error_message: Error description

        Returns:
            True if successful, False if message not found
        """
        try:
            db_service = get_database_service()
            with db_service.uow() as r:
                message = r.notifications.messages.update_message(message_id, {
                    'status': MessageStatus.FAILED.value,
                    'last_error': error_message,
                    'processed_at': datetime.now(timezone.utc)
                })

                if message:
                    self._logger.warning(
                        "Message %s marked as failed: %s",
                        message_id, error_message
                    )
                    return True
                else:
                    self._logger.warning("Message %s not found for failure update", message_id)
                    return False

        except Exception as e:
            self._logger.error("Failed to mark message %s as failed: %s", message_id, e)
            raise

    def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get queue statistics.

        Returns:
            Dictionary with queue statistics
        """
        try:
            db_service = get_database_service()
            with db_service.uow() as r:
                stats = {}

                # Count messages by status
                for status in MessageStatus:
                    count = r.s.query(Message).filter(
                        Message.status == status.value
                    ).count()
                    stats[f"{status.value.lower()}_count"] = count

                # Count messages by priority
                for priority in MessagePriority:
                    count = r.s.query(Message).filter(
                        Message.priority == priority.value
                    ).count()
                    stats[f"{priority.value.lower()}_priority_count"] = count

                # Count overdue messages
                current_time = datetime.now(timezone.utc)
                overdue_count = r.s.query(Message).filter(
                    Message.status == MessageStatus.PENDING.value,
                    Message.scheduled_for < current_time - timedelta(minutes=5)
                ).count()
                stats["overdue_count"] = overdue_count

                # Count retry-eligible messages
                retry_eligible_count = r.s.query(Message).filter(
                    Message.status == MessageStatus.FAILED.value,
                    Message.retry_count < Message.max_retries
                ).count()
                stats["retry_eligible_count"] = retry_eligible_count

                return stats

        except Exception:
            self._logger.exception("Failed to get queue stats:")
            raise

    def _validate_message_data(self, message_data: Dict[str, Any]) -> None:
        """
        Validate message data before enqueueing.

        Args:
            message_data: Message data to validate

        Raises:
            ValueError: If message data is invalid
        """
        required_fields = ['message_type', 'channels', 'content']

        for field in required_fields:
            if field not in message_data:
                raise ValueError(f"Missing required field: {field}")

        # Validate message_type
        if not isinstance(message_data['message_type'], str) or not message_data['message_type'].strip():
            raise ValueError("message_type must be a non-empty string")

        # Validate channels
        if not isinstance(message_data['channels'], list) or not message_data['channels']:
            raise ValueError("channels must be a non-empty list")

        for channel in message_data['channels']:
            if not isinstance(channel, str) or not channel.strip():
                raise ValueError("All channels must be non-empty strings")

        # Validate content
        if not isinstance(message_data['content'], dict):
            raise ValueError("content must be a dictionary")

        # Validate optional fields
        if 'recipient_id' in message_data and message_data['recipient_id'] is not None:
            if not isinstance(message_data['recipient_id'], str):
                raise ValueError("recipient_id must be a string")

        if 'template_name' in message_data and message_data['template_name'] is not None:
            if not isinstance(message_data['template_name'], str):
                raise ValueError("template_name must be a string")

        if 'message_metadata' in message_data and message_data['message_metadata'] is not None:
            if not isinstance(message_data['message_metadata'], dict):
                raise ValueError("message_metadata must be a dictionary")

    def sanitize_message_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize message content to prevent injection attacks.

        Args:
            content: Message content to sanitize

        Returns:
            Sanitized content dictionary
        """
        def sanitize_value(value):
            """Recursively sanitize values."""
            if isinstance(value, str):
                # Basic HTML/script tag removal
                import re
                value = re.sub(r'<script[^>]*>.*?</script>', '', value, flags=re.IGNORECASE | re.DOTALL)
                value = re.sub(r'<[^>]+>', '', value)  # Remove HTML tags
                return value.strip()
            elif isinstance(value, dict):
                return {k: sanitize_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [sanitize_value(item) for item in value]
            else:
                return value

        return sanitize_value(content)


# Global message queue instance
message_queue = MessageQueue()