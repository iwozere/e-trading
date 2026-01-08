"""
Shared Database Message Queue Client

Provides common utilities for polling and updating messages in the database queue.
Used by both Telegram Bot and Notification Service to access the shared message queue.

This client implements the shared DB utilities pattern (Variant D from MIGRATION_PLAN.md Phase 4):
- Keep telegram bot and notification service separate
- Extract common DB query and update utilities
- Each service maintains its own polling loop but uses shared DB operations
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from contextlib import contextmanager

from src.data.db.models.model_notification import (
    Message, MessagePriority, MessageStatus
)
from src.data.db.services.database_service import get_database_service
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class MessageQueueClient:
    """
    Shared client for accessing the message queue database.

    Provides common operations for polling pending messages,
    updating message status, and handling channel-specific queries.
    """

    def __init__(self):
        """Initialize the message queue client."""
        self._logger = setup_logger(f"{__name__}.MessageQueueClient")
        self._db_service = get_database_service()

    def get_pending_messages_for_channels(
        self,
        channels: List[str],
        limit: int = 10,
        priority: Optional[MessagePriority] = None
    ) -> List[Dict[str, Any]]:
        """
        Get pending messages that include any of the specified channels.

        Args:
            channels: List of channel names to filter by (e.g., ["telegram"])
            limit: Maximum number of messages to retrieve
            priority: Optional priority filter

        Returns:
            List of message dictionaries ready for processing
            (converted to dicts to avoid SQLAlchemy session detachment issues)

        Example:
            # Get pending Telegram messages
            messages = client.get_pending_messages_for_channels(["telegram"], limit=10)

            # Get pending Email/SMS messages
            messages = client.get_pending_messages_for_channels(["email", "sms"], limit=20)
        """
        try:
            current_time = datetime.now(timezone.utc)

            with self._db_service.uow() as r:
                # Get all pending messages
                all_pending = r.notifications.messages.get_pending_messages(
                    current_time=current_time,
                    priority=priority,
                    limit=limit * 2  # Get more to filter by channels
                )

                # Filter to messages that contain at least one of our channels
                # Convert to dictionaries immediately to avoid session detachment
                filtered_messages = []
                for message in all_pending:
                    # message.channels is a list like ["telegram"] or ["email", "telegram"]
                    if any(channel in message.channels for channel in channels):
                        # Convert to dictionary with all needed fields
                        message_dict = {
                            'id': message.id,
                            'message_type': message.message_type,
                            'priority': message.priority,
                            'channels': message.channels,
                            'recipient_id': message.recipient_id,
                            'content': message.content,
                            'message_metadata': message.message_metadata,
                            'scheduled_for': message.scheduled_for,
                            'retry_count': message.retry_count,
                            'max_retries': message.max_retries,
                            'created_at': message.created_at,
                            'status': message.status
                        }
                        filtered_messages.append(message_dict)
                        if len(filtered_messages) >= limit:
                            break

                if filtered_messages:
                    self._logger.info("Found %s pending messages for channels %s", len(filtered_messages), channels)
                #else:
                #    self._logger.debug("No pending messages found for channels %s", channels)

                return filtered_messages

        except Exception:
            self._logger.exception("Failed to get pending messages for channels %s:", channels)
            return []

    def mark_message_processing(self, message_id: int) -> bool:
        """
        Mark a message as currently being processed.

        Args:
            message_id: ID of the message to mark

        Returns:
            True if successful, False otherwise
        """
        try:
            current_time = datetime.now(timezone.utc)

            with self._db_service.uow() as r:
                updated = r.notifications.messages.update_message(message_id, {
                    'status': MessageStatus.PROCESSING.value,
                    'processed_at': current_time
                })

                if updated:
                    self._logger.debug("Marked message %s as processing", message_id)
                    return True
                else:
                    self._logger.warning("Failed to mark message %s as processing", message_id)
                    return False

        except Exception:
            self._logger.exception("Failed to mark message %s as processing:", message_id)
            return False

    def mark_message_delivered(
        self,
        message_id: int,
        delivery_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Mark a message as successfully delivered.

        Args:
            message_id: ID of the message to mark
            delivery_metadata: Optional metadata about the delivery

        Returns:
            True if successful, False otherwise
        """
        try:
            current_time = datetime.now(timezone.utc)

            update_data = {
                'status': MessageStatus.DELIVERED.value,
                'delivered_at': current_time
            }

            if delivery_metadata:
                update_data['delivery_metadata'] = delivery_metadata

            with self._db_service.uow() as r:
                updated = r.notifications.messages.update_message(message_id, update_data)

                if updated:
                    self._logger.info("Marked message %s as delivered", message_id)
                    return True
                else:
                    self._logger.warning("Failed to mark message %s as delivered", message_id)
                    return False

        except Exception:
            self._logger.exception("Failed to mark message %s as delivered:", message_id)
            return False

    def mark_message_failed(
        self,
        message_id: int,
        error_message: str,
        increment_retry: bool = True
    ) -> bool:
        """
        Mark a message as failed.

        Args:
            message_id: ID of the message to mark
            error_message: Error description
            increment_retry: Whether to increment the retry count

        Returns:
            True if successful, False otherwise
        """
        try:
            current_time = datetime.now(timezone.utc)

            update_data = {
                'status': MessageStatus.FAILED.value,
                'error_message': error_message,
                'failed_at': current_time
            }

            if increment_retry:
                # Get current message to increment retry count
                with self._db_service.uow() as r:
                    message = r.notifications.messages.get_message(message_id)
                    if message:
                        update_data['retry_count'] = message.retry_count + 1

            with self._db_service.uow() as r:
                updated = r.notifications.messages.update_message(message_id, update_data)

                if updated:
                    self._logger.warning(
                        "Marked message %s as failed: %s",
                        message_id, error_message
                    )
                    return True
                else:
                    self._logger.warning("Failed to mark message %s as failed", message_id)
                    return False

        except Exception:
            self._logger.exception("Failed to mark message %s as failed:", message_id)
            return False

    def get_message(self, message_id: int) -> Optional[Message]:
        """
        Get a message by ID.

        Args:
            message_id: Message ID

        Returns:
            Message object or None if not found
        """
        try:
            with self._db_service.uow() as r:
                return r.notifications.messages.get_message(message_id)
        except Exception:
            self._logger.exception("Failed to get message %s:", message_id)
            return None

    def get_pending_count_for_channels(self, channels: List[str]) -> int:
        """
        Get count of pending messages for specified channels.

        Args:
            channels: List of channel names

        Returns:
            Count of pending messages
        """
        try:
            current_time = datetime.now(timezone.utc)

            with self._db_service.uow() as r:
                all_pending = r.notifications.messages.get_pending_messages(
                    current_time=current_time,
                    limit=1000  # High limit to get accurate count
                )

                # Count messages that contain at least one of our channels
                count = sum(
                    1 for message in all_pending
                    if any(channel in message.channels for channel in channels)
                )

                return count

        except Exception:
            self._logger.exception("Failed to get pending count for channels %s:", channels)
            return 0

    @contextmanager
    def process_message(self, message_id: int):
        """
        Context manager for processing a message with automatic status updates.

        Usage:
            with client.process_message(message_id):
                # Send message via channel
                send_telegram_message(...)
            # Automatically marked as delivered on success
            # Automatically marked as failed on exception

        Args:
            message_id: ID of the message to process
        """
        # Mark as processing
        self.mark_message_processing(message_id)

        try:
            yield
            # Success - mark as delivered
            self.mark_message_delivered(message_id)

        except Exception as e:
            # Failed - mark as failed
            error_msg = str(e)
            self.mark_message_failed(message_id, error_msg, increment_retry=True)
            raise  # Re-raise to let caller handle


# Global singleton instance
_message_queue_client = None


def get_message_queue_client() -> MessageQueueClient:
    """
    Get the global message queue client instance.

    Returns:
        MessageQueueClient singleton
    """
    global _message_queue_client

    if _message_queue_client is None:
        _message_queue_client = MessageQueueClient()
        _logger.info("Initialized shared message queue client")

    return _message_queue_client
