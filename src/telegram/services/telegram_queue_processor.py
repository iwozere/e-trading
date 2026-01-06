"""
Telegram Queue Processor

Polls the database for pending Telegram messages and sends them via aiogram.

This implements Phase 4 of MIGRATION_PLAN.md - making the telegram bot handle
queued Telegram messages from the database, completing the hybrid architecture where:
- Telegram Bot owns ALL Telegram sending (instant + queued)
- Notification Service owns Email/SMS only

Architecture:
- Polls database every 5 seconds for messages with channels=["telegram"]
- Filters to PENDING status
- Sends via aiogram
- Marks as DELIVERED/FAILED in database using shared MessageQueueClient
"""

import asyncio
from typing import Optional, Dict, Any
from datetime import datetime
import base64
from io import BytesIO

from aiogram import Bot
from aiogram.types import BufferedInputFile, FSInputFile
from aiogram.exceptions import TelegramAPIError, TelegramBadRequest

from src.notification.service.message_queue_client import get_message_queue_client
from src.notification.logger import setup_logger
from src.data.db.services.users_service import users_service

_logger = setup_logger(__name__)


class TelegramQueueProcessor:
    """
    Processes Telegram messages from database queue.

    Used for heavy commands like /report and /screener that are queued
    to the database and then delivered asynchronously.
    """

    def __init__(self, bot: Bot, poll_interval: int = 5):
        """
        Initialize the telegram queue processor.

        Args:
            bot: Aiogram Bot instance for sending messages
            poll_interval: Seconds between database polls (default: 5)
        """
        self.bot = bot
        self.poll_interval = poll_interval
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._message_queue_client = get_message_queue_client()
        self._logger = setup_logger(f"{__name__}.TelegramQueueProcessor")

        self._logger.info(
            "Telegram queue processor initialized (poll interval: %s seconds)",
            poll_interval
        )

    async def start(self):
        """Start the queue processor."""
        if self._running:
            self._logger.warning("Queue processor already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        self._logger.info("Telegram queue processor started")

    async def stop(self):
        """Stop the queue processor."""
        if not self._running:
            return

        self._logger.info("Stopping telegram queue processor...")
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        self._logger.info("Telegram queue processor stopped")

    async def _poll_loop(self):
        """Main polling loop."""
        self._logger.info("Telegram queue processor polling loop started")

        while self._running:
            try:
                # Get pending Telegram messages
                messages = self._message_queue_client.get_pending_messages_for_channels(
                    channels=["telegram"],
                    limit=10  # Process up to 10 messages per poll
                )

                if messages:
                    self._logger.info("Processing %s pending Telegram messages", len(messages))

                    for message in messages:
                        try:
                            await self._process_message(message)
                        except Exception:
                            self._logger.exception(
                                "Failed to process message %s:",
                                message.id
                            )

                # Sleep before next poll
                await asyncio.sleep(self.poll_interval)

            except asyncio.CancelledError:
                break
            except Exception:
                self._logger.exception("Error in telegram queue processor poll loop:")
                # Back off on error
                await asyncio.sleep(self.poll_interval * 2)

        self._logger.info("Telegram queue processor polling loop stopped")

    async def _process_message(self, message: Dict[str, Any]):
        """
        Process a single message from the queue.

        Args:
            message: Message dictionary from database
        """
        message_id = message['id']
        self._logger.info("Processing message %s", message_id)

        # Mark as processing
        self._message_queue_client.mark_message_processing(message_id)

        try:
            # Extract telegram-specific data
            telegram_chat_id = self._extract_telegram_chat_id(message)
            if not telegram_chat_id:
                error_msg = "Missing telegram_chat_id in message metadata"
                self._logger.error("Message %s: %s", message_id, error_msg)
                self._message_queue_client.mark_message_failed(
                    message_id,
                    error_msg,
                    increment_retry=False  # Don't retry if chat_id is missing
                )
                return

            # Build message text
            message_text = self._build_message_text(message)

            # Extract optional parameters
            message_metadata = message.get('message_metadata') or {}
            reply_to_message_id = message_metadata.get("reply_to_message_id")

            # Check for attachments
            content = message.get('content') or {}
            attachments = content.get("attachments", {})

            if attachments:
                # Send with attachments
                await self._send_with_attachments(
                    telegram_chat_id,
                    message_text,
                    attachments,
                    reply_to_message_id
                )
            else:
                # Send text only
                await self.bot.send_message(
                    chat_id=telegram_chat_id,
                    text=message_text,
                    reply_to_message_id=reply_to_message_id,
                    parse_mode="HTML"
                )

            # Mark as delivered
            self._message_queue_client.mark_message_delivered(message_id)
            self._logger.info("Successfully delivered message %s to Telegram", message_id)

        except (TelegramAPIError, TelegramBadRequest) as e:
            # Telegram API error
            error_msg = f"Telegram API error: {str(e)}"
            self._logger.error("Message %s: %s", message_id, error_msg)
            self._message_queue_client.mark_message_failed(
                message_id,
                error_msg,
                increment_retry=True
            )

        except Exception as e:
            # Other errors
            error_msg = f"Unexpected error: {str(e)}"
            self._logger.exception("Message %s: %s", message_id, error_msg)
            self._message_queue_client.mark_message_failed(
                message_id,
                error_msg,
                increment_retry=True
            )

    def _extract_telegram_chat_id(self, message: Dict[str, Any]) -> Optional[int]:
        """
        Extract telegram_chat_id from message metadata or resolve from recipient_id.

        Args:
            message: Message dictionary

        Returns:
            Telegram chat ID or None if not found/could not be resolved
        """
        # 1. Check metadata first (direct chat_id override)
        message_metadata = message.get('message_metadata') or {}
        telegram_chat_id = message_metadata.get("telegram_chat_id")
        if telegram_chat_id:
            try:
                return int(telegram_chat_id)
            except (ValueError, TypeError):
                self._logger.warning("Invalid telegram_chat_id in metadata: %s", telegram_chat_id)

        # 2. Check recipient_id
        recipient_id = message.get('recipient_id')
        if recipient_id:
            # If it's a numeric string that looks like a Telegram ID (usually > 100k)
            # but we'll try to resolve it as a User ID first if it's small,
            # or if it's explicitly a User ID.

            try:
                recipient_id_int = int(recipient_id)

                # If recipient_id is small (e.g. < 1,000,000), it's likely our internal User ID
                # Telegram IDs are typically much larger integers.
                if recipient_id_int < 1000000:
                    self._logger.debug("Attempting to resolve User ID %s to Telegram Chat ID", recipient_id_int)
                    channels = users_service.get_user_notification_channels(recipient_id_int)
                    if channels and channels.get('telegram_chat_id'):
                        resolved_id = channels['telegram_chat_id']
                        self._logger.debug("Resolved User ID %s to Telegram Chat ID %s", recipient_id_int, resolved_id)
                        return int(resolved_id)

                # Otherwise, treat it as a direct Chat ID
                return recipient_id_int
            except (ValueError, TypeError):
                # Probably not a numeric ID
                pass

        return None

    def _build_message_text(self, message: Dict[str, Any]) -> str:
        """
        Build the message text from message content.

        Args:
            message: Message dictionary

        Returns:
            Formatted message text
        """
        content = message.get('content') or {}
        title = content.get("title", "")
        body = content.get("message", "")

        if title and body:
            return f"<b>{title}</b>\n\n{body}"
        elif title:
            return title
        else:
            return body or "Empty message"

    async def _send_with_attachments(
        self,
        chat_id: int,
        text: str,
        attachments: Dict[str, Any],
        reply_to_message_id: Optional[int] = None
    ):
        """
        Send message with attachments.

        Args:
            chat_id: Telegram chat ID
            text: Message text
            attachments: Dictionary of filename -> attachment data
            reply_to_message_id: Optional message ID to reply to
        """
        # Handle nested dictionary format: {"files": ["path1", "path2"]}
        if isinstance(attachments, dict) and "files" in attachments and len(attachments) == 1:
            files_list = attachments["files"]
            if isinstance(files_list, list):
                # Convert to flat dictionary for processing
                flat_attachments = {}
                for f in files_list:
                    p = Path(f)
                    flat_attachments[p.name] = str(f)
                attachments = flat_attachments

        # Send each attachment as a document
        for filename, attachment_data in attachments.items():
            try:
                # Process attachment data
                file_obj = await self._process_attachment(filename, attachment_data)

                if file_obj:
                    # Send document
                    await self.bot.send_document(
                        chat_id=chat_id,
                        document=file_obj,
                        caption=text if text else None,
                        reply_to_message_id=reply_to_message_id,
                        parse_mode="HTML"
                    )
                    self._logger.debug("Sent attachment %s to chat %s", filename, chat_id)

                    # Only send caption once with first attachment
                    text = None

            except Exception:
                self._logger.exception("Failed to send attachment %s:", filename)
                # Continue with other attachments even if one fails

        # If there's still text and no attachments were sent, send as text message
        if text:
            await self.bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_to_message_id=reply_to_message_id,
                parse_mode="HTML"
            )

    async def _process_attachment(
        self,
        filename: str,
        attachment_data: Any
    ) -> Optional[BufferedInputFile]:
        """
        Process attachment data into aiogram input file.

        Args:
            filename: Filename for the attachment
            attachment_data: Attachment data (bytes, dict with base64, or file path)

        Returns:
            BufferedInputFile or None if processing failed
        """
        try:
            actual_data = None

            if isinstance(attachment_data, dict):
                # Attachment stored as base64 in database
                if attachment_data.get("type") == "base64":
                    actual_data = base64.b64decode(attachment_data["data"])
                    self._logger.debug(
                        "Decoded base64 attachment: %s (size: %d bytes)",
                        filename, len(actual_data)
                    )
                elif attachment_data.get("type") == "file_path":
                    # File path stored in database
                    file_path = attachment_data.get("path")
                    if file_path:
                        with open(file_path, 'rb') as f:
                            actual_data = f.read()
                        self._logger.debug(
                            "Read file attachment: %s from %s",
                            filename, file_path
                        )
            elif isinstance(attachment_data, bytes):
                # Raw bytes (direct usage)
                actual_data = attachment_data
                self._logger.debug(
                    "Using raw bytes attachment: %s (size: %d bytes)",
                    filename, len(actual_data)
                )

            if actual_data:
                return BufferedInputFile(actual_data, filename=filename)
            else:
                self._logger.warning("Could not process attachment %s", filename)
                return None

        except Exception:
            self._logger.exception("Failed to process attachment %s:", filename)
            return None

    def get_stats(self) -> Dict[str, Any]:
        """
        Get processor statistics.

        Returns:
            Dictionary with processor stats
        """
        return {
            "running": self._running,
            "poll_interval": self.poll_interval,
            "pending_count": self._message_queue_client.get_pending_count_for_channels(
                ["telegram"]
            ) if self._running else 0
        }
