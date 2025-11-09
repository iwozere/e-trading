"""
Telegram Channel Plugin

Telegram notification channel implementation using aiogram.
Supports message splitting, attachments, health monitoring, and dynamic chat IDs.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import asyncio

from aiogram import Bot
from aiogram.types import BufferedInputFile, FSInputFile
from aiogram.exceptions import TelegramAPIError, TelegramBadRequest, TelegramForbiddenError

from src.notification.channels.base import (
    NotificationChannel, DeliveryResult, ChannelHealth, MessageContent,
    DeliveryStatus, ChannelHealthStatus
)
from src.notification.channels.config import ConfigValidator, CommonValidationRules
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class TelegramChannel(NotificationChannel):
    """
    Telegram notification channel using aiogram.

    Supports:
    - Message splitting for long messages (4096 char limit)
    - Photo and document attachments
    - Dynamic chat IDs per message
    - Reply functionality
    - Health monitoring via Telegram API
    - Rate limiting and retry mechanisms
    """

    # Telegram API limits
    MAX_MESSAGE_LENGTH = 4000  # Leave buffer for safety (actual limit is 4096)
    MAX_CAPTION_LENGTH = 1024
    RATE_LIMIT_DEFAULT = 30  # messages per minute

    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate Telegram channel configuration.

        Args:
            config: Configuration dictionary to validate

        Raises:
            ValueError: If configuration is invalid
        """
        validator = ConfigValidator(self.channel_name)

        # Required fields
        validator.require_field(
            "bot_token",
            str,
            description="Telegram bot token from @BotFather",
            min_length=40,
            max_length=50
        )

        validator.optional_field(
            "default_chat_id",
            str,
            description="Default chat ID for messages (can be overridden per message)",
            min_length=1
        )

        # Optional fields
        validator.optional_field(
            "parse_mode",
            str,
            description="Default parse mode (HTML, Markdown, or None)",
            allowed_values=["HTML", "Markdown", None]
        )

        validator.optional_field(
            "disable_web_page_preview",
            bool,
            description="Disable web page preview in messages"
        )

        validator.optional_field(
            "disable_notification",
            bool,
            description="Send messages silently"
        )

        validator.optional_field(
            "protect_content",
            bool,
            description="Protect message content from forwarding"
        )

        validator.optional_field(
            "message_thread_id",
            int,
            description="Default message thread ID for forum groups"
        )

        # Add common validation rules
        validator.add_rule(CommonValidationRules.timeout_seconds(default=30))
        validator.add_rule(CommonValidationRules.rate_limit(default=self.RATE_LIMIT_DEFAULT))
        validator.add_rule(CommonValidationRules.max_retries(default=3))

        # Validate and update config
        validated_config = validator.validate(config)
        self.config.update(validated_config)

        # Initialize bot instance
        self.bot = Bot(token=self.config["bot_token"])

    async def send_message(
        self,
        recipient: str,
        content: MessageContent,
        message_id: Optional[str] = None,
        priority: str = "NORMAL"
    ) -> DeliveryResult:
        """
        Send a message via Telegram.

        Args:
            recipient: Chat ID (can be user ID, group ID, or channel username)
            content: Message content to send
            message_id: Optional message ID for tracking
            priority: Message priority (affects retry behavior)

        Returns:
            DeliveryResult with delivery status and metadata
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Use recipient as chat_id, or fall back to default
            chat_id = recipient or self.config["default_chat_id"]

            # Extract metadata for Telegram-specific options
            metadata = content.metadata or {}
            reply_to_message_id = metadata.get("reply_to_message_id")
            message_thread_id = metadata.get("message_thread_id", self.config.get("message_thread_id"))

            # Handle attachments first
            if content.has_attachments:
                return await self._send_with_attachments(
                    chat_id, content, reply_to_message_id, message_thread_id, start_time
                )

            # Send text message with splitting if needed
            telegram_message = await self._send_text_message_with_splitting(
                chat_id=chat_id,
                text=content.text,
                reply_to_message_id=reply_to_message_id,
                message_thread_id=message_thread_id
            )

            response_time = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)

            return DeliveryResult(
                success=True,
                status=DeliveryStatus.DELIVERED,
                external_id=str(telegram_message.message_id),
                response_time_ms=response_time,
                metadata={
                    "chat_id": chat_id,
                    "message_id": telegram_message.message_id,
                    "date": telegram_message.date.isoformat() if telegram_message.date else None,
                    "message_thread_id": message_thread_id
                }
            )

        except TelegramForbiddenError as e:
            error_msg = f"Bot was blocked or chat not found: {str(e)}"
            _logger.warning("Telegram forbidden error for chat %s: %s", recipient, error_msg)

            return DeliveryResult(
                success=False,
                status=DeliveryStatus.BOUNCED,
                error_message=error_msg,
                response_time_ms=int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            )

        except TelegramBadRequest as e:
            error_msg = f"Bad request to Telegram API: {str(e)}"
            _logger.warning("Telegram bad request for chat %s: %s", recipient, error_msg)

            return DeliveryResult(
                success=False,
                status=DeliveryStatus.FAILED,
                error_message=error_msg,
                response_time_ms=int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            )

        except TelegramAPIError as e:
            error_msg = f"Telegram API error: {str(e)}"
            _logger.error("Telegram API error for chat %s: %s", recipient, error_msg)

            return DeliveryResult(
                success=False,
                status=DeliveryStatus.FAILED,
                error_message=error_msg,
                response_time_ms=int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            )

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            _logger.exception("Unexpected error sending Telegram message to %s: %s", recipient, error_msg)

            return DeliveryResult(
                success=False,
                status=DeliveryStatus.FAILED,
                error_message=error_msg,
                response_time_ms=int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            )

    async def _send_with_attachments(
        self,
        chat_id: str,
        content: MessageContent,
        reply_to_message_id: Optional[int],
        message_thread_id: Optional[int],
        start_time: datetime
    ) -> DeliveryResult:
        """Send message with attachments."""
        try:
            for filename, attachment_data in content.attachments.items():
                if isinstance(attachment_data, bytes):
                    # Send as photo if it looks like an image
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                        await self._send_photo_with_caption(
                            chat_id, attachment_data, filename, content.text,
                            reply_to_message_id, message_thread_id
                        )
                    else:
                        # Send as document
                        await self._send_document_with_caption(
                            chat_id, attachment_data, filename, content.text,
                            reply_to_message_id, message_thread_id
                        )

                elif isinstance(attachment_data, str):
                    # File path - send as document
                    await self._send_file_with_caption(
                        chat_id, attachment_data, filename, content.text,
                        reply_to_message_id, message_thread_id
                    )

            response_time = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)

            return DeliveryResult(
                success=True,
                status=DeliveryStatus.DELIVERED,
                external_id=f"attachment_{int(start_time.timestamp())}",
                response_time_ms=response_time,
                metadata={
                    "chat_id": chat_id,
                    "attachment_count": len(content.attachments),
                    "message_thread_id": message_thread_id
                }
            )

        except Exception as e:
            error_msg = f"Failed to send attachments: {str(e)}"
            _logger.error("Error sending attachments to chat %s: %s", chat_id, error_msg)

            return DeliveryResult(
                success=False,
                status=DeliveryStatus.FAILED,
                error_message=error_msg,
                response_time_ms=int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            )

    async def _send_photo_with_caption(
        self,
        chat_id: str,
        photo_data: bytes,
        filename: str,
        caption: str,
        reply_to_message_id: Optional[int],
        message_thread_id: Optional[int]
    ):
        """Send photo with caption, handling long captions."""
        if len(caption) > self.MAX_CAPTION_LENGTH:
            # Send text first, then photo with short caption
            await self._send_text_message_with_splitting(
                chat_id, caption, reply_to_message_id, message_thread_id
            )

            await self.bot.send_photo(
                chat_id=chat_id,
                photo=BufferedInputFile(photo_data, filename=filename),
                caption=f"📷 {filename}",
                message_thread_id=message_thread_id,
                parse_mode=self.config.get("parse_mode"),
                disable_notification=self.config.get("disable_notification", False),
                protect_content=self.config.get("protect_content", False)
            )
        else:
            await self.bot.send_photo(
                chat_id=chat_id,
                photo=BufferedInputFile(photo_data, filename=filename),
                caption=caption,
                reply_to_message_id=reply_to_message_id,
                message_thread_id=message_thread_id,
                parse_mode=self.config.get("parse_mode"),
                disable_notification=self.config.get("disable_notification", False),
                protect_content=self.config.get("protect_content", False)
            )

    async def _send_document_with_caption(
        self,
        chat_id: str,
        document_data: bytes,
        filename: str,
        caption: str,
        reply_to_message_id: Optional[int],
        message_thread_id: Optional[int]
    ):
        """Send document with caption."""
        if len(caption) > self.MAX_CAPTION_LENGTH:
            # Send text first, then document with short caption
            await self._send_text_message_with_splitting(
                chat_id, caption, reply_to_message_id, message_thread_id
            )

            await self.bot.send_document(
                chat_id=chat_id,
                document=BufferedInputFile(document_data, filename=filename),
                caption=f"📄 {filename}",
                message_thread_id=message_thread_id,
                disable_notification=self.config.get("disable_notification", False),
                protect_content=self.config.get("protect_content", False)
            )
        else:
            await self.bot.send_document(
                chat_id=chat_id,
                document=BufferedInputFile(document_data, filename=filename),
                caption=caption,
                reply_to_message_id=reply_to_message_id,
                message_thread_id=message_thread_id,
                disable_notification=self.config.get("disable_notification", False),
                protect_content=self.config.get("protect_content", False)
            )

    async def _send_file_with_caption(
        self,
        chat_id: str,
        file_path: str,
        filename: str,
        caption: str,
        reply_to_message_id: Optional[int],
        message_thread_id: Optional[int]
    ):
        """Send file from path with caption."""
        if len(caption) > self.MAX_CAPTION_LENGTH:
            # Send text first, then file with short caption
            await self._send_text_message_with_splitting(
                chat_id, caption, reply_to_message_id, message_thread_id
            )

            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                await self.bot.send_photo(
                    chat_id=chat_id,
                    photo=FSInputFile(file_path, filename=filename),
                    caption=f"📷 {filename}",
                    message_thread_id=message_thread_id,
                    parse_mode=self.config.get("parse_mode"),
                    disable_notification=self.config.get("disable_notification", False),
                    protect_content=self.config.get("protect_content", False)
                )
            else:
                await self.bot.send_document(
                    chat_id=chat_id,
                    document=FSInputFile(file_path, filename=filename),
                    caption=f"📄 {filename}",
                    message_thread_id=message_thread_id,
                    disable_notification=self.config.get("disable_notification", False),
                    protect_content=self.config.get("protect_content", False)
                )
        else:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                await self.bot.send_photo(
                    chat_id=chat_id,
                    photo=FSInputFile(file_path, filename=filename),
                    caption=caption,
                    reply_to_message_id=reply_to_message_id,
                    message_thread_id=message_thread_id,
                    parse_mode=self.config.get("parse_mode"),
                    disable_notification=self.config.get("disable_notification", False),
                    protect_content=self.config.get("protect_content", False)
                )
            else:
                await self.bot.send_document(
                    chat_id=chat_id,
                    document=FSInputFile(file_path, filename=filename),
                    caption=caption,
                    reply_to_message_id=reply_to_message_id,
                    message_thread_id=message_thread_id,
                    disable_notification=self.config.get("disable_notification", False),
                    protect_content=self.config.get("protect_content", False)
                )

    async def _send_text_message_with_splitting(
        self,
        chat_id: str,
        text: str,
        reply_to_message_id: Optional[int] = None,
        message_thread_id: Optional[int] = None
    ):
        """Send text message with automatic splitting for long messages."""
        if len(text) <= self.MAX_MESSAGE_LENGTH:
            # Message is short enough, send normally
            return await self.bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_to_message_id=reply_to_message_id,
                message_thread_id=message_thread_id,
                parse_mode=self.config.get("parse_mode"),
                disable_web_page_preview=self.config.get("disable_web_page_preview", False),
                disable_notification=self.config.get("disable_notification", False),
                protect_content=self.config.get("protect_content", False)
            )

        # Split long message into parts
        parts = self._split_message(text, self.MAX_MESSAGE_LENGTH)
        last_message = None

        for i, part in enumerate(parts):
            # Add part indicator for multi-part messages
            if len(parts) > 1:
                part_text = f"📄 Part {i+1}/{len(parts)}\n\n{part}"
            else:
                part_text = part

            # Only use reply_to_message_id for the first part
            current_reply_id = reply_to_message_id if i == 0 else None

            try:
                last_message = await self.bot.send_message(
                    chat_id=chat_id,
                    text=part_text,
                    reply_to_message_id=current_reply_id,
                    message_thread_id=message_thread_id,
                    parse_mode=self.config.get("parse_mode"),
                    disable_web_page_preview=self.config.get("disable_web_page_preview", False),
                    disable_notification=self.config.get("disable_notification", False),
                    protect_content=self.config.get("protect_content", False)
                )

                # Small delay between messages to avoid rate limiting
                if i < len(parts) - 1:
                    await asyncio.sleep(0.5)

            except Exception as e:
                _logger.error("Failed to send message part %d/%d: %s", i+1, len(parts), e)
                # Try to send a truncated version if splitting failed
                if i == 0:  # Only for first part
                    truncated_text = text[:self.MAX_MESSAGE_LENGTH-100] + "\n\n... [Message truncated due to length]"
                    last_message = await self.bot.send_message(
                        chat_id=chat_id,
                        text=truncated_text,
                        reply_to_message_id=reply_to_message_id,
                        message_thread_id=message_thread_id,
                        parse_mode=self.config.get("parse_mode"),
                        disable_web_page_preview=self.config.get("disable_web_page_preview", False),
                        disable_notification=self.config.get("disable_notification", False),
                        protect_content=self.config.get("protect_content", False)
                    )
                break

        return last_message

    def _split_message(self, text: str, max_length: int) -> List[str]:
        """
        Split a long message into parts while preserving formatting and readability.

        Args:
            text: The message text to split
            max_length: Maximum length per part

        Returns:
            List of message parts
        """
        if len(text) <= max_length:
            return [text]

        parts = []
        current_part = ""
        lines = text.split('\n')

        for line in lines:
            # If adding this line would exceed the limit
            if len(current_part) + len(line) + 1 > max_length:
                if current_part:
                    parts.append(current_part.strip())
                    current_part = ""

                # If a single line is too long, split it
                if len(line) > max_length:
                    # Split long line by words
                    words = line.split()
                    temp_line = ""

                    for word in words:
                        if len(temp_line) + len(word) + 1 <= max_length:
                            temp_line += (word + " ")
                        else:
                            if temp_line:
                                parts.append(temp_line.strip())
                                temp_line = word + " "

                    if temp_line:
                        current_part = temp_line
                else:
                    current_part = line
            else:
                current_part += line + '\n'

        # Add the last part
        if current_part.strip():
            parts.append(current_part.strip())

        return parts

    async def check_health(self) -> ChannelHealth:
        """
        Check Telegram channel health by testing the bot API.

        Returns:
            ChannelHealth with current status and metrics
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Test bot API by getting bot info
            bot_info = await self.bot.get_me()

            response_time = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)

            return ChannelHealth(
                status=ChannelHealthStatus.HEALTHY,
                last_check=datetime.now(timezone.utc),
                response_time_ms=response_time,
                metadata={
                    "bot_id": bot_info.id,
                    "bot_username": bot_info.username,
                    "bot_first_name": bot_info.first_name,
                    "can_join_groups": bot_info.can_join_groups,
                    "can_read_all_group_messages": bot_info.can_read_all_group_messages,
                    "supports_inline_queries": bot_info.supports_inline_queries
                }
            )

        except TelegramAPIError as e:
            response_time = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            error_msg = f"Telegram API error: {str(e)}"

            # Determine if this is a temporary or permanent issue
            if "unauthorized" in str(e).lower() or "forbidden" in str(e).lower():
                status = ChannelHealthStatus.DOWN
            else:
                status = ChannelHealthStatus.DEGRADED

            return ChannelHealth(
                status=status,
                last_check=datetime.now(timezone.utc),
                response_time_ms=response_time,
                error_message=error_msg
            )

        except Exception as e:
            response_time = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            error_msg = f"Unexpected error: {str(e)}"

            return ChannelHealth(
                status=ChannelHealthStatus.DOWN,
                last_check=datetime.now(timezone.utc),
                response_time_ms=response_time,
                error_message=error_msg
            )

    def get_rate_limit(self) -> int:
        """
        Get rate limit for Telegram channel.

        Returns:
            Rate limit in messages per minute
        """
        return self.config.get("rate_limit_per_minute", self.RATE_LIMIT_DEFAULT)

    def supports_feature(self, feature: str) -> bool:
        """
        Check if Telegram channel supports a feature.

        Args:
            feature: Feature name to check

        Returns:
            True if feature is supported
        """
        supported_features = {
            "html": True,
            "markdown": True,
            "attachments": True,
            "photos": True,
            "documents": True,
            "replies": True,
            "threading": True,  # Forum groups
            "formatting": True,
            "splitting": True,
            "priority": False,  # Telegram doesn't have native priority
            "read_receipts": False,
            "typing_indicators": False,
            "voice_messages": True,
            "video_messages": True,
            "stickers": True,
            "animations": True,
            "polls": True,
            "location": True,
            "contact": True
        }

        return supported_features.get(feature, False)

    def get_max_message_length(self) -> int:
        """
        Get maximum message length for Telegram.

        Returns:
            Maximum message length in characters
        """
        return self.MAX_MESSAGE_LENGTH

    def format_message(self, content: MessageContent) -> MessageContent:
        """
        Format message for Telegram (apply parse mode if HTML).

        Args:
            content: Original message content

        Returns:
            Formatted message content
        """
        # If HTML is provided and parse_mode is HTML, use HTML content
        if content.html and self.config.get("parse_mode") == "HTML":
            return MessageContent(
                text=content.html,
                subject=content.subject,
                html=content.html,
                attachments=content.attachments,
                metadata=content.metadata
            )

        # Otherwise return as-is
        return content

    async def close(self):
        """Close the bot session."""
        if hasattr(self, 'bot') and self.bot:
            await self.bot.session.close()