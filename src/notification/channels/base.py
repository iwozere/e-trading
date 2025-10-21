"""
Notification Channel Plugin Base Classes

Abstract base classes and data structures for notification channel plugins.
Provides the interface that all channel implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class DeliveryStatus(str, Enum):
    """Delivery status enumeration."""
    PENDING = "PENDING"
    SENT = "SENT"
    DELIVERED = "DELIVERED"
    FAILED = "FAILED"
    BOUNCED = "BOUNCED"


class ChannelHealthStatus(str, Enum):
    """Channel health status enumeration."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    DOWN = "DOWN"


@dataclass
class DeliveryResult:
    """Result of a message delivery attempt."""

    success: bool
    status: DeliveryStatus
    external_id: Optional[str] = None
    response_time_ms: Optional[int] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate delivery result data."""
        if self.success and self.status in [DeliveryStatus.FAILED, DeliveryStatus.BOUNCED]:
            raise ValueError("Success cannot be True when status is FAILED or BOUNCED")

        if not self.success and self.status in [DeliveryStatus.SENT, DeliveryStatus.DELIVERED]:
            raise ValueError("Success cannot be False when status is SENT or DELIVERED")

    @property
    def is_successful(self) -> bool:
        """Check if delivery was successful."""
        return self.success and self.status in [DeliveryStatus.SENT, DeliveryStatus.DELIVERED]

    @property
    def is_retryable(self) -> bool:
        """Check if delivery failure is retryable."""
        return not self.success and self.status == DeliveryStatus.FAILED


@dataclass
class ChannelHealth:
    """Channel health information."""

    status: ChannelHealthStatus
    last_check: datetime
    response_time_ms: Optional[int] = None
    error_message: Optional[str] = None
    failure_count: int = 0
    metadata: Optional[Dict[str, Any]] = None

    @property
    def is_healthy(self) -> bool:
        """Check if channel is healthy."""
        return self.status == ChannelHealthStatus.HEALTHY

    @property
    def is_available(self) -> bool:
        """Check if channel is available for delivery."""
        return self.status in [ChannelHealthStatus.HEALTHY, ChannelHealthStatus.DEGRADED]


@dataclass
class MessageContent:
    """Structured message content for delivery."""

    text: str
    subject: Optional[str] = None
    html: Optional[str] = None
    attachments: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate message content."""
        if not self.text and not self.html:
            raise ValueError("Either text or html content must be provided")

    @property
    def has_attachments(self) -> bool:
        """Check if message has attachments."""
        return bool(self.attachments)

    @property
    def is_html(self) -> bool:
        """Check if message contains HTML content."""
        return bool(self.html)


class NotificationChannel(ABC):
    """
    Abstract base class for notification channel plugins.

    All channel implementations must inherit from this class and implement
    the required abstract methods.
    """

    def __init__(self, channel_name: str, config: Dict[str, Any]):
        """
        Initialize the notification channel.

        Args:
            channel_name: Name of the channel (e.g., 'telegram', 'email')
            config: Channel-specific configuration dictionary
        """
        self.channel_name = channel_name
        self.config = config
        self._logger = setup_logger(f"{__name__}.{channel_name}")

        # Validate configuration on initialization
        self.validate_config(config)

    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate channel configuration.

        Args:
            config: Configuration dictionary to validate

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    @abstractmethod
    async def send_message(
        self,
        recipient: str,
        content: MessageContent,
        message_id: Optional[str] = None,
        priority: str = "NORMAL"
    ) -> DeliveryResult:
        """
        Send a message through this channel.

        Args:
            recipient: Recipient identifier (chat_id, email, phone, etc.)
            content: Message content to send
            message_id: Optional message ID for tracking
            priority: Message priority (CRITICAL, HIGH, NORMAL, LOW)

        Returns:
            DeliveryResult with delivery status and metadata
        """
        pass

    @abstractmethod
    async def check_health(self) -> ChannelHealth:
        """
        Check the health status of this channel.

        Returns:
            ChannelHealth with current status and metrics
        """
        pass

    @abstractmethod
    def get_rate_limit(self) -> int:
        """
        Get the rate limit for this channel in messages per minute.

        Returns:
            Rate limit in messages per minute
        """
        pass

    @abstractmethod
    def supports_feature(self, feature: str) -> bool:
        """
        Check if this channel supports a specific feature.

        Args:
            feature: Feature name to check (e.g., 'html', 'attachments', 'replies')

        Returns:
            True if feature is supported, False otherwise
        """
        pass

    def format_message(self, content: MessageContent) -> MessageContent:
        """
        Format message content for this channel.

        Default implementation returns content as-is. Channels can override
        this to apply channel-specific formatting.

        Args:
            content: Original message content

        Returns:
            Formatted message content
        """
        return content

    def get_max_message_length(self) -> Optional[int]:
        """
        Get maximum message length for this channel.

        Returns:
            Maximum message length in characters, or None if unlimited
        """
        return None

    def split_long_message(self, content: MessageContent) -> List[MessageContent]:
        """
        Split a long message into multiple parts if needed.

        Args:
            content: Message content to split

        Returns:
            List of message parts (single item if no splitting needed)
        """
        max_length = self.get_max_message_length()
        if not max_length or len(content.text) <= max_length:
            return [content]

        # Simple text splitting - channels can override for smarter splitting
        parts = []
        text = content.text

        while len(text) > max_length:
            # Find a good break point (space, newline, etc.)
            break_point = max_length
            for i in range(max_length - 1, max_length - 100, -1):
                if i < len(text) and text[i] in [' ', '\n', '\t', '.', ',', ';']:
                    break_point = i
                    break

            part_text = text[:break_point].strip()
            if part_text:
                parts.append(MessageContent(
                    text=part_text,
                    subject=content.subject,
                    metadata=content.metadata
                ))

            text = text[break_point:].strip()

        if text:
            parts.append(MessageContent(
                text=text,
                subject=content.subject,
                html=content.html,  # Only include HTML in last part
                attachments=content.attachments,  # Only include attachments in last part
                metadata=content.metadata
            ))

        return parts or [content]

    async def send_message_with_retry(
        self,
        recipient: str,
        content: MessageContent,
        message_id: Optional[str] = None,
        priority: str = "NORMAL",
        max_retries: int = 3
    ) -> DeliveryResult:
        """
        Send message with automatic retry on failure.

        Args:
            recipient: Recipient identifier
            content: Message content
            message_id: Optional message ID
            priority: Message priority
            max_retries: Maximum retry attempts

        Returns:
            Final delivery result
        """
        last_result = None

        for attempt in range(max_retries + 1):
            try:
                result = await self.send_message(recipient, content, message_id, priority)

                if result.is_successful:
                    if attempt > 0:
                        self._logger.info(
                            "Message delivered successfully on attempt %s for channel %s",
                            attempt + 1, self.channel_name
                        )
                    return result

                if not result.is_retryable:
                    self._logger.warning(
                        "Message delivery failed with non-retryable error for channel %s: %s",
                        self.channel_name, result.error_message
                    )
                    return result

                last_result = result

                if attempt < max_retries:
                    self._logger.warning(
                        "Message delivery failed for channel %s (attempt %s/%s): %s",
                        self.channel_name, attempt + 1, max_retries + 1, result.error_message
                    )

            except Exception as e:
                error_msg = f"Unexpected error during delivery: {str(e)}"
                last_result = DeliveryResult(
                    success=False,
                    status=DeliveryStatus.FAILED,
                    error_message=error_msg
                )

                if attempt < max_retries:
                    self._logger.error(
                        "Unexpected error in channel %s (attempt %s/%s): %s",
                        self.channel_name, attempt + 1, max_retries + 1, e
                    )

        self._logger.error(
            "Message delivery failed after %s attempts for channel %s",
            max_retries + 1, self.channel_name
        )

        return last_result or DeliveryResult(
            success=False,
            status=DeliveryStatus.FAILED,
            error_message="All retry attempts failed"
        )

    def __str__(self) -> str:
        """String representation of the channel."""
        return f"{self.__class__.__name__}(name='{self.channel_name}')"

    def __repr__(self) -> str:
        """Detailed string representation of the channel."""
        return f"{self.__class__.__name__}(name='{self.channel_name}', config_keys={list(self.config.keys())})"


class ChannelRegistry:
    """
    Registry for managing notification channel plugins.

    Provides plugin discovery, loading, and management functionality.
    """

    def __init__(self):
        """Initialize the channel registry."""
        self._channels: Dict[str, type] = {}
        self._instances: Dict[str, NotificationChannel] = {}
        self._logger = setup_logger(f"{__name__}.ChannelRegistry")

    def register_channel(self, channel_name: str, channel_class: type) -> None:
        """
        Register a channel plugin class.

        Args:
            channel_name: Name of the channel
            channel_class: Channel class that inherits from NotificationChannel

        Raises:
            ValueError: If channel class is invalid
        """
        if not issubclass(channel_class, NotificationChannel):
            raise ValueError(f"Channel class must inherit from NotificationChannel")

        self._channels[channel_name] = channel_class
        self._logger.info("Registered channel plugin: %s", channel_name)

    def unregister_channel(self, channel_name: str) -> None:
        """
        Unregister a channel plugin.

        Args:
            channel_name: Name of the channel to unregister
        """
        if channel_name in self._channels:
            del self._channels[channel_name]

        if channel_name in self._instances:
            del self._instances[channel_name]

        self._logger.info("Unregistered channel plugin: %s", channel_name)

    def get_channel(self, channel_name: str, config: Dict[str, Any]) -> NotificationChannel:
        """
        Get or create a channel instance.

        Args:
            channel_name: Name of the channel
            config: Channel configuration

        Returns:
            Channel instance

        Raises:
            ValueError: If channel is not registered
        """
        if channel_name not in self._channels:
            raise ValueError(f"Channel '{channel_name}' is not registered")

        # Create new instance each time to ensure fresh config
        channel_class = self._channels[channel_name]
        instance = channel_class(channel_name, config)

        self._instances[channel_name] = instance
        return instance

    def list_channels(self) -> List[str]:
        """
        List all registered channel names.

        Returns:
            List of registered channel names
        """
        return list(self._channels.keys())

    def is_registered(self, channel_name: str) -> bool:
        """
        Check if a channel is registered.

        Args:
            channel_name: Name of the channel

        Returns:
            True if channel is registered, False otherwise
        """
        return channel_name in self._channels

    async def check_all_health(self) -> Dict[str, ChannelHealth]:
        """
        Check health of all active channel instances.

        Returns:
            Dictionary mapping channel names to health status
        """
        health_results = {}

        for channel_name, instance in self._instances.items():
            try:
                health = await instance.check_health()
                health_results[channel_name] = health
            except Exception as e:
                self._logger.error("Health check failed for channel %s: %s", channel_name, e)
                health_results[channel_name] = ChannelHealth(
                    status=ChannelHealthStatus.DOWN,
                    last_check=datetime.now(timezone.utc),
                    error_message=str(e)
                )

        return health_results


# Global channel registry instance
channel_registry = ChannelRegistry()