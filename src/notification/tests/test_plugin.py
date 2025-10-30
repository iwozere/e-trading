"""
Test Channel Plugin

A simple test implementation of the notification channel interface.
Used for testing and as an example for plugin development.
"""

from typing import Dict, Any
from datetime import datetime, timezone
import asyncio

from src.notification.channels.base import (
    NotificationChannel, DeliveryResult, ChannelHealth, MessageContent,
    DeliveryStatus, ChannelHealthStatus
)
from src.notification.channels.config import ConfigValidator, CommonValidationRules


class TestChannel(NotificationChannel):
    """
    Test notification channel for testing and development.

    This channel simulates message delivery without actually sending anything.
    Useful for testing the notification system and as a plugin example.
    """

    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate test channel configuration.

        Args:
            config: Configuration dictionary to validate

        Raises:
            ValueError: If configuration is invalid
        """
        validator = ConfigValidator(self.channel_name)

        # Add validation rules
        validator.optional_field(
            "simulate_delay_ms",
            int,
            description="Simulated delivery delay in milliseconds",
            min_value=0,
            max_value=5000
        )

        validator.optional_field(
            "failure_rate",
            float,
            description="Simulated failure rate (0.0 to 1.0)",
            min_value=0.0,
            max_value=1.0
        )

        validator.optional_field(
            "max_message_length",
            int,
            description="Maximum message length for testing splitting",
            min_value=10,
            max_value=10000
        )

        validator.add_rule(CommonValidationRules.timeout_seconds())
        validator.add_rule(CommonValidationRules.rate_limit(default=100))
        validator.add_rule(CommonValidationRules.max_retries())

        # Validate and update config
        validated_config = validator.validate(config)
        self.config.update(validated_config)

    async def send_message(
        self,
        recipient: str,
        content: MessageContent,
        message_id: str = None,
        priority: str = "NORMAL"
    ) -> DeliveryResult:
        """
        Simulate sending a message.

        Args:
            recipient: Recipient identifier
            content: Message content to send
            message_id: Optional message ID for tracking
            priority: Message priority

        Returns:
            DeliveryResult with simulated delivery status
        """
        start_time = datetime.now(timezone.utc)

        # Simulate processing delay
        delay_ms = self.config.get("simulate_delay_ms", 100)
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000.0)

        # Simulate failure rate
        import random
        failure_rate = self.config.get("failure_rate", 0.0)

        if random.random() < failure_rate:
            return DeliveryResult(
                success=False,
                status=DeliveryStatus.FAILED,
                error_message="Simulated delivery failure",
                response_time_ms=delay_ms
            )

        # Simulate successful delivery
        response_time = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)

        return DeliveryResult(
            success=True,
            status=DeliveryStatus.DELIVERED,
            external_id=f"test_{message_id or 'unknown'}_{int(datetime.now(timezone.utc).timestamp())}",
            response_time_ms=response_time,
            metadata={
                "recipient": recipient,
                "message_length": len(content.text),
                "priority": priority,
                "simulated": True
            }
        )

    async def check_health(self) -> ChannelHealth:
        """
        Check test channel health (always healthy).

        Returns:
            ChannelHealth with healthy status
        """
        return ChannelHealth(
            status=ChannelHealthStatus.HEALTHY,
            last_check=datetime.now(timezone.utc),
            response_time_ms=self.config.get("simulate_delay_ms", 100),
            metadata={
                "simulated": True,
                "config": self.config
            }
        )

    def get_rate_limit(self) -> int:
        """
        Get rate limit for test channel.

        Returns:
            Rate limit in messages per minute
        """
        return self.config.get("rate_limit_per_minute", 100)

    def supports_feature(self, feature: str) -> bool:
        """
        Check if test channel supports a feature.

        Args:
            feature: Feature name to check

        Returns:
            True if feature is supported
        """
        supported_features = {
            "html": True,
            "attachments": True,
            "replies": False,
            "formatting": True,
            "splitting": True,
            "priority": True
        }

        return supported_features.get(feature, False)

    def get_max_message_length(self) -> int:
        """
        Get maximum message length for testing.

        Returns:
            Maximum message length in characters
        """
        return self.config.get("max_message_length", 1000)

    def format_message(self, content: MessageContent) -> MessageContent:
        """
        Format message for test channel (adds test prefix).

        Args:
            content: Original message content

        Returns:
            Formatted message content
        """
        formatted_text = f"[TEST] {content.text}"

        return MessageContent(
            text=formatted_text,
            subject=content.subject,
            html=f"<p><strong>[TEST]</strong> {content.html}</p>" if content.html else None,
            attachments=content.attachments,
            metadata=content.metadata
        )