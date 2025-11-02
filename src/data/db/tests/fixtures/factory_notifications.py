"""
Test data factories for Notification models.

Provides factory functions to create test data for Message and MessageDeliveryStatus models.
"""
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List

from src.data.db.models.model_notification import MessagePriority, MessageStatus, DeliveryStatus


class MessageFactory:
    """Factory for creating Message test data."""

    @staticmethod
    def create_data(
        message_type: str = "alert",
        priority: str = "NORMAL",
        channels: Optional[List[str]] = None,
        recipient_id: str = "user_123",
        content: Optional[Dict[str, Any]] = None,
        status: str = "PENDING",
        **kwargs
    ) -> Dict[str, Any]:
        """Create message data dictionary."""
        return {
            "message_type": message_type,
            "priority": priority,
            "channels": channels or ["telegram", "email"],
            "recipient_id": recipient_id,
            "content": content or {"title": "Test", "body": "Test message"},
            "status": status,
            "scheduled_for": datetime.now(timezone.utc),
            "max_retries": 3,
            "retry_count": 0,
            "message_metadata": {},
            **kwargs
        }

    @staticmethod
    def alert_message(recipient_id: str = "user_123", priority: str = "HIGH") -> Dict[str, Any]:
        """Create an alert message."""
        return MessageFactory.create_data(
            message_type="alert",
            priority=priority,
            channels=["telegram"],
            recipient_id=recipient_id,
            content={
                "title": "Price Alert",
                "body": "AAPL reached $150",
                "ticker": "AAPL"
            }
        )

    @staticmethod
    def screener_result_message(recipient_id: str = "user_123") -> Dict[str, Any]:
        """Create a screener result message."""
        return MessageFactory.create_data(
            message_type="screener_result",
            priority="NORMAL",
            channels=["telegram", "email"],
            recipient_id=recipient_id,
            content={
                "title": "Screener Results",
                "body": "Found 5 stocks matching criteria",
                "results_count": 5
            }
        )

    @staticmethod
    def critical_alert(recipient_id: str = "user_123") -> Dict[str, Any]:
        """Create a critical alert message."""
        return MessageFactory.create_data(
            message_type="system_alert",
            priority="CRITICAL",
            channels=["telegram", "email", "sms"],
            recipient_id=recipient_id,
            content={
                "title": "Critical System Alert",
                "body": "Trading bot encountered critical error",
                "severity": "critical"
            }
        )

    @staticmethod
    def scheduled_report(recipient_id: str = "user_123", scheduled_for: Optional[datetime] = None) -> Dict[str, Any]:
        """Create a scheduled report message."""
        return MessageFactory.create_data(
            message_type="report",
            priority="LOW",
            channels=["email"],
            recipient_id=recipient_id,
            content={
                "title": "Daily Report",
                "body": "Your daily trading summary",
                "report_type": "daily"
            },
            scheduled_for=scheduled_for or (datetime.now(timezone.utc) + timedelta(hours=1))
        )


class DeliveryStatusFactory:
    """Factory for creating MessageDeliveryStatus test data."""

    @staticmethod
    def create_data(
        message_id: int = 1,
        channel: str = "telegram",
        status: str = "PENDING",
        **kwargs
    ) -> Dict[str, Any]:
        """Create delivery status data dictionary."""
        return {
            "message_id": message_id,
            "channel": channel,
            "status": status,
            "created_at": datetime.now(timezone.utc),
            **kwargs
        }

    @staticmethod
    def pending_delivery(message_id: int, channel: str = "telegram") -> Dict[str, Any]:
        """Create a pending delivery status."""
        return DeliveryStatusFactory.create_data(
            message_id=message_id,
            channel=channel,
            status=DeliveryStatus.PENDING.value
        )

    @staticmethod
    def delivered_status(message_id: int, channel: str = "telegram", response_time_ms: int = 150) -> Dict[str, Any]:
        """Create a delivered status."""
        return DeliveryStatusFactory.create_data(
            message_id=message_id,
            channel=channel,
            status=DeliveryStatus.DELIVERED.value,
            delivered_at=datetime.now(timezone.utc),
            response_time_ms=response_time_ms,
            external_id=f"ext_{channel}_{message_id}"
        )

    @staticmethod
    def failed_delivery(message_id: int, channel: str = "telegram", error: str = "Connection timeout") -> Dict[str, Any]:
        """Create a failed delivery status."""
        return DeliveryStatusFactory.create_data(
            message_id=message_id,
            channel=channel,
            status=DeliveryStatus.FAILED.value,
            error_message=error
        )


# Convenient aliases
message_factory = MessageFactory()
delivery_status_factory = DeliveryStatusFactory()
