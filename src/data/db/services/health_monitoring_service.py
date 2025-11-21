"""
Health Monitoring Service

Service layer for monitoring notification system health across all channels.
Provides queue metrics, channel ownership status, and health assessments.
"""

from typing import Dict, Any, List
from datetime import datetime, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.data.db.services.base_service import BaseDBService, with_uow, handle_db_error
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class HealthMonitoringService(BaseDBService):
    """
    Service for monitoring notification system health.

    Provides health metrics for:
    - Telegram Bot (telegram channel)
    - Notification Service (email, sms channels)
    - Overall queue health
    """

    def __init__(self, db_service=None):
        """Initialize the health monitoring service."""
        super().__init__(db_service)

    @with_uow
    @handle_db_error
    def get_telegram_health(self) -> Dict[str, Any]:
        """
        Get Telegram Bot health status.

        Returns:
            Dictionary with:
            - service: Service name
            - status: "healthy", "degraded", or "error"
            - status_reason: Reason if degraded/error
            - timestamp: Current timestamp
            - channels: Channel ownership info
            - queue: Queue metrics
        """
        channels = ["telegram"]
        queue_metrics = self.uow.notifications.messages.get_queue_health_for_channels(channels)

        # Determine health status based on metrics
        status, status_reason = self._assess_health_status(queue_metrics)

        return {
            "service": "telegram_bot",
            "status": status,
            "status_reason": status_reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "channels": {
                "owned": ["telegram"],
                "email_owned_by": "notification_service",
                "sms_owned_by": "notification_service",
                "telegram_status": "enabled"
            },
            "queue": queue_metrics
        }

    @with_uow
    @handle_db_error
    def get_notification_service_health(self, enabled_channels: List[str] = None) -> Dict[str, Any]:
        """
        Get Notification Service health status.

        Args:
            enabled_channels: List of channels owned by notification service
                            (default: ["email"])

        Returns:
            Dictionary with:
            - service: Service name
            - status: "healthy", "degraded", or "error"
            - status_reason: Reason if degraded/error
            - timestamp: Current timestamp
            - channels: Channel ownership info
            - queue: Queue metrics
        """
        if enabled_channels is None:
            enabled_channels = ["email"]

        queue_metrics = self.uow.notifications.messages.get_queue_health_for_channels(enabled_channels)

        # Determine health status based on metrics
        status, status_reason = self._assess_health_status(queue_metrics)

        return {
            "service": "notification_service",
            "status": status,
            "status_reason": status_reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "channels": {
                "owned": enabled_channels,
                "telegram_owned_by": "telegram_bot",
                "email_status": "enabled" if "email" in enabled_channels else "disabled",
                "sms_status": "enabled" if "sms" in enabled_channels else "disabled"
            },
            "queue": queue_metrics
        }

    @with_uow
    @handle_db_error
    def get_comprehensive_health(self, notification_enabled_channels: List[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive health status for all services.

        Args:
            notification_enabled_channels: List of channels owned by notification service

        Returns:
            Dictionary with:
            - status: Overall status
            - timestamp: Current timestamp
            - services: Health info for each service
            - summary: Overall summary metrics
        """
        telegram_health = self.get_telegram_health()
        notification_health = self.get_notification_service_health(notification_enabled_channels)

        all_healthy = (
            telegram_health["status"] == "healthy" and
            notification_health["status"] == "healthy"
        )

        return {
            "status": "healthy" if all_healthy else "degraded",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "services": {
                "telegram_bot": telegram_health,
                "notification_service": notification_health
            },
            "summary": {
                "all_services_healthy": all_healthy,
                "total_pending_messages": (
                    telegram_health["queue"]["pending"] +
                    notification_health["queue"]["pending"]
                ),
                "channel_ownership": {
                    "telegram_bot": ["telegram"],
                    "notification_service": notification_health["channels"]["owned"]
                }
            }
        }

    @with_uow
    @handle_db_error
    def get_database_health(self) -> Dict[str, Any]:
        """
        Get database health and message statistics.

        Returns:
            Dictionary with:
            - status: Health status
            - timestamp: Current timestamp
            - database: Connection and message statistics
        """
        from src.data.db.models.model_notification import Message, MessageStatus

        try:
            # Get message statistics by status
            total_messages = self.uow.s.query(Message).count()

            pending_messages = self.uow.s.query(Message).filter(
                Message.status == MessageStatus.PENDING.value
            ).count()

            processing_messages = self.uow.s.query(Message).filter(
                Message.status == MessageStatus.PROCESSING.value
            ).count()

            delivered_messages = self.uow.s.query(Message).filter(
                Message.status == MessageStatus.DELIVERED.value
            ).count()

            failed_messages = self.uow.s.query(Message).filter(
                Message.status == MessageStatus.FAILED.value
            ).count()

            # Check for locked messages (currently being processed)
            locked_messages = self.uow.s.query(Message).filter(
                Message.locked_by.isnot(None)
            ).count()

            return {
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "database": {
                    "connection": "connected",
                    "total_messages": total_messages,
                    "pending_messages": pending_messages,
                    "processing_messages": processing_messages,
                    "delivered_messages": delivered_messages,
                    "failed_messages": failed_messages,
                    "locked_messages": locked_messages
                }
            }

        except Exception as e:
            _logger.exception("Error getting database health:")
            return {
                "status": "error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "database": {
                    "connection": "error",
                    "error": str(e)
                }
            }

    def _assess_health_status(self, queue_metrics: Dict[str, Any]) -> tuple[str, str]:
        """
        Assess health status based on queue metrics.

        Args:
            queue_metrics: Queue metrics dictionary

        Returns:
            Tuple of (status, status_reason)
        """
        stuck_messages = queue_metrics.get("stuck_messages", 0)
        pending = queue_metrics.get("pending", 0)
        failed_last_hour = queue_metrics.get("failed_last_hour", 0)
        delivered_last_hour = queue_metrics.get("delivered_last_hour", 0)

        # Check for stuck messages (highest priority)
        if stuck_messages > 0:
            return "degraded", f"{stuck_messages} messages stuck in processing"

        # Check for high queue backlog
        if pending > 100:
            return "degraded", f"High queue backlog: {pending} messages"

        # Check for high failure rate with no deliveries
        if failed_last_hour > 10 and delivered_last_hour == 0:
            return "degraded", f"High failure rate: {failed_last_hour} failures in last hour"

        # All checks passed
        return "healthy", None
