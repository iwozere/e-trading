"""
Notification Health Service
---------------------------

Service for querying Notification Service health from database service layer.

This service provides health monitoring for the Notification Service by delegating
to the database service layer, following the proper repository pattern.
"""

from typing import Dict, Any, List
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.data.db.services.health_monitoring_service import HealthMonitoringService
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class NotificationHealthService:
    """
    Service for monitoring Notification Service health.

    Delegates to HealthMonitoringService in the database layer
    to follow proper separation of concerns.
    """

    def __init__(self):
        """Initialize the health service."""
        self.health_monitoring_service = HealthMonitoringService()
        # Default to email only (can be made configurable via environment variable)
        # TODO: Load from config if NOTIFICATION_ENABLED_CHANNELS is set
        self.enabled_channels = ["email"]

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get Notification Service health status.

        Returns:
            Health status dictionary with:
            - service: Service name
            - status: "healthy", "degraded", or "error"
            - status_reason: Reason for degraded/error status
            - timestamp: Current timestamp
            - channels: Channel ownership information
            - queue: Queue metrics (pending, processing, failed, delivered)
        """
        try:
            return self.health_monitoring_service.get_notification_service_health(
                enabled_channels=self.enabled_channels
            )

        except Exception:
            _logger.exception("Error getting Notification Service health status:")
            # Return error status if database query fails
            from datetime import datetime, timezone
            return {
                "service": "notification_service",
                "status": "error",
                "status_reason": "Failed to query health status from database",
                "error": "Database query failed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "channels": {
                    "owned": self.enabled_channels,
                    "telegram_owned_by": "telegram_bot",
                    "email_status": "unknown",
                    "sms_status": "unknown"
                },
                "queue": {
                    "pending": -1,
                    "processing": -1,
                    "failed_last_hour": -1,
                    "delivered_last_hour": -1,
                    "stuck_messages": -1
                }
            }
