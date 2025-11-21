"""
Telegram Health Service
-----------------------

Service for querying Telegram bot health from database service layer.

This service provides health monitoring for the Telegram Bot by delegating
to the database service layer, following the proper repository pattern.
"""

from typing import Dict, Any
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.data.db.services.health_monitoring_service import HealthMonitoringService
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class TelegramHealthService:
    """
    Service for monitoring Telegram bot health.

    Delegates to HealthMonitoringService in the database layer
    to follow proper separation of concerns.
    """

    def __init__(self):
        """Initialize the health service."""
        self.health_monitoring_service = HealthMonitoringService()

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get Telegram bot health status.

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
            return self.health_monitoring_service.get_telegram_health()

        except Exception:
            _logger.exception("Error getting Telegram health status:")
            # Return error status if database query fails
            from datetime import datetime, timezone
            return {
                "service": "telegram_bot",
                "status": "error",
                "status_reason": "Failed to query health status from database",
                "error": "Database query failed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "channels": {
                    "owned": ["telegram"],
                    "email_owned_by": "notification_service",
                    "sms_owned_by": "notification_service",
                    "telegram_status": "unknown"
                },
                "queue": {
                    "pending": -1,
                    "processing": -1,
                    "failed_last_hour": -1,
                    "delivered_last_hour": -1,
                    "stuck_messages": -1
                }
            }
