"""
Health Check API Routes
-----------------------

Unified health check endpoints for all system services.

Consolidates:
- Main API health check
- Notification service health
- Telegram bot health
- Channel ownership and queue status
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
from datetime import datetime, timezone
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.api.auth import get_current_user
from src.data.db.models.model_users import User
from src.api.services.telegram_health_service import TelegramHealthService
from src.api.services.notification_health_service import NotificationHealthService
from src.notification.logger import setup_logger

# For database connectivity check
from src.data.db.services.database_service import get_database_service
from src.data.db.models.model_notification import Message
from src.data.db.services.health_monitoring_service import HealthMonitoringService

_logger = setup_logger(__name__)

# Create router
router = APIRouter(prefix="/api/health", tags=["health"])

# Initialize health services
telegram_health_service = TelegramHealthService()
notification_health_service = NotificationHealthService()


@router.get("")
async def health_check():
    """
    Basic health check endpoint.

    Tests:
    - API is running
    - Database connectivity

    This is a public endpoint (no authentication required) for monitoring tools.
    """
    try:
        # Test database connection
        db_service = get_database_service()

        with db_service.uow() as uow:
            # Simple query to test database connectivity
            message_count = uow.s.query(Message).count()

        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "database": "connected",
            "api": "operational",
            "total_messages": message_count
        }

    except Exception as e:
        _logger.exception("Health check failed:")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
            "api": "operational",
            "database": "unavailable"
        }


@router.get("/channels")
async def get_channels_health(current_user: User = Depends(get_current_user)):
    """
    Get comprehensive health status for all notification channels.

    Shows:
    - Which service owns which channels (Telegram Bot vs Notification Service)
    - Queue status for each service
    - Pending/processing/failed message counts
    - Overall system health

    This is the main endpoint for operational monitoring of the notification system.
    """
    try:
        # Get health status from both services
        telegram_health = telegram_health_service.get_health_status()
        notification_health = notification_health_service.get_health_status()

        # Determine overall health
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

    except Exception:
        _logger.exception("Error getting channels health:")
        raise HTTPException(status_code=500, detail="Failed to get channels health")


@router.get("/telegram")
async def get_telegram_bot_health(current_user: User = Depends(get_current_user)):
    """
    Get Telegram Bot health status.

    Shows:
    - Channel ownership (telegram)
    - Queue status (pending, processing, failed, delivered)
    - Stuck message detection
    """
    try:
        health_status = telegram_health_service.get_health_status()
        return health_status

    except Exception:
        _logger.exception("Error getting Telegram bot health:")
        raise HTTPException(status_code=500, detail="Failed to get Telegram bot health")


@router.get("/notification")
async def get_notification_service_health(current_user: User = Depends(get_current_user)):
    """
    Get Notification Service health status.

    Shows:
    - Channel ownership (email, sms)
    - Queue status (pending, processing, failed, delivered)
    - Stuck message detection
    """
    try:
        health_status = notification_health_service.get_health_status()
        return health_status

    except Exception:
        _logger.exception("Error getting notification service health:")
        raise HTTPException(status_code=500, detail="Failed to get notification service health")


@router.get("/database")
async def get_database_health(current_user: User = Depends(get_current_user)):
    """
    Get database health and connectivity status.

    Shows:
    - Connection status
    - Message table statistics
    - Queue health metrics
    """
    try:
        from src.data.db.services.health_monitoring_service import HealthMonitoringService

        health_monitoring_service = HealthMonitoringService()
        return health_monitoring_service.get_database_health()

    except Exception:
        _logger.exception("Error getting database health:")
        raise HTTPException(status_code=500, detail="Failed to get database health")
