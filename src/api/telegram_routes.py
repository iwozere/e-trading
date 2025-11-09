#!/usr/bin/env python3
"""
Telegram Bot Management API Routes
---------------------------------

FastAPI router for Telegram bot management endpoints.
Provides REST API for user management, alerts, schedules, broadcasts, and audit logs.

Features:
- User management (list, verify, approve, reset)
- Alert management (list, toggle, delete, create)
- Schedule management (list, toggle, delete, create)
- Broadcast messaging
- Audit logging and statistics
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.security import HTTPBearer
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.api.auth import get_current_user, require_admin
from src.data.db.models.model_users import User
from src.api.services.telegram_app_service import TelegramAppService

_logger = setup_logger(__name__)

# Create router
router = APIRouter(prefix="/api/telegram", tags=["telegram"])

# Initialize application service
telegram_app_service = TelegramAppService()

# Security
security = HTTPBearer()

# Pydantic models for API
from pydantic import BaseModel, Field

class TelegramUser(BaseModel):
    """Telegram user model."""
    telegram_user_id: str
    email: Optional[str] = None
    verified: bool = False
    approved: bool = False
    language: str = "en"
    is_admin: bool = False
    max_alerts: int = 5
    max_schedules: int = 5
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

class UserStats(BaseModel):
    """User statistics model."""
    total_users: int
    verified_users: int
    approved_users: int
    pending_approvals: int
    admin_users: int

class TelegramAlert(BaseModel):
    """Telegram alert model."""
    id: int
    user_id: str
    ticker: str
    price: Optional[float] = None
    condition: str
    active: bool = True
    email: bool = False
    alert_type: Optional[str] = None
    timeframe: Optional[str] = None
    config_json: Optional[str] = None
    alert_action: Optional[str] = None
    re_arm_config: Optional[str] = None
    is_armed: bool = True
    last_price: Optional[float] = None
    last_triggered_at: Optional[str] = None
    created: Optional[str] = None

class AlertStats(BaseModel):
    """Alert statistics model."""
    total_alerts: int
    active_alerts: int
    triggered_today: int
    rearm_cycles: int

class TelegramSchedule(BaseModel):
    """Telegram schedule model."""
    id: int
    user_id: str
    ticker: str
    scheduled_time: str
    period: Optional[str] = None
    active: bool = True
    email: bool = False
    indicators: Optional[str] = None
    interval: Optional[str] = None
    provider: Optional[str] = None
    schedule_type: Optional[str] = None
    list_type: Optional[str] = None
    config_json: Optional[str] = None
    schedule_config: Optional[str] = None
    created: Optional[str] = None

class ScheduleStats(BaseModel):
    """Schedule statistics model."""
    total_schedules: int
    active_schedules: int
    executed_today: int
    failed_executions: int

class CommandAudit(BaseModel):
    """Command audit model."""
    id: int
    telegram_user_id: str
    command: str
    full_message: Optional[str] = None
    is_registered_user: bool = False
    user_email: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    response_time_ms: Optional[int] = None
    created: str

class AuditStats(BaseModel):
    """Audit statistics model."""
    total_commands: int
    successful_commands: int
    failed_commands: int
    recent_activity_24h: int
    top_commands: List[Dict[str, Any]]

class BroadcastMessage(BaseModel):
    """Broadcast message model."""
    message: str = Field(..., description="Message to broadcast")

class BroadcastResult(BaseModel):
    """Broadcast result model."""
    message: str
    total_recipients: int
    successful_deliveries: int
    failed_deliveries: int

# --- USER MANAGEMENT ENDPOINTS ---

@router.get("/users")
async def get_telegram_users(
    status: Optional[str] = Query(None, description="Filter users by status: all, verified, approved, pending"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Items per page"),
    current_user: User = Depends(require_admin)
):
    """
    Get list of Telegram bot users with optional filtering and pagination.

    Filters:
    - all: All users (default)
    - verified: Only verified users
    - approved: Only approved users
    - pending: Users pending approval
    """
    try:
        _logger.info("Getting Telegram users with status: %s, page: %d, page_size: %d", status, page, page_size)

        # Use application service
        users_data = telegram_app_service.get_users_list(status)

        # Apply pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_users = users_data[start_idx:end_idx]

        # Convert to response model
        result_users = [
            TelegramUser(**user_data)
            for user_data in paginated_users
        ]

        # Create paginated response
        result = {
            "data": result_users,
            "total": len(users_data),
            "page": page,
            "limit": page_size,
            "has_more": end_idx < len(users_data)
        }

        _logger.info("Retrieved %d Telegram users (page %d of %d total)", len(result_users), page, len(users_data))
        return result

    except Exception as e:
        _logger.exception("Error getting Telegram users:")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/users/{user_id}/verify")
async def verify_telegram_user(
    user_id: str,
    current_user: User = Depends(require_admin)
):
    """Manually verify a Telegram user's email."""
    try:
        _logger.info("Manually verifying Telegram user: %s", user_id)

        # Use application service
        result = telegram_app_service.verify_user(user_id)

        _logger.info("Successfully verified Telegram user: %s", user_id)
        return result

    except Exception as e:
        _logger.exception("Error verifying Telegram user %s:", user_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/users/{user_id}/approve")
async def approve_telegram_user(
    user_id: str,
    current_user: User = Depends(require_admin)
):
    """Approve a Telegram user for access."""
    try:
        _logger.info("Approving Telegram user: %s", user_id)

        # Use application service to approve user
        result = telegram_app_service.approve_user(user_id)

        _logger.info("Successfully approved Telegram user: %s", user_id)
        return result

    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("Error approving Telegram user %s:", user_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/users/{user_id}/reset-email")
async def reset_telegram_user_email(
    user_id: str,
    current_user: User = Depends(require_admin)
):
    """Reset a Telegram user's email verification status."""
    try:
        _logger.info("Resetting email for Telegram user: %s", user_id)

        # Use application service to reset user email
        result = telegram_app_service.reset_user_email(user_id)

        _logger.info("Successfully reset email for Telegram user: %s", user_id)
        return result

    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("Error resetting email for Telegram user %s:", user_id)
        raise HTTPException(status_code=500, detail=str(e))


# --- STATISTICS ENDPOINTS ---

@router.get("/stats/users", response_model=UserStats)
async def get_telegram_user_stats(
    current_user: User = Depends(get_current_user)
):
    """Get Telegram user statistics."""
    try:
        _logger.info("Getting Telegram user statistics")

        # Use application service
        stats_data = telegram_app_service.get_user_stats()
        stats = UserStats(**stats_data)

        _logger.info("Retrieved Telegram user statistics: %s", stats.dict())
        return stats

    except Exception as e:
        _logger.exception("Error getting Telegram user statistics:")
        raise HTTPException(status_code=500, detail=str(e))
# --- ALERT MANAGEMENT ENDPOINTS ---

@router.get("/alerts")
async def get_telegram_alerts(
    status: Optional[str] = Query(None, description="Filter alerts by status: all, active, inactive"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Items per page"),
    current_user: User = Depends(get_current_user)
):
    """
    Get list of Telegram bot alerts with optional filtering and pagination.

    Filters:
    - all: All alerts (default)
    - active: Only active alerts
    - inactive: Only inactive alerts
    """
    try:
        _logger.info("Getting Telegram alerts with status: %s, page: %d, page_size: %d", status, page, page_size)

        # Use application service to get alerts
        alerts_data = telegram_app_service.get_alerts_list(status, page, page_size)

        # Convert to response model
        result_alerts = [
            TelegramAlert(**alert_data)
            for alert_data in alerts_data
        ]

        # Create paginated response
        result = {
            "data": result_alerts,
            "total": len(alerts_data),  # This should be updated in the service to return total count
            "page": page,
            "limit": page_size,
            "has_more": len(result_alerts) == page_size  # Simple check
        }

        _logger.info("Retrieved %d Telegram alerts", len(result_alerts))
        return result

    except Exception as e:
        _logger.exception("Error getting Telegram alerts:")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/{alert_id}/toggle")
async def toggle_telegram_alert(
    alert_id: int,
    current_user: User = Depends(require_admin)
):
    """Toggle a Telegram alert's active status."""
    try:
        _logger.info("Toggling Telegram alert: %d", alert_id)

        # Use application service to toggle alert
        result = telegram_app_service.toggle_alert(alert_id)

        _logger.info("Successfully toggled Telegram alert: %d", alert_id)
        return result

    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("Error toggling Telegram alert %d:", alert_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/alerts/{alert_id}")
async def delete_telegram_alert(
    alert_id: int,
    current_user: User = Depends(require_admin)
):
    """Delete a Telegram alert."""
    try:
        _logger.info("Deleting Telegram alert: %d", alert_id)

        # Use application service to delete alert
        result = telegram_app_service.delete_alert(alert_id)

        _logger.info("Successfully deleted Telegram alert: %d", alert_id)
        return result

    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("Error deleting Telegram alert %d:", alert_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts/{alert_id}/config")
async def get_telegram_alert_config(
    alert_id: int,
    current_user: User = Depends(get_current_user)
):
    """Get detailed configuration for a Telegram alert."""
    try:
        _logger.info("Getting Telegram alert config: %d", alert_id)

        # For now, return a simple response since this endpoint needs more complex implementation
        # TODO: Implement proper alert config retrieval through telegram_app_service
        raise HTTPException(status_code=501, detail="Alert config endpoint not yet implemented")

        _logger.info("Retrieved Telegram alert config: %d", alert_id)
        return config

    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("Error getting Telegram alert config %d:", alert_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/alerts", response_model=AlertStats)
async def get_telegram_alert_stats(
    current_user: User = Depends(get_current_user)
):
    """Get Telegram alert statistics."""
    try:
        _logger.info("Getting Telegram alert statistics")

        # Use application service to get alert stats
        stats_data = telegram_app_service.get_alert_stats()
        stats = AlertStats(**stats_data)

        _logger.info("Retrieved Telegram alert statistics: %s", stats.dict())
        return stats

    except Exception as e:
        _logger.exception("Error getting Telegram alert statistics:")
        raise HTTPException(status_code=500, detail=str(e))


# --- SCHEDULE MANAGEMENT ENDPOINTS ---

@router.get("/schedules")
async def get_telegram_schedules(
    status: Optional[str] = Query(None, description="Filter schedules by status: all, active, inactive"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Items per page"),
    current_user: User = Depends(get_current_user)
):
    """
    Get list of Telegram bot schedules with optional filtering and pagination.

    Filters:
    - all: All schedules (default)
    - active: Only active schedules
    - inactive: Only inactive schedules
    """
    try:
        _logger.info("Getting Telegram schedules with status: %s, page: %d, page_size: %d", status, page, page_size)

        # Use application service to get schedules
        schedules_data = telegram_app_service.get_schedules_list(status)

        # Apply pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_schedules = schedules_data[start_idx:end_idx]

        # Convert to response model
        result_schedules = [
            TelegramSchedule(**schedule_data)
            for schedule_data in paginated_schedules
        ]

        # Create paginated response
        result = {
            "data": result_schedules,
            "total": len(schedules_data),
            "page": page,
            "limit": page_size,
            "has_more": end_idx < len(schedules_data)
        }

        _logger.info("Retrieved %d Telegram schedules (page %d of %d total)", len(result_schedules), page, len(schedules_data))
        return result

    except Exception as e:
        _logger.exception("Error getting Telegram schedules:")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/schedules/{schedule_id}/toggle")
async def toggle_telegram_schedule(
    schedule_id: int,
    current_user: User = Depends(require_admin)
):
    """Toggle a Telegram schedule's active status."""
    try:
        _logger.info("Toggling Telegram schedule: %d", schedule_id)

        # TODO: Implement schedule toggle through telegram_app_service
        raise HTTPException(status_code=501, detail="Schedule toggle endpoint not yet implemented")

    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("Error toggling Telegram schedule %d:", schedule_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/schedules/{schedule_id}")
async def delete_telegram_schedule(
    schedule_id: int,
    current_user: User = Depends(require_admin)
):
    """Delete a Telegram schedule."""
    try:
        _logger.info("Deleting Telegram schedule: %d", schedule_id)

        # TODO: Implement schedule deletion through telegram_app_service
        raise HTTPException(status_code=501, detail="Schedule deletion endpoint not yet implemented")

    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("Error deleting Telegram schedule %d:", schedule_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/schedules/{schedule_id}")
async def update_telegram_schedule(
    schedule_id: int,
    schedule_data: Dict[str, Any],
    current_user: User = Depends(require_admin)
):
    """Update a Telegram schedule."""
    try:
        _logger.info("Updating Telegram schedule: %d", schedule_id)

        # TODO: Implement schedule update through telegram_app_service
        raise HTTPException(status_code=501, detail="Schedule update endpoint not yet implemented")

    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("Error updating Telegram schedule %d:", schedule_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/schedules", response_model=ScheduleStats)
async def get_telegram_schedule_stats(
    current_user: User = Depends(get_current_user)
):
    """Get Telegram schedule statistics."""
    try:
        _logger.info("Getting Telegram schedule statistics")

        # Use application service to get schedule stats
        stats_data = telegram_app_service.get_schedule_stats()
        stats = ScheduleStats(**stats_data)

        _logger.info("Retrieved Telegram schedule statistics: %s", stats.dict())
        return stats

    except Exception as e:
        _logger.exception("Error getting Telegram schedule statistics:")
        raise HTTPException(status_code=500, detail=str(e))


# --- BROADCAST MESSAGING ENDPOINTS ---

@router.post("/broadcast", response_model=BroadcastResult)
async def send_telegram_broadcast(
    broadcast: BroadcastMessage,
    current_user: User = Depends(require_admin)
):
    """Send broadcast message to all Telegram bot users."""
    try:
        _logger.info("Sending Telegram broadcast message")

        # Use application service to send broadcast
        result_data = telegram_app_service.send_broadcast(broadcast.message)

        # Log the broadcast for audit purposes
        _logger.info("Broadcast sent to %d users by admin %s", result_data["total_recipients"], current_user.username)

        result = BroadcastResult(**result_data)

        _logger.info("Broadcast result: %s", result.dict())
        return result

    except Exception as e:
        _logger.exception("Error sending Telegram broadcast:")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/broadcast/history")
async def get_telegram_broadcast_history(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(50, ge=1, le=100, description="Items per page"),
    current_user: User = Depends(get_current_user)
):
    """Get broadcast message history with pagination."""
    try:
        _logger.info("Getting Telegram broadcast history - page: %d, limit: %d", page, limit)

        # Calculate offset for pagination
        offset = (page - 1) * limit

        # Use application service to get broadcast history
        history_data = telegram_app_service.get_broadcast_history(limit=limit, offset=offset)

        # Create paginated response
        result = {
            "data": history_data,
            "total": len(history_data),  # This is approximate since we don't have total count
            "page": page,
            "limit": limit,
            "has_more": len(history_data) == limit  # Simple check
        }

        _logger.info("Retrieved %d broadcast history entries", len(history_data))
        return result

    except Exception as e:
        _logger.exception("Error getting Telegram broadcast history:")
        raise HTTPException(status_code=500, detail=str(e))


# --- AUDIT LOGGING ENDPOINTS ---

@router.get("/audit")
async def get_telegram_audit_logs(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Items per page"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    command: Optional[str] = Query(None, description="Filter by command"),
    success_only: Optional[bool] = Query(None, description="Filter by success status"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    current_user: User = Depends(get_current_user)
):
    """
    Get Telegram bot command audit logs with filtering and pagination.

    Supports filtering by:
    - user_id: Specific Telegram user ID
    - command: Specific command name
    - success_only: Only successful commands (true/false)
    - start_date/end_date: Date range filter
    """
    try:
        _logger.info("Getting Telegram audit logs - page: %d, page_size: %d", page, page_size)

        # Calculate offset for pagination
        offset = (page - 1) * page_size

        # Use application service to get audit logs
        logs_data = telegram_app_service.get_audit_logs(
            limit=page_size,
            offset=offset,
            user_id=user_id,
            command=command,
            success_only=success_only
        )

        # Create paginated response
        result = {
            "data": logs_data,
            "total": len(logs_data),  # This is approximate since we don't have total count
            "page": page,
            "limit": page_size,
            "has_more": len(logs_data) == page_size  # Simple check
        }

        _logger.info("Retrieved %d Telegram audit logs", len(logs_data))
        return result

    except Exception as e:
        _logger.exception("Error getting Telegram audit logs:")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/audit", response_model=AuditStats)
async def get_telegram_audit_stats(
    current_user: User = Depends(get_current_user)
):
    """Get Telegram command audit statistics."""
    try:
        _logger.info("Getting Telegram audit statistics")

        # Use application service to get audit stats
        stats_data = telegram_app_service.get_audit_stats()

        stats = AuditStats(
            total_commands=stats_data.get('total_commands', 0),
            successful_commands=stats_data.get('successful_commands', 0),
            failed_commands=stats_data.get('failed_commands', 0),
            recent_activity_24h=stats_data.get('recent_activity_24h', 0),
            top_commands=stats_data.get('top_commands', [])
        )

        _logger.info("Retrieved Telegram audit statistics: %s", stats.dict())
        return stats

    except Exception as e:
        _logger.exception("Error getting Telegram audit statistics:")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users/{user_id}/audit")
async def get_user_audit_logs(
    user_id: str,
    limit: int = Query(50, ge=1, le=100, description="Number of logs to retrieve"),
    current_user: User = Depends(get_current_user)
):
    """Get command audit logs for a specific user."""
    try:
        _logger.info("Getting audit logs for Telegram user: %s", user_id)

        # Use application service to get user audit logs
        logs_data = telegram_app_service.get_user_audit_logs(user_id, limit)

        # Create paginated response
        result = {
            "data": logs_data,
            "total": len(logs_data),
            "page": 1,
            "limit": limit,
            "has_more": len(logs_data) == limit
        }

        _logger.info("Retrieved %d audit logs for user %s", len(logs_data), user_id)
        return result

    except Exception as e:
        _logger.exception("Error getting audit logs for user %s:", user_id)
        raise HTTPException(status_code=500, detail=str(e))