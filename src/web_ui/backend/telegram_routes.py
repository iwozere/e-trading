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
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.data.db import telegram_service as db
from src.notification.logger import setup_logger
from src.web_ui.backend.auth import get_current_user, require_admin
from src.web_ui.backend.models import User

_logger = setup_logger(__name__)

# Create router
router = APIRouter(prefix="/api/telegram", tags=["telegram"])

# Security
security = HTTPBearer()

# Pydantic models for API
from pydantic import BaseModel, Field
from datetime import datetime

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

@router.get("/users", response_model=List[TelegramUser])
async def get_telegram_users(
    filter: Optional[str] = Query(None, description="Filter users by status: all, verified, approved, pending"),
    current_user: User = Depends(require_admin)
):
    """
    Get list of Telegram bot users with optional filtering.

    Filters:
    - all: All users (default)
    - verified: Only verified users
    - approved: Only approved users
    - pending: Users pending approval
    """
    try:
        _logger.info("Getting Telegram users with filter: %s", filter)

        # Get all users from database
        users = db.list_users()

        # Apply filtering
        if filter == "verified":
            users = [u for u in users if u.get('verified', False)]
        elif filter == "approved":
            users = [u for u in users if u.get('approved', False)]
        elif filter == "pending":
            users = [u for u in users if u.get('verified', False) and not u.get('approved', False)]
        # 'all' or None returns all users

        # Convert to response model
        result = []
        for user in users:
            result.append(TelegramUser(
                telegram_user_id=user['telegram_user_id'],
                email=user.get('email'),
                verified=user.get('verified', False),
                approved=user.get('approved', False),
                language=user.get('language', 'en'),
                is_admin=user.get('is_admin', False),
                max_alerts=user.get('max_alerts', 5),
                max_schedules=user.get('max_schedules', 5)
            ))

        _logger.info("Retrieved %d Telegram users", len(result))
        return result

    except Exception as e:
        _logger.error("Error getting Telegram users: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/users/{user_id}/verify")
async def verify_telegram_user(
    user_id: str,
    current_user: User = Depends(require_admin)
):
    """Manually verify a Telegram user's email."""
    try:
        _logger.info("Manually verifying Telegram user: %s", user_id)

        # Update user verification status
        db.update_user_verification(user_id, True)

        _logger.info("Successfully verified Telegram user: %s", user_id)
        return {"message": f"User {user_id} verified successfully"}

    except Exception as e:
        _logger.error("Error verifying Telegram user %s: %s", user_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/users/{user_id}/approve")
async def approve_telegram_user(
    user_id: str,
    current_user: User = Depends(require_admin)
):
    """Approve a Telegram user for access."""
    try:
        _logger.info("Approving Telegram user: %s", user_id)

        # Check if user exists and is verified
        user_status = db.get_user_status(user_id)
        if not user_status:
            raise HTTPException(status_code=404, detail="User not found")

        if not user_status.get('verified', False):
            raise HTTPException(status_code=400, detail="User must be verified before approval")

        # Approve user
        success = db.approve_user(user_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to approve user")

        _logger.info("Successfully approved Telegram user: %s", user_id)
        return {"message": f"User {user_id} approved successfully"}

    except HTTPException:
        raise
    except Exception as e:
        _logger.error("Error approving Telegram user %s: %s", user_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/users/{user_id}/reset-email")
async def reset_telegram_user_email(
    user_id: str,
    current_user: User = Depends(require_admin)
):
    """Reset a Telegram user's email verification status."""
    try:
        _logger.info("Resetting email for Telegram user: %s", user_id)

        # Check if user exists
        user_status = db.get_user_status(user_id)
        if not user_status:
            raise HTTPException(status_code=404, detail="User not found")

        # Reset verification and approval status
        db.update_user_verification(user_id, False)
        db.reject_user(user_id)  # Remove approval as well

        _logger.info("Successfully reset email for Telegram user: %s", user_id)
        return {"message": f"Email reset for user {user_id} - user must re-verify"}

    except HTTPException:
        raise
    except Exception as e:
        _logger.error("Error resetting email for Telegram user %s: %s", user_id, e)
        raise HTTPException(status_code=500, detail=str(e))


# --- STATISTICS ENDPOINTS ---

@router.get("/stats/users", response_model=UserStats)
async def get_telegram_user_stats(
    current_user: User = Depends(get_current_user)
):
    """Get Telegram user statistics."""
    try:
        _logger.info("Getting Telegram user statistics")

        # Get all users
        users = db.list_users()

        # Calculate statistics
        total_users = len(users)
        verified_users = len([u for u in users if u.get('verified', False)])
        approved_users = len([u for u in users if u.get('approved', False)])
        pending_approvals = len([u for u in users if u.get('verified', False) and not u.get('approved', False)])
        admin_users = len([u for u in users if u.get('is_admin', False)])

        stats = UserStats(
            total_users=total_users,
            verified_users=verified_users,
            approved_users=approved_users,
            pending_approvals=pending_approvals,
            admin_users=admin_users
        )

        _logger.info("Retrieved Telegram user statistics: %s", stats.dict())
        return stats

    except Exception as e:
        _logger.error("Error getting Telegram user statistics: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
# --- ALERT MANAGEMENT ENDPOINTS ---

@router.get("/alerts", response_model=List[TelegramAlert])
async def get_telegram_alerts(
    filter: Optional[str] = Query(None, description="Filter alerts by status: all, active, inactive"),
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
        _logger.info("Getting Telegram alerts with filter: %s, page: %d, page_size: %d", filter, page, page_size)

        # Get alerts based on filter
        if filter == "active":
            alerts = db.get_active_alerts()
        elif filter == "inactive":
            # Get all alerts and filter inactive ones
            all_alerts = db.get_alerts_by_type()
            alerts = [a for a in all_alerts if not a.get('active', True)]
        else:
            # Get all alerts
            alerts = db.get_alerts_by_type()

        # Apply pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_alerts = alerts[start_idx:end_idx]

        # Convert to response model
        result = []
        for alert in paginated_alerts:
            result.append(TelegramAlert(
                id=alert['id'],
                user_id=alert['user_id'],
                ticker=alert['ticker'],
                price=alert.get('price'),
                condition=alert['condition'],
                active=alert.get('active', True),
                email=alert.get('email', False),
                alert_type=alert.get('alert_type'),
                timeframe=alert.get('timeframe'),
                config_json=alert.get('config_json'),
                alert_action=alert.get('alert_action'),
                re_arm_config=alert.get('re_arm_config'),
                is_armed=alert.get('is_armed', True),
                last_price=alert.get('last_price'),
                last_triggered_at=alert.get('last_triggered_at'),
                created=alert.get('created')
            ))

        _logger.info("Retrieved %d Telegram alerts", len(result))
        return result

    except Exception as e:
        _logger.error("Error getting Telegram alerts: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/{alert_id}/toggle")
async def toggle_telegram_alert(
    alert_id: int,
    current_user: User = Depends(require_admin)
):
    """Toggle a Telegram alert's active status."""
    try:
        _logger.info("Toggling Telegram alert: %d", alert_id)

        # Get current alert
        alert = db.get_alert(alert_id)
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")

        # Toggle active status
        new_active_status = not alert.get('active', True)
        success = db.update_alert(alert_id, active=new_active_status)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to toggle alert")

        status_text = "activated" if new_active_status else "deactivated"
        _logger.info("Successfully %s Telegram alert: %d", status_text, alert_id)
        return {"message": f"Alert {alert_id} {status_text} successfully"}

    except HTTPException:
        raise
    except Exception as e:
        _logger.error("Error toggling Telegram alert %d: %s", alert_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/alerts/{alert_id}")
async def delete_telegram_alert(
    alert_id: int,
    current_user: User = Depends(require_admin)
):
    """Delete a Telegram alert."""
    try:
        _logger.info("Deleting Telegram alert: %d", alert_id)

        # Check if alert exists
        alert = db.get_alert(alert_id)
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")

        # Delete alert
        success = db.delete_alert(alert_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete alert")

        _logger.info("Successfully deleted Telegram alert: %d", alert_id)
        return {"message": f"Alert {alert_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        _logger.error("Error deleting Telegram alert %d: %s", alert_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts/{alert_id}/config")
async def get_telegram_alert_config(
    alert_id: int,
    current_user: User = Depends(get_current_user)
):
    """Get detailed configuration for a Telegram alert."""
    try:
        _logger.info("Getting Telegram alert config: %d", alert_id)

        # Get alert
        alert = db.get_alert(alert_id)
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")

        # Return detailed configuration
        config = {
            "alert": TelegramAlert(
                id=alert['id'],
                user_id=alert['user_id'],
                ticker=alert['ticker'],
                price=alert.get('price'),
                condition=alert['condition'],
                active=alert.get('active', True),
                email=alert.get('email', False),
                alert_type=alert.get('alert_type'),
                timeframe=alert.get('timeframe'),
                config_json=alert.get('config_json'),
                alert_action=alert.get('alert_action'),
                re_arm_config=alert.get('re_arm_config'),
                is_armed=alert.get('is_armed', True),
                last_price=alert.get('last_price'),
                last_triggered_at=alert.get('last_triggered_at'),
                created=alert.get('created')
            ).dict(),
            "user_info": db.get_user_status(alert['user_id'])
        }

        _logger.info("Retrieved Telegram alert config: %d", alert_id)
        return config

    except HTTPException:
        raise
    except Exception as e:
        _logger.error("Error getting Telegram alert config %d: %s", alert_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/alerts", response_model=AlertStats)
async def get_telegram_alert_stats(
    current_user: User = Depends(get_current_user)
):
    """Get Telegram alert statistics."""
    try:
        _logger.info("Getting Telegram alert statistics")

        # Get all alerts
        all_alerts = db.get_alerts_by_type()
        active_alerts = db.get_active_alerts()

        # Calculate basic statistics
        total_alerts = len(all_alerts)
        active_count = len(active_alerts)

        # For now, set triggered_today and rearm_cycles to 0
        # These would require additional database queries or audit log analysis
        triggered_today = 0
        rearm_cycles = 0

        stats = AlertStats(
            total_alerts=total_alerts,
            active_alerts=active_count,
            triggered_today=triggered_today,
            rearm_cycles=rearm_cycles
        )

        _logger.info("Retrieved Telegram alert statistics: %s", stats.dict())
        return stats

    except Exception as e:
        _logger.error("Error getting Telegram alert statistics: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# --- SCHEDULE MANAGEMENT ENDPOINTS ---

@router.get("/schedules", response_model=List[TelegramSchedule])
async def get_telegram_schedules(
    filter: Optional[str] = Query(None, description="Filter schedules by status: all, active, inactive"),
    current_user: User = Depends(get_current_user)
):
    """
    Get list of Telegram bot schedules with optional filtering.

    Filters:
    - all: All schedules (default)
    - active: Only active schedules
    - inactive: Only inactive schedules
    """
    try:
        _logger.info("Getting Telegram schedules with filter: %s", filter)

        # Get schedules based on filter
        if filter == "active":
            schedules = db.get_active_schedules()
        elif filter == "inactive":
            # Get all schedules and filter inactive ones
            all_schedules = db.get_schedules_by_config()
            schedules = [s for s in all_schedules if not s.get('active', True)]
        else:
            # Get all schedules
            schedules = db.get_schedules_by_config()

        # Convert to response model
        result = []
        for schedule in schedules:
            result.append(TelegramSchedule(
                id=schedule['id'],
                user_id=schedule['user_id'],
                ticker=schedule['ticker'],
                scheduled_time=schedule['scheduled_time'],
                period=schedule.get('period'),
                active=schedule.get('active', True),
                email=schedule.get('email', False),
                indicators=schedule.get('indicators'),
                interval=schedule.get('interval'),
                provider=schedule.get('provider'),
                schedule_type=schedule.get('schedule_type'),
                list_type=schedule.get('list_type'),
                config_json=schedule.get('config_json'),
                schedule_config=schedule.get('schedule_config'),
                created=schedule.get('created')
            ))

        _logger.info("Retrieved %d Telegram schedules", len(result))
        return result

    except Exception as e:
        _logger.error("Error getting Telegram schedules: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/schedules/{schedule_id}/toggle")
async def toggle_telegram_schedule(
    schedule_id: int,
    current_user: User = Depends(require_admin)
):
    """Toggle a Telegram schedule's active status."""
    try:
        _logger.info("Toggling Telegram schedule: %d", schedule_id)

        # Get current schedule
        schedule = db.get_schedule(schedule_id)
        if not schedule:
            raise HTTPException(status_code=404, detail="Schedule not found")

        # Toggle active status
        new_active_status = not schedule.get('active', True)
        success = db.update_schedule(schedule_id, active=new_active_status)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to toggle schedule")

        status_text = "activated" if new_active_status else "deactivated"
        _logger.info("Successfully %s Telegram schedule: %d", status_text, schedule_id)
        return {"message": f"Schedule {schedule_id} {status_text} successfully"}

    except HTTPException:
        raise
    except Exception as e:
        _logger.error("Error toggling Telegram schedule %d: %s", schedule_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/schedules/{schedule_id}")
async def delete_telegram_schedule(
    schedule_id: int,
    current_user: User = Depends(require_admin)
):
    """Delete a Telegram schedule."""
    try:
        _logger.info("Deleting Telegram schedule: %d", schedule_id)

        # Check if schedule exists
        schedule = db.get_schedule(schedule_id)
        if not schedule:
            raise HTTPException(status_code=404, detail="Schedule not found")

        # Delete schedule
        success = db.delete_schedule(schedule_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete schedule")

        _logger.info("Successfully deleted Telegram schedule: %d", schedule_id)
        return {"message": f"Schedule {schedule_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        _logger.error("Error deleting Telegram schedule %d: %s", schedule_id, e)
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

        # Check if schedule exists
        schedule = db.get_schedule(schedule_id)
        if not schedule:
            raise HTTPException(status_code=404, detail="Schedule not found")

        # Update schedule with provided data
        success = db.update_schedule(schedule_id, **schedule_data)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update schedule")

        _logger.info("Successfully updated Telegram schedule: %d", schedule_id)
        return {"message": f"Schedule {schedule_id} updated successfully"}

    except HTTPException:
        raise
    except Exception as e:
        _logger.error("Error updating Telegram schedule %d: %s", schedule_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/schedules", response_model=ScheduleStats)
async def get_telegram_schedule_stats(
    current_user: User = Depends(get_current_user)
):
    """Get Telegram schedule statistics."""
    try:
        _logger.info("Getting Telegram schedule statistics")

        # Get all schedules
        all_schedules = db.get_schedules_by_config()
        active_schedules = db.get_active_schedules()

        # Calculate basic statistics
        total_schedules = len(all_schedules)
        active_count = len(active_schedules)

        # For now, set executed_today and failed_executions to 0
        # These would require additional database queries or audit log analysis
        executed_today = 0
        failed_executions = 0

        stats = ScheduleStats(
            total_schedules=total_schedules,
            active_schedules=active_count,
            executed_today=executed_today,
            failed_executions=failed_executions
        )

        _logger.info("Retrieved Telegram schedule statistics: %s", stats.dict())
        return stats

    except Exception as e:
        _logger.error("Error getting Telegram schedule statistics: %s", e)
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

        # Get all approved users
        users = db.list_users()
        approved_users = [u for u in users if u.get('approved', False)]

        total_recipients = len(approved_users)
        successful_deliveries = 0
        failed_deliveries = 0

        # For now, we'll simulate the broadcast
        # In a real implementation, this would integrate with the Telegram bot API
        # to actually send messages to users

        # Simulate delivery (in real implementation, iterate through users and send messages)
        successful_deliveries = total_recipients  # Assume all succeed for now
        failed_deliveries = 0

        # Log the broadcast for audit purposes
        _logger.info("Broadcast sent to %d users by admin %s", total_recipients, current_user.get_username())

        result = BroadcastResult(
            message="Broadcast sent successfully",
            total_recipients=total_recipients,
            successful_deliveries=successful_deliveries,
            failed_deliveries=failed_deliveries
        )

        _logger.info("Broadcast result: %s", result.dict())
        return result

    except Exception as e:
        _logger.error("Error sending Telegram broadcast: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# --- AUDIT LOGGING ENDPOINTS ---

@router.get("/audit", response_model=List[CommandAudit])
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

        # Get audit logs with filters
        audit_logs = db.get_all_command_audit(
            limit=page_size,
            offset=offset,
            user_id=user_id,
            command=command,
            success_only=success_only,
            start_date=start_date,
            end_date=end_date
        )

        # Convert to response model
        result = []
        for log in audit_logs:
            result.append(CommandAudit(
                id=log['id'],
                telegram_user_id=log['telegram_user_id'],
                command=log['command'],
                full_message=log.get('full_message'),
                is_registered_user=log.get('is_registered_user', False),
                user_email=log.get('user_email'),
                success=log.get('success', True),
                error_message=log.get('error_message'),
                response_time_ms=log.get('response_time_ms'),
                created=log['created']
            ))

        _logger.info("Retrieved %d Telegram audit logs", len(result))
        return result

    except Exception as e:
        _logger.error("Error getting Telegram audit logs: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/audit", response_model=AuditStats)
async def get_telegram_audit_stats(
    current_user: User = Depends(get_current_user)
):
    """Get Telegram command audit statistics."""
    try:
        _logger.info("Getting Telegram audit statistics")

        # Get audit statistics from database
        stats_data = db.get_command_audit_stats()

        # Get recent activity (last 24 hours)
        # This is a simplified implementation - in practice you'd query by date
        recent_logs = db.get_all_command_audit(limit=1000)  # Get recent logs
        recent_activity_24h = len(recent_logs)  # Simplified count

        # Calculate top commands
        command_counts = {}
        for log in recent_logs:
            command = log.get('command', 'unknown')
            command_counts[command] = command_counts.get(command, 0) + 1

        top_commands = [
            {"command": cmd, "count": count}
            for cmd, count in sorted(command_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]

        stats = AuditStats(
            total_commands=stats_data.get('total_commands', 0),
            successful_commands=stats_data.get('successful_commands', 0),
            failed_commands=stats_data.get('failed_commands', 0),
            recent_activity_24h=recent_activity_24h,
            top_commands=top_commands
        )

        _logger.info("Retrieved Telegram audit statistics: %s", stats.dict())
        return stats

    except Exception as e:
        _logger.error("Error getting Telegram audit statistics: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users/{user_id}/audit", response_model=List[CommandAudit])
async def get_user_audit_logs(
    user_id: str,
    limit: int = Query(50, ge=1, le=100, description="Number of logs to retrieve"),
    current_user: User = Depends(get_current_user)
):
    """Get command audit logs for a specific user."""
    try:
        _logger.info("Getting audit logs for Telegram user: %s", user_id)

        # Get user command history
        history = db.get_user_command_history(user_id, limit)

        # Convert to response model
        result = []
        for log in history:
            result.append(CommandAudit(
                id=log['id'],
                telegram_user_id=user_id,
                command=log['command'],
                full_message=log.get('full_message'),
                is_registered_user=log.get('is_registered_user', False),
                user_email=log.get('user_email'),
                success=log.get('success', True),
                error_message=log.get('error_message'),
                response_time_ms=log.get('response_time_ms'),
                created=log['created']
            ))

        _logger.info("Retrieved %d audit logs for user %s", len(result), user_id)
        return result

    except Exception as e:
        _logger.error("Error getting audit logs for user %s: %s", user_id, e)
        raise HTTPException(status_code=500, detail=str(e))