"""
Notification API Routes
----------------------

FastAPI routes for notification message management.
Provides endpoints for creating, tracking, and managing notifications
through the notification service.
"""

from fastapi import APIRouter, HTTPException, status, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from pathlib import Path
import sys
import httpx

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.web_ui.backend.auth import get_current_user, require_trader_or_admin
from src.data.db.models.model_users import User
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

# Create router
router = APIRouter(prefix="/api/notifications", tags=["notifications"])

# Direct database access instead of proxying to notification service
from src.data.db.repos.repo_notification import NotificationRepository
from src.data.db.core.database import session_scope
from src.data.db.models.model_notification import (
    Message, MessageDeliveryStatus, ChannelHealth,
    MessageStatus, DeliveryStatus, ChannelHealthStatus
)
from sqlalchemy.orm import Session
from sqlalchemy import func, desc


# Pydantic models for API
class NotificationCreate(BaseModel):
    """Notification creation request model."""
    message_type: str = Field(..., description="Type of notification (alert, trade, system, etc.)")
    priority: str = Field(default="normal", description="Message priority (low, normal, high, urgent)")
    channels: List[str] = Field(..., description="Delivery channels (telegram, email, sms)")
    recipient_id: str = Field(..., description="Recipient user ID")
    template_name: Optional[str] = Field(None, description="Template name for structured messages")
    content: Dict[str, Any] = Field(..., description="Message content and variables")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    scheduled_for: Optional[datetime] = Field(None, description="Schedule delivery for specific time")


class NotificationResponse(BaseModel):
    """Notification response model."""
    message_id: int
    status: str
    channels: List[str]
    priority: str
    created_at: str
    scheduled_for: str
    processed_at: Optional[str]
    retry_count: int
    last_error: Optional[str]


class DeliveryStatusResponse(BaseModel):
    """Delivery status response model."""
    delivery_id: int
    channel: str
    status: str
    delivered_at: Optional[str]
    response_time_ms: Optional[int]
    error_message: Optional[str]
    external_id: Optional[str]


class NotificationStats(BaseModel):
    """Notification statistics model."""
    total_messages: int
    delivered_messages: int
    failed_messages: int
    pending_messages: int
    success_rate: float
    channels_health: Dict[str, str]


# Helper function to get database session
def get_notification_repo() -> NotificationRepository:
    """Get notification repository instance."""
    with session_scope() as session:
        return NotificationRepository(session)


# Helper function to call notification service
async def _call_notification_service(method: str, endpoint: str, data: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Call the notification service API.

    Args:
        method: HTTP method (GET, POST, etc.)
        endpoint: API endpoint
        data: Request body data
        params: Query parameters

    Returns:
        Response data

    Raises:
        HTTPException: If the service call fails
    """
    try:
        # Get notification service URL from config or use default
        service_url = "http://localhost:8001"  # Default notification service URL

        url = f"{service_url}/api/v1{endpoint}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            if method.upper() == "GET":
                response = await client.get(url, params=params)
            elif method.upper() == "POST":
                response = await client.post(url, json=data, params=params)
            elif method.upper() == "PUT":
                response = await client.put(url, json=data, params=params)
            elif method.upper() == "DELETE":
                response = await client.delete(url, params=params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            if response.status_code == 404:
                raise HTTPException(status_code=404, detail="Resource not found")
            elif response.status_code >= 400:
                error_detail = "Notification service error"
                try:
                    error_data = response.json()
                    error_detail = error_data.get("detail", error_detail)
                except:
                    pass
                raise HTTPException(status_code=response.status_code, detail=error_detail)

            return response.json()

    except httpx.TimeoutException:
        _logger.error("Timeout calling notification service: %s %s", method, endpoint)
        raise HTTPException(status_code=504, detail="Notification service timeout")
    except httpx.ConnectError:
        _logger.error("Connection error calling notification service: %s %s", method, endpoint)
        raise HTTPException(status_code=503, detail="Notification service unavailable")
    except HTTPException:
        raise
    except Exception as e:
        _logger.error("Error calling notification service: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")


# Health check endpoint (must be before parameterized routes)

@router.get("/health")
async def notification_routes_health():
    """
    Health check for notification routes and service connectivity.

    Returns:
        Health status of notification routes and service
    """
    try:
        # Test connection to notification service
        result = await _call_notification_service("GET", "/health")

        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "notification_service": result,
            "routes": "operational"
        }

    except Exception as e:
        _logger.error("Notification routes health check failed: %s", e)
        return {
            "status": "unhealthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
            "routes": "operational",
            "notification_service": "unavailable"
        }


# Channel endpoints

@router.get("/channels/health")
async def get_channels_health(current_user: User = Depends(get_current_user)):
    """
    Get health status for all notification channels.

    Args:
        current_user: Current authenticated user

    Returns:
        Channel health information
    """
    try:
        result = await _call_notification_service("GET", "/channels/health")
        return {"channels_health": result}

    except HTTPException:
        raise
    except Exception as e:
        _logger.error("Error getting channels health: %s", e)
        raise HTTPException(status_code=500, detail="Failed to get channels health")


@router.get("/channels")
async def list_notification_channels(current_user: User = Depends(get_current_user)):
    """
    List available notification channels.

    Args:
        current_user: Current authenticated user

    Returns:
        List of available channels with configuration
    """
    try:
        result = await _call_notification_service("GET", "/channels")
        return {"channels": result}

    except HTTPException:
        raise
    except Exception as e:
        _logger.error("Error listing channels: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list channels")


# Statistics endpoints

@router.get("/stats", response_model=NotificationStats)
async def get_notification_statistics(
    channel: Optional[str] = None,
    days: int = 30,
    current_user: User = Depends(get_current_user)
):
    """
    Get notification delivery statistics.

    Args:
        channel: Filter by channel (optional)
        days: Number of days to analyze
        current_user: Current authenticated user

    Returns:
        Notification statistics
    """
    try:
        params = {"days": days}
        if channel:
            params["channel"] = channel

        result = await _call_notification_service("GET", "/stats", params=params)

        # Extract relevant statistics
        delivery_stats = result.get("delivery_statistics", {})
        message_stats = result.get("message_statistics", {})

        total_messages = sum(message_stats.values()) if message_stats else 0
        delivered_messages = delivery_stats.get("delivered", 0)
        failed_messages = delivery_stats.get("failed", 0)
        pending_messages = message_stats.get("pending", 0)

        success_rate = (delivered_messages / total_messages) if total_messages > 0 else 0.0

        # Get channel health
        try:
            health_result = await _call_notification_service("GET", "/channels/health")
            channels_health = {
                ch["channel"]: ch["status"]
                for ch in health_result
            }
        except:
            channels_health = {}

        return NotificationStats(
            total_messages=total_messages,
            delivered_messages=delivered_messages,
            failed_messages=failed_messages,
            pending_messages=pending_messages,
            success_rate=success_rate,
            channels_health=channels_health
        )

    except HTTPException:
        raise
    except Exception as e:
        _logger.error("Error getting notification statistics: %s", e)
        raise HTTPException(status_code=500, detail="Failed to get notification statistics")


# Administrative endpoints

@router.post("/admin/cleanup")
async def cleanup_old_notifications(
    days_to_keep: int = 30,
    current_user: User = Depends(require_trader_or_admin)
):
    """
    Clean up old delivered notifications.

    Args:
        days_to_keep: Number of days of notifications to keep
        current_user: Current authenticated user (admin required)

    Returns:
        Cleanup status
    """
    try:
        if days_to_keep < 1:
            raise HTTPException(status_code=400, detail="Days to keep must be at least 1")

        result = await _call_notification_service(
            "POST",
            "/admin/cleanup",
            params={"days_to_keep": days_to_keep}
        )

        _logger.info(
            "Notification cleanup initiated by user %s: days_to_keep=%s",
            current_user.username or current_user.email,
            days_to_keep
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        _logger.error("Error initiating cleanup: %s", e)
        raise HTTPException(status_code=500, detail="Failed to initiate cleanup")


@router.get("/admin/processor/stats")
async def get_processor_statistics(current_user: User = Depends(require_trader_or_admin)):
    """
    Get message processor statistics.

    Args:
        current_user: Current authenticated user (admin required)

    Returns:
        Processor statistics and performance metrics
    """
    try:
        result = await _call_notification_service("GET", "/processor/stats")
        return result

    except HTTPException:
        raise
    except Exception as e:
        _logger.error("Error getting processor statistics: %s", e)
        raise HTTPException(status_code=500, detail="Failed to get processor statistics")


# Convenience endpoint models
class AlertNotificationRequest(BaseModel):
    """Alert notification request model."""
    title: str
    message: str
    severity: str = "normal"
    channels: List[str] = ["telegram"]
    recipient_id: Optional[str] = None


class TradeNotificationRequest(BaseModel):
    """Trade notification request model."""
    action: str
    symbol: str
    quantity: float
    price: float
    strategy_name: str
    channels: List[str] = ["telegram"]
    recipient_id: Optional[str] = None


# Convenience endpoints for common notification types

@router.post("/alert")
async def send_alert_notification(
    alert_data: AlertNotificationRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Send an alert notification (convenience endpoint).

    Args:
        alert_data: Alert notification data
        current_user: Current authenticated user

    Returns:
        Notification creation result
    """
    try:
        notification = NotificationCreate(
            message_type="alert",
            priority=alert_data.severity,
            channels=alert_data.channels,
            recipient_id=alert_data.recipient_id or str(current_user.id),
            content={
                "title": alert_data.title,
                "message": alert_data.message,
                "severity": alert_data.severity
            },
            metadata={
                "alert_type": "manual",
                "source": "web_ui"
            }
        )

        return await create_notification(notification, current_user)

    except Exception as e:
        _logger.error("Error sending alert notification: %s", e)
        raise HTTPException(status_code=500, detail="Failed to send alert notification")


@router.post("/trade")
async def send_trade_notification(
    trade_data: TradeNotificationRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Send a trade notification (convenience endpoint).

    Args:
        trade_data: Trade notification data
        current_user: Current authenticated user

    Returns:
        Notification creation result
    """
    try:
        notification = NotificationCreate(
            message_type="trade",
            priority="normal",
            channels=trade_data.channels,
            recipient_id=trade_data.recipient_id or str(current_user.id),
            template_name="trade_execution",
            content={
                "action": trade_data.action,
                "symbol": trade_data.symbol,
                "quantity": trade_data.quantity,
                "price": trade_data.price,
                "strategy_name": trade_data.strategy_name,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            metadata={
                "trade_type": trade_data.action,
                "source": "web_ui"
            }
        )

        return await create_notification(notification, current_user)

    except Exception as e:
        _logger.error("Error sending trade notification: %s", e)
        raise HTTPException(status_code=500, detail="Failed to send trade notification")


# Notification management endpoints (parameterized routes must come last)

@router.post("/", response_model=Dict[str, Any])
async def create_notification(
    notification: NotificationCreate,
    current_user: User = Depends(get_current_user)
):
    """
    Create and enqueue a new notification message.

    Args:
        notification: Notification data
        current_user: Current authenticated user

    Returns:
        Message creation result with ID and status
    """
    try:
        # Prepare message data for notification service
        message_data = {
            "message_type": notification.message_type,
            "priority": notification.priority,
            "channels": notification.channels,
            "recipient_id": notification.recipient_id,
            "template_name": notification.template_name,
            "content": notification.content,
            "metadata": {
                **notification.metadata,
                "created_by_user_id": current_user.id,
                "created_by_username": current_user.username or current_user.email
            }
        }

        if notification.scheduled_for:
            message_data["scheduled_for"] = notification.scheduled_for.isoformat()

        # Call notification service
        result = await _call_notification_service("POST", "/messages", data=message_data)

        _logger.info(
            "Notification created by user %s: message_id=%s, type=%s, channels=%s",
            current_user.username or current_user.email,
            result.get("message_id"),
            notification.message_type,
            notification.channels
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        _logger.error("Error creating notification: %s", e)
        raise HTTPException(status_code=500, detail="Failed to create notification")


@router.get("/", response_model=List[NotificationResponse])
async def list_notifications(
    status: Optional[str] = None,
    priority: Optional[str] = None,
    recipient_id: Optional[str] = None,
    message_type: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    current_user: User = Depends(get_current_user)
):
    """
    List notifications with optional filtering.

    Args:
        status: Filter by message status
        priority: Filter by message priority
        recipient_id: Filter by recipient ID
        message_type: Filter by message type
        limit: Maximum number of results
        offset: Number of results to skip
        current_user: Current authenticated user

    Returns:
        List of notifications
    """
    try:
        params = {
            "limit": min(limit, 1000),  # Cap at 1000
            "offset": offset
        }

        if status:
            params["status"] = status
        if priority:
            params["priority"] = priority
        if recipient_id:
            params["recipient_id"] = recipient_id
        if message_type:
            params["message_type"] = message_type

        result = await _call_notification_service("GET", "/messages", params=params)
        return [NotificationResponse(**msg) for msg in result]

    except HTTPException:
        raise
    except Exception as e:
        _logger.error("Error listing notifications: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list notifications")


@router.get("/{message_id}", response_model=NotificationResponse)
async def get_notification_status(
    message_id: int,
    current_user: User = Depends(get_current_user)
):
    """
    Get notification status and details.

    Args:
        message_id: Message ID
        current_user: Current authenticated user

    Returns:
        Notification details and status
    """
    try:
        result = await _call_notification_service("GET", f"/messages/{message_id}/status")
        return NotificationResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        _logger.error("Error getting notification status: %s", e)
        raise HTTPException(status_code=500, detail="Failed to get notification status")


@router.get("/{message_id}/delivery", response_model=List[DeliveryStatusResponse])
async def get_notification_delivery_status(
    message_id: int,
    current_user: User = Depends(get_current_user)
):
    """
    Get delivery status for all channels of a notification.

    Args:
        message_id: Message ID
        current_user: Current authenticated user

    Returns:
        List of delivery statuses per channel
    """
    try:
        result = await _call_notification_service("GET", f"/messages/{message_id}/delivery")
        return [DeliveryStatusResponse(**ds) for ds in result]

    except HTTPException:
        raise
    except Exception as e:
        _logger.error("Error getting delivery status: %s", e)
        raise HTTPException(status_code=500, detail="Failed to get delivery status")
