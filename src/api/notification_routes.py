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

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.api.auth import get_current_user, require_trader_or_admin
from src.data.db.models.model_users import User
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

# Create router
router = APIRouter(prefix="/api/notifications", tags=["notifications"])

# Direct database access for database-centric architecture
from src.data.db.services.database_service import get_database_service
from src.data.db.models.model_notification import (
    Message, MessageDeliveryStatus,
    MessageStatus, DeliveryStatus, MessagePriority
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


# Database-centric helper functions (no HTTP calls)
def _validate_channels(channels: List[str]) -> List[str]:
    """
    Validate and filter enabled channels.

    Args:
        channels: List of requested channels

    Returns:
        List of enabled channels
    """
    # For now, assume all channels are valid
    # In a real implementation, this would check channel configuration
    enabled_channels = []
    valid_channels = ["telegram", "email", "sms", "slack", "discord"]

    for channel in channels:
        if channel in valid_channels:
            enabled_channels.append(channel)
        else:
            _logger.warning("Unknown channel requested: %s", channel)

    return enabled_channels

def _convert_priority(priority_str: str) -> MessagePriority:
    """
    Convert priority string to MessagePriority enum.

    Args:
        priority_str: Priority as string

    Returns:
        MessagePriority enum value
    """
    priority_map = {
        "low": MessagePriority.LOW,
        "normal": MessagePriority.NORMAL,
        "high": MessagePriority.HIGH,
        "critical": MessagePriority.CRITICAL,
        "urgent": MessagePriority.CRITICAL  # Map urgent to critical
    }

    return priority_map.get(priority_str.lower(), MessagePriority.NORMAL)


# Health check endpoint (must be before parameterized routes)

@router.get("/health")
async def notification_routes_health():
    """
    Health check for notification routes and database connectivity.

    Returns:
        Health status of notification routes and database
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
            "routes": "operational",
            "total_messages": message_count
        }

    except Exception as e:
        _logger.exception("Notification routes health check failed:")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
            "routes": "operational",
            "database": "unavailable"
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
        db_service = get_database_service()

        with db_service.uow() as uow:
            from src.data.db.repos.repo_system_health import SystemHealthRepository
            from src.data.db.services.system_health_service import SystemHealthService

            # Initialize health service
            health_repo = SystemHealthRepository(uow.s)
            health_service = SystemHealthService(health_repo)

            # Get notification channels health
            channels_health = health_service.get_notification_channels_health()

            return {
                "channels_health": channels_health,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    except Exception as e:
        _logger.exception("Error getting channels health:")
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
        db_service = get_database_service()

        with db_service.uow() as uow:
            # Get channel configurations from database
            channel_configs = uow.notifications.channel_configs.list_channel_configs()

            # Convert to response format
            channels = []
            for channel_config in channel_configs:
                channels.append({
                    "channel": channel_config.channel,
                    "enabled": channel_config.enabled,
                    "rate_limit_per_minute": channel_config.rate_limit_per_minute,
                    "max_retries": channel_config.max_retries,
                    "timeout_seconds": channel_config.timeout_seconds
                })

            # Add default channels if not in database
            default_channels = ["telegram", "email", "sms"]
            existing_channels = {ch["channel"] for ch in channels}

            for channel_name in default_channels:
                if channel_name not in existing_channels:
                    channels.append({
                        "channel": channel_name,
                        "enabled": True,  # Default to enabled
                        "rate_limit_per_minute": 60,
                        "max_retries": 3,
                        "timeout_seconds": 30
                    })

            return {"channels": channels}

    except Exception as e:
        _logger.exception("Error listing channels:")
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
        db_service = get_database_service()

        with db_service.uow() as uow:
            # Get delivery statistics
            delivery_stats = uow.notifications.delivery_status.get_delivery_statistics(
                channel=channel, days=days
            )

            # Get message statistics
            message_stats = {}
            for status in MessageStatus:
                count = uow.s.query(Message).filter(Message.status == status.value).count()
                message_stats[status.value] = count

            total_messages = sum(message_stats.values()) if message_stats else 0
            delivered_messages = delivery_stats.get("delivered", 0)
            failed_messages = delivery_stats.get("failed", 0)
            pending_messages = message_stats.get("PENDING", 0)

            success_rate = (delivered_messages / total_messages) if total_messages > 0 else 0.0

            # Get channel health
            try:
                from src.data.db.repos.repo_system_health import SystemHealthRepository
                from src.data.db.services.system_health_service import SystemHealthService

                health_repo = SystemHealthRepository(uow.s)
                health_service = SystemHealthService(health_repo)
                channels_health_data = health_service.get_notification_channels_health()

                channels_health = {
                    ch["channel"]: ch["status"]
                    for ch in channels_health_data
                }
            except Exception as e:
                _logger.warning("Could not get channel health: %s", e)
                channels_health = {}

            return NotificationStats(
                total_messages=total_messages,
                delivered_messages=delivered_messages,
                failed_messages=failed_messages,
                pending_messages=pending_messages,
                success_rate=success_rate,
                channels_health=channels_health
            )

    except Exception as e:
        _logger.exception("Error getting notification statistics:")
        raise HTTPException(status_code=500, detail="Failed to get notification statistics")


# Analytics endpoints (consolidated from notification service)

@router.get("/analytics/delivery-rates")
async def get_delivery_rates_analytics(
    channel: Optional[str] = None,
    user_id: Optional[str] = None,
    days: int = 30,
    current_user: User = Depends(get_current_user)
):
    """
    Get comprehensive delivery rate analytics.

    Args:
        channel: Filter by specific channel
        user_id: Filter by specific user
        days: Number of days to analyze
        current_user: Current authenticated user

    Returns:
        Delivery rate analytics with channel and user breakdowns
    """
    try:
        # Import analytics service
        from src.notification.service.analytics import notification_analytics

        # Get delivery rate analytics
        analytics_result = await notification_analytics.get_delivery_rates(
            channel=channel,
            user_id=user_id,
            days=days
        )

        return {
            "analytics": analytics_result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        _logger.exception("Error getting delivery rates analytics:")
        raise HTTPException(status_code=500, detail="Failed to get delivery rates analytics")


@router.get("/analytics/response-times")
async def get_response_time_analytics(
    channel: Optional[str] = None,
    days: int = 30,
    current_user: User = Depends(get_current_user)
):
    """
    Get detailed response time analytics.

    Args:
        channel: Filter by specific channel
        days: Number of days to analyze
        current_user: Current authenticated user

    Returns:
        Response time statistics and percentiles
    """
    try:
        # Import analytics service
        from src.notification.service.analytics import notification_analytics

        # Get response time analytics
        analytics_result = await notification_analytics.get_response_time_analysis(
            channel=channel,
            days=days
        )

        return {
            "analytics": analytics_result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        _logger.exception("Error getting response time analytics:")
        raise HTTPException(status_code=500, detail="Failed to get response time analytics")


@router.get("/analytics/trends")
async def get_trend_analytics(
    metric: str = "success_rate",
    channel: Optional[str] = None,
    days: int = 30,
    current_user: User = Depends(get_current_user)
):
    """
    Get trend analysis for notification metrics.

    Args:
        metric: Metric to analyze (success_rate, response_time, message_count)
        channel: Filter by specific channel
        days: Number of days to analyze
        current_user: Current authenticated user

    Returns:
        Trend analysis with direction, strength, and time series data
    """
    try:
        # Validate metric parameter
        valid_metrics = ["success_rate", "response_time", "message_count"]
        if metric not in valid_metrics:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid metric. Must be one of: {', '.join(valid_metrics)}"
            )

        # Import analytics service
        from src.notification.service.analytics import notification_analytics

        # Get trend analysis
        trend_analysis = await notification_analytics.get_trend_analysis(
            metric=metric,
            days=days,
            channel=channel
        )

        return {
            "analytics": trend_analysis.to_dict(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("Error getting trend analytics:")
        raise HTTPException(status_code=500, detail="Failed to get trend analytics")


@router.get("/analytics/aggregated")
async def get_aggregated_analytics(
    granularity: str = "daily",
    channel: Optional[str] = None,
    days: int = 30,
    current_user: User = Depends(get_current_user)
):
    """
    Get time-aggregated notification statistics.

    Args:
        granularity: Time granularity (hourly, daily, weekly, monthly)
        channel: Filter by specific channel
        days: Number of days to analyze
        current_user: Current authenticated user

    Returns:
        Time-aggregated statistics with summary metrics
    """
    try:
        # Validate granularity parameter
        valid_granularities = ["hourly", "daily", "weekly", "monthly"]
        if granularity not in valid_granularities:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid granularity. Must be one of: {', '.join(valid_granularities)}"
            )

        # Import analytics service
        from src.notification.service.analytics import notification_analytics, TimeGranularity

        # Convert string to enum
        granularity_enum = TimeGranularity(granularity)

        # Get aggregated analytics
        analytics_result = await notification_analytics.get_aggregated_statistics(
            granularity=granularity_enum,
            days=days,
            channel=channel
        )

        return {
            "analytics": analytics_result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("Error getting aggregated analytics:")
        raise HTTPException(status_code=500, detail="Failed to get aggregated analytics")


@router.get("/analytics/channel-comparison")
async def get_channel_performance_comparison(
    days: int = 30,
    current_user: User = Depends(get_current_user)
):
    """
    Get performance comparison across all notification channels.

    Args:
        days: Number of days to analyze
        current_user: Current authenticated user

    Returns:
        Channel performance comparison with rankings and scores
    """
    try:
        # Import analytics service
        from src.notification.service.analytics import notification_analytics

        # Get channel performance comparison
        analytics_result = await notification_analytics.get_channel_performance_comparison(
            days=days
        )

        return {
            "analytics": analytics_result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        _logger.exception("Error getting channel performance comparison:")
        raise HTTPException(status_code=500, detail="Failed to get channel performance comparison")


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

        db_service = get_database_service()

        with db_service.uow() as uow:
            # Clean up old delivered messages
            deleted_count = uow.notifications.messages.cleanup_old_messages(days_to_keep)
            uow.commit()

        _logger.info(
            "Notification cleanup completed by user %s: %s messages deleted, days_to_keep=%s",
            current_user.username or current_user.email,
            deleted_count,
            days_to_keep
        )

        return {
            "status": "completed",
            "deleted_count": deleted_count,
            "days_to_keep": days_to_keep,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("Error during cleanup:")
        raise HTTPException(status_code=500, detail="Failed to cleanup notifications")


@router.get("/admin/processor/stats")
async def get_processor_statistics(current_user: User = Depends(require_trader_or_admin)):
    """
    Get message processor statistics from database.

    Args:
        current_user: Current authenticated user (admin required)

    Returns:
        Processor statistics and performance metrics
    """
    try:
        db_service = get_database_service()

        with db_service.uow() as uow:
            # Get basic message statistics
            total_messages = uow.s.query(Message).count()
            pending_messages = uow.s.query(Message).filter(Message.status == MessageStatus.PENDING.value).count()
            processing_messages = uow.s.query(Message).filter(Message.status == MessageStatus.PROCESSING.value).count()
            delivered_messages = uow.s.query(Message).filter(Message.status == MessageStatus.DELIVERED.value).count()
            failed_messages = uow.s.query(Message).filter(Message.status == MessageStatus.FAILED.value).count()

            # Get locked messages (currently being processed)
            locked_messages = uow.s.query(Message).filter(Message.locked_by.isnot(None)).count()

            # Calculate success rate
            processed_messages = delivered_messages + failed_messages
            success_rate = (delivered_messages / processed_messages) if processed_messages > 0 else 0.0

            return {
                "processor_stats": {
                    "total_messages": total_messages,
                    "pending_messages": pending_messages,
                    "processing_messages": processing_messages,
                    "delivered_messages": delivered_messages,
                    "failed_messages": failed_messages,
                    "locked_messages": locked_messages,
                    "success_rate": success_rate
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "note": "Statistics retrieved from database (database-centric architecture)"
            }

    except Exception as e:
        _logger.exception("Error getting processor statistics:")
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
        _logger.exception("Error sending alert notification:")
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
        _logger.exception("Error sending trade notification:")
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
        # Validate and filter channels
        enabled_channels = _validate_channels(notification.channels)
        if not enabled_channels:
            raise HTTPException(status_code=400, detail="No valid channels specified")

        # Convert priority
        priority = _convert_priority(notification.priority)

        # Prepare message data for database
        message_data = {
            "message_type": notification.message_type,
            "priority": priority.value,
            "channels": enabled_channels,
            "recipient_id": notification.recipient_id,
            "template_name": notification.template_name,
            "content": notification.content,
            "message_metadata": {
                **notification.metadata,
                "created_by_user_id": current_user.id,
                "created_by_username": current_user.username or current_user.email
            },
            "scheduled_for": notification.scheduled_for or datetime.now(timezone.utc),
            "status": MessageStatus.PENDING.value
        }

        # Create message in database
        db_service = get_database_service()

        with db_service.uow() as uow:
            db_message = uow.notifications.messages.create_message(message_data)
            uow.commit()

        _logger.info(
            "Notification created by user %s: message_id=%s, type=%s, channels=%s",
            current_user.username or current_user.email,
            db_message.id,
            notification.message_type,
            enabled_channels
        )

        return {
            "message_id": db_message.id,
            "status": "enqueued",
            "channels": enabled_channels,
            "priority": db_message.priority,
            "scheduled_for": db_message.scheduled_for.isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("Error creating notification:")
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
        db_service = get_database_service()

        with db_service.uow() as uow:
            # Convert string parameters to enums if provided
            status_enum = None
            if status:
                try:
                    status_enum = MessageStatus(status.upper())
                except ValueError:
                    raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

            priority_enum = None
            if priority:
                try:
                    priority_enum = _convert_priority(priority)
                except ValueError:
                    raise HTTPException(status_code=400, detail=f"Invalid priority: {priority}")

            # Get messages from database
            messages = uow.notifications.messages.list_messages(
                status=status_enum,
                priority=priority_enum,
                recipient_id=recipient_id,
                message_type=message_type,
                limit=min(limit, 1000),  # Cap at 1000
                offset=offset
            )

            # Convert to response format
            notifications = []
            for msg in messages:
                notifications.append(NotificationResponse(
                    message_id=msg.id,
                    status=msg.status,
                    channels=msg.channels,
                    priority=msg.priority,
                    created_at=msg.created_at.isoformat(),
                    scheduled_for=msg.scheduled_for.isoformat(),
                    processed_at=msg.processed_at.isoformat() if msg.processed_at else None,
                    retry_count=msg.retry_count,
                    last_error=msg.last_error
                ))

            return notifications

    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("Error listing notifications:")
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
        db_service = get_database_service()

        with db_service.uow() as uow:
            message = uow.notifications.messages.get_message(message_id)

            if not message:
                raise HTTPException(status_code=404, detail="Message not found")

            return NotificationResponse(
                message_id=message.id,
                status=message.status,
                channels=message.channels,
                priority=message.priority,
                created_at=message.created_at.isoformat(),
                scheduled_for=message.scheduled_for.isoformat(),
                processed_at=message.processed_at.isoformat() if message.processed_at else None,
                retry_count=message.retry_count,
                last_error=message.last_error
            )

    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("Error getting notification status:")
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
        db_service = get_database_service()

        with db_service.uow() as uow:
            # Check if message exists
            message = uow.notifications.messages.get_message(message_id)
            if not message:
                raise HTTPException(status_code=404, detail="Message not found")

            # Get delivery statuses
            delivery_statuses = uow.notifications.delivery_status.get_delivery_statuses_by_message(message_id)

            # Convert to response format
            responses = []
            for ds in delivery_statuses:
                responses.append(DeliveryStatusResponse(
                    delivery_id=ds.id,
                    channel=ds.channel,
                    status=ds.status,
                    delivered_at=ds.delivered_at.isoformat() if ds.delivered_at else None,
                    response_time_ms=ds.response_time_ms,
                    error_message=ds.error_message,
                    external_id=ds.external_id
                ))

            return responses

    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("Error getting delivery status:")
        raise HTTPException(status_code=500, detail="Failed to get delivery status")
