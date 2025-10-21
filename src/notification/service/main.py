"""
Notification Service Main Application

FastAPI application for the notification service.
Provides REST API endpoints for message enqueueing, status tracking, and health monitoring.
"""

from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
from sqlalchemy import desc, asc, func, case
import asyncio
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.notification.service.config import config
from src.notification.service.dependencies import (
    init_database, get_notification_repo, get_config
)
from src.data.db.repos.repo_notification import NotificationRepository
from src.data.db.models.model_notification import (
    Message, MessageDeliveryStatus,
    MessageCreate, MessageResponse, MessageUpdate,
    DeliveryStatusResponse, ChannelHealthResponse,
    MessagePriority, MessageStatus, DeliveryStatus
)

from src.notification.logger import setup_logger
_logger = setup_logger(__name__)

# Global services (will be initialized in lifespan)
message_processor = None
health_monitor = None


async def _register_channel_plugins():
    """Register available channel plugins."""
    try:
        from src.notification.channels.base import channel_registry
        from src.notification.channels.telegram_channel import TelegramChannel
        from src.notification.channels.email_channel import EmailChannel
        from src.notification.channels.sms_channel import SMSChannel

        # Register channel plugins
        channel_registry.register_channel('telegram', TelegramChannel)
        channel_registry.register_channel('email', EmailChannel)
        channel_registry.register_channel('sms', SMSChannel)

        _logger.info("Registered channel plugins: %s", channel_registry.list_channels())

    except Exception as e:
        _logger.error("Error registering channel plugins: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global message_processor, health_monitor

    # Startup
    _logger.info("Starting Notification Service...")

    # Initialize database
    init_database(
        database_url=config.database.url,
        echo=config.database.echo,
        pool_size=config.database.pool_size,
        max_overflow=config.database.max_overflow
    )
    _logger.info("Database initialized")

    # Register channel plugins
    await _register_channel_plugins()

    # Initialize message processor
    from src.notification.service.processor import message_processor as mp
    message_processor = mp
    await message_processor.start()
    _logger.info("Message processor started")

    # Initialize health monitor
    from src.notification.service.health_monitor import health_monitor
    await health_monitor.start()
    _logger.info("Health monitor started")

    yield

    # Shutdown
    _logger.info("Shutting down Notification Service...")
    if message_processor:
        await message_processor.shutdown()
    if health_monitor:
        await health_monitor.stop()


# Create FastAPI application
app = FastAPI(
    title="Notification Service API",
    description="REST API for notification message management and delivery",
    version=config.version,
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle ValueError exceptions."""
    return JSONResponse(
        status_code=400,
        content={"error": "Bad Request", "detail": str(exc)}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    _logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": "An unexpected error occurred"}
    )


# Health check endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": config.service_name,
        "version": config.version,
        "status": "running"
    }


@app.get("/api/v1/health")
async def health_check():
    """Service health check endpoint."""
    try:
        # Test database connection
        with get_notification_repo() as repo:
            # Simple query to test database connectivity
            repo.session.execute("SELECT 1")

        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": config.service_name,
            "version": config.version,
            "database": "connected"
        }
    except Exception as e:
        _logger.error("Health check failed: %s", e)
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "service": config.service_name,
                "version": config.version,
                "error": str(e)
            }
        )


# Message management endpoints
@app.post("/api/v1/messages", response_model=Dict[str, Any])
async def enqueue_message(
    message: MessageCreate,
    background_tasks: BackgroundTasks,
    repo: NotificationRepository = Depends(get_notification_repo)
):
    """
    Enqueue a new message for delivery.

    Args:
        message: Message data
        background_tasks: FastAPI background tasks
        repo: Notification repository

    Returns:
        Message ID and status
    """
    try:
        # Validate channels are enabled
        enabled_channels = []
        for channel in message.channels:
            if config.is_channel_enabled(channel):
                enabled_channels.append(channel)
            else:
                _logger.warning("Channel %s is not enabled, skipping", channel)

        if not enabled_channels:
            raise ValueError("No enabled channels specified")

        # Create message data
        message_data = message.dict()
        message_data['channels'] = enabled_channels

        # Set default scheduled_for if not provided
        if not message_data.get('scheduled_for'):
            message_data['scheduled_for'] = datetime.now(timezone.utc)

        # Create message in database
        db_message = repo.messages.create_message(message_data)
        repo.commit()

        # Message processing is handled automatically by the processor workers

        _logger.info("Message %s enqueued successfully", db_message.id)

        return {
            "message_id": db_message.id,
            "status": "enqueued",
            "channels": enabled_channels,
            "priority": db_message.priority
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        repo.rollback()
        _logger.error("Failed to enqueue message: %s", e)
        raise HTTPException(status_code=500, detail="Failed to enqueue message")


@app.get("/api/v1/messages/{message_id}/status", response_model=MessageResponse)
async def get_message_status(
    message_id: int,
    repo: NotificationRepository = Depends(get_notification_repo)
):
    """
    Get message status and details.

    Args:
        message_id: Message ID
        repo: Notification repository

    Returns:
        Message details and status
    """
    try:
        message = repo.messages.get_message(message_id)
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")

        return MessageResponse.from_orm(message)

    except HTTPException:
        raise
    except Exception as e:
        _logger.error("Failed to get message status: %s", e)
        raise HTTPException(status_code=500, detail="Failed to get message status")


@app.get("/api/v1/messages", response_model=List[MessageResponse])
async def list_messages(
    status: Optional[MessageStatus] = None,
    priority: Optional[MessagePriority] = None,
    recipient_id: Optional[str] = None,
    message_type: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    repo: NotificationRepository = Depends(get_notification_repo)
):
    """
    List messages with optional filtering.

    Args:
        status: Filter by message status
        priority: Filter by message priority
        recipient_id: Filter by recipient ID
        message_type: Filter by message type
        limit: Maximum number of results
        offset: Number of results to skip
        repo: Notification repository

    Returns:
        List of messages
    """
    try:
        messages = repo.messages.list_messages(
            status=status,
            priority=priority,
            recipient_id=recipient_id,
            message_type=message_type,
            limit=min(limit, 1000),  # Cap at 1000
            offset=offset
        )

        return [MessageResponse.from_orm(msg) for msg in messages]

    except Exception as e:
        _logger.error("Failed to list messages: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list messages")


@app.get("/api/v1/messages/{message_id}/delivery", response_model=List[DeliveryStatusResponse])
async def get_message_delivery_status(
    message_id: int,
    repo: NotificationRepository = Depends(get_notification_repo)
):
    """
    Get delivery status for all channels of a message.

    Args:
        message_id: Message ID
        repo: Notification repository

    Returns:
        List of delivery statuses
    """
    try:
        # Check if message exists
        message = repo.messages.get_message(message_id)
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")

        # Get delivery statuses
        delivery_statuses = repo.delivery_status.get_delivery_statuses_by_message(message_id)

        return [DeliveryStatusResponse.from_orm(ds) for ds in delivery_statuses]

    except HTTPException:
        raise
    except Exception as e:
        _logger.error("Failed to get delivery status: %s", e)
        raise HTTPException(status_code=500, detail="Failed to get delivery status")


# Channel management endpoints
@app.get("/api/v1/channels", response_model=List[Dict[str, Any]])
async def list_channels(
    repo: NotificationRepository = Depends(get_notification_repo)
):
    """
    List available notification channels.

    Args:
        repo: Notification repository

    Returns:
        List of channel configurations
    """
    try:
        # Get channel configurations from database
        channel_configs = repo.channel_configs.list_channel_configs()

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
                    "enabled": config.is_channel_enabled(channel_name),
                    "rate_limit_per_minute": config.get_rate_limit_for_channel(channel_name),
                    "max_retries": 3,
                    "timeout_seconds": 30
                })

        return channels

    except Exception as e:
        _logger.error("Failed to list channels: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list channels")


@app.get("/api/v1/channels/health", response_model=List[ChannelHealthResponse])
async def get_channels_health(
    repo: NotificationRepository = Depends(get_notification_repo)
):
    """
    Get health status for all channels.

    Args:
        repo: Notification repository

    Returns:
        List of channel health statuses
    """
    try:
        health_records = repo.channel_health.list_channel_health()
        return [ChannelHealthResponse.from_orm(health) for health in health_records]

    except Exception as e:
        _logger.error("Failed to get channel health: %s", e)
        raise HTTPException(status_code=500, detail="Failed to get channel health")


# Statistics endpoints
@app.get("/api/v1/stats", response_model=Dict[str, Any])
async def get_delivery_statistics(
    channel: Optional[str] = None,
    days: int = 30,
    repo: NotificationRepository = Depends(get_notification_repo)
):
    """
    Get delivery statistics.

    Args:
        channel: Filter by channel (optional)
        days: Number of days to look back
        repo: Notification repository

    Returns:
        Delivery statistics
    """
    try:
        if days < 1 or days > 365:
            raise ValueError("Days must be between 1 and 365")

        stats = repo.delivery_status.get_delivery_statistics(channel=channel, days=days)

        # Add message statistics
        message_stats = {}
        for status in MessageStatus:
            count = repo.session.query(repo.messages.session.query(repo.messages.__class__).filter(
                repo.messages.__class__.status == status.value
            ).count())
            message_stats[status.value] = count

        return {
            "delivery_statistics": stats,
            "message_statistics": message_stats,
            "period_days": days,
            "channel_filter": channel
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _logger.error("Failed to get statistics: %s", e)
        raise HTTPException(status_code=500, detail="Failed to get statistics")


# Delivery History API endpoints
@app.get("/api/v1/history/messages", response_model=Dict[str, Any])
async def get_message_history(
    user_id: Optional[str] = None,
    channel: Optional[str] = None,
    status: Optional[MessageStatus] = None,
    message_type: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = 100,
    offset: int = 0,
    order_by: str = "created_at",
    order_desc: bool = True,
    repo: NotificationRepository = Depends(get_notification_repo)
):
    """
    Get message history with filtering and pagination.

    Args:
        user_id: Filter by recipient user ID
        channel: Filter by specific channel
        status: Filter by message status
        message_type: Filter by message type
        start_date: Filter messages created after this date
        end_date: Filter messages created before this date
        limit: Maximum number of results (max 1000)
        offset: Number of results to skip
        order_by: Field to order by (created_at, scheduled_for, processed_at)
        order_desc: Order in descending order
        repo: Notification repository

    Returns:
        Paginated message history with metadata
    """
    try:
        # Validate parameters
        if limit < 1 or limit > 1000:
            raise ValueError("Limit must be between 1 and 1000")

        if offset < 0:
            raise ValueError("Offset must be non-negative")

        if order_by not in ["created_at", "scheduled_for", "processed_at"]:
            raise ValueError("Invalid order_by field")

        # Build query filters
        query = repo.session.query(Message)

        if user_id:
            query = query.filter(Message.recipient_id == user_id)

        if status:
            query = query.filter(Message.status == status.value)

        if message_type:
            query = query.filter(Message.message_type == message_type)

        if start_date:
            query = query.filter(Message.created_at >= start_date)

        if end_date:
            query = query.filter(Message.created_at <= end_date)

        if channel:
            # Filter messages that include the specified channel
            # Use PostgreSQL array contains operation with proper casting
            from sqlalchemy import text
            query = query.filter(text("channels @> ARRAY[:channel]")).params(channel=channel)

        # Get total count for pagination
        total_count = query.count()

        # Apply ordering
        order_field = getattr(Message, order_by)
        if order_desc:
            query = query.order_by(desc(order_field))
        else:
            query = query.order_by(asc(order_field))

        # Apply pagination
        messages = query.offset(offset).limit(limit).all()

        # Convert to response format
        message_responses = [MessageResponse.from_orm(msg) for msg in messages]

        return {
            "messages": message_responses,
            "pagination": {
                "total": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total_count
            },
            "filters": {
                "user_id": user_id,
                "channel": channel,
                "status": status.value if status else None,
                "message_type": message_type,
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None
            }
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _logger.error("Failed to get message history: %s", e)
        raise HTTPException(status_code=500, detail="Failed to get message history")


@app.get("/api/v1/history/deliveries", response_model=Dict[str, Any])
async def get_delivery_history(
    message_id: Optional[int] = None,
    user_id: Optional[str] = None,
    channel: Optional[str] = None,
    status: Optional[DeliveryStatus] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = 100,
    offset: int = 0,
    order_by: str = "created_at",
    order_desc: bool = True,
    repo: NotificationRepository = Depends(get_notification_repo)
):
    """
    Get delivery history with filtering and pagination.

    Args:
        message_id: Filter by specific message ID
        user_id: Filter by recipient user ID (requires join with messages)
        channel: Filter by specific channel
        status: Filter by delivery status
        start_date: Filter deliveries created after this date
        end_date: Filter deliveries created before this date
        limit: Maximum number of results (max 1000)
        offset: Number of results to skip
        order_by: Field to order by (created_at, delivered_at)
        order_desc: Order in descending order
        repo: Notification repository

    Returns:
        Paginated delivery history with metadata
    """
    try:
        # Validate parameters
        if limit < 1 or limit > 1000:
            raise ValueError("Limit must be between 1 and 1000")

        if offset < 0:
            raise ValueError("Offset must be non-negative")

        if order_by not in ["created_at", "delivered_at"]:
            raise ValueError("Invalid order_by field")

        # Build query with optional join
        if user_id:
            # Need to join with messages table to filter by user_id
            query = repo.session.query(MessageDeliveryStatus).join(Message).filter(
                Message.recipient_id == user_id
            )
        else:
            query = repo.session.query(MessageDeliveryStatus)

        if message_id:
            query = query.filter(MessageDeliveryStatus.message_id == message_id)

        if channel:
            query = query.filter(MessageDeliveryStatus.channel == channel)

        if status:
            query = query.filter(MessageDeliveryStatus.status == status.value)

        if start_date:
            query = query.filter(MessageDeliveryStatus.created_at >= start_date)

        if end_date:
            query = query.filter(MessageDeliveryStatus.created_at <= end_date)

        # Get total count for pagination
        total_count = query.count()

        # Apply ordering
        order_field = getattr(MessageDeliveryStatus, order_by)
        if order_desc:
            query = query.order_by(desc(order_field))
        else:
            query = query.order_by(asc(order_field))

        # Apply pagination
        deliveries = query.offset(offset).limit(limit).all()

        # Convert to response format
        delivery_responses = [DeliveryStatusResponse.from_orm(delivery) for delivery in deliveries]

        return {
            "deliveries": delivery_responses,
            "pagination": {
                "total": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total_count
            },
            "filters": {
                "message_id": message_id,
                "user_id": user_id,
                "channel": channel,
                "status": status.value if status else None,
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None
            }
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _logger.error("Failed to get delivery history: %s", e)
        raise HTTPException(status_code=500, detail="Failed to get delivery history")


@app.get("/api/v1/history/export", response_model=Dict[str, Any])
async def export_history(
    format: str = "json",
    user_id: Optional[str] = None,
    channel: Optional[str] = None,
    status: Optional[MessageStatus] = None,
    message_type: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = 10000,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    repo: NotificationRepository = Depends(get_notification_repo)
):
    """
    Export message and delivery history for analytics.

    Args:
        format: Export format (json, csv)
        user_id: Filter by recipient user ID
        channel: Filter by specific channel
        status: Filter by message status
        message_type: Filter by message type
        start_date: Filter messages created after this date
        end_date: Filter messages created before this date
        limit: Maximum number of records to export (max 50000)
        background_tasks: FastAPI background tasks
        repo: Notification repository

    Returns:
        Export status and download information
    """
    try:
        # Validate parameters
        if format not in ["json", "csv"]:
            raise ValueError("Format must be 'json' or 'csv'")

        if limit < 1 or limit > 50000:
            raise ValueError("Limit must be between 1 and 50000")

        # For small exports, return data immediately
        if limit <= 1000:
            # Build query
            query = repo.session.query(Message)

            if user_id:
                query = query.filter(Message.recipient_id == user_id)

            if status:
                query = query.filter(Message.status == status.value)

            if message_type:
                query = query.filter(Message.message_type == message_type)

            if start_date:
                query = query.filter(Message.created_at >= start_date)

            if end_date:
                query = query.filter(Message.created_at <= end_date)

            if channel:
                # Filter messages that include the specified channel
                # Use PostgreSQL array contains operation with proper casting
                from sqlalchemy import text
                query = query.filter(text("channels @> ARRAY[:channel]")).params(channel=channel)

            # Get messages with delivery statuses
            messages = query.order_by(desc(Message.created_at)).limit(limit).all()

            export_data = []
            for message in messages:
                message_data = {
                    "message_id": message.id,
                    "message_type": message.message_type,
                    "priority": message.priority,
                    "channels": message.channels,
                    "recipient_id": message.recipient_id,
                    "template_name": message.template_name,
                    "content": message.content,
                    "metadata": message.message_metadata,
                    "created_at": message.created_at.isoformat(),
                    "scheduled_for": message.scheduled_for.isoformat(),
                    "status": message.status,
                    "retry_count": message.retry_count,
                    "max_retries": message.max_retries,
                    "last_error": message.last_error,
                    "processed_at": message.processed_at.isoformat() if message.processed_at else None,
                    "deliveries": []
                }

                # Add delivery statuses
                for delivery in message.delivery_statuses:
                    delivery_data = {
                        "delivery_id": delivery.id,
                        "channel": delivery.channel,
                        "status": delivery.status,
                        "delivered_at": delivery.delivered_at.isoformat() if delivery.delivered_at else None,
                        "response_time_ms": delivery.response_time_ms,
                        "error_message": delivery.error_message,
                        "external_id": delivery.external_id,
                        "created_at": delivery.created_at.isoformat()
                    }
                    message_data["deliveries"].append(delivery_data)

                export_data.append(message_data)

            if format == "json":
                return {
                    "export_type": "immediate",
                    "format": format,
                    "record_count": len(export_data),
                    "data": export_data,
                    "filters": {
                        "user_id": user_id,
                        "channel": channel,
                        "status": status.value if status else None,
                        "message_type": message_type,
                        "start_date": start_date.isoformat() if start_date else None,
                        "end_date": end_date.isoformat() if end_date else None
                    }
                }
            else:  # CSV format
                import csv
                import io

                output = io.StringIO()
                writer = csv.writer(output)

                # Write header
                writer.writerow([
                    "message_id", "message_type", "priority", "channels", "recipient_id",
                    "template_name", "created_at", "scheduled_for", "status", "retry_count",
                    "delivery_id", "delivery_channel", "delivery_status", "delivered_at",
                    "response_time_ms", "error_message", "external_id"
                ])

                # Write data
                for message_data in export_data:
                    if message_data["deliveries"]:
                        for delivery in message_data["deliveries"]:
                            writer.writerow([
                                message_data["message_id"],
                                message_data["message_type"],
                                message_data["priority"],
                                ",".join(message_data["channels"]),
                                message_data["recipient_id"],
                                message_data["template_name"],
                                message_data["created_at"],
                                message_data["scheduled_for"],
                                message_data["status"],
                                message_data["retry_count"],
                                delivery["delivery_id"],
                                delivery["channel"],
                                delivery["status"],
                                delivery["delivered_at"],
                                delivery["response_time_ms"],
                                delivery["error_message"],
                                delivery["external_id"]
                            ])
                    else:
                        # Message without deliveries
                        writer.writerow([
                            message_data["message_id"],
                            message_data["message_type"],
                            message_data["priority"],
                            ",".join(message_data["channels"]),
                            message_data["recipient_id"],
                            message_data["template_name"],
                            message_data["created_at"],
                            message_data["scheduled_for"],
                            message_data["status"],
                            message_data["retry_count"],
                            None, None, None, None, None, None, None
                        ])

                csv_data = output.getvalue()
                output.close()

                return {
                    "export_type": "immediate",
                    "format": format,
                    "record_count": len(export_data),
                    "data": csv_data,
                    "filters": {
                        "user_id": user_id,
                        "channel": channel,
                        "status": status.value if status else None,
                        "message_type": message_type,
                        "start_date": start_date.isoformat() if start_date else None,
                        "end_date": end_date.isoformat() if end_date else None
                    }
                }

        else:
            # For large exports, process in background
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            user_part = user_id or 'all'
            export_id = f"export_{timestamp}_{user_part}"

            def background_export():
                # This would be implemented to generate large exports
                # For now, return a placeholder
                _logger.info("Background export %s started", export_id)

            background_tasks.add_task(background_export)

            return {
                "export_type": "background",
                "export_id": export_id,
                "status": "processing",
                "message": "Large export started in background. Check status with export_id.",
                "estimated_completion": "5-10 minutes"
            }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _logger.error("Failed to export history: %s", e)
        raise HTTPException(status_code=500, detail="Failed to export history")


@app.get("/api/v1/history/summary", response_model=Dict[str, Any])
async def get_history_summary(
    user_id: Optional[str] = None,
    channel: Optional[str] = None,
    days: int = 30,
    repo: NotificationRepository = Depends(get_notification_repo)
):
    """
    Get summary statistics for message and delivery history.

    Args:
        user_id: Filter by recipient user ID
        channel: Filter by specific channel
        days: Number of days to analyze
        repo: Notification repository

    Returns:
        Summary statistics for the specified period
    """
    try:
        if days < 1 or days > 365:
            raise ValueError("Days must be between 1 and 365")

        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

        # Message statistics
        message_query = repo.session.query(Message).filter(
            Message.created_at >= cutoff_date
        )

        if user_id:
            message_query = message_query.filter(Message.recipient_id == user_id)

        if channel:
            # Filter messages that include the specified channel
            # Use PostgreSQL array contains operation with proper casting
            from sqlalchemy import text
            message_query = message_query.filter(text("channels @> ARRAY[:channel]")).params(channel=channel)

        # Count messages by status
        message_stats = {}
        for status in MessageStatus:
            count = message_query.filter(Message.status == status.value).count()
            message_stats[status.value] = count

        total_messages = message_query.count()

        # Delivery statistics
        delivery_query = repo.session.query(MessageDeliveryStatus).filter(
            MessageDeliveryStatus.created_at >= cutoff_date
        )

        if user_id:
            delivery_query = delivery_query.join(Message).filter(
                Message.recipient_id == user_id
            )

        if channel:
            delivery_query = delivery_query.filter(MessageDeliveryStatus.channel == channel)

        # Count deliveries by status
        delivery_stats = {}
        for status in DeliveryStatus:
            count = delivery_query.filter(MessageDeliveryStatus.status == status.value).count()
            delivery_stats[status.value] = count

        total_deliveries = delivery_query.count()

        # Calculate success rates
        message_success_rate = (
            message_stats.get(MessageStatus.DELIVERED.value, 0) / total_messages
            if total_messages > 0 else 0.0
        )

        delivery_success_rate = (
            delivery_stats.get(DeliveryStatus.DELIVERED.value, 0) / total_deliveries
            if total_deliveries > 0 else 0.0
        )

        # Get channel breakdown if not filtering by channel
        channel_breakdown = {}
        if not channel:
            channel_stats = repo.session.query(
                MessageDeliveryStatus.channel,
                func.count(MessageDeliveryStatus.id).label('total'),
                func.sum(
                    case(
                        (MessageDeliveryStatus.status == DeliveryStatus.DELIVERED.value, 1),
                        else_=0
                    )
                ).label('delivered')
            ).filter(
                MessageDeliveryStatus.created_at >= cutoff_date
            )

            if user_id:
                channel_stats = channel_stats.join(Message).filter(
                    Message.recipient_id == user_id
                )

            channel_stats = channel_stats.group_by(MessageDeliveryStatus.channel).all()

            for row in channel_stats:
                success_rate = row.delivered / row.total if row.total > 0 else 0.0
                channel_breakdown[row.channel] = {
                    "total_deliveries": row.total,
                    "successful_deliveries": row.delivered,
                    "success_rate": success_rate
                }

        return {
            "period": {
                "days": days,
                "start_date": cutoff_date.isoformat(),
                "end_date": datetime.now(timezone.utc).isoformat()
            },
            "filters": {
                "user_id": user_id,
                "channel": channel
            },
            "message_statistics": {
                "total": total_messages,
                "by_status": message_stats,
                "success_rate": message_success_rate
            },
            "delivery_statistics": {
                "total": total_deliveries,
                "by_status": delivery_stats,
                "success_rate": delivery_success_rate
            },
            "channel_breakdown": channel_breakdown
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _logger.error("Failed to get history summary: %s", e)
        raise HTTPException(status_code=500, detail="Failed to get history summary")


# Advanced Analytics endpoints
@app.get("/api/v1/analytics/delivery-rates", response_model=Dict[str, Any])
async def get_delivery_rates(
    channel: Optional[str] = None,
    user_id: Optional[str] = None,
    days: int = 30
):
    """
    Get comprehensive delivery rate analysis.

    Args:
        channel: Filter by specific channel
        user_id: Filter by specific user
        days: Number of days to analyze

    Returns:
        Delivery rate statistics per channel and user
    """
    try:
        from src.notification.service.analytics import notification_analytics

        if days < 1 or days > 365:
            raise ValueError("Days must be between 1 and 365")

        rates = await notification_analytics.get_delivery_rates(
            channel=channel, user_id=user_id, days=days
        )
        return rates

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _logger.error("Failed to get delivery rates: %s", e)
        raise HTTPException(status_code=500, detail="Failed to get delivery rates")


@app.get("/api/v1/analytics/response-times", response_model=Dict[str, Any])
async def get_response_time_analysis(
    channel: Optional[str] = None,
    days: int = 30
):
    """
    Get detailed response time analysis.

    Args:
        channel: Filter by specific channel
        days: Number of days to analyze

    Returns:
        Response time statistics and percentiles
    """
    try:
        from src.notification.service.analytics import notification_analytics

        if days < 1 or days > 365:
            raise ValueError("Days must be between 1 and 365")

        analysis = await notification_analytics.get_response_time_analysis(
            channel=channel, days=days
        )
        return analysis

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _logger.error("Failed to get response time analysis: %s", e)
        raise HTTPException(status_code=500, detail="Failed to get response time analysis")


@app.get("/api/v1/analytics/aggregated", response_model=Dict[str, Any])
async def get_aggregated_statistics(
    granularity: str = "daily",
    days: int = 30,
    channel: Optional[str] = None
):
    """
    Get time-aggregated statistics.

    Args:
        granularity: Time granularity (hourly, daily, weekly, monthly)
        days: Number of days to analyze
        channel: Filter by specific channel

    Returns:
        Time-series aggregated statistics
    """
    try:
        from src.notification.service.analytics import notification_analytics, TimeGranularity

        if days < 1 or days > 365:
            raise ValueError("Days must be between 1 and 365")

        # Validate granularity
        try:
            time_granularity = TimeGranularity(granularity.lower())
        except ValueError:
            raise ValueError(f"Invalid granularity. Must be one of: {[g.value for g in TimeGranularity]}")

        stats = await notification_analytics.get_aggregated_statistics(
            granularity=time_granularity, days=days, channel=channel
        )
        return stats

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _logger.error("Failed to get aggregated statistics: %s", e)
        raise HTTPException(status_code=500, detail="Failed to get aggregated statistics")


@app.get("/api/v1/analytics/trends/{metric}", response_model=Dict[str, Any])
async def get_trend_analysis(
    metric: str,
    days: int = 30,
    channel: Optional[str] = None
):
    """
    Get trend analysis for a specific metric.

    Args:
        metric: Metric to analyze (success_rate, response_time, message_count)
        days: Number of days to analyze
        channel: Filter by specific channel

    Returns:
        Trend analysis with direction, strength, and statistics
    """
    try:
        from src.notification.service.analytics import notification_analytics

        if days < 1 or days > 365:
            raise ValueError("Days must be between 1 and 365")

        # Validate metric
        valid_metrics = ["success_rate", "response_time", "message_count"]
        if metric not in valid_metrics:
            raise ValueError(f"Invalid metric. Must be one of: {valid_metrics}")

        trend = await notification_analytics.get_trend_analysis(
            metric=metric, days=days, channel=channel
        )
        return trend.to_dict()

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _logger.error("Failed to get trend analysis: %s", e)
        raise HTTPException(status_code=500, detail="Failed to get trend analysis")


@app.get("/api/v1/analytics/channel-comparison", response_model=Dict[str, Any])
async def get_channel_performance_comparison(days: int = 30):
    """
    Compare performance across all channels.

    Args:
        days: Number of days to analyze

    Returns:
        Channel performance comparison with rankings
    """
    try:
        from src.notification.service.analytics import notification_analytics

        if days < 1 or days > 365:
            raise ValueError("Days must be between 1 and 365")

        comparison = await notification_analytics.get_channel_performance_comparison(days=days)
        return comparison

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _logger.error("Failed to get channel performance comparison: %s", e)
        raise HTTPException(status_code=500, detail="Failed to get channel performance comparison")


# Administrative endpoints
@app.post("/api/v1/admin/cleanup")
async def cleanup_old_messages(
    background_tasks: BackgroundTasks,
    days_to_keep: int = 30,
    repo: NotificationRepository = Depends(get_notification_repo)
):
    """
    Clean up old delivered messages.

    Args:
        days_to_keep: Number of days of messages to keep
        background_tasks: FastAPI background tasks
        repo: Notification repository

    Returns:
        Cleanup status
    """
    try:
        if days_to_keep < 1:
            raise ValueError("Days to keep must be at least 1")

        # Run cleanup in background
        def cleanup_task():
            deleted_count = repo.messages.cleanup_old_messages(days_to_keep)
            repo.commit()
            _logger.info("Cleanup completed: %s messages deleted", deleted_count)

        background_tasks.add_task(cleanup_task)

        return {
            "status": "cleanup_started",
            "days_to_keep": days_to_keep,
            "message": "Cleanup task started in background"
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _logger.error("Failed to start cleanup: %s", e)
        raise HTTPException(status_code=500, detail="Failed to start cleanup")


@app.get("/api/v1/processor/stats", response_model=Dict[str, Any])
async def get_processor_statistics():
    """
    Get message processor statistics.

    Returns:
        Processor statistics and performance metrics
    """
    try:
        if not message_processor:
            raise HTTPException(status_code=503, detail="Message processor not available")

        stats = message_processor.get_stats()

        return {
            "processor_stats": stats,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        _logger.error("Failed to get processor statistics: %s", e)
        raise HTTPException(status_code=500, detail="Failed to get processor statistics")


# Fallback and Recovery Management endpoints
@app.get("/api/v1/fallback/dead-letters", response_model=Dict[str, Any])
async def get_dead_letter_messages(
    limit: int = 100,
    offset: int = 0
):
    """
    Get messages from the dead letter queue.

    Args:
        limit: Maximum number of messages to return (max 1000)
        offset: Number of messages to skip

    Returns:
        Paginated list of dead letter messages
    """
    try:
        if limit < 1 or limit > 1000:
            raise ValueError("Limit must be between 1 and 1000")

        if offset < 0:
            raise ValueError("Offset must be non-negative")

        if not message_processor:
            raise HTTPException(status_code=503, detail="Message processor not available")

        messages = message_processor.fallback_manager.get_dead_letter_messages(limit, offset)
        total_count = len(message_processor.fallback_manager._dead_letter_queue)

        return {
            "messages": messages,
            "pagination": {
                "total": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total_count
            }
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _logger.error("Failed to get dead letter messages: %s", e)
        raise HTTPException(status_code=500, detail="Failed to get dead letter messages")


@app.post("/api/v1/fallback/dead-letters/{message_id}/reprocess")
async def reprocess_dead_letter_message(
    message_id: int,
    force_channels: Optional[List[str]] = None
):
    """
    Manually reprocess a message from the dead letter queue.

    Args:
        message_id: Message ID to reprocess
        force_channels: Optional list of channels to force use

    Returns:
        Reprocessing result
    """
    try:
        if not message_processor:
            raise HTTPException(status_code=503, detail="Message processor not available")

        success, message = await message_processor.fallback_manager.reprocess_dead_letter_message(
            message_id, message_processor._channel_instances, force_channels
        )

        return {
            "success": success,
            "message": message,
            "message_id": message_id,
            "forced_channels": force_channels
        }

    except Exception as e:
        _logger.error("Failed to reprocess dead letter message %s: %s", message_id, e)
        raise HTTPException(status_code=500, detail="Failed to reprocess message")


@app.get("/api/v1/fallback/retry-queue", response_model=Dict[str, Any])
async def get_retry_queue_status():
    """
    Get retry queue status information.

    Returns:
        Retry queue status and statistics
    """
    try:
        if not message_processor:
            raise HTTPException(status_code=503, detail="Message processor not available")

        status = message_processor.fallback_manager.get_retry_queue_status()
        return status

    except Exception as e:
        _logger.error("Failed to get retry queue status: %s", e)
        raise HTTPException(status_code=500, detail="Failed to get retry queue status")


@app.get("/api/v1/fallback/statistics", response_model=Dict[str, Any])
async def get_fallback_statistics():
    """
    Get comprehensive fallback and recovery statistics.

    Returns:
        Fallback statistics including success rates and recent attempts
    """
    try:
        if not message_processor:
            raise HTTPException(status_code=503, detail="Message processor not available")

        stats = message_processor.fallback_manager.get_fallback_statistics()
        return stats

    except Exception as e:
        _logger.error("Failed to get fallback statistics: %s", e)
        raise HTTPException(status_code=500, detail="Failed to get fallback statistics")


@app.post("/api/v1/fallback/rules/{channel}")
async def configure_fallback_rule(
    channel: str,
    fallback_channels: List[str],
    strategy: str = "priority_order",
    max_attempts: int = 3,
    retry_delay_seconds: int = 60,
    enabled: bool = True
):
    """
    Configure fallback rule for a channel.

    Args:
        channel: Primary channel name
        fallback_channels: List of fallback channel names
        strategy: Fallback strategy (priority_order, round_robin, health_based, load_balanced)
        max_attempts: Maximum fallback attempts
        retry_delay_seconds: Delay between retry attempts
        enabled: Whether the rule is enabled

    Returns:
        Configuration result
    """
    try:
        if not message_processor:
            raise HTTPException(status_code=503, detail="Message processor not available")

        from src.notification.service.fallback_manager import FallbackRule, FallbackStrategy

        # Validate strategy
        try:
            strategy_enum = FallbackStrategy(strategy)
        except ValueError:
            raise ValueError(f"Invalid strategy. Must be one of: {[s.value for s in FallbackStrategy]}")

        # Create fallback rule
        rule = FallbackRule(
            primary_channel=channel,
            fallback_channels=fallback_channels,
            strategy=strategy_enum,
            max_attempts=max_attempts,
            retry_delay_seconds=retry_delay_seconds,
            enabled=enabled
        )

        # Configure the rule
        message_processor.fallback_manager.configure_fallback_rule(rule)

        return {
            "success": True,
            "message": f"Fallback rule configured for channel {channel}",
            "rule": {
                "primary_channel": channel,
                "fallback_channels": fallback_channels,
                "strategy": strategy,
                "max_attempts": max_attempts,
                "retry_delay_seconds": retry_delay_seconds,
                "enabled": enabled
            }
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _logger.error("Failed to configure fallback rule for %s: %s", channel, e)
        raise HTTPException(status_code=500, detail="Failed to configure fallback rule")


@app.delete("/api/v1/fallback/rules/{channel}")
async def remove_fallback_rule(channel: str):
    """
    Remove fallback rule for a channel.

    Args:
        channel: Channel name

    Returns:
        Removal result
    """
    try:
        if not message_processor:
            raise HTTPException(status_code=503, detail="Message processor not available")

        success = message_processor.fallback_manager.remove_fallback_rule(channel)

        if success:
            return {
                "success": True,
                "message": f"Fallback rule removed for channel {channel}"
            }
        else:
            raise HTTPException(status_code=404, detail=f"No fallback rule found for channel {channel}")

    except HTTPException:
        raise
    except Exception as e:
        _logger.error("Failed to remove fallback rule for %s: %s", channel, e)
        raise HTTPException(status_code=500, detail="Failed to remove fallback rule")


@app.post("/api/v1/fallback/global-channels")
async def set_global_fallback_channels(channels: List[str]):
    """
    Set global fallback channels.

    Args:
        channels: List of channel names to use as global fallbacks

    Returns:
        Configuration result
    """
    try:
        if not message_processor:
            raise HTTPException(status_code=503, detail="Message processor not available")

        message_processor.fallback_manager.set_global_fallback_channels(channels)

        return {
            "success": True,
            "message": "Global fallback channels configured",
            "channels": channels
        }

    except Exception as e:
        _logger.error("Failed to set global fallback channels: %s", e)
        raise HTTPException(status_code=500, detail="Failed to set global fallback channels")


if __name__ == "__main__":
    import uvicorn

    # Run the server
    uvicorn.run(
        "main:app",
        host=config.server.host,
        port=config.server.port,
        reload=config.server.reload,
        log_level=config.server.log_level
    )