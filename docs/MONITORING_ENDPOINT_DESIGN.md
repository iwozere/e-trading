# Monitoring Endpoint Design - Channel Health Check

## Overview

This document provides a detailed implementation plan for adding a health check endpoint that shows channel ownership and queue status for both Telegram Bot and Notification Service.

---

## Part 3: Detailed Implementation Plan

### Architecture Decision

**Where to add the endpoint?**

We have **two separate services** that need to report their status:

1. **Telegram Bot** - Runs its own async aiogram application
2. **Notification Service** - Has its own FastAPI/Flask server

**Recommended Approach: Add endpoint to BOTH services**

Each service exposes its own `/health/channels` endpoint showing what it owns.

---

## Implementation for Notification Service

### File: `src/notification/service/health_endpoints.py` (NEW)

```python
"""
Health check endpoints for notification service.
Shows channel ownership and queue status.
"""

from fastapi import APIRouter, Depends
from typing import Dict, Any
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from src.notification.service.database import get_db
from src.notification.service.config import get_config
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/channels")
async def get_channel_health(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Get health status for notification service channels.

    Returns:
        Channel ownership, queue status, and health metrics
    """
    config = get_config()

    # Get enabled channels from config
    enabled_channels = config.enabled_channels

    # Query database for queue status
    queue_stats = _get_queue_stats(db, enabled_channels)

    # Get service uptime
    uptime_seconds = _get_service_uptime()

    return {
        "service": "notification_service",
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "channels": {
            "owned": enabled_channels,
            "telegram_owned_by": "telegram_bot",  # Explicitly state telegram is NOT owned
            "email_status": "enabled" if "email" in enabled_channels else "disabled",
            "sms_status": "enabled" if "sms" in enabled_channels else "disabled"
        },
        "queue": {
            "pending": queue_stats["pending"],
            "processing": queue_stats["processing"],
            "failed_last_hour": queue_stats["failed_last_hour"],
            "delivered_last_hour": queue_stats["delivered_last_hour"]
        },
        "uptime_seconds": uptime_seconds,
        "config": {
            "poll_interval": config.processing.poll_interval,
            "batch_size": config.processing.batch_size
        }
    }


def _get_queue_stats(db: Session, channels: list) -> Dict[str, int]:
    """
    Get queue statistics for owned channels.

    Args:
        db: Database session
        channels: List of owned channels (e.g., ["email"])

    Returns:
        Dictionary with queue counts
    """
    from src.notification.service.database import Message

    try:
        # Build query to filter by owned channels
        # WHERE 'email' = ANY(channels) OR 'sms' = ANY(channels)
        channel_filters = []
        for channel in channels:
            channel_filters.append(Message.channels.contains([channel]))

        # Pending count
        pending_query = db.query(Message).filter(
            Message.status == "PENDING"
        )
        if channel_filters:
            from sqlalchemy import or_
            pending_query = pending_query.filter(or_(*channel_filters))
        pending_count = pending_query.count()

        # Processing count
        processing_query = db.query(Message).filter(
            Message.status == "PROCESSING"
        )
        if channel_filters:
            processing_query = processing_query.filter(or_(*channel_filters))
        processing_count = processing_query.count()

        # Failed in last hour
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        failed_query = db.query(Message).filter(
            Message.status == "FAILED",
            Message.updated_at >= one_hour_ago
        )
        if channel_filters:
            failed_query = failed_query.filter(or_(*channel_filters))
        failed_count = failed_query.count()

        # Delivered in last hour
        delivered_query = db.query(Message).filter(
            Message.status == "DELIVERED",
            Message.delivered_at >= one_hour_ago
        )
        if channel_filters:
            delivered_query = delivered_query.filter(or_(*channel_filters))
        delivered_count = delivered_query.count()

        return {
            "pending": pending_count,
            "processing": processing_count,
            "failed_last_hour": failed_count,
            "delivered_last_hour": delivered_count
        }

    except Exception as e:
        _logger.exception("Error getting queue stats:")
        return {
            "pending": -1,
            "processing": -1,
            "failed_last_hour": -1,
            "delivered_last_hour": -1
        }


# Global variable to track service start time
_service_start_time = datetime.utcnow()


def _get_service_uptime() -> int:
    """Get service uptime in seconds."""
    return int((datetime.utcnow() - _service_start_time).total_seconds())
```

### Integration into Notification Service

**File: `src/notification/notification_db_centric_bot.py`**

Add to the FastAPI app initialization:

```python
from src.notification.service.health_endpoints import router as health_router

# In the FastAPI app setup:
app.include_router(health_router)
```

---

## Implementation for Telegram Bot

### File: `src/telegram/services/health_handler.py` (NEW)

```python
"""
Health check handler for telegram bot.
Shows Telegram channel ownership and queue status.
"""

from typing import Dict, Any
from datetime import datetime, timedelta

from src.notification.service.message_queue_client import get_message_queue_client
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class TelegramHealthHandler:
    """
    Provides health check information for Telegram bot.
    """

    def __init__(self, queue_processor):
        """
        Initialize health handler.

        Args:
            queue_processor: TelegramQueueProcessor instance
        """
        self.queue_processor = queue_processor
        self.message_queue_client = get_message_queue_client()
        self.start_time = datetime.utcnow()

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive health status for telegram bot.

        Returns:
            Dictionary with health metrics
        """
        try:
            # Get queue stats from queue processor
            processor_stats = self.queue_processor.get_stats()

            # Get pending telegram messages from database
            pending_count = self.message_queue_client.get_pending_count_for_channels(
                ["telegram"]
            )

            # Calculate uptime
            uptime_seconds = int((datetime.utcnow() - self.start_time).total_seconds())

            return {
                "service": "telegram_bot",
                "status": "healthy" if processor_stats["running"] else "stopped",
                "timestamp": datetime.utcnow().isoformat(),
                "channels": {
                    "owned": ["telegram"],
                    "email_owned_by": "notification_service",  # Explicitly state email is NOT owned
                    "sms_owned_by": "notification_service",
                    "telegram_status": "enabled"
                },
                "queue": {
                    "pending": pending_count,
                    "processor_running": processor_stats["running"],
                    "poll_interval": processor_stats["poll_interval"]
                },
                "uptime_seconds": uptime_seconds,
                "aiogram_version": self._get_aiogram_version()
            }

        except Exception as e:
            _logger.exception("Error getting health status:")
            return {
                "service": "telegram_bot",
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    def _get_aiogram_version(self) -> str:
        """Get aiogram version."""
        try:
            import aiogram
            return aiogram.__version__
        except Exception:
            return "unknown"
```

### Integration into Telegram Bot

**File: `src/telegram/telegram_bot.py`**

Add command handler:

```python
from src.telegram.services.health_handler import TelegramHealthHandler

# After initializing queue processor:
health_handler = TelegramHealthHandler(queue_processor)

# Add command handler:
@dp.message(Command("health"))
async def health_command(message: types.Message):
    """
    Show health status of telegram bot.
    Admin command to check service health.
    """
    # Optional: Add admin-only check
    # if str(message.from_user.id) not in ADMIN_USER_IDS:
    #     await message.reply("❌ Admin command only")
    #     return

    health_status = health_handler.get_health_status()

    # Format response
    status_emoji = "✅" if health_status["status"] == "healthy" else "❌"

    response = f"""
{status_emoji} **Telegram Bot Health Status**

**Service**: {health_status['service']}
**Status**: {health_status['status']}
**Uptime**: {health_status['uptime_seconds']}s

**Owned Channels**: {', '.join(health_status['channels']['owned'])}

**Queue Status**:
• Pending: {health_status['queue']['pending']}
• Processor: {"Running" if health_status['queue']['processor_running'] else "Stopped"}
• Poll interval: {health_status['queue']['poll_interval']}s

**Note**: Email/SMS handled by Notification Service
    """.strip()

    await message.reply(response, parse_mode="Markdown")
```

---

## Alternative: Web Endpoint for Telegram Bot

If you want an HTTP endpoint (not just Telegram command), you can add a simple Flask/FastAPI server:

### File: `src/telegram/services/health_server.py` (NEW)

```python
"""
Lightweight HTTP server for telegram bot health checks.
Runs alongside the aiogram bot.
"""

from flask import Flask, jsonify
import threading
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

app = Flask(__name__)


class HealthServer:
    """
    Simple HTTP server for health checks.
    Runs on separate thread alongside telegram bot.
    """

    def __init__(self, health_handler, port=8081):
        """
        Initialize health server.

        Args:
            health_handler: TelegramHealthHandler instance
            port: Port to listen on (default: 8081)
        """
        self.health_handler = health_handler
        self.port = port
        self._setup_routes()

    def _setup_routes(self):
        """Setup Flask routes."""

        @app.route("/health/channels", methods=["GET"])
        def get_channels():
            """Get channel health status."""
            return jsonify(self.health_handler.get_health_status())

        @app.route("/health", methods=["GET"])
        def get_health():
            """Simple health check."""
            return jsonify({"status": "ok"})

    def start(self):
        """Start the health check server in a background thread."""
        def run_server():
            _logger.info("Starting health check server on port %s", self.port)
            app.run(host="0.0.0.0", port=self.port, debug=False)

        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        _logger.info("Health check server started")
```

### Integration

```python
# In telegram_bot.py startup:
from src.telegram.services.health_server import HealthServer

health_handler = TelegramHealthHandler(queue_processor)
health_server = HealthServer(health_handler, port=8081)
health_server.start()
```

---

## Example Response

### Notification Service (`GET /health/channels`)

```json
{
  "service": "notification_service",
  "status": "healthy",
  "timestamp": "2025-11-20T21:00:00",
  "channels": {
    "owned": ["email"],
    "telegram_owned_by": "telegram_bot",
    "email_status": "enabled",
    "sms_status": "disabled"
  },
  "queue": {
    "pending": 3,
    "processing": 1,
    "failed_last_hour": 0,
    "delivered_last_hour": 45
  },
  "uptime_seconds": 86400,
  "config": {
    "poll_interval": 5,
    "batch_size": 10
  }
}
```

### Telegram Bot (`GET http://localhost:8081/health/channels`)

```json
{
  "service": "telegram_bot",
  "status": "healthy",
  "timestamp": "2025-11-20T21:00:00",
  "channels": {
    "owned": ["telegram"],
    "email_owned_by": "notification_service",
    "sms_owned_by": "notification_service",
    "telegram_status": "enabled"
  },
  "queue": {
    "pending": 2,
    "processor_running": true,
    "poll_interval": 5
  },
  "uptime_seconds": 172800,
  "aiogram_version": "3.1.1"
}
```

---

## Dashboard Query

### SQL Query for Operational Dashboard

```sql
-- Channel distribution and performance metrics (last 24 hours)
SELECT
    channels,
    status,
    COUNT(*) as message_count,
    AVG(EXTRACT(EPOCH FROM (delivered_at - created_at))) as avg_delivery_time_sec,
    MIN(created_at) as first_message,
    MAX(created_at) as last_message
FROM messages
WHERE created_at > NOW() - INTERVAL '24 hours'
GROUP BY channels, status
ORDER BY channels, status;

-- Results will look like:
-- channels        | status     | message_count | avg_delivery_time_sec | first_message | last_message
-- {telegram}      | DELIVERED  | 245           | 6.2                   | ...           | ...
-- {telegram}      | FAILED     | 3             | NULL                  | ...           | ...
-- {email}         | DELIVERED  | 18            | 12.5                  | ...           | ...
-- {email}         | PENDING    | 2             | NULL                  | ...           | ...
-- {telegram,email}| DELIVERED  | 5             | 8.1                   | ...           | ...
```

---

## Summary

### What We're Adding

1. **Notification Service**:
   - New file: `src/notification/service/health_endpoints.py`
   - Endpoint: `GET /health/channels`
   - Shows: Email/SMS ownership and queue stats

2. **Telegram Bot** (Two Options):

   **Option A - Telegram Command**:
   - New file: `src/telegram/services/health_handler.py`
   - Command: `/health` (via Telegram)
   - Shows: Telegram ownership and queue stats

   **Option B - HTTP Endpoint** (Recommended):
   - New files: `src/telegram/services/health_handler.py` + `health_server.py`
   - Endpoint: `GET http://localhost:8081/health/channels`
   - Shows: Same data, accessible via HTTP

### Benefits

- ✅ **Clear Visibility**: See exactly which service owns which channels
- ✅ **Queue Monitoring**: Track pending/processing/failed messages per service
- ✅ **Uptime Tracking**: Monitor how long each service has been running
- ✅ **Operational Alerts**: Can alert if queues get too large or services go down
- ✅ **Debugging**: Quickly identify which service is having issues

### Next Steps

1. Implement notification service endpoint (already has FastAPI)
2. Choose Telegram bot approach (command vs HTTP endpoint)
3. Add monitoring/alerting based on these endpoints
4. Create Grafana/dashboard to visualize the metrics

---

**Recommendation**: Use **Option B (HTTP endpoint)** for Telegram Bot so you can easily integrate with monitoring tools like Prometheus, Grafana, or custom dashboards.
