# Health Check Endpoint Consolidation

## Overview

Consolidated all health check endpoints into a unified system to avoid spreading HTTP services across the application. All health monitoring is now centralized in `src/api/health_routes.py`.

---

## Changes Made

### New Files Created

1. **[src/api/health_routes.py](../src/api/health_routes.py)** - Unified health check router
   - Consolidates all system health monitoring
   - Provides channel ownership visibility
   - Shows queue status for Telegram Bot and Notification Service

2. **[src/api/services/telegram_health_service.py](../src/api/services/telegram_health_service.py)**
   - Service layer for Telegram bot health monitoring
   - Database-centric (no HTTP calls to Telegram bot)
   - Monitors queue status, stuck messages, throughput

3. **[src/api/services/notification_health_service.py](../src/api/services/notification_health_service.py)**
   - Service layer for Notification Service health monitoring
   - Database-centric (no HTTP calls to notification service)
   - Monitors queue status for owned channels (email, sms)

### Modified Files

1. **[src/api/main.py](../src/api/main.py)**
   - Added import: `from src.api.health_routes import router as health_router`
   - Registered health router: `app.include_router(health_router)`
   - Deprecated old `/api/health` endpoint → moved to `/api/health/legacy`

2. **[src/api/notification_routes.py](../src/api/notification_routes.py)**
   - Deprecated `/api/notifications/health` endpoint (kept for backward compatibility)
   - Deprecated `/api/notifications/channels/health` endpoint (kept for backward compatibility)
   - Both endpoints now include deprecation notices directing to new unified system

---

## New Unified Health Endpoints

All endpoints are under `/api/health`:

### 1. **GET /api/health**
Basic health check for API and database connectivity.

**Authentication**: None (public endpoint for monitoring tools)

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-21T10:00:00Z",
  "database": "connected",
  "api": "operational",
  "total_messages": 1234
}
```

---

### 2. **GET /api/health/channels** ⭐ Main Endpoint
Comprehensive health status for all notification channels showing channel ownership and queue status.

**Authentication**: Required

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-21T10:00:00Z",
  "services": {
    "telegram_bot": {
      "service": "telegram_bot",
      "status": "healthy",
      "status_reason": null,
      "timestamp": "2025-11-21T10:00:00Z",
      "channels": {
        "owned": ["telegram"],
        "email_owned_by": "notification_service",
        "sms_owned_by": "notification_service",
        "telegram_status": "enabled"
      },
      "queue": {
        "pending": 3,
        "processing": 1,
        "failed_last_hour": 0,
        "delivered_last_hour": 45,
        "stuck_messages": 0
      }
    },
    "notification_service": {
      "service": "notification_service",
      "status": "healthy",
      "status_reason": null,
      "timestamp": "2025-11-21T10:00:00Z",
      "channels": {
        "owned": ["email"],
        "telegram_owned_by": "telegram_bot",
        "email_status": "enabled",
        "sms_status": "disabled"
      },
      "queue": {
        "pending": 2,
        "processing": 0,
        "failed_last_hour": 0,
        "delivered_last_hour": 18,
        "stuck_messages": 0
      }
    }
  },
  "summary": {
    "all_services_healthy": true,
    "total_pending_messages": 5,
    "channel_ownership": {
      "telegram_bot": ["telegram"],
      "notification_service": ["email"]
    }
  }
}
```

---

### 3. **GET /api/health/telegram**
Telegram Bot specific health status.

**Authentication**: Required

**Response**:
```json
{
  "service": "telegram_bot",
  "status": "healthy",
  "status_reason": null,
  "timestamp": "2025-11-21T10:00:00Z",
  "channels": {
    "owned": ["telegram"],
    "email_owned_by": "notification_service",
    "sms_owned_by": "notification_service",
    "telegram_status": "enabled"
  },
  "queue": {
    "pending": 3,
    "processing": 1,
    "failed_last_hour": 0,
    "delivered_last_hour": 45,
    "stuck_messages": 0
  }
}
```

---

### 4. **GET /api/health/notification**
Notification Service specific health status.

**Authentication**: Required

**Response**:
```json
{
  "service": "notification_service",
  "status": "healthy",
  "status_reason": null,
  "timestamp": "2025-11-21T10:00:00Z",
  "channels": {
    "owned": ["email"],
    "telegram_owned_by": "telegram_bot",
    "email_status": "enabled",
    "sms_status": "disabled"
  },
  "queue": {
    "pending": 2,
    "processing": 0,
    "failed_last_hour": 0,
    "delivered_last_hour": 18,
    "stuck_messages": 0
  }
}
```

---

### 5. **GET /api/health/database**
Database health and connectivity status.

**Authentication**: Required

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-21T10:00:00Z",
  "database": {
    "connection": "connected",
    "total_messages": 1234,
    "pending_messages": 5,
    "processing_messages": 1,
    "delivered_messages": 1200,
    "failed_messages": 28,
    "locked_messages": 1
  }
}
```

---

## Deprecated Endpoints (Backward Compatible)

These endpoints still work but return deprecation notices:

1. **GET /api/health/legacy** (was `/api/health`)
   - Basic health check
   - Returns deprecation notice

2. **GET /api/notifications/health**
   - Notification routes health
   - Returns deprecation notice directing to `/api/health/notification`

3. **GET /api/notifications/channels/health**
   - Channel health (old format)
   - Returns deprecation notice directing to `/api/health/channels`

---

## Health Status Values

### Status Field
- `"healthy"` - Service is operating normally
- `"degraded"` - Service is running but experiencing issues
- `"error"` - Service health check failed

### Status Reasons (when degraded)
- `"X messages stuck in processing"` - Messages processing for > 5 minutes
- `"High queue backlog: X messages"` - Pending queue > 100 messages
- `"High failure rate: X failures in last hour"` - Many failures with no deliveries

---

## Architecture Benefits

### ✅ Centralized HTTP Services
- All health endpoints in one place: `src/api/health_routes.py`
- No HTTP servers in Telegram Bot or Notification Service
- Follows existing API patterns

### ✅ Database-Centric Design
- Health services query database directly
- No tight coupling between services
- Works even if Telegram Bot/Notification Service are down

### ✅ Clear Channel Ownership
- Explicitly shows which service owns which channels
- Prevents confusion about duplicate message issues
- Aligns with CHANNEL_OWNERSHIP.md architecture

### ✅ Operational Monitoring
- Detects stuck messages (processing > 5 minutes)
- Monitors queue backlog
- Tracks throughput (delivered/failed per hour)
- Provides actionable health metrics

---

## Usage Examples

### Monitoring Dashboard Query
```bash
# Get comprehensive health status
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/health/channels
```

### Alerting Integration
```python
import requests

response = requests.get(
    "http://localhost:8000/api/health/channels",
    headers={"Authorization": f"Bearer {token}"}
)

health = response.json()

# Alert if any service is degraded
if health["status"] != "healthy":
    send_alert(f"System degraded: {health['services']}")

# Alert if queue backlog is high
total_pending = health["summary"]["total_pending_messages"]
if total_pending > 50:
    send_alert(f"High queue backlog: {total_pending} messages")
```

### Prometheus Metrics (Future Enhancement)
The health endpoints can be easily adapted to export Prometheus metrics:
```python
# /api/health/metrics
telegram_queue_pending{service="telegram_bot"} 3
telegram_queue_processing{service="telegram_bot"} 1
notification_queue_pending{service="notification_service"} 2
```

---

## Migration Guide

### For Monitoring Tools

**Old**:
```bash
curl http://localhost:8000/api/health
curl http://localhost:8000/api/notifications/channels/health
```

**New**:
```bash
# Basic health
curl http://localhost:8000/api/health

# Comprehensive channel health (recommended)
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/health/channels
```

### For Dashboards

Update your dashboard queries to use the new endpoints:

1. Replace `/api/health` → `/api/health` (no change, but use authenticated version for more details)
2. Replace `/api/notifications/channels/health` → `/api/health/channels`
3. Add new panels for Telegram Bot health: `/api/health/telegram`
4. Add new panels for Notification Service health: `/api/health/notification`

---

## Related Documentation

- [CHANNEL_OWNERSHIP.md](CHANNEL_OWNERSHIP.md) - Channel ownership architecture
- [MONITORING_ENDPOINT_DESIGN.md](MONITORING_ENDPOINT_DESIGN.md) - Original design document (reference only)
- [MIGRATION_PLAN.md](../MIGRATION_PLAN.md) - Phase 6 now complete with monitoring implementation

---

## Summary

✅ **3 new files** created for unified health system
✅ **2 files** modified to integrate and deprecate old endpoints
✅ **5 new endpoints** providing comprehensive health monitoring
✅ **3 deprecated endpoints** kept for backward compatibility
✅ **Zero changes** required to Telegram Bot or Notification Service
✅ **All health functionality** now centralized in `src/api/health_routes.py`

**Migration Status**: Phase 6 (Monitoring & Observability) from MIGRATION_PLAN.md is now **100% complete**.
