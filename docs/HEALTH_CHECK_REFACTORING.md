# Health Check Refactoring - Database Layer Separation

## Overview

Refactored health check system to follow proper database service/repository architecture. All database operations now go through the established service layer pattern, ensuring proper separation of concerns.

---

## Architecture Layers

### Layer 1: Database Repository (`src/data/db/repos/`)
**Responsibility**: Direct database queries using SQLAlchemy ORM

**File**: [src/data/db/repos/repo_notification.py](../src/data/db/repos/repo_notification.py)

**New Method Added**:
```python
def get_queue_health_for_channels(self, channels: List[str]) -> Dict[str, Any]:
    """
    Get queue health metrics for specific channels.

    Queries database for:
    - pending: Count of pending messages
    - processing: Count of messages currently processing
    - failed_last_hour: Count of failures in last hour
    - delivered_last_hour: Count of deliveries in last hour
    - stuck_messages: Count of messages stuck in processing (> 5 minutes)
    """
```

---

### Layer 2: Database Service (`src/data/db/services/`)
**Responsibility**: Business logic, transaction management (UoW pattern), error handling

**File**: [src/data/db/services/health_monitoring_service.py](../src/data/db/services/health_monitoring_service.py) (NEW)

**Methods**:
- `get_telegram_health()` - Telegram Bot health status
- `get_notification_service_health(enabled_channels)` - Notification Service health status
- `get_comprehensive_health()` - Combined health for all services
- `get_database_health()` - Database connectivity and message statistics
- `_assess_health_status(queue_metrics)` - Health assessment logic

**Pattern**: Extends `BaseDBService`, uses `@with_uow` decorator for transaction management

---

### Layer 3: API Service (`src/api/services/`)
**Responsibility**: API-specific logic, delegates to database services

**Files**:
- [src/api/services/telegram_health_service.py](../src/api/services/telegram_health_service.py) - Updated
- [src/api/services/notification_health_service.py](../src/api/services/notification_health_service.py) - Updated

**Changes**:
- Removed all direct SQLAlchemy queries
- Now delegates to `HealthMonitoringService`
- Handles error responses if database layer fails
- Maintains API contract (same response format)

---

### Layer 4: API Routes (`src/api/`)
**Responsibility**: HTTP endpoints, authentication, request/response handling

**File**: [src/api/health_routes.py](../src/api/health_routes.py)

**No Changes Required**: Routes already use API services, which now properly delegate to database layer

---

## Data Flow

### Before Refactoring ❌
```
HTTP Request → API Route → API Service → Direct SQLAlchemy Query
                                              ↓
                                          Database
```

**Problem**: API layer directly querying database, bypassing service/repository pattern

---

### After Refactoring ✅
```
HTTP Request → API Route → API Service → DB Service → Repository → Database
                                           ↓              ↓
                                      (Business      (SQLAlchemy
                                       Logic,         Queries)
                                        UoW)
```

**Benefits**:
- ✅ Proper separation of concerns
- ✅ Transaction management via UoW pattern
- ✅ Consistent error handling via `@handle_db_error`
- ✅ Reusable database logic (can be called from CLI, workers, etc.)
- ✅ Testable at each layer

---

## Files Modified

### 1. **src/data/db/repos/repo_notification.py**
**Added**: `get_queue_health_for_channels()` method to `MessageRepository` class

**Location**: Lines 245-318

**Purpose**: Encapsulates all queue health queries in one reusable method

**Usage**:
```python
# From service layer:
queue_metrics = self.uow.notifications.messages.get_queue_health_for_channels(["telegram"])
```

---

### 2. **src/data/db/services/health_monitoring_service.py** (NEW FILE)
**Created**: New service for health monitoring operations

**Size**: 252 lines

**Extends**: `BaseDBService` (proper service pattern)

**Key Features**:
- Uses `@with_uow` decorator for transaction management
- Uses `@handle_db_error` decorator for error handling
- Delegates to repositories for all database access
- Contains health assessment business logic

---

### 3. **src/api/services/telegram_health_service.py**
**Before**: 150+ lines with direct SQLAlchemy queries

**After**: 76 lines delegating to `HealthMonitoringService`

**Reduction**: ~50% less code, zero database coupling

---

### 4. **src/api/services/notification_health_service.py**
**Before**: 200+ lines with direct SQLAlchemy queries and helper methods

**After**: 81 lines delegating to `HealthMonitoringService`

**Reduction**: ~60% less code, zero database coupling

---

### 5. **src/api/health_routes.py**
**Modified**: `get_database_health()` endpoint

**Before**: Direct database queries in route

**After**: Delegates to `HealthMonitoringService.get_database_health()`

---

## Benefits of Refactoring

### 1. **Proper Layering** ✅
```
API Layer (src/api/)
  ↓ delegates to
Database Service Layer (src/data/db/services/)
  ↓ uses
Repository Layer (src/data/db/repos/)
  ↓ queries
Database
```

### 2. **Transaction Management** ✅
All database operations now use Unit of Work pattern via `@with_uow` decorator:
```python
@with_uow
def get_telegram_health(self) -> Dict[str, Any]:
    # self.uow is automatically managed
    queue_metrics = self.uow.notifications.messages.get_queue_health_for_channels(["telegram"])
    # Transaction auto-commits on success, rolls back on error
```

### 3. **Error Handling** ✅
Consistent error handling via `@handle_db_error` decorator:
```python
@with_uow
@handle_db_error
def get_telegram_health(self) -> Dict[str, Any]:
    # Database errors automatically caught, logged, and wrapped
```

### 4. **Reusability** ✅
Health monitoring logic can now be used from:
- API endpoints (current use case)
- CLI commands
- Background workers
- Cron jobs
- Monitoring scripts

Example:
```python
# Can be called from anywhere
from src.data.db.services.health_monitoring_service import HealthMonitoringService

health_service = HealthMonitoringService()
telegram_health = health_service.get_telegram_health()
```

### 5. **Testability** ✅
Each layer can be tested independently:
- Repository tests: Mock database session
- Service tests: Mock repository methods
- API service tests: Mock database service
- Route tests: Mock API services

---

## Code Comparison

### Before: Direct Database Queries in API Service ❌
```python
# src/api/services/telegram_health_service.py (OLD)
def get_health_status(self) -> Dict[str, Any]:
    try:
        db_service = get_database_service()

        with db_service.uow() as uow:
            # Direct SQLAlchemy queries
            pending_count = uow.s.query(Message).filter(
                Message.channels.contains(["telegram"]),
                Message.status == MessageStatus.PENDING.value
            ).count()

            processing_count = uow.s.query(Message).filter(
                Message.channels.contains(["telegram"]),
                Message.status == MessageStatus.PROCESSING.value
            ).count()

            # ... many more queries ...
```

### After: Proper Service Delegation ✅
```python
# src/api/services/telegram_health_service.py (NEW)
def get_health_status(self) -> Dict[str, Any]:
    try:
        return self.health_monitoring_service.get_telegram_health()
    except Exception:
        _logger.exception("Error getting Telegram health status:")
        # Return error response
```

---

## Repository Method Implementation

### Location: src/data/db/repos/repo_notification.py

```python
def get_queue_health_for_channels(self, channels: List[str]) -> Dict[str, Any]:
    """
    Get queue health metrics for specific channels.

    Uses optimized SQLAlchemy queries with:
    - func.count() for efficient counting
    - Array contains operator for channel filtering
    - Time-based filtering for hourly metrics
    """
    from sqlalchemy import func

    current_time = datetime.now(timezone.utc)
    one_hour_ago = current_time - timedelta(hours=1)
    five_min_ago = current_time - timedelta(minutes=5)

    # Build channel filters
    channel_filters = [Message.channels.contains([ch]) for ch in channels]

    # Pending count (using efficient scalar query)
    pending_count = self.session.query(func.count(Message.id)).filter(
        Message.status == MessageStatus.PENDING.value,
        or_(*channel_filters)
    ).scalar() or 0

    # ... similar for other metrics ...
```

**Optimizations**:
- Uses `func.count()` instead of `.count()` for better performance
- Uses `.scalar()` to return single value directly
- Single query per metric (no N+1 queries)
- Proper null handling with `or 0`

---

## Service Layer Implementation

### Location: src/data/db/services/health_monitoring_service.py

```python
class HealthMonitoringService(BaseDBService):
    """Service for monitoring notification system health."""

    @with_uow
    @handle_db_error
    def get_telegram_health(self) -> Dict[str, Any]:
        """
        Get Telegram Bot health status.

        Delegates to repository for queue metrics,
        then applies business logic for health assessment.
        """
        channels = ["telegram"]
        queue_metrics = self.uow.notifications.messages.get_queue_health_for_channels(channels)

        # Business logic: assess health based on metrics
        status, status_reason = self._assess_health_status(queue_metrics)

        # Return formatted response
        return {
            "service": "telegram_bot",
            "status": status,
            "status_reason": status_reason,
            # ... full response structure ...
        }
```

**Pattern**:
- `@with_uow` - Manages database transaction
- `@handle_db_error` - Catches and logs database errors
- `self.uow.notifications.messages` - Accesses repository via UoW bundle
- `_assess_health_status()` - Business logic separated from data access

---

## Health Assessment Logic

### Location: src/data/db/services/health_monitoring_service.py

```python
def _assess_health_status(self, queue_metrics: Dict[str, Any]) -> tuple[str, str]:
    """
    Assess health status based on queue metrics.

    Priority order:
    1. Stuck messages (highest priority)
    2. High queue backlog
    3. High failure rate with no deliveries
    """
    stuck_messages = queue_metrics.get("stuck_messages", 0)
    pending = queue_metrics.get("pending", 0)
    failed_last_hour = queue_metrics.get("failed_last_hour", 0)
    delivered_last_hour = queue_metrics.get("delivered_last_hour", 0)

    # Check for stuck messages (highest priority)
    if stuck_messages > 0:
        return "degraded", f"{stuck_messages} messages stuck in processing"

    # Check for high queue backlog
    if pending > 100:
        return "degraded", f"High queue backlog: {pending} messages"

    # Check for high failure rate with no deliveries
    if failed_last_hour > 10 and delivered_last_hour == 0:
        return "degraded", f"High failure rate: {failed_last_hour} failures in last hour"

    # All checks passed
    return "healthy", None
```

**Business Rules**:
- Stuck messages = processing for > 5 minutes
- High backlog = > 100 pending messages
- High failure rate = > 10 failures in last hour with 0 deliveries

---

## Testing Strategy

### Unit Tests for Repository
```python
def test_get_queue_health_for_channels(mock_session):
    repo = MessageRepository(mock_session)

    # Mock query results
    mock_session.query.return_value.filter.return_value.scalar.return_value = 5

    health = repo.get_queue_health_for_channels(["telegram"])

    assert health["pending"] == 5
    assert "stuck_messages" in health
```

### Unit Tests for Service
```python
def test_get_telegram_health(mock_uow):
    service = HealthMonitoringService()
    service.uow = mock_uow

    # Mock repository response
    mock_uow.notifications.messages.get_queue_health_for_channels.return_value = {
        "pending": 5,
        "processing": 1,
        "failed_last_hour": 0,
        "delivered_last_hour": 10,
        "stuck_messages": 0
    }

    health = service.get_telegram_health()

    assert health["service"] == "telegram_bot"
    assert health["status"] == "healthy"
```

### Integration Tests for API
```python
def test_health_endpoint(client):
    response = client.get("/api/health/telegram", headers=auth_headers)

    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "telegram_bot"
    assert "queue" in data
```

---

## Summary

✅ **1 repository method added** - `get_queue_health_for_channels()`
✅ **1 new service file created** - `health_monitoring_service.py`
✅ **2 API services refactored** - Removed all direct database queries
✅ **1 API route updated** - Delegate database health to service layer
✅ **Zero breaking changes** - All endpoints maintain same response format
✅ **Proper architecture** - Service → Repository → Database

**Code Reduction**:
- API services: ~55% less code
- Database coupling: 0% (was 100%)
- Reusability: Can now be used from anywhere in the application

**Architecture Compliance**:
- ✅ Follows existing `BaseDBService` pattern
- ✅ Uses `@with_uow` for transaction management
- ✅ Uses `@handle_db_error` for error handling
- ✅ Accesses database only through repositories
- ✅ Business logic in service layer
- ✅ Data access in repository layer
