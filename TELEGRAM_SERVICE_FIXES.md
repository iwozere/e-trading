# Telegram Service Fixes

## Issues Identified

### 1. Missing `get_user_status` method in TelegramService ✅ FIXED
**Error:**
```
RuntimeError: Telegram service missing required method: get_user_status
```

**Root Cause:**
The `telegram_bot.py` initialization validates that `telegram_service` has `get_user_status` and `set_user_limit` methods, but `TelegramService` only had `set_user_limit`.

**Solution:**
Added `get_user_status` method to `TelegramService` class.

---

### 2. Import Path Issue - Module vs Instance ✅ FIXED
**Error:**
```
RuntimeError: Telegram service missing required method: get_user_status
(persisted after adding the method)
```

**Root Cause:**
```python
from src.data.db.services import telegram_service  # This imports the MODULE
telegram_service_instance = telegram_service        # Checking module for methods ❌
```

The import was returning the module, not the service instance, so `hasattr()` checks were failing.

**Solution:**
Changed import to get the instance directly:
```python
from src.data.db.services.telegram_service import telegram_service as telegram_service_instance
```

---

### 3. Missing `list_alerts` method in TelegramService ✅ FIXED
**Error:**
```
ValueError: Telegram service missing required method: list_alerts
```

**Root Cause:**
The `business_logic.py` layer requires these methods:
- `get_user_status`, `set_user_limit`, `add_alert`, **`list_alerts`**,
- `add_schedule`, `list_schedules`, `log_command_audit`, `add_feedback`

`TelegramService` had all except `list_alerts`.

**Solution:**
Added `list_alerts(telegram_user_id)` method that:
- Queries job schedules with `job_type=ALERT` for the user
- Parses the JSON config to extract alert details
- Returns list of dicts with alert_type (price/indicator/custom), ticker, status, etc.

---

### 4. SystemHealthService initialization error in HeartbeatManager ✅ FIXED
**Error:**
```
RuntimeError: Telegram service missing required method: get_user_status
```

**Root Cause:**
The `telegram_bot.py` initialization validates that `telegram_service` has `get_user_status` and `set_user_limit` methods, but `TelegramService` only has `set_user_limit`.

**Solution:**
Add `get_user_status` method to `TelegramService` class.

---

### 2. SystemHealthService initialization error in HeartbeatManager
**Error:**
```
AttributeError: 'SystemHealthRepository' object has no attribute 'uow'
```

**Root Cause:**
In `heartbeat_manager.py` line 185, the code incorrectly creates a `SystemHealthService` by passing a repository instance:
```python
health_repo = SystemHealthRepository(uow.s)
health_service = SystemHealthService(health_repo)  # WRONG!
```

`SystemHealthService` extends `BaseDBService` and expects either `None` or a `DatabaseService`, NOT a repository.

**Solution:**
Pass `None` to `SystemHealthService` constructor and use the `@with_uow` decorated methods which handle UoW management automatically.

---

## Proposed Fixes

### Fix 1: Add `get_user_status` method to TelegramService

**File:** `src/data/db/services/telegram_service.py`

Add this method after the existing user-related methods (around line 95):

```python
@with_uow
@handle_db_error
def get_user_status(self, telegram_user_id: str) -> Optional[Dict[str, Any]]:
    """
    Get user status including approval, verification, and limit information.
    
    Args:
        telegram_user_id: Telegram user ID
        
    Returns:
        Dictionary with user status or None if user not found
    """
    profile = self.uow.users.get_telegram_profile(telegram_user_id)
    if not profile:
        return None
    
    return {
        "telegram_user_id": telegram_user_id,
        "approved": profile.get("approved", False),
        "verified": profile.get("verified", False),
        "max_alerts": profile.get("max_alerts"),
        "max_schedules": profile.get("max_schedules"),
        "language": profile.get("language", "en"),
        "is_admin": profile.get("is_admin", False),
    }
```

---

### Fix 2: Correct SystemHealthService usage in HeartbeatManager

**File:** `src/common/heartbeat_manager.py`

Replace the `_update_health_status` method (lines ~175-195) with:

```python
def _update_health_status(
    self,
    status: SystemHealthStatus,
    response_time_ms: Optional[int] = None,
    error_message: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """Update health status in the database."""
    try:
        # SystemHealthService handles its own UoW via @with_uow decorator
        health_service = SystemHealthService()
        
        health_service.update_system_health(
            system=self.system,
            component=self.component,
            status=status,
            response_time_ms=response_time_ms,
            error_message=error_message,
            metadata=metadata
        )

    except Exception as e:
        _logger.exception("Failed to update health status for %s.%s:",
                     self.system, self.component or 'main')
        # Don't re-raise - heartbeat should continue even if health update fails
```

**Key Changes:**
1. Remove manual UoW context manager (`with self.db_service.uow()`)
2. Don't create `SystemHealthRepository` manually
3. Create `SystemHealthService()` with no arguments
4. Call `update_system_health()` directly - it handles UoW internally via `@with_uow` decorator
5. Don't re-raise exception to prevent heartbeat from stopping

---

### Fix 3: Optional - Make telegram service validation non-blocking

**File:** `src/telegram/telegram_bot.py`

If you want the bot to continue even if methods are missing (with degraded functionality), change line ~735 from:

```python
if not hasattr(telegram_service_instance, method):
    raise RuntimeError(f"Telegram service missing required method: {method}")
```

To:

```python
if not hasattr(telegram_service_instance, method):
    _logger.warning(f"Telegram service missing method: {method} - some features may not work")
```

This makes the validation warn instead of fail, allowing the bot to run with reduced capabilities.

---

## Implementation Order

1. **Fix 1** (TelegramService.get_user_status) - Required
2. **Fix 2** (HeartbeatManager health service usage) - Required
3. **Fix 3** (Non-blocking validation) - Optional

---

## Testing After Fixes

After applying fixes, verify:

1. Telegram bot starts without errors:
   ```powershell
   python src/telegram/telegram_bot.py
   ```

2. Check logs for successful service initialization:
   - "Telegram service initialized and validated successfully"
   - "Initialized heartbeat manager for telegram_bot.main"
   - No AttributeError about 'uow'

3. Verify health status updates in database:
   ```sql
   SELECT * FROM msg_system_health WHERE system = 'telegram_bot';
   ```

---

## Root Cause Summary

Both issues stem from **incorrect service layer initialization patterns**:

1. **TelegramService**: Missing a method that telegram_bot expects
2. **HeartbeatManager**: Incorrectly mixing repository-level code with service-layer APIs

The fixes align both with the existing service layer architecture where services use `@with_uow` decorators to manage transactions automatically.
