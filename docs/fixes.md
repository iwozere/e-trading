# Pending Fixes — `src/api/main.py`

Three categories of pre-existing Pyright warnings in `src/api/main.py`.
None of them cause runtime crashes today, but they hide real bugs and make
the type checker useless as a safety net for future edits.

---

## Fix 1 — `StrategyManager` used as a type after being set to `None`

**Pyright rule**: `reportInvalidTypeForm` — Line 82

### Root cause

The import guard pattern at lines 58–65 re-assigns the class name on failure:

```python
try:
    from src.trading.strategy_manager import StrategyManager
    TRADING_SYSTEM_AVAILABLE = True
except ImportError as e:
    StrategyManager = None          # ← class name now holds None
    TRADING_SYSTEM_AVAILABLE = False

strategy_manager: Optional[StrategyManager] = None   # ← Pyright sees Optional[None]
```

When `StrategyManager = None`, using it in `Optional[StrategyManager]` is an invalid
type expression.

### Fix

Keep the runtime guard but don't clobber the class name. Use `TYPE_CHECKING`
so the type-checker always sees the real import:

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.trading.strategy_manager import StrategyManager

try:
    from src.trading.strategy_manager import StrategyManager  # noqa: F811
    TRADING_SYSTEM_AVAILABLE = True
except ImportError as e:
    _logger.warning("Trading system not available: %s", e)
    TRADING_SYSTEM_AVAILABLE = False

strategy_manager: Optional["StrategyManager"] = None
```

---

## Fix 2 — Strategy and monitoring service globals used without `None` checks

**Pyright rule**: `reportOptionalMemberAccess` — Lines 368, 385, 404, 428, 448,
472, 498, 520, 548, 551, 584, 598, 613, 627, 640, 653 (and more)

### Root cause

The globals are typed as `Optional` because they start as `None` before `lifespan()`
runs. The `lifespan()` function always initialises them, but Pyright cannot prove
that — it sees every endpoint as potentially calling the service before startup:

```python
strategy_service: Optional[StrategyManagementService] = None
monitoring_service: Optional[SystemMonitoringService] = None

@app.get("/api/strategies")
async def list_strategies(...):
    strategies = strategy_service.get_all_strategies_status()  # ← could be None
```

### Fix

Replace direct global access with FastAPI dependency functions that narrow
the type to non-optional and return HTTP 503 when called before startup:

```python
def _require_strategy_service() -> StrategyManagementService:
    if strategy_service is None:
        raise HTTPException(status_code=503, detail="Strategy service not initialised")
    return strategy_service


def _require_monitoring_service() -> SystemMonitoringService:
    if monitoring_service is None:
        raise HTTPException(status_code=503, detail="Monitoring service not initialised")
    return monitoring_service
```

Then update every affected endpoint to inject the service via `Depends`:

```python
# Before
@app.get("/api/strategies")
async def list_strategies(current_user: User = Depends(get_current_user)):
    strategies = strategy_service.get_all_strategies_status()

# After
@app.get("/api/strategies")
async def list_strategies(
    current_user: User = Depends(get_current_user),
    svc: StrategyManagementService = Depends(_require_strategy_service),
):
    strategies = svc.get_all_strategies_status()
```

Affected endpoints (replace `strategy_service.` → `svc.`):
- `list_strategies` (line 368)
- `create_strategy` (line 385)
- `get_strategy` (line 404)
- `update_strategy` (line 428)
- `delete_strategy` (line 448)
- `start_strategy` (line 472)
- `stop_strategy` (line 498)
- `restart_strategy` (line 520)
- `update_strategy_parameters` (line 584)
- `validate_strategy_config` (line 613)
- `get_strategy_templates` (line 598)

Affected endpoints (replace `monitoring_service.` → `mon.`):
- `get_system_status` (line 548)
- `get_system_metrics` (line 551)
- `get_system_alerts` (line 627 / 640)
- `acknowledge_system_alert` (line 653)

---

## Fix 3 — `api_heartbeat_manager` possibly unbound

**Pyright rule**: `reportPossiblyUnboundVariable` — Line 184

### Root cause

`api_heartbeat_manager` is assigned inside a `try` block. If the block raises
before reaching that line, the variable never gets bound. It is then accessed
in the shutdown section (still inside `try: ... except: pass`, so no crash,
but the except silently swallows the `NameError`):

```python
try:
    api_heartbeat_manager = HeartbeatManager(...)   # line 165 — may not be reached
    api_heartbeat_manager.start_heartbeat()
except Exception:
    _logger.exception("Failed to initialise heartbeat manager:")

# shutdown
try:
    api_heartbeat_manager.stop_heartbeat()          # line 184 — possibly unbound
except:
    pass
```

### Fix

Initialise to `None` before the `try` block and guard the shutdown call:

```python
api_heartbeat_manager = None
try:
    api_heartbeat_manager = HeartbeatManager(...)
    api_heartbeat_manager.set_health_check_function(api_service_health_check)
    api_heartbeat_manager.start_heartbeat()
    _logger.info("Heartbeat manager started for API service")
except Exception:
    _logger.exception("Failed to initialise heartbeat manager:")

# shutdown
if api_heartbeat_manager is not None:
    api_heartbeat_manager.stop_heartbeat()
    _logger.info("Stopped API service heartbeat")
```

---

## Summary

| # | Rule | Lines | Runtime risk | Effort |
|---|------|-------|--------------|--------|
| 1 | `reportInvalidTypeForm` | 82 | None today | Low — 5 lines |
| 2 | `reportOptionalMemberAccess` | 368–653 | None today (services always init'd) | Medium — ~15 endpoints |
| 3 | `reportPossiblyUnboundVariable` | 184 | Low (bare `except: pass` hides it) | Low — 3 lines |
