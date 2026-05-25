# Architectural Review: `src/trading/` and `src/common/`

**Date:** 2026-05-25  
**Scope:** `src/trading/` and `src/common/` submodules  
**Focus:** Design, architecture, performance, security, bad patterns  
**Reviewer:** Claude Sonnet 4.6

---

## Executive Summary

Both modules show solid foundational design—a well-layered broker abstraction, a clean strategy-instance lifecycle model, a composable risk pipeline, a circuit-breaker-equipped sentiment adapter manager, and a comprehensive health-monitoring skeleton. However, several **critical correctness bugs** and **dangerous design patterns** must be addressed before this system can safely run live or paper trades at scale. The most urgent issues are a hardcoded market-price constant that invalidates all paper-trading simulations, a double-execution loop race condition, deprecated asyncio usage, and a paper-trading PnL calculation bug that always returns $0.

---

## Priority Levels

| Level | Label | Meaning |
|-------|-------|---------|
| P1 | 🔴 Critical | Correctness bug or data-loss risk — fix before any live use |
| P2 | 🟠 High | Reliability / security flaw — fix in next sprint |
| P3 | 🟡 Medium | Design smell / maintainability debt — plan to fix |
| P4 | 🟢 Low | Convention / style / minor enhancement |

---

## `src/trading/` Issues

---

### ✅ RESOLVED — P1-T1 — `_get_simulated_market_price()` uses hardcoded $100 base price

**File:** `src/trading/broker/base_broker.py:1539–1547`

```python
def _get_simulated_market_price(self, symbol: str) -> float:
    base_price = 100.0          # ← ALWAYS $100
    volatility = 0.02
    price_change = random.gauss(0, volatility)
    return base_price * (1 + price_change)
```

All Backtrader paper-trading order simulation (`_simulate_backtrader_order_execution`) routes through this method. For BTCUSDT at $60,000, every simulated execution uses ~$100, making portfolio P&L, slippage calculations, and commission amounts completely wrong. This single bug invalidates all paper-trading backtests that run through the Backtrader bridge.

**Fix:** Replace with the actual current-bar close price from the Backtrader data feed. The `next()` method already has access to the data feed via `backtrader_owner` stored in the order's metadata; extract `data.close[0]` and pass it through, or store a reference to the active data feed in the broker during `cerebro.run()`.

---

### ✅ RESOLVED — P1-T2 — Double execution loop: Backtrader + BaseTradingBot running simultaneously

**File:** `src/trading/strategy_instance.py:117–118`

```python
asyncio.create_task(self._run_backtrader_async())   # cerebro.run() on BT thread
asyncio.create_task(self._start_trading_bot_loop()) # BaseTradingBot.run() on executor
```

Both tasks start for the same instance. `cerebro.run()` drives the Backtrader strategy which fires signals via `on_signal_callback=self.trading_bot.add_signal`. Simultaneously, `BaseTradingBot.run()` calls `process_signals()` and `execute_trade()`. This creates:

- **Race condition** on `active_positions` and `trade_history` between the two loops
- **Double execution** of buy/sell signals — the same signal queued in Backtrader can also be dequeued and executed by the `BaseTradingBot` loop
- **Two independent heartbeat/DB update streams** for the same bot ID

**Fix:** Use Backtrader as the sole signal source when `cerebro` is active. The `BaseTradingBot.run()` loop should only run when Backtrader is *not* managing the strategy (i.e., for pure live-data bots without Cerebro). Gate `_start_trading_bot_loop` on `self.cerebro is None`.

---

### ✅ RESOLVED — P1-T3 — Deprecated `asyncio.get_event_loop()` breaks on Python 3.12+

**File:** `src/trading/live_trading_bot.py:64–70, 82–85, 97–100`

```python
loop = asyncio.get_event_loop()   # Deprecated since 3.10, raises RuntimeError in 3.12+
if loop.is_running():
    asyncio.ensure_future(self.manager.start_instance(self.instance_id))
else:
    loop.run_until_complete(self.manager.start_instance(self.instance_id))
```

`asyncio.get_event_loop()` without a running loop raises `DeprecationWarning` in 3.10 and `RuntimeError` in 3.12+. The code also conflates the CLI entry point with the web-UI entry point by checking `sys.argv[0]`.

**Fix:** Replace with `asyncio.run(self.manager.start_instance(...))` for CLI use-cases, and expose a proper `async def start(self)` for callers already inside an event loop.

---

### ✅ RESOLVED — P1-T4 — Paper-trading PnL bug: realized PnL is always $0 for closed positions

**File:** `src/trading/broker/base_broker.py:1582–1587`

```python
position.quantity -= executed_qty
if position.quantity <= 0:
    realized_pnl = (executed_price - position.average_price) * abs(position.quantity)
    # Bug: position.quantity is exactly 0 here → pnl is always 0
```

After subtracting `executed_qty`, `position.quantity` becomes `<= 0`. The PnL formula multiplies by `abs(position.quantity)`, which is 0 when fully closed — so every full position close records $0 realized PnL in the paper portfolio.

**Fix:** Capture the quantity before subtraction:
```python
qty_closed = min(executed_qty, position.quantity)
position.quantity -= qty_closed
realized_pnl = (executed_price - position.average_price) * qty_closed
```

---

### 🟠 P2-T1 — Triple notification duplication on every trade

**File:** `src/trading/base_trading_bot.py:474–496, 560–608`

On each BUY or SELL, all three of the following fire:
1. `notify_trade_event()` → `_schedule_notification_to_owner()` (if `trade_notification_hook` is None)
2. `position_notification_manager.notify_position_opened/closed()` (always fires if manager exists)
3. `trade_notification_hook()` (if set, e.g. `StrategyInstance._schedule_user_trade_notification`)

The intent is that `notify_trade_event` is suppressed when the hook is set, but `PositionNotificationManager` always fires regardless. The result: users who configure both a hook and a `PositionNotificationManager` receive duplicate Telegram/email messages for every trade.

**Fix:** Define a single notification path per event. Either remove `PositionNotificationManager` from `BaseTradingBot` (keep it in `BaseBroker` only) or remove `notify_trade_event` and route everything through the hook pattern.

---

### 🟠 P2-T2 — `InstanceService` singleton silently ignores dependency injection

**File:** `src/trading/instance_service.py:27–43`

```python
class InstanceService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, notification_client=None, trade_repository=None):
        if hasattr(self, '_initialized') and self._initialized:
            return   # ← silently ignores new dependencies
```

`StrategyManager` creates `InstanceService(notification_client=..., trade_repository=...)`. If a second `StrategyManager` is created with different dependencies (e.g., in tests, or after hot-reload), `InstanceService` retains the original client and repository. All `StrategyInstance` objects created afterward will use the stale clients.

**Fix:** Remove the singleton pattern from `InstanceService`. A single `StrategyManager` should own a single `InstanceService` as a regular instance attribute.

---

### 🟠 P2-T3 — Commission hardcoded at 0.1% of gross PnL — wrong formula and non-configurable

**File:** `src/trading/base_trading_bot.py:505–506`

```python
commission = gross_pnl * 0.001  # 0.1% commission
```

Two problems:
1. Commission should be a percentage of **trade notional** (`price × size`), not of P&L. When a trade loses money (`gross_pnl < 0`), this formula produces a negative commission (i.e., a commission *rebate*).
2. The value 0.001 is hardcoded instead of coming from the broker config (`paper_trading_config.commission_rate`).

**Fix:**
```python
commission = price * position["size"] * self.paper_trading_config.commission_rate
```
And expose `paper_trading_config` on `BaseTradingBot` from the broker or config.

---

### 🟠 P2-T4 — `execution_persistence._append_to_json_list()` is not atomic

**File:** `src/trading/execution_persistence.py:59–88`

The read → modify → write pattern without an atomic rename means a crash during the write leaves a partially-written JSON file. On restart, the corrupted file causes all trade/order history to be silently reset to `[]`.

**Fix:** Write to a `.tmp` file, then atomically rename:
```python
tmp_path = file_path.with_suffix('.tmp')
with open(tmp_path, "w", encoding="utf-8") as f:
    json.dump(all_items, f, default=str, indent=2)
tmp_path.replace(file_path)  # atomic on POSIX; near-atomic on Windows
```

---

### 🟠 P2-T5 — `MetricsRegistry._save_metrics()` called on every trade — O(n bots) disk write

**File:** `src/trading/metrics_tracker.py:133–140`

Every call to `record_trade()` serializes **all** bot metrics to disk. For 20 active bots each making 100 trades/day, this produces 2,000 full-file rewrites per day. At scale this will saturate I/O.

**Fix:** Use a dirty-flag + periodic flush (e.g., flush every N seconds from a background thread), or move metrics to the database which already tracks them per-trade.

---

### 🟠 P2-T6 — `RiskController.real_time_adjustments()` uses config equity, not actual balance

**File:** `src/trading/risk/controller.py:100–108`

```python
volatility_size = volatility_scaling.volatility_scaled_position(
    account_equity=self.config.get('account_equity', 100000),  # ← stale config value
    ...
)
```

The pre-trade checks correctly receive `account_equity` as a parameter (current balance), but `real_time_adjustments` reads it from the initial config. After a series of losses, the volatility-scaled position size will be inflated relative to the actual reduced equity.

**Fix:** Add `account_equity: float` as a parameter to `real_time_adjustments()`.

---

### 🟡 P3-T1 — `BaseTradingBot` violates Single Responsibility Principle

**File:** `src/trading/base_trading_bot.py` (~1000 lines)

The class handles: signal management, order execution, position tracking, three notification paths, JSON state persistence, database persistence, balance management, P&L calculation, Backtrader integration, stop-loss/take-profit monitoring, and metrics tracking.

**Fix:** Extract into focused sub-components:
- `TradeExecutor` — order placement and fill handling
- `PositionTracker` — position state + persistence
- `NotificationDispatcher` — single unified notification path
- `BotMetrics` — balance and PnL tracking

---

### 🟡 P3-T2 — `BaseBroker.buy()` / `sell()` are 200-line duplicates

**File:** `src/trading/broker/base_broker.py:1079–1331`

The two methods are identical except for `OrderSide.BUY` vs `OrderSide.SELL`. This is ~200 lines of duplicated code.

**Fix:** Extract a private `_create_bt_order(side: OrderSide, owner, data, size, price, ...)` method and call it from both `buy()` and `sell()`.

---

### 🟡 P3-T3 — Dead code in `save_state()`

**File:** `src/trading/base_trading_bot.py:642–643`

```python
folder = os.path.join("logs", "json")
os.makedirs(folder, exist_ok=True)
```

This folder creation is never used — the actual write goes to `self.state_file` which correctly uses `TRADING_STATE_DIR`. The `folder` variable is unused.

**Fix:** Remove the two lines.

---

### 🟡 P3-T4 — Crash-recovery marker file is fragile

**File:** `src/trading/strategy_manager.py:344–365`

The `.trading_service_running` marker at project root:
- Is not reliable in containerized deployments (ephemeral filesystem)
- Cannot support multiple concurrent `StrategyManager` instances (they share the same file)
- A SIGKILL or OOM kill won't clean up the file, which is the intended behavior, but any external process touching that path would cause false crash detection

**Fix:** Persist the "running" flag in the database (`trading_bots` table `status` column). On startup, query for bots in `status='running'` — that is inherently crash-recovery mode without needing a filesystem flag.

---

### 🟡 P3-T5 — `adapter_manager.py` imports `donotshare` secrets at module-level

**File:** `src/common/sentiments/adapters/adapter_manager.py:23`

```python
import config.donotshare.donotshare as secrets
```

This top-level import means the secrets file is loaded for any code that imports `AdapterManager` or `register_default_adapters`. The file name `donotshare` suggests it contains sensitive credentials; loading it eagerly increases the attack surface.

**Fix:** Move the import inside `add_adapter()`, guarded by the specific adapter that needs it, or use `os.getenv()` instead of a secrets module.

---

### 🟡 P3-T6 — `PositionNotificationManager` in `BaseBroker` silently never sends notifications

**File:** `src/trading/broker/base_broker.py:711`

```python
self.notification_manager = PositionNotificationManager(config, notification_client)
```

`notification_client` defaults to `None`. In `BrokerFactory.get_broker()`, no `notification_client` is passed to broker constructors. `PositionNotificationManager._send_notifications()` checks `if self.notification_client:` and just logs a warning otherwise. Position-opened/closed notifications from the broker layer silently fail for all deployments.

**Fix:** Either pass the `NotificationServiceClient` through the factory, or document that broker-level notifications are opt-in and remove the misleading default config keys `position_opened: true`.

---

### 🟡 P3-T7 — Risk sub-packages missing `__init__.py` (convention violation)

**Files:** `src/trading/risk/pre_trade/`, `real_time/`, `post_trade/`, `src/trading/risk/`

None of these directories contain `__init__.py`. Per `CLAUDE.md` section 2.2 and section 10, all packages must have `__init__.py`. While Python 3 namespace packages allow imports without them, the convention is not followed consistently and tooling (e.g. pytest collection, mypy, certain IDEs) may behave unexpectedly.

**Fix:** Add empty `__init__.py` to all four directories.

---

### 🟢 P4-T1 — `update_positions()` stop-loss trigger at -150% PnL is effectively disabled

**File:** `src/trading/base_trading_bot.py:848–853`

```python
if hasattr(self.strategy, "sl_atr_mult") and pnl <= -self.strategy.sl_atr_mult * 100:
    self.execute_trade("sell", ...)
```

`sl_atr_mult` is an ATR multiplier (e.g., 1.5). Multiplied by 100, the threshold becomes -150 — a loss of 150%, which is impossible for a long position. The stop-loss in this code path never fires.

**Fix:** The ATR-based stop-loss should be computed as an absolute price level, not a percentage. Use the stop price stored in the position entry data, or delegate stop-loss entirely to `RiskController.real_time_adjustments()`.

---

### 🟢 P4-T2 — `_initialize_bot_instance()` uses unnecessary duck-typing for dict

**File:** `src/trading/base_trading_bot.py:237`

```python
getattr(self.config, 'get', lambda x, y=None: y)('config_file', None)
```

`self.config` is typed as `Dict[str, Any]` and always passed as a dict. The duck-typing workaround for non-dict configs is unnecessary complexity.

**Fix:** `self.config.get('config_file', None)`.

---

### 🟢 P4-T3 — Emoji characters in `strategy_manager.py` log messages

**File:** `src/trading/strategy_manager.py` (multiple locations)

Log messages like `"📊 Strategy Monitor: ..."`, `"🔄 CRASH RECOVERY MODE: ..."` may cause encoding errors in log destinations configured for ASCII/Latin-1 (e.g. certain syslog or Windows Event Log configurations).

**Fix:** Remove emojis from log messages. Use `_logger.warning("CRASH RECOVERY MODE: ...")` and add emojis only to user-facing Telegram notifications.

---

## `src/common/` Issues

---

### 🟠 P2-C1 — `CircuitBreaker` state machine uses inverted/non-standard naming

**File:** `src/common/sentiments/adapters/adapter_manager.py:36–93`

The standard circuit-breaker states are:
- `CLOSED` → healthy, calls allowed
- `OPEN` → tripped, calls blocked
- `HALF_OPEN` → recovery probe, limited calls allowed

The implementation uses:
- `HEALTHY` (≈ `CLOSED`)
- `FAILED` (≈ `OPEN`, but named from `AdapterStatus` which also has `CIRCUIT_OPEN`)
- `CIRCUIT_OPEN` (≈ `HALF_OPEN`)

The naming is the reverse of industry convention: `CIRCUIT_OPEN` *allows* calls (up to `half_open_max_calls`), while `FAILED` *blocks* calls. Any developer extending this class will likely introduce bugs by relying on the name.

**Fix:** Rename states to `CLOSED`, `OPEN`, `HALF_OPEN`, or at minimum add clear docstrings to `can_execute()` explaining the custom naming.

---

### 🟠 P2-C2 — `asyncio.create_task()` called from synchronous context in `AdapterManager.__init__()` and `remove_adapter()`

**File:** `src/common/sentiments/adapters/adapter_manager.py:183, 241`

```python
# In __init__:
asyncio.create_task(self._global_coordinator.start_monitoring())

# In remove_adapter:
asyncio.create_task(self._adapters[name].close())
```

`asyncio.create_task()` requires a running event loop. `AdapterManager` is instantiated at module level via `get_adapter_manager()`, which is called from non-async code. This raises `RuntimeError: no running event loop` in Python 3.10+.

**Fix:** In `__init__`, store the coroutine and start it lazily in an `async def start()` method. In `remove_adapter`, use `asyncio.ensure_future()` guarded by a running-loop check, or schedule cleanup on the next async tick.

---

### 🟡 P3-C1 — `fetch_messages_from_adapter()` and `fetch_summary_from_adapter()` are 120-line duplicates

**File:** `src/common/sentiments/adapters/adapter_manager.py:246–396`

Both methods implement identical circuit-breaker guard, global rate-limit acquire/release, health recording, and error recording logic. Only the adapter method called differs (`adapter.fetch_messages(...)` vs `adapter.fetch_summary(...)`).

**Fix:** Extract a private `async def _protected_fetch(self, adapter_name, adapter_method_name, *args)` and call it from both.

---

### 🟡 P3-C2 — Four stub health checkers always return `UNKNOWN` status

**File:** `src/common/health_monitor.py:348–391`

`_check_telegram_bot_health`, `_check_api_service_health`, `_check_web_ui_health`, and `_check_trading_bot_health` all return:
```python
return HealthCheckResult(status=SystemHealthStatus.UNKNOWN, error_message="Health check not implemented yet")
```

These are registered in `_register_default_checkers()` and run on every `check_all_systems_health()` call, polluting health dashboards with permanent `UNKNOWN` entries.

**Fix:** Either implement them or remove them from `_register_default_checkers()` and let callers register them when ready via `register_health_checker()`.

---

### 🟡 P3-C3 — `_check_database_health()` uses raw SQL string incompatible with SQLAlchemy 2.x

**File:** `src/common/health_monitor.py:266`

```python
result = uow.s.execute("SELECT 1").scalar()
```

SQLAlchemy 2.0 requires `text()` for raw SQL:
```python
from sqlalchemy import text
result = uow.s.execute(text("SELECT 1")).scalar()
```

Without this, the health check will fail silently or raise a warning that is caught by the blanket `except`.

**Fix:** Wrap in `text()`.

---

### 🟡 P3-C4 — `RecommendationEngine` uses hardcoded indicator weights

**File:** `src/common/recommendation/engine.py:175`

```python
composite_score = (technical_score * 0.4 + fundamental_score * 0.6)
```

The 40/60 technical/fundamental split is a magic number with no configuration path. Different asset classes (crypto vs equities) or user preferences require different weightings.

**Fix:** Accept `technical_weight` and `fundamental_weight` as constructor parameters with sensible defaults.

---

### 🟡 P3-C5 — Multiple sentiment files use `sys.path.append` at module level

**Files:** `src/common/sentiments/collect_sentiment_async.py:32–33`, `src/common/sentiments/adapters/adapter_manager.py:16–17`

```python
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))
```

Per `CLAUDE.md` section 2.1, `sys.path` manipulation should use `parents[n]` (which these do correctly), but the insertion at *module import time* as a side-effect is problematic — importing either module modifies the global `sys.path` for the entire process. Normally, the project root is already on `sys.path` via the runner or `pytest.ini`.

**Fix:** Remove module-level `sys.path.append` calls from `src/` files. If needed for standalone scripts, gate behind `if __name__ == "__main__":`.

---

### 🟢 P4-C1 — Naive `datetime.now()` in `PerformanceMetrics` and `ExecutionPersistenceService`

**Files:** `src/trading/metrics_tracker.py:39`, `src/trading/execution_persistence.py:104, 122`

```python
last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
order_data["persisted_at"] = datetime.now().isoformat()
```

Per `CLAUDE.md` section 2.3, `datetime.now(timezone.utc)` must be used. Naive datetimes cause subtle bugs in time-series comparisons across DST boundaries or timezone differences.

**Fix:** Replace all `datetime.now()` with `datetime.now(timezone.utc)`.

---

### 🟢 P4-C2 — `CircuitBreaker` does not reset `failure_count` on half-open success

**File:** `src/common/sentiments/adapters/adapter_manager.py:72–79`

```python
def record_success(self) -> None:
    if self.state == AdapterStatus.CIRCUIT_OPEN:
        self.half_open_calls += 1
        if self.half_open_calls >= self.config.half_open_max_calls:
            self.state = AdapterStatus.HEALTHY
            self.failure_count = 0   # ← only reset here
```

When `record_success()` is called in `HEALTHY` state (normal operation), `failure_count` is reset correctly. But in `CIRCUIT_OPEN` (half-open), `failure_count` is only reset when transitioning back to `HEALTHY` *after* `half_open_max_calls` successes. On the intermediate successes, the count remains non-zero. This is benign but inconsistent.

---

## Cross-Cutting Issues

---

### 🟠 P2-X1 — No integration tests covering the full trade execution path

The unit tests in `src/trading/broker/tests/` test broker connectivity and individual methods. There are no integration tests that:
1. Start a `StrategyInstance` with a mock broker and mock data feed
2. Inject a synthetic signal
3. Assert the trade is executed, the position is recorded, the notification is sent, and the DB is updated

This gap means the double-execution bug (P1-T2) and the commission bug (P2-T3) would not be caught by the test suite.

**Fix:** Add an integration test fixture with `MockBroker`, an in-memory trade repository, and a `MockDataFeed` that emits a fixed price series.

---

### 🟡 P3-X1 — Two parallel persistence layers for the same trade data (DB + JSON files)

`BaseTradingBot` writes completed trades to:
1. The database via `trade_repository.update_trade()` (primary)
2. `execution_persistence.save_trade()` → `logs/json/trades.json` (legacy)

The two stores are not kept in sync (e.g., DB failure does not roll back the JSON write, and vice versa). This creates diverging records and confusion during recovery.

**Fix:** Designate the database as the single source of truth. Deprecate and eventually remove the JSON persistence once the DB recovery path (`_load_open_positions_from_db`) is confirmed stable.

---

### 🟡 P3-X2 — `BaseTradingBot` and `StrategyInstance` both update DB bot status independently

`BaseTradingBot.run()` updates `last_heartbeat` and `current_balance` in the DB every loop iteration. `StrategyInstance._heartbeat_loop()` also updates `last_heartbeat` from its thread. Two independent writers to the same DB row creates unnecessary contention and potential last-write-wins overwrite of `current_balance`.

**Fix:** Centralize heartbeat and status updates in `StrategyInstance`; have `BaseTradingBot` expose an in-memory status object that `StrategyInstance` reads and persists.

---

## Prioritized Fix Backlog

| # | ID | Severity | Description | Effort |
|---|-----|----------|-------------|--------|
| 1 | P1-T1 | 🔴 Critical | `_get_simulated_market_price()` hardcoded $100 | S |
| 2 | P1-T4 | 🔴 Critical | Paper PnL always $0 for closed positions | S |
| 3 | P1-T2 | 🔴 Critical | Double execution loop race condition | M |
| 4 | P1-T3 | 🔴 Critical | Deprecated `asyncio.get_event_loop()` | S |
| 5 | P2-T1 | 🟠 High | Triple notification duplication | S |
| 6 | P2-T3 | 🟠 High | Commission formula wrong and hardcoded | S |
| 7 | P2-T4 | 🟠 High | Non-atomic JSON write — data loss on crash | S |
| 8 | P2-C2 | 🟠 High | `asyncio.create_task()` in sync `AdapterManager` init | S |
| 9 | P2-C1 | 🟠 High | `CircuitBreaker` inverted state naming | S |
| 10 | P2-T2 | 🟠 High | `InstanceService` singleton drops DI | M |
| 11 | P2-T5 | 🟠 High | `MetricsRegistry` full file write per trade | M |
| 12 | P2-T6 | 🟠 High | `RiskController.real_time_adjustments` uses stale equity | S |
| 13 | P2-X1 | 🟠 High | No integration tests for trade execution path | L |
| 14 | P3-T1 | 🟡 Medium | `BaseTradingBot` SRP violation / decompose | L |
| 15 | P3-T2 | 🟡 Medium | `buy()` / `sell()` 200-line duplication | S |
| 16 | P3-T4 | 🟡 Medium | Crash marker → DB-backed crash detection | M |
| 17 | P3-T5 | 🟡 Medium | `donotshare` module-level import | S |
| 18 | P3-T6 | 🟡 Medium | Broker `PositionNotificationManager` silently does nothing | S |
| 19 | P3-T7 | 🟡 Medium | Missing `__init__.py` in risk sub-packages | S |
| 20 | P3-C1 | 🟡 Medium | 120-line duplicate fetch methods in `AdapterManager` | S |
| 21 | P3-C2 | 🟡 Medium | 4 stub health checkers polluting health reports | S |
| 22 | P3-C3 | 🟡 Medium | Raw SQL incompatible with SQLAlchemy 2.x | S |
| 23 | P3-C4 | 🟡 Medium | Hardcoded 40/60 indicator weights | S |
| 24 | P3-C5 | 🟡 Medium | `sys.path.append` at module level in `src/` files | S |
| 25 | P3-X1 | 🟡 Medium | Two diverging persistence layers (DB + JSON) | M |
| 26 | P3-X2 | 🟡 Medium | Dual DB heartbeat writers contending on same row | M |
| 27 | P4-T1 | 🟢 Low | Stop-loss threshold at -150% never triggers | S |
| 28 | P4-T3 | 🟢 Low | Emoji chars in log messages risk encoding issues | S |
| 29 | P4-C1 | 🟢 Low | Naive `datetime.now()` without timezone | S |
| 30 | P4-T2 | 🟢 Low | Unnecessary duck-typing for dict in `_initialize_bot_instance` | S |

**Effort key:** S = Small (< 1 day) · M = Medium (1–3 days) · L = Large (> 3 days)

---

*This document was generated by automated architectural review on 2026-05-25.*
