# Trading Module — Refactoring Plan

_Generated: 2026-03-25 | Senior Software Architect Review_

---

## Overview

The `src/trading` module is functional and well-integrated, but carries architectural debt that increases maintenance cost and risk as the system grows. This document lists the key issues, steps to resolve them, estimated effort, and a recommended execution order.

---

## 🔴 High Priority — Critical / Must Fix

### 1. `BaseTradingBot` — "God Class" Decomposition

**Problem**: 758-line class handling signal processing, trade execution, DB persistence, notifications, and risk — all in one class. Violates the Single Responsibility Principle (SRP).

**Steps**:
1. Extract `TradeExecutor` — encapsulates buy/sell logic
2. Extract `StateManager` — handles save/load from DB and JSON
3. Extract `NotificationDispatcher` — wraps `notification_client` calls
4. Wire all three via Dependency Injection (DI) into `BaseTradingBot`

**Effort**: 🟡 2–3 days

---

### 2. Dual Persistence — Risk of State Drift

**Problem**: Active positions are tracked in both the database and `logs/json/state.json`. This creates a risk of inconsistency after crashes or restarts.

**Steps**:
1. Remove JSON as a source of truth for active positions (keep for debug audit only)
2. DB is the single source of truth at startup
3. Add a reconciliation check on bot startup: compare DB open trades vs. local JSON and log discrepancies

**Effort**: 🟢 0.5–1 day

---

### 3. `RiskController` is Not Wired Into the Trade Path

**Problem**: `RiskController` is instantiated in `__init__` but `execute_trade()` never calls `pre_trade_checks()`. Risk management is effectively bypassed.

**Steps**:
1. Call `risk_controller.pre_trade_checks()` before every `execute_trade()` call
2. Return early (skip the trade) if the check returns `0.0`
3. Log the rejection reason clearly
4. Add a unit test that asserts a trade is blocked when a risk limit is exceeded

**Effort**: 🟢 0.5 day

---

### 4. `asyncio.run()` Inside Synchronous Trade Path

**Problem**: `asyncio.run()` is called from `execute_trade()`, which is a sync function. This crashes at runtime when called from an async context (e.g., from `StrategyManager`).

```python
# ❌ Current — dangerous in async contexts
asyncio.run(self.position_notification_manager.notify_position_opened(position_data))
```

**Steps**:
1. Replace with `asyncio.create_task()` in async contexts
2. In sync contexts (the base bot run loop), use a dedicated background thread or a thread-safe notification queue
3. Test both code paths (sync `run()` loop and async `StrategyManager`)

**Effort**: 🟡 1 day

---

## 🟡 Medium Priority — Significant Improvements

### 5. Dual Strategy Registry

**Problem**: `live_trading_bot.py` maintains a hardcoded `STRATEGY_REGISTRY` dict while `StrategyHandler` already provides a proper plugin registry. They can drift independently.

```python
# ❌ Hardcoded — bypasses StrategyHandler
STRATEGY_REGISTRY = {
    "CustomStrategy": CustomStrategy,
    "AdvancedStrategyFramework": AdvancedStrategyFramework,
}
```

**Steps**:
1. Delete `STRATEGY_REGISTRY` from `live_trading_bot.py`
2. Route all strategy resolution through `strategy_handler.get_strategy_class()`

**Effort**: 🟢 2 hours

---

### 6. `StrategyInstance` — Second God Class

**Problem**: `StrategyInstance` in `strategy_manager.py` handles data feed management, reconnection logic, Backtrader setup, heartbeat threading, DB updates, and trade notifications — all in one class.

**Steps**:
1. Extract `DataFeedMonitor` — encapsulates health checks and reconnection
2. Extract `HeartbeatReporter` — manages the DB heartbeat background thread
3. `StrategyInstance` delegates to both

**Effort**: 🟡 1–2 days

---

### 7. Fragile Stop-Loss Enforcement via Duck Typing

**Problem**: `update_positions()` uses `hasattr(self.strategy, 'sl_atr_mult')` to determine stop-loss. If the strategy doesn't expose this attribute, the SL is silently skipped — a critical safety gap.

**Steps**:
1. Define a `StopLossConfig` dataclass (or Pydantic model) with `sl_pct`, `tp_pct`, `use_atr` fields
2. Source this from the config rather than the strategy instance
3. Pass through `RiskController.real_time_adjustments()` for consistency

**Effort**: 🟡 1 day

---

## 🟢 Low Priority — Polish & Observability

### 8. Hardcoded Commission in Trade Execution

**Problem**: `commission = gross_pnl * 0.001` is hardcoded in `execute_trade()`.

**Steps**:
1. Add `commission_pct` (default `0.001`) to the trading config schema
2. Read from config in `execute_trade()`

**Effort**: 🟢 1 hour

---

### 9. Structured Logging and Metrics

**Problem**: All observability is through free-text logs with no structured context fields, making it hard to filter by `bot_id` or `trade_id` in log aggregators.

**Steps**:
1. Add structured log context (bot_id, pair, trade_id) using a logging adapter or `structlog`
2. Optionally emit Prometheus metrics: trade count, net PnL, last heartbeat timestamp

**Effort**: 🟡 1–2 days

---

## Summary & Execution Order

| Priority | # | Issue | Effort |
|---|---|---|---|
| 🔴 | 3 | RiskController bypassed | 🟢 0.5d |
| 🔴 | 2 | Dual persistence / state drift | 🟢 0.5d |
| 🟡 | 5 | Dual strategy registry | 🟢 2h |
| 🔴 | 4 | `asyncio.run()` in sync flow | 🟡 1d |
| 🟡 | 7 | Fragile SL enforcement | 🟡 1d |
| 🟡 | 6 | `StrategyInstance` bloat | 🟡 1–2d |
| 🔴 | 1 | God class `BaseTradingBot` | 🟡 2–3d |
| 🟢 | 8 | Hardcoded commission | 🟢 1h |
| 🟢 | 9 | Structured logging | 🟡 1–2d |

**Recommended start**: Items 3, 2, and 5 have the best ROI — low effort, clear scope, and no design risk. Begin there before tackling the larger class decompositions (1, 6).
