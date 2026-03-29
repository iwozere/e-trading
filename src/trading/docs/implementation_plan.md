# Phase 4: Quality & Observability

This phase focuses on hardening the system, improving monitoring, and refactoring core components to reduce technical debt.

## Status and next steps (cross-checked with code and `architectural_review.md`)

The bullets below are ordered for execution: unresolved or partial work first, then what is already done.

### Remaining work (step by step)

1. **`BaseBroker` conditional inheritance (larger refactor)**  
   Still switches base class between `backtrader.broker.BrokerBase` and `abc.ABC` depending on install. **Next:** composition over inheritance (architectural review section 4.6, recommendation 16) without breaking `cerebro` registration — needs a dedicated change + test updates.

2. **Risk layer — depth beyond execution gates**  
   Buys use `pre_trade_checks`; sells use new `pre_exit_checks`. Modules under `risk/` remain mostly stubs; post-trade / real-time hooks from `RiskController` are not wired into `BaseTradingBot` yet.

3. **Observability verification (manual)**  
   Confirm in a real or batch run that aggregated metrics and append logs behave as intended (`metrics.json`, execution persistence log paths under `logs/`).

4. **Test breadth (ongoing)**  
   Phase 4 adds orchestration integration tests and metrics unit tests; architectural review still called out broader coverage (brokers, risk, handlers). Expand as modules are touched.

### Recently addressed (2026-03-29)

- **`BaseTradingBot` state path:** imports `TRADING_STATE_DIR` from `constants`; per-bot directory under `data/bots/<bot_id>/`.
- **`StrategyManager` DB status updates:** `_coerce_db_bot_id` / `_update_bot_status_db` skip persistence for non-numeric instance IDs; DB poll loop fixed (`bot_id not in self.strategy_instances` replaced broken `exists` name).
- **`binance_broker.py`:** explicit `binance.enums` imports (no star import).
- **`broker_factory._env`:** docstring aligned with env-only behavior.
- **Sell-side risk:** `RiskController.pre_exit_checks` + call from `execute_trade` before placing orders.

### Phase 4 proposed scope — implementation status

| Area | Status |
|------|--------|
| `execution_persistence.py` + delegation from `BaseTradingBot` | Done |
| `metrics_tracker.py` + `metrics_registry` on trade close | Done |
| `instance_service.py`, `strategy_registry.py`, thinner `strategy_manager.py` | Done (manager still multi-purpose but much smaller than original “god class” size) |
| `tests/integration/test_orchestration_flow.py` | Done |
| `tests/unit/trading/test_metrics_tracker.py` | Done |
| Automated verification | `pytest` on the two test paths above passes in dev |

### Architectural review criticals — quick mapping

| Original issue | Status |
|----------------|--------|
| `trade_id` / BUY branch NameError | Fixed |
| `asyncio.run()` from inside running loop | Mitigated via `_run_async` (thread fallback when no loop) |
| `state_file` not set | Fixed via `TRADING_STATE_DIR` in `BaseTradingBot` |
| Live `get_order_status` stubs | Fixed (Binance / IBKR) |
| `get_open_trades` dict vs ORM | Fixed (`_load_open_positions_from_db` supports both) |
| Credentials: `donotshare` at import | Improved: `dotenv` + env vars in `broker_factory` |
| Sensitive parameters in logs | Fixed (`_sanitize_params`) |
| Risk bypassed entirely | Partial (pre-trade on buy; `pre_exit_checks` on sell) |
| `active_positions` thread safety | Fixed (`threading.RLock`) |
| Dual `LiveTradingBot` vs `StrategyManager` | Largely unified (wrapper + `StrategyInstance` runs `BaseTradingBot`) |
| Batch simulation `sys.modules` monkey-patch | Addressed: `MockRepository` injected as `trade_repository` |
| Absolute paths for config/state/logs | Partial (`constants.py`, persistence logs); state dir import must be completed |
| `orders.json` concurrency | Addressed via centralized persistence service |

## User Review Required

> [!IMPORTANT]
> **Metrics Storage**: I propose a local-first approach using a `metrics.json` file for performance tracking before moving to a full Prometheus exporter. Does this meet your immediate "observability" needs?
> 
> **Refactoring Risk**: Splitting `StrategyManager.py` is a structural change. I will ensure backward compatibility for existing manifest loading paths.

## Proposed Changes

### 1. Centralized Execution Persistence
Address Item 15 in the architectural review. `orders.json` and `trades.json` are currently handled via direct file writes in `BaseTradingBot`, which is risky under high concurrency.

#### [NEW] [execution_persistence.py](file:///c:/dev/cursor/e-trading/src/trading/execution_persistence.py)
- A thread-safe service for logging trades and orders.
- Uses `threading.Lock` and atomic write patterns to prevent corruption.
- Supports both JSON (structured) and append-only (robust) formats.

#### [MODIFY] [base_trading_bot.py](file:///c:/dev/cursor/e-trading/src/trading/base_trading_bot.py)
- Delegate order/trade persistence to the new service.

---

### 2. Observability & Performance Metrics
Implement a cross-instance tracking system for bot health and PnL.

#### [NEW] [metrics_tracker.py](file:///c:/dev/cursor/e-trading/src/trading/metrics_tracker.py)
- `PerformanceMetrics`: Dataclass for PnL, Win Rate, Max Drawdown.
- `MetricsRegistry`: Singleton/Service to aggregate metrics from all active `StrategyInstance` objects.

#### [MODIFY] [base_trading_bot.py](file:///c:/dev/cursor/e-trading/src/trading/base_trading_bot.py)
- Push performance updates to the `MetricsRegistry` on every trade close.

---

### 3. Decoupling Strategy Manager (Refactoring)
Address Items 13 and 19. Split the God class into focused services.

#### [NEW] [instance_service.py](file:///c:/dev/cursor/e-trading/src/trading/instance_service.py)
- Logic for instantiating, starting, and stopping `StrategyInstance` objects.

#### [NEW] [strategy_registry.py](file:///c:/dev/cursor/e-trading/src/trading/strategy_registry.py)
- Logic for discovery and validation of strategy classes and parameters.

#### [MODIFY] [strategy_manager.py](file:///c:/dev/cursor/e-trading/src/trading/strategy_manager.py)
- Becomes a thin orchestration layer delegating to the new services.

---

### 4. Integration Testing
Validate the Phase 3 unification (Backtrader -> Signal -> Bot -> Broker).

#### [NEW] [test_orchestration_flow.py](file:///c:/dev/cursor/e-trading/tests/integration/test_orchestration_flow.py)
- A pytest-based integration test using `MockBroker` and a sample CSV feed.
- Asserts that a signal generated in Backtrader correctly results in a filled order in the `MockBroker` through the `BaseTradingBot` queue.

## Open Questions

- **Prometheus**: Do you want a real Prometheus HTTP exporter now, or is a structured log/JSON file sufficient for your initial monitoring setup?
- **God Class Splitting**: Should I prioritize keeping `StrategyManager` fully backward compatible (meaning it still exposes the same public methods) while moving implementation to sub-modules?

## Verification Plan

### Automated Tests
- `pytest tests/integration/test_orchestration_flow.py`
- `pytest tests/unit/trading/test_metrics_tracker.py`

### Manual Verification
- Run a batch simulation and verify that `metrics.json` is correctly updated with aggregated PnL.
- Verify `logs/orders.log` and `logs/trades.log` are populated using the new persistence service.
