# Implementation plan: BaseBroker and conditional inheritance

**Scope:** Remove the runtime switch where `BaseBroker` inherits either `backtrader.broker.BrokerBase` or `abc.ABC`, and replace it with a **stable domain base class** plus an optional **Backtrader-facing adapter**. This addresses architectural review section 4.6 and recommendation 16.

**Primary references:** `src/trading/broker/base_broker.py`, `src/trading/broker/backtrader_broker_bridge.py`, `src/trading/broker/tests/test_backtrader_integration.py`, call sites using `cerebro.setbroker`.

### Implementation status (complete)

These items match sections 3–8 and the definition of done below.

| Item | Where |
|------|--------|
| Core is always `abc.ABC` | `BaseBroker` in `base_broker.py` |
| No `BaseBrokerClass` / conditional `BrokerBase` MRO | removed |
| Single probe for optional backtrader | `backtrader_availability.py` → `BACKTRADER_AVAILABLE`; `base_broker` imports the flag only (no `import backtrader` there) |
| Bridge loads `backtrader` only when available | `backtrader_broker_bridge.py`: `import backtrader as bt` inside `if BACKTRADER_AVAILABLE:` |
| Factory contract documented | `get_broker` docstring in `broker_factory.py`: return value must be wrapped with `wrap_broker_for_cerebro` before `cerebro.setbroker` |
| Typing protocol for cores | `SupportsBacktraderBridge` in `backtrader_bridge_protocol.py`; `wrap_broker_for_cerebro(core: SupportsBacktraderBridge)` |
| Inventory of bridge surface | Appendix A below |
| Integration smoke: `get_broker` + wrap + `cerebro.run` | `tests/integration/test_cerebro_broker_bridge_smoke.py` |
| `MockBroker` compatible with bridge | legacy `buy`/`sell` dict helpers renamed to `simulate_market_buy` / `simulate_market_sell` so `BaseBroker.buy`/`sell` are not shadowed |
| Tests / example | `test_backtrader_integration.py`, `docs/examples/backtrader_integration_example.py` |
| Call sites | `base_trading_bot.py`, `strategy_instance.py` |

**Optional later:** subclass `BackBroker`, or implement `setcash` on the bridge if callers must mutate cash on the attached broker after `setbroker`.

---

## 1. Problem

- Previously, `BaseBroker` inherited either `bt.broker.BrokerBase` or `ABC` depending on whether `backtrader` imported.
- Effects:
  - **Typing and static analysis** cannot express a single stable type for `BaseBroker`.
  - **Method resolution order** and `super()` behavior differ between environments.
  - **Tests and docs** must branch on `BACKTRADER_AVAILABLE` for `isinstance(..., bt.broker.BrokerBase)`.
- Callers still need a real `BrokerBase` instance when integrating with `bt.Cerebro` (see `BaseTradingBot.run_backtrader_engine`, `StrategyInstance._setup_backtrader`), so the fix must preserve **Cerebro compatibility**, not drop it.

---

## 2. Target design (composition)

Introduce two layers:

| Layer | Role |
|--------|------|
| **`BaseBrokerCore`** (or keep the name `BaseBroker` for minimal churn) | Always subclasses `abc.ABC`. Holds paper/live logic, order objects, notifications, execution metrics—everything that does not need to be a Backtrader broker. |
| **`BacktraderBrokerBridge`** | Always subclasses `bt.broker.BrokerBase` when backtrader is installed. Holds a **reference** to the core broker and implements or overrides the minimum Backtrader broker contract by **delegating** to the core (and forwarding Cerebro lifecycle hooks). |

Public API for applications using Cerebro:

- Either **`get_backtrader_broker(core: BaseBroker) -> bt.broker.BrokerBase`** in `base_broker.py` (or a small `backtrader_bridge.py`), or
- **`core.as_backtrader_broker()`** returning the bridge, lazily constructed and cached on the core instance.

**Rule:** No conditional **base class** of the main implementation; optional backtrader is only pulled in via the bridge module / factory.

---

## 3. Step-by-step implementation

### Step 1 — Inventory and contracts

1. In `base_broker.py`, list every method and attribute that exists **because** of Backtrader (`buy`, `sell`, `cancel`, `getcash`, `getvalue`, `next`, order hooks, etc.). Mark which are overrides of `bt.broker.BrokerBase` (consult Backtrader docs / `inspect` on `bt.broker.BrokerBase`).
2. Grep the repo for:
   - `setbroker`, `cerebro.broker =`, `is_backtrader_mode`, `BACKTRADER_AVAILABLE`, `isinstance(..., BaseBroker)`.
3. Record which code paths assign a **factory broker** (`BinanceBroker`, `MockBroker`, etc.) directly to Cerebro today—those types must either inherit the new core base only and use the same bridge, or the bridge must wrap the **runtime instance** returned by `get_broker`.

Deliverable: short checklist in a comment or this doc (subsection listing methods Cerebro actually invokes in your smoke tests).

### Step 2 — Extract `BaseBrokerCore`

1. Move **non–Backtrader-specific** state and methods onto `BaseBrokerCore(ABC)`.
2. Remove any `isinstance(self, bt.broker.BrokerBase)` branching from **`__init__`**. Initialization should not depend on inheriting Backtrader’s broker base.
3. Keep `BACKTRADER_AVAILABLE` only for:
   - optional imports in the bridge module,
   - mapping enums (`OrderStatus` ↔ `bt.Order.Status`), and helpers that need `bt` symbols.

Goal: `BaseBrokerCore` imports and tests cleanly **without** subclassing Backtrader.

### Step 3 — Implement `BacktraderBrokerBridge`

1. New class (same file or `backtrader_broker_bridge.py`):

   ```text
   class BacktraderBrokerBridge(bt.broker.BrokerBase):
       def __init__(self, core: BaseBrokerCore):
           super().__init__()
           self._core = core
   ```

2. Implement **only** what Cerebro calls on your brokers. Minimum set to verify first:
   - Cash and value accessors used by strategies/analyzers (`getcash`, `getvalue`, or whatever your strategies touch).
   - Order entry points strategies use (`buy`, `sell`, `cancel` if exposed through the strategy-broker path).
3. Delegate each override to `_core`, translating types (Backtrader `Order` / data feed handles ↔ internal `Order` / symbol strings).
4. Ensure **`_core`** does not call methods that assume `self` is the bridge; avoid recursion (bridge calls core; core should not call `buy` on `self` expecting Backtrader semantics unless explicitly designed).

### Step 4 — Wire concrete brokers (`BinanceBroker`, `MockBroker`, …)

1. Change concrete classes to inherit **`BaseBrokerCore`** only (not `BrokerBase`).
2. If the class currently relies on `super().__init__()` reaching `bt.broker.BrokerBase`, replace with `BaseBrokerCore.__init__` only.
3. In `BaseTradingBot.run_backtrader_engine` and `StrategyInstance._setup_backtrader`:
   - If `self.broker` is a `BaseBrokerCore`, pass **`BacktraderBrokerBridge(self.broker)`** (or the factory) to Cerebro instead of the raw instance **unless** config explicitly uses built-in `cerebro.broker` only.

### Step 5 — API compatibility and factory

1. **Factory (`get_broker`)**: Decide whether it returns `BaseBrokerCore` only (recommended) and callers wrap for Cerebro, or returns a small tuple—avoid returning two types without documentation.
2. **Deprecation path**: If external code does `isinstance(x, BaseBroker)`, keep `class BaseBroker(BaseBrokerCore): pass` as an alias for one release, or re-export `BaseBrokerCore` as `BaseBroker` after renaming in a single mechanical PR.
3. **`is_backtrader_mode`**: Redefine as “this core is wrapped for Cerebro” or “running inside bridge”; avoid `isinstance(self, bt.broker.BrokerBase)` on the core.

### Step 6 — Typing and `py.typed` / protocols (optional but valuable)

1. Define a `Protocol` for “what the bridge needs from the core” (e.g. `place_paper_order`, internal cash accessors) to keep the bridge thin and testable.
2. Annotate `get_backtrader_broker(core: BaseBrokerCore) -> bt.broker.BrokerBase` behind `TYPE_CHECKING` or a string annotation if needed for optional backtrader.

### Step 7 — Tests

1. Update `test_backtrader_integration.py`:
   - When backtrader is available, assert **`isinstance(bridge, bt.broker.BrokerBase)`** for the bridge, not necessarily for `BaseBrokerCore`.
   - Keep tests for `buy` / `sell` / `cancel` on either the bridge forwarding to core or on core’s public API—pick one documented entry point.
2. Add an integration smoke test:

   ```text
   cerebro = bt.Cerebro()
   core = MockBroker(...)
   cerebro.setbroker(BacktraderBrokerBridge(core))
   cerebro.adddata(...)
   cerebro.addstrategy(...)
   cerebro.run()
   ```

3. Run existing backtrader-related tests and at least one path that uses `get_broker` + Cerebro.

### Step 8 — Documentation and cleanup

1. Update module docstring in `base_broker.py`: describe **core + bridge**, remove “conditional inheritance” wording.
2. Delete `BaseBrokerClass` global and the try/except import block from the core module; move backtrader import into the bridge module guarded by lazy import or try/except **only there**.
3. If `docs/examples/backtrader_integration_example.py` uses raw `BaseBroker`, update it to show the bridge.

---

## 4. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Missing a `BrokerBase` method Cerebro invokes at runtime | Step 1 inventory + smoke `cerebro.run()` with each broker type; compare trace or Backtrader source for your version. |
| Double counting cash / positions (core vs Cerebro state) | Single source of truth in core; bridge is stateless aside from reference to core. |
| Large diff | Land Step 2–3 with `MockBroker` only, then expand to Binance/IBKR in follow-up PRs. |
| Optional backtrader in CI | Keep `@unittest.skipUnless(BACKTRADER_AVAILABLE, ...)`; add one job that installs backtrader. |

---

## 5. Definition of done

- `BaseBrokerCore` (or renamed equivalent) **always** subclasses `ABC`; no conditional base class for the main implementation.
- Cerebro integration uses **`BacktraderBrokerBridge`** (or equivalent) implementing `bt.broker.BrokerBase`.
- No `isinstance(self, bt.broker.BrokerBase)` inside core `__init__`.
- Tests and examples updated; `grep` shows no `BaseBrokerClass = ...` pattern in production code.
- Manual or automated run: at least one full `cerebro.run()` with a factory-created broker via the bridge.

---

## Appendix A — Bridge delegation inventory (`BacktraderBrokerBridge`)

Methods overridden on `bt.broker.BrokerBase` and delegated to the core (verified against smoke strategies that only call `next`, plus standard analyzer/strategy cash queries):

| Bridge method | Core target | Notes |
|---------------|-------------|--------|
| `getcash` | `_bt_getcash` | Required abstract broker API |
| `getvalue` | `_bt_getvalue` | Portfolio value |
| `getposition` | `_bt_getposition` | Returns `bt.position.Position` |
| `buy` | `buy` | Strategy orders; signature matches Backtrader |
| `sell` | `sell` | Same |
| `cancel` | `cancel` | Order cancel |
| `next` | `next` | Per-bar processing / paper fill simulation on core |
| `get_notification` | `get_notification` | Custom notification channel used by this codebase |
| *(constructor / bar sync)* | `startingcash`, `cash` | Set from `_bt_getcash()` after `super().__init__()`; `next()` refreshes `cash` from the core so writers see updates |

`BrokerBase` defines other methods (`submit`, `start`, `stop`, …). Subclasses inherit defaults until a failing stack trace proves another override is needed; extend the bridge when that happens.

**Repo grep snapshot (attach patterns):** production code should use `wrap_broker_for_cerebro` + `setbroker`, not `cerebro.broker = <core>`. Examples under `docs/examples` may still use the default `BackBroker` for demos that do not use `get_broker`.

---

## 6. Out of scope (for later)

- Reimplementing Backtrader’s full broker API surface (only what you use).
- Changing IBKR/Binance live execution paths beyond what is required to inherit `BaseBroker` and construct the bridge.
