# Pipeline p13_bdsh — Architectural Review & Refactoring Plan

_Generated: 2026-03-25 | Senior Software Architect Review_

---

## 1. Problem Summary

Pipeline p13 (`VIX-Threshold Scaling Strategy / BDSH`) is a backtesting and live-signal pipeline that:
- Uses daily VIX Z-Score to determine equity exposure across 3 entry tiers.
- Applies an ATR-based stop-loss with a cooldown mechanism.
- Persists state to a JSON file and generates CSV + chart artefacts.
- Is run via `run_p13.py` with a CLI interface.

The pipeline is well-reasoned algorithmically, but has structural debt that limits testability, composability, and production readiness.

---

## 2. Assumptions

- Pipeline is currently used as a **backtesting + production signal** tool, not wired to live order execution.
- The `DataManager` abstraction for data ingestion is shared with the broader trading system.
- The strategy is expected to support multiple tickers independently.

---

## 3. Architecture Overview

```
run_p13.py (CLI)
    └── P13Pipeline (Orchestrator)
            ├── data_loader.py   → Downloads & preprocesses OHLCV + VIX
            ├── VIXScalingEngine → Z-score, signals, ATR SL, backtest loop, metrics, plotting
            └── config.py        → All magic numbers and paths (module-level globals)
```

The design is mostly linear and sound, but `VIXScalingEngine` is overloaded and `config.py` has critical structural issues.

---

## 4. Issues, Improvements & Steps

---

### 🔴 Issue 1: `INITIAL_CAPITAL` Defined Twice in `config.py`

**Problem**: The constant `INITIAL_CAPITAL = 100000.0` appears on **line 19 and line 23**. The second silently overwrites the first. This is a latent bug.

```python
INITIAL_CAPITAL = 100000.0  # line 19 ← overwritten
...
INITIAL_CAPITAL = 100000.0  # line 23 ← actual value used
```

Also, `SLIPPAGE_COMMISSION` (line 20) and `SLIPPAGE_PCT` (line 24) are two separate names for the same concept — but only `SLIPPAGE_COMMISSION` is used by `VIXScalingEngine`. `SLIPPAGE_PCT` is orphaned.

**Steps**:
1. Remove the duplicate `INITIAL_CAPITAL` assignment.
2. Pick one name (`SLIPPAGE_PCT`) and update `VIXScalingEngine` to use it.
3. Delete `SLIPPAGE_COMMISSION`.

**Effort**: 🟢 15 minutes

---

### 🔴 Issue 2: `config.py` Executes Side Effects at Import Time

**Problem**: `os.makedirs(RESULTS_DIR, exist_ok=True)` is called at module level — meaning importing the config creates directories on disk, which breaks unit tests and makes the module non-deterministic.

```python
# config.py — executed on every import
os.makedirs(RESULTS_DIR, exist_ok=True)  # ❌ side effect at import time
```

Also, `import os` and `from pathlib import Path` appear at the end of the file instead of the top, which violates PEP 8 and is confusing.

**Steps**:
1. Move all `import` statements to the top of the file.
2. Remove `os.makedirs(...)` from module level.
3. Create the directory inside `P13Pipeline.__init__()` or `run_p13.main()` instead.

**Effort**: 🟢 30 minutes

---

### 🔴 Issue 3: `config.__dict__` Passed as a Config Bag

**Problem**: `P13Pipeline` passes `config.__dict__` directly to `VIXScalingEngine`. This includes Python dunder attributes and is fragile — any module-level variable added to `config.py` gets silently injected.

```python
self.engine = VIXScalingEngine(config.__dict__)  # ❌
```

**Steps**:
1. Define a `Pydantic` model or a `dataclass` (`P13Config`) that declares all valid parameters explicitly.
2. Instantiate it in `P13Pipeline.__init__()` from the config module's values.
3. Pass the typed config object to `VIXScalingEngine`.

**Effort**: 🟡 1–2 hours

---

### 🔴 Issue 4: `P13Pipeline.save_state()` Has a Misplaced `else` Clause

**Problem**: There is a dangling `else` on line 108 of `p13_pipeline.py` that belongs to the `try/except`, not to the saving logic. This logs a misleading warning even if the save succeeded.

```python
def save_state(self, state: dict):
    try:
        with open(...) as f:
            json.dump(state, f, indent=4)
        logger.info(...)
    except Exception as e:
        logger.error(...)
    else:
        logger.warning("No tickers were successfully processed.")  # ❌ runs on success!
```

**Steps**:
1. Remove the dangling `else` from `save_state()`.
2. Move the "No tickers processed" warning to the `P13Pipeline.run()` method where `results_summary` is checked.

**Effort**: 🟢 5 minutes

---

### 🟡 Issue 5: `VIXScalingEngine` Violates SRP — "God Class" for the Engine

**Problem**: The engine handles 5 distinct responsibilities:
1. Z-score calculation (`calculate_vix_zscore`)
2. ATR calculation (`calculate_atr`)
3. Signal generation (`generate_signals`)
4. Full backtest simulation (`run_backtest`)
5. Metrics (`calculate_metrics`)
6. Visualization (`plot_results`)

`plot_results` is especially problematic — it carries state from `run_backtest` via `self.markers`, creating an implicit coupling: you **must** call `run_backtest` before `plot_results` or it will crash.

**Steps**:
1. Extract `class P13Plotter` with `plot_results(results, markers, ticker, output_path)` — receive markers as an explicit argument, not via `self.markers`.
2. Optionally extract `class P13Metrics` with `calculate_metrics(results)`.
3. Keep `VIXScalingEngine` focused on the simulation loop only.

**Effort**: 🟡 2–3 hours

---

### 🟡 Issue 6: `generate_signals()` is Unused

**Problem**: `generate_signals()` exists as a public method on `VIXScalingEngine` but **is never called** — the signal logic is duplicated directly inside `run_backtest()`. This creates two diverging codepaths for the same logic.

**Steps**:
1. Refactor `run_backtest()` to call `generate_signals()` for the target exposure step.
2. Or delete `generate_signals()` and document that signal logic lives in the backtest loop.

**Effort**: 🟢 1–2 hours

---

### 🟡 Issue 7: `example_strategy.py` Uses `yfinance` Directly — Bypasses `DataManager`

**Problem**: `example_strategy.py` calls `yf.download()` directly while the rest of the codebase uses `DataManager`. This is an inconsistency that creates two data paths and bypasses the caching layer.

**Steps**:
1. Rewrite `get_live_signals()` to use `DataManager.get_ohlcv()` instead of `yfinance`.
2. Or clearly label the file as a standalone prototype/example and add a comment warning not to use it in production.

**Effort**: 🟢 1 hour

---

### 🟡 Issue 8: Tier Logic Duplicated in Two Places

**Problem**: The entry tier accumulation logic (sorted tiers, `accumulated_allocation`) is copy-pasted identically in both `generate_signals()` and `run_backtest()`. If tier logic changes, both need to be updated.

```python
# Duplicated in generate_signals() AND run_backtest()
sorted_tiers = sorted(self.entry_tiers.values(), key=lambda x: x['z_threshold'])
accumulated = 0.0
for tier in sorted_tiers:
    if prev_z > tier['z_threshold']:
        accumulated += tier['allocation']
```

**Steps**:
1. Extract a private method `_compute_target_exposure(z: float) -> float` on `VIXScalingEngine`.
2. Call it from both `generate_signals()` and `run_backtest()`.

**Effort**: 🟢 30 minutes

---

### 🟢 Issue 9: No Unit Tests

**Problem**: There are no automated tests for the engine. The `docs/test-ideas.md` exists but no test files have been created.

**Steps**:
1. Create `tests/ml/pipeline/p13_bdsh/test_vix_engine.py`.
2. Test at minimum:
   - `calculate_vix_zscore()` with known input → known output
   - `calculate_atr()` with known OHLC data
   - `_compute_target_exposure()` for each tier boundary
   - Stop-loss trigger logic in `run_backtest()` (provide a synthetic DataFrame where price drops below SL)
3. Test `P13Pipeline.save_state()` / `load_state()` round-trip.

**Effort**: 🟡 1–2 days

---

### 🟢 Issue 10: Reporting is `print`/logger only — No Structured Output

**Problem**: The `--live` mode in `run_p13.py` prints a formatted string to stdout. For production use, this should be a structured dict/JSON that can be consumed by another service (e.g., to trigger Telegram alerts).

**Steps**:
1. Add `--output json` flag to `run_p13.py`.
2. Emit the live state as JSON to stdout or a file.
3. Wire to the existing `NotificationServiceClient` for Telegram delivery of daily signals.

**Effort**: 🟡 2–4 hours

---

## 5. Summary & Execution Order

| Priority | # | Issue | Effort |
|---|---|---|---|
| 🔴 | 4 | Misplaced `else` in `save_state` | 🟢 5 min |
| 🔴 | 1 | Duplicate `INITIAL_CAPITAL` / orphaned constants | 🟢 15 min |
| 🔴 | 2 | `os.makedirs` at import time + imports at end | 🟢 30 min |
| 🟡 | 8 | Tier logic duplicated | 🟢 30 min |
| 🟡 | 7 | `example_strategy.py` bypasses DataManager | 🟢 1h |
| 🟡 | 6 | `generate_signals()` unused / duplicated | 🟢 1–2h |
| 🔴 | 3 | `config.__dict__` passed as config bag | 🟡 1–2h |
| 🟡 | 5 | `VIXScalingEngine` God class + implicit marker state | 🟡 2–3h |
| 🟢 | 10 | Structured output / notification hook | 🟡 2–4h |
| 🟢 | 9 | No unit tests | 🟡 1–2 days |

**Recommended start**: Items 4 → 1 → 2 → 8 are all under 1 hour combined and eliminate real bugs. Do those first before the structural refactoring (3, 5).
