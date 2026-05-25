# Architecture Review: `src/notification/` & `src/screeners/`

**Date:** 2026-05-25  
**Reviewer:** AI Architecture Review (Claude Sonnet 4.6)  
**Scope:** `src/notification/` · `src/screeners/`  
**Status:** 🔄 In progress

---

## Commits

| SHA | Message |
|-----|---------|
| _(pending)_ | Fix critical and high issues in notification and screeners modules |

---

## 🔴 P1 — Critical

### Issue 1 — `await` inside `threading.RLock` in async service classes

**Files:** `src/notification/service/rate_limiter.py`, `src/notification/service/batch_processor.py`  
**Severity:** 🔴 Critical  
**Impact:** `threading.RLock` is a blocking OS primitive that does not yield to the asyncio event loop. Awaiting inside it is correct only by coincidence of being single-threaded; it breaks under `run_in_executor`, in tests that use multiple threads, and silently serialises all lock-holders instead of cooperatively suspending them. Six service classes share this anti-pattern.

**Root cause — rate_limiter.py:**
```python
# _get_or_create_bucket() — await inside threading.RLock
with self._lock:
    bucket = await self._load_bucket_from_db(user_id, channel)  # ← blocks lock
```

**Root cause — batch_processor.py:**
```python
# add_message() — chained awaits under threading.RLock
with self._lock:
    await self._process_pending_messages(channel)
    # which itself awaits _complete_batch() → _batch_processor_callback()
```

**Fix:** Replace every `threading.RLock` in async classes with `asyncio.Lock` and use `async with` throughout.

**Status:** ✅ Fixed — see commit details below.

---

### Issue 2 — Synchronous blocking CPU work (`cerebro.run`) on the asyncio event loop

**File:** `src/screeners/logic/strategy_bridge.py:75`  
**Severity:** 🔴 Critical  
**Impact:** `backtrader.Cerebro.run()` is a long, synchronous, CPU-bound call executed directly inside an `async` coroutine. With `asyncio.Semaphore(50)` concurrency, up to 50 coroutines all block the event loop simultaneously — notifications, health checks, the broker keepalive, and all other async operations stall for the full scan duration.

**Root cause:**
```python
# strategy_bridge.run() — sync call inside async context (called from _process_symbol)
results = cerebro.run(runonce=True, preload=True)  # synchronous, CPU-bound
```

**Fix:** Run `bridge.run()` inside a `ProcessPoolExecutor` via `loop.run_in_executor()`. A process pool (not thread pool) is required because Backtrader is CPU-bound and the GIL prevents true thread parallelism.

**Status:** ✅ Fixed — see commit details below.

---

## 🟠 P2 — High

### Issue 3 — `asyncio.create_task()` called inside `threading.RLock` with stale-capture race

**File:** `src/notification/service/rate_limiter.py:313`  
**Severity:** 🟠 High  
**Impact:** `asyncio.create_task()` is called while holding the threading lock. The task captures the mutable `bucket` object by reference; by the time the task runs, `bucket.tokens` may have been mutated again, saving a wrong value to the DB.

**Root cause:**
```python
with self._lock:
    bucket.consume(tokens)
    asyncio.create_task(self._save_bucket_to_db(user_id, channel, bucket))
    # bucket is mutated further on next check_rate_limit before task runs
```

**Fix:** Move the `create_task` outside the `async with` lock scope and pass the token count as a scalar argument, not the mutable bucket object. Addressed as part of Issue 1 fix (lock migration).

**Status:** ✅ Fixed — resolved during Issue 1 fix.

---

### Issue 4 — Unbounded `_violations` list — memory leak in `RateLimiter`

**File:** `src/notification/service/rate_limiter.py`  
**Severity:** 🟠 High  
**Impact:** Every HIGH/CRITICAL message that bypasses rate limiting appends a `RateLimitViolation` to the in-memory `self._violations` list. `cleanup_old_violations()` is never called automatically, so the list grows without bound in long-running service processes.

**Fix:** Convert `self._violations` to a `collections.deque(maxlen=10000)` to enforce a hard cap, and add a call to `cleanup_old_violations()` in the `MessageProcessor._cleanup_worker` hourly pass.

**Status:** ✅ Fixed — see commit details below.

---

### Issue 5 — Three incompatible `MessagePriority` / `MessageType` enum definitions

**Files:** `src/notification/model.py`, `src/notification/service/client.py`, `src/data/db/models/model_notification.py`  
**Severity:** 🟠 High  
**Impact:** `isinstance()` checks fail across module boundaries. `model.py:33` mistakenly aliases `MessagePriority = NotificationType` (a type aliased to a *different* enum). The `URGENT` value exists only in `model.py`, silently dropped elsewhere.

**Fix:** `service/client.py` re-exports from `notification/model.py` instead of defining its own enums. The `MessagePriority = NotificationType` alias in `model.py` is corrected.

**Status:** ✅ Fixed — see commit details below.

---

### Issue 6 — Naive `datetime.now()` (no timezone) in screeners

**Files:** `src/screeners/ibkr_screener_service.py`, `src/screeners/logic/strategy_bridge.py`  
**Severity:** 🟠 High  
**Impact:** Produces timezone-naive datetimes. Mixing with the timezone-aware datetimes from the notification service and database layer raises `TypeError` at comparison boundaries and stores incorrect timestamps.

**Fix:** Replace all `datetime.now()` calls with `datetime.now(timezone.utc)`.

**Status:** ✅ Fixed — see commit details below.

---

### Issue 7 — Results directory is CWD-relative in screener

**File:** `src/screeners/ibkr_screener_service.py:44`  
**Severity:** 🟠 High  
**Impact:** `os.path.join('results', 'screeners', 'ibkr')` resolves against the process's working directory at runtime. Running from any directory other than project root silently creates a second `results/` tree.

**Root cause:**
```python
self.results_dir = os.path.join('results', 'screeners', 'ibkr')
```

**Fix:** Derive path from `__file__` using `pathlib`: `Path(__file__).resolve().parents[2] / "results" / "screeners" / "ibkr"`. Also migrate `os.path` calls to `pathlib` per convention.

**Status:** ✅ Fixed — see commit details below.

---

## 🟡 P3 — Medium (tracked, not in this PR)

| # | Description | File |
|---|---|---|
| 8 | `process_database_message` duplicates `_process_single_message` (~140 lines DRY violation) | `service/processor.py` |
| 9 | Hardcoded `enabled_channels = ['email']` — architecture in a comment | `service/processor.py:145` |
| 10 | `donotshare.py` imported at module level in `service/config.py` — blocks testing | `service/config.py:13` |
| 11 | No signal de-duplication — identical alerts fire every scan interval | `ibkr_screener_service.py` |
| 12 | `IBKRDownloader()` created without configuration — hidden coupling to donotshare | `ibkr_screener_service.py:41` |

---

## 🔵 P4 — Low / Convention (tracked, not in this PR)

| # | Description | File |
|---|---|---|
| 13 | Dead code (unreachable lines) after `return` in `get_multiprocessing_logger()` | `logger.py:507` |
| 14 | `print_log()` function uses `print()` directly — violates no-print convention | `logger.py:99` |
| 15 | `os.path` should be `pathlib` throughout screeners | `ibkr_screener_service.py`, `discovery/static.py` |
| 16 | Missing `__init__.py` files in `src/screeners/` packages | `screeners/`, `screeners/logic/`, `screeners/discovery/` |
| 17 | No `tests/` directory in `src/screeners/` | `screeners/` |
| 18 | `ContextAwareLogger` clones `RotatingFileHandler` instances to same file — race on rotation | `logger.py:527` |
