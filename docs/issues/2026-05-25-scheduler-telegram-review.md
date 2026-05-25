# Architectural Review: `src/scheduler/` and `src/telegram/`

**Date:** 2026-05-25  
**Modules reviewed:** `src/scheduler/` · `src/telegram/`  
**Reviewer:** Claude (claude-sonnet-4-6)  
**Scope:** Design, architecture, performance, security, bad patterns

---

## Fix Status

| ID | Status | Fixed in commit |
|----|--------|----------------|
| P1-SCHED-1 | ✅ Fixed | path-traversal guard + arg validation in `_execute_data_processing_job` |
| P1-SCHED-2 | ✅ Fixed | singleton guard in `__init__` + `_deregister_instance()` class method, called in `stop()` |
| P1-TG-1 | ✅ Fixed | Bot token log line replaced with `"Bot token present"` in `telegram_bot.py` |
| P1-TG-2 | ✅ Fixed | `random.randint` → `secrets.randbelow` in `account.py`, `user_service.py`, `create_admin.py` |
| P1-TG-3 | ✅ Fixed | `/api/status` removed from `_OPEN_PATHS`; new unauthenticated `/api/health` probe added |
| P1-TG-4 | ✅ Fixed | Magic `"admin"` recipient replaced with `telegram_svc.get_admin_user_ids()` loop |

---

## Summary

Both modules are structurally sound — the scheduler follows a clean lifecycle pattern with DI, and the Telegram bot correctly separates handlers, business logic, and a service layer. However, several **critical security** issues need immediate attention before production hardening, plus a cluster of architectural weaknesses that cause reliability and maintainability debt.

---

## P1 — Critical

### [P1-SCHED-1] Subprocess injection via unvalidated `script_path` / `script_args`
**File:** `src/scheduler/scheduler_service.py:636–667`

`task_params` originates from the database (written by the user or admin). `script_path` and `script_args` are passed directly to `asyncio.create_subprocess_exec` with no allow-list, no path canonicalization, and no sandboxing. A malicious or compromised DB row can execute arbitrary code with the scheduler process's privileges.

```python
# CURRENT — dangerous
cmd = [python_executable, str(script_full_path)] + script_args
```

**Fix:**
- Maintain an explicit allow-list of permitted scripts in config.
- Resolve `script_full_path` relative to `PROJECT_ROOT` and assert it stays inside the allowed subtree (`resolved.is_relative_to(PROJECT_ROOT / "scripts")`).
- Validate `script_args` against a schema before use.
- Consider running scripts in a restricted subprocess environment (`subprocess.PIPE` stdin, environment whitelist).

---

### [P1-SCHED-2] Global `_service_instance` is overwritten on every `SchedulerService` instantiation
**File:** `src/scheduler/scheduler_service.py:117–118`

```python
global _service_instance
_service_instance = self
```

Any second instantiation (e.g., test setup, CLI `reload` command) silently replaces the singleton. If `execute_job_wrapper` fires between instantiation and the first `start()`, it will call the new (not-yet-started) instance. This is a latent race condition.

**Fix:** Guard with `if _service_instance is None` or use an explicit `register_instance()` class method with a `RuntimeError` on double-registration.

---

### [P1-TG-1] Bot token partially written to log
**File:** `src/telegram/telegram_bot.py:59`

```python
_logger.info("Bot token: %s…", TELEGRAM_BOT_TOKEN[:10])
```

Log aggregators (ELK, Datadog, Loki) store this permanently. Ten characters is enough for enumeration attacks. Remove this line entirely; the token's presence can be inferred from bot startup success.

---

### [P1-TG-2] Verification code generated with non-CSPRNG `random`
**File:** `src/telegram/handlers/account.py:94`

```python
verification_code = f"{random.randint(100000, 999999):06d}"
```

Python's `random` is a Mersenne Twister — not cryptographically secure. An attacker who observes enough codes can reconstruct the RNG state and predict future codes.

**Fix:**
```python
import secrets
verification_code = f"{secrets.randbelow(900000) + 100000:06d}"
```

Duplicated in `src/telegram/screener/user_service.py:70` — fix both.

---

### [P1-TG-3] `/api/status` is unauthenticated and leaks internal system data
**File:** `src/telegram/api/middleware.py:11`, `src/telegram/api/routes.py:139–167`

`_OPEN_PATHS = {"/api/status", "/api/test"}` exempts the status endpoint from the API key check. The response includes:
- Adapter names loaded in `IndicatorService`
- Total registered user count
- Notification queue stats
- Full service health booleans

This is a reconnaissance goldmine. Unauthenticated health probes should return only `{"status": "ok"}`.

**Fix:** Move detailed status behind the API key. Expose a minimal `GET /health` → `200 OK` or `503` for load balancers.

---

### [P1-TG-4] Admin approval notification sent to magic string `"admin"`
**File:** `src/telegram/handlers/account.py:251`

```python
await client.send_notification(
    ...
    recipient_id="admin",
)
```

`"admin"` is not a valid Telegram chat ID or numeric user ID. The notification will silently fail (notification service can't resolve it), so **admins are never notified of approval requests**. The user sees a success message but the request goes unreviewed.

**Fix:** Resolve admin chat IDs at startup from the DB (`telegram_service.list_admin_users()`) or store them in config. Notify all admins by iterating their real chat IDs.

---

## P2 — High

### [P2-SCHED-1] DB change listener uses deprecated `asyncio.get_event_loop()`
**File:** `src/scheduler/scheduler_service.py:882`

```python
loop = asyncio.get_event_loop()
await conn.add_listener('scheduler_updates',
    lambda *args: loop.call_soon_threadsafe(asyncio.ensure_future, self.reload_schedules()))
```

`asyncio.get_event_loop()` is deprecated in Python 3.10+ (raises `DeprecationWarning` if no running loop). `asyncio.ensure_future()` is also deprecated in favor of `asyncio.create_task()`.

**Fix:**
```python
loop = asyncio.get_running_loop()
await conn.add_listener('scheduler_updates',
    lambda *args: loop.call_soon_threadsafe(
        asyncio.create_task, self.reload_schedules()
    ))
```

---

### [P2-SCHED-2] `reload_schedules()` has no debounce — burst DB NOTIFYs trigger O(N) reloads
**File:** `src/scheduler/scheduler_service.py:885–887`

Every `NOTIFY scheduler_updates` event immediately fires a full reload: DB query + APScheduler job removal + re-registration. Under any batch DB operation, a burst of 100 NOTIFYs triggers 100 sequential reloads. This is a DoS vector against the scheduler itself.

**Fix:** Debounce with a short (2–5s) `asyncio.Event` + `asyncio.sleep` pattern:

```python
async def _debounced_reload(self, delay: float = 3.0) -> None:
    """Deduplicate rapid-fire reload requests."""
    self._reload_pending = True
    await asyncio.sleep(delay)
    if self._reload_pending:
        self._reload_pending = False
        await self.reload_schedules()
```

---

### [P2-SCHED-3] No timeout applied to job execution despite config field `max_evaluation_time`
**File:** `src/scheduler/scheduler_service.py:453–555`, `src/scheduler/config.py:54`

`AlertConfig.max_evaluation_time = 120` is defined but never applied. `_execute_job` and `_execute_alert_job` have no `asyncio.wait_for(..., timeout=...)` wrapper. A hung data provider call will block the event loop slot for that job indefinitely, eventually starving other jobs.

**Fix:**
```python
result = await asyncio.wait_for(
    self._execute_alert_job(schedule, run_record),
    timeout=self.config.alert.max_evaluation_time
)
```
Wrap at the `_execute_job` level so all job types are covered.

---

### [P2-SCHED-4] Non-atomic job reload — window where all jobs are absent
**File:** `src/scheduler/scheduler_service.py:229–234`

```python
await self._clear_all_jobs()           # APScheduler is now empty
count = await self._load_and_register_schedules()  # may fail
```

If `_load_and_register_schedules()` raises (DB down, schema error), the scheduler runs with zero registered jobs until restart. During that window, scheduled alerts silently miss their fire times.

**Fix:** Load the new schedule list first, then atomically replace in APScheduler. Use `replace_existing=True` (already done for individual jobs) and only call `remove_job` for IDs no longer in the DB rather than wiping all jobs.

---

### [P2-TG-1] Two parallel implementations of alert business logic
**File:** `src/telegram/handlers/alerts.py` vs `src/telegram/screener/alert_manager.py`

`handlers/alerts.py` directly instantiates `AlertsService`, `DataManager`, `JobsService` on every command, bypassing the `TelegramBusinessLogic` facade. `screener/alert_manager.py` routes through `telegram_service`. The two paths have divergent logic (e.g., `alert_manager.py` supports `add_indicator`, `pause`, `resume`; `handlers/alerts.py` does not). Users can end up with inconsistent behavior depending on code path.

**Fix:** Remove the direct-service path from `handlers/alerts.py`. Route everything through `TelegramBusinessLogic` / `AlertManager`.

---

### [P2-TG-2] `handle_command()` facade re-instantiates all sub-services on every call
**File:** `src/telegram/screener/business_logic.py:139–143`

```python
async def handle_command(parsed: ParsedCommand) -> Dict[str, Any]:
    ts, ids = get_service_instances()
    facade = TelegramBusinessLogic(ts, ids)  # creates UserService, AlertManager, etc. every call
    return await facade.handle_command(parsed)
```

`TelegramBusinessLogic.__init__` instantiates five service objects on each invocation. Under load this is unnecessary object churn and obscures the single shared instance pattern.

**Fix:** Cache the facade as a module-level singleton, recreated only when service instances change.

---

### [P2-TG-3] `TelegramQueueProcessor` has no circuit breaker — hammers DB on outage
**File:** `src/telegram/services/telegram_queue_processor.py:119–122`

On any exception, the processor sleeps `poll_interval * 2` (10s) and retries immediately with no cap. A sustained DB outage produces a log storm and connection exhaustion.

**Fix:** Implement exponential backoff with a ceiling:
```python
backoff = min(self.poll_interval * (2 ** consecutive_errors), 300)
await asyncio.sleep(backoff)
```

---

### [P2-TG-4] Fragile Telegram chat ID resolution heuristic
**File:** `src/telegram/services/telegram_queue_processor.py:234–244`

```python
if recipient_id_int < 1000000:
    # treat as internal User ID, look up telegram_chat_id
```

The threshold `1_000_000` is arbitrary and undocumented. Real Telegram user IDs start at ~100M for bots; group chat IDs are negative. As the internal user table grows past 1M rows, this will silently mis-route messages by treating internal user IDs as direct Telegram chat IDs.

**Fix:** Store and communicate the ID type explicitly (e.g., `recipient_type: "user_id" | "telegram_chat_id"` in the message metadata), or always store `telegram_chat_id` at message creation time.

---

### [P2-TG-5] No rate limiting on `/alerts add`
**File:** `src/telegram/handlers/alerts.py:56–89`

`/register` checks `count_codes_last_hour()`, but `/alerts add` has no throttle. A user can flood the DB with thousands of alerts. This also has a direct cost implication at evaluation time (every alert fires a data-provider query).

**Fix:** Add a per-user alert limit check before creation, reusing the existing `set_user_limit("alerts", N)` mechanism. Return an error if the user is at or above their limit.

---

### [P2-TG-6] `misc.py` contains a redundant manual dispatch table
**File:** `src/telegram/handlers/misc.py:132–146`

`unknown_command` duplicates aiogram's command routing with a hand-rolled `handlers_map` dict. New commands must be added in two places (aiogram `register()` + this dict), and the two will silently diverge. The case-insensitive matching also conflicts with Telegram's documented case-insensitive command handling.

**Fix:** Remove the manual dispatch table. Rely on aiogram's `Command(ignore_case=True)` filter, which handles case-insensitivity natively.

---

## P3 — Medium

### [P3-SCHED-1] `MessagePriority` enum re-defined inside scheduler module
**File:** `src/scheduler/scheduler_service.py:42–47`

An identical `MessagePriority` enum already exists in `src/notification/service/client.py`. Re-defining it locally creates import confusion and drift when the canonical values change.

**Fix:** Import from `src.notification.service.client`.

---

### [P3-SCHED-2] `_send_notification` is a 200-line God method
**File:** `src/scheduler/scheduler_service.py:1052–1270`

The method simultaneously: formats a multi-section Markdown message, categorises technical indicators into trend/momentum/volatility buckets, builds a priority mapping, and writes a DB record. This violates SRP and makes testing impossible without a full DB.

**Fix:** Extract to a `NotificationFormatter` class with a separate `format_alert_message(notification_data) -> str` method. Keep `_send_notification` only for the DB write.

---

### [P3-SCHED-3] `coalesce=False` with no missed-run cap
**File:** `src/scheduler/scheduler_service.py:312–316`

With `coalesce=False`, every missed execution is replayed individually after downtime. After a 24-hour outage, a 1-minute alert schedule generates 1,440 catch-up executions, hammering the data provider. The comment acknowledges the trade-off but doesn't impose any guardrail.

**Fix:** Add a `max_catchup_runs: int = 5` job default (`misfire_grace_time`) or set a per-job `max_instances` guard so APScheduler naturally limits concurrent replay.

---

### [P3-SCHED-4] Module-level `SchedulerServiceConfig()` instantiation with side effects on import
**File:** `src/scheduler/config.py:181`

```python
config = SchedulerServiceConfig()
```

Instantiating at module level triggers `get_database_url()`, env var reads, and filesystem checks every time any module imports `src.scheduler.config`. This makes unit testing and mocking painful.

**Fix:** Remove the module-level instance; let callers construct `SchedulerServiceConfig()` explicitly. If a shared default is needed, use a lazy getter:
```python
_config: Optional[SchedulerServiceConfig] = None
def get_config() -> SchedulerServiceConfig:
    global _config
    if _config is None:
        _config = SchedulerServiceConfig()
    return _config
```

---

### [P3-TG-1] Bare `except:` and one-liner exception handlers in `business_logic.py`
**File:** `src/telegram/screener/business_logic.py:106, 116`

```python
except: return {"status": "error", "message": "Failed to save feedback."}
```

Bare `except:` also catches `KeyboardInterrupt`, `SystemExit`, and `GeneratorExit`. Per project CLAUDE.md §7, catch specific exceptions. Also, one-liner except bodies obscure stack traces in logs.

**Fix:**
```python
except Exception:
    _logger.exception("Error saving feedback for user %s:", tid)
    return {"status": "error", "message": "Failed to save feedback."}
```

---

### [P3-TG-2] f-string used in exception log call (violates project convention)
**File:** `src/telegram/screener/business_logic.py:67`

```python
_logger.exception(f"Error handling command {parsed.command}")
```

Project CLAUDE.md §3.2 requires lazy `%s` formatting in all log calls. f-strings evaluate eagerly even when the log level is disabled.

**Fix:**
```python
_logger.exception("Error handling command %s", parsed.command)
```

---

### [P3-TG-3] `sys.path.append` instead of `sys.path.insert` in `notifications.py`
**File:** `src/telegram/screener/notifications.py:5`

```python
sys.path.append(str(PROJECT_ROOT))
```

`append` adds to the end of `sys.path`; if a same-named module exists elsewhere on the path, it will shadow the intended one. Project convention (CLAUDE.md §2.1) requires `sys.path.insert(0, str(PROJECT_ROOT))`. Also applies to other modules using `sys.path.append`.

---

### [P3-TG-4] `RecommendationEngine` instantiated at module import level
**File:** `src/telegram/screener/notifications.py:15`

```python
recommendation_engine = RecommendationEngine()
```

Module-level object instantiation triggers side effects (potential DB/file I/O, heavy imports) at import time for every module that imports `notifications`. Makes testing with mocks harder and slows cold-start.

**Fix:** Move to lazy initialization inside the functions that use it, or inject via parameter.

---

## P4 — Low

### [P4-SCHED-1] CLI uses `print()` instead of project logger
**File:** `src/scheduler/cli.py:34–36, 55–69, 85–86, 99–119`

`print()` calls throughout the CLI violate project CLAUDE.md §3 ("Never use print()"). CLI output is swallowed by process supervisors and log aggregators.

**Fix:** Replace `print()` with `_logger.info()` / `_logger.warning()` / `_logger.error()`.

---

### [P4-SCHED-2] `src/scheduler/` missing `README.md` at module root
**File:** `src/scheduler/`

Per CLAUDE.md §10 and §12.4, each `src/` submodule must include a `README.md` at root level. The scheduler has docs in `src/scheduler/docs/` (MONITORING, TROUBLESHOOTING, etc.) but no module-level `README.md`.

---

### [P4-TG-1] `src/telegram/` missing top-level `README.md`
**File:** `src/telegram/`

Same as above — `src/telegram/` lacks a module-level `README.md`.

---

### [P4-TG-2] Missing top-level `docs/` subfolder for `src/telegram/`
**File:** `src/telegram/`

The screener sub-package (`src/telegram/screener/docs/`) has docs but the parent `src/telegram/` has no `docs/Requirements.md`, `docs/Design.md`, or `docs/Tasks.md`. Per CLAUDE.md §10, every `src/` module needs this structure.

---

## Issue Priority Table

| ID | Module | Severity | Category | One-line description |
|----|--------|----------|----------|---------------------|
| P1-SCHED-1 | scheduler | **Critical** | Security | Subprocess injection via unvalidated `script_path` |
| P1-SCHED-2 | scheduler | **Critical** | Reliability | Global `_service_instance` overwritten on re-instantiation |
| P1-TG-1 | telegram | **Critical** | Security | Bot token (first 10 chars) written to log |
| P1-TG-2 | telegram | **Critical** | Security | Verification code uses non-CSPRNG `random` |
| P1-TG-3 | telegram | **Critical** | Security | `/api/status` unauthenticated, leaks internals |
| P1-TG-4 | telegram | **Critical** | Reliability | Admin approval notif sent to magic string `"admin"` |
| P2-SCHED-1 | scheduler | High | Reliability | DB listener uses deprecated `get_event_loop()` / `ensure_future` |
| P2-SCHED-2 | scheduler | High | Reliability | No debounce on `reload_schedules()` — burst NOTIFY DoS |
| P2-SCHED-3 | scheduler | High | Reliability | `max_evaluation_time` config field never applied |
| P2-SCHED-4 | scheduler | High | Reliability | Non-atomic reload — brief window with zero jobs |
| P2-TG-1 | telegram | High | Architecture | Two parallel `/alerts` implementations with divergent logic |
| P2-TG-2 | telegram | High | Performance | `handle_command()` re-creates all sub-services per call |
| P2-TG-3 | telegram | High | Reliability | Queue processor has no backoff ceiling — DB outage storm |
| P2-TG-4 | telegram | High | Reliability | Fragile `< 1_000_000` heuristic for chat ID resolution |
| P2-TG-5 | telegram | High | Security | No rate limiting on `/alerts add` |
| P2-TG-6 | telegram | High | Architecture | `misc.py` manual dispatch table duplicates aiogram routing |
| P3-SCHED-1 | scheduler | Medium | Architecture | `MessagePriority` re-defined, already in notification module |
| P3-SCHED-2 | scheduler | Medium | Design | `_send_notification` God method (200+ lines, violates SRP) |
| P3-SCHED-3 | scheduler | Medium | Reliability | `coalesce=False` without missed-run replay cap |
| P3-SCHED-4 | scheduler | Medium | Architecture | Module-level `SchedulerServiceConfig()` — import side effects |
| P3-TG-1 | telegram | Medium | Style/Safety | Bare `except:` in `business_logic.py` — swallows signals |
| P3-TG-2 | telegram | Medium | Style | f-string in `_logger.exception()` — violates project convention |
| P3-TG-3 | telegram | Medium | Style | `sys.path.append` instead of `sys.path.insert` |
| P3-TG-4 | telegram | Medium | Architecture | `RecommendationEngine` instantiated at module import level |
| P4-SCHED-1 | scheduler | Low | Style | CLI uses `print()` instead of project logger |
| P4-SCHED-2 | scheduler | Low | Docs | Missing `README.md` at `src/scheduler/` root |
| P4-TG-1 | telegram | Low | Docs | Missing `README.md` at `src/telegram/` root |
| P4-TG-2 | telegram | Low | Docs | Missing `docs/` subfolder structure in `src/telegram/` |

---

## Recommended Fix Order

**Sprint 1 (block deployment):**
- P1-TG-1: Remove bot token logging
- P1-TG-2: Switch to `secrets` module for verification codes
- P1-TG-3: Put `/api/status` behind API key
- P1-TG-4: Fix admin notification recipient resolution
- P1-SCHED-1: Add script allow-list validation
- P1-SCHED-2: Guard global `_service_instance` assignment

**Sprint 2 (reliability hardening):**
- P2-SCHED-1: `get_running_loop()` + `create_task()`
- P2-SCHED-2: Add reload debounce
- P2-SCHED-3: Wrap job execution in `asyncio.wait_for`
- P2-SCHED-4: Make reload atomic
- P2-TG-3: Add backoff ceiling to queue processor
- P2-TG-5: Add rate limiting to `/alerts add`

**Sprint 3 (architecture cleanup):**
- P2-TG-1: Consolidate alert implementation paths
- P2-TG-2: Cache `TelegramBusinessLogic` singleton
- P2-TG-4: Explicit recipient type metadata
- P2-TG-6: Remove manual dispatch table in `misc.py`
- P3-SCHED-2: Extract `NotificationFormatter`
- P3-SCHED-4: Remove module-level config instantiation

**Sprint 4 (housekeeping):**
- P3-SCHED-1, P3-TG-1, P3-TG-2, P3-TG-3, P3-TG-4
- P4-SCHED-1 (print → logger in CLI)
- P4-SCHED-2, P4-TG-1, P4-TG-2 (add missing docs)
