# Remaining Issues — Detailed Task Plan

## Overview

Three bodies of work remain after the P1/P2/P3 fixes applied in the previous sessions:

| Work area | Severity | Files affected |
|---|---|---|
| **A** — Extract [telegram_bot.py](file:///c:/dev/cursor/e-trading/src/telegram/telegram_bot.py) god file | 🔴 Architectural | [src/telegram/telegram_bot.py](file:///c:/dev/cursor/e-trading/src/telegram/telegram_bot.py) → new modules |
| **B** — Convert processor functions to dict-based arg access | 🟠 Correctness/Consistency | [src/telegram/telegram_bot.py](file:///c:/dev/cursor/e-trading/src/telegram/telegram_bot.py) (all `process_*` functions) |
| **C** — Review and split [screener/business_logic.py](file:///c:/dev/cursor/e-trading/src/telegram/screener/business_logic.py) | 🟠 Architectural | [src/telegram/screener/business_logic.py](file:///c:/dev/cursor/e-trading/src/telegram/screener/business_logic.py) |
| **D** — Implement scheduler alert state persistence | 🟠 Correctness | [src/scheduler/scheduler_service.py](file:///c:/dev/cursor/e-trading/src/scheduler/scheduler_service.py), DB schema |

---

## Work Area A — Extract [telegram_bot.py](file:///c:/dev/cursor/e-trading/src/telegram/telegram_bot.py) into separate modules

**Why**: The file is 1,658 lines and mixes command handlers, HTTP API routes, service init, health checks, email helpers, the bot entry point, and all business logic. This makes it untestable and hard to navigate.

**Proposed package layout** (no new Python package — just new files in `src/telegram/`):

```
src/telegram/
  telegram_bot.py          ← keep as thin entry point (~100 lines)
  handlers/
    __init__.py
    account.py             ← /register, /verify, /language, /request_approval, /info
    alerts.py              ← /alerts (all sub-actions)
    schedules.py           ← /schedules (delegates to notifications.py)
    admin.py               ← /admin
    content.py             ← /report, /screener (delegates to screener/)
    misc.py                ← /start, /help, /feedback, /feature, unknown
  api/
    __init__.py
    routes.py              ← api_send_message, api_broadcast, api_notify, api_status
    middleware.py          ← api_key_middleware (already written, just move it)
  services/
    (existing telegram_queue_processor.py, models.py stay)
  lifecycle.py             ← initialize_services(), perform_service_health_checks(),
                              get_service_instances(), get_notification_client()
```

### Tasks — Area A

- [ ] **A1** Create `src/telegram/handlers/__init__.py` (empty)
- [ ] **A2** Create `src/telegram/handlers/account.py`
  - Move: `process_info_command_immediate`, `process_register_command_immediate`, `process_verify_command_immediate`, `process_language_command_immediate`, `process_request_approval_command_immediate`
  - Move: `cmd_info`, `cmd_register`, `cmd_verify`, `cmd_language`, `cmd_request_approval` handler registrations
- [ ] **A3** Create `src/telegram/handlers/alerts.py`
  - Move: `process_alerts_command_immediate`
  - Move: `cmd_alerts` handler registration
- [ ] **A4** Create `src/telegram/handlers/admin.py`
  - Move: `process_admin_command_immediate`
  - Move: `cmd_admin` handler registration
- [ ] **A5** Create `src/telegram/handlers/content.py`
  - Move: `send_email_notification_if_requested`, `send_email_notification_with_attachments`, `extract_attachments_from_telegram_message`
  - Move: `cmd_report`, `cmd_screener` handler registrations
- [ ] **A6** Create `src/telegram/handlers/misc.py`
  - Move: `process_feedback_command_immediate`, `process_feature_command_immediate`, `process_unknown_command_immediate`
  - Move: `cmd_start`, `cmd_help`, `cmd_feedback`, `cmd_feature`, `cmd_schedules`, `unknown_command`, `all_messages` handler registrations
- [ ] **A7** Create `src/telegram/api/__init__.py` (empty)
- [ ] **A8** Create `src/telegram/api/middleware.py`
  - Move: `api_key_middleware`, `_OPEN_PATHS`
- [ ] **A9** Create `src/telegram/api/routes.py`
  - Move: all `api_*` functions, `api_app` creation and router setup
- [ ] **A10** Create `src/telegram/lifecycle.py`
  - Move: `initialize_services`, `perform_service_health_checks`, `check_telegram_service_health`, `check_indicator_service_health`, `get_service_instances`, `get_notification_client`
- [ ] **A11** Rewrite `telegram_bot.py` as thin entry point
  - Imports from the new modules
  - Globals: `bot`, `dp`, `notification_client`, `telegram_service_instance`, `indicator_service_instance`
  - `main()` function only
  - `if __name__ == "__main__": asyncio.run(main())`
- [ ] **A12** Run syntax check on all new files
- [ ] **A13** Run existing tests (`src/telegram/tests/`) to confirm no regressions

---

## Work Area B — Convert processor functions to dict-based argument access

**Why**: After P2.2 (route handlers now use `parse_command`), the downstream processor functions still extract values via positional `args[1]`, `args[2]` etc. This inconsistency means the parsing improvement is only half-applied and these functions remain fragile (changing argument order breaks them silently).

**Pattern to apply** in each function:

```python
# Before (fragile positional indexing)
email = args[1].strip()
language = args[2].strip().lower() if len(args) > 2 else "en"

# After (dict access from EnterpriseCommandParser)
parsed = parse_command(message.text)
email = parsed.args.get("email_address", "").strip()
language = parsed.args.get("language", "en").strip().lower()
```

### Tasks — Area B

- [x] **B1** `process_register_command_immediate`
- [x] **B2** `process_verify_command_immediate`
- [x] **B3** `process_language_command_immediate`
- [x] **B4** `process_alerts_command_immediate`
- [x] **B5** `process_feedback_command_immediate`
- [x] **B6** Signature updates (Option 1 applied)
- [x] **B7** Update all call sites

> [!NOTE]
> B6 offers two approaches. **Option 1** (recommended): pass the already-parsed `ParsedCommand` object from the route handler to the processor. **Option 2**: have each processor re-parse from `message.text`. Option 1 avoids double-parsing; Option 2 keeps processor functions self-contained for testing.

---

## Work Area C — Review and split [screener/business_logic.py](file:///c:/dev/cursor/e-trading/src/telegram/screener/business_logic.py)

**Why**: At ~4,000 lines (184 KB), this file is an extreme "god file" containing duplicated logic (standalone functions vs. `TelegramBusinessLogic` class methods). It mixes user management, alert lifecycle, scheduling, real-time reports, and screener orchestration.

**Proposed split in `src/telegram/screener/`**:
- `user_service.py`: Registration, verification, approval, info, language, and admin user management.
- `alert_manager.py`: Price and indicator alert lifecycle (add, edit, list, delete, pause, resume).
- `schedule_manager.py`: Scheduled report and screener lifecycle.
- `report_engine.py`: Multi-ticker technical/fundamental analysis and formatting (the core `/report` logic).
- `screener_engine.py`: Specialized screener orchestration and predefined strategy configs.
- `business_logic.py` (Reduced): Keep as the central entry point (`handle_command`) that delegates to the new service-specific handlers.

### Tasks — Area C

- [x] **C1** Audit file and produce inventory (Completed)
- [ ] **C2** Create `src/telegram/screener/user_service.py` and move user-related logic
- [ ] **C3** Create `src/telegram/screener/alert_manager.py` and move alert-related logic
- [ ] **C4** Create `src/telegram/screener/schedule_manager.py` and move schedule-related logic
- [ ] **C5** Create `src/telegram/screener/report_engine.py` and move report/analysis logic
- [ ] **C6** Create `src/telegram/screener/screener_engine.py` and move screener-specific logic
- [ ] **C7** Refactor `business_logic.py` to be a thin facade importing from the above
- [ ] **C8** Verify all imports in `notifications.py` and handlers are correct
- [ ] **C9** syntax check and basic verification

> [!IMPORTANT]
> C1-C3 are a research/planning sub-phase. Do not write code until the split plan is approved.

---

## Work Area D — Scheduler alert state persistence

**Why**: `_update_schedule_state` in `SchedulerService` is a no-op stub (logs only). Alert rearm logic (e.g., "fire once per bar") is computed in memory and lost on restart, causing alerts to re-trigger every evaluation cycle after a service restart.

### Tasks — Area D

#### D1 — Database schema
- [ ] Add `state_json TEXT` column to the `Schedule` table via a migration
  ```sql
  ALTER TABLE schedules ADD COLUMN state_json TEXT;
  ```
- [ ] Add the field to the `Schedule` SQLAlchemy model

#### D2 — Service layer
- [ ] Add `update_schedule_state(schedule_id: int, state_json: str)` method to `JobsService`
- [ ] Add `get_schedule_state(schedule_id: int) -> dict` method to `JobsService`

#### D3 — SchedulerService integration
- [ ] In `_update_schedule_state`, replace the log stub with:
  ```python
  state_json = json.dumps(state_updates, ensure_ascii=False, default=str)
  self.jobs_service.update_schedule_state(schedule_id, state_json)
  ```
- [ ] On job startup (inside `execute_job_wrapper` or `execute_alert_job`), load existing state from DB before evaluating alert rules
- [ ] Pass loaded state into `AlertEvaluator` so rearm tracking survives restarts

#### D4 — Verification
- [ ] Write a test that simulates: trigger alert → restart service → verify alert does not re-trigger immediately
