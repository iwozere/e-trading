# Tasks

## Implementation Status

### ✅ COMPLETED FEATURES

- [x] User registration flow — `/register <email>`, CSPRNG verification codes, `/verify <code>`
- [x] Account management — `/info`, `/language`, `/request_approval`
- [x] Admin commands — `/admin users|pending|approve|reject|broadcast`
- [x] Alert management — `/alerts add|list|delete|pause|resume` via `AlertManager`
- [x] Schedule management — `/schedules` via `ScheduleManager`
- [x] On-demand reports — `/report <ticker>` with optional email delivery
- [x] Screener execution — `/screener <name/json>`
- [x] Internal HTTP API — `/api/send_message`, `/api/broadcast`, `/api/notify`,
      `/api/status` (authenticated), `/api/health` (open)
- [x] Queue processor — DB-queue polling with exponential backoff and DB-first chat ID resolution
- [x] Security hardening (P1) — bot token not logged, CSPRNG codes, `/api/status` authenticated,
      admin notifications to real IDs
- [x] P2 architectural fixes — lazy business-logic facade, alert handler consolidation,
      queue-processor backoff, chat ID DB-first resolution, handlers_map removed
- [x] P3 quality fixes — bare `except:` replaced, f-string logging fixed, `sys.path.insert`,
      lazy `RecommendationEngine`

### 🔄 IN PROGRESS

- [ ] Rate limiting on `/alerts add` beyond per-user quota (token-bucket or sliding-window)
- [ ] Internationalization (i18n) — Russian language support (`ru`) is scaffolded but messages
      are still English-only

### 🚀 PLANNED ENHANCEMENTS

- [ ] Webhook mode — replace long-polling with webhook for lower latency in production
- [ ] `/alerts history` — show recently triggered alerts per user
- [ ] Inline keyboard support — confirmation dialogs for alert deletion
- [ ] `/report` caching — cache indicator computation results for repeated same-ticker requests

## Technical Debt

- [ ] `msg.from_user` is typed `Optional[User]` in aiogram; all handler dispatches do
      `msg.from_user.id` without guard — add a null check helper in `handlers/common.py`
- [ ] Pre-existing `Optional[str]` → `str` mismatches in `business_logic.py` admin handlers
      (parsed args from dict are `Optional[str]` but service methods expect `str`)
- [ ] `notifications.py` has unreachable code paths and unused variables (Pyright warnings)

## Known Issues

- `telegram_queue_processor.py` line 115/116: `.id` accessed on `Dict[str, Any]` —
  pre-existing type issue; the dict is expected to have an `id` key but Pyright cannot infer it.
- `handlers/misc.py`: `msg.from_user` can be `None` in channel posts — pre-existing across
  all handlers; needs a guard in `audit_command_wrapper`.

## Testing Requirements

- [ ] Unit tests for `get_business_logic()` cache invalidation logic
- [ ] Unit tests for queue processor exponential backoff
- [ ] Unit tests for chat ID DB-first resolution
- [ ] Integration test for `/register` → `/verify` → `/request_approval` flow
- [ ] Integration test for X-API-Key middleware (missing key, wrong key, correct key)

## Documentation Updates

- [x] `README.md` — module overview and quick start
- [x] `docs/Requirements.md` — dependencies and constraints
- [x] `docs/Design.md` — architecture and design decisions
- [x] `docs/Tasks.md` — this file
