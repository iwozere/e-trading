# Design

## Purpose

The `src/telegram/` module is the primary user-facing interface for the e-trading platform.
It bridges end-users (via Telegram) with the platform's data, alerts, schedules, and reports,
and exposes an authenticated internal HTTP API for other services (scheduler, admin scripts)
to push notifications.

## Architecture

### High-Level Architecture

```
User (Telegram)
       │
       ▼
  telegram_bot.py          ← aiogram Bot + Dispatcher, long-polling
       │
       ├── handlers/       ← Command routing layer (one file per concern)
       │     ├── account.py    /register /verify /info /language /request_approval
       │     ├── alerts.py     /alerts — delegates to AlertManager
       │     ├── content.py    /report /screener
       │     ├── admin.py      /admin …
       │     └── misc.py       /help /feedback and unknown-command handler
       │
       ├── screener/       ← Business-logic layer
       │     ├── business_logic.py   TelegramBusinessLogic facade (cached singleton)
       │     ├── user_service.py     User state, verification, admin checks
       │     ├── alert_manager.py    Alert CRUD with quota enforcement
       │     ├── schedule_manager.py Schedule CRUD
       │     ├── report_engine.py    On-demand technical analysis
       │     ├── screener_engine.py  Screener execution
       │     └── notifications.py   Notification helpers, lazy RecommendationEngine
       │
       ├── services/       ← Background workers
       │     └── telegram_queue_processor.py   DB-queue polling + delivery
       │
       ├── api/            ← Internal HTTP API (aiohttp)
       │     ├── middleware.py   X-API-Key authentication
       │     └── routes.py       /api/send_message, /api/broadcast, /api/notify,
       │                         /api/status (authenticated), /api/health (open)
       │
       └── lifecycle.py    ← Singleton service instances + notification client
```

### Command Routing Flow

1. aiogram receives a Telegram update and matches it to a `Command()` filter.
2. The handler in `handlers/` calls `audit_command_wrapper()` (logging, user auth check).
3. The processor function delegates to the appropriate service via `get_service_instances()`
   or the cached `get_business_logic()` facade.
4. Results are sent back to the user via `message.reply()`.

### Internal HTTP API Flow

1. The scheduler (or admin script) POSTs to `/api/notify` with an `X-API-Key` header.
2. `api/middleware.py` validates the key; rejects with 401 if missing or wrong.
3. The route handler calls `get_notification_client()` and enqueues the notification.
4. The queue processor picks it up and delivers it via aiogram.

## Component Design

### `lifecycle.py` — Service Singletons

Holds module-level `_telegram_service`, `_indicator_service`, and `_notification_client`
references.  `get_service_instances()` returns them; they are initialized once in
`initialize_services()` and reused for the lifetime of the process.

### `business_logic.py` — Lazy Cached Facade

`get_business_logic()` returns a cached `TelegramBusinessLogic` instance, rebuilding only
when the underlying service instances change identity.  This avoids constructing 5 sub-service
objects on every command invocation.

### `telegram_queue_processor.py` — Outbound Queue Delivery

Polls the notification DB every `poll_interval` seconds for pending Telegram messages.
Uses exponential backoff (ceiling 300 s) on consecutive errors.  Chat ID resolution is
DB-first: looks up `telegram_chat_id` via `users_service`; falls back to treating
`recipient_id` as a direct chat ID.

### `api/middleware.py` — X-API-Key Guard

All routes except `/api/health` and `/api/test` require the `X-API-Key` header to match
the configured secret.  `/api/health` returns only `{"status": "ok"}` — no internal data.

## Design Decisions

| Decision | Rationale |
|---|---|
| aiogram 3.x | Modern async-native Telegram framework; `Command(ignore_case=True)` handles case-insensitivity natively |
| aiohttp for internal API | Lightweight; already in the dependency tree; co-runs in the same event loop |
| DB-first chat ID resolution | Eliminates the fragile `< 1_000_000` heuristic; correct as user table grows |
| Lazy `get_business_logic()` | Avoids per-request object churn while remaining stateless from the caller's perspective |
| `secrets.randbelow()` for codes | CSPRNG compliance; Mersenne Twister (`random`) is not suitable for security-sensitive codes |
| Bot token never logged | Prevents token leakage into log aggregators (ELK, Datadog, Loki) |

## Integration Patterns

- **Inbound (Telegram → Bot)**: long-polling via aiogram's `Dispatcher.start_polling()`.
- **Outbound (Bot → User)**: direct `message.reply()` for command responses; queue processor
  for scheduled/async notifications.
- **Service-to-service (Scheduler → Bot)**: HTTP POST to `/api/notify` with `X-API-Key`.
- **Error handling**: every handler is wrapped in `audit_command_wrapper()` which catches
  unhandled exceptions and replies with a generic error message while logging the full
  traceback at ERROR level.
