# Requirements

## Python Dependencies

- `aiogram` >= 3.0 — Telegram Bot API framework (async, Dispatcher-based routing)
- `aiohttp` >= 3.x — Internal REST API server
- `asyncpg` — PostgreSQL async driver (used indirectly via DB services)
- `sqlalchemy` — ORM layer (used via `src.data.db`)

## Internal Module Dependencies

- `src.data.db.services.telegram_service` — User records, verification codes, admin IDs
- `src.data.db.services.users_service` — User notification channel resolution
- `src.data.db.services.jobs_service` — Schedule CRUD (via ScheduleManager)
- `src.data.db.services.notification_service` — Notification message persistence
- `src.notification.service.client` — `NotificationClient` for delivery (Telegram, email)
- `src.notification.logger` — Shared logging setup
- `src.model.telegram_bot` — `ParsedCommand` dataclass
- `src.common.alerts.alert_evaluator` — Alert condition evaluation
- `src.common.alerts.cron_parser` — Cron expression validation
- `src.common.recommendation.engine` — `RecommendationEngine` for report annotations
- `src.scheduler` — Receives inbound HTTP calls from the scheduler service

## External Services

- **Telegram Bot API** — Primary user interface; requires a valid bot token from @BotFather.
- **PostgreSQL** — Stores user records, verification codes, alert definitions, schedule state,
  and queued notification messages.
- **Notification Service** — Downstream service for email and Telegram delivery.

## Security Requirements

- Bot token must **never** appear in logs (use `"Bot token present"` log message only).
- Verification codes generated with `secrets.randbelow()` (CSPRNG).
- Internal HTTP API protected by `X-API-Key` header for all endpoints except `/api/health`.
- Admin notifications dispatched to real Telegram user IDs (resolved from DB), never magic strings.

## Performance Requirements

- The `TelegramBusinessLogic` facade is cached per service-instance identity
  (`get_business_logic()`) to avoid per-request object construction.
- The queue processor uses exponential backoff (max 300 s) to avoid hammering the DB during
  outages.
- `RecommendationEngine` is lazy-initialized on first use to avoid heavy import-time overhead.

## System Requirements

- Python 3.10+ (uses `asyncio.get_running_loop()`, `asyncio.create_task()`).
- PostgreSQL 13+ with `LISTEN/NOTIFY` support.
- Network access to `api.telegram.org` from the deployment environment.
