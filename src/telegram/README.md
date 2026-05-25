# Telegram Module

## Overview

The `src/telegram/` module is the Telegram bot interface for the e-trading platform.
It handles user registration and authentication, command routing, alert and schedule
management, screener execution, report delivery, and admin operations — all via the
Telegram messaging API (aiogram 3.x).

## Features

- **User lifecycle** — `/register`, `/verify` (CSPRNG codes), `/request_approval`.
- **Account management** — `/info`, `/language`, admin approval workflow.
- **Alert management** — `/alerts add|list|delete|pause|resume` via `AlertManager`.
- **Schedule management** — `/schedules` via `ScheduleManager`.
- **Screener & reports** — `/screener`, `/report` with optional email delivery.
- **Internal HTTP API** — aiohttp endpoints (`/api/send_message`, `/api/broadcast`,
  `/api/notify`, `/api/status`, `/api/health`) with X-API-Key authentication.
- **Queue processor** — polls DB for pending Telegram messages; exponential backoff
  on errors; DB-first chat-ID resolution.
- **Admin commands** — `/admin users|pending|approve|reject|broadcast`.

## Quick Start

```python
# Start the bot (normally invoked from the service entry point)
from src.telegram.telegram_bot import main
import asyncio
asyncio.run(main())
```

## Module Layout

```
src/telegram/
├── telegram_bot.py             # Entry point: Bot, Dispatcher, polling loop
├── lifecycle.py                # Singleton service instances + notification client
├── command_parser.py           # Command text → ParsedCommand parser
├── api/
│   ├── middleware.py           # X-API-Key authentication middleware
│   └── routes.py               # aiohttp REST endpoints
├── handlers/
│   ├── common.py               # audit_command_wrapper, shared helpers
│   ├── account.py              # /info /register /verify /language /request_approval
│   ├── alerts.py               # /alerts — delegates to AlertManager
│   ├── content.py              # /report /screener
│   ├── admin.py                # /admin …
│   └── misc.py                 # /help /feedback /feature and unknown-command handler
├── screener/
│   ├── business_logic.py       # TelegramBusinessLogic facade + lazy get_business_logic()
│   ├── user_service.py         # UserService — user state, verification, admin
│   ├── alert_manager.py        # AlertManager — full alert CRUD with quota enforcement
│   ├── schedule_manager.py     # ScheduleManager — schedule CRUD
│   ├── report_engine.py        # ReportEngine — on-demand technical analysis
│   ├── screener_engine.py      # ScreenerEngine — screener execution
│   ├── notifications.py        # Notification helpers; lazy RecommendationEngine
│   └── tests/                  # Screener-layer unit tests
├── services/
│   └── telegram_queue_processor.py  # DB-queue polling and delivery with backoff
└── tests/                      # Bot-level integration tests
```

## Integration

This module integrates with:

- `src.data.db.services.telegram_service` — user records, verification codes, admin IDs.
- `src.data.db.services.users_service` — user notification channel resolution.
- `src.notification.service.client` — notification delivery (Telegram, email).
- `src.scheduler` — receives inbound HTTP calls from the scheduler for alert notifications.
- `src.common.alerts` — alert evaluation and cron parsing.
- `src.common.recommendation.engine` — indicator recommendations in reports.

## Configuration

Key environment variables:

| Variable | Description |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Bot token from @BotFather |
| `TELEGRAM_API_KEY` | X-API-Key secret for the internal HTTP API |
| `TELEGRAM_API_PORT` | Port for the internal aiohttp server (default: 8080) |
| `DATABASE_URL` | PostgreSQL connection string |
| `NOTIFICATION_SERVICE_URL` | Base URL for the notification service |

## Related Documentation

- [Requirements](docs/Requirements.md) — Dependencies and constraints
- [Design](docs/Design.md) — Architecture and design decisions
- [Tasks](docs/Tasks.md) — Implementation roadmap
