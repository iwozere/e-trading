# Requirements

## Python Dependencies
Already satisfied by the root `requirements.txt`:

- `ib_insync >= 0.9.86` - IBKR client
- `pyyaml` - watchlist + config YAML parsing
- `pandas` - price DataFrame handling (via `DataManager`)
- `apscheduler`, `croniter` - scheduler integration
- `aiogram`, `aiosmtplib` - Telegram / email notification channels
- `aiohttp` - `NotificationServiceClient` transport

## External Module Dependencies
- `src.trading.broker.ibkr_broker` - live positions + average cost
- `src.data.data_manager` - batched OHLCV / latest close
- `src.notification.service.client` - notification dispatch
- `src.notification.logger` - logger factory
- `src.scheduler.scheduler_service` - job dispatcher (one branch added for
  `target.startswith("portfolio.")`)
- `src.data.db.services.jobs_service` - schedule CRUD
- `src.data.db.models.model_jobs` - `Schedule`, `JobType`, `ScheduleCreate`

## External Services
- IBKR TWS or Gateway reachable at `IBKR_HOST:IBKR_PORT` with the configured
  `IBKR_CLIENT_ID`. Live positions and average cost come from
  `IB.positions()`.
- Market data providers used by `DataManager` (Yahoo / Polygon / Alpaca / etc.)
  for the latest daily close.
- Telegram Bot API (`TELEGRAM_BOT_TOKEN`) and SMTP server (`SMTP_*` env vars)
  for notification delivery.
- PostgreSQL / SQLite database that backs `JobsService` (the existing
  `job_schedules` table is reused).

## Security Requirements
- No new secrets introduced. All credentials are reused from
  `config.donotshare.donotshare` (IBKR, Telegram, SMTP).
- Watchlist YAML is stored in the repo config folder and is considered
  non-sensitive (symbol + average price only).

## Performance Requirements
- Runs once a day, processing a short list (tens) of tickers. No throughput
  concerns.
- Each IBKR round-trip and price fetch must have reasonable timeouts so a
  single slow provider cannot stall the whole run.
- Per-symbol price failures must not fail the entire run.
