# Requirements

## Python Dependencies
Already satisfied by the root `requirements.txt` (same set `pnl_alert` and
P19 already use):

- `ib_insync >= 0.9.86` (or `ib_async`) — IBKR client
- `pyyaml` — config YAML parsing
- `pandas` — via `EarningsCalendar` / `DataManager`
- `apscheduler`, `croniter` — scheduler integration
- `aiogram`, `aiosmtplib` — Telegram / email notification channels
- `aiohttp` — `NotificationServiceClient` transport
- `requests` — `EarningsCalendar`'s FMP calls (via `pnl_alert.flex_downloader`
  reuse, and `p05_ai_selector.signals.earnings_calendar`)

No new dependency was added for this module.

## External Module Dependencies
- `src.portfolio.pnl_alert.flex_downloader` / `ibkr_xml_loader` /
  `position_aggregator` — holdings pipeline, reused directly (not
  re-implemented)
- `src.ml.pipeline.p05_ai_selector.signals.earnings_calendar.EarningsCalendar`
  — earnings dates for held tickers
- `src.notification.service.client` — notification dispatch
- `src.notification.logger` — logger factory
- `src.scheduler.scheduler_service` — job dispatcher (`target ==
  "portfolio.management"` branch inside the existing `portfolio.*` handler)
- `src.data.db.services.jobs_service` — schedule CRUD
- `src.data.db.models.model_jobs` — `Schedule`, `JobType`, `ScheduleCreate`

## External Services
- **Live** IBKR TWS or Gateway reachable at `IBKR_HOST:IBKR_PORT` (port
  `4001` for Gateway, `7496` for TWS) with a dedicated
  `IBKR_LIVE_STOP_GUARD_CLIENT_ID`. **Never** connects to the paper account
  (`IBKR_PAPER_PORT`) — real stop orders only exist on live.
  - **Precondition: Master API Client ID.** For a stop placed manually via
    the TWS/Gateway GUI to be visible to `reqAllOpenOrders()`, the connecting
    clientId must be configured as the account's Master API Client ID
    (Gateway/TWS → Configure → API → Settings). Without this, every
    manually-set stop is wrongly reported as missing — this is a one-time
    manual setup step on the Gateway, not something the code can detect or
    work around.
  - Read-only connection (`readonly=True`) — this module can never place,
    modify, or cancel an order, by construction, not just by convention.
- IBKR Flex Web Service (`IBKR_FLEX_TOKEN` + `IBKR_FLEX_QUERY_ID`) — reused
  from `pnl_alert`, same daily "Open Positions" XML export.
- FMP API (`FMP_API_KEY`) via `EarningsCalendar` — earnings calendar,
  already paid for and used by the P05 AI Selector pipeline.
- Telegram Bot API (`TELEGRAM_BOT_TOKEN`) and SMTP server (`SMTP_*` env vars)
  for notification delivery.
- SQLite/PostgreSQL database that backs `JobsService` (the existing
  `job_schedules` table is reused; no new tables required for the MVP —
  see `Design.md` "Phase 2 (optional)").

## Security Requirements
- All credentials are reused from `config.donotshare.donotshare`. One new
  variable was added: `IBKR_LIVE_STOP_GUARD_CLIENT_ID` (an integer clientId,
  not a secret in itself, but kept in the same env-var-only convention as
  the rest of the IBKR configuration).
- This module never writes orders, so it carries no "wrong side" execution
  risk the way a trading bot would — the only failure mode is a wrong or
  missed *reminder*, not a wrong trade.

## Performance Requirements
- Polls on a 15-minute cadence during ~12:00–22:00 UTC on weekdays (see
  `Design.md`), processing a short list (tens) of held tickers. No
  throughput concerns.
- The live IBKR connection is only opened when at least one ticker's
  T-1day/T-1hour trigger is actually in-window this run — most polls do no
  IBKR I/O at all.
- `reqAllOpenOrders()` and the earnings calendar lookup must have reasonable
  timeouts so a single slow/unreachable dependency doesn't stall the run;
  both failure modes are non-fatal (see `Design.md` "Error Handling").

## Known Limitations
- **Session (BMO/AMC) detection is not implemented.** `EarningsCalendar`
  doesn't currently surface a session/time-of-day field from FMP, so every
  earnings event is treated as `session="unknown"`, which
  `earnings_window.resolve_anchor_utc` anchors to market open (09:30 ET) —
  the safer (earlier) of the two anchors. A ticker that actually reports
  after market close will still get correct T-1day/T-1hour triggers relative
  to the *open* anchor, which is up to ~6.5 hours earlier than its real AMC
  anchor — earlier reminders, never later ones. See `Tasks.md`.
