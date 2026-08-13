# Portfolio Management — Stop-Loss & Earnings Reminder

## Overview
Reminds the user, via Telegram + Email, when a currently-held IBKR ticker has
an upcoming earnings release and its live position doesn't have a working
protective stop order covering it. **Reminder only — it never places,
modifies, or cancels an order.** The user sets stop-loss orders manually;
this system's job is to make sure that's never forgotten right before an
earnings-driven price gap.

## Features
- Checks stop coverage **only** around each held ticker's own earnings date:
  T-1 day and T-1 hour before (BMO/AMC-aware), not on a continuous baseline
  poll — see `docs/Design.md` for why.
- Reads live protective orders (`STP` / `STP LMT` / `TRAIL` / `TRAIL LIMIT`)
  from the **live** IBKR Gateway, read-only — never paper, since real stops
  only exist on the live account.
- Reuses `pnl_alert`'s Flex Query XML holdings pipeline, `p05_ai_selector`'s
  FMP-backed earnings calendar, the shared `NotificationServiceClient`, and
  the existing `job_schedules` / APScheduler wiring — no new infrastructure.
- Fires every time the trigger condition is met (no dedup) — see
  `docs/Design.md` "Decisions" for the plan to revisit this later.
- Exposes a CLI (`python -m src.portfolio.management`) for manual runs.

## Quick Start

1. Set `IBKR_LIVE_STOP_GUARD_CLIENT_ID` (see "Configuration" below) — a
   clientId dedicated to this module, distinct from any live trading bot.
2. On the **live** Gateway/TWS, set this clientId (or whichever one connects)
   as the account's **Master API Client ID** (Configure → API → Settings) —
   without this, manually-placed stops are invisible to the API and every
   held ticker will be wrongly reported as uncovered. See
   `docs/Requirements.md` "External Services".
3. Edit `src/portfolio/management/config/management.yaml` if you want to
   change the lookahead window, trigger tolerance, or channels.
4. Run once manually to validate the setup:

```bash
python -m src.portfolio.management --dry-run
```

5. Insert the polling schedule into the `job_schedules` table:

```bash
python -m src.portfolio.management.seed_schedule
```

6. Reload the scheduler so it picks up the new row:

```bash
python -m src.scheduler.cli reload
```

## Integration
- `src.portfolio.pnl_alert.flex_downloader` / `ibkr_xml_loader` /
  `position_aggregator` — holdings, reused as-is (not re-implemented)
- `src.ml.pipeline.p05_ai_selector.signals.earnings_calendar` — earnings
  dates for held tickers (FMP-backed, monthly cache)
- `src.notification.service.client` — Telegram + Email delivery
- `src.scheduler.scheduler_service` — APScheduler host (dispatch branch for
  `target == "portfolio.management"`)
- `src.data.db.services.jobs_service` — inserting / updating the schedule row

## Configuration
- YAML: `src/portfolio/management/config/management.yaml`
- Environment variables reused: `IBKR_HOST`, `IBKR_PORT` (the **live**
  Gateway vars — see `docs/brainstorm.md` "Live account only" for why this
  is deliberately not `IBKR_PAPER_PORT`), `TELEGRAM_BOT_TOKEN`, `SMTP_*`,
  `FMP_API_KEY` (via `EarningsCalendar`), `IBKR_FLEX_TOKEN` /
  `IBKR_FLEX_QUERY_ID` (via `pnl_alert`'s Flex downloader)
- Environment variable added: `IBKR_LIVE_STOP_GUARD_CLIENT_ID` — a clientId
  dedicated to this module's live, read-only connection, distinct from
  `IBKR_CLIENT_ID` so it can never collide with a live trading bot session.

## Related Documentation
- [Brainstorm](docs/brainstorm.md) — original scoping discussion, the Flex
  Query "no open orders" finding, and what was deliberately rejected
- [Requirements](docs/Requirements.md) — technical requirements
- [Design](docs/Design.md) — architecture and design decisions
- [Tasks](docs/Tasks.md) — implementation roadmap
