# Portfolio PnL Alert

## Overview
Sends one combined Telegram + Email notification once per weekday listing every
ticker whose current market price is at least +10% above the user's average buy
price. Holdings are pulled live from IBKR and merged with a YAML watchlist of
manually tracked positions.

## Features
- Pulls live positions (average cost) from IBKR via `IBKRBroker`
- Merges a user-editable YAML watchlist for positions held outside IBKR
- Fetches latest daily close via the shared `DataManager.get_ohlcv`
- Sends a single digest message sorted by PnL% descending
- Runs on the existing APScheduler service - a single row in the
  `job_schedules` table is the only runtime wiring required
- Also exposes a CLI (`python -m src.portfolio.pnl_alert`) for manual runs

## Quick Start

1. Edit `src/portfolio/pnl_alert/config/watchlist.yaml` and add your
   manually tracked positions (symbol + average buy price).
2. Edit `src/portfolio/pnl_alert/config/pnl_alert.yaml` if you want to change
   the threshold, channels, or cron.
3. Run once manually to validate the setup:

```bash
python -m src.portfolio.pnl_alert --dry-run
```

4. Insert the daily schedule into the `job_schedules` table:

```bash
python -m src.portfolio.pnl_alert.seed_schedule
```

5. Reload the scheduler so it picks up the new row:

```bash
python -m src.scheduler.cli reload
```

## Integration
- `src.trading.broker.ibkr_broker` - live positions and average cost
- `src.data.data_manager` - market price lookup
- `src.notification.service.client` - Telegram + Email delivery
- `src.scheduler.scheduler_service` - APScheduler host (one dispatch branch
  added for `target == "portfolio.pnl_alert"`)
- `src.data.db.services.jobs_service` - inserting / updating the schedule row

## Configuration
- YAML files under `src/portfolio/pnl_alert/config/`
- Environment variables reused (no new vars introduced):
  - `IBKR_HOST`, `IBKR_PORT`, `IBKR_CLIENT_ID`
  - `TELEGRAM_BOT_TOKEN`
  - `SMTP_SERVER`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`
  - `NOTIFICATION_SERVICE_URL` (falls back to `http://localhost:5003`)

## Watchlist symbol gotchas

Prices are fetched via the shared `DataManager`, which routes requests to
the providers configured in `config/data/provider_rules.yaml` (Yahoo is the
primary for daily stock data).

- **US-listed equities** use the bare ticker, e.g. `NVDA`, `AAPL`, `SRPT`.
- **Non-US listings** need the Yahoo exchange suffix, e.g. `VUSD.L` for the
  London-listed Vanguard S&P 500 ETF, `SAP.DE` for XETRA. Putting `VUSD`
  alone will yield 404s on US providers and the symbol will be silently
  excluded from the alert (logged as a WARNING).
- If a symbol is mis-typed or halted the pipeline simply skips it and
  continues with the rest of the watchlist. Check the run logs for
  `No current price for N symbols (excluded from alert): [...]`.

## Related Documentation
- [Specification](docs/alert-specification.md) - Full functional spec
- [Requirements](docs/Requirements.md) - Technical requirements
- [Design](docs/Design.md) - Architecture and design
- [Tasks](docs/Tasks.md) - Implementation roadmap
