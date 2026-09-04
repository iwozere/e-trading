# Portfolio PnL Alert

## Overview
Sends one combined Telegram + Email notification once per weekday listing
**every** currently held ticker, sorted by PnL% descending — positions at or
above +10% above the user's average buy price are highlighted. Each held
ticker also shows any insider (Form 4) buying/selling from the trailing 30
days. Holdings come from IBKR alone: the daily Flex Query XML export,
optionally topped up with same-day live broker positions.

## Features
- Refreshes `Open_Positions.xml` daily from IBKR's Flex Web Service before
  each run (`flex_downloader.py`), so the XML export is never stale
- Pulls live positions (average cost) from IBKR via `IBKRBroker`, merged on
  top of the XML export (live wins on same symbol)
- Disambiguates IBKR tickers that collide across exchanges (e.g. `GOLD` on
  LSE vs NYSE) using the Flex Query `listingExchange` column — see
  "IBKR symbol collisions" below
- Fetches latest daily close via the shared `DataManager.get_ohlcv`
- Sends a single digest of every held position, sorted by PnL% descending,
  with rows at/above `threshold_pct` highlighted
- Attaches trailing 30-day insider (Form 4) activity — both buys and sells,
  any insider role — under each held ticker that has any, sourced from
  P18's shared EDGAR daily cache (no new network polling); 10b5-1 plan
  trades are labeled, not filtered out
- Runs on the existing APScheduler service - a single row in the
  `job_schedules` table is the only runtime wiring required
- Also exposes a CLI (`python -m src.portfolio.pnl_alert`) for manual runs

## Quick Start

1. Set `IBKR_FLEX_TOKEN` / `IBKR_FLEX_QUERY_ID` (see "Configuration" below)
   so `flex_downloader.py` can pull your "Open Positions" Flex Query.
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
- `src.data.downloader.edgar_downloader.EdgarDownloader` - Form 4 insider
  activity for held tickers, read from P18's shared daily cache
- `src.notification.service.client` - Telegram + Email delivery
- `src.scheduler.scheduler_service` - APScheduler host (one dispatch branch
  added for `target == "portfolio.pnl_alert"`)
- `src.data.db.services.jobs_service` - inserting / updating the schedule row

## Configuration
- YAML files under `src/portfolio/pnl_alert/config/`
- Environment variables reused:
  - `IBKR_HOST`, `IBKR_PORT`, `IBKR_CLIENT_ID`
  - `TELEGRAM_BOT_TOKEN`
  - `SMTP_SERVER`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`
  - `NOTIFICATION_SERVICE_URL` (falls back to `http://localhost:5003`)
- Environment variables added for `flex_downloader.py`:
  - `IBKR_FLEX_TOKEN` - Flex Web Service token (Client Portal > Performance
    & Reports > Flex Queries)
  - `IBKR_FLEX_QUERY_ID` - Query id of the "Open Positions" Flex Query.
  - If either is unset, the download step is skipped (logged at INFO) and
    the run falls back to whatever `Open_Positions.xml` is already on disk.

## Symbol pricing gotchas

Prices are fetched via the shared `DataManager`, which routes requests to
the providers configured in `config/data/provider_rules.yaml` (Yahoo is the
primary for daily stock data).

- **US-listed equities** resolve on the bare IBKR ticker, e.g. `NVDA`, `AAPL`.
- **Non-US listings** need a Yahoo exchange suffix, e.g. `VUSD.L` for the
  London-listed Vanguard S&P 500 ETF. `ibkr_xml_loader.py` derives this
  automatically from the Flex Query `listingExchange` column — see "IBKR
  symbol collisions" below. Without that column enabled, non-US symbols will
  404 on US providers and be silently excluded from the alert (WARNING).
- If a symbol is halted or otherwise unpriceable the pipeline simply skips it
  and continues with the rest of the holdings. Check the run logs for
  `No current price for N symbols (excluded from alert): [...]`.

## IBKR symbol collisions (e.g. `GOLD`)

IBKR tickers are not globally unique. `GOLD` is both a London-listed gold
ETC (`listingExchange="LSEETF"`, price ~$4) and Barrick Gold Corp on NYSE
(price ~$40) — a bare `GOLD` handed to the data provider silently resolves
to whichever one *it* considers canonical, which can produce a false PnL
alert instead of a missing-price warning.

`ibkr_xml_loader.py` resolves this using the position's `listingExchange`
Flex Query column (see `_resolve_provider_symbol` and the module docstring
for the full precedence: manual override map, then listing-exchange suffix,
then bare symbol + WARNING for anything unrecognized). This requires the
**"Listing Exchange"** (and ideally **"Currency"**) columns to be enabled on
the "Open Positions" Flex Query template:

IBKR Client Portal → Performance & Reports → Flex Queries → edit the "Open
Positions" query → Open Positions section → check "Listing Exchange" and
"Currency" → Save.

Without that column, `listingExchange` is empty for every position and the
loader falls back to the bare symbol (previous behavior).

## Related Documentation
- [Specification](docs/alert-specification.md) - Full functional spec
- [Requirements](docs/Requirements.md) - Technical requirements
- [Design](docs/Design.md) - Architecture and design
- [Tasks](docs/Tasks.md) - Implementation roadmap
