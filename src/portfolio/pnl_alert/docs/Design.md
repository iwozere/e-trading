# Design

## Purpose
Give the user a daily digest of every holding currently above a configurable
profit threshold (default +10%). Holdings come from IBKR alone: the daily
Flex Query XML export, optionally topped up with same-day live positions.

## Architecture

### High-Level Architecture
```
APScheduler  ->  runner.run_once(cfg)
                     |
                     +--> flex_downloader (refresh Open_Positions.xml)
                     |
                     +--> position_aggregator
                     |       |
                     |       +--> ibkr_xml_loader (Flex Query XML export)
                     |       +--> IBKRBroker.get_positions / ib.positions()
                     |
                     +--> price_fetcher  (DataManager.get_ohlcv, last close)
                     |
                     +--> pnl_evaluator  (pure function)
                     |
                     +--> notifier       (NotificationServiceClient)
```

### Component Design
- **config.py** - `PnLAlertConfig` dataclass + `load_config(path)` YAML loader.
- **flex_downloader.py** - calls IBKR's Flex Web Service (SendRequest /
  GetStatement) to refresh `Open_Positions.xml` at the start of every run.
  Writes both a fixed filename and a `Open_Positions-YYYY-MM-DD.xml`
  date-stamped copy. Best-effort: any failure (missing credentials, network
  error, IBKR error response) is logged and swallowed so the run falls back
  to whatever XML is already on disk.
- **ibkr_xml_loader.py** - parses the Flex Query "Open Positions" XML export
  (refreshed daily by `flex_downloader.py`) into `RawIbkrPosition` objects.
- **position_aggregator.py** - takes the merged XML + live IBKR positions,
  filters to STK. Produces `list[Holding]`.
- **price_fetcher.py** - wraps `DataManager.get_ohlcv`, returns
  `dict[str, float]` keyed by symbol, resilient to per-symbol failures.
- **pnl_evaluator.py** - pure function:
  `evaluate(holdings, prices, threshold) -> list[AlertRow]`. Sorts by
  `pnl_pct` descending.
- **notifier.py** - formats one plain-text body + one HTML body, dispatches
  via `NotificationServiceClient.send_notification(...)`.
- **runner.py** - orchestrates the steps above; usable from the CLI, tests,
  and the scheduler.
- **cli.py** / **__main__.py** - `python -m src.portfolio.pnl_alert` with
  `--dry-run`, `--threshold`, `--config` flags.
- **seed_schedule.py** - idempotent inserter/updater of the row in
  `job_schedules`.

## Data Flow
- Input: IBKR positions (symbol, quantity, avg_price) from the XML export and
  optionally the live broker, current-close dict (symbol -> price).
- Output: a single notification message + a small `RunSummary` dict
  (stored by the scheduler as run-result JSON).

## Design Decisions

### Co-locate config YAMLs with the module
`src/portfolio/pnl_alert/config/*.yaml` keeps user-editable files next to the
code that consumes them and avoids polluting the top-level `config/` tree.

### Reuse `JobType.ALERT` rather than adding a new enum value
The `job_schedules` table has a hard DB check constraint:
```
CHECK (job_type IN ('report','screener','alert','notification','data_processing','backup'))
```
Adding a new `portfolio_pnl_alert` JobType would require a schema migration.
Using `job_type = "alert"` with `target = "portfolio.pnl_alert"` as the
dispatch key is zero-migration and semantically correct.

### No dedup / state
Per user's explicit choice: notify every day for every symbol currently above
threshold. No "first-crossing" tracking.

### IBKR connection is best-effort
If the live IBKR broker is unreachable the pipeline proceeds with whatever the
Flex Query XML export already has on disk and logs a WARNING. This keeps the
daily digest useful even when TWS is offline.

### No second holdings source (watchlist removed 2026-08-13)
A YAML watchlist for manually tracked / outside-IBKR positions existed
originally to cover positions the Flex Query wouldn't see. In practice 8 of
its 12 entries just duplicated IBKR XML positions (pure noise - IBKR always
won on conflict anyway), and the other 4 turned out to be closed/stale
positions nobody was removing. Rather than keep a second, driftable source of
truth, it was removed outright: IBKR (XML + live) is now the only source.
`Holding.source` is still a field (always `"ibkr"` today) so a second source
could be reintroduced later without changing the evaluator/notifier.

### Flex Query download runs inline, not as a separate scheduled job
`flex_downloader.download_open_positions_xml` is called at the top of
`runner.run_once`, right before `ibkr_xml_loader` reads the file, rather than
via its own cron entry (unlike, e.g., P20's VIX ingestion). This guarantees
the XML is always fresh relative to the alert that consumes it, with no risk
of the download and the alert schedules drifting out of sync. The download
itself is best-effort — failures fall back to the last file already on disk,
mirroring the "IBKR connection is best-effort" decision above.

### Pure evaluator
`pnl_evaluator.evaluate` has no I/O and is trivially unit-testable.

### Listing-exchange based symbol disambiguation
IBKR tickers collide across exchanges (`GOLD` = LSE gold ETC *and* NYSE
Barrick Gold Corp). Resolving purely by symbol risks a silently wrong price
from the data provider and a false PnL alert. `ibkr_xml_loader.py` reads the
`listingExchange` Flex Query attribute (when the "Listing Exchange" column is
enabled on the report template) and appends the matching provider suffix
(`LSEETF`/`LSE` → `.L`, etc.) via `_LISTING_EXCHANGE_SUFFIX_MAP`. A small
`_IBKR_SYMBOL_MAP` remains for verified one-off overrides that take priority
over the automatic resolution (e.g. `VUSD` → `VUSD.L`). Any non-US exchange
not yet in the map is logged as a WARNING rather than guessed — a missing
price is a safer failure mode than a mispriced one. If the Flex Query
template doesn't have the column enabled, `listingExchange` is empty and
resolution falls back to the bare symbol.

## Integration Patterns
- Scheduler integration is a single branch inside the existing `ALERT`
  handler: if `schedule.target.startswith("portfolio.")`, dispatch to this
  module's `runner.run_once`. All existing alerts keep going through the
  `AlertEvaluator`.
- Notifications are sent via the shared `NotificationServiceClient`, so the
  same Telegram and SMTP plumbing the rest of the app uses.

## Error Handling
- Flex Query XML unreadable/missing: logged, added to `RunSummary.errors`;
  run continues with whatever live IBKR positions are available (may be zero).
- Live IBKR unreachable: WARNING, run continues with the XML export alone.
- Per-symbol price failure: WARNING, symbol excluded, run continues.
- All prices fail: ERROR, run emits a critical notification saying "price
  fetch failed" and exits non-zero.
