# Design

## Purpose
Give the user a daily digest of every holding currently above a configurable
profit threshold (default +10%). Live positions from IBKR and manually
tracked positions in a YAML watchlist are merged into one unified view.

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
                     |       +--> watchlist_loader (YAML)
                     |
                     +--> price_fetcher  (DataManager.get_ohlcv, last close)
                     |
                     +--> pnl_evaluator  (pure function)
                     |
                     +--> notifier       (NotificationServiceClient)
```

### Component Design
- **config.py** - `PnLAlertConfig` dataclass + `load_config(path)` YAML loader.
- **watchlist_loader.py** - validates schema, returns `list[WatchlistEntry]`.
- **flex_downloader.py** - calls IBKR's Flex Web Service (SendRequest /
  GetStatement) to refresh `Open_Positions.xml` at the start of every run.
  Writes both a fixed filename and a `Open_Positions-YYYY-MM-DD.xml`
  date-stamped copy. Best-effort: any failure (missing credentials, network
  error, IBKR error response) is logged and swallowed so the run falls back
  to whatever XML is already on disk.
- **ibkr_xml_loader.py** - parses the Flex Query "Open Positions" XML export
  (refreshed daily by `flex_downloader.py`) into `RawIbkrPosition` objects.
- **position_aggregator.py** - calls IBKR, filters to STK, merges with watchlist
  (IBKR wins on conflicts). Produces `list[Holding]`.
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
- Input: IBKR positions (symbol, quantity, avg_price), YAML watchlist
  (symbol, avg_price, optional notes), current-close dict (symbol -> price).
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
If IBKR is unreachable the pipeline proceeds with the watchlist alone and logs
a WARNING. This keeps the daily digest useful even when TWS is offline.

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

## Integration Patterns
- Scheduler integration is a single branch inside the existing `ALERT`
  handler: if `schedule.target.startswith("portfolio.")`, dispatch to this
  module's `runner.run_once`. All existing alerts keep going through the
  `AlertEvaluator`.
- Notifications are sent via the shared `NotificationServiceClient`, so the
  same Telegram and SMTP plumbing the rest of the app uses.

## Error Handling
- Missing watchlist file or invalid YAML: log CRITICAL + optionally send a
  CRITICAL notification so the user sees the failure. Run exits non-zero.
- IBKR unreachable: WARNING, run continues with watchlist only.
- Per-symbol price failure: WARNING, symbol excluded, run continues.
- All prices fail: ERROR, run emits a critical notification saying "price
  fetch failed" and exits non-zero.
