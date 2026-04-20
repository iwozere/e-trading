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
                     +--> position_aggregator
                     |       |
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
