# Tasks

## Implementation Status

### COMPLETED FEATURES
- [x] Package scaffolding (`__init__.py`, README, docs, config YAMLs)
- [x] Config dataclasses + YAML loader (`config.py`)
- [x] Watchlist loader with schema validation (`watchlist_loader.py`)
- [x] Position aggregator merging IBKR + watchlist, STK-only (`position_aggregator.py`)
- [x] Price fetcher over `DataManager.get_ohlcv` (`price_fetcher.py`)
- [x] Pure PnL evaluator (`pnl_evaluator.py`)
- [x] Notifier (combined Telegram + Email message) (`notifier.py`)
- [x] Orchestrator `runner.run_once` + CLI (`runner.py`, `cli.py`, `__main__.py`)
- [x] Scheduler integration: dispatch branch + `seed_schedule.py`
- [x] Unit tests (evaluator, aggregator, notifier format, watchlist loader)

### PLANNED ENHANCEMENTS
- [ ] Optional `quantity` field for watchlist entries (today defaults to 1)
- [ ] First-crossing dedup / state (today notifies daily)
- [ ] Downside alerts (e.g. below -10%)
- [ ] FX / multi-currency support
- [ ] Interactive Telegram controls (ack / snooze)
- [ ] Optional `exchange` field in watchlist entries to avoid hand-typed
      Yahoo suffixes like `.L`, `.DE`

## Technical Debt
- [ ] IBKR sec-type filtering uses `ib.positions()` directly; a future
      enhancement is to thread sec-type through `IBKRBroker.Position.metadata`.

## Known Issues
- None at the time of writing.

## Testing Requirements
- [x] Unit tests for the pure evaluator
- [x] Unit tests for watchlist YAML loader (happy / invalid paths)
- [x] Unit tests for IBKR+watchlist merge precedence
- [x] Unit tests for notification message formatting
- [ ] Integration smoke test against a live IBKR paper account (manual)
