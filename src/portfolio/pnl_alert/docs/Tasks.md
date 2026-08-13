# Tasks

## Implementation Status

### COMPLETED FEATURES
- [x] Package scaffolding (`__init__.py`, README, docs, config YAMLs)
- [x] Config dataclasses + YAML loader (`config.py`)
- [x] Position aggregator, STK-only (`position_aggregator.py`)
- [x] Price fetcher over `DataManager.get_ohlcv` (`price_fetcher.py`)
- [x] Pure PnL evaluator (`pnl_evaluator.py`)
- [x] Notifier (combined Telegram + Email message) (`notifier.py`)
- [x] Orchestrator `runner.run_once` + CLI (`runner.py`, `cli.py`, `__main__.py`)
- [x] Scheduler integration: dispatch branch + `seed_schedule.py`
- [x] Daily Open_Positions.xml refresh from IBKR Flex Web Service (`flex_downloader.py`)
- [x] IBKR symbol disambiguation via `listingExchange` (e.g. `GOLD` LSE vs NYSE) (`ibkr_xml_loader.py`)
- [x] Unit tests (evaluator, aggregator, notifier format, XML loader, flex downloader)
- [x] Removed the YAML watchlist source (2026-08-13) — IBKR (XML + live) is
      now the sole holdings source; see `Design.md` for rationale

### PLANNED ENHANCEMENTS
- [ ] First-crossing dedup / state (today notifies daily)
- [ ] Downside alerts (e.g. below -10%)
- [ ] FX / multi-currency support
- [ ] Interactive Telegram controls (ack / snooze)

## Technical Debt
- [ ] IBKR sec-type filtering uses `ib.positions()` directly; a future
      enhancement is to thread sec-type through `IBKRBroker.Position.metadata`.

## Known Issues
- None at the time of writing.

## Testing Requirements
- [x] Unit tests for the pure evaluator
- [x] Unit tests for IBKR Flex Query XML loader (parsing, merging, exchange disambiguation)
- [x] Unit tests for position aggregation (STK filter, non-positive qty/price)
- [x] Unit tests for notification message formatting
- [ ] Integration smoke test against a live IBKR paper account (manual)
