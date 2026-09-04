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
- [x] Full daily digest, not threshold-gated (2026-09-04) — every priced
      holding is now included every run, sorted by PnL% desc; `threshold_pct`
      only controls the `flagged` highlight, not inclusion
- [x] Insider (Form 4) activity per held ticker (2026-09-04) — trailing
      30-day buys/sells for currently-held tickers only, sourced from P18's
      shared EDGAR daily cache (`insider_activity.py`); 10b5-1 plan trades
      are shown but labeled separately, not filtered out
- [x] Widened `EdgarDownloader`'s Form 4 parser to also capture role flags
      (`is_director`/`is_officer`/`is_ten_percent_owner`/`officer_title`) and
      the true per-transaction `transaction_date` (distinct from `filed_date`)

### PLANNED ENHANCEMENTS
- [ ] First-crossing dedup / state (today notifies daily)
- [ ] Downside alerts (e.g. below -10%)
- [ ] FX / multi-currency support
- [ ] Interactive Telegram controls (ack / snooze)

## Technical Debt
- [ ] IBKR sec-type filtering uses `ib.positions()` directly; a future
      enhancement is to thread sec-type through `IBKRBroker.Position.metadata`.

## Known Issues
- **Fixed 2026-09-04**: P19's structural profiler (`_load_form4_window` /
  `_load_dg_window`) was fetching *today's* still-open date as part of its
  100-day lookback, caching a partial same-day snapshot as final and
  poisoning the shared Form4/13D-G cache this module's insider-activity
  feature reads from. Both loaders now skip any date >= today (UTC). The 12
  poisoned cache days (2026-08-19 through 2026-09-03) were deleted and
  self-heal on the next run.

## Testing Requirements
- [x] Unit tests for the pure evaluator (including `flagged` semantics)
- [x] Unit tests for IBKR Flex Query XML loader (parsing, merging, exchange disambiguation)
- [x] Unit tests for position aggregation (STK filter, non-positive qty/price)
- [x] Unit tests for notification message formatting (full digest + insider section)
- [x] Unit tests for insider activity loading (`test_insider_activity.py`)
- [x] Unit tests for the orchestrator (`test_runner.py`)
- [ ] Integration smoke test against a live IBKR paper account (manual)
