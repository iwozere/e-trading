# Tasks

## Implementation Status

### ✅ COMPLETED FEATURES
- [x] Harvesters for P05, P06, P10, P17, P19, P21 (2026-10-21)
- [x] `signal_ledger` / `signal_book_nav` tables, migration `010_signal_ledger`
- [x] Forward-return evaluator with next-open entry and the 5× price-jump guard
- [x] Scorecard (cost-adjusted excess, Newey-West cohort t, verdicts) vs VT/SPY/IWM
- [x] Paper book simulation (USD 2,000 / position, max 40, rest in VT)
- [x] Nightly job + scheduler spec, `--full` backfill
- [x] API `/api/signals/*` and Monitoring → Signals tab

### 🔄 IN PROGRESS
- [ ] Deploy: migration on the Pi, register the job, run `--full` once

### 🚀 PLANNED ENHANCEMENTS
- [ ] P22 and P23 harvesters once their FMP-based data is ready
- [ ] P05 sub-streams (e.g. confidence ≥ 8) to see whether the LLM's confidence ranks the picks
- [ ] Optional IBKR paper execution of the streams with a proven edge (orders on gateway port 4002, `orderRef` per stream)
- [ ] Alert (Telegram/email) when a stream's verdict changes

## Technical Debt
- [ ] Price cache keeps every ticker forever; prune tickers with no signal in the last year

## Known Issues
- Yahoo price data for penny stocks: unadjusted reverse splits (handled by the jump guard, which can also drop a genuine 5× move)
- Survivorship: ~3 % of signals end as `no_data` (delisted before any bar could be fetched); they are not counted
- History is short (P05 since 2026-06-26): verdicts can change as data accumulates

## Testing Requirements
- [x] Unit tests: harvesters, evaluator, scorecard, portfolio, job end-to-end with a fake service/fetcher
- [x] API route tests
- [ ] Repo tests in `src/data/db/tests/repos` (need the Postgres test DB setup)

## Documentation Updates
- [x] README, Requirements, Design, Tasks
