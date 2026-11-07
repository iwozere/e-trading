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
- [x] Deploy: migration on the Pi, register the job, run `--full` once
- [x] P24 harvester (`p24_daily`, `p24_weekly`)
- [x] P22 harvester: tier 1–3 entries (`p22_tier_entry`, sub-stream `p22_tier3`), 250-session book horizon (2026-11-07)
- [x] P15 reviewed: data-download bundle only, nothing to harvest (2026-11-07)
- [x] Exit-policy framework (2026-11-07): capped + uncapped shadow books per stream and policy (horizon, fixed 20/60/120/250,
      bracket, trail_atr, native = P22 tier drop / P24 paper exit / signal decay), trades with exit reason + MAE/MFE,
      `signal_book_trades` / `signal_book_stats` (migration 017), API policies/trades/positions
- [x] 60/120/250-session horizons and 20-session MAE/MFE per signal; OHLC price cache; nightly refresh of every ticker
      with a signal in the last 400 days (fixes stale marks of positions held past 20 sessions)

### 🚀 PLANNED ENHANCEMENTS (roadmap agreed 2026-11-07, in order)
- [ ] P22 deal-announced exit (needs a deal-event feed; today the native exit is the tier drop)
- [ ] Kill/promote rules per stream (minimum closed trades, cost-adjusted excess t, drawdown vs VT), optional SPRT
- [ ] Random-entry baseline per stream (same universe, same dates) as a selection-skill control
- [ ] Alert (Telegram/email) when a stream's verdict changes
- [ ] Web UI: per-stream equity vs VT and baseline, exit-policy × stream matrix, open positions, trade drill-down, verdict badges
- [ ] Optional IBKR paper execution of promoted streams only (gateway port 4002, `orderRef` per stream), fills reconciled
      against the ledger's assumed next-open to measure real slippage
- [ ] P23 harvester once its FMP-based data is ready
- [ ] P05 sub-streams (e.g. confidence ≥ 8) to see whether the LLM's confidence ranks the picks

## Technical Debt
- [ ] Price cache keeps every ticker forever; prune tickers with no signal in the last year

## Known Issues
- The price cache starts 2025-12, so the 120/250-session horizons and long-hold policies have few closed trades yet
- Uncapped stats still skip a re-signal of a ticker already held (no pyramiding), by design
- Yahoo price data for penny stocks: unadjusted reverse splits (handled by the jump guard, which can also drop a genuine 5× move)
- Survivorship: ~3 % of signals end as `no_data` (delisted before any bar could be fetched); they are not counted
- History is short (P05 since 2026-06-26): verdicts can change as data accumulates

## Testing Requirements
- [x] Unit tests: harvesters, evaluator, scorecard, portfolio, job end-to-end with a fake service/fetcher
- [x] API route tests
- [x] Repo tests in `src/data/db/tests/repos` (run against the local Postgres test DB)

## Documentation Updates
- [x] README, Requirements, Design, Tasks
