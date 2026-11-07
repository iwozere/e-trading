# Tasks

## Implementation Status

### ✅ COMPLETED FEATURES
- [x] Stages 1–7: universe, market data, technicals, short squeeze, dilution, scoring, reporting
- [x] Stage 5.5 CatalystAgent (8-K items + keywords from the shared daily 8-K index)
- [x] Stage 8 Tier A/B email via the notification service
- [x] Survival filter actually narrows the universe (cash runway, debt/cash)
- [x] `backtest.py` best-case backtest; `strategy_sim.py` stop/trailing simulator + Optuna (spec §15.0–15.1)
- [x] Extension gate - already-run names capped at Tier C, never EXPLOSIVE (2026-11-06, spec §11.1)
- [x] Pre-breakout setup score, shadow mode + signal-ledger `p17_setup` stream (2026-11-06, spec §11.2)
- [x] `replay.py` point-in-time replay with forward returns (2026-11-06)
- [x] OHLCV always re-downloaded (no stale or partial intraday bars from the shared cache), no look-ahead past
  `target_date` (2026-11-07, CR-2026-11-06 §5)

### 🔄 IN PROGRESS
- [x] Deploy the extension gate to the Pi (2026-11-07)
- [ ] Deploy the OHLCV freshness fix; confirm the log shows one full batch download and few/no lagging tickers
- [ ] Confirm the first gated runs (fewer Tier A/B emails, `extended=` / `setups=` in the `Result:` log line)
- [ ] Watch the signal-ledger scorecard for `p17_tier_ab` (gated) and `p17_setup` once ≥ 20 cohorts mature

### 🚀 PLANNED ENHANCEMENTS
- [ ] Scheduled catalysts known in advance: PDUFA / AdCom dates, earnings dates, insider Form 4 buys, 13D filings
- [ ] Intraday trigger: P19 alerts when watchlist names cross time-of-day-normalised RVOL + price levels
- [ ] Two-tier universe: looser liquidity (median volume / ~$300–500k dollar volume, pre-jump market cap & float) for
  the watchlist, strict liquidity only for the actionable alert
- [ ] 8-K body-text classification (`docs/catalyst_8k_improvement_plan.md`, proposal)
- [ ] Strategy automation resume point (spec §15.2): walk-forward, paper trading, gap-risk model

## Technical Debt
- [ ] Shared UnifiedCache stores partial intraday daily bars for every pipeline that reads it during market hours
  (P17 now bypasses it; other readers are still exposed) — consider a "complete after close" marker in the cache
- [ ] Tier A practically never fires (spec §24.5 calibration decision still open)
- [ ] `ReportingAgent.get_alert_candidates` and `P17AlertConfig.min_alert_score` are unused

## Known Issues
- ~3% of the universe is missing from the snapshot each run (Yahoo fetch failures / rate limits), including some
  names that later produced signals.
- Replay is limited to tickers that passed the universe filters on each day (only those rows are saved).
- Sample sizes are small (20 gated episodes over 6 months); treat the gate result as encouraging, not proven.

## Testing Requirements
- [x] Unit tests for scoring (incl. extension gate and setup), technicals (incl. setup features), replay helpers,
  notification, catalyst, config, pipeline, backtest, strategy sim
- [x] Tests for the OHLCV freshness rule (`tests/test_market_agent.py`)

## Documentation Updates
- [x] README / Requirements / Design / Tasks (2026-11-06)
- [ ] Update spec §24 calibration backlog once the Tier A decision is made
