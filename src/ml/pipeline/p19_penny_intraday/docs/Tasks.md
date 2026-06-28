# Tasks — P19 Intraday Penny-Stock Monitor

## Implementation Status

### ✅ COMPLETED (Phase 0 — scaffold)
- [x] Pipeline specification (`docs/pipeline-specification.md`)
- [x] Feed-latency/capability probe (`tools/latency_probe.py`) + 2026-06-28 findings (§13.1)
- [x] Config dataclasses (`config.py`) + tests
- [x] `IntradaySignal` model (`models/`) + tests
- [x] CLI scaffold (`run_p19.py`) with run modes
- [x] Submodule docs (README, Requirements, Design, Tasks)

### 🔄 NEXT — Phase 1 (watchlist + shadow logger, NO alerts)
- [ ] Watchlist Builder: consume P17 daily output + pre-market gappers < $5; hard
      filters; cap to N; write `watchlist.json` with baseline context
- [ ] `run-once` shadow loop: poll Finnhub `/quote` for price action
- [ ] RVOL-so-far: intraday volume-profile baseline (bootstrap from `daily_avg × cdf`)
- [ ] Shadow store (SQLite/Parquet) + `eod-backfill` of O/H/L/C
- [ ] Wire a market-hours intraday cron (UTC, DST-aware)

### 🚀 PLANNED
- [ ] Phase 2: Trigger Engine (price thrust + delayed-volume confirm) + dedup state +
      Telegram alerts + daily cap
- [ ] Phase 3: enrichment via P17 `CatalystAgent` / `ShortSqueezeAgent` /
      `DilutionAgent`; intraday EFTS 8-K polling; sentiment context attach
- [ ] Phase 4: Optuna threshold calibration on shadow data; LULD halt detection;
      optional LLM alert summarizer

## Known Issues / Constraints
- **Primary feed = IBKR Gateway (delayed, free)** — gives 5m bars *with volume* (spec
  §13.2). Binding limits: ~100 market-data lines (→ watchlist N ≤ 100), historical
  pacing ~60/10min (→ stream, don't poll). Gateway must be up; daily restart →
  handle reconnects; unique clientId (p19=19).
- Free REST tiers lack real-time volume (§13.1) — kept only as fallback / price
  cross-check.
- **Run `latency_probe --ibkr` on the Pi during market hours** to confirm real delay
  and volume presence before Phase 1.

## Open Questions (spec §17)
- [ ] Best batch/snapshot endpoints + measured market-hours latency per provider
- [ ] Gappers / most-active source (provider screener vs pre-market scan)
- [ ] Volume-profile bootstrap window (days of history before real profile)
- [ ] Shadow store backend: SQLite vs Parquet vs planned DuckDB layer

## Testing Requirements
- [x] config + model unit tests
- [ ] Phase 1: watchlist builder, RVOL calc, shadow store round-trip
- [ ] Phase 2: trigger gate + dedup/escalation state machine
