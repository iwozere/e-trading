# Tasks — P19 Intraday Penny-Stock Monitor

## Implementation Status

### ✅ COMPLETED (Phase 0 — scaffold)
- [x] Pipeline specification (`docs/pipeline-specification.md`)
- [x] Feed-latency/capability probe (`tools/latency_probe.py`) + 2026-06-28 findings (§13.1)
- [x] Config dataclasses (`config.py`) + tests
- [x] `IntradaySignal` model (`models/`) + tests
- [x] CLI scaffold (`run_p19.py`) with run modes
- [x] Submodule docs (README, Requirements, Design, Tasks)

### 🔄 IN PROGRESS — Phase 1 (watchlist + shadow logger, NO alerts)
- [x] **Watchlist Builder** (`watchlist_builder.py`): P17 source (Tier B/C + explosive)
      + IBKR-scanner gappers + manual pins; hard filters; dedup/precedence; priority
      rank; cap to N; `watchlist.json` with baseline context. Wired to
      `run_p19 build-watchlist`. Tested on real P17 output (gappers need the Pi Gateway).
- [x] **`run-once` shadow loop** (`shadow_loop.py`): one delayed IBKR `reqMktData`
      snapshot per watchlist name (one line each, ≤100) → %-move / RVOL-so-far →
      append to SQLite. Feed/store/EOD-fetcher injectable; tested with a fake feed.
- [x] **Metrics** (`metrics.py`): %-from-open/prev-close, RVOL-so-far with a *linear*
      session-fraction placeholder (real U-shaped profile comes from shadow data).
- [x] **Shadow store** (`shadow_store.py`): SQLite at `results/p19_penny_intraday/
      shadow.sqlite` + `eod-backfill` of O/H/L/C (via DataManager, DATA_CACHE_DIR).
- [ ] **Verify on the Pi during market hours**: `run-once --mode shadow` logs rows;
      check RVOL/volume units (`ibkr_volume_lot_size`, default 100) against live numbers.
- [ ] Enrich **gappers' baseline** (avg_volume / prior_close) via DataManager — they
      currently log `rvol=0` until baseline is fetched (P17 names already have it).
- [ ] Wire a market-hours intraday cron (UTC, DST-aware) for `run-once` + `eod-backfill`.

> **Verify on the Pi:** run `python src/ml/pipeline/p19_penny_intraday/run_p19.py
> build-watchlist` with the Gateway up to confirm the IBKR gappers scanner adds
> pre-market movers beyond the P17 names.

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
