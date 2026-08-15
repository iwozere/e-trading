# Tasks — P10 EMPS3

## Implementation Status

### ✅ COMPLETED FEATURES
- [x] Universe download and fundamental filtering (reuses shared modules)
- [x] TRF dark-pool volume correction
- [x] AccumulationAnalyzer with Vol Z-Score, RV, and Absorption Ratio
- [x] Bollinger Band squeeze and inside-day detection
- [x] Pre-breakout scoring (0–100)
- [x] Rolling memory / Phase 1.5 early warning detection
- [x] Chunked OHLCV download with checkpointing (Raspberry Pi-friendly)
- [x] Diagnostic output (`08_absorption_diagnostics.csv`) with per-ticker pass/fail reasons
- [x] Telegram + Email alert dispatch
- [x] Results path injected via constructor (refactoring.md Issue 2 — already fixed)

---

### 🔴 CRITICAL BUGS (fix before any threshold tuning)

- [x] **NaN-safe filter guards** — Added `np.isnan()` guard at top of `_check_accumulation` before all comparisons. `accumulation_analyzer.py:324`
- [x] **`vol_zscore > 0` guard before AR calculation** — `ar = vol_zscore / rv if (rv > 0 and vol_zscore > 0) else 0.0`. `accumulation_analyzer.py:257`
- [x] **Fix async double-call in `alerts.py`** — Replaced per-channel loop with single `send_notification(channels=['telegram','email'])` call; removed `finally: close()` from `_send_notifications`; added explicit `EMPS3AlertSender.close()` method called from pipeline `finally` block. `alerts.py, emps3_pipeline.py`

---

### 🔴 THRESHOLD RECALIBRATION (required to produce any signals)

Based on 37-run diagnostic analysis (see `signal-analysis.md`), current thresholds made it statistically near-impossible to pass all gates simultaneously:

- [x] **Raise `atr_ratio` hard cutoff**: `0.02` → `0.04` (2% → 4%). Moved to `config.max_atr_ratio`; was hardcoded in analyzer. `config.py:33, accumulation_analyzer.py:335`
- [x] **Replace 52-week high gate with 20-day high gate**: `max_distance_from_resistance` now applies to `dist_local_high` (20-day high). New threshold: `0.15`. `config.py:36, accumulation_analyzer.py:349`
- [x] **Relax daily price range gate**: `max_price_impact` `0.03` → `0.05`. `config.py:31`

Expected outcome after these three changes: **3–10 candidates per run** on normal market days.

---

### 🔴 PHASE 2 ALERT INVESTIGATION (2026-08-14/15)

Stage 3 (`AccumulationAnalyzer`) and Phase 1 (`07_phase1_watchlist.csv`) work — the pipeline was
producing daily candidates. Phase 2 (`08_phase2_alerts.csv`) did not: **1 alert in ~10 weeks**
of production runs (2026-06-02 → 2026-08-13).

**Root cause:** the deprecated `EMPS3Pipeline` shim (since it started delegating wholesale to
`EMPS2Pipeline(analyzer_type='accumulation')`, commit `931a9c5`, 2026-06-13) was silently reusing
`p06_emps2`'s shared `RollingMemoryConfig` for Phase 2 detection — no accumulation-mode-specific
override existed. That config's `max_phase2_lag_days=7` is mechanically incompatible with
`phase1_min_appearances=5` inside a 14-day lookback: a ticker typically doesn't rack up its 5th
appearance (and become Phase-1-eligible) until 8-14 days after `first_seen`, so it usually blows
past the 7-day lag cap before it's even eligible. Confirmed by ablation replay against production
data. Same root cause independently found and fixed for p06_emps2 — see
`p06_emps2/docs/TIMING_ANALYSIS.md` 2026-08-14 re-measurement.

- [x] **Decouple p10's Phase 2 config from p06's** — `emps3_pipeline.py` now builds its own
  `RollingMemoryConfig` via `_p10_rolling_memory_config()` instead of inheriting whatever
  `EMPS2PipelineConfig.create_default()` uses.
- [x] **Fix the mechanical lag-cap bug**: `max_phase2_lag_days` `7` → `10` (mirrors the p06 fix;
  same structural cause, independent of universe/quality differences).
- [x] **Grid-swept `vol_zscore` / `vol_acceleration` / `drift` for a p10-specific quality
  improvement** — no combination tested beat the p06-inherited defaults (`vol_zscore=3.0`,
  `vol_acceleration=1.3`, `drift=5.0`) on 10d forward returns; every looser variant traded more
  volume for a worse (more negative) mean return. Left at defaults; **not resolved**, see below.

**Still open — structural mismatch, not a tuning problem:** even after the lag fix, p10's Phase 2
quality is weak (10d: 60% win / **−1.8%** mean; 20d: 50% win / **−6.0%** mean, n=10-11 — small,
noisy sample, mostly clustered on one date). p06's PREMIUM/HIGH price-drift heuristic is inverted
for p10 (PREMIUM alerts underperformed HIGH in the same test set) — a pullback during a genuine
momentum accumulation (p06's thesis) reads as "institutional buying on weakness", but a pullback
during a low-volatility squeeze (p10's thesis) more plausibly reads as "the squeeze is failing".
p10's Phase 2 gate (`vol_zscore≥3.0`) also confirms a breakout **after** it has already spiked,
which contradicts the "catch it before the spike" precursor premise Stage 3 is built around.
Fixing this needs a genuinely different Phase 2 definition for p10, not a retuned copy of p06's —
see the dormant `EMPS3RollingMemoryScanner`/Phase 1.5 (trend-based: ATR contracting + vol z-score
rising, no single-day spike) below as the likely starting point, plus the wiring/output-contract
and test-coverage work needed to bring it back live.

---

### 🟡 SIGNAL QUALITY IMPROVEMENTS (after bugs and thresholds are fixed)

- [ ] **Intraday range compression** — Compute price compression from 1h bar ranges (std of recent 20 intraday ranges) rather than the single daily H-L bar. The coiled spring effect is intraday; the daily bar is too coarse. Adds ~2h implementation.
- [ ] **Tightening trend confirmation** — Add a soft check: daily bar ranges have been contracting over the last 5 days (linear slope of ranges < −0.0005). Reduces false positives on single-day quiet bars that don't represent a sustained setup. Adds to scoring, not a hard gate.
- [ ] **BB width trend check** — Supplement the `bb_squeeze` flag (fires only at 12-month extreme) with a "BB contracting" flag: `bb_width[-1] < bb_width[-5]`. Fires more frequently as a leading indicator.
- [ ] **Slope magnitude filter in Phase 1.5** — Add minimum magnitude to ATR slope: `abs(atr_slope) > 0.001`. Current threshold `max_atr_slope: -0.0001` accepts noise-level slopes. `rolling_memory.py`, `config.py:58`

---

### 🟡 CODE QUALITY (non-blocking but needed)

- [x] **Remove dead `trf_surge` variable** — Already removed when `AccumulationAnalyzer` was moved to `p06_emps2/accumulation_analyzer.py`. Not present in the live code.
- [ ] **Extract shared modules from p06 imports** — `accumulation_analyzer.py` still imports `get_trf_correction_factor` from shared (currently OK), but verify no remaining direct p06 imports exist. Run import audit.
- [ ] **`EMPS3RollingMemoryConfig` deduplication** — Shares fields with p06 `RollingMemoryConfig`; create a `BaseRollingMemoryConfig` in `shared/config.py`.

---

## Technical Debt

- [x] Unit tests for `AccumulationAnalyzer._check_accumulation` edge cases — added `p06_emps2/tests/test_check_accumulation_edge_cases.py` (NaN inputs, negative zscore, XRXDW regression, good-candidate integration)
- [ ] No unit tests for `RollingMemoryScanner.detect_phase1_5_candidates` slope direction logic
- [ ] Diagnostic CSV column set is not validated — columns vary depending on which error path was taken, making aggregation fragile

## Known Issues

- **[RESOLVED]** ~~Pipeline has never produced a legitimate signal~~ — NaN guard, threshold recalibration, and all critical bugs fixed. Confirmed producing daily Stage 3 / Phase 1 candidates in production.
- **[UPDATED 2026-08-15]** `EMPS3RollingMemoryScanner` / Phase 1.5 (`rolling_memory.py`) is dead
  code, not merely dormant — it stopped being reachable once `EMPS3Pipeline` became a shim
  delegating wholesale to `EMPS2Pipeline` (commit `931a9c5`, 2026-06-13). Nothing in the live
  call path imports it; Stage 3 output alone won't revive it. See "🔴 PHASE 2 ALERT
  INVESTIGATION" above — reviving it (with test coverage and an output-contract fix) is the
  leading candidate for a real p10-specific Phase 2 signal.
- **Phase 2 alerts fire at a p06-inherited quality ceiling (~break-even), not a good one** — see
  "🔴 PHASE 2 ALERT INVESTIGATION" above. The 2026-08-15 lag-cap fix restores mechanical
  reachability but does not fix signal quality.

## Testing Requirements

- [x] Unit test: `test_nan_metrics_are_rejected` — ✅ passes
- [x] Unit test: `test_negative_vol_zscore_rejects` — ✅ passes, AR=0.0 confirmed
- [x] Integration test: `test_apply_filters_passes_good_candidate` — ✅ mocked DataManager, good candidate passes all gates
- [x] Regression test: `test_low_price_warrant_nan_metrics_regression` — ✅ XRXDW-like warrant with NaN RV is now rejected
- [x] **37-scenario threshold calibration suite** — `tests/test_threshold_calibration.py` — Groups A–I covering all 7 filter gates from both sides plus grid-sweep comparisons — ✅ all 37 pass (2026-06-13)

## Documentation Updates

- [x] `signal-analysis.md` — Root cause analysis and proposed changes (created 2026-05-20)
- [ ] Update `pipeline-specification.md` thresholds table after recalibration is deployed
- [ ] Add a "Signal Statistics" section to `README.md` once the pipeline is producing real signals
