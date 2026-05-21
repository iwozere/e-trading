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

### 🟡 SIGNAL QUALITY IMPROVEMENTS (after bugs and thresholds are fixed)

- [ ] **Intraday range compression** — Compute price compression from 1h bar ranges (std of recent 20 intraday ranges) rather than the single daily H-L bar. The coiled spring effect is intraday; the daily bar is too coarse. Adds ~2h implementation.
- [ ] **Tightening trend confirmation** — Add a soft check: daily bar ranges have been contracting over the last 5 days (linear slope of ranges < −0.0005). Reduces false positives on single-day quiet bars that don't represent a sustained setup. Adds to scoring, not a hard gate.
- [ ] **BB width trend check** — Supplement the `bb_squeeze` flag (fires only at 12-month extreme) with a "BB contracting" flag: `bb_width[-1] < bb_width[-5]`. Fires more frequently as a leading indicator.
- [ ] **Slope magnitude filter in Phase 1.5** — Add minimum magnitude to ATR slope: `abs(atr_slope) > 0.001`. Current threshold `max_atr_slope: -0.0001` accepts noise-level slopes. `rolling_memory.py`, `config.py:58`

---

### 🟡 CODE QUALITY (non-blocking but needed)

- [ ] **Remove dead `trf_surge` variable** — Computed but never used in scoring or filtering. Either implement the intended rolling 3-day TRF comparison or delete. `accumulation_analyzer.py:283`
- [ ] **Extract shared modules from p06 imports** — `accumulation_analyzer.py` still imports `get_trf_correction_factor` from shared (currently OK), but verify no remaining direct p06 imports exist. Run import audit.
- [ ] **`EMPS3RollingMemoryConfig` deduplication** — Shares fields with p06 `RollingMemoryConfig`; create a `BaseRollingMemoryConfig` in `shared/config.py`.

---

## Technical Debt

- [ ] No unit tests for `AccumulationAnalyzer._check_accumulation` edge cases (NaN inputs, zero RV, single-bar history)
- [ ] No unit tests for `RollingMemoryScanner.detect_phase1_5_candidates` slope direction logic
- [ ] Diagnostic CSV column set is not validated — columns vary depending on which error path was taken, making aggregation fragile

## Known Issues

- **Pipeline has never produced a legitimate signal** (37 runs, 12,821 evaluations, 0 valid passes). All 4 historical "PASSED" records are for XRXDW (warrant, ~$0.11) where all metrics are NaN — they pass via the NaN comparison bug. See `signal-analysis.md` for full root cause analysis.
- **Phase 1.5 rolling memory is functionally dead** — it has never had any inputs because Stage 3 never passes any candidates. It will activate automatically once the Stage 3 fixes above are deployed.

## Testing Requirements

- [ ] Unit test: `test_nan_metrics_are_rejected` — pass a metrics dict with NaN vol_zscore, verify FAILED/nan_metrics
- [ ] Unit test: `test_negative_vol_zscore_rejects` — negative zscore should produce AR=0 and be rejected
- [ ] Integration test: run pipeline on a known historical date with relaxed thresholds, verify ≥1 candidate produced
- [ ] Regression test: confirm XRXDW (or any similar low-price warrant) no longer passes with NaN metrics

## Documentation Updates

- [x] `signal-analysis.md` — Root cause analysis and proposed changes (created 2026-05-20)
- [ ] Update `pipeline-specification.md` thresholds table after recalibration is deployed
- [ ] Add a "Signal Statistics" section to `README.md` once the pipeline is producing real signals
