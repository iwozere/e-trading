# Tasks

## Implementation Status

### ✅ COMPLETED FEATURES
- [x] M0 Skeleton, `config.py`, module docs
- [x] M1 Indicators (ADX/DI/CCI via TA-Lib, PPO with own EMA) + tests
- [x] M2 Geometry (causal ZigZag, reference peak, low since peak, channel width) + causality tests
- [x] M3 Detector (all §4 conditions, one signal per knife, re-arm) + synthetic-knife and truncation tests
- [x] M4 Exits and trade simulation, tiered costs on as-traded price + tests
- [x] M5 Data layer (FMP vintage, split adjust, raw columns, weekly resample, eligibility, survivorship table) + tests
- [x] M6 Backtest runner (process pool, variants, drop-only + matched-random controls, SPY excess) + CLI
- [x] Validation module (metrics, month-block bootstrap, paired CI, cost repricing, gate) and `report.md`

### 🔄 IN PROGRESS
- [ ] First full default run and review with the owner (`results/p24_falling_knife/backtest/20261105-default/`)
- [ ] Owner decision: 2021 estimated delisted coverage 44.5% (< 50% threshold) → headline window 2021+ or 2022+?

### 🚀 PLANNED ENHANCEMENTS
- [ ] M7 Calibration grid on train (plateau pick, `calibration_frozen.json`), holdout run once, owner gate
- [x] M7 Book-example check (§7.6) — see spec §14.2
- [x] M7 `chart.py` 4-panel charts + `run_p24_explain.py --chart-dir`
- [ ] M7 20 sample charts in the report
- [x] M7 Portfolio layer (§6.3) `portfolio.py` + report section (realized equity; daily marks still open)
- [ ] Phase 2 (after gate): live scan, paper positions, notifications, signal-ledger harvester, `p24_specs.py`
- [ ] Phase 3: index mode, multi-timeframe confirmation, on-the-way-down statistics

## Technical Debt
- [ ] Bearish-divergence exit scans the bars since the last swing high each bar (O(hold²)); fine at 126 bars

## Known Issues
- FMP vintage has almost no delisted names before 2016 and thin coverage until 2021; pre-2021 results are an optimistic
  upper bound (bankrupt knives missing).
- Weekly bars resampled from daily; a delisted security's final short week is kept.
- Returns exclude dividends (split-adjusted prices).
- `JUMP_NO_SPLIT` + quiet-volume rule is a heuristic; a missing split traded on heavy volume would slip through.

## Testing Requirements
- [x] Unit tests for every module (63 tests)
- [ ] Truncation test on real vintage securities (currently synthetic + random-walk series)

## Documentation Updates
- [x] README, Requirements, Design, Tasks
- [ ] Add calibration and gate results once M7 runs
