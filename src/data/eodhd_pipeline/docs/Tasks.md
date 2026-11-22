# Tasks

## Implementation Status

### ✅ COMPLETED
- [x] Client: symbol lists, bars, splits, dividends, `/user`; pacing, retries, quota/plan errors (2026-11-22)
- [x] Universe with stable ids; resumable download with budget, quota and free-plan checks
- [x] Raw-basis conversion (volume back to as traded, dividends from `unadjustedValue`)
- [x] Store in `FmpStore` reader shape; mapping to FMP vintage ids; delisted coverage table
- [x] Cross-check: adjustment against `adjusted_close`; closes, volumes, splits, dividends against FMP
- [x] CLI `quota | universe | download | build | crosscheck | status`
- [x] Unit tests on a mock API (97% coverage); pyright, mypy, flake8 clean
- [x] Free-key dry run in a scratch root: universe 22,595 codes; 5 stocks end to end; closes and volumes equal to
      FMP on 4 pairs. Found and fixed: the daily reserve swallowed small quotas; the free plan cuts a whole-history
      request without a warning (now detected via `/user`)
- [x] Sharadar `eodhd-check` uses this client (one copy of the code)

### 🔄 WAITING FOR THE OWNER
- [ ] Buy one EODHD EOD month ($19.99), then follow the [Runbook](Runbook.md)

### 🚀 PLANNED — on purchase day (needs the paid key)
- [ ] Confirm the conversion facts on real data: split format, `unadjustedValue`, split-adjusted volume (crosscheck)
- [ ] Measure raw size, download time and failures; fill the [VERIFY] items in Requirements
- [ ] Record the delisted coverage per period (`status`) in the project memory, not in git

### 🚀 PLANNED — after the download
- [ ] Rerun the survivor-bias question on the recorded results (quality-value, momentum, falling knife) with the
      delisted names included
- [ ] Delisting returns: EODHD has none; decide on a rule (last close, or a haircut for bankruptcy-like ends)
- [ ] Quality flags (zero volume runs, stale prices, jumps without a split) if the cross-check shows a need

## Known Issues
- ADRs are typed "Common Stock" by EODHD and come in as `stock`; P24's plain-symbol filter drops most (5-letter
  `...Y` symbols stay).
- A code's meaning can change between symbol-list downloads (a live `ABC` delists and becomes `ABC_old`, a new
  company takes `ABC`). Ids are per code, so a later `universe` run can give the old company a second id. Within
  one purchase month this is rare and not detected.
- No delisting reasons, no deal consideration: the delisting return is unknown.
- No OTC history: the post-delisting life of a stock is absent, as in the other stores.

## Testing Requirements
- [x] Client error classes, retries, pacing, quota
- [x] Conversion rules (splits, dividends, raw volume)
- [x] Resume, budget, quota and plan stops
- [x] Store reads, P24 helpers on the store, mapping, cross-checks, CLI exit codes
- [ ] Reader parity on real delisted names (paid key)

## Documentation Updates
- [ ] After purchase day: measured sizes and times in Requirements, cross-check numbers in the memory
