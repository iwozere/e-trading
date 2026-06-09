# Tasks

## Implementation Status

### ✅ COMPLETED
- [x] Extended `EdgarDownloader` with 13F, Form 4, 13D/G methods
- [x] `OpenFigiMapper` — CUSIP→ticker with CSV.gz cache
- [x] `PositionDeltaCalculator` — Q-over-Q delta engine
- [x] `ExitScreener` — reduction threshold filter
- [x] `ConsensusDetector` — multi-institution overlap aggregator
- [x] `VolumeAnomalyDetector` — daily spike detection via DataManager
- [x] `Form4Monitor` — Form 4 + 13D/G daily watcher
- [x] `CompositeScorer` — weighted signal aggregator
- [x] `InstitutionalFlowPipeline` — daily orchestrator
- [x] `run_p18_scan.py` — scheduler entry point
- [x] Unit tests for all processors and scorer
- [x] README, Requirements, Design, Tasks documentation

### 🔄 NEXT STEPS (before first production run)

- [ ] **Scheduler registration** — run the SQL from README.md to register the
      daily job in the production database
- [ ] **Backfill** — run with `--as-of-date` for last 2 quarters to warm up the
      13F cache and consensus baseline:
      ```bash
      python run_p18_scan.py --as-of-date 2024-09-30 --force-refresh
      python run_p18_scan.py --as-of-date 2024-12-31 --force-refresh
      ```
- [ ] **Verify EDGAR EFTS response format** — the pipeline handles the expected
      JSON structure; run a test query and confirm field names match production
- [ ] **OPENFIGI_API_KEY** — register a free key and add to `.env` for 10× rate
      limit improvement during backfill

### 🚀 PHASE 5 — ADVANCED (Deferred)

- [ ] FINRA ATS dark pool data ingestion — monthly CSV from FINRA transparency portal
- [ ] Options flow — put/call ratio spike detection (requires paid data feed)
- [ ] IPO calendar scraper + underwriter 13F correlation
- [ ] Passive index fund filter — exclude institutions with >90% overlap to SPY
      holdings to separate mechanical sells from informational ones
- [ ] ML model — predict post-exit price trajectory from historical 13F patterns
- [ ] `watchlist_tickers` fuzzy match in `Form4Monitor.get_13dg_drops()`

## Known Limitations

- **13F lag**: 45-day filing deadline means signals confirm what happened in the
  prior quarter, not the current one.
- **Form 4 parsing coverage**: Not all Form 4 XML variants are tested. Unusual
  filing structures may result in zero rows being parsed — check `_parse_form4_xml`.
- **CUSIP edge cases**: Non-equity CUSIPs (bonds, preferred shares) are stored as
  `ticker=None` and excluded from scoring. This may miss convertible bond unwinds.
- **AUM proxy**: Total portfolio value from the infotable is used as an AUM proxy.
  Multi-manager platforms (family offices) may be under- or over-counted.

## Testing Requirements

- [ ] End-to-end test with a real historical quarter (e.g., Q4 2023)
- [ ] Integration test: full daily run against cached EDGAR data
- [ ] Performance test: time a full backfill run for 500 institutions
