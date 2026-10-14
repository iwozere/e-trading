# Tasks

Work packages follow `docs/FMP_DATA_PIPELINE_IMPLEMENTATION_PLAN.md` §10. Gate G1 (owner buys Premium) comes
only after every Phase 1 item below is done.

## Implementation Status

### ✅ COMPLETED FEATURES
- [x] Phase 0a recon — plan §1 (2026-10-13)
- [x] WP-7 `adjust()` + `yfinance_to_raw()` — all spec §8.3 test cases and vectors; matches FMP `full` exactly and
      `dividend-adjusted` within 2.5e-5 on real AAPL data (2026-10-14)
- [x] WP-1 `fmp probe-free` — 119/200 calls, `docs/fmp_probe_report.md`, 36 synthetic fixtures (2026-10-14)
- [x] `docs/decisions.md` with D12–D24 and Q1–Q6

### 🔄 IN PROGRESS
- [ ] `docs/datamanager_integration.md` — write up plan §1 with the full legacy-caller list (G1 item)

### 🚀 PLANNED ENHANCEMENTS
- [ ] WP-3 mock FMP server (`tests/mock_server.py`) driven by `tests/fixtures/`; must serve gzip + chunked
      without `Content-Length` (probe F10) and 402 plain-text bodies (F5)
- [ ] WP-2 async client, token bucket, retry/circuit breaker, SQLite manifest; promote `probe_client.classify`
- [ ] WP-4 planner + `fmp plan --dry-run` (byte projections from probe F10: ~25 B/row wire, ~171 B/row decoded)
- [ ] WP-5 raw archive + Windows-safe symbol encoding
- [ ] WP-6 builder: per-endpoint field mapping (D20), int/float casting, nullable `declarationDate`, future
      `delistedDate`s (F13), non-US exchange filter
- [ ] WP-8 DataManager adapter + golden-output snapshot of the 797 cached symbols
- [ ] WP-9 yfinance increments (needs a narrow `auto_adjust=False` method on `YahooDataDownloader`)
- [ ] WP-A validators, `fmp verify --final`
- [ ] Phase 2 `fmp probe-paid` covering the eight deferred items in `docs/fmp_probe_report.md`

## Technical Debt
- [ ] `probe_client` is synchronous and single-purpose by design (D24); do not reuse it for the bulk download.

## Known Issues
- The free key cannot distinguish an invalid symbol from a non-entitled one (probe F6).
- Ticker-reuse behaviour (§4.2.8) is unexamined: the free key restricts `symbol-change` and paging.

## Testing Requirements
- [x] Unit tests for `adjust()`, `yfinance_to_raw()`, probe client/analysis/report (85 tests, 99% coverage)
- [ ] Kill-and-resume and bandwidth-stop integration tests against the mock server (WP-2/WP-3)

## Documentation Updates
- [ ] `docs/storage_schema.md`, `docs/runbook.md` (WP-6, Phase 5)
