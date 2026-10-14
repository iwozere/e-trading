# Tasks

Work packages follow `docs/FMP_DATA_PIPELINE_IMPLEMENTATION_PLAN.md` §10. Gate G1 (owner buys Premium) comes
only after every Phase 1 item below is done.

## Implementation Status

### ✅ COMPLETED FEATURES
- [x] Phase 0a recon — plan §1 (2026-10-13)
- [x] WP-7 `adjust()` + `yfinance_to_raw()` — all spec §8.3 test cases and vectors; matches FMP `full` exactly and
      `dividend-adjusted` within 2.5e-5 on real AAPL data (2026-10-14)
- [x] WP-1 `fmp probe-free` — 119/200 calls, `docs/fmp_probe_report.md`, 36 synthetic fixtures (2026-10-14)
- [x] `docs/decisions.md` with D12–D35 and Q1–Q6
- [x] WP-3 mock FMP server (`tests/mock_server.py`): auth, 402 text bodies, paging, faults (429 + Retry-After,
      5xx bursts, 200 error body, slow), intraday row cap, gzip chunked / Content-Length, quota (2026-10-14)
- [x] WP-2 async `FmpClient` (header auth, gzip, wire/decoded byte counting, retries with jitter + Retry-After,
      circuit breaker, auth abort, date validation), no-burst `AsyncRateLimiter`, SQLite `Manifest`
      (spec §6.2 schema + `run_events`), `Downloader` worker pool with quota/auth/bandwidth stops, `fsutil`
      atomic writes; kill-and-resume proven in-process and across a real process kill (2026-10-14)
- [x] WP-5 raw archive (`archive/`): `RawArchive` sink (one gzip member per response, fsync, torn-tail repair,
      `ArchiveUnavailable` pauses the run), streaming reader + `latest_successful`, Windows-safe reversible symbol
      names (device names, case suffix, NTFS round-trip test), `ensure_free_space`; kill test repeated with the
      archive as the sink (2026-10-14)

### 🔄 IN PROGRESS
- [ ] `docs/datamanager_integration.md` — write up plan §1 with the full legacy-caller list (G1 item)

### 🚀 PLANNED ENHANCEMENTS
- [ ] WP-4 planner + `fmp plan --dry-run` (byte projections from probe F10: ~25 B/row wire, ~171 B/row decoded)
- [ ] WP-6 builder: per-endpoint field mapping (D20), int/float casting, nullable `declarationDate`, future
      `delistedDate`s (F13), non-US exchange filter
- [ ] WP-8 DataManager adapter + golden-output snapshot of the 797 cached symbols
- [ ] WP-9 yfinance increments (needs a narrow `auto_adjust=False` method on `YahooDataDownloader`)
- [ ] WP-A validators, `fmp verify --final`
- [ ] Phase 2 `fmp probe-paid` covering the eight deferred items in `docs/fmp_probe_report.md`

## Technical Debt
- [ ] `fmp run` CLI command (with Windows `SetThreadExecutionState` sleep suppression, plan §9.3) — lands with
      WP-4. Needs a per-dump lock file so two runs never append to the same archive (single writer).
- [ ] Structured JSON-lines logging and the once-a-minute progress summary / `fmp status` (spec §13).
- [ ] `probe_client` is synchronous and single-purpose by design (D24); do not reuse it for the bulk download.

## Known Issues
- Mock assumptions not yet verifiable (free key): quota body shape, Premium page-size cap, intraday row cap
  and truncation direction (newest rows kept), intraday field names. All are `MockConfig` knobs — set them from
  `fmp probe-paid` results.
- The free key cannot distinguish an invalid symbol from a non-entitled one (probe F6).
- Ticker-reuse behaviour (§4.2.8) is unexamined: the free key restricts `symbol-change` and paging.

## Testing Requirements
- [x] Unit tests for `adjust()`, `yfinance_to_raw()`, probe client/analysis/report (85 tests)
- [x] Mock server tests (29) — whole package 114 tests, 99% coverage
- [x] Client, manifest and runner tests incl. kill-and-resume and bandwidth stop — package total 181 tests, 99%
- [x] Archive and symbol-encoding tests (torn tails, CRC damage, chunk-spanning members, low disk, vanished root,
      write rollback, NTFS round trip) — package total 275 tests, 99%

## Documentation Updates
- [ ] `docs/storage_schema.md`, `docs/runbook.md` (WP-6, Phase 5)
