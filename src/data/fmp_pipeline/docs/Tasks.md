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
- [x] WP-4 planner (`planner/`) + `fmp plan [--dry-run] [--offline]`: bootstrap lists via the downloader,
      universe assembly + `universe_report.md`, tiers A–E, task generation in §6.3.5 order (delisted first),
      fixed-grid daily/intraday windows, run-time paging and intraday truncation follow-ups, wire/decoded
      projection + schedule + disk need in `budget_plan.md`, per-dump writer lock; dry run against the mock and
      the free key (2026-10-14)
- [x] `docs/decisions.md` D36–D42
- [x] `fmp run` (`executor.py`: block-by-block in spec order, `--phase`/`--tier`/`--max-gb`, disk check per block
      from the plan's projection, `power.keep_awake`), `fmp status` (`status.py`, read-only), `fmp retry-failed`;
      D43–D44 (2026-10-14)
- [x] WP-6 builder (`builders/`) + `fmp build [--no-switch]`: universe and stable `security_id`s
      (`security_ids.json`, reuse segmentation behind a flag, symbol history), splits/dividends, raw daily and
      intraday Parquet per security with the five flags and `missing_splits_candidates.csv`, every other endpoint
      as a lossless record table (`_extra`), structural validation, temp dir + atomic `current.txt` switch,
      `build_info.json` with per-file sha256; plan → run → build chain on the mock; D45–D49 (2026-10-15)

### 🔄 IN PROGRESS
- [ ] `docs/datamanager_integration.md` — write up plan §1 with the full legacy-caller list (G1 item)

### 🚀 PLANNED ENHANCEMENTS
- [ ] WP-8 DataManager adapter + golden-output snapshot of the 797 cached symbols
- [ ] WP-9 yfinance increments (needs a narrow `auto_adjust=False` method on `YahooDataDownloader`)
- [ ] WP-A validators, `fmp verify --final`
- [ ] Phase 2 `fmp probe-paid` covering the eight deferred items in `docs/fmp_probe_report.md`

## Technical Debt
- [ ] Bisect-forward for the first intraday window (spec §6.5.4) deliberately not built (D39); revisit after the
      paid probe if 1h depth is per-symbol and much shorter than daily.
- [ ] `fmp plan` reads every profile file back from the archive on each run (~30k small files); cache resolved
      attributes in the manifest if that becomes slow.
- [ ] Structured JSON-lines logging and the once-a-minute progress summary during `fmp run` (spec §13);
      `fmp status` exists.
- [ ] Optional `fmp run --replan`: re-plan offline after blocks 0 and 3 so one command can run a whole dump.
- [ ] `probe_client` is synchronous and single-purpose by design (D24); do not reuse it for the bulk download.
- [ ] Flag thresholds (D45) are the spec defaults: calibrate on a sample of the real dump and record in
      decisions.md; `STALE` for 1h (35 bars) is a guess.
- [ ] Turn on `build.segment_reused_tickers` only after the paid probe answers §4.2 item 8 (D46).
- [ ] Optional consolidated yearly price partitions if DuckDB globbing over ~30k files is slow (spec §7.1).

## Known Issues
- Mock assumptions not yet verifiable (free key): quota body shape, Premium page-size cap, intraday row cap
  and truncation direction (newest rows kept), intraday field names. All are `MockConfig` knobs — set them from
  `fmp probe-paid` results.
- The free key cannot distinguish an invalid symbol from a non-entitled one (probe F6).
- Ticker-reuse behaviour (§4.2.8) is unexamined: the free key restricts `symbol-change` and paging.
- Field names of `stock-list`, `actively-trading-list`, `etf-list`, `symbol-change` and the constituent lists are
  unverified (402 on the free key); `planner/universe.py` accepts the common spellings. Check
  `universe_report.md` after the first paid `fmp plan`, including the "Exchanges seen" table.
- Intraday time zone and bar label (`build.intraday_tz`, `build.intraday_label`) are assumptions until the paid
  probe's DST-week check (spec §8.4).
- The probe-derived `profile` fixture has every boolean set (ETF, fund and ADR at once); tests that need a plain
  stock inject their own profile through `MockConfig.extra_fixtures`.

## Testing Requirements
- [x] Unit tests for `adjust()`, `yfinance_to_raw()`, probe client/analysis/report (85 tests)
- [x] Mock server tests (29) — whole package 114 tests, 99% coverage
- [x] Client, manifest and runner tests incl. kill-and-resume and bandwidth stop — package total 181 tests, 99%
- [x] Archive and symbol-encoding tests (torn tails, CRC damage, chunk-spanning members, low disk, vanished root,
      write rollback, NTFS round trip) — package total 275 tests, 99%
- [x] Planner tests: windowing grids, config loader, universe rules and tiers, generator order and deferrals,
      projection/schedule, follow-ups through the real `Downloader`, plan → run → re-plan cycle, CLI and lock
      — package total 354 tests; planner modules 97–100%
- [x] Executor/status/retry tests (block order, tier scope, `--max-gb`, disk refusal, unplanned dump, status
      reader during a live run, CLI exit codes, sleep suppression) — package total 365 tests, 99%
- [x] Builder tests: every flag with synthetic hits and near misses, id registry stability, segmentation on/off,
      rename chains, lossless record tables, window drops, DST conversion, clipping, structural validation,
      rejected builds, byte-identical rebuild, mock plan → run → build, CLI — package total 404 tests, 99%

## Documentation Updates
- [ ] `docs/storage_schema.md` (vintage tables and columns, `_extra`, flags, `security_id` rules; partly in
      `builders/` docstrings now), `docs/runbook.md` (Phase 5)
