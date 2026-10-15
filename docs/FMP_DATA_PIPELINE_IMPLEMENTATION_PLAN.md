# FMP Historical Data Pipeline — Implementation Plan v1

**Date:** 2026-10-13
**Based on:** `docs/FMP_DATA_PIPELINE_SPEC-2.md` v1.1 (2026-10-11)
**Status:** Phase 0a recon **done** (section 1). **WP-1 and WP-7 done 2026-10-14**: `src/data/fmp_pipeline/`
exists with `adjust/` and `probe/`; results in `docs/fmp_probe_report.md` and `docs/decisions.md` (D13–D24).
**WP-3 mock server, WP-2 client/manifest/downloader, WP-5 raw archive, WP-4 planner (`fmp plan`) `fmp run`/`status`/`retry-failed`, WP-6 builder (`fmp build`), WP-8 DataManager adapter and WP-9 increments done** (D25–D60). Next: WP-A validators (`fmp validate`, `fmp verify`)
(section 10). Section 1 refutes two spec assumptions and must be read before
any further code is written.

---

## 0. How this plan maps to the spec

The spec defines six phases and three gates. Only **Phase 0a, 0b and Phase 1** are buildable before the
owner buys Premium, and the spec is explicit that everything buildable MUST be finished before purchase.
This plan therefore front-loads all of it and stops at **G1**.

| Spec phase | This plan | Blocking? |
|---|---|---|
| 0a Repository recon | **Section 1 — complete** (findings below, `docs/datamanager_integration.md` still to be written up) | no |
| 0b Free-key probe | Section 4, Work Package WP-1 | needs the existing free `FMP_API_KEY` |
| 1 Build | Sections 5–9, WP-2 … WP-9 | no network beyond the free key |
| G1 | Section 10 | **owner decision: buy** |
| 2–6 | Section 11 (outline only) | after purchase |

Sections 2 and 3 are the decisions this plan needs from the owner *before* WP-1 starts, and the reuse map.

---

## 1. Phase 0a recon — results

Done against the live repo and the real cache on disk. Two spec assumptions are **refuted**; both change the
design, so they are listed first.

### 1.1 FINDING A (refutes spec §4.1, §9.3): legacy `close` is split **and dividend** adjusted

The spec's empirical test is: "compare `close` before 2020-08-31 with the known raw price (about $400 before
the 4:1 split). If legacy `close` is about $100, it is split-adjusted."

Measured from `c:/data-cache/ohlcv/AAPL/1d/2020.csv.gz`:

| Date | Legacy `close` | Raw (actual) | Split-only (raw/4) |
|---|---|---|---|
| 2020-08-28 | **121.1715** | ~499.23 | 124.8075 |

121.1715 / 124.8075 = 0.9709 — a ~2.9% residual drag, which is AAPL's cumulative dividend yield between
2020-08 and the file's download date (`2026-03-26`, per the sidecar `2020.metadata.json`). Confirmed in code:
`src/data/downloader/yahoo_data_downloader.py:347` passes `auto_adjust=True` ("explicit: use split/dividend-adjusted
prices"). Volume agrees too: 187,630,000 = raw 46.9M × 4, i.e. split-adjusted.

Consequences:

1. The legacy series is **total-return adjusted**, not split-adjusted. `adjust="legacy"` (spec §9.3) MUST map
   to `mode="total"`, not `"split"`. Writing it the way the spec expects would silently shift every legacy
   call by the dividend drag.
2. There is **no `adjClose` column** in the legacy files at all. The header is exactly
   `timestamp,open,high,low,close,volume`. The spec's expected case ("`close` = split-adjusted,
   `adjClose` = split-and-dividend-adjusted") does not exist on disk. The adapter must not invent an
   `adjClose` column by default — that would break the `src/data/ohlcv_contract.py` contract, which 147 files
   depend on.
3. Legacy data is **not reproducible**: a re-download of the same symbol/year produces different numbers,
   because the adjustment is computed as of download time. This is a strong independent argument for the
   spec's D3 (store raw, adjust at read time) and it means V-YF-XCHECK (§11) must compare our
   `close_total`, not `close_split`, against legacy files — comparing against `close_split` would report a
   systematic failure on every dividend-paying symbol.

**Action:** record in `docs/decisions.md` as D12; adjust §9.3's default and the V-YF-XCHECK comparand.

### 1.2 FINDING B: the repo already has FMP code, and one piece of it is broken

| Component | Location | State |
|---|---|---|
| `FMPDataDownloader` | `src/data/downloader/fmp_data_downloader.py` (822 lines) | sync `requests`, API key as **query param**, `get_ohlcv` calls `/stable/historical-price-full` at line 86 — **a dead endpoint (404)** |
| `FMPClient` (P22) | `src/ml/pipeline/p22_biotech_ma/ingest/fmp_client.py` (242 lines) | sync `httpx`, narrow (prices/ratios/estimates/search), key as query param |
| `get_with_retry` | `src/ml/pipeline/p22_biotech_ma/ingest/http_retry.py` | reusable retry helper |
| `fmp_limiter` | `src/ml/pipeline/p22_biotech_ma/ingest/rate_limits.py` | reusable rate limiter |

The dead-endpoint bug is pre-existing and outside this plan's scope, but it is in code this plan touches the
neighbourhood of — see section 12 (pre-existing issues).

### 1.3 FINDING C: live-verified FMP facts already in the repo (shrinks the Phase 0b probe)

P22's client docstring records facts verified against the live API on 2026-08-31. These pre-answer or
re-shape several spec §3/§5 `[VERIFY]` items:

- `/stable/historical-price-full` is **dead (404)**, even for valid symbols. The live endpoint is
  `/stable/historical-price-eod/full` — which the spec already uses. Confirms the spec's path, invalidates
  the repo's old one.
- `/stable/historical-price-eod/full` returns a **bare list** (no `historical` wrapper) of
  `{symbol, date, open, high, low, close, volume, change, changePercent, vwap}` — and **no `adjClose`
  field**. So FMP's `full` endpoint cannot be used to cross-check our adjClose; only
  `historical-price-eod/dividend-adjusted` can (spec §5.3 already says this; now confirmed as necessary,
  not optional).
- **The current free key shows per-symbol entitlement, not a sample-symbol list**: MRNA and PFE return 200
  while AMGN, GILD and SRPT return **402** ("not available under your current subscription") across every
  date range and both `/full` and `/light`. Spec §4.2 assumes "about 90 sample symbols"; the real
  restriction is a per-symbol allow-list that must be **discovered empirically**. WP-1 is designed around
  this (section 4.1).

### 1.4 DataManager, cache and symbol conventions

- **Entry point:** `DataManager.get_ohlcv(symbol, timeframe, start_date, end_date, force_refresh=False)` →
  `src/data/data_manager.py:159`. Also `get_ohlcv_batch` (:322), `get_fundamentals` (:802),
  `get_live_feed` (:749), `get_cache_stats`/`clear_cache` (:2342, :2346). Module-level factories
  `get_data_manager()` / `get_provider_selector()` at :2386 / :2369.
- **Return contract:** `src/data/ohlcv_contract.py` — tz-naive `DatetimeIndex` (bar open), lowercase
  `open, high, low, close, volume`, **no** `timestamp` column. Enforced by `_normalize_ohlcv` (:687).
- **Cache-on-miss flow** (:194–:250): load cache → detect prefix/suffix/intermediate gaps against
  `_GAP_TOLERANCE_BARS` → fetch only missing segments from the provider chain → merge-safe cache write.
  The FMP store slots in as a *cache-like layer ahead of the gap detector*, not as another provider.
- **Legacy cache:** `UnifiedCache` (`src/data/cache/unified_cache.py`), layout
  `<DATA_CACHE_DIR>/ohlcv/<SYMBOL>/<timeframe>/<year>.csv.gz` plus `<year>.metadata.json`.
  `DATA_CACHE_DIR` = `c:/data-cache` (`config/donotshare/donotshare.py:14`). Writes are already atomic
  (`_save_compressed_csv` → `os.replace`). **797 symbols** currently cached under `ohlcv/`.
- **Symbol convention:** cache directories are the symbol verbatim as callers pass it; indices keep `^`
  (`^GSPC`, `^VIX` used across P21/P20). Class-share separator convention to be confirmed per-symbol in
  WP-1 (yfinance uses `BRK-B`; FMP's form is a `[VERIFY]` item).
- **Regression surface:** `get_ohlcv` appears in **147 source files**; **55** construct a `DataManager`.
  But only **8 source files touch `UnifiedCache` directly** (`data_manager.py`, `ohlcv_contract.py`, four
  `src/data/utils/` scripts, `p17_penny_stocks/agents/market_agent.py`, and two test modules). That
  asymmetry is the good news: almost everything goes through `get_ohlcv`, so a single chokepoint plus a
  golden-output snapshot (section 8.3) covers the whole surface. The 8 direct cache users are the only
  places needing individual review.

### 1.5 Runtime environment (writer machine, measured)

| Item | Value |
|---|---|
| OS / arch | Windows 11 Pro 10.0.26200, AMD64 |
| Python | 3.13.1 (`.venv`) |
| pandas | 3.0.5 — **pandas is the dataframe library**; polars is not installed |
| pyarrow | 25.0.1 ✅ |
| duckdb | **not installed** — new dependency |
| orjson | **not installed** — optional new dependency |
| tzdata | 2025.3 ✅ (needed for §8.4 DST handling) |
| httpx / aiohttp | 0.28.1 / 3.14.3 — both present; no new HTTP dependency needed |
| pytest | 9.1.1 |
| Free disk | **C: 134.5 GB** (system), **D: 405.2 GB**, R: 352.7 GB (**the Pi's own SSD**, mapped to Windows — see below), G: 16.4 GB |

**Correction (owner-confirmed 2026-10-13): `R:\apps\e-trading` is the Raspberry Pi's own SSD, mapped to a
Windows drive letter — it is the live production reader, not a NAS and not a backup volume.** There is no
NAS mount visible from the laptop today; NAS backups are done manually via a separately-connected USB HDD
with no stable drive letter. This refutes the plan's original §13 reading of spec §6.8/§14 ("NAS (6 TB):
backup target and hand-off point") and changes two things:

1. **Owner decision (replaces Q3 below): `backup_path` is not a fixed config value.** `fmp backup` /
   `fmp verify --final` must **prompt for the backup path at invocation time** (spec §14 already says this:
   "The command asks for the backup path and runs the check there") rather than reading a configured
   default — Appendix B's example `backup_path: <NAS mount point...>` is illustrative, not a value this repo
   can fill in today.
2. **`R:` is a real candidate hand-off channel to the Pi**, separate from backup. Since it's a direct,
   already-mounted network path onto the Pi's own disk, `fmp publish` could write straight into a staging
   folder on `R:` instead of via an intermediate NAS. This needs an explicit decision — see Q3' below
   (renumbered; the original backup-path Q3 is resolved by point 1 above).

Decided: **`data_root` = `DATA_CACHE_DIR/fmp-data`** (owner's choice, 2026-10-13), *not* `D:\fmp`. Today
that's `c:/data-cache/fmp-data`, since `DATA_CACHE_DIR` (`config/donotshare/donotshare.py:14`) still
defaults to `c:/data-cache`. The owner is separately planning to move `DATA_CACHE_DIR` itself from C: to D: —
tracked as a **follow-up task, not part of this plan's scope**; `fmp-data` moves with it automatically
since it's defined relative to `DATA_CACHE_DIR`, not as a hardcoded path. C:'s 134.5 GB free still clears
spec §6.8's ~30–47 GB estimate for this working volume with margin, just less headroom than D: would give
until that move happens.

Packaging: `pyproject.toml` declares `dependencies = []` and has **no extras mechanism** — deps live in
`requirements*.txt`. The spec's `reader` / `pipeline` install profiles therefore become two new requirements
files, not extras (section 9.2).

`pytest.ini` applies `--cov=src/data --cov-fail-under=80` to the whole suite; a new package under
`src/data/` inherits that floor, and the spec asks for 85% on the core modules anyway.

---

## 2. Decisions needed from the owner before WP-1 — ALL RESOLVED (2026-10-13)

These were *not* questions the spec answers, so per §0 they were raised rather than guessed, and are now
all decided. Carry these into `docs/decisions.md` verbatim when WP-1 starts.

| # | Decision | Status | Why it matters |
|---|---|---|---|
| Q1 | Package location | **DECIDED: `src/data/fmp_pipeline/`** (owner, 2026-10-13) | Keeps the `src/<module>/docs/` + `tests/` convention (CLAUDE.md §10) and absolute-import rule intact |
| Q2 | `data_root` | **DECIDED: `DATA_CACHE_DIR/fmp-data`** (owner, 2026-10-13) | Co-locates with the existing `ohlcv/` cache. `DATA_CACHE_DIR` itself is moving C:→D: as a separate follow-up task; `fmp-data` inherits the move automatically |
| Q3' | Backup path *(replaces the original fixed-`backup_path` question — see the §1.5 correction above)* | **DECIDED: prompt at runtime, no config default**; manual NAS backup via a separately-connected USB HDD | Matches spec §14's own wording ("the command asks for the backup path") |
| Q3'' | `fmp publish` channel | **DECIDED: direct via `R:`** (owner, 2026-10-13) | One network hop instead of two: `fmp publish` stages into a folder on `R:` and verifies checksums there; the Pi, running locally, does the atomic `current.txt` switch itself (local-OS atomicity, not network-share atomicity). The manual NAS/HDD backup stays purely a backup leg (Q3'), not in the hot path to the reader |
| Q4 | `increments.writer_host` (§10.4 — exactly one writer) | **DECIDED: the Pi** (owner, 2026-10-13 — "always online") | Pi's `reader` install profile includes the increment ingester (§13 permits this); laptop and any other reader never write to `increments/` |
| Q5 | `adjust="legacy"` semantics | **DECIDED: `total`, no caller migration** (owner, 2026-10-13) | Matches the files on disk exactly (Finding A); 147 callers see zero numeric change. A future move to split-only default is explicitly out of this plan's scope |
| Q6 | DuckDB as a new dependency | **DECIDED: yes, on both Windows and the Pi** (owner, 2026-10-13) | Prebuilt aarch64 manylinux wheel confirmed for the Pi; justification recorded in `docs/decisions.md` per §13 |

---

## 3. Reuse map

Everything here is **read, not reimplemented**.

| Spec need | Existing component | Notes |
|---|---|---|
| Logger | `src.notification.logger.setup_logger(__name__)` | Project-wide; lazy `%s` formatting, never `print()` |
| Rate limiter | `src/ml/pipeline/p22_biotech_ma/ingest/rate_limits.fmp_limiter` | Promote to `src/data/fmp_pipeline/client/` as a shared token bucket; P22 keeps importing it from the new home via a shim, same pattern `raw_zone.py` already uses |
| HTTP retry/backoff | `src/ml/pipeline/p22_biotech_ma/ingest/http_retry.get_with_retry` | Sync-only; the async client (§6.4) needs its own loop, but reuse the classification and `Retry-After` logic |
| API key resolution | `FMPDataDownloader._get_config_value("FMP_API_KEY", ...)` | One place in the repo decides where the key comes from — keep it. **But** switch to the `apikey:` **header** (§6.1) |
| Raw landing pattern | `src/data/pipeline/raw_zone.py` | Content-addressed `<root>/<source>/<date>/<hash>.json.gz`, immutable, `known_from`-stamped. **Not** directly usable: §6.6 requires append-only `*.jsonl.gz` with concatenated gzip members, one file per endpoint+symbol, to avoid hundreds of thousands of tiny files on a network share. Reuse its atomic-write and manifest-row ideas; document the divergence in `decisions.md` |
| Job scheduling for `snapshot-universe` / `ingest-increments` | `src/data/pipeline/` registry (`base_plugin.PluginSpec`, `register_jobs.py`, 72 jobs) | §10.2.7 and §12.1 need recurring runs. Register them as `PluginSpec`s instead of inventing a scheduler; `ALLOWED_SCRIPT_DIRS` already permits `src/data/` |
| yfinance download with actions | `YahooDataDownloader` | **Cannot be reused as-is** for §10: it hardcodes `auto_adjust=True` (:347) and drops `Dividends`/`Stock Splits`. §10.2.2 needs `auto_adjust=False` + actions. Add a new narrow method or a parameter rather than flipping the default (Finding A) |
| Parquet I/O | `pyarrow` 25.0.1 | Already installed; §7.3 settings (zstd ~6, 100k row groups, `use_byte_stream_split`) are all supported |
| Trading calendar for V-CAL | `pandas_market_calendars` (added for P21) | Confirm it's in `requirements.txt`; if so, reuse for NYSE/NASDAQ expected-trading-days |
| Typecheck / lint gate | `scripts/typecheck.py`, ruff (line-length 120), pyright + mypy both clean | New package must land clean in both; this is CI-enforced |

**Not reused, and why:**

| Gap | Why not | Plan |
|---|---|---|
| `FMPDataDownloader` as the pipeline client | Sync `requests`, key in query string, no bandwidth accounting, no manifest, one dead endpoint | New async client in `fmp_pipeline/client/`. Leave `FMPDataDownloader` alone (it serves other callers); fix its dead endpoint as a separate task (section 12) |
| `UnifiedCache` as the vintage store | Year-partitioned `csv.gz`, no raw/adjusted separation, no `security_id` | D2 forbids touching it. It stays the `LegacyYfinanceCache` layer verbatim |
| `P22 FMPClient` | Scoped to three P22 endpoints; sync | Superseded by the new client for pipeline use; P22 keeps its own |

---

## 4. WP-1 — Free-key probe (`fmp probe-free`)

Deliverable: `docs/fmp_probe_report.md`. Budget ≤200 calls, idempotent, responses cached to disk so re-runs
cost zero calls.

### 4.1 Entitlement discovery must come first (changed by Finding C)

The spec assumes a known ~90-symbol sample set. Reality is a per-symbol 402 allow-list. So the probe's
**first 20 calls** run a cheap entitlement sweep: one minimal request per candidate symbol across a spread
of mega-cap / mid-cap / ETF / index / delisted names, recording 200 vs 402 vs 403. Everything after that
only uses symbols that came back 200, and the report states explicitly which spec `[VERIFY]` items are
**unanswerable on the free key** and deferred to the paid probe.

### 4.2 Probe tasks (mapping to spec §4.2)

| Call budget | Task | Spec ref |
|---|---|---|
| ~20 | Entitlement sweep (above) | new, from Finding C |
| ~45 | Record exact JSON schema of every §5 endpoint reachable on the free key: field names, types, null patterns, date formats. Confirm the `[VERIFY]` parameter names (`from`/`to`/`limit`/`page`) per endpoint | §4.2.1 |
| ~10 | Measure bytes/row per daily endpoint, **both** `bytes_wire` (gzip) and `bytes_decoded` | §4.2.2, §6.3.1 |
| ~15 | Pagination behaviour on `delisted-companies`, `cik-list`, `shares-float-all`: max page size, empty-page terminator | §4.2.3 |
| ~10 | Error shapes: invalid symbol, plan-restricted endpoint, empty result, and the **HTTP 200 + `{"Error Message": ...}`** case | §4.2.4 |
| ~5 | **AAPL dividend basis**: which of `dividend` / `adjDividend` equals **$0.82** on ex-date **2020-08-07**. This single fact fixes the adjClose formula input | §4.2.5, §8.3 |
| ~10 | AAPL cross-check of `non-split-adjusted` vs `full` vs `dividend-adjusted` around 2020-08-31 (4:1) and 2014-06-09 (7:1) | §4.2.6 |
| ~5 | Any intraday endpoint on a permitted symbol — schema if it works, "paid probe" if not | §4.2.7 |
| ~20 | **Ticker reuse**: from `delisted-companies` + `symbol-change`, find symbols with multiple entries / differing names or CIKs, and record what the price endpoints return (merged, one company, separate). 3–5 concrete examples | §4.2.8, §7.3 |
| ~20 | Slack for retries and follow-ups | |

### 4.3 Output

Besides the report: a set of **synthetic** fixtures under `src/data/fmp_pipeline/tests/fixtures/` that match
every recorded schema. No FMP response is ever committed (D5, §4.2 closing note).

---

## 5. WP-2 — Client, rate limiter, manifest

`src/data/fmp_pipeline/client/` and `manifest/`.

1. **Async client** (`httpx.AsyncClient`, already a dependency): `apikey:` request header only — never the
   query param, so the key cannot reach logs, URLs or the manifest. `Accept-Encoding: gzip`. Connect
   timeout 10 s, read 60 s. Large bodies streamed to disk. Records `bytes_wire` and `bytes_decoded` per
   response. Symbols URL-encoded (`.`, `-`, `^` are all real).
2. **Token bucket**: shared, default 600 rpm, 8–16 workers, must not burst past the limit.
3. **Retry/backoff**: exponential + jitter, ≤6 attempts, for network errors / timeouts / 429 / 5xx; honour
   `Retry-After`; **circuit breaker** pausing all workers 60 s after 20 consecutive 429/5xx. Non-429 4xx
   never retried, classified instead.
4. **Response classifier**: `ok` / `ok_empty` / `plan_restricted` (402/403 or the plan message) /
   `api_error` / `quota_exhausted`. `ok_empty` is a **valid outcome**, not an error. A quota message stops
   the run immediately. All-requests-401/403 aborts the run.
5. **Manifest** (SQLite, `manifest.sqlite`): `tasks` and `bandwidth_log` exactly per §6.2. `task_id` =
   `sha1(dump_id|endpoint|symbol|canonical_params)`. On restart `running` → `pending`. Atomic writes
   (temp → fsync → `os.replace`), with the Windows `PermissionError` retry-with-backoff wrapper §13 asks for.

**Tests:** manifest state machine; rate limiter under concurrency; retry/backoff/circuit-breaker; classifier
over every error shape from WP-1. ≥85% coverage.

---

## 6. WP-3 — Mock FMP server

`src/data/fmp_pipeline/tests/mock_server.py` (stdlib `http.server`, driven by the WP-1 fixtures). Must
simulate: normal paging, empty results, 200-with-error-body, 402/403, 429 with `Retry-After`, 5xx bursts,
slow responses, truncated intraday windows (row cap), gzip with and without `Content-Length`, and quota
exhaustion. **The entire downloader test suite runs against it with no network** — this is what makes
Phase 1 completable before purchase, so it is built before WP-4, not after.

---

## 7. WP-4 … WP-6 — Planner, archive, builder

### WP-4 Planner (`planner/`) → `fmp plan [--dry-run]`, `budget_plan.md`
- Universe assembly per §7.2 from `stock-list` ∪ `delisted-companies` ∪ `etf-list` ∪ `symbol-change`
  (old+new), filtered by exchange/type, `include_otc: false`, every exclusion logged with counts into
  `universe_report.md`.
- Task generation in the strict §6.3.5 order, **delisted symbols before active ones** within each block.
- Byte projection per block and per tier under **both** counting hypotheses (wire and decoded), from
  WP-1's measured bytes/row. Prints expected disk need next to actual free space.
- Tiers A–E per §6.3.3.
- **Intraday windowing** (§6.5): windows of `floor(max_rows / bars_per_day / 1.25)` trading days; runtime
  safeguard marks a response `possibly_truncated` at ≥95% of `max_rows`, halves the window and re-queues
  both halves; bisect-forward for the start of history; skip windows outside the symbol's known first/last
  daily bar dates. `max_rows` and `bars_per_day` are **inputs from the paid probe** — before that they come
  from config, and the mock server supplies them in tests.
- **Bandwidth hard stop**: 0.90 × 50 GB over the rolling 30-day window from `bandwidth_log`, basis
  `decoded` by default. Clean exit with a status report; never silently exceeds.

### WP-5 Archive (`archive/`)
`data/fmp/raw/<dump_id>/<endpoint_path>/<encoded_symbol>.jsonl.gz`, one line per response
(`{request (no key), fetched_at, http_status, body}`), appended as **concatenated gzip members**.
Append-only: a re-fetch adds a newer line, never overwrites. Checksums in the manifest. Free-disk check
before each phase.

**Windows-safe symbol encoding** is its own small, heavily tested module: percent-encode
`< > : " / \ | ? *`, trailing dots and spaces, and prefix reserved device names (`CON`, `PRN`, `AUX`, `NUL`,
`COM1`–`COM9`, `LPT1`–`LPT9` — several of which are or were real tickers) with `_`. Plus a plan-time
case-collision detector that disambiguates symbols differing only by case with a hash suffix. Price files
are named by `security_id` and sidestep this; the archive is named by encoded symbol and does not.

### WP-6 Builder (`builders/`) → `fmp build`
- Reads **only** the archive; newest successful line per request key wins.
- Deterministic and idempotent; row counts and per-file content hashes into `_meta/build_info.json` (V-REPRO).
- Builds into a temp dir, atomically switches `current.txt` (a **text file**, never a symlink) only after
  structural validation passes.
- One security at a time, <4 GB RSS, never loads a whole price table.
- Preserves unknown fields (`_extra` JSON column on statements) so a future need doesn't force a re-download.
- **`security_id` allocation** (§7.3): stable integers, never reused, allocated once and persisted — so a
  rebuild from the same archive yields the same ids. Identity segmentation per §7.3's rules, with
  `identity_uncertain = true` where evidence is contradictory (never guess).
- **`flags` bitmask** (§7.5): five flags, deterministic, thresholds from config, each with synthetic unit
  tests. `JUMP_NO_SPLIT` hits also written to `reports/missing_splits_candidates.csv`. **Never alters data** (D11).

---

## 8. WP-7 … WP-9 — adjust(), DataManager adapter, increments

### WP-7 `adjust/` — the numeric core
Pure, vectorized, no I/O: `adjust(prices_raw, splits, dividends, mode)` with `mode ∈ {raw, split, total}`,
implementing §8.2 exactly (`split_adj_factor`, `DF(D) = 1 - d / C_prev`, volume divided by the split factor,
events after the last bar handled explicitly). Plus `yfinance_to_raw()` per §10.2.3.

This is the highest-value, lowest-dependency work package — **build it first, in parallel with WP-1**, since
it needs no network and nothing else depends on its inputs. Unit tests with hand-computed vectors:
multiple splits, reverse split, dividend on a split date, dividend with no prior bar, duplicate events, no
events; and the §8.3 real-world vectors (AAPL 7:1 2014-06-09 and 4:1 2020-08-31, TSLA 5:1 / 3:1,
AMZN 20:1, GOOGL 20:1, NVDA 10:1, AAPL $0.82 ex-2020-08-07).

### WP-8 `datamanager_adapter/` — the riskiest package
Three layers behind the **unchanged** public interface: `FmpStore` → `IncrementStore` → `LegacyYfinanceCache`,
with resolution policy `fmp_first` per §9.2 (FMP bars up to its `last_price_date`, increments strictly after,
adjustments applied to the combined raw series using the deduplicated **union** of events; unknown symbols
fall through to the legacy cache and then to today's cache-on-miss loader, writing to the legacy location per D2).

Non-negotiable constraints, given 147 calling files (and the 8 direct cache users, §1.4):

1. **Default output is byte-identical to today** for anything served from the legacy cache. The adapter is
   **off by default behind a config flag**; turning it on is a deliberate act.
2. `adjust="legacy"` ≡ `mode="total"` (Finding A), and **no new columns appear by default** — `source`,
   `flags` and `exclude_suspect` are all opt-in, so `validate_normalized_ohlcv` keeps passing.
3. Cache keys include the vintage id **and** a hash of the events used, so a new dividend or vintage
   invalidates adjusted caches (§9.5).
4. Ambiguous symbol resolution **raises**, never guesses (§9.6).
5. Bulk cross-sectional read as a single DuckDB query.

**Regression strategy for the 147 files:** rather than hand-auditing them, capture a golden-output snapshot.
A script runs `get_ohlcv` over the 797 cached symbols × the timeframes present, hashes each frame, and
stores the hashes. The same script re-runs after the adapter lands; **any** hash change with the adapter
disabled is a bug. With the adapter enabled, changes are expected only for symbols present in the FMP
vintage, and are reviewed per-symbol. This is cheap, exhaustive over real data, and re-runnable at every
later phase.

### WP-9 `increments/` — yfinance layer (§10)
`auto_adjust=False` + actions (needs a new narrow method on `YahooDataDownloader`, *not* a default flip);
convert to raw before storing; append only bars strictly after the last stored bar; new splits/dividends to
`increments/events` and unioned at read time (FMP wins on conflict, with a warning). Single writer per Q4,
`increment_seq` in `increments/_meta.json`, atomic per-security writes, readers never write. Loud warning
when the 1h gap approaches 700 days (yfinance's ~730-day limit). Registered as a `PluginSpec` in
`src/data/pipeline/` for scheduled runs. Unit tests for the gap case §10.2.5 calls out explicitly.

---

## 9. Cross-cutting

### 9.1 Validators (`validators/`) → `fmp validate`, `fmp verify --final`
All 24 checks of §11 with their IDs and severities, emitting `validation_report.md` + `.json`, plus
`universe_report.md` and `coverage_report.md`. Two adjustments forced by this plan's findings:

- **V-YF-XCHECK compares `close_total`, not `close_split`**, against the legacy files (Finding A).
- V-SPLIT-KNOWN / V-ADJ-UNIT are runnable in Phase 1 against synthetic data; everything data-dependent is
  written and tested in Phase 1 but only *meaningful* in Phase 4.

`fmp verify --final` implements the §14 checklist, including the archive-wide `ARCHIVE.sha256` in
`sha256sum -c` format, the end-to-end gzip decompression test, and re-running the check **on the NAS copy**.

### 9.2 Package split and dependencies
Two new requirements files, since `pyproject.toml` has no extras:
`requirements-fmp-reader.txt` (pandas, pyarrow, duckdb, tzdata, yfinance — all with aarch64 wheels, verified
and recorded in `decisions.md`) and `requirements-fmp-pipeline.txt` (reader + httpx, aiosqlite if needed,
orjson). The Pi installs reader only.

### 9.3 Windows specifics
`pathlib` throughout; `encoding="utf-8"` on every text open; PowerShell examples in the runbook; `tzdata`
dependency made explicit for `zoneinfo`; `current.txt` as a text file; `os.replace` with a
`PermissionError` backoff retry (antivirus/indexer); sleep/hibernation suppressed during `fmp run` and
`fmp build` via `SetThreadExecutionState`, released on exit; `data_root` kept short (`D:\fmp`).

### 9.4 Module documentation (CLAUDE.md §12.4 — mandatory)
`src/data/fmp_pipeline/README.md` plus `docs/Requirements.md`, `docs/Design.md`, `docs/Tasks.md`, and a
`tests/` subfolder. Repo-level docs per spec §13: `fmp_probe_report.md`, `datamanager_integration.md`,
`decisions.md`, `storage_schema.md`, `runbook.md`.

---

## 10. Build order and G1 acceptance

Dependency-ordered; the two tracks at the top are independent and can run concurrently.

```
WP-7 adjust()            ──┐  (no network, no deps — start immediately)
WP-1 probe-free          ──┤
                           ├─> WP-3 mock server ─> WP-2 client+manifest ─> WP-4 planner
                           │                                                    │
                           └─> WP-5 archive ─────> WP-6 builder ────────────────┤
                                                        │                       │
                                                   WP-9 increments         WP-8 adapter
                                                        └──────> WP-A validators <┘
                                                                      │
                                                                     G1
```

**G1 hand-off package** (spec §15, Phase 1):

- [ ] `docs/fmp_probe_report.md` — every §3/§5 `[VERIFY]` item confirmed, refuted, or explicitly marked
      "requires paid probe"; ≤200 calls used
- [x] `docs/datamanager_integration.md` — section 1 of this plan, written up, with the full legacy-caller list
- [ ] `docs/decisions.md` — D12 (Finding A) and Q1–Q6 as recorded decisions
- [ ] Downloader green against the mock server: resume-after-kill, retry/backoff, circuit breaker,
      plan-restricted, quota-exhausted stop, bandwidth hard stop
- [x] `fmp plan --dry-run` output against both the mock and the free key: task counts and byte projections
      per block and tier, both counting hypotheses
- [ ] `adjust()` and `yfinance_to_raw()` passing the §8.3 vectors on synthetic data
- [x] Golden-output regression snapshot captured for the 797 cached symbols, adapter disabled → zero diffs
      (1,294 series, 2026-10-15)
- [ ] ≥85% coverage on adjust / manifest / planner / windowing; ruff + pyright + mypy clean via
      `scripts/typecheck.py`
- [ ] Open-questions list for the owner

Then the owner decides what to buy. **The agent never buys, cancels, or touches the FMP account** (D10).

---

## 11. Post-purchase phases (outline only)

| Phase | Work | Gate |
|---|---|---|
| 2 | `fmp probe-paid` in the first 30–60 min: intraday intervals actually served, 1h depth, `max_rows`, window caps, bar-label convention and time zone (incl. the mandatory DST-week windows of §8.4), extended-hours bars, intraday split-adjustment, and the wire-vs-decoded byte question via `fmp calibrate-usage --dashboard-gb X`. Must be **one command** producing both reports unattended | **G2** — owner approves `budget_plan.md` |
| 3 | `fmp run` — resumable, unattended, hard-stopped at 0.90 × cap, §6.3.5 order, delisted before active | — |
| 4 | `fmp build` → vintage; `fmp validate` → no FAIL | — |
| 5 | Adapter enabled; increments live on the writer host; `fmp publish` / `fmp sync` to the Pi — **channel depends on Q3'' below (`R:` direct vs. a true NAS leg)**, backup is the manually-connected HDD, prompted at runtime (Q3') | **G3** — `fmp verify --final` prints READY TO CANCEL |
| 6 | `fmp snapshot-universe` (no key needed) and `fmp ingest-increments` registered as scheduled `PluginSpec`s; `docs/runbook.md` for the 2029 repeat dump | — |

If `1hour` turns out not to be served on Premium, **stop and report at G2** — never silently substitute
another interval (§6.3.6).

---

## 12. Pre-existing issues found during recon

Per the diagnostics-hygiene rule, these are reported rather than silently fixed or silently ignored. None is
caused by this plan; all are in code adjacent to it.

1. **`src/data/downloader/fmp_data_downloader.py:86` calls a dead endpoint.**
   `get_ohlcv` requests `/stable/historical-price-full`, which P22 live-verified as **404** on 2026-08-31.
   Daily OHLCV via `FMPDataDownloader` is therefore broken for every caller. Fix is one line:
   `/stable/historical-price-eod/full` (note its response is a bare list, not a `{historical: [...]}`
   wrapper, so the parser needs checking too).
2. **FMP API key is sent as a query parameter** in both `FMPDataDownloader` and P22's `FMPClient`. FMP
   accepts the `apikey:` header, which keeps the key out of URLs, proxy logs and exception traces. Relevant
   given the leaked-credentials incident of 2026-09.
3. **`YahooDataDownloader` hardcodes `auto_adjust=True`** (:347) and drops the `Dividends` / `Stock Splits`
   columns (:383, :418). Not a bug today, but it means the entire legacy cache is irreproducible
   total-return data (Finding A), and §10 needs the opposite setting.

Want any of these fixed as part of this work, or tracked separately?
