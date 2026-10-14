# FMP Historical Data Pipeline — Decisions Log

Decisions D1–D11 are fixed by the spec (`docs/FMP_DATA_PIPELINE_SPEC-2.md` §2) and are not repeated here. This
log records everything decided since: recon findings, owner decisions, and choices made while building. Each entry
says what was decided and why. Package: `src/data/fmp_pipeline/`.

## Recon and owner decisions (2026-10-13)

| ID | Decision | Why |
|---|---|---|
| D12 | `adjust="legacy"` means `mode="total"`. V-YF-XCHECK compares our `close_total` with legacy files, not `close_split`. | Finding A (plan §1.1): the legacy yfinance cache is split **and** dividend adjusted (`auto_adjust=True`), has no `adjClose` column, and is not reproducible. |
| Q1 | Package lives at `src/data/fmp_pipeline/`. | Keeps the `src/<module>/docs/` + `tests/` convention and absolute imports. |
| Q2 | `data_root` = `DATA_CACHE_DIR/fmp-data`. | Co-located with the existing cache. `DATA_CACHE_DIR` resolves to `d:/data-cache` as of 2026-10-14. |
| Q3' | No configured backup path; `fmp backup` / `fmp verify --final` prompt for it at runtime. | Backups go to a manually connected USB HDD with no stable drive letter (spec §14 already says "the command asks"). |
| Q3'' | `fmp publish` stages a vintage directly on `R:` (the Pi's SSD) and verifies checksums there; the Pi switches `current.txt` locally. | One network hop instead of two. There is no NAS mount on the laptop. |
| Q4 | `increments.writer_host` = the Pi. | Always online; the laptop is not. |
| Q5 | No migration of the 147 `get_ohlcv` callers. | They see zero numeric change under D12. |
| Q6 | DuckDB approved as a new dependency on Windows and the Pi. | Spec D1. A prebuilt aarch64 manylinux wheel exists. Not installed yet: no code needs it until WP-6. |

## WP-7 — `adjust()` (2026-10-14)

| ID | Decision | Why |
|---|---|---|
| D13 | Before computing `DF(D) = 1 - d / C_prev`, `C_prev` is restated on the share basis of `D` (divided by every split ratio with `C_prev`'s date < S ≤ D). | Spec §8.2 is silent on a split between `C_prev` and `D`, and the commonest case is a dividend whose ex-date **is** the split date. There, `d` is per post-split share and the raw `C_prev` is a pre-split price; without the restatement the dividend is overstated by the split ratio. |
| D14 | A dividend with no earlier bar gets `DF = 1`. Exact duplicate event rows are collapsed; two *different* dividends on one ex-date (regular + special) are both applied; two *different* split ratios on one date raise `ValueError`. | A factor only affects earlier bars, and there are none. Duplicates are a data artefact; contradictory splits cannot be resolved without guessing (D11). |
| D15 | `adjust()` ignores events dated after the last bar; `yfinance_to_raw()` does **not** ignore splits after the last bar. | Spec §8.2 for the first. For the second, yfinance has already applied every split up to download time, so all of them must be undone (spec §10.2.3, gap case §10.2.5). |
| D16 | Invalid inputs raise `ValueError` instead of being clamped: a non-positive or non-finite split ratio, a negative dividend, or a dividend ≥ the restated previous close. | D11: never silently alter source data. The builder catches and flags. |
| D17 | Adjustment compares calendar dates (bar timestamp normalized; tz-aware indexes use local wall-clock). `adjust()` accepts optional `reference_closes` (raw daily closes) for intraday `C_prev`. | One implementation for daily and intraday. Without daily closes, an extended-hours print could become `C_prev`. |
| D18 | Event tables passed to `adjust()` use neutral column names: splits `date, numerator, denominator` (FMP's own) and dividends `date, amount`. The builder maps FMP's `dividend` field to `amount`. | Keeps `adjust()` source-agnostic (FMP and yfinance increments feed the same function). Which FMP field is raw was settled by the probe (D19). |

## WP-1 — free-key probe (2026-10-14)

Full evidence: `docs/fmp_probe_report.md`.

| ID | Decision | Why |
|---|---|---|
| D19 | FMP `dividends.dividend` is the raw amount used for `DF`; `adjDividend` is never used for adjustment. | Probe F1: AAPL 2020-08-07 `dividend` = 0.82 (raw), `adjDividend` = 0.205. Probe F4: with `dividend`, our `adjust()` matches FMP `dividend-adjusted` within 2.5e-5. |
| D20 | The builder maps price fields **by endpoint**: `non-split-adjusted` → `adjOpen`/`adjHigh`/`adjLow`/`adjClose` are the **raw** OHLC; `full` → `open`…`close`. Every price is cast to float64. | Probe F2/F3: the raw endpoint uses `adj*` names for unadjusted prices; values arrive as a mix of JSON ints and floats. |
| D21 | The client validates `from`/`to` itself and the builder checks returned date ranges against requested ones. | Probe F7: an unparseable `from` is silently ignored and the default window is served. |
| D22 | Bandwidth is counted from streamed bytes, both wire (gzip) and decoded; `Content-Length` is never trusted. | Probe F10: responses are gzip + chunked with no `Content-Length`; decoded is ~7× wire. |
| D23 | No rate limiting is derived from response headers. | Probe F11: FMP sends no rate-limit headers. |
| D24 | The probe uses its own small synchronous client (`probe/probe_client.py`) with a disk cache and a cross-run call ledger. WP-2 builds the async production client and promotes `classify()` into it. | The probe needs idempotence and a hard call budget, not throughput. |

## WP-2 — client, rate limiter, manifest, runner (2026-10-14)

| ID | Decision | Why |
|---|---|---|
| D25 | The rate limiter hands out grants evenly spaced at `60 / per_minute` s with **no burst capacity**. | Spec §6.4: "do not exceed the limit even briefly; bursts count". A token bucket with any capacity can exceed the per-minute rate inside a shorter window. FMP gives no headers to adapt from (F11). |
| D26 | Response bodies are held in memory per request, not streamed to a disk file by the client; the sink (WP-5 archive) writes them. | Measured sizes are small: a full 25-year daily history is ~1.1 MB decoded (171 B/row, F10), and intraday windows are bounded by the row cap. The spec's "stream large bodies to disk" is satisfied at the archive layer. Revisit if the paid probe shows multi-hundred-MB bodies (`stock-list` is the candidate). |
| D27 | Manifest crash-atomicity comes from SQLite (WAL, `synchronous=FULL`, one transaction per state change), not temp-file + rename. `fsutil.atomic_write_bytes` (with the Windows `PermissionError` retry) is for plain files. | Renaming a live SQLite database is unsafe; its journal already gives all-or-nothing commits that survive a kill (proven by the cross-process kill test). |
| D28 | Per task: **sink first, then manifest** — at-least-once delivery. | A kill between the two re-fetches the task; the sink may see a response twice but never loses one. The archive is append-only and the builder keeps the newest line per request (spec §6.6), so duplicates are harmless. The kill tests bound duplicates by the number of workers. |
| D29 | Classification → status: `not_found` → `api_error`; `unauthorized` and exhausted retries (`rate_limited`/`server_error`/`network_error`) → `failed` (re-runnable via `requeue_failed`). Quota or auth stop **releases** the in-flight task to `pending` without counting the attempt. `AuthAbort` after 10 consecutive 401/403. | Spec §6.2 status list has no `not_found`; a stop is not the task's fault. 10 is low enough to stop fast on a revoked key, high enough not to trip on a few restricted endpoints. |
| D30 | The bandwidth guard counts **every attempt** (retries and error bodies included), checks before each claim, and re-syncs its total from `bandwidth_log` every 60 s (in-memory additions between). Basis `decoded` until the paid probe proves FMP counts wire bytes. | Every byte FMP sends counts against the cap. A SUM over a month of `bandwidth_log` before each of ~10 claims/s would be wasteful. |
| D31 | Added a `run_events` table (start, stop reason, quota time) to the manifest. | Spec §6.2 requires "record the time" of a quota stop; a dedicated table keeps `tasks` per spec. |

## WP-5 — raw archive (2026-10-14)

| ID | Decision | Why |
|---|---|---|
| D32 | Symbols that differ only by case are told apart by a rule on the symbol alone: a file name containing a lower-case letter gets `~` + the first 8 hex digits of the symbol's sha256 (`abc` → `abc~<hash>`, `ABC` → `ABC`). `find_case_collisions` stays as a plan-time assertion. Encoding is reversible and canonical (`decode_symbol` rejects anything `encode_symbol` would not produce). | The plan's "plan-time detector that disambiguates colliding symbols" makes a file name depend on the rest of the universe, so a symbol added in a later dump would rename an existing file. The rule-based suffix never changes a name, and the 32-bit hash leaves only a theoretical collision, which the assertion catches. |
| D33 | Every append is one gzip member written with a single `write`, then fsynced. The first append to a file in each process streams the file's members and truncates a torn tail (bytes after the last member whose CRC and length check out). Later appends trust the length the writer itself recorded. | The spec requires "writes are atomic"; an append to a shared file cannot be made atomic by rename. A torn tail can only come from an interrupted write, whose task was never marked done (D28), so dropping it loses nothing. Scanning only on first touch costs one decompression per resumed file, not per append. |
| D34 | A missing archive root, free space below 2 GB (checked every 30 s) or any write error raises `ArchiveUnavailable`. The runner returns the task to `pending` and stops with reason `archive`. The archive never recreates a missing root. | Spec §13: if the archive path disappears mid-run, the run must pause and resume cleanly. Failing tasks one by one would mark the rest of the dump `failed`; recreating the root would silently write to the wrong volume. |
| D35 | Every final response is archived, not only successes (402s, error bodies, network errors with an empty body). `body` is the parsed JSON (re-serialized compactly); a non-JSON body goes to `body_text`. The `sha256` in the line and in the manifest is of the response bytes as received. | Spec §6.6 line format. Failures are evidence for `fmp verify` and the paid probe; the builder filters on `classification`. Hashing the received bytes lets the manifest be cross-checked against the archive without depending on JSON formatting. |

## WP-4 — planner, `fmp plan` (2026-10-14)

| ID | Decision | Why |
|---|---|---|
| D36 | Planning is **iterative and idempotent**. `fmp plan` first fetches the bootstrap lists (`stock-list`, `actively-trading-list`, `delisted-companies`, `symbol-change`, `etf-list`, `available-exchanges`, `financial-statement-symbol-list`, the six constituent lists) as ordinary priority-0 tasks through the `Downloader` into the archive, then plans only what it can decide: a candidate whose exchange is unknown gets just its `profile`; a symbol's intraday windows are written only once all its daily raw tasks are final; no intraday task is written while `intraday.max_rows` is unset. Everything else is projected as *deferred* and written by a later `fmp plan`. The dump's `as_of` is stored in the manifest by the first plan; pending tasks of symbols that leave the universe are marked `skipped`. `--dry-run` still fetches the bootstrap lists (they belong to the dump anyway) but adds no other task. | The universe can only be decided from data: `stock-list` may carry no exchange (resolved by `profile`), intraday windows depend on the first/last daily bar (spec §6.5.5) and on the row cap (paid probe). `as_of` is part of every date window, so a drifting `as_of` would change task ids and duplicate work. |
| D37 | Request windows sit on **fixed grids**: daily (and market-cap) history in calendar blocks of `daily_window_years` (default 5) from `daily_from` for **every** symbol, dropped only when they start after delisting + 30 days; intraday in blocks of `floor(max_rows / bars_per_day / 1.25)` weekdays counted from 1990-01-01, not clipped to the symbol (only `to` is capped at `as_of`). | Grid windows give the same `from`/`to` on every re-plan, so task ids are stable. Starting daily windows at the IPO date would trust FMP's `ipoDate`; an empty early window costs one request and a few bytes, and traffic, not request count, is the binding limit (spec §6.3). 5-year windows also keep one daily response far below any row cap FMP might have (unverified for 25-year ranges, probe F8). |
| D38 | A paged list continues while a page is **non-empty** (not while it is full), up to `max_pages` (2000), and stops when a page's body equals the previous page's. | The free key silently caps `limit` at 100 (probe), so a short page proves nothing. The two guards stop an endpoint that ignores `page` (`symbol-change` paging is [VERIFY]). |
| D39 | Intraday truncation is handled by a `Reviewer` hook in the `Downloader`: a response at ≥ 95% of the row cap is relabelled `possibly_truncated` **before** the sink (so the archive line is never a "successful" one and the builder ignores it), its two halves are inserted before the task is completed, and the task ends in the new final status `possibly_truncated`. Bisect-forward for the start of history (spec §6.5.4) is **not** built: windows already start at the first daily bar (§6.5.5) and `intraday.earliest` can hold a fixed 1h depth. | Inserting follow-ups before completion means a crash can only re-plan them (idempotent), never lose them. Without bisect, an empty early window costs one request and ~2 bytes; revisit only if the paid probe shows a per-symbol 1h depth far shorter than the daily history. |
| D40 | One writer per dump is enforced by an OS byte-range lock (`msvcrt.locking` / `flock`) on `<data_root>/locks/<dump_id>.lock`, taken by `fmp plan` (and later `fmp run`/`build`). Layout under `data_root`: `raw/<dump_id>/`, `manifests/<dump_id>.sqlite`, `reports/<dump_id>/`, `locks/`. | `fmp plan` already appends to the archive, which allows a single writer (D33). The OS releases the lock when the process dies, so a crash leaves no stale lock to clean up. |
| D41 | Configuration is `config/fmp/fmp.yaml` (spec App. B) with the universe rules under `fmp.universe` instead of a separate `config/universe.yaml`. Every key has a default in `planner/config.py`; unknown keys are errors. | The repository `.gitignore` only un-ignores `config/*/*.yaml`. One file keeps the owner's edits in one place; strict keys catch typos that would otherwise silently fall back to a default. |
| D42 | The projection counts finished tasks with their **real** bytes (follow-up pages and halves included) and the rest with catalogue estimates: bytes per row measured by the probe (F10) where it saw the endpoint, otherwise assumed and marked `*` in `budget_plan.md`. Disk need = remaining wire bytes × 1.25 (one gzip member per response) + 12 GB vintage + 10 GB build temp (spec §6.8). | Re-running `fmp plan` after each phase turns the projection into measurements without extra work; the report says plainly which totals still rest on assumptions. |

## `fmp run`, `fmp status`, `fmp retry-failed` (2026-10-14)

| ID | Decision | Why |
|---|---|---|
| D43 | `fmp run` executes **one block at a time** in spec §6.3.5 order (`block = priority // 100`), each with its own `Downloader` sharing one `BandwidthGuard`; the first run-level stop ends the run. `--phase N` runs one block (with a warning if earlier blocks still have pending tasks), `--tier T` runs one intraday tier, `--max-gb G` adds a per-run limit to the rolling-window hard stop. Before each block, free space on the archive volume must cover the last plan's projected remaining wire bytes of the blocks still in scope × `archive_overhead` + 20% (`plan_remaining_bytes` in the manifest's `meta`); a dump never planned without `--dry-run` is refused. Windows sleep is suppressed for the whole run (`SetThreadExecutionState`). The run does **not** re-plan by itself: it ends by telling the owner to re-run `fmp plan`, which writes the tasks that were waiting on the new data (D36). Exit codes: 0 done, 1 stopped, 2 usage/lock/not planned, 3 disk. | Block-at-a-time makes the strict order and the per-phase disk check (spec §6.8) explicit instead of relying only on priority sort. Keeping planning a separate, explicit step means every change to the manifest's task list is visible in a fresh `budget_plan.md` before it is downloaded. |
| D44 | Opening a `Manifest` reverts `running` tasks to `pending` only for the dump's writer; `fmp status` opens it with `recover=False` and without the lock. | Otherwise checking status during a live run would put in-flight tasks back to `pending`, and they would be fetched twice. |

## Dependencies

No new third-party dependency so far: WP-1 and WP-7 use `httpx`, `numpy` and `pandas`. All three are in
`requirements.txt` (`httpx>=0.28.1` added 2026-10-14; it was previously pinned only in `requirements-full.txt`
although P22 already imported it). DuckDB (Q6) and the `requirements-fmp-reader.txt` / `requirements-fmp-pipeline.txt` split
(plan §9.2) arrive with WP-6. WP-4 uses `PyYAML` (already in `requirements.txt`) for `config/fmp/fmp.yaml`.
