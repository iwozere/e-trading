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

## Dependencies

No new third-party dependency so far: WP-1 and WP-7 use `httpx`, `numpy` and `pandas`. All three are in
`requirements.txt` (`httpx>=0.28.1` added 2026-10-14; it was previously pinned only in `requirements-full.txt`
although P22 already imported it). DuckDB (Q6) and the `requirements-fmp-reader.txt` / `requirements-fmp-pipeline.txt` split
(plan §9.2) arrive with WP-5/WP-6.
