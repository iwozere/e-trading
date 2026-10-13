# FMP Historical Data Pipeline: Technical Specification

Version 1.1, 2026-10-11 (v1.0 plus accepted review changes, see Appendix E). Audience: an autonomous programming agent. Owner: a single retail quant, personal use only.

---

## 0. How to read this document

- **MUST / SHOULD / MAY** are used in the RFC sense.
- Items marked **[VERIFY]** are facts taken from FMP's public documentation or pricing page on 2026-10-11 that have **not** been confirmed by live API calls. Phase 0 MUST confirm or refute each one and record the outcome in `docs/fmp_probe_report.md`. Where this spec and observed API behaviour disagree, observed behaviour wins; document every delta.
- Section 2 lists decisions that are already made. Do not reopen them and do not ask the owner about them.
- Section 4 defines **gates**: points where you MUST stop, report, and wait for the owner. Outside the gates, work autonomously and do not ask questions that this spec answers. When something is genuinely ambiguous, choose the default stated in this spec, record the choice in `docs/decisions.md`, and continue.
- You do not know the internal API of the owner's existing `DataManager`. Section 9 tells you how to discover it. Never guess it.

---

## 1. Context and goal

**Situation**

- The owner has a `DataManager` with a cache manager. OHLCV data (timeframe 1d only, US equities) comes from **yfinance** and is stored as `.csv.gz` files. When a requested series is missing from the cache, a loader downloads it from free sources and stores it.
- The owner will buy **FMP (Financial Modeling Prep) Premium for one month** (a second month is possible, see section 6.3), download the maximum useful history including **delisted tickers**, and then cancel. After that, the owner keeps updating **currently traded** tickers with the existing free-source loader. In 2 to 3 years the owner will repeat a full FMP dump.
- Because the subscription is time-boxed, **everything that can be built and tested before purchase MUST be finished before purchase**. The owner will buy only when the downloader is ready, tested, and has a reviewed download plan.

**Goal.** Deliver:

1. A resumable, bandwidth-aware **FMP downloader** that archives raw API responses.
2. A normalized local store (**Parquet + DuckDB**) built from the archive. The store is rebuildable from the archive at any time.
3. A **validation suite** that proves the data is complete and consistent before the subscription is cancelled.
4. **DataManager integration**: the new store becomes the base data source; existing yfinance data stays where it is and is used as an incremental and fallback layer.
5. **Maintenance tooling**: universe snapshots, yfinance incremental ingestion that is consistent with the FMP base, and a repeatable "vintage" procedure for the next FMP dump in 2 to 3 years.

**Scope of data (priority order)**

1. Symbol universe, delisted list, symbol changes, company profiles.
2. Corporate events: splits, dividends, earnings dates, M&A, IPOs.
3. Daily OHLCV, unadjusted (raw).
4. Historical index membership (S&P 500, Nasdaq, Dow Jones).
5. Fundamentals with filing dates (point-in-time safe).
6. Intraday **1-hour** OHLCV for the US universe.
7. Macro and calendars (treasury rates, economic indicators, exchange holidays and hours).

**Out of scope for this task**

- Any change to strategy logic, backtesting engines, or optimizers. The deliverable ends at the `DataManager` read interface (section 9). Backtester integration is a separate task.
- Real-time or live-trading data, order execution, and any non-US market.
- Correcting or deleting source values (see D11). Suspect data is flagged, never altered.

---

## 2. Decisions already made

| ID | Decision | Rationale |
|----|----------|-----------|
| D1 | Storage is **Parquet (zstd)** files queried through **DuckDB**. SQLite is used only for the downloader manifest and small metadata. | Columnar compression is several times smaller and faster than csv.gz; DuckDB queries Parquet without import. |
| D2 | Existing yfinance `.csv.gz` data is **not migrated, rewritten, or deleted**. It stays in place as a legacy source. | Owner's explicit instruction. |
| D3 | The base store holds **raw (unadjusted) prices plus separate corporate-event tables**. `close`, `adjClose` and all other adjusted fields are **computed by us at read time**; third-party adjusted values are never stored as the primary truth. | Every new dividend rewrites the whole adjusted history, so a stored adjClose goes stale. FMP and yfinance compute adjustments differently, so mixing them corrupts history. |
| D4 | Every FMP HTTP response is **archived raw** (gzip) before parsing. Parquet is a derived artifact. | After the subscription ends, the archive is the only copy of what was paid for. |
| D5 | Personal use only. No redistribution, no uploading data or raw responses to any hosted service or public repository. | FMP Terms of Service. |
| D6 | Each full FMP dump is a **vintage** (e.g. `2026-10`). Vintages are kept side by side; a pointer selects the active one. | The dump will be repeated in 2 to 3 years and results must stay reproducible. |
| D7 | Instruments get an **internal `security_id`**. Ticker symbols are attributes, not keys. | Tickers are reused and renamed. |
| D8 | The primary daily price source is FMP's **non-split-adjusted** endpoint. The `full` (split-adjusted) and `dividend-adjusted` daily endpoints are downloaded **only for a reconciliation sample** (section 6.3), not for the whole universe. | Bandwidth budget (see section 6.3). |
| D9 | Language is Python. Follow the conventions of the existing repository (dependency manager, formatter, test runner, dataframe library). | Fit with the owner's codebase. |
| D10 | Cancellation of the subscription is the **owner's** action and happens only after `fmp verify --final` passes (section 14). The agent never cancels, buys, or changes anything on the FMP account. | Safety. |
| D11 | The pipeline **never corrects, drops, or overwrites source values**. Suspected bad data (bad ticks, missing splits, OHLC violations) is only **flagged** in a `flags` column; consumers decide whether to exclude it (section 7.5). | Silent "fixes" destroy information and make results irreproducible. |

---

## 3. Facts about FMP (as of 2026-10-11) [VERIFY]

Source: FMP pricing page and API documentation. Re-check before purchase.

| Item | Free (Basic) | Starter | Premium | Ultimate |
|------|--------------|---------|---------|----------|
| Price, personal use | $0 | $29/mo ($19 billed annually) | **$69/mo** ($49 billed annually) | $139/mo ($99 billed annually) |
| Rate limit | 250 calls/day | 300 calls/min | **750 calls/min** | 3,000 calls/min |
| History depth | 5 years | 5 years | **30+ years** | 30+ years |
| Coverage | ~90 sample symbols (AAPL, TSLA, AMZN, ...) | US | US, UK, Canada | Global |
| Intraday charts | not included | not included | **included** (exact intervals unclear, see below) | included, plus **1-minute** |
| Bulk and batch endpoints | no | no | no (listed under Ultimate) | yes (`eod-bulk`, `profile-bulk`, and others) |
| Bandwidth cap, trailing 30 days | 500 MB | 20 GB | **50 GB** | 150 GB |

Critical unknowns that Phase 0 and the paid probe MUST resolve:

1. Which intraday intervals Premium actually serves (`1min`, `5min`, `15min`, `30min`, `1hour`, `4hour`). The pricing table is ambiguous. `1min` is listed as an Ultimate feature.
2. How far back `1hour` history goes. Community reports range from about 15 to about 20 years. Not confirmed.
3. The maximum number of rows returned by one intraday request and the maximum date window.
4. Whether the 50 GB cap is measured on compressed (wire) bytes or on decompressed JSON bytes.
5. Whether Premium includes all US exchanges, delisted symbols in the price endpoints, and `historical-*-constituent` endpoints.
6. Whether intraday bars are split-adjusted, and whether a `nonadjusted` parameter exists.
7. Bar-label convention (start vs end of bar), time zone of the `date` field, and whether pre-market and after-hours bars are included.

All authenticated calls use the base URL `https://financialmodelingprep.com/stable/`. Authorization MUST use the **request header** `apikey: <KEY>`, not the query parameter, so that the key never appears in URLs, logs, or the manifest. (The query-parameter form `?apikey=` is also accepted by FMP.)

---

## 4. Phases and gates

```
Phase 0a  Repository recon (no network)                          -> docs/datamanager_integration.md
Phase 0b  Free-key API probe (<= 200 calls)                      -> docs/fmp_probe_report.md
Phase 1   Build downloader, store, validators (offline + mocks)  -> tested code
GATE G1   Owner reviews probe report, plan, and dry run
          Owner buys Premium (or Ultimate)
Phase 2   Paid probe (first 30-60 minutes after purchase)        -> docs/fmp_paid_probe_report.md, budget_plan.md
GATE G2   Owner approves budget_plan.md (what to download in month 1)
Phase 3   Full download run (resumable, unattended)
Phase 4   Build Parquet store from the archive; run validation   -> validation_report.md
Phase 5   DataManager integration; yfinance incremental layer
GATE G3   `fmp verify --final` passes; owner may cancel
Phase 6   Maintenance: universe snapshots, increments, next vintage
```

### 4.1 Phase 0a: Repository recon (no network)

Before writing code, read the existing repository and write `docs/datamanager_integration.md` containing:

- Where `DataManager`, the cache manager, and the yfinance loader live; their public method signatures; how cache-on-miss works.
- Exact on-disk layout and schema of the legacy `.csv.gz` files: path pattern, columns, dtypes, index, timezone, date format, whether `close` is split-adjusted (yfinance default) and whether `adjClose` is present and how it is named.
- Verify empirically on one legacy file (e.g. AAPL): compare `close` before 2020-08-31 with the known raw price (about $400 before the 4:1 split on that date). If legacy `close` is about $100, it is **split-adjusted**; record this. This determines the default read semantics (section 9.3).
- Which symbol conventions the code uses (`BRK-B` vs `BRK.B`, index symbols with `^`).
- The dataframe library in use (pandas or polars) and the test framework.
- The runtime environment of **both machines** (section 13): on the Windows laptop, the CPU, RAM, Python version, free disk space on the internal and external SSDs, and the NAS mount; on the Raspberry Pi, the architecture (aarch64), Python version, and where the existing `DataManager` and the yfinance loader actually run today. Record installed versions of `pyarrow`, `duckdb`, the dataframe library, and the HTTP/JSON libraries, and check that the `reader` dependencies install on aarch64.
- Every caller of the cache manager, so you can prove backward compatibility later.

### 4.2 Phase 0b: Free-key probe

The free key allows 250 calls/day and about 90 sample symbols. Use at most **200 calls**, make the probe re-runnable and idempotent (responses cached), and write the findings to `docs/fmp_probe_report.md`. The probe's purposes:

1. Record the **exact JSON schema** (field names, types, null patterns, date formats) of every endpoint in section 5, using sample symbols. Parameter names marked [VERIFY] in section 5 must be confirmed.
2. Measure **bytes per row** for each daily endpoint (wire size with gzip and decompressed size).
3. Test pagination (`page`, `limit`) for list endpoints.
4. Determine error shapes: HTTP status plus body for an invalid symbol, a plan-restricted endpoint (expect HTTP 402/403 or a JSON error message), an exhausted quota, and an empty result. FMP may return HTTP 200 with a JSON error body such as `{"Error Message": "..."}`; treat that as an error, not data.
5. Determine the semantics of `dividends` fields on AAPL: which of `dividend` or `adjDividend` equals **$0.82** for the ex-date **2020-08-07** (the pre-split cash amount). Record which one is on the raw (unadjusted) basis. This determines the adjClose formula inputs (section 8).
6. Check whether `non-split-adjusted`, `full`, and `dividend-adjusted` agree with each other and with splits on AAPL around 2020-08-31, and around the 2014-06-09 7:1 split.
7. If any intraday endpoint works on a sample symbol with the free key, record its schema; otherwise note that intraday must be probed in Phase 2.
8. **Ticker reuse.** From `delisted-companies` and `symbol-change`, list symbols that appear more than once or whose entries show different names, CIKs, or non-overlapping date ranges. For every such symbol reachable with the free key (otherwise in Phase 2), record what the price endpoints return: one merged series, only one company's history, or separate histories. Record the result with 3 to 5 concrete examples; it decides how aggressively the segmentation rule in section 7.3 applies.

Do not commit any FMP response to any repository. Test fixtures MUST be **synthetic** files that match the recorded schemas.

### 4.3 Phase 1: Build

Build everything in sections 6 to 12 and 14. It MUST be developed against:

- the recorded schemas (synthetic fixtures),
- a local **mock FMP server** (section 6.7) that can simulate rate limiting, errors, truncation, plan restrictions, and large responses,
- the free key for a small number of real calls.

Acceptance for Phase 1 is in section 15. At the end, run `fmp plan --dry-run` against the mock and against the free key and attach the output to the hand-off.

### 4.4 Gates (the agent MUST stop here)

- **G1** (end of Phase 1): deliver `docs/fmp_probe_report.md`, the dry-run plan, the test results, and a list of open questions. The owner decides what to buy.
- **G2** (end of Phase 2): deliver `docs/fmp_paid_probe_report.md` and `budget_plan.md` (section 6.3). The owner approves or edits the plan before the long run starts.
- **G3** (end of Phase 4/5): deliver the validation report and the output of `fmp verify --final`. Only the owner cancels.

If Phase 2 is executed by the owner while the agent is not available, `fmp probe-paid` MUST be a single command that produces both reports without further input, and `fmp plan` MUST produce `budget_plan.md` automatically from the probe results.

---

## 5. Endpoint catalogue

All paths are relative to `https://financialmodelingprep.com/stable/`. All are GET requests returning JSON. List endpoints that accept `page` and `limit` MUST be paginated until an empty page is returned. Parameter names for date filters (`from`, `to`) and `limit` are [VERIFY] for each endpoint; the probe must confirm them. Priority: **P0** must have, **P1** should have, **P2** optional (download only if budget remains).

### 5.1 Universe and reference

| Priority | Endpoint | Params | Use |
|----------|----------|--------|-----|
| P0 | `stock-list` | none | All symbols (worldwide; filter to US exchanges, section 7.2) |
| P0 | `actively-trading-list` | none | Currently traded symbols; difference with `stock-list` helps identify inactive ones |
| P0 | `delisted-companies` | `page`, `limit` (max page size [VERIFY]) | All delisted companies: symbol, exchange, IPO date, delisting date |
| P0 | `symbol-change` | none or paging [VERIFY] | Ticker renames and corporate-action ticker changes |
| P0 | `etf-list` | none | ETF symbols |
| P0 | `profile` | `symbol` | Per-symbol profile: exchange, sector, industry, CIK, ISIN, CUSIP, IPO date, `isEtf`, `isAdr`, `isFund`, `isActivelyTrading`. Needed for **every** symbol in the universe |
| P1 | `cik-list` | `page`, `limit` | CIK registry |
| P1 | `available-exchanges`, `available-sectors`, `available-industries`, `available-countries` | none | Reference lists |
| P1 | `shares-float-all` | `page`, `limit` | Float and shares outstanding |
| P1 | `mergers-acquisitions-latest` (paged); `mergers-acquisitions-search` | `name` | M&A events (the search endpoint takes a company name) |
| P1 | `holidays-by-exchange`, `exchange-market-hours` | `exchange` (NASDAQ, NYSE) | Trading calendar for gap checks |
| P2 | `employee-count`, `historical-employee-count` | `symbol` | Optional |
| P2 | `index-list`, `etf/info`, `etf/holdings` | `symbol` | ETF reference; holdings are snapshots, not history |

### 5.2 Corporate events

| Priority | Endpoint | Params | Use |
|----------|----------|--------|-----|
| P0 | `splits` | `symbol` | Per-symbol splits: date, numerator, denominator |
| P0 | `dividends` | `symbol` | Per-symbol dividends: ex-date (`date`), record, payment, declaration date, `dividend`, `adjDividend`, `yield`, `frequency` |
| P1 | `splits-calendar`, `dividends-calendar` | `from`, `to` | Cross-check completeness of the per-symbol data; confirm the maximum window |
| P1 | `earnings` | `symbol` | Earnings dates, EPS and revenue actual and estimate |
| P1 | `earnings-calendar` | `from`, `to` | Cross-check |
| P2 | `ipos-calendar` | `from`, `to` | IPO dates |

### 5.3 Prices

| Priority | Endpoint | Params | Use |
|----------|----------|--------|-----|
| **P0** | `historical-price-eod/non-split-adjusted` | `symbol`, `from`, `to` | **Primary daily source (raw OHLCV)** |
| P1 (sample only) | `historical-price-eod/full` | `symbol`, `from`, `to` | Split-adjusted OHLCV plus `vwap`, `change`, `changePercent`; for reconciliation only |
| P1 (sample only) | `historical-price-eod/dividend-adjusted` | `symbol`, `from`, `to` | FMP's adjusted prices; for reconciliation of our adjClose only |
| P0 for indices | same three endpoints with `^GSPC`, `^IXIC`, `^DJI`, `^VIX`, `^RUT`, plus ETFs SPY, QQQ, IWM, DIA | | Index availability on Premium is [VERIFY] |
| **P0 (intraday)** | `historical-chart/1hour` | `symbol`, `from`, `to` | **1-hour bars**; windowing in section 6.5. Availability on Premium, depth, and row cap are [VERIFY] |
| P2 | `historical-chart/5min`, `15min`, `30min`, `4hour` | same | Only if the owner later asks; see section 6.3 for the bandwidth impact |
| Skip | `historical-price-eod/light` | | Not needed |

### 5.4 Index membership

| Priority | Endpoint | Use |
|----------|----------|-----|
| P0 | `historical-sp500-constituent` | S&P 500 additions and removals with dates |
| P0 | `historical-nasdaq-constituent` | Same for FMP's Nasdaq index (check whether this is Nasdaq-100) |
| P0 | `historical-dowjones-constituent` | Same for the Dow Jones Industrial Average |
| P0 | `sp500-constituent`, `nasdaq-constituent`, `dowjones-constituent` | Current membership; the anchor for rolling the change log backwards |

### 5.4a Fundamentals (statement endpoints take `symbol`, `period` = `annual` | `quarter`, `limit`)

| Priority | Endpoint |
|----------|----------|
| P1 | `income-statement`, `balance-sheet-statement`, `cash-flow-statement` |
| P1 | `key-metrics`, `ratios`, `enterprise-values` |
| P1 | `historical-market-capitalization` (`symbol`, plus date range [VERIFY]) |
| P1 | `financial-statement-symbol-list` (to know which symbols have statements), `financial-reports-dates` |
| P2 | `income-statement-as-reported`, `balance-sheet-statement-as-reported`, `cash-flow-statement-as-reported`, `financial-statement-full-as-reported` |
| P2 | `financial-scores`, `owner-earnings`, the `*-growth` endpoints, `revenue-product-segmentation`, `revenue-geographic-segmentation` |
| P2 | `analyst-estimates`, `grades-historical`, `ratings-historical` (only if the owner uses them) |

Do not download `*-ttm` endpoints, quotes, news, or technical-indicator endpoints; they are either current snapshots or derivable.

Every statement row MUST retain `filingDate` and `acceptedDate`. Without them the data cannot be used point-in-time. A statement endpoint that does not return these fields for most rows is a finding to report at G1.

### 5.5 Macro

| Priority | Endpoint | Params |
|----------|----------|--------|
| P1 | `treasury-rates` | `from`, `to` |
| P1 | `economic-indicators` | `name` (GDP, CPI, unemployment, ...) |
| P2 | `economic-calendar`, `market-risk-premium` | |

---

## 6. Downloader

### 6.1 Command-line interface

A single entry point `fmp` (module or console script) with sub-commands. All commands MUST be idempotent and resumable.

```
fmp probe-free            # Phase 0b: <=200 calls, writes docs/fmp_probe_report.md
fmp probe-paid            # Phase 2: intraday intervals, depth, row caps, byte accounting
fmp plan [--dry-run]      # builds task list + bandwidth projection -> budget_plan.md
fmp run  [--phase N] [--tier T] [--max-gb G] [--dump-id ID]   # executes tasks from the manifest
fmp status                # progress, bytes used (rolling 30d), ETA, failures
fmp retry-failed          # re-queue failed/retryable tasks
fmp build                 # raw archive -> normalized Parquet vintage
fmp validate [--sample N] # runs the validation suite (section 11) -> validation_report.md/.json
fmp verify --final        # gate G3 checklist (section 14)
fmp snapshot-universe     # section 12.1 (free sources, no FMP key needed)
fmp ingest-increments     # section 10 (yfinance incremental layer; only on the increment writer host)
fmp backup                # pack and copy archive + vintage to the NAS, verify checksums there (section 14)
fmp publish               # laptop -> NAS: publish an immutable vintage (and increments) for readers
fmp sync                  # NAS -> this machine (e.g. the Pi, `reader` profile): copy, verify, then switch current.txt
```

Configuration lives in `config/fmp.yaml` (example in appendix B). The API key is read **only** from the environment variable `FMP_API_KEY` (or a git-ignored `.env`). The key MUST NOT appear in logs, exceptions, the manifest, or the raw archive request metadata. Add `.env`, `data/`, and the raw archive to `.gitignore`.

Where the code runs: the downloader, builder, and validators run on the **owner's Windows 11 laptop** (section 13); the Raspberry Pi runs only the `reader` profile. Assume nothing about network access in your own development sandbox beyond what you verify.

### 6.2 Task manifest (SQLite)

Persist work in `manifest.sqlite`. One row per atomic request:

```
tasks(
  task_id          TEXT PRIMARY KEY,      -- sha1(dump_id|endpoint|symbol|canonical_params)
  dump_id          TEXT,                  -- vintage, e.g. '2026-10'
  endpoint         TEXT,
  symbol           TEXT NULL,
  params_json      TEXT,                  -- canonical, WITHOUT the API key
  priority         INTEGER,               -- lower runs first
  tier             TEXT,                  -- 'A','B','C' (section 6.3)
  status           TEXT,                  -- pending|running|ok|ok_empty|api_error|plan_restricted|failed|skipped
  attempts         INTEGER,
  http_status      INTEGER NULL,
  rows             INTEGER NULL,
  bytes_wire       INTEGER NULL,          -- Content-Length or counted compressed bytes
  bytes_decoded    INTEGER NULL,          -- decompressed body size
  sha256           TEXT NULL,             -- of the stored response body
  fetched_at       TEXT NULL,             -- UTC ISO-8601
  error            TEXT NULL
)
bandwidth_log(ts TEXT, bytes_wire INTEGER, bytes_decoded INTEGER)   -- for the rolling 30-day window
```

Rules:

- `ok_empty` (HTTP 200 and an empty list or object) is a **valid outcome** and must be distinguished from an error. A symbol with no data is information.
- `plan_restricted` (HTTP 402/403 or an explicit "not available under your plan" message) is recorded per endpoint; it MUST NOT abort the run for other endpoints. If **all** requests start returning 401/403, abort the run and report (probably a revoked key or a plan lapse).
- A response that says the quota or limit has been reached MUST stop the run immediately (not retry in a loop) and record the time.
- Crashes: the manifest is the source of truth; on restart, `running` tasks revert to `pending`. Writes MUST be atomic (write to a temporary file, fsync, rename).

### 6.3 Bandwidth budget and tiers (the key constraint)

The Premium plan caps traffic at **50 GB in a trailing 30-day window** [VERIFY: counted on compressed or decompressed bytes]. A rough estimate (to be **replaced by measured values** from the probe) shows that downloading everything from section 5 does not fit in one month if bytes are counted decompressed:

| Block | Rough size as raw JSON (estimate) | Notes |
|-------|-----------------------------------|-------|
| Reference, profiles, events | 1 to 3 GB | profiles about 30k symbols |
| Daily raw (`non-split-adjusted`), full universe | 10 to 15 GB | ~30k symbols, avg 2,500 to 4,000 bars, ~150 B per bar |
| Daily `full` + `dividend-adjusted`, full universe | ~25 GB more | **why D8 limits them to a sample** |
| Fundamentals (P1 endpoints), ~12k symbols with statements | 8 to 15 GB | |
| **1h bars**, full US universe | **40 to 50 GB** | ~250 to 450 M bars x ~110 B per bar |

So the realistic plan is: **month 1** = everything except most of the 1h universe, plus 1h for the highest-value tiers; **month 2 (optional, owner's decision)** = the remaining 1h. If the cap turns out to be measured on compressed bytes, roughly 4 to 6 times more fits and everything may fit in one month. This is why the paid probe (Phase 2) measures both numbers.

Requirements:

1. Send `Accept-Encoding: gzip` and record both `bytes_wire` and `bytes_decoded` for every response. The probe compares the two against FMP's usage display (the owner can read the account dashboard) to infer which one the cap uses.
2. `fmp plan` MUST compute, from measured bytes per row per endpoint and from the real universe size, a projection table: bytes and request count per block and per tier, under both counting hypotheses. It writes `budget_plan.md`.
3. **Tiers for 1h downloads** (execution order; the owner may edit membership at G2):
   - **A**: index ETFs and indices (SPY, QQQ, IWM, DIA, sector SPDRs, `^GSPC`, `^IXIC`, `^DJI`, `^VIX`).
   - **B**: current and historical S&P 500, Nasdaq-100 and Dow constituents (from section 5.4).
   - **C**: all other US common stocks and ADRs that are currently active.
   - **D**: all other delisted US symbols.
   - **E**: remaining ETFs and unclassified instruments.
4. `fmp run` MUST enforce a **hard stop** at `safety x cap` (default 0.90 x 50 GB, configurable) measured over the **rolling 30-day window** from `bandwidth_log`, using whichever of wire or decoded bytes is configured as the conservative basis (default: decoded, until the probe proves otherwise). When the stop is hit, the run exits cleanly with a status report; it never silently exceeds the cap.
5. Order of work (strict): reference data, then events, then index membership, calendars and macro, then daily raw, then the sampled reconciliation endpoints, then fundamentals, then 1h by tier. Within a block, process **delisted symbols before active ones**: active symbols can still be refreshed from free sources after cancellation, delisted ones cannot.
6. If a given interval turns out not to be offered on Premium (for example `1hour`), **stop and report at G2**. Do not silently substitute another interval. Possible alternatives for the owner to choose from: Ultimate, or building 1h from `30min` or `15min` (higher traffic), or skipping intraday.

Estimated request counts (for scheduling only): reference and events about 100k (profile, splits, dividends per symbol); daily raw about 30k; fundamentals about 200k to 300k; 1h roughly 30k symbols x several windows each (depends on the measured row cap). At the default limit of 600 requests per minute this is on the order of a day or two of wall-clock time; traffic, not request rate, is the binding constraint.

### 6.4 Rate limiting, retries, concurrency

- Shared **token bucket**, default **600 requests/min** (80% of 750), configurable. Concurrency 8 to 16 workers (async HTTP client). Do not exceed the limit even briefly; bursts count.
- Retries: network errors, timeouts, HTTP 429 and 5xx get exponential backoff with jitter, **up to 6 attempts**. Honour `Retry-After`. After 20 consecutive 429/5xx responses, pause all workers for 60 seconds (circuit breaker).
- HTTP 4xx other than 429 are not retried; they are classified (`plan_restricted`, `api_error`).
- A timeout MUST be set (connect 10 s, read 60 s). Large bodies are streamed to disk.
- URL-encode symbols (tickers can contain `.`, `-`, `^`). Preserve the symbol exactly as FMP returns it; the mapping to yfinance conventions is done in the `securities` table (section 7.3).

### 6.5 Intraday windowing

1. The paid probe determines `max_rows_per_request` and the allowed window by requesting AAPL 1h with progressively larger `from`/`to` ranges and observing truncation (a response that is shorter than expected for the window, or whose last timestamp stops before `to`).
2. The task generator splits the history of each symbol into windows of `floor(max_rows / bars_per_day / 1.25)` trading days (25% safety margin), where `bars_per_day` is the observed number (expected 7 for the regular session, see section 8.4).
3. Runtime safeguard: if a response's row count reaches 95% of `max_rows`, mark it `possibly_truncated`, halve the window, and re-queue the two halves. Never accept a possibly-truncated window silently.
4. Start of history per symbol: use `max(IPO date, configured earliest date)`; for delisted symbols stop at the delisting date plus a small margin. If the first window is empty, bisect forward to find the first non-empty window instead of scanning year by year.
5. Do not download windows that fall entirely outside the symbol's first and last known price dates from the daily data (daily raw runs first, so those dates are known). This saves a large amount of traffic.

### 6.6 Raw archive

- Layout: `data/fmp/raw/<dump_id>/<endpoint_path>/<symbol or _>.jsonl.gz`. Each line is one response: `{"request": {...no key...}, "fetched_at": "...", "http_status": 200, "body": <parsed JSON>}`. Append new lines as **concatenated gzip members** (valid gzip) so packing many windows per symbol into one file avoids hundreds of thousands of tiny files.
- A sidecar `*.sha256` or an entry in the manifest records the checksum. Writes are atomic.
- Free-disk check before each phase; fail early with a clear message. Expected raw archive size: tens of GB.
- The archive is **append-only**. A re-run never overwrites; a re-fetch adds a new line with a newer `fetched_at`, and the builder uses the newest successful line per request key.

### 6.7 Mock FMP server for tests

Provide a small local HTTP server (for example with a standard library or a lightweight framework) driven by synthetic fixtures that can simulate: normal paging, empty results, HTTP 200 with an error body, HTTP 402/403, HTTP 429 with `Retry-After`, 5xx bursts, slow responses, truncated intraday windows (row cap), gzip with and without `Content-Length`, and quota exhaustion. The full downloader test suite MUST run against it with no network.

### 6.8 Disk budget

All figures are rough estimates to be replaced by measured values in `fmp plan`; the agent does not know the owner's hardware.

| Item | Estimate |
|------|----------|
| Raw archive (gzip of JSON), everything in section 5 for the full universe | about 10 to 25 GB (1h alone about 5 to 10 GB) |
| Parquet vintage (all tables, 1d and 1h) | about 8 to 12 GB |
| Temporary space during `fmp build` | about 10 GB |
| Backup copy of archive and vintage (separate volume) | same as the two lines above |
| **Recommended free space** | **at least 100 GB on the working volume and 40 GB on a separate backup volume** |

Owner's environment (section 13): the writer is a Windows 11 laptop with a 500 GB internal SSD and a 500 GB external SSD (the budget above fits on the internal one); the reader is a Raspberry Pi 5 with a 500 GB SSD (it needs only one vintage of about 10 to 12 GB plus increments); the NAS (6 TB) is the backup and hand-off target. Consequences for the implementation: (a) the backup target is a **network share**, so copying hundreds of thousands of small files is slow; `fmp backup` and `fmp publish` MUST pack the data into a few large files per `dump_id` (for example one tar or zip per top-level block) or use an incremental copy, and run the checksum verification of section 14 against the copy on the NAS; (b) the NAS filesystem may lack symlinks and POSIX permissions, so the `current.txt` pointer is a text file and the read-only marking is best-effort on the NAS; (c) never run DuckDB queries or the build directly on the NAS, only on a local disk.

Requirements: `fmp plan` computes the expected disk need from measured bytes and prints it next to the free space; `fmp run` and `fmp build` abort with a clear message if free space is below the need plus a 20% margin. Recommend an SSD (NVMe preferred) for the Parquet vintage and the build step, since DuckDB queries and the build are I/O-bound; the raw archive and the backup may live on slower storage.

---

## 7. Normalized store (Parquet + DuckDB)

### 7.1 Directory layout

```
data/fmp/
  raw/<dump_id>/...                      # section 6.6 (append-only archive)
  manifest.sqlite
  vintages/<dump_id>/                    # derived, rebuildable from raw/
    reference/   securities.parquet, symbol_history.parquet, profiles.parquet, delisted.parquet, ...
    events/      splits.parquet, dividends.parquet, earnings.parquet, mna.parquet
    prices/
      1d_raw/<bucket>/<security_id>.parquet      # one file per security
      1h_raw/<bucket>/<security_id>.parquet
      recon/   daily_split_adjusted_sample.parquet, daily_div_adjusted_sample.parquet
    index_membership/  index_changes.parquet, index_members_asof.parquet (optional materialization)
    fundamentals/ income_statement.parquet, balance_sheet.parquet, cash_flow.parquet, key_metrics.parquet, ratios.parquet, ...
    macro/       treasury_rates.parquet, economic_indicators.parquet, calendars.parquet
    _meta/       build_info.json (code version, input checksums, row counts), validation_report.json
  current.txt                            # pointer: a small text file containing the active <dump_id> (never a symlink: Windows and NAS)
  increments/                            # section 10 (yfinance incremental layer)
  snapshots/<YYYY-MM-DD>/                # section 12.1
```

`<bucket>` is a two-level hash or the first character of the symbol, used only to keep directory sizes reasonable (30k files in one directory is slow on some file systems). One file per security matches the cache-on-miss access pattern of the existing `DataManager`; cross-sectional queries use DuckDB globs (`read_parquet('.../prices/1d_raw/*/*.parquet')`). If glob performance over about 30k files is poor, add an optional consolidated yearly partition as a **derived** cache, never as the source of truth.

### 7.2 Universe definition

`config/universe.yaml` (defaults; owner may edit):

- Include exchanges: NASDAQ, NYSE, AMEX (NYSE American), NYSE Arca, BATS/Cboe. Exact FMP exchange codes MUST be taken from `available-exchanges` and observed `profile`/`delisted-companies` values.
- Include types: common stock, ADR, ETF. Warrants, units, rights and preferreds: keep the rows if present but tag them (`instrument_type`) so downstream code can filter. Mutual funds, crypto, forex, commodities: excluded from the universe (indices are fetched from an explicit list, section 5.3).
- Exclude OTC / pink-sheet venues by default (`include_otc: false`).
- The universe is the union of `stock-list`, `delisted-companies`, `etf-list`, and `symbol-change` old and new symbols, filtered by the rules above.
- Every exclusion decision is logged with counts (how many symbols removed by which rule) in `universe_report.md`.

### 7.3 Core tables

**`securities`** (one row per security; internal id)

| Column | Notes |
|--------|-------|
| `security_id` | Internal surrogate key (stable integer, allocated once, never reused) |
| `symbol` | Current or last known FMP symbol |
| `yf_symbol` | Symbol in yfinance convention (for example class shares `BRK-B`); derived and verified, nullable |
| `exchange`, `name`, `instrument_type` | |
| `cik`, `isin`, `cusip` | Nullable |
| `ipo_date`, `delisted_date`, `is_actively_trading` | |
| `first_price_date`, `last_price_date` | Filled from the daily raw data |
| `source_flags` | Which lists the symbol came from |

**`symbol_history`**: `security_id`, `symbol`, `valid_from`, `valid_to`, `reason` (from `symbol-change`).

**Identity rules.** FMP price endpoints are keyed by symbol only and bars carry no CIK or ISIN, so a change of issuing company **cannot be detected from price data**. Therefore:

1. **Segment by listing entries, not by prices.** When `delisted-companies`, `symbol-change`, and `profile` show several entries for the same symbol with different names or CIKs and non-overlapping date ranges, create one `security_id` per entry with `valid_from` and `valid_to`, and **clip** that symbol's bars to each range. Clipping only assigns bars to securities; the unclipped response stays untouched in the raw archive. Apply this rule only to the extent the Phase 0b/2 probe (section 4.2 item 8) shows that FMP actually returns merged series for reused tickers; record the finding in `docs/decisions.md`.
2. If ranges overlap, are missing, or the evidence is contradictory, **do not guess**: keep one security, set `identity_uncertain = true`, and report it.
3. The validator (V-IDENT in section 11) flags suspicious cases: a price gap of more than 180 calendar days in the middle of a series, a CIK or name mismatch between `delisted-companies` and `profile`, and an IPO date later than the first bar. Flag and report them; no further heuristic repair in v1.

**`splits`**: `security_id`, `date`, `numerator`, `denominator`, `ratio` (= numerator / denominator, for example 4.0 for a 4-for-1; 0.1 for a 1-for-10 reverse split), `source` (`fmp` | `yfinance`).

**`dividends`**: `security_id`, `ex_date`, `record_date`, `payment_date`, `declaration_date`, `amount_raw` (the cash amount on the raw price basis at that date, see 4.2 item 5), `amount_split_adjusted`, `frequency`, `source`.

**`prices_1d_raw`** columns: `security_id`, `date` (DATE), `open`, `high`, `low`, `close`, `volume`, `flags` (int, bitmask, section 7.5), `source`, `vintage`. Prices are float64; volume int64. Sorted by `date`, no duplicates.

**`prices_1h_raw`** columns: `security_id`, `ts_utc` (TIMESTAMP, UTC, bar **start**), `date_et` (DATE, exchange local date), `open`, `high`, `low`, `close`, `volume`, `flags` (int, bitmask), `source`, `vintage`.

Parquet writing: zstd compression (level about 6), row group about 100k rows, statistics on, sorted by time, `use_byte_stream_split` for float columns if the library supports it. Record the settings in `_meta/build_info.json`.

### 7.4 Builder (`fmp build`)

- Reads only the raw archive (never the network). Newest successful response per request key wins.
- **Deterministic and idempotent**: the same archive and code produce byte-identical (or at least content-identical) Parquet. Store row counts and content hashes per output file in `build_info.json` (V-REPRO check).
- Builds into a temporary directory and atomically switches `current` only after validation passes at the "structural" level.
- Preserves unknown extra fields: raw-field passthrough tables or a `_extra` JSON column for statements, so a later need for an unused field does not require a re-download.

### 7.5 Suspect-data flags (D11: flag, never alter)

The builder computes a `flags` bitmask per price bar, deterministically, from the data in the same vintage. Values are never changed; flagged bars stay in the store. Thresholds live in `config/fmp.yaml` and every flag is covered by unit tests with synthetic examples.

| Bit | Name | Rule (defaults; calibrate on a sample and record in `docs/decisions.md`) |
|-----|------|------|
| 0 | `OHLC_VIOLATION` | `low > min(open, close)` or `high < max(open, close)` or any price <= 0 |
| 1 | `SPIKE_REVERTED` | single-bar outlier: `abs(return) >= 35%` (daily) and robust z-score of the return against the prior 60 bars `>= 8`, and the price reverts within 1 to 2 bars (`abs(r_t + r_{t+1}) < 0.5 * abs(r_t)`), and no split or dividend within +-2 sessions |
| 2 | `JUMP_NO_SPLIT` | a close-to-close ratio near an integer or a simple fraction (2, 3, 4, 5, 10, 1/2, 1/3, 1/10, ...) within 3%, with **no** matching split event within +-2 sessions (a probable **missing split** in the events table) |
| 3 | `STALE` | at least N identical consecutive OHLC bars with zero volume on trading days (N default 5 daily; calibrate for 1h) |
| 4 | `ZERO_VOLUME` | volume = 0 on a trading day with a non-degenerate price range |

Rules:

- Real crashes and real delisting collapses must not be hidden: flags describe suspicion, they do not mean "wrong". Penny stocks and delisting runs produce many flags; that is acceptable.
- `JUMP_NO_SPLIT` findings are also written to `reports/missing_splits_candidates.csv` (symbol, date, ratio) for the owner's review; do not insert synthetic splits automatically.
- The validation report contains the flag rate per security tier and the top 50 symbols by flagged-bar share (V-FLAGS).
- DataManager default output does **not** filter flagged bars (legacy compatibility); it exposes the `flags` column on request and supports an `exclude_suspect` option (section 9.3).

---

## 8. Price normalization and adjustment

### 8.1 Principle (D3)

Stored prices are **raw**. `close` and `adjClose` are derived from raw prices and the events tables. Never persist adjusted prices into the base store, with one exception: an optional, disposable read cache keyed by a hash of the events used (section 9.5).

### 8.2 Definitions

- Split with `ratio = numerator / denominator` effective on date `S` (first trading day at the new share count). A price on a date before `S` is divided by `ratio` to be comparable with prices on or after `S`. Volume before `S` is multiplied by `ratio`.
- Dividend of `d` (cash per share on the raw basis of that date) with ex-date `D`. Let `C_prev` be the **raw** close of the last trading day **before** `D`. The dividend factor is `DF(D) = 1 - d / C_prev`.
- For a bar at date `t`:

```
split_adj_factor(t) = PRODUCT over splits with S > t of (1 / ratio_S)
div_adj_factor(t)   = PRODUCT over dividends with D > t of DF(D)
close_split(t)      = close_raw(t) * split_adj_factor(t)           # matches yfinance 'Close' semantics
adjClose(t)         = close_split(t) * div_adj_factor(t)            # total-return style, matches yfinance 'Adj Close' semantics (to be verified)
```

Open, high, and low are adjusted with the same factors. Volume uses `volume_raw / split_adj_factor` (that is, multiplied by the cumulative ratio). Events dated **after** the last bar do not affect existing bars until a new price bar is added after them; handle this explicitly.

### 8.3 Implementation requirements

- A pure, vectorized function `adjust(prices_raw, splits, dividends, mode)` with `mode` in `{"raw", "split", "total"}`. No I/O inside. It MUST be unit-tested with hand-computed examples, including: multiple splits, a reverse split, a dividend on the day of a split, a dividend with no prior bar, a duplicate event, and a symbol with no events.
- Test vectors (the agent MUST verify these against the downloaded data in the validation suite, and treat a mismatch as a bug or a data finding, not as something to hide):
  - AAPL 4-for-1 on 2020-08-31; AAPL 7-for-1 on 2014-06-09.
  - TSLA 5-for-1 on 2020-08-31 and 3-for-1 on 2022-08-25.
  - AMZN 20-for-1 on 2022-06-06; GOOGL 20-for-1 on 2022-07-18; NVDA 10-for-1 on 2024-06-10.
  - AAPL ex-dividend 2020-08-07: $0.82 per share pre-split (raw basis).
- Compare your `adjClose` with FMP's `dividend-adjusted` series for the reconciliation sample and with legacy yfinance files. Methodologies may differ legitimately (rounding, handling of special dividends), so differences are reported with statistics (median, p99, max of relative difference per symbol) and only extreme outliers fail (section 11).

### 8.4 Intraday bar conventions (to be fixed by the paid probe)

Expected for the regular US session (09:30 to 16:00 ET): bars start at 09:30, 10:30, 11:30, 12:30, 13:30, 14:30, and a **half-hour bar** at 15:30; that is **7 bars per regular day** and fewer on early-close days (13:00 ET close). The probe MUST record: the actual label convention (bar start or bar end), the time zone of the `date` field (verify against a known day, for example check the first bar of a normal day), whether pre-market and after-hours bars exist, and whether intraday prices are split-adjusted. Keep the original string in the raw archive.

**Time conversion procedure (MUST).** Never trust or apply a fixed UTC offset to FMP timestamps. Determine in the paid probe which time zone the `date` strings are expressed in (expected: exchange local time, `America/New_York`; verify with a known regular day where the first bar must be the 09:30 ET bar). Then convert with `zoneinfo` from that zone to UTC, per bar. US daylight-saving changes happen on Sundays (second Sunday of March, first Sunday of November) when the market is closed, so the real risk is a wrong or fixed offset making all bars after the change shift by one hour, and the hour-ambiguity itself never falls inside a session. The probe MUST therefore request AAPL 1h bars for the trading days around both transitions of at least two years (the Friday before and the Monday after each change) and assert that the first bar of every regular day is at 09:30 ET after conversion. If FMP's zone turns out to be something other than US/Eastern (for example UTC or a European zone), the weeks in which US and European transition dates differ are an additional mandatory test window. Test the converter with synthetic data across the same dates.

---

## 9. DataManager integration

### 9.1 Discover first

Follow section 4.1. Produce `docs/datamanager_integration.md` and do not change any `DataManager` behavior until it exists. All code changes MUST be backward compatible: every existing call (timeframe 1d, existing columns) must continue to work with identical results where legacy data is used.

### 9.2 Target architecture

Add a **provider chain** behind the existing public interface (adapt names to the repo; do not rename existing public methods):

1. `FmpStore` (new): reads the active vintage (`current`) through DuckDB or pyarrow. Returns raw prices plus events.
2. `IncrementStore` (new, section 10): yfinance-sourced bars and events newer than the vintage's last date, in the same raw format.
3. `LegacyYfinanceCache` (existing): the untouched `.csv.gz` files and the existing cache-on-miss loader.

**Resolution policy `fmp_first`** (default; configurable in one place):

- For a request `(symbol, timeframe, start, end)`: map symbol to `security_id`; take bars from `FmpStore` up to its `last_price_date` for that security; append `IncrementStore` bars **strictly after** that date; apply adjustments on the combined raw series using the **union** of events (deduplicated by `(security_id, date, type)`).
- If the symbol is unknown to the FMP store (not in the universe or no data): fall back to the legacy cache; if that misses too, run the existing cache-on-miss loader exactly as today. New downloads from that loader keep going to the legacy location (D2), unless the owner later changes it.
- If both FMP and legacy data exist for the same range, FMP wins; no automatic merging of overlapping bars from different sources into one series (provenance would be lost). The `source` of each bar is exposed in an optional column.
- Never splice FMP-provided `adjClose` with yfinance-provided `adjClose`. Adjusted values always come from `adjust()` over raw data.

### 9.3 Read semantics

Default output MUST reproduce the legacy semantics found in Phase 0a. The expected case (to be confirmed on the legacy files): `close` = split-adjusted close (as yfinance returns by default) and `adjClose` = split-and-dividend-adjusted close. Add an explicit parameter (for example `adjust="legacy"` default, `"raw"`, `"split"`, `"total"`) so strategies can request raw prices when needed. Column names, dtypes, index type and timezone handling must equal the legacy output.

Optional extras, all off by default so legacy callers see no change: a `source` column, a `flags` column (section 7.5), and `exclude_suspect=True` to drop flagged bars.

**Semantics to document in `docs/storage_schema.md` (for whoever builds backtests later):**

- Dividend data has **dates only** (ex-date, record, payment, declaration); there is no time-of-day and no session flag, and none is needed. The price effect of a dividend occurs on the **ex-date** bar.
- `adjClose` is on a total-return basis: returns computed from `adjClose` include dividends, credited at the ex-date. A consumer that uses raw or split-only `close` must account for dividends separately from the `dividends` table, otherwise dividend income is silently missing.
- Splits are effective on the first trading day at the new share count.

### 9.4 Timeframes

- `1d`: served from `1d_raw` as above.
- `1h`: served from `1h_raw` plus the 1h increments. Document which intraday adjustments are applied on read and how the daily split factors are applied to intraday bars (same `split_adj_factor` by date).
- Higher timeframes (1w, 1M) MAY be resampled from 1d by the DataManager; do not store them.

### 9.5 Caching and performance

The existing cache manager continues to cache what it loads. Cache keys MUST include the vintage id and a hash of the events used, so a new dividend or a new vintage invalidates adjusted caches correctly. Reads of one symbol over its whole history MUST complete in well under 100 ms from a warm Parquet file; document the measured numbers. Provide a bulk read API for cross-sectional access (many symbols, one date range) implemented as a single DuckDB query.

### 9.6 Symbol mapping

`securities.yf_symbol` links the owner's usual symbols to `security_id`. Rules: class-share separators (`.` vs `-`) normalized per the convention of the legacy files; index symbols keep their `^` prefix; historical symbols resolve through `symbol_history` with an optional `as_of` date. Resolving an ambiguous symbol (more than one security) MUST raise a clear error, never return a silent guess, unless `as_of` disambiguates.

---

## 10. yfinance incremental layer

Purpose: after the subscription ends, keep current tickers up to date from the free source without breaking consistency with the FMP raw base.

### 10.1 The trap

yfinance returns `Close` already **split-adjusted as of download time**, and `Volume` split-adjusted too. The FMP base is **raw**. Appending yfinance bars naively after a new split produces a series that jumps by the split ratio. Therefore every yfinance ingest MUST convert to raw before storing.

### 10.2 Ingestion rules

1. Ingest **only bars with date strictly after the last stored bar** of that security (base plus increments), plus an optional small overlap window used only for verification, never for overwriting.
2. Download with `auto_adjust=False` and actions enabled so that splits and dividends come with the prices (adapt to the library version in the repo).
3. Convert to raw: `close_raw = close_yf / split_adj_factor_yf`, where the factor is built only from splits that occurred **after each bar's date within the downloaded window and after the ingest date** (that is, splits between the bar and today). Equivalent to un-adjusting using the splits yfinance reports. Volume is converted the opposite way. Unit-test this function against synthetic series with one split, two splits, a reverse split, and no split.
4. New splits and dividends found in the downloaded actions are written to `increments/events` and unioned with FMP events at read time (deduplicate by `(security_id, date, type)`; on conflict FMP wins and a warning is logged).
5. Because only new bars are appended, previously stored raw bars remain correct when a split occurs later: raw history never needs rewriting. The only danger is a **gap** (a long period without ingestion during which a split occurred): the conversion in rule 3 handles it because it uses the splits reported in the same download. Add a test for exactly this case.
6. Each increment row carries `source='yfinance'`, `ingested_at`, and `vintage` (the vintage it extends).
7. **Run the ingestion for the entire current universe on a regular schedule (weekly is enough; daily preferred).** This is the main defence against survivorship loss: a stock that is delisted next month will already have its history stored, whereas free sources often return nothing for delisted tickers. Provide a `cron`-style example in the docs; do not create schedulers on the owner's machine.
8. 1h increments follow the same rules but yfinance only exposes about the last 730 days of 1h data. The ingestion MUST be run at least every 12 months, and warn loudly if the gap since the last stored 1h bar approaches 700 days (data would be lost forever).

### 10.3 Drift report

When the next FMP vintage arrives, `fmp build` MUST produce a **drift report** that compares the overlapping period of the old increments against the new FMP data (raw close relative difference per symbol, split/dividend event mismatches, bars missing in either source). Then the increments before the new vintage's end date are archived (moved, not deleted) and the new vintage becomes the base. This quantifies how much the free source diverged and is valuable for the owner's research.

### 10.4 Where increments run: single writer

Two machines read the data (laptop for research, Pi for live strategies), so the increment layer needs one rule to avoid divergence: **exactly one machine is the increment writer** (`increments.writer_host` in the config; the other machines are readers). Default until the owner decides: the machine that runs the existing cache-on-miss yfinance loader today. Requirements:

- Increment files are per security and per timeframe, written atomically (temporary file plus `os.replace`), and carry a monotonically increasing `increment_seq` in `increments/_meta.json`.
- Readers on other machines get increments through the same publish and sync mechanism as vintages (NAS as the hand-off, checksum verified, switched atomically); a reader never writes to `increments/`.
- If both machines must be able to ingest (for example the Pi runs daily updates and the laptop is offline), the design MUST still keep one writer at a time: a lock file on the shared NAS location with an expiry, and a conflict rule (the writer with the higher `increment_seq` wins; the other re-ingests from its last common bar). Document the chosen mode in `docs/decisions.md`; do not silently run two writers.
- The legacy yfinance cache (D2) keeps working on whichever machine uses it today; this section concerns only the new increment layer.

---

## 11. Validation suite (`fmp validate`)

Output: `validation_report.md` (human summary) and `validation_report.json` (machine-readable). Each check has an ID, a severity (`FAIL` blocks gate G3, `WARN` is reported, `INFO`), the number of items examined, the number of violations, and up to 20 example offenders. A `--sample N` option runs expensive cross-source checks on N random symbols (default 100), plus a fixed set of named symbols so that runs are comparable.

| ID | Check | Severity |
|----|-------|----------|
| V-SCHEMA | Every Parquet table matches the declared schema; no unexpected nulls in key columns; types correct | FAIL |
| V-DUP | No duplicate `(security_id, date)` or `(security_id, ts_utc)` | FAIL |
| V-ORDER | Timestamps strictly increasing within each security | FAIL |
| V-OHLC | `low <= min(open, close)`, `high >= max(open, close)`, all prices > 0, volume >= 0. Violation rate above 0.1% of bars for a symbol is a WARN; above 5% FAIL | WARN/FAIL |
| V-CAL | Daily bars exist on **expected trading days**: expected days come from NYSE/NASDAQ holidays plus weekends, between each security's first and last bar. Report missing-day ratio per security; WARN above 1%, FAIL above 10% for any security of tier A and B | WARN/FAIL |
| V-SPLIT-KNOWN | The test-vector splits and dividend in section 8.3 exist with the right ratios and dates | FAIL |
| V-SPLIT-CONSISTENCY | For each split, the raw close ratio across the split date (`close_before / close_after`) is within a factor of 0.7 to 1.4 of the declared ratio for at least 95% of splits; list outliers | WARN |
| V-DIV-SANITY | `amount_raw / C_prev < 0.5` except a short allow-list of flagged outliers; no negative amounts | WARN |
| V-ADJ-RECON | Our `adjClose` vs FMP `dividend-adjusted` on the reconciliation sample: per-symbol median relative difference below 0.1%; report p99 and max. Fail only if more than 5% of sampled symbols exceed 1% median difference (indicates a systematic error in our formula or in the dividend basis) | WARN/FAIL |
| V-YF-XCHECK | Our `close_split` vs legacy yfinance `close` for N sampled symbols on overlapping dates: share of days with relative difference above 0.5%; report symbols with systematic offsets (adjustment or dating issues) | WARN |
| V-DELIST | A list of known delisted symbols (TWTR delisted 2022, SIVB 2023, BBBY 2023, ATVI 2023; LEH 2008 as a soft check because coverage of very old failures may be absent) exists in `securities` with a delisted date and has bars up to about that date. Missing symbols are reported as WARN with the list; they are findings about FMP coverage, not bugs | WARN |
| V-UNIVERSE | Counts by exchange and type; share of universe symbols with zero daily bars (`ok_empty`); compare `stock-list` vs `actively-trading-list` vs `delisted-companies` overlaps | INFO/WARN |
| V-IDENT | Identity risks from section 7.3 (gaps over 180 days, CIK/name mismatch, IPO date after first bar; securities with `identity_uncertain = true`) | WARN |
| V-FLAGS | Flag rates from section 7.5 per tier and the top 50 flagged symbols; WARN if more than 0.5% of bars are flagged for any tier A or B security; `missing_splits_candidates.csv` exists | WARN/INFO |
| V-INDEX | Reconstruct S&P 500 membership as of several dates (2008-01-02, 2012-01-03, 2016-01-04, 2020-01-02, 2024-01-02) by rolling the change log backwards from the current list; the size must be within 500 to 505 (multiple share classes explain small excess); every reconstructed member must exist in `securities`; the share of members with price data on that date must be above 97% | WARN/FAIL |
| V-FUND-PIT | For statements: `filingDate` or `acceptedDate` is present for at least 95% of rows, and is on or after the period end date; flag rows where it is not | WARN |
| V-FUND-COV | Share of price-universe common stocks with at least one annual statement; compare with `financial-statement-symbol-list` | INFO |
| V-INTRADAY-ALIGN | For sampled 1h series: bars per full trading day equal the observed convention (expected 7); early-close days have fewer; no bars outside the session boundary found in the probe; the first bar of every regular day is at 09:30 ET after conversion, including the Mondays after each DST change (section 8.4) | FAIL (tier A/B) / WARN |
| V-INTRADAY-DAILY | Aggregate 1h bars into days and compare with the daily raw series: first open within 0.5% of daily open, last close within 0.5% of daily close, max high and min low within 0.5%, volume sum within 5% (consolidated and primary-venue volume may legitimately differ; report the distribution) | WARN |
| V-COMPLETE | Manifest: zero `pending`, `running`, `failed` tasks outside an explicit `skipped_by_plan` list; every `plan_restricted` and `ok_empty` is listed with counts per endpoint | FAIL |
| V-RAW-ARCHIVE | Every archive file matches its recorded checksum; every gzip member is readable end to end; the archive-wide `ARCHIVE.sha256` (section 14) is regenerated and consistent | FAIL |
| V-REPRO | Rebuilding one random vintage subset from the archive reproduces identical content hashes | FAIL |
| V-ADJ-UNIT | Unit tests of `adjust()` and `yfinance_to_raw()` pass | FAIL |

Also write `universe_report.md` (section 7.2) and `coverage_report.md` (per endpoint: symbols requested, `ok`, `ok_empty`, `plan_restricted`, failed, bytes).

---

## 12. Maintenance

### 12.1 Universe snapshots (survivorship protection after cancellation)

`fmp snapshot-universe` MUST work **without** an FMP key. It downloads, from free official sources, the current symbol directories and stores them under `snapshots/<date>/` with checksums:

- Nasdaq Trader symbol directory files `nasdaqlisted.txt` and `otherlisted.txt` (daily-updated, free).
- The current S&P 500, Nasdaq-100 and Dow constituents from a free source configured in `config/fmp.yaml` (the agent chooses a stable free source and documents it).

Each run writes a **diff versus the previous snapshot**: symbols added (new listings, renames) and symbols removed (delisted, renamed). For removed symbols, attempt one last download from yfinance (it may return nothing) and mark them in `securities` as `delisted_suspected` with the snapshot date. Recommend a monthly run; the owner schedules it.

### 12.2 Repeat dump in 2 to 3 years

The whole pipeline MUST be re-runnable with a new `dump_id` (for example `2029-xx`). `docs/runbook.md` describes: the pre-purchase checklist (re-verify section 3, re-run `fmp probe-free`), the plan and gates, the build, the drift report (section 10.3), the promotion of the new vintage (`current` pointer switch), and the retirement of old increments. Old vintages are kept until the owner deletes them manually.

---

## 13. Engineering requirements

- **Layout (adapt to the repo)**: a package `fmp_pipeline/` with `client/` (HTTP, rate limiter, retry), `manifest/`, `planner/`, `archive/`, `builders/`, `adjust/`, `validators/`, `increments/`, `datamanager_adapter/`, and `cli.py`. Do not scatter logic into the existing DataManager; keep the adapter thin.
- **Tests**: unit tests for `adjust()`, `yfinance_to_raw()`, windowing, rate limiter, manifest state machine, builders (with synthetic fixtures), and validators. Integration tests run the downloader against the mock server (section 6.7) including a **kill-and-resume** test (terminate the run in the middle and verify no duplicate or lost rows) and a **bandwidth stop** test. Target at least 85% line coverage on adjustment, manifest, planner, and windowing.
- **Logging**: structured logs (JSON lines) with request id, endpoint, symbol, status, rows, bytes; progress summary every minute; never log the API key or full response bodies.
- **Observability**: `fmp status` shows tasks by status, bytes in the rolling 30-day window against the cap, request rate, ETA, top failing endpoints.
- **Target environments (fixed).** There are two, with different roles:
  - **Dev and ingest machine (writer): a Windows 11 laptop** (x86-64 expected; CPU and RAM to be recorded in Phase 0a), internal SSD about 500 GB and an external SSD about 500 GB. All development, tests, optimization runs, the FMP download, `fmp build`, `fmp validate`, and `fmp verify` happen here. **Windows is the primary development platform**; every command, path, and test MUST work on Windows 11 (PowerShell) first.
  - **Production server (reader): a Raspberry Pi 5** (Ubuntu, aarch64, 4 cores, 16 GB RAM, 500 GB SSD) where the strategies run. It **only reads** the finished data (and, depending on the owner's choice, runs the yfinance increment ingestion, see section 10.4). It never downloads from FMP and never builds vintages.
  - **NAS (6 TB)**: backup target and the hand-off point between the two machines.
- **Package split.** Ship the code as one package with two install profiles: `reader` (DataManager adapter, `adjust()`, DuckDB/pyarrow access, increment ingestion; minimal dependencies) and `pipeline` (downloader, planner, builders, validators; everything else). The Pi installs `reader` only. Both profiles MUST work on Windows x86-64 and Ubuntu aarch64; every dependency in `reader` MUST have a prebuilt aarch64 wheel or be pure Python (verify `pyarrow`, `duckdb`, `pandas`/`polars`, and the JSON library, and record versions in `docs/decisions.md`). Parquet files are portable across the two platforms; the vintage is built once on Windows and read on the Pi.
- **Windows specifics (MUST handle):**
  - Use `pathlib` everywhere; open every text file with explicit `encoding="utf-8"`; do not rely on shell scripts or bash; the CLI and the runbook use PowerShell examples.
  - `zoneinfo` has no system time-zone database on Windows: depend on the `tzdata` package (needed for the DST handling in section 8.4).
  - Symbolic links need elevated rights on Windows: the `current` vintage pointer is **always a small text file** (`current.txt` containing the `dump_id`), never a symlink.
  - **File names derived from symbols MUST be Windows-safe**: encode characters illegal on NTFS (`< > : " / \ | ? *`), trailing dots and spaces, and the reserved device names (`CON`, `PRN`, `AUX`, `NUL`, `COM1`-`COM9`, `LPT1`-`LPT9`, with any extension; some of these are or were real tickers). Use one deterministic encoding (percent-encoding plus a `_` prefix for reserved names). The filesystem is case-insensitive: at plan time detect symbols that differ only by case and disambiguate them with a hash suffix. Price files are named by `security_id`, which avoids most of this; the raw archive is named by encoded symbol.
  - Paths: keep `data_root` short (for example `D:\fmp`) and either keep the 260-character limit in mind or document enabling long paths. Atomic replace uses `os.replace`; handle `PermissionError` caused by antivirus or indexers holding a file (retry with backoff). Recommend excluding `data_root` from real-time antivirus scanning and search indexing in the runbook (the owner's choice).
  - Long unattended runs: prevent sleep and hibernation during `fmp run` and `fmp build` (for example through the Windows `SetThreadExecutionState` API, released on exit) and document that Windows Update restarts can interrupt a run; the manifest makes every run resumable.
- **Memory and CPU.** The builder and validators process **one security at a time** and MUST stay under 4 GB RSS. DuckDB is configured explicitly per machine (Appendix B). Never load a whole price table into memory. Use a fast JSON parser if a wheel exists for the platform (for example `orjson`), stream large responses, and use a moderate gzip level (about 4 to 6) for the raw archive. The downloader is I/O-bound.
- **Disk placement on the laptop.** Default: everything under one `data_root` on the internal SSD (manifest, raw archive, vintages, DuckDB temp files). The external SSD is an optional second local copy of the raw archive and the vintage (an extra backup, not required for operation). Do not put the manifest or the DuckDB temp directory on a drive that can be unplugged during a run; if the archive path disappears mid-run, the run MUST pause and resume cleanly, not corrupt files.
- **Moving data from laptop to Pi.** Provide `fmp publish` (laptop to NAS) and `fmp sync` (NAS to Pi, runnable on the Pi with the `reader` profile): a vintage is **immutable**; copy it to a temporary directory, verify checksums (section 14), and only then update `current.txt` atomically. A reader that is reading while a sync runs must keep seeing the previous vintage until the switch.
- **Performance targets** (report measured times; treat misses as findings, not as reasons to change the design). On the laptop: `fmp build` for the 1d store of about 30k securities within 30 minutes and the 1h store within 2 hours (adjust after measuring the real CPU). On the Pi: a single-security 1d read over the full history under 100 ms warm; a single-security 1h read under 300 ms; a bulk read of one date range for 500 securities in a few seconds. The Pi performs no builds. Process symbol by symbol to keep memory bounded.
- **Type hints** and a linter/formatter consistent with the repo. No new heavyweight dependencies without a one-line justification in `docs/decisions.md`; preferred libraries: an async HTTP client, `pyarrow`, `duckdb`, and the dataframe library already used by the repo.
- **Security and compliance**: no secrets in code, logs or manifests; `.gitignore` covers data, `.env`, raw archives; never upload or share FMP data; do not add any redistribution feature.
- **Documentation** (all in `docs/`): `fmp_probe_report.md`, `fmp_paid_probe_report.md`, `datamanager_integration.md`, `decisions.md`, `storage_schema.md`, `runbook.md`, plus a `README` section with the quick start.

---

## 14. Gate G3: `fmp verify --final` checklist

`fmp verify --final` exits non-zero unless **all** hold, and prints each line as pass/fail:

1. No pending, running, or unexplained failed tasks (V-COMPLETE); a `skipped_by_plan` list exists and matches the approved `budget_plan.md`.
2. All FAIL-severity validators pass (section 11); all WARN findings are listed in the report for the owner's review.
3. Raw archive integrity and backup:
   - `fmp verify` writes **one archive-wide checksum file** `ARCHIVE.sha256` in standard `sha256sum -c` format (relative paths, one line per archive file) for the whole raw archive, and a second one for the active vintage's Parquet files; it records the file count and total bytes alongside, and prints the hash of the checksum file itself.
   - It also reads every gzip member end to end (a decompression test), because a plain copy can silently carry a corrupt file.
   - The owner then copies the archive, the vintage, and the checksum files to a separate volume. The command asks for the backup path and runs the check **there** (`sha256sum -c` equivalent plus file count and total bytes); any mismatch is a failure.
   - After this step the raw archive is marked read-only; later appends (a new dump) go to a new `dump_id` directory and produce a new checksum file.
   - Free disk space is adequate (section 6.8).
4. The reconciliation sample and the V-ADJ-RECON report exist.
5. `fmp build` has been re-run from the archive and reproduces identical hashes (V-REPRO).
6. The DataManager adapter passes its backward-compatibility tests and the new-source tests (section 15).
7. The owner has confirmed that any item still not downloaded (for example 1h for lower tiers) is intentionally postponed to a second month or abandoned. List those items with the estimated bytes.

Only after this prints "READY TO CANCEL" does the owner cancel the FMP subscription.

---

## 15. Acceptance criteria

**Phase 1 (before purchase)**

- `fmp probe-free` runs within 200 calls and produces a complete probe report; every [VERIFY] item in sections 3 and 5 is either confirmed, refuted, or explicitly marked "requires paid probe".
- The downloader passes all tests against the mock server, including resume after kill, retry/backoff, circuit breaker, plan-restricted handling, quota-exhausted stop, and the bandwidth hard stop.
- `fmp plan --dry-run` prints the task counts and byte projections per block and tier for both counting hypotheses (compressed and decompressed), using measured bytes per row.
- `adjust()` and `yfinance_to_raw()` are unit-tested and pass the section 8.3 vectors on synthetic data.
- `docs/datamanager_integration.md` is complete and the adapter design is described, with a list of every legacy caller to be regression-tested.

**Phase 2 and 3**

- The paid probe resolves the open questions in section 3 and the report is delivered at G2.
- The full run follows `budget_plan.md`, never exceeds the safety cap, and ends with every task in a terminal state.

**Phase 4 and 5**

- `fmp build` creates a vintage; `fmp validate` has no FAIL results; the report is delivered.
- DataManager integration: a regression suite proves legacy calls return **identical** results to before the change for data served from the legacy cache; new-source calls return the right columns and semantics (section 9.3); an end-to-end test shows that after a synthetic new split the combined series (base plus yfinance increment) is continuous.
- 1d and 1h reads for one symbol over the full history complete within the documented performance targets.

---

## Appendix A: Open questions the agent must answer by probing (not by asking)

1. Which intraday intervals does the owner's plan serve, from which date, with what row cap? (paid probe)
2. Is the bandwidth cap measured on compressed or decompressed bytes? (compare local counters with the dashboard; ask the owner to read the dashboard value and enter it with `fmp calibrate-usage --dashboard-gb X`)
3. Which `dividends` field is on the raw basis? (free probe, AAPL 2020-08-07)
4. Does `historical-price-eod/non-split-adjusted` return delisted symbols? Which exchanges?
5. Do the three `historical-*-constituent` endpoints and the index price symbols work on the plan?
6. Does the `from`/`to` range work for `splits-calendar`, `dividends-calendar`, `earnings-calendar`, and what is the maximum window?
7. What are the actual exchange code strings and symbol formats for class shares?
8. What does legacy `close` represent in the owner's files (split-adjusted or raw)? (Phase 0a)
9. Time zone and label convention of intraday bars; are extended-hours bars present? (paid probe)
10. For tickers reused by different companies, does FMP return a merged series, one company's history, or separate histories? (section 4.2 item 8)
11. Is the intraday time zone US/Eastern, and does the first bar of every regular day land at 09:30 ET after conversion in the weeks around each DST change? (section 8.4)

## Appendix B: Example `config/fmp.yaml`

```yaml
fmp:
  base_url: https://financialmodelingprep.com/stable/
  api_key_env: FMP_API_KEY
  dump_id: "2026-10"
  rate_limit_per_min: 600        # 80% of Premium's 750
  workers: 12
  bandwidth:
    cap_gb: 50
    safety: 0.90
    basis: decoded               # 'wire' only after the probe proves the cap counts compressed bytes
    window_days: 30
  history:
    daily_from: "2000-01-01"     # cheap to over-fetch for daily; Premium offers 30+ years
    intraday_from: "earliest"    # resolved by the paid probe
  intraday:
    interval: "1hour"
    tiers: [A, B, C, D, E]
  flags:                         # section 7.5; thresholds are defaults to be calibrated
    daily_spike_abs_return: 0.35
    daily_spike_robust_z: 8
    revert_window_bars: 2
    jump_ratio_tolerance: 0.03
    stale_min_bars_daily: 5
  reconciliation_sample:
    size: 300
    fixed_symbols: [AAPL, MSFT, TSLA, AMZN, NVDA, GOOGL, JPM, XOM, KO, T, GE, BRK-B]
duckdb:                          # one profile per machine, selected by hostname or an environment variable
  laptop:
    memory_limit: auto           # about half of the installed RAM
    threads: auto
    temp_directory: <local SSD path>/duckdb_tmp
  pi:
    memory_limit: 6GB            # Raspberry Pi 5, 16 GB RAM total, shared with the strategies
    threads: 4
    temp_directory: <local SSD path>/duckdb_tmp
increments:
  writer_host: <laptop|pi>       # section 10.4; exactly one writer
paths:
  data_root: data/fmp
  backup_path: <NAS mount point; 6 TB, network share>   # section 14; may not support symlinks
  legacy_yfinance_cache: <fill from docs/datamanager_integration.md>
```

## Appendix C: Quick reference of facts and test vectors

| Item | Value |
|------|-------|
| Premium rate limit | 750 requests/min [VERIFY] |
| Bandwidth cap | 50 GB per trailing 30 days (Premium) [VERIFY] |
| Regular US session | 09:30 to 16:00 ET; early close 13:00 ET |
| Expected 1h bars per full day | 7 (six full hours plus a half-hour bar at 15:30) [VERIFY] |
| yfinance 1h history limit | about 730 days |
| AAPL splits | 7-for-1 on 2014-06-09; 4-for-1 on 2020-08-31 |
| TSLA splits | 5-for-1 on 2020-08-31; 3-for-1 on 2022-08-25 |
| AMZN / GOOGL / NVDA splits | 20-for-1 on 2022-06-06 / 20-for-1 on 2022-07-18 / 10-for-1 on 2024-06-10 |
| AAPL dividend | $0.82 per share, ex-date 2020-08-07 (pre-split, raw basis) |

## Appendix D: Rough size expectations for planning (replace with measured numbers)

- Universe size: about 25,000 to 30,000 US symbols including delisted (the real count must be reported by Phase 0 or Phase 2).
- 1h bars: about 250 to 450 million rows in total. Parquet (zstd): about 4 to 6 GB. `csv.gz`: about 7 to 10 GB. Raw JSON: about 40 to 50 GB decompressed.
- 1d raw bars: about 60 to 100 million rows; Parquet in the low single-digit GB range.
- Raw archive (gzip): about 10 to 25 GB; see section 6.8 for the full disk budget and check free disk space before each phase.

## Appendix E: Changes from v1.0 (external review)

| Change | Where |
|--------|-------|
| Out-of-scope list added (no backtester or strategy changes; the deliverable ends at the DataManager interface) | section 1 |
| New decision D11: flag suspect data, never alter or drop it | section 2 |
| Ticker-reuse probe, and identity rules: segment by listing entries and clip bars; no splitting by CIK from prices (bars carry no CIK) | 4.2 item 8, 7.3 |
| Suspect-data flags (`flags` bitmask, missing-split candidates, `exclude_suspect` option) | 7.3, 7.5, 9.3, V-FLAGS, Appendix B |
| Backtester-relevant semantics of dividends and adjClose documented | 9.3 |
| Explicit time-conversion procedure with DST-week test windows | 8.4, V-INTRADAY-ALIGN, Appendix A |
| Archive-wide `ARCHIVE.sha256`, end-to-end gzip read test, checksum verification on the backup copy, archive made read-only | 14, V-RAW-ARCHIVE |
| Disk budget with recommended free space and SSD note | 6.8 |
| Two fixed environments: Windows 11 laptop = development, download, build (writer); Raspberry Pi 5 = read-only production server; NAS = backup and hand-off. Windows-specific handling, `reader`/`pipeline` package split, `fmp publish`/`fmp sync`, single increment writer, per-machine DuckDB profiles | 4.1, 10.4, 13, Appendix B |
