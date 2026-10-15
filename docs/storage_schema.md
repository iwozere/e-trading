# FMP data store: storage schema and read semantics

What is on disk after `fmp build`, what every column means, and how prices must be read. Written for whoever
builds backtests or research on this data later (spec §9.3, §13). The code is the authority. File and constant
names are given so each statement can be checked: the schemas live in `src/data/fmp_pipeline/builders/`
(`builder.py`, `events.py`, `prices.py`, `records.py`, `tables.py`) and `increments/store.py`.

FMP data is for personal use only (D5). None of these files may be uploaded, shared or committed.

## 1. Where things are

One `data_root` per machine. The default is `DATA_CACHE_DIR/fmp-data`, for example `d:/data-cache/fmp-data` on
the laptop. `--data-root` overrides it, and so does `fmp.adapter.data_root` for the DataManager.

```
<data_root>/
  raw/<dump_id>/<endpoint>/<symbol>.jsonl.gz   append-only archive of every FMP response (laptop only)
  manifests/<dump_id>.sqlite                   download task manifest + bandwidth log (laptop only)
  vintages/<dump_id>/                          the Parquet store built from raw/ (immutable once built)
  current.txt                                  the active <dump_id> (a text file, never a symlink)
  security_ids.json                            append-only security_id registry (laptop only)
  increments/                                  yfinance bars and events newer than the vintage (section 6)
  snapshots/<YYYY-MM-DD>/                      monthly universe snapshots (section 8)
  reports/<dump_id>/                           plan, build, validation and drift reports
  checksums/<dump_id>/                         ARCHIVE.sha256, VINTAGE.sha256 (sha256sum -c format)
  probe-free/, probe-paid/                     cached probe responses and ledgers (laptop only)
  locks/                                       one-writer locks (dump, increments, sync)
```

The Pi (reader) has only `vintages/`, `current.txt`, `increments/` and `snapshots/`. It gets the vintage through
`fmp sync`. It ingests increments and snapshots itself, because it is the increment writer (Q4).

A vintage is never changed after it is built. A newer dump is a new `<dump_id>` next to it. `current.txt`
selects which one is read, and old vintages stay until the owner deletes them.

## 2. Vintage layout

```
vintages/<dump_id>/
  reference/        securities, symbol_history, profiles, delisted, stock_list, actively_trading, etf_list,
                    symbol_changes, exchanges, sectors, industries, countries, statement_symbols, cik_list,
                    shares_float, index_list, market_risk_premium, holidays, market_hours, employee_count,
                    historical_employee_count, etf_info, etf_holdings
  events/           splits, dividends, earnings, mna
  prices/
    1d_raw/<bucket>/<security_id>.parquet      raw daily bars, one file per security
    1h_raw/<bucket>/<security_id>.parquet      raw hourly bars, one file per security
    recon/          daily_split_adjusted_sample, daily_div_adjusted_sample   (reconciliation sample only, D8)
  index_membership/ index_changes, index_members_current
  fundamentals/     income_statement, balance_sheet, cash_flow, key_metrics, ratios, enterprise_values,
                    *_as_reported, full_as_reported, *_growth, analyst_estimates, market_cap,
                    financial_reports_dates, financial_scores, owner_earnings,
                    revenue_product_segmentation, revenue_geographic_segmentation, grades_historical,
                    ratings_historical
  macro/            treasury_rates, economic_indicators, calendars
  _meta/build_info.json                        code version, input digests, settings, per-file rows + sha256
```

Every table is `<name>.parquet`. Only tables whose endpoint was actually downloaded exist. An archived endpoint
that no table reads is listed as `unbuilt` in `build_info.json`, and its responses stay in `raw/`.

`<bucket>` is `security_id % 256` as two lowercase hex digits, for example `prices/1d_raw/2a/42.parquet`. It
serves only to keep directories small. Find a file through `securities`, never by guessing.

Parquet settings: zstd (level `build.zstd_level`, default 6), row groups of 100,000 rows, statistics on,
byte-stream-split for float columns. They are recorded in `build_info.json`. Price files are sorted by time.

## 3. Identity

### `reference/securities.parquet`

There is one row per security. A ticker is an attribute; `security_id` is the key (D7).

| Column | Type | Meaning |
|---|---|---|
| `security_id` | int64 | Internal id. Allocated once in `security_ids.json`, never reused. A later dump keeps the id of every symbol it shares with this one. |
| `symbol` | string | Current or last known FMP symbol |
| `yf_symbol` | string | yfinance spelling: class shares `BRK.B` become `BRK-B`, and indices keep `^` |
| `exchange`, `name` | string | From the profile, or from the delisted list |
| `instrument_type` | string | `stock`, `adr`, `etf`, `fund`, `index`, `warrant`, `unit`, `right` or `preferred`. Warrants, units, rights and preferreds are recognised from the name (a heuristic), and their rows are kept. |
| `cik`, `isin`, `cusip` | string | Nullable |
| `ipo_date`, `delisted_date` | date32 | Nullable. A future `delisted_date` is a *scheduled* delisting (probe F13). |
| `is_actively_trading` | bool | Nullable |
| `first_price_date`, `last_price_date` | date32 | First and last daily bar in this vintage |
| `source_flags` | string | Which FMP lists named the symbol |
| `tier` | string | `A` to `E`, the download tier of spec §6.3 |
| `valid_from`, `valid_to` | date32 | Set only for one segment of a reused ticker. Both are inclusive; null means open-ended. |
| `identity_uncertain` | bool | The evidence about who this symbol is was contradictory (spec §7.3 rule 2) |
| `uncertain_reason` | string | Why |

Reused tickers are split into segments only when `build.segment_reused_tickers` is on, and it is off until the
paid probe shows that FMP merges such series (D46). Until then, a reused ticker is one security with
`identity_uncertain = true`, and the case is listed in `reports/<dump_id>/identity_uncertain.csv`.

### `reference/symbol_history.parquet`

Columns: `security_id`, `symbol`, `valid_from`, `valid_to` (inclusive dates; null means open-ended) and
`reason`. Rows come from `symbol-change`: one row per earlier symbol of a renamed security.

### Symbol lookup

`FmpStore.resolve` tries, in order:
1. the security's `symbol`;
2. its `yf_symbol`;
3. the same two with `.` and `-` swapped;
4. `symbol_history`, using `as_of` when one is given.

If more than one security matches and `as_of` does not narrow it to exactly one, the lookup raises
`AmbiguousSymbol`, a subclass of `ValueError`. It never guesses.

## 4. Events

### `events/splits.parquet`

| Column | Type | Meaning |
|---|---|---|
| `security_id`, `symbol` | int64, string | |
| `date` | date32 | Effective date: the **first trading day at the new share count** |
| `numerator`, `denominator` | float64 | As FMP reports them |
| `ratio` | float64 | `numerator / denominator`: 4.0 for a 4-for-1, 0.1 for a 1-for-10 reverse split |
| `source` | string | `fmp` (increments use `yfinance`) |
| `_extra` | string | Every other field of the response, as JSON |

### `events/dividends.parquet`

| Column | Type | Meaning |
|---|---|---|
| `security_id`, `symbol` | int64, string | |
| `ex_date` | date32 | The date the price effect happens |
| `record_date`, `payment_date`, `declaration_date` | date32 | Nullable. FMP sometimes sends an empty string, which is stored as null. |
| `amount_raw` | float64 | Cash per share **on the raw price basis of the ex-date**. This is FMP's `dividend` field (probe F1: AAPL 2020-08-07 = 0.82). |
| `amount_split_adjusted` | float64 | FMP's `adjDividend`, restated to today's share count |
| `frequency`, `source` | string | |
| `_extra` | string | Every other field, as JSON |

Dividends carry **dates only**: no time of day and no session flag. None is needed, because the price effect of
a dividend is on the **ex-date** bar.

`events/earnings` and `events/mna` are generic record tables (section 7.1).

## 5. Prices

All stored prices are **raw**: exactly what traded on that day, never adjusted for later splits or dividends
(D3). Adjusted values are computed when the data is read (section 9) and are never stored.

### `prices/1d_raw/<bucket>/<security_id>.parquet`

| Column | Type | Meaning |
|---|---|---|
| `security_id` | int64 | |
| `date` | date32 | Trading day (exchange-local) |
| `open`, `high`, `low`, `close` | float64 | Raw prices. They come from FMP's `non-split-adjusted` endpoint, which names them `adjOpen` … `adjClose` even though they are raw (probe F3). |
| `volume` | int64 | Raw shares |
| `flags` | int32 | Suspect-data bitmask (below) |
| `source` | string | `fmp` |
| `vintage` | string | The dump id |

The rows are sorted by `date`, with exactly one bar per date. When FMP returned two different bars for one date,
the newest response wins, and the conflicts are counted in `build_info.json`.

### `prices/1h_raw/<bucket>/<security_id>.parquet`

Same columns, except that `date` is replaced by two columns:

| Column | Type | Meaning |
|---|---|---|
| `ts_utc` | timestamp[us, UTC] | **Start** of the bar, in UTC |
| `date_et` | date32 | Exchange-local trading day the bar belongs to |

FMP's local timestamp strings are converted per bar with `zoneinfo`, never with a fixed offset. The source zone
and label are set by `build.intraday_tz` and `build.intraday_label`, and `fmp probe-paid` measures both. A label
of `end` is shifted, so that `ts_utc` is always the bar start. The expected regular session is 7 bars:
09:30, 10:30, … 14:30, plus a half-hour bar at 15:30 ET. Early-close days have fewer. The original strings stay
in `raw/`.

### `flags` bitmask

The flags never change a value (D11); they only mark bars as suspect. Thresholds live in `fmp.build.flags`.

| Bit | Value | Name | Rule (defaults) |
|---|---|---|---|
| 0 | 1 | `OHLC_VIOLATION` | `low > min(open, close)`, `high < max(open, close)`, or a price missing or ≤ 0 |
| 1 | 2 | `SPIKE_REVERTED` | Return ≥ 35% with a robust z-score ≥ 8 against the previous 60 returns, reverted within 1 to 2 bars, and no split or dividend within ±2 sessions |
| 2 | 4 | `JUMP_NO_SPLIT` | Close ratio within 3% of 2, 3, 4, 5, 10, 3/2 or their inverses, with no split within ±2 sessions: a probable missing split. Also listed in `reports/<dump_id>/missing_splits_candidates.csv`. |
| 3 | 8 | `STALE` | At least 5 daily (35 hourly) consecutive bars with identical OHLC and zero volume |
| 4 | 16 | `ZERO_VOLUME` | Volume 0 while `high > low` |

Daily bars get all five flags. Hourly bars get bits 0, 3 and 4 only (D45). Penny stocks and delisting runs
legitimately collect many flags, so a flag means "look at this", not "wrong".

## 6. Increment layer (`<data_root>/increments/`)

Bars come from yfinance and continue the vintage after its last bar. They are converted back to raw prices
before they are stored (spec §10.1), so they line up with the vintage's raw prices.

```
increments/
  1d/<bucket>/<security_id>.parquet       vintage daily schema    + ingested_at (timestamp, UTC)
  1h/<bucket>/<security_id>.parquet       vintage intraday schema + ingested_at
  events/<bucket>/<security_id>.parquet   security_id, type (split|dividend), date, numerator, denominator,
                                          amount (raw basis), source, ingested_at
  _meta.json                              increment_seq, writer host, last run report
  archive/<old dump_id>/...               increments a newer vintage covers (moved by fmp archive-increments)
  archive/before-<new dump_id>/...        events a newer vintage covers
```

Rules that affect readers:
- Every increment bar is dated strictly after the security's last stored bar. Stored bars are never rewritten,
  and `source = 'yfinance'`.
- `vintage` names the vintage the bar extends.
- Only `writer_host` writes (D57). Every other machine gets the layer by `fmp sync --increments`.

## 7. Other vintage tables

### 7.1 Generic record tables

Every non-price endpoint becomes one table, without losing anything (`records.py`):

- column `security_id` (int64, null if the symbol is not a security of this vintage);
- a label column where several endpoints feed one table: `index` in `index_membership/*` (`sp500`, `nasdaq`,
  `dowjones`), and `calendar` in `macro/calendars` (`splits`, `dividends`, `earnings`, `ipos`, `economic`);
- typed snake_case columns for the fields named in `tables.py`, for example `symbol`, `date`, `period`,
  `fiscal_year`, `filing_date`, `accepted_date`;
- in numeric tables (statements, metrics, macro, the recon sample), one float64 column per numeric FMP field,
  **under FMP's own field name**;
- `_extra`: every other field, plus any value that did not fit its typed column, as compact JSON with sorted keys.
  A field nobody uses today can be recovered from it without downloading again.

Within each request, only the newest successful response counts. Rows dated outside their own request's window
are dropped and counted (FMP ignores a bad `from`, probe F7).

### 7.2 Point-in-time use of fundamentals

Statement tables carry `filing_date` (date) and `accepted_date` (string, `YYYY-MM-DD HH:MM:SS`). A backtest must
join on these columns, not on `date` (the period end). Otherwise it uses numbers before they were published.
V-FUND-PIT reports rows where they are missing or earlier than the period end.

### 7.3 Index membership

`index_members_current` lists today's members, with `date_first_added`. `index_changes` is the change log.
Membership on a past date is reconstructed by rolling the log backwards from the current list; V-INDEX checks
that reconstruction for the S&P 500 at five dates. The fields describing what was added and removed are kept in
`_extra` with FMP's names.

## 8. Snapshots (`<data_root>/snapshots/<YYYY-MM-DD>/`)

These are written by `fmp snapshot-universe`, which needs no FMP key (D69 to D71):

| File | Content |
|---|---|
| raw source files | `nasdaqlisted.txt`, `otherlisted.txt`, and the index lists exactly as downloaded |
| `symbols.csv` | Parsed US symbol directory, test issues removed |
| `constituents.csv` | S&P 500, Nasdaq-100 and Dow members |
| `diff.json`, `diff.md` | Added and removed symbols compared with the previous snapshot |
| `last_bars/` | One last yfinance download per removed symbol, stored raw like increments |
| `delisted_suspected.csv` | Removed symbols, with the snapshot date |
| `SHA256SUMS` | Checksums of all of the above |

The vintage is immutable, so a suspected delisting is recorded here and not in `securities`.

## 9. Read semantics

### 9.1 Through the DataManager

`DataManager.get_ohlcv` and `get_ohlcv_batch` serve FMP data only when `fmp.adapter.enabled` is true (D50). When
it is false, nothing changes for any caller. When it is true:

- **Resolution (`fmp_first`).**
  1. Take the vintage's raw bars up to its last bar.
  2. Append increment bars **strictly after** that bar.
  3. Adjust the combined raw series with the union of both event sets. FMP wins on a shared date.

  A symbol or timeframe the vintage does not have falls through to the legacy yfinance cache and its
  cache-on-miss loader, unchanged.
- **Output:**
  - a tz-naive `DatetimeIndex` named `timestamp`: the date for daily bars, the **UTC bar start** for hourly
    bars, as in the legacy files;
  - columns `open`, `high`, `low`, `close`, `volume`;
  - the range is inclusive at both ends.
- **`adjust=`** (keyword only):

  | Value | `close` is | Volume |
  |---|---|---|
  | `legacy` (default) | same as `total`, matching the legacy files (D12) | split-adjusted |
  | `raw` | the traded price on that day | raw |
  | `split` | split-adjusted (yfinance `Close` without dividends) | split-adjusted |
  | `total` | split **and** dividend adjusted (total-return basis) | split-adjusted |

  Open, high and low use the same factors as close.
- **Opt-in extras:**
  - `include_source=True` adds `source` (`fmp` or `yfinance`);
  - `include_flags=True` adds `flags`;
  - `exclude_suspect=True` drops every bar with any flag set;
  - `as_of=` disambiguates a reused or renamed symbol.

  By default nothing is filtered (legacy compatibility).
- **Caching:** an in-memory LRU of adjusted histories, keyed by vintage, security, timeframe, mode, a hash of the
  events used, and the increment tail. A new dividend or a new vintage can never be served stale. Nothing
  FMP-derived is written into the legacy cache.

### 9.2 Adjustment definitions (`src/data/fmp_pipeline/adjust/adjust.py`)

Notation: a split with `ratio` is effective on date `S`. A dividend of `d` (raw basis) has ex-date `D`, and
`C_prev` is the **raw** close of the last trading day before `D`.

```
split_factor(t) = product over splits with S > t of 1 / ratio
DF(D)           = 1 - d / C_prev
div_factor(t)   = product over dividends with D > t of DF(D)
close_split(t)  = close_raw(t) * split_factor(t)
close_total(t)  = close_split(t) * div_factor(t)
volume_split(t) = volume_raw(t) / split_factor(t)
```

Events dated after the last bar do not change existing bars. Our `adjust()` reproduces FMP's split-adjusted
series exactly, and its dividend-adjusted series within 2.5e-5 (probe F4).

### 9.3 What a backtest must know

- **`total` and `legacy` returns include dividends**, credited on the ex-date. Do not add dividend cash on top.
- **`raw` and `split` closes do not include dividends.** A strategy that trades on them must book dividend cash
  from `events/dividends` (`amount_raw` on the ex-date, per share held at the close before it). Otherwise
  dividend income silently disappears.
- **A split is effective on the first trading day at the new share count.** In raw prices, that day's bar is
  the first one at the new price level.
- **Hourly bars are adjusted with the daily factors of their exchange-local date** (spec §9.4). For a dividend,
  `C_prev` is the previous **daily** session close, so all hourly bars of one day share one factor.
- **Adjusted history is relative to today.** The next dividend or split changes every earlier `split` or
  `total` value; only `raw` never changes. Store raw prices plus the mode if a result must be reproduced exactly
  later.
- **Survivorship:** delisted securities are in `securities`, with `delisted_date` and their bars up to about that
  date. A universe selected "as of" a past date must use `ipo_date`, `delisted_date`, `first_price_date`,
  `last_price_date` and index membership. Using today's lists brings back survivorship bias.
- **Flags are advisory.** Decide per study whether to use `exclude_suspect` or to filter specific bits
  (`flags & 4` marks a probable missing split, for example).

### 9.4 Direct queries (DuckDB, read-only)

Run queries on a local disk, never on the NAS or `R:`. Resolve the vintage from `current.txt` first.

```python
from pathlib import Path

import duckdb

root = Path("d:/data-cache/fmp-data")
v = root / "vintages" / (root / "current.txt").read_text(encoding="utf-8").strip()
con = duckdb.connect()
sec = con.execute(f"SELECT security_id, symbol, delisted_date FROM '{v}/reference/securities.parquet' "
                  "WHERE symbol = 'AAPL'").df()
bars = con.execute(f"SELECT * FROM read_parquet('{v}/prices/1d_raw/*/*.parquet') "
                   "WHERE date = DATE '2020-03-16'").df()      # cross-section, raw prices
```

Raw prices from direct queries are **not adjusted**. To adjust them, use `adjust()` together with the events
tables, or use the DataManager. Never mix with FMP's `recon/` adjusted values, which exist for a sample only.

## 10. Reports and checksums

| Path | Written by | Content |
|---|---|---|
| `reports/<dump_id>/budget_plan.md`, `universe_report.md` | `fmp plan` | Bandwidth and request projection; universe exclusions by rule |
| `reports/<dump_id>/missing_splits_candidates.csv`, `identity_uncertain.csv` | `fmp build` | For the owner's review |
| `reports/<dump_id>/drift_report.md/.json` | `fmp build` | Earlier increments compared with the new vintage (D73) |
| `reports/<dump_id>/validation_report.md/.json`, `coverage_report.md` | `fmp validate`, `fmp verify` | Spec §11 |
| `checksums/<dump_id>/ARCHIVE.sha256`, `VINTAGE.sha256` | `fmp verify`, `fmp backup` | `sha256sum -c` format, relative paths |
| `vintages/<dump_id>/_meta/build_info.json` | `fmp build` | Code version, input digest, settings, rows and sha256 per output file (V-REPRO) |
