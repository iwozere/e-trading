# P15 Pipeline — Tasks

## ✅ Completed

- [x] **Daily bundle runner** (`p15_daily.py`) — Mon–Fri 13:00 UTC, 10 jobs
- [x] **Weekly bundle runner** (`p15_weekly.py`) — Friday 13:30 UTC, 6 jobs
- [x] **Self-healing gap detection** — each daily job fills up to 60 days of gaps per run;
      cutoff 2010-01-01, gaps healed most-recent-first
- [x] **Pipeline file log** — `results/p15_hidden_deps/pipeline.log` (RotatingFileHandler)
- [x] **yfinance prices** — 57-ticker P15 universe, range-fill into parquet
- [x] **CBOE put/call ratio** — full daily file replacement
- [x] **CNN Fear & Greed** — incremental append (daily) + full rebuild (weekly)
- [x] **GDELT v2 GKG & Events** — range fill via `download_gkg_range` / `download_events_range`
- [x] **FRED daily/weekly/monthly/quarterly** — incremental update, combined parquet rebuilt weekly
- [x] **SEC EDGAR submissions + facts** — daily submissions, quarterly facts via date-check
- [x] **FINRA TRF** — weekday range fill; silently skipped when credentials absent
- [x] **AAII investor sentiment** — weekly full download (Thursday publish cycle)
- [x] **GDELT 1.0 GKG cache** — one-shot backfill via `populate_gdelt1_cache.py` (2013-04-01 to present)
- [x] **YYYY.csv.gz storage format** — all downloaders (CBOE, Fear & Greed, AAII, FRED, GDELT v2, FINRA,
      yfinance prices) store tabular cache in per-year `YYYY.csv.gz` files via shared `yearly_csv` utility;
      human-readable, consistent with OHLCV cache convention

---

## 🔄 In Progress

*(nothing currently)*

---

## 🚀 Planned Enhancements

### ALFRED — Point-in-Time FRED Vintage Data

**Priority**: High — required for backtest correctness

**Why**: The current `FredDownloader` uses the standard FRED API (latest-revised values).
For heavily-revised macro series, this introduces look-ahead bias: a backtest on 2022-03-14
would see the revised CPI figure published months later, not the one available to traders
on that date. ALFRED (Archival FRED) provides point-in-time values using the same API key.

**Affected series** (only 6 of the 28 tracked series are significantly revised):

| Series      | Description         | Notes                          |
|-------------|---------------------|--------------------------------|
| `PAYEMS`    | Nonfarm payrolls    | Revised ±100k, up to 3 months |
| `CPIAUCSL`  | CPI                 | Minor but persistent revisions |
| `CPILFESL`  | Core CPI            | Same                           |
| `PCEPILFE`  | Core PCE            | Same                           |
| `UNRATE`    | Unemployment rate   | Minor                          |
| `INDPRO`    | Industrial prod.    | Occasional large revisions     |

Market/financial series (`DFF`, `T10Y2Y`, yield spreads) are never revised — no change needed.

**What changes**:
- New `vintages/` cache under `DATA_CACHE_DIR/fred/` with `(date, vintage_date)` MultiIndex
- New `FredDownloader.build_combined_realtime(as_of: date)` for point-in-time snapshots
- New quarterly job in the daily bundle to refresh vintage history
- Full spec in [`src/data/docs/Tasks.md`](../../../../data/docs/Tasks.md)

---

### Historical Backfill Scripts

**Priority**: Medium

One-shot scripts for sources without existing backfill tooling:

- [ ] `populate_prices_cache.py` — yfinance prices from 2010-01-01 to today
      (mirrors `populate_gdelt1_cache.py` pattern; daily bundle caps at 60 days so
      a fresh install takes ~months to catch up otherwise)
- [ ] `populate_fear_greed_cache.py` — run `download(full_rebuild=True)` once,
      then daily incremental handles the rest
- [ ] `populate_finra_cache.py` — iterate 2014-04-01 to today, weekdays only;
      heavy (~2500 API calls), run once manually

---

### Data Quality Monitoring

**Priority**: Medium

- [ ] Add a `p15_healthcheck.py` script that reads the latest run's
      `pipeline.log`, checks each source's watermark against yesterday,
      and emits a summary / alert if any source is more than 2 days stale
- [ ] Store per-run watermarks in `results/p15_hidden_deps/watermarks.json`
      so health checks don't require reading full parquet files

---

### GDELT v1 GKG Integration into Daily Bundle

**Priority**: Low

Currently populated via a separate one-shot script. Could be added as job #11
in `p15_daily.py` with a gap window using `_GDELT_V1_START = date(2013, 4, 1)`.
The `Gdelt1Downloader.download_gkg_range()` already handles skip-if-cached.
