# DataManager integration of the FMP vintage

Spec §9 (`docs/FMP_DATA_PIPELINE_SPEC-2.md`), plan §1.4 and WP-8 (`docs/FMP_DATA_PIPELINE_IMPLEMENTATION_PLAN.md`),
decisions D12, D50–D55 and D56–D60 (`docs/decisions.md`). Status 2026-10-15: adapter and yfinance increment layer
built, adapter **off by default**; golden regression snapshot captured and clean.

## 1. The DataManager as it is (Phase 0a recon, refreshed 2026-10-15)

- **Entry points** (`src/data/data_manager.py`): `DataManager.get_ohlcv(symbol, timeframe, start_date, end_date,
  force_refresh=False)` and `get_ohlcv_batch(symbols, timeframe, start_date, end_date, force_refresh=False)`;
  module factory `get_data_manager()`. Fundamentals, live feeds and derivatives are untouched by this work.
- **Return contract** (`src/data/ohlcv_contract.py`): tz-naive `DatetimeIndex` named `timestamp` (bar open; the
  legacy files hold `datetime64[us]`), lowercase `open`, `high`, `low`, `close` (float64), `volume` (int64), no
  `timestamp` column. Daily bars are at midnight; hourly bars are the **UTC** bar start (a regular-session 1h file
  starts at `13:30` in summer, `14:30` in winter).
- **Cache-on-miss flow**: `UnifiedCache.get` → prefix/suffix/intermediate gap detection with `_GAP_TOLERANCE_BARS`
  → missing segments from the provider chain → merge-safe cache write. Legacy cache layout:
  `<DATA_CACHE_DIR>/ohlcv/<SYMBOL>/<timeframe>/<year>.csv.gz` (+ `.metadata.json`); `DATA_CACHE_DIR` is
  `d:/data-cache` on the writer laptop.
- **What the legacy files hold — Finding A (D12):** yfinance `auto_adjust=True`, i.e. prices **split and
  dividend adjusted as of the download date**, no `adjClose` column. So `adjust="legacy"` means `total`, not
  `split`, and the legacy series are not reproducible (a re-download moves every number by the dividends paid
  since).
- **Cache size** (2026-10-15): 797 symbols, 1,294 `(symbol, timeframe)` series: 779 × 1d, 499 × 1h, 4 each of
  5m/15m/30m/4h.

## 2. What WP-8 adds

```
DataManager.get_ohlcv(symbol, tf, start, end, force_refresh=False, *, adjust="legacy",
                      include_source=False, include_flags=False, exclude_suspect=False, as_of=None)
  │
  ├─ fmp_adapter is None (default) or force_refresh ───────────────────────► legacy path (unchanged)
  │
  └─ FmpOhlcvAdapter.get_ohlcv(...)            src/data/fmp_pipeline/datamanager_adapter/adapter.py
        ├─ FmpStore.resolve(symbol, as_of)     fmp_store.py — vintage named by current.txt
        │     unknown symbol, timeframe not served, or no price file for it ─► None ─► legacy path
        ├─ raw bars (vintage) + increment bars strictly after the last one (IncrementStore, WP-9)
        ├─ events: vintage ∪ increments, deduplicated by date, FMP wins
        └─ adjust(raw, splits, dividends, mode)   (legacy → total) ─► contract frame
```

- **Hook.** `DataManager.__init__` sets `self.fmp_adapter = _load_fmp_adapter()`. The previous bodies of
  `get_ohlcv` and `get_ohlcv_batch` moved verbatim into `_get_ohlcv_legacy` / `_get_ohlcv_batch_legacy`; the
  public methods only add the FMP check in front (D50). The loader imports the FMP package lazily and logs any
  failure instead of raising, so the adapter can never stop a `DataManager` from starting.
- **Policy `fmp_first`** (spec §9.2, the only policy): a symbol the vintage knows, in a timeframe it has a file
  for, comes from FMP — including an empty frame when the range has no bars (a delisted security does not fall
  through to a fresh yfinance download). Everything else takes the legacy path, which keeps writing new downloads
  into the legacy cache (D2). FMP-derived data is **never** written into the legacy cache.
- **`force_refresh=True`** always takes the legacy path: the vintage is immutable, a refresh means "ask the
  providers".
- **Options** (spec §9.3, all opt-in): `adjust` = `legacy` (default, = `total`), `raw`, `split`, `total`;
  `include_flags` adds the suspect-data bitmask (spec §7.5), `exclude_suspect` drops flagged bars, `as_of`
  disambiguates a reused or renamed symbol, `include_source` adds `source` (`fmp`/`yfinance` on the FMP path,
  `legacy` on the legacy path). `adjust ≠ legacy`, `include_flags` and `exclude_suspect` **raise `ValueError`**
  for a symbol the vintage does not serve: the legacy files cannot provide them, and silently returning
  total-adjusted prices for `adjust="raw"` would be wrong.
- **Symbols** (spec §9.6, D53): own symbol → `yf_symbol` → class-share separator swapped (`BRK-B` ↔ `BRK.B`) →
  `symbol_history`. Several matches without a disambiguating `as_of` raise `AmbiguousSymbol` (a `ValueError`, so
  the `retry_on_exception` decorator does not retry it). With reuse segmentation off (the default, D46) a ticker
  is never split, so this only happens for former symbols shared by two securities.
- **Intraday** (spec §9.4): hourly bars are adjusted with the daily split/dividend factors **by exchange-local
  date**, with `C_prev` from the daily session closes; the index is the UTC bar start like the legacy 1h files.
- **Batch** (spec §9.5): `get_ohlcv_batch` serves the FMP symbols through one DuckDB query over their daily files
  (plus one read per events table) and the rest through the legacy batch, in request order.
- **Caching** (spec §9.5, D52): adjusted full histories in an in-memory LRU (256 entries) keyed by vintage,
  security, timeframe, mode, a hash of the events used and the increment tail. A new vintage or a new dividend can
  therefore never be served from a stale entry.

## 3. Turning it on

1. A built vintage: `fmp build` has run and `<data_root>/current.txt` names it.
2. The reader dependencies on that machine: `pip install -r requirements-fmp-reader.txt` (pyarrow, duckdb,
   tzdata).
3. `config/fmp/fmp.yaml`:

   ```yaml
   fmp:
     adapter:
       enabled: true
       data_root: ""          # "" = DATA_CACHE_DIR/fmp-data
       timeframes: [1d, 1h]   # timeframes served from FMP; anything else stays on the legacy path
   ```
4. Re-run the golden snapshot (§5) and review every changed series: only symbols the vintage serves may differ.

**Live jobs need current increments.** The adapter appends `<data_root>/increments/` (WP-9) after the vintage's
last bar, so turn it on for live jobs only on a machine where `fmp ingest-increments` runs (the writer host,
D57) or where increments are synced to from it. Without them a request past the vintage's last bar returns only
what the vintage has. Backtests that end before the vintage date are always safe.

### Keeping increments current (spec §10.2.7)

On the writer host only (`fmp.increments.writer_host`), daily after the US close. The repository's scheduler has
the job registered but **disabled** (`FMP Increments Ingest` in `src/data/pipeline/specs/fmp_specs.py`, 23:00 UTC
Mon–Fri); enable its `job_schedules` row on that machine. A plain cron equivalent:

```
# m  h  dom mon dow  command   (UTC; the Pi's checkout)
  0  23  *   *  1-5  cd /opt/e-trading && .venv/bin/python -m src.data.fmp_pipeline.cli ingest-increments
```

Exit codes: 0 done, 1 stopped by Yahoo throttling (the next run resumes), 2 not the writer, locked or no vintage.
Run at least every 12 months: yfinance keeps about 730 days of hourly bars, and the run logs an error from 700
days of hourly gap on.

## 4. Measured performance (synthetic vintage, writer laptop, 2026-10-15)

| Read | Time |
|---|---|
| One security, whole history (6,783 daily bars, 104 dividends, 1 split), first read | ~40–46 ms |
| Same, served from the adjusted-history cache | ~21–23 ms |
| Batch, 50 securities × 2 years (26k rows), one DuckDB query | ~1.2 s |

Spec §9.5's target ("well under 100 ms" for one symbol's whole history) is met. Batch time is dominated by
`adjust()` normalizing tiny event tables (~14 ms per security of pandas overhead); a faster event path is a
Tasks item, not a correctness issue.

## 5. Regression strategy: golden-output snapshot

`fmp golden capture|compare` (`src/data/fmp_pipeline/datamanager_adapter/golden.py`) calls `get_ohlcv` for every
cached `(symbol, timeframe)` over its whole range **offline** (the DataManager gets no providers, so nothing is
downloaded or written back) and hashes what callers can observe: columns, dtypes, index type and name, index and
values. Hashes and row counts only go to `<data_root>/reports/golden/ohlcv_hashes.json`.

| Run (2026-10-15) | Result |
|---|---|
| Baseline, unmodified `DataManager` | 1,294 series hashed, none raising |
| Compare, unmodified code (determinism check) | 1,294 same, 0 changed |
| Compare after the hook, adapter **disabled** | **1,294 same, 0 changed, 0 missing, 0 added** |

Any difference with the adapter disabled is a bug. With it enabled, only symbols the vintage serves may change,
and each is reviewed (expected: tiny numeric differences from a newer adjustment date and FMP vs yfinance data).
The local cache keeps growing as jobs run, so re-capture the baseline before a review session.

```powershell
.venv\Scripts\python.exe -m src.data.fmp_pipeline.cli golden capture   # baseline
.venv\Scripts\python.exe -m src.data.fmp_pipeline.cli golden compare   # exit 1 on any difference
```

## 6. Caller inventory (2026-10-15)

The plan's recon counted 147 files mentioning `get_ohlcv`. Today: **106** files under `src/` (excluding
`fmp_pipeline`) plus 7 outside it; **39** of the `src/` files are provider downloaders that implement their own
`get_ohlcv` and are not `DataManager` callers. **17** use `get_ohlcv_batch`. Every `DataManager` call goes through
the one hooked chokepoint, which is why the golden snapshot covers the whole surface.

**Direct `UnifiedCache` users** (bypass the adapter, keep reading the legacy cache — consistent with D2):
`src/data/data_manager.py` (the legacy path), `src/data/ohlcv_contract.py` (contract docs/helpers),
`src/data/utils/annual_data.py`, `src/ml/pipeline/p17_penny_stocks/agents/market_agent.py`.

**Files constructing a `DataManager` (`DataManager(` or `get_data_manager(`), 44:**

```
src/common/alerts/alert_runner.py                      src/ml/pipeline/p13_bdsh/example_strategy.py
src/common/common.py                                   src/ml/pipeline/p14_ath/ath_pipeline.py
src/common/fundamentals.py                             src/ml/pipeline/p15_hidden_deps/p15_daily.py
src/data/data_manager.py                               src/ml/pipeline/p16_taleb/src/data_loader.py
src/data/examples/fundamentals_config_example.py       src/ml/pipeline/p18_institutional_flow_tracker/pipeline.py
src/data/examples/fundamentals_example.py              src/ml/pipeline/p18_institutional_flow_tracker/processors/volume_anomaly_detector.py
src/data/feed/base_live_data_feed.py                   src/ml/pipeline/p19_penny_intraday/label_backfill.py
src/data/tests/integration/test_data_manager_integration.py  src/ml/pipeline/p19_penny_intraday/shadow_loop.py
src/data/tests/performance/test_cache_performance.py   src/ml/pipeline/p19_penny_intraday/tests/test_watchlist_builder.py
src/data/tests/test_base_live_data_feed_cerebro.py     src/ml/pipeline/p19_penny_intraday/watchlist_builder.py
src/data/tests/test_data_manager_rate_limits.py        src/ml/pipeline/p20_kestrel/ingest/eod_ingest.py
src/data/tests/test_enhanced_fundamentals.py           src/ml/pipeline/p20_kestrel/tests/test_eod_ingest.py
src/data/tests/test_enhanced_symbol_classification.py  src/ml/pipeline/shared/pipeline_summary_generator.py
src/data/tests/test_fundamentals_cache.py              src/portfolio/pnl_alert/price_fetcher.py
src/data/tests/verify_datamanager_derivatives.py       src/portfolio/pnl_alert/tests/test_runner.py
src/data/utils/annual_data.py                          src/scheduler/main.py
src/data/utils/data_downloader.py                      src/strategy_pack/run.py
src/data/utils/periodic_fundamentals_refresh.py        src/strategy_pack/tests/test_signal_serialization.py
src/data/utils/populate_cache.py                       src/telegram/screener/enhanced_screener.py
src/data/utils/refresh_fundamentals_cache.py
src/ml/pipeline/p05_ai_selector/stages/stage1_prefilter.py
src/ml/pipeline/p05_ai_selector/stages/stage2_scorer.py
src/ml/pipeline/p06_emps2/emps2_pipeline.py
src/ml/pipeline/p12_order_flow/data_ingestion.py
src/ml/pipeline/p13_bdsh/data_loader.py
```

Regenerate: `git grep -lE "DataManager\(|get_data_manager\(" -- src scripts`.

## 7. Open items

- `fmp publish` / sync of vintage and increments from the writer to readers (Phase 5), and the drift report
  when the next vintage arrives (spec §10.3).
- Verify `yf_symbol` against the legacy cache directory names after the first real build (class-share separator
  is still a paid-probe item, F12).
- 1h is served only once the vintage has `1h_raw` files, i.e. after the paid probe sets `intraday.max_rows`.
- V-YF-XCHECK (WP-A) compares FMP `total` against the legacy files (Finding A).
