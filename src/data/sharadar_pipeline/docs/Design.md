# Design

## Purpose

Turn a short Sharadar subscription into as many reliable research answers as possible, and leave nothing behind
that the licence forbids. See [Findings](Findings.md) for the facts this design rests on.

## Architecture

```
Sharadar bulk API ──> raw/ (zips, untouched)
                        │
                        ▼
                    store/ (parquet: prices_raw, splits, dividends, delistings,
                            fundamentals_ar, daily, securities, sp500, insiders, holdings)
                        │
        ┌───────────────┼────────────────┐
        ▼               ▼                ▼
   id mapping      cross-check      SharadarStore reader
   (CIK ↔ FMP      vs FMP vintage   (same shape as FmpStore)
   security_id)    (report in data        │
                   root only)             ▼
                                   research scripts ──> results/ (aggregates only, kept)
```

Data root: `<DATA_CACHE_DIR>/sharadar-data/` with `raw/`, `store/`, `reports/`, `manifest.json`. Everything under
it is deleted by the purge. Kept results go to the project `results/` tree.

### Components

| Component | Responsibility |
|---|---|
| `client.py` | Paged queries, bulk status, bulk download (follows the redirect), retries on 429 and 5xx |
| `download.py` | Bulk download of all tables (resumable) and the free sample |
| `manifest.py` | Files, sizes, hashes, download time, subscription dates, deletion deadline |
| `convert.py` | Pure conversions: raw prices, splits, dividends, delistings, securities, AR fundamentals |
| `build.py` | Raw CSV or zip to the Parquet store, large tables in chunks |
| `mapping.py` | Sharadar security to FMP vintage `security_id` |
| `store.py` | `SharadarStore`: `bulk_daily`, `events_many`, `fundamentals`, `delisting`, `delisting_return`, `resolve` |
| `crosscheck.py` | Adjustment check against `closeadj`; vendor check against the FMP vintage |
| `purge.py` | Delete the data root and verify; refuses any folder not named `sharadar-data` |
| `cli.py` | `status`, `download`, `bulk-status`, `build`, `crosscheck`, `set-end`, `purge` |

## Data Flow

### Prices

- Raw close is `closeunadj`. Raw open, high, low are the split-adjusted value times `closeunadj / close`.
  Raw volume is `volume × close / closeunadj`.
- Split and total-return adjustment is done at read time with the existing `adjust()`, from the converted events,
  exactly as for the FMP vintage. `closeadj` is kept only as a check on our own adjustment.

### Events

| Sharadar action | Becomes |
|---|---|
| `split` | split row: ratio = `value`, on the ex-date (confirmed against FMP in the dry run) |
| `dividend` | dividend row: ex-date, amount converted from split-adjusted to raw basis with later splits |
| `spinoffdividend` | dividend-like row, so total return covers spinoffs |
| `delisted`, `regulatorydelisting`, `voluntarydelisting`, `bankruptcyliquidation`, `acquisitionby`, `mergerto` | delisting row: last trade date and reason |
| `acquisitioncash`, `acquisitioncashusd`, `acquisitionstock` | consideration per share, used for the delisting return; any other `acquisition…` type sets `other_consideration` |
| `tickerchangefrom`, `tickerchangeto` | symbol history for `resolve` |

### Identity

1. `security_id` is Sharadar's `permaticker`. It is never replaced by an FMP id, so the two id spaces cannot be
   confused; `fmp_security_id` is an extra, nullable column.
2. CIK comes from the `secfilings` URL.
3. Candidates are vintage stocks and ADRs with the same CIK whose price range overlaps.
4. Assignment is one-to-one and greedy: same symbol first, then the longest overlap. Without the symbol rule
   GOOGL was matched to FMP's GOOG (found in the dry run).
5. `symbol` is the ticker as traded: the number Sharadar appends to a reused delisted ticker is removed
   (`sharadar_ticker` keeps the original), so `is_common_stock` accepts delisted names.
6. Share classes stay separate (`permaticker` is per class); portfolios dedupe by CIK as in the 2026-11 screens.

### Fundamentals

- Only ARQ, ART, ARY are loaded. `available_from = date + 1 trading day`.
- MR dimensions are not loaded at all, so a restated value cannot leak into a backtest.
- ADR and Canadian filers have annual data only; quarterly screens run on domestic filers.

## Design Decisions

- **Separate data root, never merged into the vintage.** The licence requires deletion; a merged vintage could not
  be cleaned.
- **Bulk files, not paged queries.** A handful of requests for the whole history.
- **Raw basis as the stored truth.** Same rule as the FMP pipeline, so `adjust()` and all research code are shared.
- **Same reader shape as `FmpStore`.** Existing screens (P24 helpers, event bench) run with a store swap.
- **Build and test before buying.** The free sample serves the full `tickers` and `descriptions` tables and about
  five years of all tables for sample tickers, which is enough for unit tests and an end-to-end dry run.
- **Research queue fixed before purchase.** With about 60 days of access, deciding what to test after the clock
  starts wastes the window. See [Tasks](Tasks.md).
- **Results are aggregates.** Per-period, per-bucket statistics and model files. No per-ticker-per-day tables in
  kept outputs.

## Integration Patterns

- Read-only use of `src.data.fmp_pipeline` (vintage securities, `adjust()`).
- No DataManager hook, no scheduler job, nothing on the live path. This is research-only.
- Errors: a failed download resumes from the manifest; a failed cross-check stops the build before research starts.
