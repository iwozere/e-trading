# Design

## Purpose

A kept, survivorship-free daily price layer for US common stocks. The FMP vintage has almost no stock that
stopped trading before 2021, so every backtest on it so far is an upper bound. Sharadar fills the gap too, but its
licence forces deleting the data 30 days after the subscription ends; EODHD lets a private user keep it.

## Architecture

```
universe  ->  download  ->  build  ->  store  ->  research (P24 helpers, event bench)
   |             |            |
 2 calls     3 calls/stock   FMP vintage (mapping, cross-check)
```

| File | Role |
|---|---|
| `config.py` | Constants, data root, key |
| `client.py` | `EodhdClient`: symbol lists, bars, splits, dividends, `/user`; pacing, retries, error classes |
| `universe.py` | Filter the symbol lists; `IdRegistry` (code -> id) |
| `download.py` | Resumable per-security download with budget, quota and plan checks |
| `convert.py` | Pure conversions to the raw basis |
| `mapping.py` | EODHD -> FMP vintage ids; `name_similarity` |
| `build.py` | Parquet store and `build_info.json` |
| `store.py` | `EodhdStore` reader (`FmpStore` shape) |
| `crosscheck.py` | Adjustment and vendor checks |
| `cli.py` | `quota | universe | download | build | crosscheck | status` |

### Data root

```
<DATA_CACHE_DIR>/eodhd-data/
  ids.json                     code -> security id, only ever added to
  universe.parquet             the codes to download
  raw/symbols/                 both symbol lists as delivered, by date
  raw/history/NN/<id>.json.gz  bars, splits and dividends of one security as delivered
  raw/progress.jsonl           one line per finished attempt
  store/                       built from raw/ (replaced by every build)
  reports/                     cross-check output
```

Raw files are named by id, not code: `CON`, `PRN`, `AUX`, `NUL` are real tickers and reserved file names on Windows.

## EODHD facts the conversion relies on

From EODHD's documentation, checked on 2026-11-22. Volume and splits are verified against the FMP vintage by
`crosscheck` on purchase day.

- `/eod`: `open`, `high`, `low`, `close` are as traded; `volume` is split-adjusted; `adjusted_close` is adjusted for
  splits and dividends. On the free key, closes and volumes matched the FMP vintage on all four pairs tried (one year,
  no splits in it).
- `/splits`: `split` is `"new/old"`, for example `"2.000000/1.000000"`, dated on the ex-date. Ratios need not be
  integers (`"104.000000/100.000000"`).
- `/div`: `date` is the ex-date; `value` is split-adjusted to today's shares; `unadjustedValue` is what was paid in
  the shares of the day; `currency` is given.
- A delisted symbol whose ticker was reused is renamed `ABC_old`, `ABC_old1`, … (1,768 such delisted common stocks).
- The free plan answers a whole-history request with one year of bars and **no** warning row (a request with a
  `from` date before that year gets a warning row). The download therefore checks `/user` for the plan first.

## Data Flow

1. `universe`: both lists -> keep `Type` = Common Stock on NYSE, NASDAQ, NYSE MKT, AMEX, BATS, NYSE ARCA ->
   one row per code (a code on both lists counts as live) -> ids from `IdRegistry`.
2. `download`: delisted codes first, then live. Per code: `/eod` (whole history), then `/splits` and `/div` unless
   the code is unknown (404 -> `missing`, one call). Work goes to a thread pool in batches; before each batch the
   remaining budget (smaller of `--max-calls` and calls left today minus a reserve) decides how many codes fit at
   3 calls each.
3. `build`: per security, splits -> raw volume (`volume / product of later split ratios`) -> raw-basis dividends
   (`unadjustedValue`, else `value` times the later split ratios) -> bucket files by `security_id % 64`.
4. `store`: reads like `FmpStore`; `adjust()` turns raw bars plus events into split- or total-return-adjusted bars.

## Design Decisions

- **Own ids, not FMP's.** EODHD has no permanent id and no CIK. Ids start at 1,000,001 so they never collide with
  vintage ids (about 20,000). `fmp_security_id` is an extra column.
- **Mapping by symbol, dates and name.** A candidate pair shares the symbol and overlaps in time; it is accepted if
  the names agree (token overlap 0.5 or more), or if it is the only candidate and overlaps a year or more. Greedy and
  one to one. A reused ticker fails the date overlap.
- **Raw basis everywhere.** As in the FMP vintage and the Sharadar store: adjustment happens when the data is read,
  so a missing split can be fixed without a rebuild of anything but this store.
- **Keep the payloads as delivered.** `raw/` is the source of truth; conversion rules can change and the store is
  rebuilt without spending calls.
- **Common stock on the main exchanges only by default.** OTC venues add about 16,000 delisted codes with poor data
  and no research use yet. `universe --all-exchanges` and `--types` widen it; ids of existing codes do not change.
- **Stop, don't fail.** Budget, quota and plan problems end the run with what is finished kept; failed codes are
  retried next run. Exit code 3 means "run again", 2 means "this key cannot do it".
- **No quality flags yet.** `flags` is 0. P24's `prepare_daily` already marks price jumps without a split;
  `crosscheck` shows where EODHD and FMP disagree.

## Integration Patterns

- Readers depend only on the store shape (`dir`, `bulk_daily`, `events_many`), shared with `FmpStore` and
  `SharadarStore`.
- The Sharadar `eodhd-check` imports `EodhdClient` and `name_similarity` from here.
