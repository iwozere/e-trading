# P18 Institutional Flow Tracker — Implementation Plan

## Revised Decisions (vs. initial ideas in input.md)

| Topic | Original idea | Revised decision |
|-------|--------------|-----------------|
| EDGAR downloader | New `edgar_13f_downloader.py` | Extend existing `src/data/downloader/edgar_downloader.py` |
| OpenFIGI mapper | New file in pipeline folder | `src/data/downloader/openfigi_mapper.py`, shared downloader |
| Holdings storage | DuckDB | CSV.gz files (same as AAII, CBOE, etc.) |
| Execution cadence | Quarterly | **Daily** — with layered signal freshness (see below) |

---

## Architecture: Layered Signal Freshness

The key insight: 13F data is quarterly, but the pipeline should run **daily** by combining
three data streams with different update frequencies.

```
Layer 1 — QUARTERLY (13F baseline, updated as filings arrive)
  SEC EDGAR: new 13F-HR filings published daily during the 45-day window
  after each quarter-end (Q1 → by May 15, Q2 → by Aug 15, etc.)
  Daily job checks for new filings; delta/consensus updates as they trickle in.

Layer 2 — NEAR-REAL-TIME (Form 4 + Schedule 13D/G, filed within 2 days)
  Form 4:  insider / director buy/sell transactions
  13D/G:   >5% ownership stake changes
  Both are actionable immediately and available daily via EDGAR submissions.

Layer 3 — DAILY (volume anomaly on flagged tickers)
  DataManager (existing) fetches OHLCV for watchlist stocks.
  Spike detection on stocks already flagged by layers 1 and 2.
```

**Result:** the scheduler job runs daily. During filing windows it becomes richer
(more institutions file their 13F each day); outside filing windows it still provides
Form 4, 13D/G and volume-anomaly signals without sitting idle for 2.5 months.

---

## File Structure

```
src/data/downloader/
    edgar_downloader.py          ← EXTEND: add 13F-HR methods
    openfigi_mapper.py           ← NEW: CUSIP→ticker, shared downloader

src/ml/pipeline/p18_institutional_flow_tracker/
    run_p18_scan.py              ← scheduler entry point
    pipeline.py                  ← orchestrator (daily run)
    config.py                    ← thresholds, AUM filter, paths
    processors/
        position_delta_calculator.py   ← Q vs Q-1 holdings delta
        exit_screener.py               ← reduction threshold filter
        consensus_detector.py          ← multi-institution overlap
        volume_anomaly_detector.py     ← daily spike detection (uses DataManager)
        form4_monitor.py               ← Form 4 + 13D/G daily watcher
    scoring/
        composite_scorer.py            ← weighted signal aggregator
    tests/
        test_position_delta.py
        test_exit_screener.py
        test_consensus_detector.py
        test_composite_scorer.py
        test_form4_monitor.py
    docs/
        input.md                 ← ideas (already exists)
        plan.md                  ← this file
        Requirements.md
        Design.md
        Tasks.md
    README.md
```

---

## Cache Layout (CSV.gz)

All files follow project-standard `DATA_CACHE_DIR/<provider>/<key>.csv.gz` pattern.

```
DATA_CACHE_DIR/
    edgar/
        company_tickers.json               ← already present
        companyfacts/                       ← already present
        submissions/                        ← already present
        13f/
            <YYYY>_Q<N>/
                <cik_padded10>.csv.gz       ← per-institution holdings snapshot
            deltas/
                <YYYY>_Q<N>.csv.gz          ← aggregated position deltas for a quarter
            consensus/
                <YYYY>_Q<N>.csv.gz          ← consensus exit signals for a quarter
            form4/
                <YYYY>-<MM>-<DD>.csv.gz     ← Form 4 filings for a day
            13dg/
                <YYYY>-<MM>-<DD>.csv.gz     ← 13D/G amendments for a day
    openfigi/
        cusip_map.csv.gz                    ← accumulated CUSIP→ticker mapping
```

Holdings CSV.gz columns: `cik`, `institution_name`, `quarter`, `ticker`, `cusip`,
`shares`, `value_usd`, `pct_of_portfolio`

Delta CSV.gz columns: `cik`, `institution_name`, `ticker`, `shares_prev`, `shares_curr`,
`delta_pct`, `exit_type`, `value_usd_prev`, `pct_of_portfolio_prev`

---

## Component Specifications

### 1. `edgar_downloader.py` — New 13F Methods

Add to the existing `EdgarDownloader` class:

```python
def download_13f_index(self, year: int, quarter: int, force: bool = False) -> pd.DataFrame:
    """
    Fetch the EDGAR full-text search index for 13F-HR filings in a given quarter.
    Endpoint: https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=13F-HR&dateb=&owner=include&count=40&search_text=
    Or better: use the EDGAR full-text search API:
    https://efts.sec.gov/LATEST/search-index?q=%2213F-HR%22&dateRange=custom&startdt=YYYY-MM-DD&enddt=YYYY-MM-DD&forms=13F-HR
    Returns DataFrame: cik, institution_name, accession_number, filed_date, quarter
    Cache: edgar/13f/<YYYY>_Q<N>/index.csv.gz
    """

def download_13f_infotable(self, cik: int, accession_number: str,
                            year: int, quarter: int, force: bool = False) -> Optional[Path]:
    """
    Download and cache the infotable XML for one 13F-HR filing.
    URL pattern: https://www.sec.gov/Archives/edgar/data/<cik>/<accession_no_nodash>/infotable.xml
    Cache: edgar/13f/<YYYY>_Q<N>/<cik_padded10>.csv.gz
    """

def parse_13f_infotable(self, xml_path: Path, cik: int, institution_name: str,
                         quarter: str) -> pd.DataFrame:
    """
    Parse infotable XML into a holdings DataFrame.
    Columns: cusip, nameOfIssuer, shares, value_usd, investmentDiscretion, cik,
             institution_name, quarter
    """

def load_13f_holdings(self, cik: int, year: int, quarter: int,
                       force_refresh: bool = False) -> Optional[pd.DataFrame]:
    """
    Load cached 13F holdings for a CIK+quarter from CSV.gz. Downloads if absent.
    """

def get_new_13f_filings_today(self, as_of_date: Optional[date] = None) -> pd.DataFrame:
    """
    Check EDGAR for 13F-HR filings submitted on a given date (default: today).
    Used by the daily scheduler job to detect new filings incrementally.
    Returns DataFrame: cik, institution_name, accession_number, filed_date
    """

def download_form4_filings(self, as_of_date: Optional[date] = None,
                            force: bool = False) -> pd.DataFrame:
    """
    Download Form 4 filings for a given date from EDGAR full-text search.
    Cache: edgar/13f/form4/<YYYY>-<MM>-<DD>.csv.gz
    Columns: cik, ticker, insider_name, transaction_type, shares, price, filed_date
    """

def download_13dg_filings(self, as_of_date: Optional[date] = None,
                           force: bool = False) -> pd.DataFrame:
    """
    Download Schedule 13D/G amendments filed on a given date.
    Cache: edgar/13f/13dg/<YYYY>-<MM>-<DD>.csv.gz
    Columns: cik, ticker, filer_name, ownership_pct, amendment_type, filed_date
    """
```

### 2. `openfigi_mapper.py` — New Shared Downloader

`src/data/downloader/openfigi_mapper.py`

```python
class OpenFigiMapper(BaseDataDownloader):
    """
    Maps CUSIPs to tickers via the OpenFIGI API v3.
    Maintains a persistent CSV.gz cache of all resolved mappings.

    Cache: DATA_CACHE_DIR/openfigi/cusip_map.csv.gz
    Columns: cusip, ticker, name, exchange_code, security_type, resolved_at

    API: https://api.openfigi.com/v3/mapping (free, rate-limited to 25 req/min
         without key; 250 req/min with free key)
    Batch size: up to 100 CUSIPs per request.
    """

    def get_provider_name(self) -> str:
        return "openfigi"

    def get_supported_intervals(self) -> List[str]:
        return []

    def get_ohlcv(self, ...) -> pd.DataFrame:
        # Not supported — returns empty DataFrame
        ...

    def map_cusips(self, cusips: List[str], force_refresh: bool = False) -> Dict[str, Optional[str]]:
        """
        Resolve CUSIPs to tickers.  Only fetches unknowns (not in cache).
        Returns {cusip: ticker_or_None}.
        """

    def load_cache(self) -> pd.DataFrame:
        """Load the full CSV.gz mapping table."""

    def _fetch_batch(self, cusips: List[str]) -> List[Dict]:
        """POST up to 100 CUSIPs to OpenFIGI, return raw response."""
```

Key details:
- API key from `BaseDataDownloader._get_config_value("OPENFIGI_API_KEY")`
- Cache is **append-only**: new CUSIPs are appended, existing ones are not re-fetched unless `force_refresh=True`
- Non-equity CUSIPs (bonds, ETFs with no ticker) stored as `ticker=None` to avoid re-fetching

### 3. Processors

#### `position_delta_calculator.py`
- `calculate(year: int, quarter: int) -> pd.DataFrame`
- Loads quarter `N` and `N-1` snapshots from CSV.gz cache
- Merges on `(cik, ticker)`, computes `delta_pct = (shares_curr - shares_prev) / shares_prev`
- `exit_type`: `full_exit` | `partial_exit` | `new_position` | `unchanged`
- Saves result to `edgar/13f/deltas/<YYYY>_Q<N>.csv.gz`

#### `exit_screener.py`
- `screen(delta_df: pd.DataFrame, threshold: float = 0.30, min_position_pct: float = 0.005) -> pd.DataFrame`
- Filters: `delta_pct <= -threshold` AND `pct_of_portfolio_prev >= min_position_pct`
- Ensures noise from tiny positions is excluded

#### `consensus_detector.py`
- `detect(exits_df: pd.DataFrame, min_institutions: int = 3) -> pd.DataFrame`
- Groups by `ticker`, counts distinct `cik` values
- Output columns: `ticker`, `institution_count`, `total_value_sold_usd`, `avg_exit_pct`
- Saves to `edgar/13f/consensus/<YYYY>_Q<N>.csv.gz`

#### `volume_anomaly_detector.py`
- `detect(tickers: List[str], lookback_days: int = 20, spike_multiplier: float = 3.5) -> pd.DataFrame`
- Uses `DataManager` (existing) to fetch OHLCV
- Computes rolling average volume, flags last 5 days
- Output columns: `ticker`, `volume_spike_ratio`, `price_change_5d_pct`, `above_spike_days`

#### `form4_monitor.py`
- `get_significant_sells(as_of_date: date, min_value_usd: float = 1_000_000) -> pd.DataFrame`
- Reads from `edgar_downloader.download_form4_filings()`
- Filters for `transaction_type in ('S', 'S-')` (open market sales)
- Enriches with ticker via `openfigi_mapper` or `edgar_downloader.load_company_tickers()`

### 4. `composite_scorer.py`

Signal weights (configurable in `config.py`):

| Signal | Points |
|--------|--------|
| 3+ institution consensus exit | 40 |
| Large single exit (>$500M value sold) | 25 |
| Volume spike confirmed (last 5 days) | 20 |
| Form 4 insider sell this week | 10 |
| 13D/G ownership stake drop | 10 |
| Seasonal redemption window (Oct–Dec, Mar, Sep) | 5 |
| Price down >15% from 52-week high | 5 |
| Max possible | 115 |

- Alert threshold: `total_score >= 60`
- Output columns: `ticker`, `total_score`, `signals_active`, `signal_detail` (JSON string)

### 5. `pipeline.py` — Daily Orchestrator

```python
class InstitutionalFlowPipeline:
    def run(self, user_id: int, as_of_date: Optional[date] = None,
            force_refresh: bool = False) -> dict:
        """
        Daily run logic:
        1. Check for new 13F-HR filings today → parse + cache new ones
        2. If new filings → recalculate position deltas + consensus for current quarter
        3. Download Form 4 filings for today
        4. Download 13D/G amendments for today
        5. Load latest consensus watchlist (current quarter)
        6. Run volume anomaly detection on watchlist tickers
        7. Compute composite scores
        8. Return result dict with high_score_count for notification_rules
        """
```

### 6. `run_p18_scan.py` — Entry Point

Follows `p10_emps3/run_emps3_scan.py` pattern exactly:
- `argparse`: `--user-id`, `--as-of-date` (ISO format, default today), `--force-refresh`
- Calls `InstitutionalFlowPipeline().run(...)`
- Prints `__SCHEDULER_RESULT__: <json>` to stdout
- Result keys for `notification_rules`: `high_score_count`, `new_13f_filings_today`,
  `form4_sells_count`, `top_ticker`, `top_score`

---

## Scheduler Registration

```sql
-- Daily at 07:00 UTC (after US pre-market, EDGAR updates available)
INSERT INTO job_schedules (user_id, name, job_type, cron, task_params, enabled)
VALUES (
  1, 'P18 Institutional Flow Daily', 'data_processing',
  '0 7 * * *',
  '{
    "script_path": "src/ml/pipeline/p18_institutional_flow_tracker/run_p18_scan.py",
    "script_args": [],
    "notification_rules": {
      "conditions": [{
        "check_field": "high_score_count",
        "operator": ">=",
        "threshold": 1,
        "channels": ["telegram"]
      }]
    }
  }',
  true
);
```

---

## Build Order

```
Phase 1 — Data Layer (Week 1)
  ├── edgar_downloader.py: add 13F index + infotable methods
  ├── edgar_downloader.py: add Form 4 + 13D/G methods
  ├── openfigi_mapper.py: CUSIP→ticker with CSV.gz cache
  └── Tests for all three

Phase 2 — Signal Generation (Week 2)
  ├── position_delta_calculator.py
  ├── exit_screener.py
  ├── consensus_detector.py
  ├── volume_anomaly_detector.py
  └── Tests for all four

Phase 3 — Scoring & Daily Pipeline (Week 3)
  ├── form4_monitor.py
  ├── composite_scorer.py
  ├── pipeline.py (daily orchestrator)
  └── run_p18_scan.py (entry point)

Phase 4 — Integration & Docs (Week 4)
  ├── Scheduler registration (SQL)
  ├── Telegram message template
  ├── End-to-end test run (backfill last 2 quarters)
  └── README.md, Requirements.md, Design.md, Tasks.md

Phase 5 — Advanced (Deferred)
  ├── FINRA ATS dark pool data (src/data/downloader/finra_data_downloader.py may help)
  ├── Options flow / put volume anomaly
  └── IPO calendar scraper + underwriter correlation
```

---

## Key Integration Points

| This pipeline | Uses | Via |
|--------------|------|-----|
| Volume anomaly | `DataManager` | existing facade |
| Ticker data | `DataManager` | existing facade |
| EDGAR data | `EdgarDownloader` (extended) | `src/data/downloader/edgar_downloader.py` |
| CUSIP mapping | `OpenFigiMapper` | `src/data/downloader/openfigi_mapper.py` |
| Logging | `setup_logger(__name__)` | all modules |
| Scheduling | `SchedulerService` `DATA_PROCESSING` job | daily cron |
| Notifications | `notification_rules` in task_params | existing notification engine |

---

## Open Questions

1. Minimum AUM threshold for institutions to track (>$1B recommended to reduce noise)?
2. How to handle amended 13F filings (13F-HR/A) — replace or flag separately?
3. OPENFIGI_API_KEY — free key sufficient (250 req/min) or do we need paid?
4. Should Form 4 signals stand alone (alert without 13F confirmation) or only reinforce 13F exits?
5. How to weight passive index funds whose sells are mechanical, not informational?
   Possible filter: exclude institutions whose holdings mirror a known index (e.g., SPY overlap > 90%).
