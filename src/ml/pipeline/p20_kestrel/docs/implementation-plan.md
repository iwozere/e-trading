# P20 Kestrel — Implementation Plan v2

**Date:** 2026-07-02  
**Based on:** `pipeline-specification.md` v1.2 (all 14 v1.1 discrepancies resolved)  
**Supersedes:** implementation-plan.md v1

---

## 1. Remaining Discrepancies in v1.2

One structural issue was found during the codebase audit. All other v1.1 issues are resolved.

**D15. `AsyncTrendsAdapter` uses unofficial Google Trends scrape, not pytrends**  
`src/common/sentiments/adapters/async_trends.py` rolls its own cookie + token fetch against the Trends UI (not the `pytrends` library). The spec names pytrends (§7.3). Both are unofficial scrapers of the same Google UI. The existing adapter already implements the 429 backoff and cookie handling. **Plan: reuse `AsyncTrendsAdapter` and add anchor-normalization on top — do not introduce a second Trends scraper via pytrends.**

---

## 2. What Already Exists — Reuse Map

This section maps every spec responsibility to either an existing component to reuse, or a net-new build. **Nothing in this list should be re-implemented from scratch.**

### 2.1 Data access & OHLCV

| Spec need | Existing component | Notes |
|---|---|---|
| OHLCV retrieval for all tickers | `src.data.data_manager.DataManager.get_ohlcv_batch()` | Already caches to `DATA_CACHE_DIR/ohlcv/`; P20 reads, never re-downloads |
| Fundamentals (mcap, sector, revenue growth) | `src.common.fundamentals.get_fundamentals_unified()` → `DataManager.get_fundamentals()` | Multi-provider merge already done |
| 50DMA / 200DMA / RSI / etc. | `src.common.technicals.calculate_technicals_talib()` | TALib wrappers ready; called per ticker on the OHLCV DataFrame |
| FINRA TRF (short-vol ratio) | `src.data.downloader.finra_trf_downloader.FinraTRFDownloader` | P15 already downloads daily; P20 reads cache only |
| Fear & Greed, CBOE put/call | P15 file cache (`DATA_CACHE_DIR/fear_greed/`, `DATA_CACHE_DIR/cboe/`) | P20 reads directly from files |

### 2.2 EDGAR & filings

| Spec need | Existing component | Notes |
|---|---|---|
| 8-K discovery | P15 cache at `DATA_CACHE_DIR/edgar/8k/index/YYYY-MM-DD.csv.gz` | Columns: `cik, company, accession_number, items, description, filed_date, primary_document`; P20 reads only |
| Form 4 insider transactions | P18 cache at `DATA_CACHE_DIR/edgar/13f/form4/YYYY-MM-DD.csv.gz` | Columns: `ticker, issuer_cik, insider_name, transaction_code, shares, price_per_share, total_value_usd, filed_date`; P20 reads only. **Caveat:** current downloader retains only sell codes (`S`, `S-`); see §7 for required change to support buys |
| 13D/G activist filings | P18 cache at `DATA_CACHE_DIR/edgar/13f/13dg/YYYY-MM-DD.csv.gz` | Columns: `cik, entity_name, accession_number, filed_date, form_type`; metadata only — target company ticker not present; body fetch needed to identify subject issuer |
| 8-K body text (for LLM) | `EdgarDownloader` — new `fetch_8k_body()` method | Index has `primary_document` filename; body URL = `{EDGAR_ARCHIVES_BASE}/{cik}/{acc_norm}/{primary_document}`; private `_fetch_filing_xml()` retry/rate-limit logic is reused internally |
| 13D/G body text (for issuer extraction) | Same `fetch_8k_body()` method | 13D/G filings are HTML/XML; same fetch pattern |
| CIK resolution | `EdgarDownloader.resolve_tickers_to_ciks()` | Ready |
| Company names for alias table | `EdgarDownloader` submissions cache (`DATA_CACHE_DIR/edgar/submissions/`) | Company names in existing CIK JSON |

### 2.3 Social sentiment adapters

All four adapters already exist in `src/common/sentiments/adapters/` with async I/O, rate limiting, and retry logic built in. **P20 `social_poll` is a thin orchestration wrapper — no new adapters.**

| Spec source | Existing adapter | Key method to call |
|---|---|---|
| StockTwits | `AsyncStocktwitsAdapter` | `fetch_summary(ticker)` → message count + bull/bear ratio |
| Reddit | `AsyncRedditAdapter` | `fetch_summary(ticker)` → mention count (OAuth managed internally) |
| ApeWisdom | `AsyncApeWisdomAdapter` | `fetch_summary(ticker)` or bulk page fetch → top-100 mentions |
| Google Trends (anchor-normalized) | `AsyncTrendsAdapter` | `fetch_summary(ticker)` → raw interest score; P20 adds anchor rescaling on top |

Circuit breaker & health monitoring: `src.common.sentiments.adapters.adapter_manager.AdapterManager` and `CircuitBreaker` — already implemented; `social_poll` instantiates `AdapterManager` and calls through it.

### 2.4 Alpha Vantage budget

| Spec need | Existing component | Notes |
|---|---|---|
| AV NEWS_SENTIMENT fetch | `src.data.downloader.alpha_vantage_data_downloader.AlphaVantageDataDownloader` | Has `get_news_sentiment(ticker)` or equivalent; verify method name |
| Finnhub earnings calendar | `src.data.downloader.finnhub_data_downloader.FinnhubDataDownloader` | Free endpoint; use for `catalyst_sync` |

### 2.5 Database layer

| Spec need | Existing component | Notes |
|---|---|---|
| DB session management | `src.data.db.core.database.session_scope()` | Context manager; use directly in P20 jobs |
| Service base pattern | `src.data.db.services.base_service.BaseDBService` | P20 services inherit this for UoW and error handling |
| Job/schedule registration | `src.data.db.models.model_jobs.Schedule` + `job_schedules` table | P20 jobs inserted as `job_type='script'` rows with `task_params.script_path` |
| Schema migrations | Alembic at `src/data/db/migrations/` | P20 adds one new migration file; touches only `k20_*` tables |

### 2.6 Notifications

| Spec need | Existing component | Notes |
|---|---|---|
| Push alerts + email | `src.notification.service.client` (`NotificationServiceClient`) | All channels handled; P20 constructs message, calls client |
| Logger | `src.notification.logger.setup_logger(__name__)` | Project-wide standard |

### 2.7 Scheduler execution model

Jobs are registered in `job_schedules` with `job_type='script'` and `task_params.script_path` pointing to a Python script inside `src/ml/pipeline/` (which is on the allowlist in `scheduler_service.py:808`). The scheduler runs them as subprocesses via `_execute_data_processing_job`. Each P20 job script must have a `main()` and `if __name__ == '__main__': main()` entry point, following the p15_daily.py pattern. They print `__SCHEDULER_RESULT__:{json}` on success.

---

## 3. Net-New Build — What P20 Actually Needs to Create

Everything not in §2 is new code in `src/ml/pipeline/p20_kestrel/`.

### 3.1 New module structure

```
src/ml/pipeline/p20_kestrel/
├── __init__.py
├── config.py                     # DATA_CACHE_DIR path, LLM model names, budget cap, flag REVISIONS_FEED_AVAILABLE
├── db/
│   ├── __init__.py
│   ├── models.py                 # k20_* SQLAlchemy ORM models
│   └── repos.py                  # thin repository layer over session_scope()
├── ingest/
│   ├── __init__.py
│   ├── universe_loader.py        # Phase 1: Nasdaq CSV + fundamentals → k20_universe
│   ├── eod_ingest.py             # Phase 1: DataManager OHLCV + technicals → k20_signals
│   ├── filings_ingest.py         # Phase 4: P15 8-K index → body fetch → Form4/13D parse
│   └── calendar_sync.py          # Phase 6: Finnhub earnings + PDUFA scrape → k20_catalysts
├── sentiment/
│   ├── __init__.py
│   ├── alias_builder.py          # Phase 3: company_aliases from fundamentals + EDGAR names
│   ├── gdelt_processor.py        # Phase 3: GKG files → alias match → k20_sentiment_daily
│   ├── social_poll.py            # Phase 7: orchestrates existing adapters → k20_sentiment_daily
│   ├── trends_poll.py            # Phase 7: AsyncTrendsAdapter + anchor normalization
│   ├── av_budgeted.py            # Phase 7: AV downloader + k20_request_budget gate
│   └── sentiment_aggregator.py   # Phase 7: §7.6 crowding formula; NOT the common aggregator
├── screening/
│   ├── __init__.py
│   ├── sleeve_a.py               # Phase 4: hard filters + §4.2.1 scoring
│   ├── sleeve_b.py               # Phase 6: B1/B2/B3 logic, idempotent T-minus
│   └── sleeve_c.py               # Phase 1: RS rank + regime filter; crowding overlay in Phase 7
├── llm/
│   ├── __init__.py
│   ├── client.py                 # Anthropic SDK wrapper: budget check, call, cache, store
│   ├── classifier_8k.py          # Phase 5: 8-K/PR classification prompt + output parser
│   ├── dossier.py                # Phase 5: candidate dossier prompt + §8.2 parser
│   ├── risk_diff.py              # Phase 5: 10-K/Q risk-factor diff
│   └── prompts.py                # all prompt templates as module-level string constants
├── risk/
│   ├── __init__.py
│   └── risk_checker.py           # Phase 2: stops/targets/LLM invalidations + push alerts
├── reporting/
│   ├── __init__.py
│   ├── daily_digest.py           # Phase 1: build + send digest (Telegram + email)
│   ├── weekly_report.py          # Phase 8: perf + calibration + spend
│   └── data_health.py            # Phase 1: freshness guard, budget usage
├── pos/
│   ├── __init__.py
│   └── pos_commands.py           # Phase 2: /pos command grammar parser → k20_positions
│                                 #   + YAML positions.yml ingester
├── jobs/
│   ├── __init__.py
│   └── register_jobs.py          # one-time script: INSERT job_schedules rows for all P20 jobs
│                                 #   idempotent (ON CONFLICT DO NOTHING)
├── tests/
│   ├── __init__.py
│   ├── test_db_models.py
│   ├── test_alias_builder.py
│   ├── test_gdelt_processor.py
│   ├── test_sleeve_a.py
│   ├── test_sleeve_c.py
│   ├── test_scoring.py
│   ├── test_llm_client.py
│   ├── test_classifier_8k.py
│   ├── test_dossier.py
│   ├── test_daily_digest.py
│   └── test_data_health.py
├── README.md
└── docs/
    ├── pipeline-specification.md
    ├── implementation-plan.md    (this file)
    ├── Requirements.md
    ├── Design.md
    └── Tasks.md

# Standalone job scripts (each has main() + __SCHEDULER_RESULT__ output):
src/ml/pipeline/p20_kestrel/
├── run_universe_refresh.py       # Phase 1
├── run_ingest_eod.py             # Phase 1
├── run_momentum_rank.py          # Phase 1
├── run_data_health.py            # Phase 1
├── run_digest_send.py            # Phase 1
├── run_risk_check.py             # Phase 2
├── run_gdelt_backfill.py         # Phase 3 (one-time)
├── run_gdelt_process.py          # Phase 3
├── run_alias_refresh.py          # Phase 3
├── run_ingest_filings.py         # Phase 4
├── run_screen_turnaround.py      # Phase 4
├── run_llm_classify_filings.py   # Phase 5
├── run_llm_dossiers.py           # Phase 5
├── run_catalyst_sync.py          # Phase 6
├── run_screen_spinoffs.py        # Phase 6
├── run_social_poll.py            # Phase 7
├── run_av_sentiment.py           # Phase 7
├── run_trends_watchlist.py       # Phase 7
├── run_sentiment_aggregate.py    # Phase 7
└── run_weekly_report.py          # Phase 8
```

### 3.2 Database migration

New file: `src/data/db/migrations/versions/002_kestrel_schema.py`  
Contains only `k20_*` tables; does not touch any existing table.

---

## 4. Phased Build Plan

### Phase 1 — Schema + EOD Ingest + Momentum + Digest

#### `db/models.py`

Define all `k20_*` tables as SQLAlchemy mapped classes. Notable additions vs. spec §3.1 already incorporated in v1.2:

| Table | Implementation notes |
|---|---|
| `k20_universe` | `status` as a `CheckConstraint`; `updated_at` server default |
| `k20_company_aliases` | Composite PK `(ticker, alias)`; `normalized_alias` computed column (lowercase, suffix-stripped) — stored not virtual for query performance |
| `k20_alias_blocklist` | `match_policy` enum as `CheckConstraint` |
| `k20_signals` | Composite PK `(ticker, date, signal_type)`; index on `(ticker, date)` for range scans |
| `k20_sentiment_daily` | `UNIQUE(ticker, date, source)`; `top_domains` as `JSONB` |
| `k20_catalysts` | `t10_alerted_at`, `t3_alerted_at`, `datechange_alerted_at` nullable timestamps |
| `k20_watchlist` | FK `dossier_run_id → k20_llm_runs.id`; `state` CheckConstraint |
| `k20_positions` | Standalone table; no FK to `trading_positions` |
| `k20_llm_runs` | `UNIQUE(task_type, input_ref)`; `cost_usd` for budget tracking |
| `k20_request_budget` | Composite PK `(source, date)` |
| `k20_job_runs` | Composite PK `(job, run_date)`; used by DAG (§11.1) |
| `k20_alerts_log` | `ts` with index; no PK (append-only log) |

`db/repos.py` — thin wrappers around `session_scope()` for each table; no SQLAlchemy `Session` leaked outside.

#### `ingest/universe_loader.py`

```
run_universe_refresh.py → universe_loader.run()
  1. Read Nasdaq CSV from config (DATA_CACHE_DIR or PROJECT_ROOT config path)
  2. For each ticker: DataManager.get_fundamentals(ticker) → mcap, sector, industry
  3. Upsert into k20_universe (ON CONFLICT(ticker) DO UPDATE SET ...)
  4. Mark tickers absent from CSV as status='delisted' (soft delete)
  Returns: {tickers_upserted, tickers_delisted}
```

#### `ingest/eod_ingest.py`

```
run_ingest_eod.py → eod_ingest.run(as_of_date)
  1. Load k20_universe tickers
  2. DataManager.get_ohlcv_batch(tickers, '1d', start=2y_ago, end=as_of_date)
     — reads from cache; no new downloads
  3. For each ticker, call calculate_technicals_talib(df):
     - sma_50, sma_200 → price_vs_50dma, price_vs_200dma flags
     - 2y high from df.high.rolling(504).max()
     - drawdown_from_2y_high = (close - 2y_high) / 2y_high
     - adv_20d = df.volume.rolling(20).mean() × df.close
     - 3m_return = close.pct_change(63)
     - 6m_return = close.pct_change(126)
  4. Read TRF from P15 cache per ticker → short_volume_ratio
  5. Upsert signal rows into k20_signals (skip already-present dates)
  6. Write k20_job_runs(job='ingest_eod', run_date=today, status='ok')
```

#### `screening/sleeve_c.py`

```
run_momentum_rank.py → sleeve_c.run()
  Reads k20_signals; computes RS-score percentile rank across universe.
  Regime filter: SPY sma_200 flag from k20_signals.
  Breadth: count(tickers WHERE price_vs_200dma > 0) / total universe.
  Phase 1: crowding_overlay = None (placeholder until Phase 7).
  Writes/updates k20_watchlist rows for sleeve='C'.
  Technical eligibility: price > sma_50 > sma_200, adv_20d ≥ $20M,
    positive revenue_growth from k20_universe fundamentals.
```

#### `reporting/data_health.py`

```
run_data_health.py → data_health.run()
  Check freshness of: k20_signals (last date per source), P15 GKG
  watermark, k20_sentiment_daily watermarks, k20_request_budget usage.
  Flag any source where max(date) < today - 2 days.
  Write flagged rows to k20_alerts_log(trigger='data_stale').
  LLM budget: sum(cost_usd) from k20_llm_runs this calendar month;
    emit warning if > 80% of LLM_MONTHLY_BUDGET_USD.
```

#### `reporting/daily_digest.py`

```
run_digest_send.py → daily_digest.run()
  Assembles sections from DB:
  1. Regime: SPY sma_200 flag + breadth (k20_signals), Fear&Greed + put/call
     (read from P15 CSV cache files directly).
  2. Positions: k20_positions JOIN k20_signals for current price,
     compute distance-to-stop. Sentiment flag: k20_sentiment_daily.tone_z20 < -3.
     [Phase 1: "no positions tracked" if table empty]
  3. Catalysts (next 5 days): k20_catalysts WHERE event_date BETWEEN today AND today+5.
     [Phase 1: empty]
  4. New candidates: k20_watchlist WHERE state='candidate' ORDER BY score DESC LIMIT 3.
     [Phase 1: only Sleeve C momentum candidates, no LLM verdict yet]
  5. Sentiment anomalies: top-3 mention_z20 spikes from k20_sentiment_daily.
     [Phase 1: "pending Phase 3" if no rows]
  6. Data-health: k20_alerts_log WHERE trigger='data_stale' AND ts > today.
  Renders Telegram Markdown + email HTML.
  Sends via src.notification.service.client.NotificationServiceClient.
```

#### `jobs/register_jobs.py`

Inserts all P20 jobs into `job_schedules` table as `job_type='script'` rows. Idempotent: `INSERT ... ON CONFLICT (user_id, name) DO NOTHING`. Run once at deploy time, not on every boot.

Phase 1 jobs registered in this script:

| job name | cron | script_path |
|---|---|---|
| `k20_universe_refresh` | `30 8 * * 0` | `src/ml/pipeline/p20_kestrel/run_universe_refresh.py` |
| `k20_ingest_eod` | `30 23 * * *` | `src/ml/pipeline/p20_kestrel/run_ingest_eod.py` |
| `k20_momentum_rank` | `50 23 * * *` | `src/ml/pipeline/p20_kestrel/run_momentum_rank.py` |
| `k20_data_health` | `0 7 * * *` | `src/ml/pipeline/p20_kestrel/run_data_health.py` |
| `k20_digest_send` | `30 7 * * *` | `src/ml/pipeline/p20_kestrel/run_digest_send.py` |

---

### Phase 2 — Risk Tracker + `/pos` Commands + Push Alerts

#### `pos/pos_commands.py`

```
parse_pos_add(text) → PositionEntry dataclass
  Grammar per §9.4: /pos add TICKER SLEEVE ENTRY_PX SIZE_PCT [stop=PX] [t1=PX] [t2=PX] [trail=PCT]
  Defaults: stop=entry×0.75, t1=entry×1.35, t2=entry×1.60, trail=20.
  Validates: ticker in k20_universe; sleeve in {A,B,C}; all numerics positive.
  Returns confirmation card text for bot reply (user taps ✅ to write row).
  On confirm: INSERT k20_positions + UPDATE k20_watchlist.state='active_position'.

ingest_positions_yml(path) → int (rows upserted)
  Reads YAML list; each entry maps to k20_positions columns.
  Fallback path for bulk edits without Telegram.
```

Telegram bot wiring (not in scope for Phase 2 itself — the parse function is ready; the bot handler calling it lives in the existing Telegram bot module).

#### `risk/risk_checker.py`

```
run_risk_check.py → risk_checker.run()
  For each open k20_position:
    current_px = latest k20_signals value for ticker
    if current_px <= stop_px → push alert (stop touched)
    if current_px >= t1_px and realized_thirds == 0 → push alert (first scale-out)
    if current_px >= t2_px and realized_thirds == 1 → push alert (second scale-out)
    intraday -12%: (current_px - open_px) / open_px < -0.12 → push alert
    LLM invalidation: k20_llm_runs WHERE ticker=t AND task_type='classify_8k'
      AND verdict→thesis_impact='invalidates' AND materiality='high'
      AND ts > position.entry_date AND ts > last_check → push alert
  Push alerts via NotificationServiceClient with full order-ticket draft.
  Write to k20_alerts_log for each push.
```

---

### Phase 3 — GDELT Pipeline + Alias Table

#### `sentiment/alias_builder.py`

```
run_alias_refresh.py → alias_builder.run()
  Sources:
    (a) get_fundamentals_unified(ticker) → company_name for all k20_universe tickers
    (b) EdgarDownloader: load submissions cache (existing JSON files) → entityType + name
  Normalization function normalize_alias(s):
    lowercase → strip trailing legal suffixes (Inc|Corp|Ltd|LLC|plc|SA|NV|AG|Co)
    via regex → collapse whitespace → strip punctuation
  Build alias_type rows: legal_name (full), short_name (common brand from split),
    former_name (if AKA in EDGAR). Deduplicate.
  Upsert into k20_company_aliases.
  Query k20_alias_blocklist; for aliases in blocklist, apply match_policy.
  Weekly spot-check sample: log 20 random GKG-matched articles to be reviewed.
```

#### `sentiment/gdelt_processor.py`

```
run_gdelt_process.py → gdelt_processor.run(run_date)
  1. Check k20_job_runs(job='gdelt_process', run_date) — skip if already 'ok'.
  2. Find unprocessed GKG dates: dates in DATA_CACHE_DIR/gdelt/gkg/ with
     .gkg.csv.gz files WHERE date NOT IN (
       SELECT DISTINCT date FROM k20_sentiment_daily WHERE source='gdelt').
  3. For each unprocessed date (oldest-first, up to 3 days per run to avoid
     overrunning the 07:00 deadline):
     a. Load GKG file with pandas, select V2Organizations, V2Themes, V2Tone, url, domain.
     b. Filter: rows where V2Organizations matches any alias in k20_company_aliases
        after normalization, applying k20_alias_blocklist policies.
        Also keep rows with finance V2Themes regardless of ticker match (macro).
        Fuzzy pass (≥0.93 similarity) logged to separate audit table.
     c. Tone parse: V2Tone split → tone, pos_score, neg_score, polarity.
     d. Aggregate per (ticker, date): count=mentions, mean=avg_tone, std=tone_std,
        top 5 domains by article count as JSON → top_domains.
     e. Upsert into k20_sentiment_daily(source='gdelt').
     f. Compute rolling 20-day z-scores (min_periods=15) for mention_z20, tone_z20.
        Update existing rows in k20_sentiment_daily.
  4. Write k20_job_runs(job='gdelt_process', run_date, status='ok/failed').

  Backfill variant (run_gdelt_backfill.py):
    Fetches historical GKG from GDELT public archive for 60 days before P15's
    earliest cached file. Calls same processing pipeline. One-time job.
```

**`sentiment_aggregate` DAG dependency (§11.1):** `run_sentiment_aggregate.py` reads `k20_job_runs` for today's `gdelt_process`, `social_poll`, and `av_sentiment_budgeted` rows. If any are `running` it polls every 5 minutes. At 07:00 Europe/Zurich it proceeds regardless, marking missing sources in `components_used` per §7.6.

---

### Phase 4 — Sleeve A Screen + Form 4 / 13D Parsers

#### `ingest/filings_ingest.py`

Three separate data flows — each reads its own pre-existing cache:

**Flow A — Form 4 insider buys (from P18 cache)**
```
run_ingest_filings.py → filings_ingest.run_form4(as_of_date)
  Read DATA_CACHE_DIR/edgar/13f/form4/{as_of_date}.csv.gz directly.
  Filter to tickers in k20_universe; filter transaction_code IN ('P', 'A')
    (open-market purchase and grant codes — buys only; sells are a P18 concern).
  Aggregate per ticker over trailing 90 days (rolling window over cached files):
    insider_buy_count, insider_buy_value_usd.
  Upsert k20_signals(ticker, date, signal_type='insider_buy_90d', value=buy_value_usd).
  Note: if the cache for a date only contains sells (current default), buys will be
    absent. This is resolved by the EdgarDownloader change described in §7.
```

**Flow B — 8-K body fetch + LLM queue (from P15 cache)**
```
run_ingest_filings.py → filings_ingest.run_8k(as_of_date)
  Read DATA_CACHE_DIR/edgar/8k/index/{as_of_date}.csv.gz.
  Filter to tickers in (k20_watchlist ∪ k20_positions).
  For each row:
    If accession_number already in k20_llm_runs.input_ref → skip.
    EdgarDownloader.fetch_8k_body(cik, accession_number, primary_document) → text.
    Cache text at DATA_CACHE_DIR/edgar/8k/bodies/{accession_number_norm}.txt.gz.
    Store reference: INSERT k20_llm_runs(task_type='classify_8k_pending',
      ticker, input_ref=accession_number, output_json=NULL, verdict=NULL).
    classifier_8k.py picks these up in Phase 5.
  Form 10 detection: if items contains '2.01' and description ilike '%form 10%':
    flag as spinoff — store k20_llm_runs(task_type='form10_dossier_pending').
```

**Flow C — 13D/G body fetch → target issuer mapping (from P18 cache)**
```
run_ingest_filings.py → filings_ingest.run_13dg(as_of_date)
  Read DATA_CACHE_DIR/edgar/13f/13dg/{as_of_date}.csv.gz.
  Filter entity_name against curated activist watchlist (config/activists.json).
  For each matching activist row:
    EdgarDownloader.fetch_8k_body(cik, accession_number, primary_document=None)
      → HTML/XML text (method detects extension from filing index).
    Parse subject issuer CIK from body (SGML header <SUBJECT-COMPANY> block or XML).
    Resolve issuer CIK → ticker via submissions cache.
    If ticker in k20_universe:
      INSERT k20_catalysts(event_type='activist_13d'/'passive_13g',
        ticker, catalyst_detail={'activist': entity_name, 'form_type': form_type}).
```

All three flows write `k20_job_runs(job='ingest_filings', run_date)` together at end.

#### `screening/sleeve_a.py`

```
run_screen_turnaround.py → sleeve_a.run()
  Hard filters from k20_signals + k20_universe:
    drawdown_from_2y_high ∈ [−0.75, −0.40]
    mcap ≥ 500M, adv_20d ≥ 10M
    net_debt_ebitda < 3 OR net_cash OR interest_coverage > 4  (from fundamentals)
    revenue_yoy_growth ≥ −0.15  (from fundamentals)
    gross_margin > 0, gross_margin_delta_yoy > −0.05  (from fundamentals)
    sma_50 rising OR ≥2 higher weekly lows  (from k20_signals)
    NOT (price < falling sma_50)

  Scoring (§4.2 + §4.2.1 interim mode):
    revisions:       0 if REVISIONS_FEED_AVAILABLE=false (score renormalized ×100/70)
    insider:         k20_catalysts insider_buy net 90d count/value
    balance_sheet:   composite from fundamentals
    technical_base:  sma_50 rising flag + weekly_lows_higher flag
    buyback:         k20_catalysts event_type='buyback_auth' 90d
    short_covering:  short_volume_ratio trend from k20_signals
    attention_vacuum: k20_sentiment_daily.mention_z20 < −1 sustained ≥3 days

  Interim renormalization: score_interim = round(score_partial × 100/70)
    where score_partial = sum over the 70 non-revisions points.
    Dossier/order ticket carry "⚠ revisions:n/a" tag.

  Upsert k20_watchlist(ticker, sleeve='A', score, state='screening/candidate').
  Tickers with score ≥ 60 → state='candidate' (queued for Phase 5 LLM dossier).
```

---

### Phase 5 — LLM Layer

#### `llm/client.py`

Central LLM access layer wrapping the Anthropic SDK:

```python
class KestrelLLMClient:
    def call(self, task_type, input_ref, model, prompt, max_tokens) -> dict:
        # 1. Cache check: SELECT output_json FROM k20_llm_runs
        #    WHERE task_type=task_type AND input_ref=input_ref
        # 2. Budget check: SUM(cost_usd) this month < LLM_MONTHLY_BUDGET_USD
        #    Classification continues to 120%; dossiers pause at 100%.
        # 3. anthropic.Anthropic().messages.create(...)
        # 4. Parse JSON response; one retry on parse failure.
        # 5. INSERT k20_llm_runs(ts, ticker, task_type, input_ref, output_json,
        #        model, tokens_in, tokens_out, cost_usd, verdict)
        # 6. Return parsed dict.
```

Uses `anthropic` SDK (already a project dependency based on project context). Model IDs from `config.py`:
- `HAIKU_MODEL = "claude-haiku-4-5-20251001"` for 8-K classification and guidance-delta proxy
- `SONNET_MODEL = "claude-sonnet-4-6"` for dossiers, risk diffs, Form-10 summaries

#### `llm/classifier_8k.py`

```
run_llm_classify_filings.py → classifier_8k.run()
  SELECT pending classify_8k rows from k20_llm_runs WHERE verdict IS NULL.
  For each:
    text = fetch body (already cached by filings_ingest).
    Truncate to 6k tokens (only relevant 8-K items: Item 1.01, 1.02, 5.02, 8.01).
    KestrelLLMClient.call(task_type='classify_8k', model=HAIKU_MODEL, prompt=CLASSIFY_8K_PROMPT)
    Expected output: {event_type, materiality, thesis_impact, one_liner}
    If materiality='high' AND thesis_impact='invalidates':
      INSERT k20_alerts_log(trigger='llm_invalidation', ticker, payload=output)
      → risk_checker picks this up on next run.
```

#### `llm/dossier.py`

```
run_llm_dossiers.py → dossier.run()
  SELECT DISTINCT ticker FROM k20_watchlist WHERE score >= 60 AND state='candidate'
    AND NOT EXISTS (SELECT 1 FROM k20_llm_runs WHERE task_type='candidate_dossier'
                    AND ticker=w.ticker AND ts > now() - interval '7 days').
  For each ticker (batch, Sunday 12:00):
    Build input: fundamentals snapshot + last 2 filing summaries from k20_llm_runs
      + k20_sentiment_daily latest row + k20_signals price stats.
    input_ref = sha256(ticker + date + accession_nos).
    KestrelLLMClient.call(task_type='candidate_dossier', model=SONNET_MODEL, ...)
    Expected output: §8.2 JSON contract.
    Validate: all red_flags have source pointer; reject requires non-empty invalidation.
    UPDATE k20_watchlist SET llm_verdict=verdict, dossier_run_id=llm_run.id.
    If verdict='advance' AND score >= 75:
      INSERT k20_alerts_log(trigger='candidate_advance').
      → digest_send picks up next morning; if intraday push needed, call NotificationClient.
```

---

### Phase 6 — Sleeve B Calendars + Spin-off Monitor

#### `ingest/calendar_sync.py`

```
run_catalyst_sync.py → calendar_sync.run()
  Earnings: FinnhubDataDownloader.get_earnings_calendar(tickers=watchlist+positions)
    → event_type='earnings', event_date, confidence='high'
    Upsert k20_catalysts. Detect date changes vs existing rows:
      if new event_date != existing event_date → state='date_changed',
        stamp datechange_alerted_at, reset t10_alerted_at and t3_alerted_at.
  PDUFA: scrape pdufa.bio (circuit breaker; on failure, log warning, keep existing).
    → event_type='pdufa'
  Index changes: RSS parse → event_type='index_addition'/'index_removal'.
  Spin-off dates: scan k20_llm_runs for Form 10 filings → event_type='spinoff_distribution'.
  T-minus enforcement (§5.1 idempotent):
    days_to_event = event_date - today
    if days_to_event <= 10 AND t10_alerted_at IS NULL:
      push alert, UPDATE k20_catalysts SET t10_alerted_at=now()
    if days_to_event <= 3 AND t3_alerted_at IS NULL:
      push alert, UPDATE k20_catalysts SET t3_alerted_at=now()
```

#### `screening/sleeve_b.py`

B1 FDA: candidates meeting mcap/cash/event-window criteria from `k20_catalysts`; crowding skip from `k20_sentiment_daily`.  
B2 Spin-offs: post-distribution volume normalization from `k20_signals`; gate on mandatory LLM Form-10 dossier.  
B3 Index + activists: filter by curated activist list (config-driven).

---

### Phase 7 — Social Polling + pytrends + Crowding Overlay

#### `sentiment/social_poll.py`

```
run_social_poll.py → social_poll.run()
  tickers = [t from k20_watchlist] + [t from k20_positions WHERE status='open']
  For each source:
    StockTwits: AdapterManager.get_adapter('stocktwits').fetch_summary(ticker)
      → mentions (message count), bullish_ratio (bull/(bull+bear))
    Reddit: AdapterManager.get_adapter('reddit').fetch_summary(ticker)
      → mentions (post count)
    ApeWisdom: AsyncApeWisdomAdapter.fetch_summary(ticker)
      → mentions (from top-100 table; cached per run)
  Upsert k20_sentiment_daily per source.
  Let AdapterManager handle circuit breaker state.
  Write k20_job_runs(job='social_poll', run_date).
```

**Key:** P20 does NOT create new adapters. `AdapterManager` from `src.common.sentiments.adapters.adapter_manager` is instantiated with the three existing adapters. P20 only adds the DB persistence layer.

#### `sentiment/trends_poll.py`

```
run_trends_watchlist.py → trends_poll.run()
  Anchor term: "stock market" (fixed constant in config.py).
  Batches of 4 tickers + anchor per request via AsyncTrendsAdapter.
  Anchor rescaling: norm_i = raw_i × (anchor_ref / anchor_batch)
    where anchor_ref = median of "stock market" interest over trailing 52 weeks
    (stored in k20_sentiment_daily(ticker='__anchor__', source='trends')).
  Jittered sleep 30–60s between batches.
  On HTTP 429: exponential backoff, max 5 retries, then abort and set
    k20_job_runs(job='trends_watchlist', status='skipped', error='rate_limited').
  Upsert k20_sentiment_daily(source='trends') with weekly rows.
  Write k20_job_runs.
```

#### `sentiment/av_budgeted.py`

```
run_av_sentiment.py → av_budgeted.run()
  Check k20_request_budget(source='alpha_vantage', date=today):
    if used >= 20: abort (quota exhausted).
  Priority queue: open positions first (≤8), then top watchlist by score,
    then rotating watchlist (round-robin from yesterday's last-served index).
  For each ticker until quota or limit:
    AlphaVantageDataDownloader.get_news_sentiment(ticker)
    UPDATE k20_request_budget SET used=used+1.
    Upsert k20_sentiment_daily(source='av_news').
  Carry-over: tickers not served → stored in k20_request_budget.notes (JSON list)
    as tomorrow's priority head.
  Write k20_job_runs.
```

#### `sentiment/sentiment_aggregator.py`

This is **not** `src.common.sentiments.processing.sentiment_aggregator.SentimentAggregator`. That class computes a quality-weighted per-message sentiment score; P20 needs a crowding z-score aggregation per §7.6. They are different things.

```
run_sentiment_aggregate.py → sentiment_aggregator.run()
  DAG wait: poll k20_job_runs for today's gdelt_process/social_poll/av_sentiment.
    Proceed at 07:00 regardless; mark missing sources.
  For each ticker in k20_universe:
    Pull latest mention_z20 from k20_sentiment_daily per source.
    Check staleness: social ≤3 days, gdelt ≤2 days, trends ≤10 days.
    usable = {source: z for source, z in components.items()
              if z is not None and age_days(source, ticker) <= STALENESS[source]}
    if len(usable) < 2: crowding = NULL  (fail-open for blocking)
    else: crowding = mean(usable.values())
    Store: k20_signals(ticker, date, signal_type='crowding', value=crowding, sleeve=NULL)
    Store audit: k20_signals(signal_type='crowding_components', value=json(usable.keys()))
  Apply sleeve rules:
    Sleeve C: crowding > pct95_of_ticker_history → write k20_signals(signal_type='crowding_block')
    Sleeve A: attention vacuum (mention_z20 < -1 for ≥3 days) already scored in sleeve_a.py;
      this job refreshes the signal for the next weekly screen.
  Write k20_job_runs.
```

---

### Phase 8 — Weekly Reporting + LLM Calibration + NYSE/AMEX

#### `reporting/weekly_report.py`

```
run_weekly_report.py → weekly_report.run()
  1. Performance: k20_positions with realized/unrealized P&L; benchmark from k20_signals SPY.
  2. Sleeve attribution: per-sleeve P&L breakdown.
  3. Funnel stats: COUNT per state in k20_universe → k20_watchlist → k20_llm_runs → k20_alerts_log.
  4. 2-week catalyst calendar: k20_catalysts next 14 days.
  5. LLM calibration: k20_llm_runs dossiers ≥4 weeks old;
     join with k20_signals returns at +4w and +12w; compute accuracy by verdict tier.
     Log to digest + store in k20_signals(signal_type='llm_calibration_4w').
  6. LLM spend: SUM(cost_usd) by model this month from k20_llm_runs.
  7. GDELT alias precision sample: 20 random k20_sentiment_daily(source='gdelt') rows
     with top_domains URLs for manual spot-check.
  8. Interim score overlap report (when REVISIONS_FEED_AVAILABLE goes true):
     show both score_interim and score_full for 2 weeks.
  Send via NotificationServiceClient (email only for weekly pack; Telegram summary).
```

**NYSE/AMEX extension**: `universe_loader.py` accepts a list of ticker-file paths from `config.py`; adding NYSE/AMEX means adding their CSV paths to that list. No schema change needed.

---

## 5. Job Registration Summary

All P20 jobs use `job_type='script'`. `register_jobs.py` inserts them into `job_schedules`. Jobs are in `src/ml/pipeline/` which is on the scheduler's `_ALLOWED_SCRIPT_DIRS` allowlist.

| Scheduler job name | cron (UTC) | Phase | Script |
|---|---|---|---|
| `k20_universe_refresh` | `30 8 * * 0` | 1 | `run_universe_refresh.py` |
| `k20_ingest_eod` | `30 23 * * *` | 1 | `run_ingest_eod.py` |
| `k20_momentum_rank` | `50 23 * * *` | 1 | `run_momentum_rank.py` |
| `k20_data_health` | `0 7 * * *` | 1 | `run_data_health.py` |
| `k20_digest_send` | `30 7 * * *` | 1 | `run_digest_send.py` |
| `k20_risk_check` | `45 23 * * *` | 2 | `run_risk_check.py` |
| `k20_gdelt_backfill` | (one-time, no cron) | 3 | `run_gdelt_backfill.py` |
| `k20_gdelt_process` | `0 6 * * *` | 3 | `run_gdelt_process.py` |
| `k20_alias_refresh` | `0 9 * * 0` | 3 | `run_alias_refresh.py` |
| `k20_ingest_filings` | `0 11-23/2 * * 1-5` | 4 | `run_ingest_filings.py` |
| `k20_screen_turnaround` | `0 10 * * 0` | 4 | `run_screen_turnaround.py` |
| `k20_llm_classify_filings` | `15 11-23/2 * * 1-5` | 5 | `run_llm_classify_filings.py` |
| `k20_llm_dossiers` | `0 12 * * 0` | 5 | `run_llm_dossiers.py` |
| `k20_catalyst_sync` | `30 6 * * *` | 6 | `run_catalyst_sync.py` |
| `k20_screen_spinoffs` | `0 11 * * 0` | 6 | `run_screen_spinoffs.py` |
| `k20_social_poll` | `30 5 * * *` | 7 | `run_social_poll.py` |
| `k20_av_sentiment` | `45 5 * * *` | 7 | `run_av_sentiment.py` |
| `k20_trends_watchlist` | `0 9 * * 6` | 7 | `run_trends_watchlist.py` |
| `k20_sentiment_aggregate` | `15 6 * * *` (DAG gated) | 7 | `run_sentiment_aggregate.py` |
| `k20_weekly_report` | `0 18 * * 0` | 8 | `run_weekly_report.py` |

*Crons in UTC; spec times in Europe/Zurich (UTC+1/+2 depending on DST — verify at registration.*

---

## 6. Cross-Cutting Implementation Rules

1. **Every `run_*.py`** script follows the p15_daily.py pattern: `_setup_file_logging()`, `main()`, `print(f"__SCHEDULER_RESULT__:{json.dumps(summary)}")`.
2. **Logger**: `_logger = setup_logger(__name__)` — never `print()`.
3. **DB access**: always via `session_scope()` from `src.data.db.core.database`; no long-lived sessions.
4. **No writes to P15/P17/P18 tables** — P20 is read-only on all non-`k20_*` tables.
5. **`config.py`** centralizes: `DATA_CACHE_DIR`, `LLM_MONTHLY_BUDGET_USD` (default 75), `REVISIONS_FEED_AVAILABLE` (default False), `HAIKU_MODEL`, `SONNET_MODEL`, `STALENESS_DAYS` dict.
6. **Pathlib** for all file paths; absolute paths from `Path(__file__).resolve().parents[N]`.
7. **Type hints + docstrings** on all public functions per CLAUDE.md §5 and §6.
8. **Tests** in `tests/` for every module; mock `session_scope()` for DB tests.
9. **`k20_job_runs`** written at start (`status='running'`) and end (`status='ok'/'failed'`) of every job script.
10. **P20 is read-only on all P15 and P18 caches** (`edgar/8k/index/`, `edgar/13f/form4/`, `edgar/13f/13dg/`, `gdelt/`, etc.); two targeted additions to `EdgarDownloader` are the only changes outside `src/ml/pipeline/p20_kestrel/` — see §7.

---

## 7. Changes to Existing Classes Outside P20

Both changes are in **`src/data/downloader/edgar_downloader.py`** only.

### 7.1 New method: `fetch_8k_body()`

```python
def fetch_8k_body(
    self,
    cik: Union[int, str],
    accession_no: str,
    primary_document: Optional[str] = None,
) -> Optional[str]:
    """
    Fetch and cache the primary document text for an 8-K, 13D/G, or Form 10 filing.

    Checks a local body cache at DATA_CACHE_DIR/edgar/8k/bodies/{acc_norm}.txt.gz
    before hitting EDGAR archives. Reuses the existing _fetch_filing_xml() retry and
    rate-limit logic. If primary_document is None, falls back to fetching the filing
    index page to discover the primary document filename.

    Args:
        cik: Issuer CIK (int or zero-padded string).
        accession_no: Accession number with or without dashes.
        primary_document: Filename of primary document (e.g. 'sunc-20260505.htm'),
            read from the 8-K index csv column of the same name. Pass None to
            auto-discover from the filing index page.

    Returns:
        Raw text of the primary document, or None on fetch failure.
    """
```

Used by: `filings_ingest.run_8k()` (8-K bodies for LLM queue) and `filings_ingest.run_13dg()` (activist filing body for issuer extraction).

### 7.2 Parameter addition: `download_form4_filings(transaction_codes=...)`

The existing `_parse_form4_xml()` hard-codes `_SALE_CODES = {"S", "S-"}` and silently drops all purchase transactions. Sleeve A scoring requires net insider **buying** (`P` = open-market purchase, `A` = grant/award).

Change: add `transaction_codes: Optional[Set[str]] = None` parameter to `download_form4_filings()` and pass it through to `_parse_form4_xml()`. When `None`, defaults to the existing `{"S", "S-"}` — fully backwards-compatible; P18 `Form4Monitor` is unaffected.

P20 `filings_ingest.run_form4()` calls:
```python
edgar.download_form4_filings(as_of_date=date, transaction_codes={"P", "A"})
```

No schema change to the cached CSV; new columns are not needed — the existing `transaction_code` field already carries the code value.

---

## 8. Explicitly Out of Scope for P20

- No changes to `p15_daily.py` or any P15/P17/P18 pipeline code
- No changes to `Form4Monitor` (P18) — it continues to work with sells-only; the `transaction_codes` param is additive
- No Telegram bot handler wiring (Phase 2 delivers the parse function; bot integration is separate)
- No backtest harness (Phase 9 optional)
- No options data (Phase 9 optional)
- No execution engine; all orders remain manual
