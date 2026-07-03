# Design

## Purpose

P20 Kestrel is a trading intelligence pipeline that supports the human operator's stock selection and position management across three complementary strategies. It does NOT execute orders — it generates scored candidates, LLM-enhanced dossiers, Telegram alerts, and daily/weekly reports.

## Architecture

### High-Level Architecture

```
Data Sources           Pipeline                   Outputs
────────────           ────────                   ───────
Nasdaq CSV     ─→  universe_loader    ─→  k20_universe
OHLCV feeds    ─→  eod_ingest         ─→  k20_signals
EDGAR filings  ─→  filings_ingest     ─→  k20_signals / llm_queue.json
Catalyst feeds ─→  calendar_sync      ─→  k20_catalysts

GDELT GKG      ─→  gdelt_processor    ─→  k20_sentiment_daily
StockTwits     ─→  social_poll        ─→  k20_sentiment_daily
Reddit         ─→  social_poll        ─→  k20_sentiment_daily
ApeWisdom      ─→  social_poll        ─→  k20_sentiment_daily
AlphaVantage   ─→  av_budgeted        ─→  k20_sentiment_daily
Google Trends  ─→  trends_poll        ─→  k20_signals
All above      ─→  sentiment_agg      ─→  k20_signals (crowding)

k20_universe
k20_signals    ─→  sleeve_a           ─→  k20_watchlist
k20_catalysts  ─→  sleeve_b           ─→  k20_watchlist
k20_signals    ─→  sleeve_c           ─→  k20_watchlist

k20_watchlist  ─→  llm/classifier_8k  ─→  k20_llm_runs + alerts
k20_watchlist  ─→  llm/dossier        ─→  k20_llm_runs + k20_watchlist
k20_watchlist  ─→  llm/risk_diff      ─→  k20_llm_runs

k20_positions  ─→  risk_checker       ─→  alerts
Telegram cmd   ─→  pos_commands       ─→  k20_positions

k20_*          ─→  daily_digest       ─→  Notification push
k20_*          ─→  weekly_report      ─→  Notification push
```

### Component Design

**Repository Layer (`src.data.db.repos.repo_kestrel.KestrelRepo`)**
- Single module, thin wrappers, no raw SQL outside of it
- `session_scope()` handles commit/rollback
- PostgreSQL upsert via `pg_insert(...).on_conflict_do_update()`

**Sentiment Pipeline (§7)**
- Alias table built weekly by `alias_builder.py` using legal-name normalization + async fundamentals
- GDELT GKG processed with exact + fuzzy (≥0.93 SequenceMatcher) matching and blocklist
- Crowding score = mean of usable z-scores (≥2 sources required, else fail-open)
- Staleness policy per source: GDELT=2d, Social=3d, Trends=10d, AV=2d

**Scoring (§4.2.1 interim mode)**
- When `REVISIONS_FEED_AVAILABLE=False`: `score = round(partial_score × 100 / 70)`
- Interim score denominator is 70 pts (insider 20 + BS 15 + tech 15 + buyback 10 + short 5 + attention 5)

**LLM Budget (§8)**
- Monthly cap from `LLM_MONTHLY_BUDGET_USD` config
- Warn at 80%, stop dossiers at 100%, stop all at 120%
- All LLM calls cached via `k20_llm_runs` (task_type + input_ref uniqueness)

**Scheduler Integration**
- All jobs registered as `job_type='script'` in `job_schedules` (existing table)
- Scripts in `src/ml/pipeline/p20_kestrel/jobs/` (on scheduler allowlist)
- Each script prints `__SCHEDULER_RESULT__:{json}` on success
- `k20_job_runs` tracks per-job-per-date status for DAG dependency checks

## Data Flow

1. **Monday 04:00**: Universe refresh → fundamentals enrichment → k20_universe
2. **Weekday 20:00**: EOD OHLCV → technicals → k20_signals
3. **Weekday 20:30**: EDGAR filings → Form 4 signals, LLM queue
4. **Weekday 06:15**: GDELT GKG → sentiment z-scores
5. **Weekday 06:30**: Social poll → StockTwits/Reddit/ApeWisdom
6. **Weekday 07:00**: Sentinel aggregation → crowding score → k20_signals
7. **Weekday 21:00–22:00**: Sleeves A/B/C screen → k20_watchlist
8. **Weekday 22:00**: LLM 8-K classify → update watchlist state
9. **Weekday 22:30**: LLM dossiers for high-score candidates
10. **Weekday 06:30**: Daily digest sent to Telegram

## Design Decisions

1. **Human-in-the-loop**: No auto-execution. All signals → alerts, human decides.
2. **Interim scoring**: Sleeve A denominator 70 until revisions feed is available.
3. **Fail-open sentiment**: Crowding score uses available z-scores ≥2; else no crowding filter.
4. **k20_ table prefix**: Avoids collision with existing project tables.
5. **Repository pattern**: No SQLAlchemy Session objects outside repos.py.
6. **Budget-aware LLM**: Monthly $ cap with graceful degradation tiers.
7. **EDGAR via downloader**: Uses existing EdgarDownloader; Form 4 fetched for date not per-ticker.
