-- SQL script to schedule P20 Kestrel — Semi-Automated Trading Intelligence
-- User ID: 2 (akossyrev@gmail.com, Telegram: 859865894)
--
-- P20 Kestrel is a 3-sleeve trading intelligence pipeline (no auto-execution).
-- Sleeve A: Turnaround / Fallen Angels
-- Sleeve B: Event Catalysts (FDA run-ups, activists)
-- Sleeve C: Momentum (RS-ranked, regime-filtered)
--
-- All times UTC. Data ownership rule:
--   P15 daily (13:00 UTC) writes: GDELT GKG, 8-K index, Form 4, 13D/G cache files.
--   P15 weekly (13:30 UTC Fri) writes: Nasdaq screener CSV.
--   P20 reads those files only — never downloads raw data itself.
--
-- __SCHEDULER_RESULT__ field reference (common across all P20 scripts):
--   success     — bool, overall job success
--   job         — string, job name
--   rows_out    — int, rows written to DB (where applicable)
--
-- Idempotent: safe to re-run (ON CONFLICT (user_id, name) DO NOTHING).
-- Usage:
--   psql -d your_database < bin/scheduler/insert_p20_schedules.sql

-- ==============================================================================
-- MORNING CHAIN  (weekdays, UTC 06:00 – 07:30)
-- Depends on P15 daily (13:00 UTC previous day) having finished.
-- ==============================================================================

-- 1. Data Health Check — 06:00 UTC
-- Verifies P15 cache freshness (GDELT, EDGAR), AV/LLM budget state.
-- Sends Telegram warning when any source is stale; blocks morning chain implicitly
-- via k20_job_runs dependency checks in downstream jobs.
-- Result fields: success, alerts_sent, stale_sources
-- Timeout: 5 min (filesystem reads + DB query only).
INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'P20 Data Health Check',
    'data_processing',
    'src.ml.pipeline.p20_kestrel.jobs.run_data_health',
    '{
        "script_path": "src/ml/pipeline/p20_kestrel/jobs/run_data_health.py",
        "script_args": [],
        "timeout_seconds": 300,
        "notification_rules": {
            "conditions": [
                {
                    "check_field": "alerts_sent",
                    "operator": ">",
                    "threshold": 0,
                    "channels": ["telegram"],
                    "comment": "Telegram warning when any data source is stale or budget exceeded"
                }
            ]
        }
    }'::jsonb,
    '0 6 * * 1-5',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
) ON CONFLICT (user_id, name) DO NOTHING;

-- 2. GDELT GKG Processor — 06:15 UTC
-- Reads P15-downloaded GKG files, runs fuzzy-match alias resolution,
-- computes per-ticker z-scores, writes to k20_sentiment_daily.
-- Result fields: success, tickers_processed, rows_written
-- Timeout: 10 min (in-memory processing of pre-downloaded files).
INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'P20 GDELT Process',
    'data_processing',
    'src.ml.pipeline.p20_kestrel.jobs.run_gdelt_process',
    '{
        "script_path": "src/ml/pipeline/p20_kestrel/jobs/run_gdelt_process.py",
        "script_args": [],
        "timeout_seconds": 600
    }'::jsonb,
    '15 6 * * 1-5',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
) ON CONFLICT (user_id, name) DO NOTHING;

-- 3. Social Sentiment Poll — 06:30 UTC
-- Fetches StockTwits, Reddit (r/stocks, r/investing, r/wallstreetbets), ApeWisdom
-- for all watchlist tickers; rate-limited at 2.1s per ticker per source.
-- Result fields: success, tickers_polled, rows_written
-- Timeout: 15 min (rate-limited API calls; 100 tickers × 3 sources × 2.1s ≈ 10 min).
INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'P20 Social Sentiment Poll',
    'data_processing',
    'src.ml.pipeline.p20_kestrel.jobs.run_social_poll',
    '{
        "script_path": "src/ml/pipeline/p20_kestrel/jobs/run_social_poll.py",
        "script_args": [],
        "timeout_seconds": 900
    }'::jsonb,
    '30 6 * * 1-5',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
) ON CONFLICT (user_id, name) DO NOTHING;

-- 4. AlphaVantage News Sentiment — 06:45 UTC
-- Fetches AV sentiment for watchlist tickers in priority order; stops when the
-- daily 20-call quota is exhausted. Writes to k20_sentiment_daily.
-- Result fields: success, calls_made, tickers_covered, budget_pct
-- Timeout: 10 min (quota-capped; typically < 20 calls × 12s delay).
INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'P20 AV Sentiment',
    'data_processing',
    'src.ml.pipeline.p20_kestrel.jobs.run_av_sentiment',
    '{
        "script_path": "src/ml/pipeline/p20_kestrel/jobs/run_av_sentiment.py",
        "script_args": [],
        "timeout_seconds": 600
    }'::jsonb,
    '45 6 * * 1-5',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
) ON CONFLICT (user_id, name) DO NOTHING;

-- 5. Sentiment Aggregation — 07:00 UTC
-- Reads GDELT + social + AV z-scores, computes §7.6 crowding score
-- (mean of ≥2 usable z-scores; fail-open if fewer), writes to k20_signals.
-- Result fields: success, tickers_scored, rows_written
-- Timeout: 5 min (DB aggregation only).
INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'P20 Sentiment Aggregate',
    'data_processing',
    'src.ml.pipeline.p20_kestrel.jobs.run_sentiment_aggregate',
    '{
        "script_path": "src/ml/pipeline/p20_kestrel/jobs/run_sentiment_aggregate.py",
        "script_args": [],
        "timeout_seconds": 300
    }'::jsonb,
    '0 7 * * 1-5',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
) ON CONFLICT (user_id, name) DO NOTHING;

-- 6. Daily Digest Send — 07:30 UTC
-- Builds and sends the 07:30 morning digest: regime line, open positions,
-- upcoming catalysts (5d), top-3 candidates, data health summary.
-- Delivers via Telegram to admins.
-- Result fields: success, sections_included, delivery_ok
-- Timeout: 5 min (DB reads + notification API).
INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'P20 Daily Digest',
    'data_processing',
    'src.ml.pipeline.p20_kestrel.jobs.run_digest_send',
    '{
        "script_path": "src/ml/pipeline/p20_kestrel/jobs/run_digest_send.py",
        "script_args": [],
        "timeout_seconds": 300
    }'::jsonb,
    '30 7 * * 1-5',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
) ON CONFLICT (user_id, name) DO NOTHING;

-- ==============================================================================
-- GOOGLE TRENDS POLL  (early morning, weekdays)
-- ==============================================================================

-- 7. Google Trends Watchlist Poll — 03:00 UTC
-- Fetches pytrends relative interest for all watchlist tickers (30-60s jitter
-- per batch to avoid rate limiting). Writes trend_interest signals to k20_signals.
-- Result fields: success, tickers_polled, rows_written, aborted_429
-- Timeout: 30 min (jitter-based throttle; watchlist size-dependent).
INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'P20 Trends Poll',
    'data_processing',
    'src.ml.pipeline.p20_kestrel.jobs.run_trends_watchlist',
    '{
        "script_path": "src/ml/pipeline/p20_kestrel/jobs/run_trends_watchlist.py",
        "script_args": [],
        "timeout_seconds": 1800
    }'::jsonb,
    '0 3 * * 1-5',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
) ON CONFLICT (user_id, name) DO NOTHING;

-- ==============================================================================
-- EOD INGEST CHAIN  (weekdays, UTC 20:00 – 21:00)
-- Runs after US market close (20:00 UTC = 16:00 ET).
-- ==============================================================================

-- 8. EOD OHLCV + Technicals — 20:00 UTC
-- Downloads yesterday's OHLCV for all k20_universe tickers via DataManager,
-- computes SMA-50, SMA-200, RSI-14, RS scores, and upserts into k20_signals.
-- Result fields: success, tickers_updated, rows_written
-- Timeout: 60 min (batch fundamentals fetch for 500–3000 tickers).
INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'P20 EOD Ingest',
    'data_processing',
    'src.ml.pipeline.p20_kestrel.jobs.run_ingest_eod',
    '{
        "script_path": "src/ml/pipeline/p20_kestrel/jobs/run_ingest_eod.py",
        "script_args": [],
        "timeout_seconds": 3600
    }'::jsonb,
    '0 20 * * 1-5',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
) ON CONFLICT (user_id, name) DO NOTHING;

-- 9. Filings Ingest — 20:30 UTC
-- Reads P15 Form 4, 8-K index, 13D/G caches; filters to watchlist+positions tickers;
-- records insider-buy signals; queues 8-K filings for LLM classification.
-- Result fields: success, filings_8k_queued, form4_buys, activist_matches
-- Timeout: 30 min (reads P15 cache files; no EDGAR API calls).
INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'P20 Filings Ingest',
    'data_processing',
    'src.ml.pipeline.p20_kestrel.jobs.run_ingest_filings',
    '{
        "script_path": "src/ml/pipeline/p20_kestrel/jobs/run_ingest_filings.py",
        "script_args": [],
        "timeout_seconds": 1800
    }'::jsonb,
    '30 20 * * 1-5',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
) ON CONFLICT (user_id, name) DO NOTHING;

-- 10. Catalyst Calendar Sync — 20:45 UTC
-- Syncs upcoming FDA PDUFA/AdCom dates and earnings from catalyst feed into
-- k20_catalysts; fires T-10 and T-3 day countdown alerts via Telegram.
-- Result fields: success, catalysts_synced, alerts_fired
-- Timeout: 10 min (API + DB writes).
INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'P20 Catalyst Sync',
    'data_processing',
    'src.ml.pipeline.p20_kestrel.jobs.run_catalyst_sync',
    '{
        "script_path": "src/ml/pipeline/p20_kestrel/jobs/run_catalyst_sync.py",
        "script_args": [],
        "timeout_seconds": 600,
        "notification_rules": {
            "conditions": [
                {
                    "check_field": "alerts_fired",
                    "operator": ">",
                    "threshold": 0,
                    "channels": ["telegram"],
                    "comment": "Telegram alert when T-10 or T-3 catalyst countdown triggers"
                }
            ]
        }
    }'::jsonb,
    '45 20 * * 1-5',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
) ON CONFLICT (user_id, name) DO NOTHING;

-- ==============================================================================
-- SCREENING CHAIN  (weekdays, UTC 21:00 – 22:00)
-- ==============================================================================

-- 11. Sleeve A — Turnaround Screen — 21:00 UTC
-- Hard-filters k20_universe for down-40-75% fallen angels; applies interim §4.2.1
-- scoring (insider buy 20pt + balance sheet 15pt + technicals 15pt + buyback 10pt
-- + short 5pt + attention 5pt, scaled to 100); upserts to k20_watchlist.
-- Result fields: success, candidates, watchlist_entries
-- Timeout: 5 min (DB query + scoring).
INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'P20 Screen Turnaround',
    'data_processing',
    'src.ml.pipeline.p20_kestrel.jobs.run_screen_turnaround',
    '{
        "script_path": "src/ml/pipeline/p20_kestrel/jobs/run_screen_turnaround.py",
        "script_args": [],
        "timeout_seconds": 300,
        "notification_rules": {
            "conditions": [
                {
                    "check_field": "candidates",
                    "operator": ">",
                    "threshold": 0,
                    "channels": ["telegram"],
                    "comment": "Telegram alert when new Sleeve A turnaround candidates found"
                }
            ]
        }
    }'::jsonb,
    '0 21 * * 1-5',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
) ON CONFLICT (user_id, name) DO NOTHING;

-- 12. Sleeve B — Event Catalyst Screen — 21:15 UTC
-- B1: FDA run-ups ($300M–$10B mcap, PDUFA/AdCom/readout 10–90 days out).
-- B2: Spin-offs in the 20–60 day post-spin entry window (>$150M mcap).
-- B3: Activist 13D screen from k20_signals activist signals.
-- Result fields: success, b1_fda_runups, b2_spinoffs, b3_activists, total_candidates
-- Timeout: 5 min (DB query only).
INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'P20 Screen Spinoffs',
    'data_processing',
    'src.ml.pipeline.p20_kestrel.jobs.run_screen_spinoffs',
    '{
        "script_path": "src/ml/pipeline/p20_kestrel/jobs/run_screen_spinoffs.py",
        "script_args": [],
        "timeout_seconds": 300,
        "notification_rules": {
            "conditions": [
                {
                    "check_field": "b1_fda_runups",
                    "operator": ">",
                    "threshold": 0,
                    "channels": ["telegram"],
                    "comment": "Telegram alert when FDA run-up candidates found (Sleeve B1)"
                },
                {
                    "check_field": "b2_spinoffs",
                    "operator": ">",
                    "threshold": 0,
                    "channels": ["telegram"],
                    "comment": "Telegram alert when post-spin entry candidates found (Sleeve B2)"
                }
            ]
        }
    }'::jsonb,
    '15 21 * * 1-5',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
) ON CONFLICT (user_id, name) DO NOTHING;

-- 13. Sleeve C — Momentum Rank — 21:30 UTC
-- RS = 0.5×(3m pct) + 0.5×(6m pct); top-decile eligible when SPY > 200 DMA.
-- Applies crowding overlay: excludes tickers with z-score > threshold.
-- Result fields: success, eligible, ranked, watchlist_entries
-- Timeout: 5 min (DB query + RS computation).
INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'P20 Momentum Rank',
    'data_processing',
    'src.ml.pipeline.p20_kestrel.jobs.run_momentum_rank',
    '{
        "script_path": "src/ml/pipeline/p20_kestrel/jobs/run_momentum_rank.py",
        "script_args": [],
        "timeout_seconds": 300,
        "notification_rules": {
            "conditions": [
                {
                    "check_field": "watchlist_entries",
                    "operator": ">",
                    "threshold": 0,
                    "channels": ["telegram"],
                    "comment": "Telegram alert when new Sleeve C momentum candidates added"
                }
            ]
        }
    }'::jsonb,
    '30 21 * * 1-5',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
) ON CONFLICT (user_id, name) DO NOTHING;

-- ==============================================================================
-- LLM CHAIN  (weekdays, UTC 22:00 – 23:30)
-- Budget-capped: warn@80%, dossier-stop@100%, full-stop@120% of LLM_MONTHLY_BUDGET_USD.
-- ==============================================================================

-- 14. LLM 8-K Classifier — 22:00 UTC
-- Reads llm_queue.json written by filings ingest; classifies each 8-K filing
-- for thesis impact (positive / neutral / negative) using claude-haiku-4-5.
-- Caches results in k20_llm_runs; skips already-classified filings.
-- Result fields: success, classified, skipped_cached, cost_usd, budget_pct
-- Timeout: 30 min (LLM API calls; budget-capped).
INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'P20 LLM Classify Filings',
    'data_processing',
    'src.ml.pipeline.p20_kestrel.jobs.run_llm_classify_filings',
    '{
        "script_path": "src/ml/pipeline/p20_kestrel/jobs/run_llm_classify_filings.py",
        "script_args": [],
        "timeout_seconds": 1800,
        "notification_rules": {
            "conditions": [
                {
                    "check_field": "budget_pct",
                    "operator": ">=",
                    "threshold": 80,
                    "channels": ["telegram"],
                    "comment": "Telegram warning when LLM monthly budget reaches 80%"
                }
            ]
        }
    }'::jsonb,
    '0 22 * * 1-5',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
) ON CONFLICT (user_id, name) DO NOTHING;

-- 15. LLM Dossier Generation — 22:30 UTC
-- Generates full investment dossiers for high-score watchlist candidates
-- using claude-sonnet-4-6. Stops at 100% monthly budget.
-- Result fields: success, dossiers_generated, skipped_budget, cost_usd, budget_pct
-- Timeout: 60 min (LLM API; one dossier per candidate, budget-capped).
INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'P20 LLM Dossiers',
    'data_processing',
    'src.ml.pipeline.p20_kestrel.jobs.run_llm_dossiers',
    '{
        "script_path": "src/ml/pipeline/p20_kestrel/jobs/run_llm_dossiers.py",
        "script_args": [],
        "timeout_seconds": 3600,
        "notification_rules": {
            "conditions": [
                {
                    "check_field": "dossiers_generated",
                    "operator": ">",
                    "threshold": 0,
                    "channels": ["telegram"],
                    "comment": "Telegram notification when new candidate dossiers are ready"
                },
                {
                    "check_field": "budget_pct",
                    "operator": ">=",
                    "threshold": 80,
                    "channels": ["email", "telegram"],
                    "comment": "Email + Telegram warning when LLM monthly budget reaches 80%"
                }
            ]
        }
    }'::jsonb,
    '30 22 * * 1-5',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
) ON CONFLICT (user_id, name) DO NOTHING;

-- ==============================================================================
-- INTRADAY RISK CHECK  (weekdays, every 30 min during market hours)
-- ==============================================================================

-- 16. Intraday Risk Check — every 30 min, 09:00–17:00 UTC weekdays
-- Checks open positions against stop-loss (default 75% of entry), T1 target
-- (135%), T2 target (160%), and trailing stop (20%). Fires Telegram alerts
-- when any threshold is breached. Fast no-op outside market hours.
-- Result fields: success, positions_checked, alerts_fired
-- Timeout: 5 min (DB read + price fetch).
INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'P20 Risk Check',
    'data_processing',
    'src.ml.pipeline.p20_kestrel.jobs.run_risk_check',
    '{
        "script_path": "src/ml/pipeline/p20_kestrel/jobs/run_risk_check.py",
        "script_args": [],
        "timeout_seconds": 300,
        "notification_rules": {
            "conditions": [
                {
                    "check_field": "alerts_fired",
                    "operator": ">",
                    "threshold": 0,
                    "channels": ["telegram"],
                    "comment": "Telegram alert when stop-loss, T1, T2, or trailing stop is breached"
                }
            ]
        }
    }'::jsonb,
    '*/30 9-17 * * 1-5',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
) ON CONFLICT (user_id, name) DO NOTHING;

-- ==============================================================================
-- WEEKLY MAINTENANCE  (Monday 05:00 UTC)
-- Runs before the P15 weekly bundle (13:30 UTC) and before Monday's daily chain.
-- ==============================================================================

-- 17. Weekly Maintenance — Monday 05:00 UTC
-- (1) Universe loader: reads Nasdaq screener CSV written by P15 weekly,
--     enriches with fundamentals (get_fundamentals_unified), upserts k20_universe,
--     marks delisted tickers.
-- (2) Alias builder: rebuilds company alias table (k20_aliases) for GDELT matching
--     using legal-name normalization + async fundamentals.
-- Result fields: success, universe.tickers_upserted, aliases.aliases_written
-- Timeout: 60 min (3000+ tickers × async fundamentals fetch).
INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'P20 Weekly Maintenance',
    'data_processing',
    'src.ml.pipeline.p20_kestrel.jobs.run_weekly_maintenance',
    '{
        "script_path": "src/ml/pipeline/p20_kestrel/jobs/run_weekly_maintenance.py",
        "script_args": [],
        "timeout_seconds": 3600
    }'::jsonb,
    '0 5 * * 1',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
) ON CONFLICT (user_id, name) DO NOTHING;

-- ==============================================================================
-- WEEKLY REPORT  (Sunday 17:00 UTC)
-- ==============================================================================

-- 18. Weekly Report — Sunday 17:00 UTC
-- Builds and sends the Sunday performance summary: open positions P&L,
-- funnel stats (universe → screened → watchlist), 14-day catalyst calendar,
-- and LLM monthly spend vs. budget. Delivered via Telegram to admins.
-- Result fields: success, positions_count, candidates_in_funnel, delivery_ok
-- Timeout: 5 min (DB aggregation + notification).
INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'P20 Weekly Report',
    'data_processing',
    'src.ml.pipeline.p20_kestrel.jobs.run_weekly_report',
    '{
        "script_path": "src/ml/pipeline/p20_kestrel/jobs/run_weekly_report.py",
        "script_args": [],
        "timeout_seconds": 300
    }'::jsonb,
    '0 17 * * 0',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
) ON CONFLICT (user_id, name) DO NOTHING;

-- ==============================================================================
-- LLM RISK DIFF  (Sunday 18:00 UTC — after weekly report)
-- Runs weekly; 10-K/Q filings change quarterly so daily cadence is wasteful.
-- ==============================================================================

-- 19. LLM Risk Diff — Sunday 18:00 UTC
-- Fetches the two most recent 10-K or 10-Q filings for each watchlist ticker
-- via EDGAR submissions, extracts Item 1A risk sections, and runs an LLM diff
-- to surface newly added or materially escalated risk factors.
-- Results are cached in k20_llm_runs; already-classified filings are skipped.
-- Result fields: success, tickers_processed, diffs_ok, red_flags_found, errors
-- Timeout: 60 min (EDGAR HTTP + LLM API; budget-capped at 100% monthly).
INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'P20 LLM Risk Diff',
    'data_processing',
    'src.ml.pipeline.p20_kestrel.jobs.run_llm_risk_diff',
    '{
        "script_path": "src/ml/pipeline/p20_kestrel/jobs/run_llm_risk_diff.py",
        "script_args": [],
        "timeout_seconds": 3600,
        "notification_rules": {
            "conditions": [
                {
                    "check_field": "red_flags_found",
                    "operator": ">",
                    "threshold": 0,
                    "channels": ["telegram"],
                    "comment": "Telegram alert when new risk red flags are detected in watchlist filings"
                }
            ]
        }
    }'::jsonb,
    '0 18 * * 0',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
) ON CONFLICT (user_id, name) DO NOTHING;

-- ==============================================================================
-- Verification Query
-- ==============================================================================
-- SELECT id, name, cron, enabled FROM job_schedules WHERE user_id = 2 AND name LIKE 'P20%' ORDER BY id;
-- ==============================================================================
