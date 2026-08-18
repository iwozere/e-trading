-- SQL to schedule P19 v2 Phase 1.5 (Structural Integrity Profiler + T+10
-- label backfill) and Phase 3 (intraday filings poll) — design-v2.md §7 /
-- §Roadmap. Additive to insert_p19_schedules.sql (v1's three jobs) — apply
-- all files, in any order.
-- User ID: 2 (akossyrev@gmail.com, Telegram: 859865894)
--
-- Idempotent: safe to re-run (ON CONFLICT (user_id, name) DO NOTHING).
-- Usage:
--   psql -d your_database < bin/scheduler/insert_p19_v2_schedules.sql

-- ==============================================================================
-- 1. Structural Profile — pre-market, after the watchlist build
-- ==============================================================================
-- Layer 0 (spec v2 §4.0): reads that day's watchlist.json (build-watchlist,
-- 13:00 UTC, must run first) and refreshes each name's StructuralProfile —
-- weekly full refresh + daily delta check, cached per ticker
-- (results/p19_penny_intraday/structural_cache/). Entirely EDGAR + yfinance,
-- no IBKR — costs nothing against the intraday market-data-line budget.
-- A name with no cached profile yet still gets shadow-logged (decision #7),
-- so this job running late or partially failing never blocks collection.
INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'P19 Structural Profile',
    'data_processing',
    'src.ml.pipeline.p19_penny_intraday.run_p19',
    '{
        "script_path": "src/ml/pipeline/p19_penny_intraday/run_p19.py",
        "script_args": ["profile-structural"],
        "timeout_seconds": 1800
    }'::jsonb,
    '10 13 * * 1-5',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
) ON CONFLICT (user_id, name) DO NOTHING;

-- ==============================================================================
-- 2. Label Backfill — once daily
-- ==============================================================================
-- T+10 forward-return + structural-decay labels (spec v2 §12.2, §16 item 4).
-- Self-gating: LabelBackfill.run() only acts on shadow dates old enough
-- (>= ~16 calendar days) to plausibly have T+10 session data, so running this
-- daily is a safe no-op on any date that isn't ready yet. Scheduled well
-- before the open; independent of same-day market data.
INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'P19 Label Backfill',
    'data_processing',
    'src.ml.pipeline.p19_penny_intraday.run_p19',
    '{
        "script_path": "src/ml/pipeline/p19_penny_intraday/run_p19.py",
        "script_args": ["label-backfill"],
        "timeout_seconds": 1800
    }'::jsonb,
    '0 12 * * 1-5',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
) ON CONFLICT (user_id, name) DO NOTHING;

-- ==============================================================================
-- 3. Intraday Filings Poll — every 30 min during market hours (Phase 3, spec §9)
-- ==============================================================================
-- EFTS scan of the day's watchlist CIKs for 424B5/S-1/S-3 + 8-K items 3.01/3.02
-- filed intraday. Log-only (filings_poll.py's own SQLite table, separate from
-- shadow.sqlite and from the shared universe-wide 8-K index) — no Alert Manager
-- exists yet (Phase 2), so this seeds calibration/awareness, not alerts.
-- Cheap: a handful of CIK-scoped EFTS queries per run, not one per ticker.
-- 13-21 UTC covers both DST regimes, same as the shadow poll.
INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'P19 Intraday Filings Poll',
    'data_processing',
    'src.ml.pipeline.p19_penny_intraday.run_p19',
    '{
        "script_path": "src/ml/pipeline/p19_penny_intraday/run_p19.py",
        "script_args": ["filings-poll"],
        "timeout_seconds": 600
    }'::jsonb,
    '*/30 13-21 * * 1-5',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
) ON CONFLICT (user_id, name) DO NOTHING;

-- ==============================================================================
-- Verification
-- ==============================================================================
-- SELECT id, name, cron, enabled FROM job_schedules WHERE user_id = 2 AND name LIKE 'P19%';
-- ==============================================================================
