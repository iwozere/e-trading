-- SQL to schedule the P19 Intraday Penny-Stock Spike Monitor (Phase 1: shadow mode).
-- User ID: 2 (akossyrev@gmail.com, Telegram: 859865894)
--
-- Three jobs (all weekdays). Times are UTC; US regular session is 13:30–20:00 UTC
-- in summer (EDT) and 14:30–21:00 UTC in winter (EST). The crons below span both
-- DST regimes; off-session polls simply log few/zero quotes and are harmless. The
-- intraday metrics convert to ET internally, so they stay correct across DST.
--
-- The scheduler (data_processing executor) auto-injects `--user-id 2` after the
-- script args, which P19 uses for (future Phase 2) alert delivery. Phase 1 = shadow
-- logging only, no notifications.
--
-- Usage:
--   psql -d your_database < bin/scheduler/insert_p19_schedules.sql

-- ==============================================================================
-- 1. Watchlist Builder — once, pre-market
-- ==============================================================================
-- Merges P17 daily output (Tier B/C + explosive) with the IBKR pre-market gappers
-- scanner, hard-filters, ranks, caps to N<=100, and writes watchlist.json with
-- baseline context. Runs after P17's 06:00 UTC daily run, before the open.
INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'P19 Intraday Watchlist Build',
    'data_processing',
    'src.ml.pipeline.p19_penny_intraday.run_p19',
    '{
        "script_path": "src/ml/pipeline/p19_penny_intraday/run_p19.py",
        "script_args": ["build-watchlist"],
        "timeout_seconds": 600
    }'::jsonb,
    '0 13 * * 1-5',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
);

-- ==============================================================================
-- 2. Shadow Loop — every 15 min during (and around) market hours
-- ==============================================================================
-- One delayed IBKR reqMktData snapshot per watchlist name -> %-move / RVOL-so-far
-- -> append to results/p19_penny_intraday/shadow.sqlite. No alerts (Phase 1).
-- 13–21 UTC covers both DST regimes; pre/post-session polls are cheap no-ops.
INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'P19 Intraday Shadow Poll',
    'data_processing',
    'src.ml.pipeline.p19_penny_intraday.run_p19',
    '{
        "script_path": "src/ml/pipeline/p19_penny_intraday/run_p19.py",
        "script_args": ["run-once", "--mode", "shadow"],
        "timeout_seconds": 300
    }'::jsonb,
    '*/15 13-21 * * 1-5',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
);

-- ==============================================================================
-- 3. EOD Backfill — once, after the close
-- ==============================================================================
-- Fills O/H/L/C into the day's shadow rows (via DataManager / DATA_CACHE_DIR) so
-- each detection has its realised outcome for later calibration. 21:30 UTC is after
-- both the summer (20:00) and winter (21:00) closes.
INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'P19 Intraday EOD Backfill',
    'data_processing',
    'src.ml.pipeline.p19_penny_intraday.run_p19',
    '{
        "script_path": "src/ml/pipeline/p19_penny_intraday/run_p19.py",
        "script_args": ["eod-backfill"],
        "timeout_seconds": 1800
    }'::jsonb,
    '30 21 * * 1-5',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
);

-- ==============================================================================
-- Verification
-- ==============================================================================
-- SELECT id, name, cron, enabled FROM job_schedules WHERE user_id = 2 AND name LIKE 'P19%';
-- ==============================================================================
