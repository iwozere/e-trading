-- SQL script to schedule P15 Options Daily pipeline
-- User ID: 2 (akossyrev@gmail.com, Telegram: 859865894)
--
-- Runs at 06:00 UTC Mon–Fri — 7 hours before p15_daily.py (13:00 UTC).
-- By the time p15_daily.py runs, options chains for all NASDAQ-listed P15
-- tickers are already cached and will be skipped in the options_putcall job.
--
-- Usage:
--   psql -d your_database < bin/scheduler/insert_p15_options_schedules.sql

-- ==============================================================================
-- 1. P15 Options Daily
-- ==============================================================================
-- Downloads yesterday's full options chain for every optionable NASDAQ-listed
-- security (~3 000–4 000 symbols, ~1 500–2 000 with live options chains).
-- Results are stored in DATA_CACHE_DIR/options/chains/{TICKER}/{YYYY-MM-DD}.csv.gz
-- and daily put/call summaries in DATA_CACHE_DIR/options/putcall/{TICKER}_putcall.csv.gz.
-- The NASDAQ symbol list is cached at DATA_CACHE_DIR/nasdaq/{YYYY-MM-DD}.csv.gz.
-- Timeout: 6 hours (3 threads × ~3 500 symbols, network-bound).
-- ==============================================================================

INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'P15 Options Daily',
    'data_processing',
    'src.ml.pipeline.p15_hidden_deps.p15_options_daily',
    '{
        "script_path": "src/ml/pipeline/p15_hidden_deps/p15_options_daily.py",
        "script_args": [],
        "timeout_seconds": 21600,
        "notification_rules": {
            "conditions": [
                {
                    "check_field": "ok",
                    "operator": ">=",
                    "threshold": 0,
                    "channels": ["email"],
                    "comment": "Email summary on every completion"
                },
                {
                    "check_field": "error",
                    "operator": ">",
                    "threshold": 0,
                    "channels": ["email", "telegram"],
                    "comment": "Email + Telegram when any ticker errors are recorded"
                }
            ]
        }
    }'::jsonb,
    '0 6 * * 1-5',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
);

-- ==============================================================================
-- Verification Query
-- ==============================================================================
-- SELECT id, name, job_type, cron, enabled FROM job_schedules WHERE user_id = 2;
-- ==============================================================================
