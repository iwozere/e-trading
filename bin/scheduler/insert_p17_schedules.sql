-- SQL script to schedule P17 Explosive Penny Stock Screener
-- User ID: 2 (akossyrev@gmail.com, Telegram: 859865894)
--
-- Runs at 06:00 UTC Mon–Fri — overnight slot, 8 hours after NYSE/NASDAQ close.
-- Previous-day OHLCV and NASDAQ FTP universe are fully settled by this time.
-- Results are ready 3.5 hours before the 9:30 AM ET open.
--
-- Scheduler result fields (from run_p17.py __SCHEDULER_RESULT__ output):
--   success, total_candidates, tier_a_count, tier_b_count,
--   tier_c_count, explosive_count, results_dir, timestamp
--
-- Idempotent: safe to re-run (ON CONFLICT (user_id, name) DO NOTHING).
-- Usage:
--   psql -d your_database < bin/scheduler/insert_p17_schedules.sql

-- ==============================================================================
-- 1. P17 Penny Stock Screener — Daily
-- ==============================================================================
-- Downloads NASDAQ universe via FTP, enriches with yfinance OHLCV (90d),
-- fundamentals, EDGAR dilution filings, short-squeeze data, computes composite
-- scores, assigns tiers A/B/C/W, and generates CSV + JSON + Markdown reports.
-- Timeout: 2 hours (generous for 200–400 tickers + EDGAR lookups).
-- ==============================================================================

INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'P17 Penny Stock Screener Daily',
    'data_processing',
    'src.ml.pipeline.p17_penny_stocks.run_p17',
    '{
        "script_path": "src/ml/pipeline/p17_penny_stocks/run_p17.py",
        "script_args": [],
        "timeout_seconds": 7200,
        "notification_rules": {
            "conditions": [
                {
                    "check_field": "tier_a_count",
                    "operator": ">",
                    "threshold": 0,
                    "channels": ["email"],
                    "comment": "Email when Tier A (elite) candidates found"
                },
                {
                    "check_field": "explosive_count",
                    "operator": ">",
                    "threshold": 0,
                    "channels": ["email", "telegram"],
                    "comment": "Email + Telegram when explosive candidates found"
                }
            ]
        }
    }'::jsonb,
    '0 6 * * 1-5',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
) ON CONFLICT (user_id, name) DO NOTHING;

-- ==============================================================================
-- Verification Query
-- ==============================================================================
-- SELECT id, name, job_type, cron, enabled FROM job_schedules WHERE user_id = 2;
-- ==============================================================================
