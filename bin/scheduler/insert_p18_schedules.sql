-- SQL script to schedule P18 Institutional Flow Tracker
-- User ID: 2 (akossyrev@gmail.com, Telegram: 859865894)
--
-- Runs at 07:00 UTC every day (including weekends — EDGAR filings arrive daily
-- during 45-day filing windows; Form 4 filings arrive every business day).
--
-- Scheduler result fields (from run_p18_scan.py __SCHEDULER_RESULT__ output):
--   success, high_score_count, new_13f_filings_today, form4_sells_count,
--   top_ticker, top_score, results_dir, timestamp
--
-- Usage:
--   psql -d your_database < bin/scheduler/insert_p18_schedules.sql

-- ==============================================================================
-- 1. P18 Institutional Flow — Daily Scan
-- ==============================================================================
-- Checks for new 13F-HR filings today, updates position-delta consensus,
-- downloads Form 4 insider sells and Schedule 13D/G amendments for yesterday,
-- runs volume anomaly detection on the current watchlist, and scores all tickers.
-- Sends a Telegram alert when any ticker reaches composite score >= 60.
-- Timeout: 3600s (1 hour) — covers EDGAR bulk download during filing windows.
-- ==============================================================================

INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'P18 Institutional Flow Daily',
    'data_processing',
    'src.ml.pipeline.p18_institutional_flow_tracker.run_p18_scan',
    '{
        "script_path": "src/ml/pipeline/p18_institutional_flow_tracker/run_p18_scan.py",
        "script_args": [],
        "timeout_seconds": 3600,
        "notification_rules": {
            "conditions": [
                {
                    "check_field": "high_score_count",
                    "operator": ">",
                    "threshold": 0,
                    "channels": ["telegram"],
                    "comment": "Telegram alert when any ticker scores >= 60 (institutional distribution signal)"
                },
                {
                    "check_field": "high_score_count",
                    "operator": ">",
                    "threshold": 2,
                    "channels": ["email", "telegram"],
                    "comment": "Email + Telegram when 3+ tickers simultaneously flagged (broad distribution wave)"
                }
            ]
        }
    }'::jsonb,
    '0 7 * * *',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
);

-- ==============================================================================
-- Verification Query
-- ==============================================================================
-- SELECT id, name, job_type, target, cron, enabled FROM job_schedules WHERE user_id = 2 ORDER BY id DESC LIMIT 5;
-- ==============================================================================
