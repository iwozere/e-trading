-- SQL script to insert weekly screener jobs for S&P 500 and SIX (Swiss Exchange)
-- User ID: 2 (akossyrev@gmail.com, Telegram: 859865894)
--
-- Run schedule: every Saturday at 07:00 UTC (S&P 500) and 07:30 UTC (SIX)
-- Both markets are closed on Saturday, so all weekly data is settled.
-- Results are available before Monday's open in any timezone.
--
-- Notification: Telegram + Email when result_count >= 1 (no empty-result spam)
-- Email receives the full results table (sp500_selected_stocks.csv / six_selected_stocks.csv)
-- Telegram receives a summary with result count and top tickers.
--
-- Usage:
--   psql -d your_database < bin/scheduler/insert_screener_schedules.sql
--   OR
--   sqlite3 your_database.db < bin/scheduler/insert_screener_schedules.sql

-- ==============================================================================
-- 1. S&P 500 Weekly Screener
-- ==============================================================================
-- Screens ~503 S&P 500 tickers via yfinance for:
--   P/E < 25, ROE > 15%, D/E < 100%, FCF > 0, price > 50D MA > 200D MA
-- Saves results to sp500_selected_stocks.csv
-- Timeout: 3600s (1 hour) — yfinance makes one API call per ticker
-- ==============================================================================

INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'S&P 500 Weekly Screener',
    'data_processing',
    'src.screeners.sp500_stock_screener',
    '{
        "script_path": "src/screeners/sp500_stock_screener.py",
        "script_args": [],
        "timeout_seconds": 3600,
        "notification_rules": {
            "conditions": [
                {
                    "check_field": "result_count",
                    "operator": ">=",
                    "threshold": 1,
                    "channels": ["telegram", "email"],
                    "comment": "Telegram + Email when at least 1 stock passes all criteria"
                }
            ]
        }
    }'::jsonb,
    '0 7 * * 6',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
);

-- ==============================================================================
-- 2. SIX (Swiss Exchange) Weekly Screener
-- ==============================================================================
-- Screens Swiss-listed tickers via yfinance for:
--   P/E < 25, ROE > 15%, D/E < 100%, FCF > 0, price > 50D MA > 200D MA
-- Saves results to six_selected_stocks.csv
-- Staggered 30 min after S&P 500 run to avoid concurrent yfinance load.
-- Timeout: 1800s (30 min) — fewer tickers than S&P 500
-- ==============================================================================

INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'SIX Weekly Screener',
    'data_processing',
    'src.screeners.six_stock_screener',
    '{
        "script_path": "src/screeners/six_stock_screener.py",
        "script_args": [],
        "timeout_seconds": 1800,
        "notification_rules": {
            "conditions": [
                {
                    "check_field": "result_count",
                    "operator": ">=",
                    "threshold": 1,
                    "channels": ["telegram", "email"],
                    "comment": "Telegram + Email when at least 1 stock passes all criteria"
                }
            ]
        }
    }'::jsonb,
    '30 7 * * 6',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
);

-- ==============================================================================
-- Verification Query
-- ==============================================================================
-- Run this to verify the schedules were inserted correctly:
-- SELECT id, name, job_type, cron, enabled FROM job_schedules WHERE user_id = 2 AND name LIKE '%Screener%';
-- ==============================================================================
