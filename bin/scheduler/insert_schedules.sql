-- SQL script to insert 4 scheduled jobs for VIX, EMPS2, and Fundamentals
-- User ID: 2 (akossyrev@gmail.com, Telegram: 859865894)
--
-- Usage:
--   psql -d your_database < bin/scheduler/insert_schedules.sql
--   OR
--   sqlite3 your_database.db < bin/scheduler/insert_schedules.sql

-- ==============================================================================
-- 1. VIX Daily Monitor
-- ==============================================================================
-- Runs daily at 9:30 AM ET (after market open) on weekdays
-- Notifies via email if VIX > 20, via email+telegram if VIX > 25
-- ==============================================================================

INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'VIX Daily Monitor',
    'data_processing',
    'src.data.vix',
    '{
        "script_path": "src/data/vix.py",
        "script_args": [],
        "timeout_seconds": 600,
        "notification_rules": {
            "conditions": [
                {
                    "check_field": "vix_current",
                    "operator": ">=",
                    "threshold": 20,
                    "channels": ["email"],
                    "comment": "Email notification when VIX >= 20"
                },
                {
                    "check_field": "vix_current",
                    "operator": ">=",
                    "threshold": 25,
                    "channels": ["email", "telegram"],
                    "comment": "Email + Telegram when VIX >= 25"
                }
            ]
        }
    }'::jsonb,
    '30 9 * * 1-5',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
);

-- ==============================================================================
-- 2. EMPS2 Morning Scan (Market Open)
-- ==============================================================================
-- Runs daily at 9:35 AM ET on weekdays (5 minutes after market open)
-- Notifies via email for Phase 1, via email+telegram for Phase 2
-- Timeout: 4 hours (data download may take time)
-- ==============================================================================

INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'EMPS2 Morning Scan',
    'data_processing',
    'src.ml.pipeline.p06_emps2.run_emps2_scan',
    '{
        "script_path": "src/ml/pipeline/p06_emps2/run_emps2_scan.py",
        "script_args": [],
        "timeout_seconds": 14400,
        "notification_rules": {
            "conditions": [
                {
                    "check_field": "phase1_count",
                    "operator": ">",
                    "threshold": 0,
                    "channels": ["email"],
                    "comment": "Email notification for Phase 1 candidates"
                },
                {
                    "check_field": "phase2_count",
                    "operator": ">",
                    "threshold": 0,
                    "channels": ["email", "telegram"],
                    "comment": "Email + Telegram for Phase 2 candidates"
                }
            ]
        }
    }'::jsonb,
    '35 9 * * 1-5',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
);

-- ==============================================================================
-- 3. EMPS2 Evening Scan (8 PM CET)
-- ==============================================================================
-- Runs daily at 8:00 PM CET on weekdays
-- Notifies via email for Phase 1, via email+telegram for Phase 2
-- Timeout: 4 hours (data download may take time)
--
-- Note: 8 PM CET = 2 PM ET (during standard time) or 1 PM ET (during daylight saving)
-- Using 14:00 ET for simplicity (adjust as needed for DST)
-- ==============================================================================

INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'EMPS2 Evening Scan (8PM CET)',
    'data_processing',
    'src.ml.pipeline.p06_emps2.run_emps2_scan',
    '{
        "script_path": "src/ml/pipeline/p06_emps2/run_emps2_scan.py",
        "script_args": [],
        "timeout_seconds": 14400,
        "notification_rules": {
            "conditions": [
                {
                    "check_field": "phase1_count",
                    "operator": ">",
                    "threshold": 0,
                    "channels": ["email"],
                    "comment": "Email notification for Phase 1 candidates"
                },
                {
                    "check_field": "phase2_count",
                    "operator": ">",
                    "threshold": 0,
                    "channels": ["email", "telegram"],
                    "comment": "Email + Telegram for Phase 2 candidates"
                }
            ]
        }
    }'::jsonb,
    '0 14 * * 1-5',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
);

-- ==============================================================================
-- 4. Fundamentals Cache Refresh (Weekend)
-- ==============================================================================
-- Runs every Saturday at 2:00 AM UTC
-- Notifies via email on completion (success or error)
-- ==============================================================================

INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'Fundamentals Cache Refresh',
    'data_processing',
    'src.data.utils.refresh_fundamentals_cache',
    '{
        "script_path": "src/data/utils/refresh_fundamentals_cache.py",
        "script_args": [],
        "timeout_seconds": 3600,
        "notification_rules": {
            "conditions": [
                {
                    "check_field": "successful_symbols",
                    "operator": ">=",
                    "threshold": 0,
                    "channels": ["email"],
                    "comment": "Email notification on completion"
                }
            ]
        }
    }'::jsonb,
    '0 2 * * 6',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
);

-- ==============================================================================
-- Verification Query
-- ==============================================================================
-- Run this to verify the schedules were inserted correctly:
-- SELECT id, name, job_type, cron, enabled FROM job_schedules WHERE user_id = 2;
-- ==============================================================================
