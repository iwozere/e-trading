-- SQL script to insert scheduled jobs for P10 (EMPS3) pipeline
-- Runs 2 hours BEFORE EMPS2
-- User ID: 2 (akossyrev@gmail.com)

-- ==============================================================================
-- 1. EMPS3 Morning Scan
-- ==============================================================================
-- Runs daily at 07:00 UTC on weekdays and Saturday
-- Notifies via email for Phase 1, via email+telegram for Phase 2
-- ==============================================================================

INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'EMPS3 Morning Scan',
    'data_processing',
    'src.ml.pipeline.p10_emps3.run_emps3_scan',
    '{
        "script_path": "src/ml/pipeline/p10_emps3/run_emps3_scan.py",
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
    '0 7 * * 1-6',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
);

-- ==============================================================================
-- 2. EMPS3 Evening Scan
-- ==============================================================================
-- Runs daily at 18:00 UTC (6:00 PM UTC) on weekdays
-- Notifies via email for Phase 1, via email+telegram for Phase 2
-- ==============================================================================

INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'EMPS3 Mid-Day Scan',
    'data_processing',
    'src.ml.pipeline.p10_emps3.run_emps3_scan',
    '{
        "script_path": "src/ml/pipeline/p10_emps3/run_emps3_scan.py",
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
    '0 18 * * 1-5',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
);
