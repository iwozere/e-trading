-- SQL script to schedule P05 AI Selector
-- User ID: 2 (akossyrev@gmail.com, Telegram: 859865894)
--
-- Runs at 10:00 UTC on weekdays only (after P18 at 07:00 UTC and P15 daily at
-- 13:00 UTC the prior day — all signal inputs are available by this time).
--
-- Scheduler result fields (from run_p05_scan.py __SCHEDULER_RESULT__ output):
--   success, pick_count, p18_signals_count, notification_override,
--   trigger_reason, top_ticker, top_confidence, stage1_out, stage2_out,
--   llm_tokens_used, results_dir, timestamp, user_id
--
-- Notification logic (dual OR):
--   Primary : p18_signals_count >= 1  (P18 flagged institutional flow today)
--   Override: notification_override == 1  (LLM confidence >= 9 for any pick)
--   Either condition independently triggers Telegram + email.
--
-- Usage:
--   psql -d your_database < bin/scheduler/insert_p05_schedules.sql

-- ==============================================================================
-- 1. P05 AI Selector — Daily Scan
-- ==============================================================================
-- Stage 1: Liquidity + momentum pre-filter across Russell 3000 + top-20 crypto
--          (~3,020 symbols → ~200 candidates). Cold-start can take 20–40 min on
--          first run; subsequent runs are fast (DataManager gap-fill only).
-- Stage 2: Deterministic composite scoring — technicals + fundamentals + P18
--          signal boosts (200 → top-25).
-- Stage 3: Claude claude-sonnet-4-6 synthesises top-5 picks with full exit
--          strategies (entry/hold/breakers/profit targets). ~$0.05/run.
-- Stage 4: Writes results/p05_ai_selector/{date}/ and triggers notifications.
-- Timeout: 7200s (2 hours) — covers cold-start OHLCV fetch on first-ever run.
-- ==============================================================================

INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'P05 AI Selector Daily',
    'data_processing',
    'src.ml.pipeline.p05_ai_selector.run_p05_scan',
    '{
        "script_path": "src/ml/pipeline/p05_ai_selector/run_p05_scan.py",
        "script_args": [],
        "timeout_seconds": 7200,
        "notification_rules": {
            "conditions": [
                {
                    "check_field": "p18_signals_count",
                    "operator": ">=",
                    "threshold": 1,
                    "channels": ["telegram", "email"],
                    "comment": "P18 detected institutional flow today — AI picks are more actionable"
                },
                {
                    "check_field": "notification_override",
                    "operator": "==",
                    "threshold": 1,
                    "channels": ["telegram", "email"],
                    "comment": "LLM assigned confidence >= 9 for at least one pick — high-conviction signal"
                }
            ]
        }
    }'::jsonb,
    '0 10 * * 1-5',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
);

-- ==============================================================================
-- Verification Query
-- ==============================================================================
-- SELECT id, name, job_type, target, cron, enabled FROM job_schedules WHERE user_id = 2 ORDER BY id DESC LIMIT 5;
-- ==============================================================================
