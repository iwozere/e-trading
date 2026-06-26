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
-- Notification logic:
--   The P05 pipeline now sends its OWN rich notifications from Stage 4 (condensed
--   Telegram text + full HTML email with market context, per-pick exit strategies,
--   and top_picks.csv / report.md attached). It applies the dual OR-trigger
--   internally (p18_signals_count >= 1 OR an LLM pick with confidence >= 9).
--   Therefore this schedule intentionally defines NO notification_rules — leaving
--   them in would make the scheduler ALSO emit a generic key:value result dump,
--   duplicating (and uglifying) every notification. See Stage4Output.format_* and
--   P05Pipeline._send_notifications.
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
-- Stage 4: Writes results/p05_ai_selector/{date}/ and sends its own Telegram +
--          email notifications (no scheduler notification_rules — see header note).
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
        "timeout_seconds": 7200
    }'::jsonb,
    '0 10 * * 1-5',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
);

-- ==============================================================================
-- Migration for an already-deployed schedule
-- ==============================================================================
-- If the P05 schedule was inserted previously WITH notification_rules, run this
-- once to stop the duplicate generic result-dump now that the pipeline
-- self-notifies. It strips only the notification_rules key, leaving timeout etc.
--
--   UPDATE job_schedules
--      SET task_params = task_params - 'notification_rules',
--          updated_at  = CURRENT_TIMESTAMP
--    WHERE user_id = 2 AND name = 'P05 AI Selector Daily';
-- ==============================================================================

-- ==============================================================================
-- Verification Query
-- ==============================================================================
-- SELECT id, name, job_type, target, cron, enabled FROM job_schedules WHERE user_id = 2 ORDER BY id DESC LIMIT 5;
-- ==============================================================================
