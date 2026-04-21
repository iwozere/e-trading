-- =============================================================================
-- Trading Strategy Pack (SP-1..SP-4) scheduled jobs
-- =============================================================================
-- Registers the daily / weekly / monthly / quarterly strategies from
-- src/strategy_pack/ as data_processing jobs in the `job_schedules` table.
-- Intraday/bar-close rows (SP-5, SP-6) are in
-- `bin/scheduler/insert_strategy_pack_intraday_schedules.sql`.
--
-- User ID: 2  (akossyrev@gmail.com). Change before applying to another env.
--
-- Each row uses its own dedicated config file under
-- `config/strategy_pack/schedules/` so symbols/timeframes/parameters are
-- per-schedule and not coupled to `config/strategy_pack/default.json`.
--
-- Execution model:
--   SchedulerService._execute_data_processing_job runs
--       python <PROJECT_ROOT>/src/strategy_pack/run.py run -s <N> -v A \
--              -c <schedule-config.json> --user-id 2
--   (the scheduler auto-appends --user-id from job_schedules.user_id).
--
-- The pack sends its own notifications via NotificationServiceClient, so
-- task_params.notification_rules is intentionally omitted (the scheduler
-- would otherwise double-notify). Dedup is handled by DedupStore.
--
-- DBeaver: to execute every statement in this file, use "Execute SQL Script"
--          (Alt+X) or select-all + Ctrl+Enter. Plain Ctrl+Enter runs only the
--          statement under the cursor.
--
-- After applying, either wait for the `scheduler_updates` LISTEN/NOTIFY reload
-- or restart the scheduler service for the new rows to be picked up.
-- =============================================================================


-- -----------------------------------------------------------------------------
-- SP-1 Momentum-Growth Portfolio (monthly)
-- Cadence: 1st of each month at 01:00 UTC (after prior-day daily bars are
-- final for both US equities and crypto).
-- -----------------------------------------------------------------------------
INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'Strategy Pack SP-1 Monthly Momentum',
    'data_processing',
    'src.strategy_pack',
    '{
        "script_path": "src/strategy_pack/run.py",
        "script_args": [
            "run",
            "-s", "1",
            "-v", "A",
            "-c", "config/strategy_pack/schedules/sp1_monthly.json"
        ],
        "timeout_seconds": 1800
    }'::jsonb,
    '0 1 1 * *',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
);


-- -----------------------------------------------------------------------------
-- SP-2 Daily Trend (SMA/EMA)
-- Cadence: every day at 22:30 UTC (~15 min past US RTH close on DST;
-- covers winter close as well since the signal is computed on the latest
-- finalized daily bar, not mid-session).
-- -----------------------------------------------------------------------------
INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'Strategy Pack SP-2 Daily Trend',
    'data_processing',
    'src.strategy_pack',
    '{
        "script_path": "src/strategy_pack/run.py",
        "script_args": [
            "run",
            "-s", "2",
            "-v", "A",
            "-c", "config/strategy_pack/schedules/sp2_daily.json"
        ],
        "timeout_seconds": 900
    }'::jsonb,
    '30 22 * * *',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
);


-- -----------------------------------------------------------------------------
-- SP-3 Weekly Lazy Trend
-- Cadence: once a week, Sunday 22:00 UTC (well after crypto weekly close
-- and after US equities' Friday close settles).
-- -----------------------------------------------------------------------------
INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'Strategy Pack SP-3 Weekly Lazy',
    'data_processing',
    'src.strategy_pack',
    '{
        "script_path": "src/strategy_pack/run.py",
        "script_args": [
            "run",
            "-s", "3",
            "-v", "A",
            "-c", "config/strategy_pack/schedules/sp3_weekly.json"
        ],
        "timeout_seconds": 900
    }'::jsonb,
    '0 22 * * 0',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
);


-- -----------------------------------------------------------------------------
-- SP-4 Buy-and-Hold Rebalance Advisory
-- Cadence: quarterly, 1st of Jan/Apr/Jul/Oct at 02:00 UTC.
-- -----------------------------------------------------------------------------
INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'Strategy Pack SP-4 Quarterly Rebalance',
    'data_processing',
    'src.strategy_pack',
    '{
        "script_path": "src/strategy_pack/run.py",
        "script_args": [
            "run",
            "-s", "4",
            "-v", "A",
            "-c", "config/strategy_pack/schedules/sp4_quarterly.json"
        ],
        "timeout_seconds": 1800
    }'::jsonb,
    '0 2 1 1,4,7,10 *',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
);
