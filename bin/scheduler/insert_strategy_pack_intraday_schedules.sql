-- =============================================================================
-- Trading Strategy Pack (SP-5, SP-6) scheduled jobs
-- =============================================================================
-- Registers the intraday / bar-close strategies from src/strategy_pack/ as
-- data_processing jobs in the `job_schedules` table. Complements the
-- daily/weekly/monthly/quarterly rows in
-- `bin/scheduler/insert_strategy_pack_schedules.sql` (SP-1..SP-4).
--
-- User ID: 2  (akossyrev@gmail.com). Change before applying to another env.
--
-- Decision model:
--   Notification-only pack; scheduler fires a few minutes AFTER candle close so
--   the bar is final. Each row uses its own dedicated config file under
--   `config/strategy_pack/schedules/` so symbol/timeframe are per-schedule and
--   not coupled to `config/strategy_pack/default.json`.
--
-- Execution model:
--   SchedulerService._execute_data_processing_job runs
--       python <PROJECT_ROOT>/src/strategy_pack/run.py run -s <N> -v <V> \
--              -c <schedule-config.json> --user-id 2
--   (the scheduler auto-appends --user-id from job_schedules.user_id).
--
--   The pack sends its own notifications via NotificationServiceClient, so
--   task_params.notification_rules is intentionally omitted (the scheduler
--   would otherwise double-notify). Dedup is handled by DedupStore.
--
-- Providers:
--   * BTCUSDT -> Binance klines (via DataManager / provider_rules.yaml).
--   * SPY     -> yfinance 1h bars. yfinance's intraday history cap (~60 days
--                for 1h) is fine because lookback_days=45 in the config.
--
-- DST caveat (SP-5 SPY 1h):
--   Cron runs in UTC. US RTH hourly bars close at 14:30-21:00 UTC (DST) and
--   15:30-22:00 UTC (winter). We schedule the poll 02 minutes past each hour
--   across 14-22 UTC so both regimes are covered; a few off-hour runs on the
--   edges will just report "no new bar" and exit cleanly.
--
-- After applying, either wait for the `scheduler_updates` LISTEN/NOTIFY reload
-- or restart the scheduler service for the new rows to be picked up.
-- =============================================================================


-- -----------------------------------------------------------------------------
-- SP-5 Swing on BTC/USDT, 4h (Donchian breakout + volume, Variant A)
-- Cadence: 02 min past each 4h bar close -> 00/04/08/12/16/20 UTC.
-- -----------------------------------------------------------------------------
INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'Strategy Pack SP-5 Swing BTC 4h',
    'data_processing',
    'src.strategy_pack',
    '{
        "script_path": "src/strategy_pack/run.py",
        "script_args": [
            "run",
            "-s", "5",
            "-v", "A",
            "-c", "config/strategy_pack/schedules/sp5_btc_4h.json"
        ],
        "timeout_seconds": 600
    }'::jsonb,
    '2 0,4,8,12,16,20 * * *',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
);


-- -----------------------------------------------------------------------------
-- SP-5 Swing on BTC/USDT, 1h (Donchian breakout + volume, Variant A)
-- Cadence: 02 min past every hour (24/7).
-- -----------------------------------------------------------------------------
INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'Strategy Pack SP-5 Swing BTC 1h',
    'data_processing',
    'src.strategy_pack',
    '{
        "script_path": "src/strategy_pack/run.py",
        "script_args": [
            "run",
            "-s", "5",
            "-v", "A",
            "-c", "config/strategy_pack/schedules/sp5_btc_1h.json"
        ],
        "timeout_seconds": 600
    }'::jsonb,
    '2 * * * *',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
);


-- -----------------------------------------------------------------------------
-- SP-5 Swing on SPY, 1h (Donchian breakout + volume, Variant A)
-- Cadence: 02 min past each hour 14..22 UTC, Mon-Fri (covers US RTH
-- in both DST and winter time).
-- -----------------------------------------------------------------------------
INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'Strategy Pack SP-5 Swing SPY 1h',
    'data_processing',
    'src.strategy_pack',
    '{
        "script_path": "src/strategy_pack/run.py",
        "script_args": [
            "run",
            "-s", "5",
            "-v", "A",
            "-c", "config/strategy_pack/schedules/sp5_spy_1h.json"
        ],
        "timeout_seconds": 600
    }'::jsonb,
    '2 14-22 * * 1-5',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
);


-- -----------------------------------------------------------------------------
-- SP-6 EMA + SuperTrend on BTC/USDT, 4h
-- Cadence: 02 min past each 4h bar close -> 00/04/08/12/16/20 UTC.
-- -----------------------------------------------------------------------------
INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'Strategy Pack SP-6 EMA+SuperTrend BTC 4h',
    'data_processing',
    'src.strategy_pack',
    '{
        "script_path": "src/strategy_pack/run.py",
        "script_args": [
            "run",
            "-s", "6",
            "-v", "A",
            "-c", "config/strategy_pack/schedules/sp6_btc_4h.json"
        ],
        "timeout_seconds": 600
    }'::jsonb,
    '2 0,4,8,12,16,20 * * *',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
);


-- -----------------------------------------------------------------------------
-- SP-6 EMA + SuperTrend on BTC/USDT, 1d
-- Cadence: 22:30 UTC daily (same slot as SP-2, well after 00:00 UTC daily close
-- ... actually a full day later; crypto 1d bar closes at 00:00 UTC, so we fire
-- at 00:05 UTC instead). Using 00:05 UTC for a clean "right after close" fire.
-- -----------------------------------------------------------------------------
INSERT INTO job_schedules (user_id, name, job_type, target, task_params, cron, enabled, created_at, updated_at)
VALUES (
    2,
    'Strategy Pack SP-6 EMA+SuperTrend BTC 1d',
    'data_processing',
    'src.strategy_pack',
    '{
        "script_path": "src/strategy_pack/run.py",
        "script_args": [
            "run",
            "-s", "6",
            "-v", "A",
            "-c", "config/strategy_pack/schedules/sp6_btc_1d.json"
        ],
        "timeout_seconds": 600
    }'::jsonb,
    '5 0 * * *',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
);
