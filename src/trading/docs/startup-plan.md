# Startup Plan (Current)

This guide describes the current startup model for trading services.

## Modes

- Service mode (recommended): DB-driven multi-bot runtime
- Single-config mode: one config file through wrapper scripts

## 1) Service mode (DB-driven)

Entry point: `src/trading/trading_runner.py`

Typical command:

```powershell
python src/trading/trading_runner.py --user-id 12 --poll-interval 60
```

Optional:

- `--no-resume`: disable crash-recovery resume behavior

What happens:

1. load enabled bot configs from DB
2. validate + hydrate each config
3. create/start `StrategyInstance`s
4. start monitor loop and DB polling loop

## 2) Single-config mode

Entry points:

- `src/trading/trading_bot.py <config.json>`
- `src/trading/live_trading_bot.py <config.json>`

Example:

```powershell
python src/trading/trading_bot.py binance_paper_trading.json
```

## 3) Paper-trading first checklist

1. set `broker.trading_mode` to `paper`
2. verify data source connectivity
3. verify owner notification routing:
   - `user_id` mapping, or
   - direct `notify_email` / `notify_telegram_chat_id`
4. run in service mode and confirm:
   - strategy starts cleanly
   - signals produce simulated trades
   - owner gets trade alerts
   - owner + admins get error alerts

## 4) Live rollout checklist

1. run stable in paper for sufficient time window
2. confirm risk settings and limits
3. switch `broker.trading_mode` to `live`
4. deploy with one bot first
5. monitor notifications, execution, and risk logs

## 5) Operations

- stop gracefully with `Ctrl+C` so shutdown markers and statuses persist cleanly
- use logs for diagnostics; stale heartbeat triggers recovery attempts
- keep notification worker/API processing running so queued messages are delivered
