# Trading Docs

This folder contains the current documentation for the trading runtime.

## Current docs

- `ibkr-gateway-config.md`: IBKR Gateway deployment and remote API checklist
- `configuration.md`: bot configuration structure used by the runtime
- `startup-plan.md`: startup and operating plan for paper/live services

## Runtime structure

Primary execution path:

1. `src/trading/trading_runner.py` (DB-driven service runner)
2. `src/trading/strategy_manager.py` (loads bots, starts/stops instances)
3. `src/trading/strategy_instance.py` (broker + data feed + strategy loop)
4. `src/trading/base_trading_bot.py` (signals, orders, persistence, notifications)

Single-config wrappers:

- `src/trading/trading_bot.py`
- `src/trading/live_trading_bot.py`

## Notification behavior

- Trade/lifecycle notifications: bot owner only
- Error notifications:
  - bot owner when `notifications.error_notifications` is enabled
  - admins always

Owner resolution priority:

1. explicit config override (`notify_email` / `notify_telegram_chat_id`)
2. environment override (`TRADING_NOTIFY_EMAIL` / `TRADING_NOTIFY_TELEGRAM_CHAT_ID`)
3. DB user mapping from `user_id` + linked identities

## Safety reminder

- Use paper mode first: `broker.trading_mode: "paper"`
- Validate each config before enabling live trading
