# Trading Bot Configuration (Current)

This document describes the effective config shape consumed by `StrategyInstance` and `BaseTradingBot`.

Ready-to-copy example file:

- `config/trading/example_current_paper_bot.json`
- `config/trading/example_current_live_ibkr_bot.json`

## Minimal bot manifest

```json
{
  "id": "101",
  "name": "BTC paper bot",
  "user_id": 12,
  "enabled": true,
  "symbol": "BTCUSDT",
  "broker": {
    "type": "binance",
    "trading_mode": "paper",
    "cash": 10000
  },
  "strategy": {
    "type": "CustomStrategy",
    "parameters": {}
  },
  "data": {
    "data_source": "binance",
    "symbol": "BTCUSDT",
    "interval": "1h",
    "lookback_bars": 500
  },
  "notifications": {
    "position_opened": true,
    "position_closed": true,
    "error_notifications": true,
    "email_enabled": true,
    "telegram_enabled": true,
    "notify_email": null,
    "notify_telegram_chat_id": null
  }
}
```

## Key sections

- `broker`
  - `type`: broker adapter type (`binance`, `ibkr`, `backtrader`, etc.)
  - `trading_mode`: `paper` or `live`
  - `cash`: starting balance for paper/backtest-style modes
- `strategy`
  - `type`: registered strategy class key
  - `parameters`: strategy-specific payload
- `data`
  - runtime feed parameters used by `DataFeedFactory`
- `notifications`
  - `position_opened` / `position_closed`: owner trade alerts
  - `error_notifications`: owner error alerts
  - `email_enabled` / `telegram_enabled`: channel gating
  - `notify_email` / `notify_telegram_chat_id`: direct target overrides

## Notification routing rules

- Trade alerts: owner only
- Lifecycle alerts: owner only
- Error alerts:
  - owner if `error_notifications=true`
  - admins always

Owner target resolution:

1. `notifications.notify_*`
2. `TRADING_NOTIFY_EMAIL` / `TRADING_NOTIFY_TELEGRAM_CHAT_ID`
3. DB lookup via `user_id`

## Notes

- `symbol` should align across top-level `symbol`, `data.symbol`, and strategy assumptions.
- In paper mode, no live broker order is placed, but trades are persisted and notifications are still emitted.
