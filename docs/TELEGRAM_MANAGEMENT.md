# Telegram Management Module

This module provides enterprise-grade Telegram integration for trading, analytics, portfolio management, notifications, and advanced features such as voice commands and sentiment analysis.

## Directory Structure

```
management/telegram/
├── commands/
│   ├── trading_commands.py      # Start/stop/status
│   ├── analysis_commands.py     # Charts, reports
│   ├── portfolio_commands.py    # Portfolio overview
│   └── admin_commands.py        # Administrative functions
├── notifications/
│   ├── trade_alerts.py          # Trade notifications
│   ├── risk_alerts.py           # Risk warnings
│   ├── performance_reports.py   # Daily/weekly reports
│   └── market_updates.py        # Market news and analysis
└── integrations/
    ├── voice_commands.py        # Voice message processing
    ├── chart_generator.py       # Automatic chart generation
    └── sentiment_analysis.py    # Message sentiment
```

## Components

### Commands
- **trading_commands.py**: Start, stop, and check the status of trading bots via Telegram commands.
- **analysis_commands.py**: Request charts and reports directly from Telegram.
- **portfolio_commands.py**: Get portfolio overviews and summaries.
- **admin_commands.py**: Administrative functions such as adding/removing admins.

### Notifications
- **trade_alerts.py**: Receive trade execution notifications.
- **risk_alerts.py**: Get risk warnings and alerts.
- **performance_reports.py**: Receive daily or weekly performance summaries.
- **market_updates.py**: Get market news and analysis updates.

### Integrations
- **voice_commands.py**: Process and interpret voice messages for trading commands.
- **chart_generator.py**: Automatically generate and send charts.
- **sentiment_analysis.py**: Analyze the sentiment of messages and chats.

## Usage
Each module provides clear function interfaces and can be integrated into Telegram bot workflows for robust, enterprise-grade communication and automation.

## Integration Guidance

To integrate the Telegram management module into your bot or trading system:

1. **Import the relevant modules** in your Telegram bot scripts:

```python
from src.management.telegram.commands import trading_commands, analysis_commands, portfolio_commands, admin_commands
from src.management.telegram.notifications import trade_alerts, risk_alerts, performance_reports, market_updates
from src.management.telegram.integrations import voice_commands, chart_generator, sentiment_analysis
```

2. **Command Handling:**
   - Use `trading_commands` to start, stop, or check the status of trading bots from Telegram.
   - Use `analysis_commands` to send charts or reports to users.
   - Use `portfolio_commands` to provide portfolio overviews.
   - Use `admin_commands` for admin management.

3. **Notifications:**
   - Use `trade_alerts` to notify users of trade executions.
   - Use `risk_alerts` to send risk warnings.
   - Use `performance_reports` for daily/weekly summaries.
   - Use `market_updates` for market news and analysis.

4. **Advanced Integrations:**
   - Use `voice_commands` to process and interpret voice messages.
   - Use `chart_generator` to generate and send charts automatically.
   - Use `sentiment_analysis` to analyze the sentiment of user messages.

See `examples/telegram_management_example.py` for a complete usage demonstration.

## Customization & Extension

The Telegram management module is designed for easy extension and customization. Here are some ways to tailor it to your needs:

### 1. Adding New Commands
- Create a new Python file in `management/telegram/commands/` (e.g., `custom_commands.py`).
- Define your command functions with clear docstrings and interfaces.
- Register these commands in your Telegram bot handler (e.g., using python-telegram-bot's `CommandHandler`).

### 2. Adding New Notifications
- Add a new file in `management/telegram/notifications/` for your notification type.
- Implement a function to send the notification, accepting relevant parameters (user_id, data, etc.).
- Call this function from your trading or risk logic when an event occurs.

### 3. Adding New Integrations
- Place new integration modules in `management/telegram/integrations/` (e.g., for AI features, external APIs, etc.).
- Follow the pattern of clear function interfaces and docstrings.

### 4. Adapting to Bot Frameworks
- **python-telegram-bot:**
  - Use `CommandHandler` to map Telegram commands to your module functions.
  - Use `bot.send_message(chat_id, ...)` to send responses from your functions.
- **aiogram:**
  - Use `@dp.message_handler(commands=[...])` decorators to bind commands.
  - Use `await message.answer(...)` to reply using your module's output.

### 5. Example: Registering a Command (python-telegram-bot)
```python
from telegram.ext import CommandHandler
from src.management.telegram.commands import trading_commands

def start(update, context):
    user_id = update.effective_user.id
    msg = trading_commands.start_trading(user_id)
    context.bot.send_message(chat_id=update.effective_chat.id, text=msg)

# In your dispatcher setup:
dispatcher.add_handler(CommandHandler('start_trading', start))
```

### 6. Example: Sending a Notification
```python
from src.management.telegram.notifications import trade_alerts
trade_info = {'symbol': 'BTCUSDT', 'side': 'buy', 'price': 30000, 'qty': 0.1}
msg = trade_alerts.send_trade_alert(user_id, trade_info)
bot.send_message(chat_id, msg)
```

### 7. Extending Integrations
- Add new AI, analytics, or automation features by following the integration module pattern.
- For example, add a `news_sentiment.py` for real-time news analysis and alerts.

For more advanced customization, subclass or wrap the provided functions to add logging, error handling, or asynchronous support as needed. 