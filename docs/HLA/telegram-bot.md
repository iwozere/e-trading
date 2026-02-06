# Telegram Screener Bot: High-Level Architecture & Usage

## Overview
The Telegram Screener Bot is a command-driven interface for trading analytics, ticker management, and automated reporting. It integrates with Yahoo Finance and Binance to provide real-time technical and fundamental analysis.

## Architecture
The bot follows a two-tier processing model to ensure responsiveness:
1.  **Immediate Processing**: Interactive commands (registration, info, settings) are handled directly by the bot without queueing.
2.  **Queued Processing**: Computationally heavy tasks (reports, screening) use the Notification Service to manage execution and handle email delivery with attachments.

---

## Command Reference

### User Management & Security
Basic account setup and security. Commands are processed immediately.

- `/register email@example.com` - Register or update your email for reports.
- `/verify CODE` - Verify your email using the 6-digit code sent to you.
- `/info` - View your registration status, email, and approval level.
- `/request_approval` - Request admin approval for restricted features (requires verified email).
- `/language [en|ru]` - Set your preferred interface language.

### Ticker Analysis & Screening
Heavy processing commands that support `-email` notifications.

- `/report TICKER1 TICKER2 ... [flags]` - Unified analysis command.
  - **Flags**:
    - `-email`: Send the report and charts to your verified email.
    - `-provider=[yf|bnc]`: Specify data source (Yahoo Finance or Binance).
    - `-period=[1y|2y|3mo|...]`: Historical data range.
    - `-interval=[1d|1h|15m|...]`: Data granularity.
    - `-indicators=RSI,MACD,PE...`: Custom indicator list.
- `/screener JSON_CONFIG [-email]` - Run an advanced fundamental or technical screener immediately.

### Alerts & Schedules
Manage automated notifications and periodic reports.

- `/alerts` - List active price alerts.
- `/alerts add TICKER PRICE [above|below] [-email]` - Set a price-based notification.
- `/alerts delete ALERT_ID` - Remove an existing alert.
- `/schedules` - List all your scheduled tasks.
- `/schedules add TICKER TIME [-email] [flags]` - Schedule a daily report (TIME in HH:MM UTC).
- `/schedules screener LIST_TYPE TIME [-email]` - Schedule a periodic fundamental screener.

### Support & Feedback
- `/help` - Comprehensive command list and examples.
- `/feedback MESSAGE` - Send feedback or bug reports to developers.
- `/feature MESSAGE` - Suggest new features.

---

## Admin Commands
Restricted to authorized administrators.
- `/admin users` - List all registered users.
- `/admin approve USER_ID` - Approve a user's access request.
- `/admin broadcast MESSAGE` - Send a message to all bot users.
- `/admin setlimit [alerts|schedules] N [USER_ID]` - Manage system resource limits.

---

## Integration Details
- **Location**: `src/telegram/`
- **Main Bot**: `telegram_bot.py`
- **Logic Layer**: `screener/business_logic.py`
- **Parser**: `command_parser.py` (Enterprise-grade shlex-based parsing)
- **Email**: Requires Gmail/SMTP configuration in `config/donotshare/donotshare.py`.

---

## Data Providers
- **Yahoo Finance (`-yf`)**: Best for stocks, ETFs, and fundamental data (P/E, Market Cap).
- **Binance (`-bnc`)**: Optimized for real-time crypto pairs and specialized intervals (4h, 8h).
