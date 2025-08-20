# Telegram Screener Bot — Full Documentation

## Overview

The Telegram Screener Bot is a feature-rich Telegram bot for real-time and scheduled reporting on shares and cryptocurrencies. It supports price alerts, scheduled reports, email delivery, and a web-based admin panel. The bot is designed for extensibility, robust error handling, and future multi-language support.

---

## Features

- **Share & Crypto Reports:** On-demand and scheduled reports for stocks and crypto pairs, with technical and fundamental indicators.
- **Fundamental Screener:** Automated screening for undervalued stocks across different market cap categories and custom lists.
- **Email Delivery:** Send reports and alerts to verified user emails.
- **Price Alerts:** Set, pause, resume, and delete price alerts for any supported ticker.
- **Scheduling:** Schedule recurring reports (daily/weekly/monthly) at user-defined times.
- **Admin Panel:** Manage users, alerts, schedules, and view logs/statistics via Telegram or web interface.
- **Localization-ready:** All user-facing text is localizable (English default).
- **Robust Logging:** Logs all user commands, API errors, email delivery, and admin actions (30-day retention).
- **Extensible:** Modular data provider and command handler architecture.

---

## Command Reference & Usage Examples

### General Commands

| Command         | Description                                      | Example Usage         |
|-----------------|--------------------------------------------------|----------------------|
| `/start`        | Start interaction, show welcome/help.            | `/start`             |
| `/help`         | List all commands and usage examples.            | `/help`              |
| `/myinfo`       | Show your registered email and verification info.| `/myinfo`            |
| `/mydelete`     | Delete your account, alerts, and schedules.      | `/mydelete`          |

### Email Registration & Verification

| Command                                 | Description                                               | Example Usage                  |
|-----------------------------------------|-----------------------------------------------------------|-------------------------------|
| `/myregister user@email.xyz [lang]`     | Register/update email (and language). Sends 6-digit code. | `/myregister john@x.com en`   |
| `/myverify CODE`                        | Verify your email with the 6-digit code.                  | `/myverify 123456`            |
| `/language LANG`                        | Update your language preference.                          | `/language ru`                |

### Reports

| Command                                 | Description                                               | Example Usage                  |
|-----------------------------------------|-----------------------------------------------------------|-------------------------------|
| `/report TICKER1 TICKER2 ... [flags]`   | Get report for tickers. Use flags to customize.           | `/report AAPL BTCUSDT`        |
|                                         |                                                           | `/report MSFT -email`         |
|                                         |                                                           | `/report TSLA -indicators=RSI,MACD,MA50 -email` |
|                                         |                                                           | `/report BTCUSDT -provider=bnc -period=1y`      |
| `/reportlist`                           | List your recent or scheduled reports.                    | `/reportlist`                 |

**Supported Flags:**
- `-email`: Send report to your registered email.
- `-indicators=...`: Comma-separated indicators (e.g., RSI,MACD,MA50,PE,EPS).
- `-period=...`: Data period (e.g., 3mo, 1y, 2y). Default: 2y.
- `-interval=...`: Data interval (e.g., 1d, 15m). Default: 1d.
- `-provider=...`: Data provider (e.g., yf for Yahoo, bnc for Binance). If not specified: yf for tickers ≤5 chars, bnc for crypto (>5 chars).

### Fundamental Screener

| Command                                 | Description                                               | Example Usage                  |
|-----------------------------------------|-----------------------------------------------------------|-------------------------------|
| `/schedules screener LIST_TYPE [flags]` | Schedule fundamental screener for undervalued stocks.     | `/schedules screener us_small_cap -email` |
|                                         |                                                           | `/schedules screener us_large_cap 09:00` |
|                                         |                                                           | `/schedules screener custom_list 14:00 -indicators=PE,PB,ROE` |

**Supported List Types:**
- `us_small_cap`: US Small Cap stocks
- `us_medium_cap`: US Medium Cap stocks  
- `us_large_cap`: US Large Cap stocks
- `swiss_shares`: Swiss SIX exchange stocks
- `custom_list`: Custom ticker list (specified during creation)

**Screener Flags:**
- `-email`: Send screener report to your registered email.
- `-indicators=...`: Comma-separated fundamental indicators to include (e.g., PE,PB,ROE,ROA,DY).
- `-thresholds=...`: Custom screening thresholds (e.g., PE<15,PB<1.5,ROE>15).

**Fundamental Indicators Available:**
- **Valuation**: PE (P/E Ratio), Forward_PE, PB (P/B Ratio), PS (P/S Ratio), PEG
- **Financial Health**: Debt_Equity, Current_Ratio, Quick_Ratio
- **Profitability**: ROE (Return on Equity), ROA (Return on Assets), Operating_Margin, Profit_Margin
- **Growth**: Revenue_Growth, Net_Income_Growth
- **Cash Flow**: Free_Cash_Flow
- **Dividends**: Dividend_Yield, Payout_Ratio
- **DCF**: Discounted Cash Flow valuation

**Screening Criteria:**
- P/E Ratio < 15 (undervalued)
- P/B Ratio < 1.5 (undervalued)
- ROE > 15% (good profitability)
- Debt/Equity < 0.5 (low debt)
- Current Ratio > 1.5 (good liquidity)
- Positive Free Cash Flow
- Composite undervaluation score based on multiple factors

**Report Format:**
1. **Summary**: List of top 10 undervalued tickers with key metrics
2. **Detailed Analysis**: Individual analysis for each ticker including:
   - Company overview and sector
   - Key financial ratios and their interpretation
   - Buy/Sell/Hold recommendation with reasoning
   - DCF valuation and fair value estimate
   - Risk assessment

### Alerts

| Command                                 | Description                                               | Example Usage                  |
|-----------------------------------------|-----------------------------------------------------------|-------------------------------|
| `/alerts`                               | List all your active price alerts.                        | `/alerts`                     |
| `/alerts add TICKER PRICE CONDITION`    | Add a price alert (CONDITION: above/below).               | `/alerts add BTCUSDT 65000 above` |
| `/alerts edit ALERT_ID [params]`        | Edit an alert (price/condition).                          | `/alerts edit 2 70000 below`  |
| `/alerts delete ALERT_ID`               | Delete an alert by ID.                                    | `/alerts delete 2`            |
| `/alerts pause ALERT_ID`                | Pause a specific alert.                                   | `/alerts pause 2`             |
| `/alerts resume ALERT_ID`               | Resume a paused alert.                                    | `/alerts resume 2`            |

### Scheduled Reports

| Command                                 | Description                                               | Example Usage                  |
|-----------------------------------------|-----------------------------------------------------------|-------------------------------|
| `/schedules`                            | List all your scheduled reports.                          | `/schedules`                  |
| `/schedules add TICKER TIME [flags]`    | Schedule a report at a specific time (UTC, 24h).          | `/schedules add AAPL 09:00 -email` |
| `/schedules edit SCHEDULE_ID [params]`  | Edit a scheduled report.                                  | `/schedules edit 1 10:00`     |
| `/schedules delete SCHEDULE_ID`         | Delete a scheduled report by ID.                          | `/schedules delete 1`         |
| `/schedules pause SCHEDULE_ID`          | Pause a scheduled report.                                 | `/schedules pause 1`          |
| `/schedules resume SCHEDULE_ID`         | Resume a paused scheduled report.                         | `/schedules resume 1`         |

### Admin Commands (restricted)

| Command                                 | Description                                               | Example Usage                  |
|-----------------------------------------|-----------------------------------------------------------|-------------------------------|
| `/admin users`                          | List all registered users and emails.                     | `/admin users`                |
| `/admin listusers`                      | List all users as telegram_user_id - email pairs.         | `/admin listusers`            |
| `/admin resetemail TELEGRAM_USER_ID`    | Reset a user's email.                                     | `/admin resetemail 123456789` |
| `/admin verify TELEGRAM_USER_ID`        | Manually verify a user's email.                           | `/admin verify 123456789`     |
| `/admin setlimit alerts N`              | Set global default max alerts per user.                   | `/admin setlimit alerts 10`   |
| `/admin setlimit alerts N TELEGRAM_USER_ID` | Set per-user max alerts.                              | `/admin setlimit alerts 5 123456789` |
| `/admin setlimit schedules N`           | Set global default max scheduled reports per user.        | `/admin setlimit schedules 10`|
| `/admin setlimit schedules N TELEGRAM_USER_ID` | Set per-user max scheduled reports.                  | `/admin setlimit schedules 7 123456789` |
| `/admin broadcast MESSAGE`              | Send a broadcast message to all users.                    | `/admin broadcast Maintenance at 8pm UTC.` |
| `/admin help`                           | List all admin commands and syntax.                       | `/admin help`                 |

### Feedback & Feature Requests

| Command                                 | Description                                               | Example Usage                  |
|-----------------------------------------|-----------------------------------------------------------|-------------------------------|
| `/feedback MESSAGE`                     | Send feedback or bug report to admin/developer.           | `/feedback Please add MA!`    |
| `/feature MESSAGE`                      | Suggest a new feature.                                    | `/feature Support for EURUSD` |

---

## Fundamental Screener Examples

### Basic Usage
```
/schedules screener us_small_cap
```
Schedules a daily fundamental screener for US small cap stocks at default time (09:00 UTC).

### With Email Delivery
```
/schedules screener us_large_cap -email
```
Schedules a screener for US large cap stocks and sends results to your registered email.

### Custom Indicators
```
/schedules screener us_medium_cap -indicators=PE,PB,ROE,ROA,Dividend_Yield
```
Schedules a screener focusing on specific fundamental indicators.

### Custom Time
```
/schedules screener swiss_shares 14:00 -email
```
Schedules a screener for Swiss shares at 2 PM UTC with email delivery.

### Custom List
```
/schedules screener custom_list 08:00 -indicators=PE,PB,ROE
```
Schedules a screener for a custom ticker list (you'll be prompted to specify the list during creation).

---

## Fundamental Analysis Guide

### Key Ratios Explained

**Valuation Ratios:**
- **P/E Ratio**: Price-to-Earnings ratio. Lower values suggest undervaluation.
  - < 15: Undervalued
  - 15-25: Fair value
  - > 25: Potentially overvalued

- **P/B Ratio**: Price-to-Book ratio. Measures market value vs book value.
  - < 1: Undervalued
  - 1-3: Fair value
  - > 3: Potentially overvalued

- **P/S Ratio**: Price-to-Sales ratio. Useful for companies with no earnings.
  - < 1: Undervalued
  - 1-3: Fair value
  - > 3: Potentially overvalued

**Financial Health:**
- **Debt/Equity**: Total debt divided by shareholders' equity.
  - < 0.5: Low debt (good)
  - 0.5-1.0: Moderate debt
  - > 1.0: High debt (risky)

- **Current Ratio**: Current assets divided by current liabilities.
  - > 1.5: Good liquidity
  - 1.0-1.5: Adequate liquidity
  - < 1.0: Poor liquidity

**Profitability:**
- **ROE**: Return on Equity. Higher values indicate better profitability.
  - > 15%: Excellent
  - 10-15%: Good
  - < 10%: Poor

- **ROA**: Return on Assets. Measures efficiency of asset utilization.
  - > 5%: Good
  - 2-5%: Average
  - < 2%: Poor

**Growth:**
- **Revenue Growth**: Year-over-year revenue growth rate.
  - > 10%: Strong growth
  - 5-10%: Moderate growth
  - < 5%: Slow growth

**Dividends:**
- **Dividend Yield**: Annual dividend as percentage of stock price.
  - > 4%: High yield
  - 2-4%: Moderate yield
  - < 2%: Low yield

### Buy/Sell/Hold Recommendations

**BUY Criteria:**
- P/E < 15 and P/B < 1.5
- ROE > 15% and ROA > 5%
- Debt/Equity < 0.5
- Positive revenue and earnings growth
- Positive free cash flow
- Composite score > 7/10

**HOLD Criteria:**
- Fair valuation metrics
- Stable financials
- Moderate growth prospects
- Composite score 5-7/10

**SELL Criteria:**
- P/E > 25 or P/B > 3
- ROE < 10% or ROA < 2%
- High debt levels
- Declining revenue/earnings
- Negative free cash flow
- Composite score < 5/10

---

## Technical Details

### Data Sources
- **Fundamental Data**: Yahoo Finance (yfinance)
- **Ticker Lists**: CSV files for US stocks, SIX API for Swiss shares
- **Rate Limits**: Standard yfinance rate limits (sequential processing)

### Error Handling
- Tickers with missing fundamental data are skipped
- Errors are logged to the system log file
- Partial failures don't stop the entire screening process

### Performance
- Sequential processing to respect API rate limits
- Future enhancements planned for caching and parallel processing
- Sector-average comparisons planned for future releases

---

## Future Enhancements

### Planned Features
- **Sector Comparison**: Compare metrics against sector averages
- **Percentile Rankings**: Rank stocks within their peer group
- **Caching**: Cache fundamental data to reduce API calls
- **Custom Thresholds**: User-defined screening criteria
- **Portfolio Integration**: Track screened stocks in user portfolios
- **Alert Integration**: Price alerts for screened stocks
- **Export Options**: CSV/Excel export of screening results

### Advanced Screening
- **Multi-factor Models**: Composite scoring systems
- **Technical + Fundamental**: Combined technical and fundamental analysis
- **Risk Assessment**: Volatility and beta analysis
- **ESG Screening**: Environmental, Social, and Governance factors
- **International Markets**: Support for additional markets beyond US and Switzerland

---

## Architecture & API Integration

### Components
- **Telegram Bot API:** Receives commands, sends messages, manages user interaction.
- **Backend Server:** Handles business logic, command parsing, user state, and orchestrates API/data provider requests.
- **Data Provider APIs:** Fetches real-time/historical data for stocks and cryptos (Yahoo, Binance, etc.).
- **Email Service:** Sends reports and alerts to verified user emails.
- **Database:** Stores users, alerts, schedules, logs, and cache data.
- **Cache Layer:** Caches recent API responses to reduce calls and improve performance.

### Data Provider Logic
- **Stocks:** Yahoo Finance, Alpha Vantage, Polygon.io, Finnhub, Twelve Data (free tiers).
- **Cryptos:** Binance, CoinGecko.
- **Failover:** If one API fails/exceeds quota, try next provider.
- **Caching:** Cache ticker data per provider/interval (default: 1 day).
- **Ticker Classification:** 1-4 chars = stock; >4 chars = crypto pair.

### Email & Notification
- **Verification:** 6-digit code sent via email (valid 1 hour, max 5/hour).
- **Report Delivery:** Reports sent as HTML emails (charts inline/attached).
- **Alert Notifications:** Alerts via Telegram and/or email.
- **Logging:** All sent emails and notifications are logged for admin review.

---

## Data Model & ER Diagram

### Table Definitions

#### users
| Field              | Type         | Description                                   |
|--------------------|-------------|-----------------------------------------------|
| user_id            | INTEGER PK  | Internal unique user identifier               |
| telegram_user_id   | TEXT UNIQUE | Telegram user ID (from Telegram API)          |
| email              | TEXT        | Registered email address                      |
| validation_sent    | DATETIME    | When verification code was sent               |
| validation_received| DATETIME    | When user successfully verified email         |
| verification_code  | TEXT        | Last sent 6-digit code (for validation)       |
| is_verified        | BOOLEAN     | Email verification status                     |
| language           | TEXT        | User's language code (e.g., 'en', 'ru')       |
| is_admin           | BOOLEAN     | Is the user an admin?                         |
| max_alerts         | INTEGER     | Per-user max number of alerts (nullable)      |
| max_schedules      | INTEGER     | Per-user max number of schedules (nullable)   |

#### alerts
| Field      | Type         | Description                                   |
|------------|--------------|-----------------------------------------------|
| alert_id   | INTEGER PK   | Unique alert identifier                       |
| ticker     | TEXT         | Ticker symbol (e.g., AAPL, BTCUSDT)           |
| user_id    | INTEGER FK   | References users.user_id                      |
| price      | REAL         | Price threshold for alert                     |
| condition  | TEXT         | 'above' or 'below'                            |
| is_active  | BOOLEAN      | Is the alert currently active?                |
| created    | DATETIME     | When the alert was created                    |
| updated_at | DATETIME     | Last update timestamp                         |

#### schedules
| Field          | Type         | Description                                 |
|----------------|--------------|---------------------------------------------|
| schedule_id    | INTEGER PK   | Unique schedule identifier                  |
| ticker         | TEXT         | Ticker symbol                               |
| scheduled_time | TEXT         | Time for scheduled report (e.g., '09:00')   |
| period         | TEXT         | daily/weekly/monthly                        |
| user_id        | INTEGER FK   | References users.user_id                    |
| is_active      | BOOLEAN      | Is the schedule currently active?           |
| created        | DATETIME     | When the schedule was created               |
| updated_at     | DATETIME     | Last update timestamp                       |

#### settings
| Field         | Type         | Description                                   |
|-------------- |------------- |-----------------------------------------------|
| key           | TEXT PK      | Setting name (e.g., 'max_alerts')             |
| value         | TEXT         | Setting value                                 |

### Entity-Relationship (ER) Diagram

```
+---------------------+         +-------------------+         +-------------------+
|       users         |         |      alerts       |         |    schedules      |
+---------------------+         +-------------------+         +-------------------+
| user_id (PK)        |<---+    | alert_id (PK)     |         | schedule_id (PK)  |
| telegram_user_id    |    |    | ticker            |         | ticker            |
| email               |    +----| user_id (FK)      |         | scheduled_time    |
| validation_sent     |         | price             |         | period            |
| validation_received |         | condition         |         | user_id (FK)      |
| verification_code   |         | is_active         |         | is_active         |
| is_verified         |         | created           |         | created           |
+---------------------+         | updated_at        |         +-------------------+
                                +-------------------+
```
- **users** (1) --- (N) **alerts**
- **users** (1) --- (N) **schedules**

---

## Email Templates

### Verification Code
- **Subject:** Your Alkotrader Email Verification Code
- **Body:**
  - Plain: "Your verification code is: 123456"
  - HTML: Code in bold, valid for 1 hour.

### Report Delivery
- **Subject:** Alkotrader Report for [TICKER]
- **Body:**
  - Plain/HTML: Report content, charts inline or attached.

### Price Alert
- **Subject:** Alkotrader Price Alert: [TICKER] [above/below] [PRICE]
- **Body:**
  - Plain/HTML: Current price, threshold, trigger info.

### Admin Broadcast
- **Subject:** Alkotrader Announcement
- **Body:**
  - Plain/HTML: Broadcast content.

---

## Admin Panel Features & UI

- **Authentication:** Admin login, session management.
- **User Management:** View/edit/delete users, export, manual verification.
- **Alerts/Schedules:** View/edit/pause/resume/delete, add for user, view history.
- **Broadcast:** Send to all/segment, direct message, view history.
- **Logs:** View/filter/download logs (user commands, API errors, email delivery, admin actions).
- **Statistics:** User/usage/API stats, charts.
- **Settings:** Set global/per-user limits, manage API keys, feature toggles.
- **Feedback:** View/respond to user feedback/feature requests.
- **Audit Trail:** All admin actions logged.

---

## Localization (i18n)

- All user-facing text uses translation keys (not raw text).
- User language stored in DB; `/language LANG` to update.
- Use `gettext`/`Babel` for message catalogs.
- Fallback to English if translation missing.
- Directory: `src/telegram_screener/locales/` with `.po`/`.mo` files.

---

## Technical Notes & Code Reuse

- **Reuse existing modules:** Data downloaders, emailer, logger, screener_db, notification manager.
- **Separation of concerns:** Business logic separate from Telegram API handling.
- **Unit tests:** Required for business logic and command parsing.
- **Error handling:** All exceptions logged, user-friendly messages.
- **Caching:** API responses cached per ticker/provider/interval.
- **Downloader interface:** All providers implement a common interface.
- **Command handler registry:** Dynamic registration for extensibility.
- **Notification logging:** All outgoing notifications logged.

---

## Implementation Steps

1. Implement core commands and user flows.
2. Integrate all data downloaders/providers.
3. Implement report generation (technicals/fundamentals, formatting).
4. Integrate emailer/notification manager.
5. Use screener_db for all data storage.
6. Implement alerts and scheduling.
7. Build admin features (CLI/web panel).
8. Set up logging and error handling.
9. Enforce security, validation, and rate limits.
10. Prepare for deployment (local server, logging enabled).

---

## Extensibility & Future Roadmap

- **Multi-language support:** Add new languages via message catalogs.
- **New data providers:** Add by implementing the downloader interface.
- **UI/UX improvements:** Inline keyboards, autocomplete, last ticker memory.
- **Hot-reload:** Planned for config changes.
- **User feedback:** Mechanism for feature requests/bug reports.

---

## Documentation

For detailed technical information about the Telegram screener module:

- **[Requirements.md](Requirements.md)** - Dependencies, API keys, database setup, and deployment requirements
- **[Design.md](Design.md)** - Architecture, design decisions, data flow, and technical specifications
- **[Tasks.md](Tasks.md)** - Development roadmap, known issues, technical debt, and implementation timeline

## Contributing

When contributing to the screener module:
1. Follow the architecture patterns documented in Design.md
2. Update Requirements.md if adding new dependencies
3. Add development tasks to Tasks.md for tracking
4. Ensure all business logic is testable and well-documented
5. Update this README.md for user-facing changes

## References & Further Reading
- [OpenBB Telegram Bot Docs](https://docs.openbb.co/bot/usage/telegram)
- [CoinGecko Bot Guide](https://www.coingecko.com/learn/build-crypto-telegram-bot)
- [Telegram Bot API](https://core.telegram.org/bots/api)
- [Python Telegram Bot](https://github.com/python-telegram-bot/python-telegram-bot)
