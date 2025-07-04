# Telegram Bot for Shares & Crypto Reports — Comprehensive Requirements

## 1. Command Structure & User Interaction

- **Command list:**  
  - `/start` or `/help`: Show all available commands and parameters.  
  - `/info`: Show registered email and verification status/dates.  
  - `/register user@email.xyz`: Register or update email for the Telegram user and send 6-digit verification code (valid 1 hour).  
  - `/verify CODE`: Verify email with the code.  
  - `/report TICKER1 TICKER2 ... [-email] [-indicators=...]`: Request report for specified tickers (shares or cryptos). Optional flags: send report to email, select indicators.  
  - `/alerts`: List active price alerts.  
  - `/alerts add TICKER PRICE CONDITION`: Add alert (condition = above/below).  
  - `/alerts edit ALERT_ID ...`: Edit alert.  
  - `/alerts delete ALERT_ID`: Delete alert.  
  - `/schedules`: List scheduled reports.  
  - `/schedules add TICKER TIME`: Add scheduled report.  
  - `/schedules edit SCHEDULE_ID ...`: Edit schedule.  
  - `/schedules delete SCHEDULE_ID`: Delete schedule.  
- **User selection of indicators:** Users can specify which technical/fundamental indicators to include in reports.  
- **Explicit commands only:** No inline commands like `$AAPL`; only explicit commands like `/report AAPL`.  
- **Limits:** Default max 5 active alerts and 5 scheduled reports per user; configurable by admin.  
- **Unknown commands:** Bot replies with help message and logs the attempt for admin review.

## 2. API Integration & Data Handling

- **Ticker classification:**  
  - 1-4 characters → share ticker (e.g., MSFT).  
  - More than 4 characters → crypto pair (e.g., BTCUSDC).  
- **Data providers:**  
  - Shares: Yahoo Finance, Alpha Vantage, Polygon.io, Finnhub, Twelve Data (free tiers only).  
  - Cryptos: Binance, CoinGecko.  
- **Failover:** If one API exceeds quota or fails, automatically try next provider.  
- **Caching:** Cache ticker data per provider and response for report interval (default 1 day).  
- **Crypto support:** Include spot prices and derivatives/futures if available.  
- **Error handling:** User-friendly messages for invalid ticker, rate limits, or API errors.

## 3. Email Reporting

- **Format:** Email subject and body customizable by user; default subject: “Alkotarder report”.  
- **Logging:** Keep log of sent emails per user for admin troubleshooting.  
- **Notifications:** Notify users in Telegram on successful email delivery or errors.

## 4. Validation & Security

- **Verification code:** 6-digit code sent via email, valid for 1 hour.  
- **Notifications:** Notify user if registration/verification attempt is made with a new email for their Telegram ID.  
- **Failed attempts:** Log failed verification attempts for admin review.  
- **Cooldown:** Lockout for 10 minutes after 5 wrong code entries.

## 5. Admin Features

- **Broadcast:** Admins can broadcast messages to all users.  
- **Admin panel:** Accessible both via Telegram commands and web interface.  
- **User management:** Admins can reset user emails and manually verify users.

## 6. Notifications & Alerts

- **Alert notifications:** Send alerts via both Telegram message and email.  
- **Pause/Resume:** Users can pause or resume alerts and scheduled reports.

## 7. Logging & Monitoring

- **Log contents:** User commands, API errors, email delivery status, admin actions.  
- **Retention:** Logs kept for 30 days.

## 8. User Experience

- **Last ticker memory:** Not supported initially (future CR).  
- **Autocomplete/suggestions:** Not supported initially (future CR).

## 9. Development & Deployment

- **Config changes:** Require restart for now; hot-reloading planned for future.  
- **Testing:**  
  - Business logic separated from Telegram interface.  
  - Unit tests for core logic and command parsing modules mandatory.

## 10. Future-Proofing

- **Languages:** Designed to support multiple languages in future.  
- **Data providers:** Easy to add new stock/crypto data providers later.  
- **User feedback:** Mechanism for users to request features or report bugs directly via the bot.

If you want, I can also help draft:

- Detailed command syntax and usage examples  
- Data model diagrams for users, alerts, schedules  
- API integration design outline  
- Email template samples  
- Admin panel feature list and UI mockups  

Just let me know! This requirements summary should give your development team a clear, actionable blueprint to start building your Telegram bot.

***********************************************
-----------------------------------------------
# Clarifications & Implementation Notes

This section summarizes the latest implementation clarifications for the Telegram Screener Bot:

- **Email Report Format:** Reports are sent as HTML emails, with charts included inline (as embedded images) or as attachments.
- **Admin Panel:** Admin features are available both via bot commands and a web interface (to be implemented in `src/frontend/webgui`).
- **Scheduled Reports:** Users can set up recurring reports (daily/weekly/monthly) at user-defined times.
- **Price Alerts:** Alerts can be sent via Telegram and/or email. Users can select their preferred channels.
- **User Deletion:** Users can delete their account/data via a command (see `01.Detailed_command_syntax_and_usage_examples.md`).
- **Verification Code Delivery:** The 6-digit code for email verification is always sent via email (never via Telegram).
- **Localization:** All user-facing text should be localizable. English is the only language for now, but the structure should support future localization.
- **API Key Management:** All API keys are managed via `.env` and constants in `config/donotshare/donotshare.py`.
- **UI/UX:** The bot should support both inline keyboards and text commands for all user actions.
- **Testing:** Comprehensive unit, integration, and UI tests should be implemented for all components, including command parsing, business logic, and user interface.

---

# (Original requirements follow)

I want to develop telegram bot, which would give me information about shares or cryptos by request. it should also be able to send this report via email. 1 email per telegram user is allowed. validation via 6 digits code. ask me questions. i want to create comprehensive requirements.

To create a comprehensive requirements document for your Telegram bot that delivers share/crypto information and sends reports via email (with 1 email per Telegram user and 6-digit code validation), you should clarify several aspects. Here are the main questions and considerations to ensure your requirements are complete and actionable:
**1. User Story & Use Cases**
- What is the main workflow for a typical user? (e.g., start bot → request info → receive report → send report to email)- Should users be able to request both stock and crypto data, or only one at a time?
/start or /help should print all available commands and their parameters
unknown command should do the same
/info - prints email, verification dates
/register user@email.xyz sets/updates email (only 1) for telegram_user_id and sends verification code to this email. code is valid for 1 hour.
/verify CODE - verifies email
- What kind of information should the bot provide for shares and cryptos? (e.g., price, volume, open/close, charts, news)- Should users be able to set up recurring reports or only on-demand requests?

For cryptos something like this (technical indicators):
📉 Technical Analysis:
RSI: 70.64 - BUY - Oversold - Strong buy signal
Stochastic %K: 95.27, %D: 94.78 - description like for RSI
ADX: 19.27, +DI: 32.15, -DI: 16.60 - description like for RSI
OBV: 67957600 - description like for RSI
ADR: 0.97, Avg ADR: 1.01 - description like for RSI
MA(50): $121.66 - description like for RSI
MA(200): $118.59 - description like for RSI
MACD: 1.7089, Signal: 1.5532, Hist: 0.1557 - description like for RSI
Trend: Uptrend - description like for RSI

For shares apart from technicals show also fundamental indicators like this:
💵 Price: $128.85
🏦 P/E: 21.62, Forward P/E: 0.00 - description what does it mean
💸 Market Cap: $0.00B
📊 EPS: $0.00, Div Yield: 183.00%

**2. Data Sources & APIs**
- Which APIs or data providers should be used for stock and cryptocurrency data? (e.g., CoinGecko, Yahoo Finance, Alpha Vantage)[1][2]
If ticker is 1-4 charachters, then treat it as share (e.g. MSFT), if more than treat as crypto pair (e.g. BTCUSDC)
- Are there specific exchanges or markets you want to support?
Yahoo Finance, Alpha Vantage, Poligon.io, Finnhub, Twelve Data - for shares, only free tariffs.
Binance, CoinGecko - for cryptos
- Should the bot support custom symbols or only a predefined list?[2]
**3. Telegram Bot Interaction**
- What commands or buttons should the bot support? (e.g., /start, /help, /price, /report, /email)
- Should the bot use inline keyboards for navigation and selection?[2]
- What languages should the bot support?
**4. Email Integration**
- Should the report be sent as plain text, HTML, or as an attachment (PDF, CSV, etc.)?
- Which email service or provider should be used for sending emails? (e.g., SMTP, SendGrid, Gmail API)[3]
- Should users be able to update their email address, or is it fixed after initial registration?
- Should there be a confirmation email after registration or for every report sent?
**5. Email Validation & Security**
- Should the 6-digit code be sent via email or Telegram? (Your description suggests email, but clarify.)
  Email
- How should the bot handle expired or invalid codes?
  Do not use them. For the next request generate a new one
- What is the desired expiration time for the code (e.g., 10 minutes)?
  1 hour
- Should there be rate limiting or anti-abuse mechanisms for email registration/validation?
  Yes. No more than 5 per hour
- Should you use Telegram's login widget for authentication, or manage user accounts yourself?[4]
  I will manage user account myself. But tell me more about telegram authentication widget
**6. User Management**
- Should each user be limited to one Telegram account and one email address?
  No. Account identifier is a telegram_user_id. One email address per telegram_user_id
- Should users be able to delete or change their email address?
  Yes
- Should admins have access to a list of registered users/emails?
  Yes
**7. Privacy & Compliance**
- Do you need to comply with any privacy regulations (GDPR, etc.) regarding email storage and user data?
  No
- Should users be able to delete their data?
  Yes
**8. Notifications & Alerts**
- Should users be able to set price alerts for shares/cryptos?[1]
  Yes
- Should the bot send scheduled updates or only on-demand reports?
  Yes
**9. Deployment & Maintenance**
- Where will the bot be hosted (e.g., Heroku, AWS, local server)?[5]
  Local server
- Should the bot support logging and error reporting for admin review?
  Ideally yes
- Should there be an admin panel for managing users and monitoring usage?
  Yes
**10. Scalability & Performance**
- What is the expected number of users?
  10-50
- Should the bot handle high-frequency requests or just basic queries?
  Only basic queries
**11. Example User Flow**
Here's a sample flow to help clarify requirements:
1. User starts the bot and is prompted to register an email.
   No, he may use bot without registration. Email functionality will not be available for him until he registers and confirms the email address. Existing functionality.
3. User enters email; bot sends a 6-digit code to that email for verification[6].
   Yes. Existing functionality.
5. User enters the code in Telegram; if valid, email is linked to Telegram account.
   Yes. Existing functionality.
7. User requests a share/crypto report via command or button.
   Via telegram command. Existing functionality.
9. Bot fetches data from selected APIs and prepares a report.
    Yes. Existing functionality.
11. User can request to have the report sent to their registered email.
    Yes, as a part of command, providing "-email" flag in the command.
13. User receives the report via email.
    Right

**12. Technical requirements
1. Use standard logging from the project. 
Use lazy % message initialization.
Example:
from src.notification.logger import setup_logger
logger = setup_logger("telegram_bot")

try:
	...code goes here...
except Exception as e:
    logger.error("Notification manager not initialized: %s", e, exc_info=True)

2. Use standard emailer from the project.
try:
    notification_manager = asyncio.get_event_loop().run_until_complete(
        initialize_notification_manager(
            telegram_token=TELEGRAM_BOT_TOKEN,
            telegram_chat_id=TELEGRAM_CHAT_ID,
            email_sender=SMTP_USER,
            email_receiver=None  # Will be set per user
        )
    )
except Exception as e:
    logger.error("Notification manager not initialized: %s", e, exc_info=True)

3. Use screener_db module to interact with SQLite db.
Table users: user_id, telegram_user_id, email, validation_sent, validation_received, verification_code;
Table schedules: ticker, scheduled_time, user_id (FK to users.user_id);
Table alerts: ticker, user_id (FK to users.user_id), price, condition ("above" or "below");

4. Implement data downlaoders for Alpha Vantage, Poligon.io, Finnhub, Twelve Data - similar to existing YahooDataDownloader
CoinGecko - similar to existing BinanceDataDownloader



**Please answer or clarify the following:**
- What data fields should be included in the share/crypto report?
  Existing functionality.
- What is your preferred API or data provider for stocks and cryptos?
  Existing functionality.
- Should the bot support both instant and scheduled reports?
  Yes
- Should users be able to change or remove their email after registration?
  Yes
- What is your preferred email sending method or provider?
  Existing functionality.
- Should the bot support multiple languages?
  Not for now. Only English.
- Do you want to use Telegram's official login/auth widget for easier user validation?[4]
  Not yet
- Should there be admin features (e.g., user management, usage stats)?
  Yes
- Any specific UI/UX preferences for the Telegram bot interface?
  No. Set of the commands and their structure should be defined in the requirements later.

Once you provide these details, a complete and actionable requirements document can be drafted for your project.
Citations:
[1] GitHub - paragrudani1/telegram-crypto-alert-bot: A Telegram bot for tracking cryptocurrency prices and setting up price alerts. The bot uses the CoinGecko API to fetch cryptocurrency prices and the python-telegram-bot library to interact with users. Features include getting current crypto prices, setting up price alerts, viewing and managing active alerts, and periodic checking of price alerts. https://github.com/paragrudani1/telegram-crypto-alert-bot  
[2] GitHub - THEFZNKHAN/RealTimeMarketBot: This Telegram bot allows users to fetch real-time stock data directly within Telegram. Users can select from a list of predefined stocks or enter a custom stock symbol to retrieve relevant information such as open, high, low, close prices, and volume. https://github.com/THEFZNKHAN/RealTimeMarketBot  
[3] Telegram bot api and Email integration https://latenode.com/integrations/telegram-bot-api/email  
[4] Telegram Part 3. Authentication. Bot to send private messages https://blog.corsego.com/ruby-on-rails-telegram-login  
[5] Title: How I Created My Own Trading Bot on Telegram to ... https://www.reddit.com/r/solana/comments/1cwe8ik/title_how_i_created_my_own_trading_bot_on/  
[6] account.sendVerifyEmailCode - Telegram APIs https://core.telegram.org/method/account.sendVerifyEmailCode  
[7] How to Build a Telegram Media Sharing Bot with Admin ... https://www.youtube.com/watch?v=ntg6-7OHywI[8] How to send a Telegram message to a username using bot api or ... https://stackoverflow.com/questions/72199108/how-to-send-a-telegram-message-to-a-username-using-bot-api-or-telegram-api  
[9] From BotFather to 'Hello World' https://core.telegram.org/bots/tutorial  
[10] Authorization via Telegram Gateway: Quick-start Guide https://core.telegram.org/gateway/verification-tutorial  
[11] How To Build an AI Chatbot for Telegram [Easy Steps] https://www.voiceflow.com/blog/telegram-chatbot  
[12] Email List Validation to Telegram Bot FREE Integrations - Pabbly https://www.pabbly.com/connect/integrations/email-list-validation/telegram-bot/  
[13] Telegram bot for displaying cryptocurrencies prices and charts https://github.com/ebellocchia/telegram_crypto_price_bot  
[14] Stock Prices Telegram | CoinTrendzBot.com https://cointrendzbot.com/features/stockprice  
[15] Telegram Registration with Email Verification Support (2025 Edition ) No Manual Work https://www.youtube.com/watch?v=ZDG9bMAznh0  
[16] Pabbly Email Verification to Telegram Bot FREE Integrations | Pabbly Connect https://www.pabbly.com/connect/integrations/pabbly-email-verification/telegram-bot/  
[17] How to Build a Telegram Bot in Under 5 Minutes (Quick & ... https://www.youtube.com/watch?v=rKmxRCfITek  
[18] Building a Telegram bot in Python to track your portfolio https://duarteocarmo.com/blog/building-a-telegram-bot-in-python-to-track-your-portfolio.html  
[19] send automatic message for all user and group with telegram bot #20 https://github.com/irazasyed/telegram-bot-sdk/issues/20  
[20] Telegram Bot API Verification - AI Prompt https://docsbot.ai/prompts/technical/telegram-bot-api-verification  
[21] How to integrate Email Validation & Telegram - Integrately https://integrately.com/integrations/email-validation/telegram  
[22] User Authorization - Telegram APIs https://core.telegram.org/api/auth  
[23] Implementing Secure E2E Telegram Process https://www.reddit.com/r/TelegramBots/comments/15vcrke/implementing_secure_e2e_telegram_process_best/  
[24] Integrating with Telegram Bot https://docs.esputnik.com/docs/integrating-with-telegram-bot  
[25] Email List Verify and Telegram Bot Integration https://www.make.com/en/integrations/email-list-verify/telegram  
[26] Exploring the Safety of Telegram Bots https://botpenguin.com/blogs/exploring-the-safety-of-telegram-bots  
[27] How can I get verified on Telegram? - Mad Penguin https://www.madpenguin.org/how-can-i-get-verified-on-telegram/  
[28] Telegram Bot to Email List Validation FREE Integrations https://www.pabbly.com/connect/integrations/telegram-bot/email-list-validation/  
