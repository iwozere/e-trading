# Telegram Screener Bot Documentation

## Overview

The Telegram Screener Bot is an advanced tool for managing, analyzing, and monitoring stock and crypto tickers directly from Telegram. It provides users with:
- Per-user ticker management (add, delete, list)
- Real-time technical and fundamental analysis
- Detailed chart generation
- Email reporting with charts and recommendations
- Support for both Yahoo Finance (YF) and Binance (BNC) tickers
- Robust error handling and logging
- **Secure email registration and verification**

---

## Features

- **Per-user ticker management**: Each user can maintain their own list of tickers, organized by provider (YF/BNC).
- **Comprehensive analysis**: For each ticker, the bot provides:
  - Technical indicators (RSI, MACD, Bollinger Bands, Stochastic, ADX, OBV, ADR, etc.)
  - Fundamental data (for YF tickers: price, company, market cap, P/E, EPS, dividend yield, etc.)
  - Per-indicator recommendations and overall signal
- **Chart generation**: Generates and sends detailed charts for each ticker, visualizing all major indicators.
- **Email reporting**: Users can receive analysis and charts via email (after secure registration and verification).
- **Security**: Each Telegram user can register and verify only one email, which is securely bound to their Telegram account.
- **Verification**: Email verification is required before any reports can be sent to email.

---

## Commands

### Registration & Verification

- `/my-register email@example.com`
  - Register or update your email address.
  - Triggers a verification email with a 6-digit code (valid for 1 hour).
  - You must verify your email before receiving reports.

- `/my-verify CODE`
  - Enter the 6-digit code you received by email to verify your address.
  - If the code is valid and not expired, your email is marked as verified.
  - The bot will notify you of success or failure.

- `/my-info`
  - Shows your registered email, verification status, and timestamps for registration and verification.

### Ticker Management

- `/my-list [-PROVIDER]`
  - List your saved tickers (optionally filter by provider: `-yf` or `-bnc`).

- `/my-add -PROVIDER TICKER`
  - Add a ticker to your list (e.g., `/my-add -yf AAPL`).

- `/my-delete -PROVIDER TICKER`
  - Remove a ticker from your list.

### Analysis Command

- **/analyze** is the unified command for all analysis:
  - `/analyze` — analyze all your tickers
  - `/analyze -yf` — analyze all your YF tickers
  - `/analyze AAPL` — analyze just AAPL
  - `/analyze AAPL -period=1y -interval=1h` — analyze AAPL with custom settings
  - `/analyze -email` — send all results to your email
  - `/analyze BTCUSDT -period=1y -interval=1d -email` — analyze BTCUSDT with custom settings and email
- All previous functionality of `/my-status` and `/my-analyze` is now handled by `/analyze`.
- You can combine provider, period, interval, and email flags as needed.

### Help & Info

- `/start`
  - Welcome message and quickstart guide.

- `/help`
  - Detailed help, including registration, verification, and all commands.

---

## Email Registration & Verification Flow

1. **Register**: Use `/my-register email@example.com` to set or update your email.
2. **Verification Email**: The bot sends a verification email with subject `e-Trading: email verification` and a 6-digit code (valid for 1 hour).
3. **Verify**: Use `/my-verify CODE` in Telegram to verify your email.
4. **Check Status**: Use `/my-info` to see your email and verification status.
5. **Receive Reports**: Use `-email` flag in `/analyze` to receive reports at your verified email.

**Note:**
- Only one email per Telegram user is allowed.
- You can update your email at any time by re-registering and re-verifying.
- If you try to use `-email` without a verified email, the bot will notify you.

---

## Email Reports with Attachments

- When you use the `/analyze` command with the `-email` flag, the bot sends a single email containing the analysis for all requested tickers.
- The email includes:
  - Fundamentals (if available)
  - Technicals (all major indicators)
  - The generated chart for each ticker as an attachment
- Charts are attached as image files (PNG) to the email, and the email body contains the formatted analysis for each ticker.
- Temporary chart files are only deleted after the email is successfully sent, ensuring reliable delivery and avoiding file access errors.
- This is powered by the notification system's support for email attachments.

**Best Practices:**
- Make sure your registered email can receive attachments.
- If you do not receive the charts, check your spam folder or email provider's attachment limits.

---

## Technical Details

- **Database**: User info (including email and verification status) is stored in the `users` table in SQLite.
- **Verification Code**: 6-digit numeric code, expires after 1 hour.
- **Security**: Only the Telegram user who registered the email can trigger email reports for their account.
- **No email argument**: The bot no longer accepts an email address as a command argument; use `-email` flag and your registered/verified email will be used.

---

## Error Handling & User Feedback

- If you try to use `-email` without registering, you'll get: `No email registered. Use /my-register to set your email.`
- If you try to use `-email` without verifying, you'll get: `Your email is not verified. Use /my-verify CODE to verify.`
- If your verification code is expired or invalid, the bot will notify you.
- All errors are logged for troubleshooting.

---

## Example Usage

```
/my-register alice@example.com
# (Check your email, then:)
/my-verify 123456
/my-info
/analyze -email
/analyze -yf -email
/analyze AAPL -period=1y -interval=1h -email
/analyze BTCUSDT -period=1y -interval=1d -email
```

---

## Advanced Notes

- **MultiIndex Handling**: The bot handles yfinance MultiIndex DataFrames for YF tickers, ensuring compatibility with TA-Lib.
- **File Cleanup**: Temporary chart files are only deleted after successful delivery to avoid file access errors on Windows.
- **Extensibility**: The bot is designed for easy addition of new providers, indicators, or output formats.
- **Testing**: The `tests/test_screener_bot.py` script can be used to simulate bot commands and verify output.

---

## Troubleshooting

- **No Data Available**: If a ticker is delisted or has no recent data, the bot will notify the user and skip it.
- **Chart Generation Errors**: If TA-Lib or pandas-ta fails, the bot logs the error and continues.
- **Email Issues**: Ensure SMTP credentials are correct in `config/donotshare/donotshare.py`.
- **File Access Errors**: These are now prevented by proper file cleanup logic.

---

## Contact & Support

For support, bug reports, or feature requests, contact the bot administrator or open an issue in the project repository.

## Per-Ticker and Per-Command Period/Interval

- You can specify custom data periods and intervals for each ticker.
- **Defaults:** If not specified, the bot uses 2 years (`2y`) and daily bars (`1d`).
- **Override on add:**
  - `/my-add -yf AAPL -period=1y -interval=15m` adds AAPL with 1 year of 15-minute bars.
  - `/my-add -bnc BTCUSDT` uses the default (2y, 1d).
- **Override on analysis:**
  - `AAPL -period=6mo -interval=1h` analyzes AAPL with 6 months of 1-hour bars (does not change DB).
- **/my-list:**
  - Shows period/interval for each ticker if not default.
- **How it works:**
  - The bot stores period/interval per ticker in the database if provided in /my-add.
  - If not set, the default is used.
  - You can override for a single analysis by providing flags in the command. 