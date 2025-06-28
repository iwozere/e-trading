# Telegram Screener Bot Documentation

## Overview

The Telegram Screener Bot is an advanced tool for managing, analyzing, and monitoring stock and crypto tickers directly from Telegram. It provides users with:
- Per-user ticker management (add, delete, list)
- Real-time technical and fundamental analysis
- Detailed chart generation
- Email reporting with charts and recommendations
- Support for both Yahoo Finance (YF) and Binance (BNC) tickers
- Robust error handling and logging

---

## Features

- **Per-user ticker management**: Each user can maintain their own list of tickers, organized by provider (YF/BNC).
- **Comprehensive analysis**: For each ticker, the bot provides:
  - Technical indicators (RSI, MACD, Bollinger Bands, Stochastic, ADX, OBV, ADR, etc.)
  - Fundamental data (for YF tickers: price, company, market cap, P/E, EPS, dividend yield, etc.)
  - Per-indicator recommendations and overall signal
- **Chart generation**: Generates and sends detailed charts for each ticker, visualizing all major indicators.
- **Email integration**: Users can request analysis results and charts to be sent to their email.
- **Error handling**: All actions and errors are logged. User-facing errors are clear and actionable.
- **Multi-provider support**: Works with both Yahoo Finance (stocks/ETFs) and Binance (crypto).

---

## Commands

### `/my-add -PROVIDER TICKER`
Add a ticker to your list for a specific provider.
- Example: `/my-add -yf AAPL` or `/my-add -bnc BTCUSDT`

### `/my-delete -PROVIDER TICKER`
Remove a ticker from your list for a specific provider.
- Example: `/my-delete -yf AAPL`

### `/my-list [-PROVIDER]`
List all your tickers, or only those for a specific provider.
- Example: `/my-list` or `/my-list -bnc`

### `/my-status [-PROVIDER] [EMAIL]`
Analyze all your tickers (optionally for a provider) and optionally email the results (with charts).
- Example: `/my-status` or `/my-status -yf user@email.com`

### `/my-analyze -PROVIDER TICKER [EMAIL]`
Analyze a single ticker and optionally email the result and chart.
- Example: `/my-analyze -bnc BTCUSDT user@email.com`

---

## Analysis Details

### Technical Indicators (YF & BNC)
- **RSI**: Relative Strength Index
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: Upper, Middle, Lower bands
- **Stochastic Oscillator**: %K and %D
- **ADX**: Average Directional Index (+DI, -DI)
- **OBV**: On-Balance Volume
- **ADR**: Average Daily Range
- **SMA/EMA**: Simple/Exponential Moving Averages (20, 50, 200)
- **Volume**: Daily trading volume

### Fundamental Data (YF only)
- **Current Price**
- **Company Name**
- **Market Cap**
- **P/E Ratio**
- **Forward P/E**
- **Earnings Per Share (EPS)**
- **Dividend Yield**

### Recommendations
- Each indicator provides a recommendation (BUY/SELL/HOLD) with a reason.
- An overall recommendation is generated based on all indicators.

---

## Chart Generation

- Charts are generated using Matplotlib and TA-Lib (for YF) or TA-Lib and pandas-ta (for BNC).
- Each chart includes:
  - Price with Bollinger Bands and SMAs
  - RSI, MACD, Stochastic, ADX, OBV, Volume, ADR
  - Current values and recommendations
- Charts are sent as images to Telegram and as attachments in emails.

---

## Email Reporting

- If an email is provided with `/my-status` or `/my-analyze`, the bot sends a detailed HTML report with:
  - Fundamental and technical analysis (one indicator per line)
  - All generated charts as attachments
- Email formatting uses HTML for clear, readable output.
- SMTP credentials are loaded from `config/donotshare/donotshare.py`.

---

## Error Handling & Logging

- All actions and errors are logged to `logs/log/my_screener.log`.
- User-facing errors are clear (e.g., "No data available for TICKER", "Failed to generate chart for TICKER").
- Temporary files for charts are only deleted after successful message/email delivery, preventing file access errors.
- If a chart or analysis fails, the bot continues processing other tickers.

---

## Architecture & Key Files

- `src/screener/telegram/bot.py`: Main bot logic, command handlers, Telegram and email integration
- `src/screener/telegram/technicals.py`: Technical indicator calculation and recommendation logic
- `src/screener/telegram/chart.py`: Chart generation for YF and BNC tickers
- `src/screener/telegram/combine.py`: Formatting for comprehensive analysis (email/Telegram)
- `src/notification/emailer.py`: Email sending logic
- `config/screener/my_screener.json`: Per-user ticker storage
- `logs/log/my_screener.log`: Log file for all bot actions/errors

---

## Usage Examples

### Add and Analyze a Ticker
```
/my-add -yf AAPL
/my-status
```

### Analyze and Email Results
```
/my-analyze -bnc BTCUSDT user@email.com
```

### List All Tickers
```
/my-list
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

For issues, feature requests, or contributions, please open an issue or pull request on the project repository. 