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
- **Admin Panel:** Web-based interface for user management, approvals, alerts, schedules, and system monitoring.
- **Case-Insensitive Commands:** All bot commands work regardless of case (e.g., `/REPORT`, `/Report`, `/report`).
- **User Approval System:** Secure workflow for approving users to access restricted features.
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
| `/request_approval`                     | Request admin approval after email verification.          | `/request_approval`            |

### Reports

| Command                                 | Description                                               | Example Usage                  |
|-----------------------------------------|-----------------------------------------------------------|-------------------------------|
| `/report TICKER1 TICKER2 ... [flags]`   | Get report for tickers. Use flags to customize.           | `/report AAPL BTCUSDT`        |
|                                         |                                                           | `/report MSFT -email`         |
|                                         |                                                           | `/report TSLA -indicators=RSI,MACD,MA50 -email` |
|                                         |                                                           | `/report BTCUSDT -provider=bnc -period=1y`      |
|                                         |                                                           | `/report -config='{"report_type":"analysis","tickers":["AAPL","MSFT"],"period":"1y","indicators":["RSI","MACD"],"email":true}'` |
| `/reportlist`                           | List your recent or scheduled reports.                    | `/reportlist`                 |

**Supported Flags:**
- `-email`: Send report to your registered email.
- `-indicators=...`: Comma-separated indicators (e.g., RSI,MACD,MA50,PE,EPS).
- `-period=...`: Data period (e.g., 3mo, 1y, 2y). Default: 2y.
- `-interval=...`: Data interval (e.g., 1d, 15m). Default: 1d.
- `-provider=...`: Data provider (e.g., yf for Yahoo, bnc for Binance). If not specified: yf for tickers ≤5 chars, bnc for crypto (>5 chars).
- `-config=JSON_STRING`: Use JSON configuration for advanced options.

**JSON Configuration:**
The `/report` command supports advanced JSON configuration for complex report requirements:

```json
{
  "report_type": "analysis",
  "tickers": ["AAPL", "MSFT"],
  "period": "1y",
  "interval": "1d",
  "provider": "yf",
  "indicators": ["RSI", "MACD", "BollingerBands"],
  "fundamental_indicators": ["PE", "PB", "ROE"],
  "email": true,
  "include_chart": true,
  "include_fundamentals": true,
  "include_technicals": true
}
```

**Supported JSON Fields:**
- `report_type`: "analysis", "screener", or "custom"
- `tickers`: Array of ticker symbols
- `period`: Data period ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
- `interval`: Data interval ("1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo")
- `provider`: Data provider ("yf", "alpha_vantage", "polygon")
- `indicators`: Array of technical indicators
- `fundamental_indicators`: Array of fundamental indicators
- `email`: Boolean to send to email
- `include_chart`: Boolean to include charts
- `include_fundamentals`: Boolean to include fundamental analysis
- `include_technicals`: Boolean to include technical analysis

### Fundamental Screener

| Command                                 | Description                                               | Example Usage                  |
|-----------------------------------------|-----------------------------------------------------------|-------------------------------|
| `/schedules screener LIST_TYPE [flags]` | Schedule fundamental screener for undervalued stocks.     | `/schedules screener us_small_cap -email` |

### Enhanced Screener with FMP Integration

| Command                                 | Description                                               | Example Usage                  |
|-----------------------------------------|-----------------------------------------------------------|-------------------------------|
| `/schedules enhanced_screener CONFIG_JSON` | Schedule enhanced screener with FMP pre-filtering and advanced analysis. | `/schedules enhanced_screener '{"screener_type":"hybrid","list_type":"us_medium_cap","fmp_criteria":{"marketCapMoreThan":2000000000,"peRatioLessThan":20,"limit":50},"fundamental_criteria":[{"indicator":"PE","operator":"max","value":15,"weight":1.0,"required":true}],"max_results":10,"min_score":7.0}'` |
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

### Enhanced Screener (JSON Configuration)

| Command                                 | Description                                               | Example Usage                  |
|-----------------------------------------|-----------------------------------------------------------|-------------------------------|
| `/schedules enhanced_screener CONFIG_JSON` | Schedule enhanced screener with JSON configuration.       | `/schedules enhanced_screener '{"screener_type":"hybrid","list_type":"us_medium_cap",...}'` |

**Enhanced Screener Types:**
- `fundamental`: Fundamental analysis only (P/E, ROE, etc.)
- `technical`: Technical analysis only (RSI, MACD, etc.)
- `hybrid`: Combined fundamental and technical analysis

**Supported Fundamental Criteria:**
- **Operators**: `max`, `min`, `range`
- **Indicators**: PE, Forward_PE, PB, PS, PEG, Debt_Equity, Current_Ratio, Quick_Ratio, ROE, ROA, Operating_Margin, Profit_Margin, Revenue_Growth, Net_Income_Growth, Free_Cash_Flow, Dividend_Yield, Payout_Ratio

**Supported Technical Criteria:**
- **Indicators**: RSI, MACD, BollingerBands, SMA, EMA, ADX, ATR, Stochastic, WilliamsR, CCI, ROC, MFI
- **Conditions**: `<`, `>`, `range`, `above`, `below_lower_band`, `not_above_upper_band`, `between_bands`

**Enhanced Screener Features:**
- **Weighted Scoring**: Each criterion has a weight (0.0-1.0) for fine-tuned control
- **Required vs Optional**: Mark criteria as required or optional
- **Composite Scoring**: 0-10 scale with configurable minimum score
- **DCF Valuation**: Automatic discounted cash flow calculations
- **Buy/Sell/Hold Recommendations**: Based on composite scores
- **Flexible Timeframes**: Configurable periods and intervals
- **FMP Integration**: Professional pre-filtering using Financial Modeling Prep API

## FMP Integration

The Enhanced Screener now includes **FMP (Financial Modeling Prep) Integration** for professional-grade stock screening. FMP provides sophisticated pre-filtering capabilities that dramatically improve screening performance and accuracy.

### **FMP Integration Benefits:**

🚀 **90% Performance Improvement**: Single FMP API call vs. processing all tickers individually
📊 **Professional Screening**: Uses FMP's sophisticated screening algorithms
🎯 **Pre-filtered Results**: Only analyze stocks that meet your initial criteria
⚡ **Faster Execution**: Process only relevant tickers instead of entire lists
💰 **Cost Effective**: Reduce API calls and processing time

### **How FMP Integration Works:**

```
Stage 1: FMP Pre-filtering (Single API call)
├── Use FMP criteria to get pre-filtered tickers
├── Criteria from user JSON OR predefined strategies OR defaults
└── Returns list of tickers meeting FMP criteria

Stage 2: Enhanced Analysis (Only for FMP results)
├── Get OHLCV data for FMP-filtered tickers
├── Calculate technical indicators
├── Apply additional fundamental analysis
└── Generate final composite scores
```

### **FMP Configuration Options:**

#### **1. Custom FMP Criteria:**
```json
{
  "fmp_criteria": {
    "marketCapMoreThan": 2000000000,
    "peRatioLessThan": 20,
    "returnOnEquityMoreThan": 0.12,
    "debtToEquityLessThan": 0.5,
    "limit": 50
  }
}
```

#### **2. Predefined FMP Strategies:**
```json
{
  "fmp_strategy": "conservative_value"
}
```

#### **3. Default Configurations:**
- **fundamental**: Basic fundamental screening defaults
- **hybrid**: Balanced fundamental + technical defaults
- **technical**: Technical screening defaults
- **conservative_value**: Conservative value defaults
- **growth**: Growth stock defaults

### **Available FMP Strategies:**

| Strategy | Description | Key Criteria |
|----------|-------------|--------------|
| `conservative_value` | Conservative value stocks with strong fundamentals | PE < 12, PB < 1.2, ROE > 15% |
| `growth_at_reasonable_price` | Growth stocks with reasonable valuations | PE < 20, ROE > 12%, Revenue Growth > 10% |
| `dividend_aristocrats` | High-quality dividend-paying stocks | Dividend Yield > 4%, Payout Ratio < 60% |
| `deep_value` | Deep value stocks with very low valuations | PE < 8, PB < 0.8, PS < 1.0 |
| `quality_growth` | High-quality growth stocks | ROE > 18%, Operating Margin > 20% |
| `small_cap_value` | Small-cap value stocks with growth potential | Market Cap: $500M-$2B, PE < 15 |
| `defensive_stocks` | Defensive stocks with low volatility | Beta < 0.8, Debt/Equity < 0.4 |
| `momentum_quality` | Quality stocks with positive momentum | ROE > 15%, Revenue Growth > 12% |
| `international_value` | International value stocks | PE < 12, ROE > 12% |
| `tech_growth` | Technology growth stocks | PE < 30, Revenue Growth > 15% |
| `financial_stocks` | Financial sector stocks | PE < 15, ROE > 12% |

### **FMP Criteria Reference:**

| Criteria | Description | Example |
|----------|-------------|---------|
| `marketCapMoreThan` | Minimum market capitalization | `2000000000` (2B) |
| `marketCapLowerThan` | Maximum market capitalization | `10000000000` (10B) |
| `peRatioLessThan` | Maximum P/E ratio | `15` |
| `peRatioMoreThan` | Minimum P/E ratio | `8` |
| `priceToBookRatioLessThan` | Maximum P/B ratio | `1.5` |
| `returnOnEquityMoreThan` | Minimum ROE | `0.15` (15%) |
| `debtToEquityLessThan` | Maximum debt/equity | `0.5` |
| `currentRatioMoreThan` | Minimum current ratio | `1.5` |
| `dividendYieldMoreThan` | Minimum dividend yield | `0.03` (3%) |
| `betaLessThan` | Maximum beta | `1.0` |
| `limit` | Maximum number of results | `50` |

### **FMP Integration Examples:**

#### **Basic FMP Screening:**
```bash
/schedules enhanced_screener '{"screener_type": "hybrid", "list_type": "us_medium_cap"}'
```

#### **Custom FMP Criteria:**
```bash
/schedules enhanced_screener '{"fmp_criteria": {"marketCapMoreThan": 5000000000, "peRatioLessThan": 12}, "screener_type": "hybrid", "list_type": "us_medium_cap"}'
```

#### **FMP Strategy + Enhanced Analysis:**
```bash
/schedules enhanced_screener '{"fmp_strategy": "conservative_value", "fundamental_criteria": [{"indicator": "PE", "operator": "max", "value": 12}], "technical_criteria": [{"indicator": "RSI", "parameters": {"period": 14}, "condition": {"operator": "<", "value": 70}}]}'
```

#### **Advanced FMP + Technical:**
```bash
/schedules enhanced_screener '{"screener_type": "hybrid", "list_type": "us_small_cap", "fmp_criteria": {"marketCapMoreThan": 500000000, "marketCapLowerThan": 2000000000, "peRatioLessThan": 20, "returnOnEquityMoreThan": 0.10, "limit": 40}, "fundamental_criteria": [{"indicator": "PE", "operator": "max", "value": 15, "weight": 1.0, "required": true}], "technical_criteria": [{"indicator": "RSI", "parameters": {"period": 14}, "condition": {"operator": "<", "value": 70}, "weight": 0.6, "required": false}], "max_results": 20, "min_score": 6.5}'
```

### **FMP Integration Setup:**

1. **API Key**: Set `FMP_API_KEY` environment variable
2. **Configuration**: Use `fmp_criteria` or `fmp_strategy` in your screener config
3. **Execution**: Run enhanced screener with FMP integration

### **Fallback Behavior:**

If FMP is unavailable (no API key, network issues, etc.), the system automatically falls back to traditional screening methods, ensuring continuous operation.

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
| `/alerts add_indicator TICKER CONFIG_JSON [flags]` | Add indicator-based alert with JSON configuration.    | `/alerts add_indicator AAPL '{"type":"indicator","indicator":"RSI","parameters":{"period":14},"condition":{"operator":"<","value":30},"alert_action":"BUY","timeframe":"15m"}' -email` |
| `/alerts edit ALERT_ID [params]`        | Edit an alert (price/condition).                          | `/alerts edit 2 70000 below`  |
| `/alerts delete ALERT_ID`               | Delete an alert by ID.                                    | `/alerts delete 2`            |
| `/alerts pause ALERT_ID`                | Pause a specific alert.                                   | `/alerts pause 2`             |
| `/alerts resume ALERT_ID`               | Resume a paused alert.                                    | `/alerts resume 2`            |

#### 🔄 Re-Arm Alert System (Default Behavior)

**All new price alerts automatically use the enhanced re-arm system to prevent notification spam:**

**How Re-Arm Alerts Work:**
- **Crossing Detection**: Alerts trigger only when price **crosses** the threshold (not just exceeds it)
- **Automatic Re-Arming**: Alert re-arms when price moves back across hysteresis level
- **Smart Hysteresis**: 0.25% buffer prevents noise from small fluctuations
- **No Spam**: No repeated notifications while price stays above/below threshold

**Example: AAPL "above $150" alert with 0.25% hysteresis**
1. **Setup**: Alert created and ARMED
2. **Price $149.50**: Alert remains ARMED (below threshold)
3. **Price $150.25**: Alert TRIGGERS! 📢 Notification sent, alert DISARMED
4. **Price $151.00**: No notification (alert disarmed)
5. **Price $149.63**: Alert RE-ARMS (crossed hysteresis level: 150 - 0.25%)
6. **Price $150.10**: Alert TRIGGERS again! 📢

**Notification Example:**
```
🚨 Price Alert Triggered!

Ticker: AAPL
Current Price: $150.25
Alert: above $150.00
Alert ID: #123

Alert will re-arm below $149.63.
```

**Default Re-Arm Settings:**
- **Hysteresis**: 0.25% of threshold price
- **Cooldown**: 15 minutes between triggers
- **Re-arm**: Enabled by default for all new alerts
- **Channels**: Telegram + Email (if configured)

**Alert Flags:**
- `-email`: Send alert notification to email.
- `-timeframe=...`: Set timeframe (5m, 15m, 1h, 4h, 1d). Default: 15m.
- `-action_type=...`: Set action (BUY, SELL, HOLD, notify). Default: notify.

**JSON Configuration for Indicator Alerts:**
The `/alerts add_indicator` command supports advanced JSON configuration for complex indicator-based alerts:

```json
{
  "type": "indicator",
  "indicator": "RSI",
  "parameters": {"period": 14},
  "condition": {"operator": "<", "value": 30},
  "alert_action": "BUY",
  "timeframe": "15m"
}
```

**Supported JSON Fields:**
- `type`: "indicator" or "price"
- `indicator`: Technical indicator name ("RSI", "MACD", "BollingerBands", "SMA", "EMA", "ADX", "ATR", "Stochastic", "WilliamsR")
- `parameters`: Indicator-specific parameters (e.g., {"period": 14} for RSI)
- `condition`: Alert condition with operator and value
- `alert_action`: Action to take ("BUY", "SELL", "HOLD", "notify")
- `timeframe`: Data timeframe ("5m", "15m", "1h", "4h", "1d")

**Complex Alert Examples:**

**RSI Oversold Alert:**
```json
{
  "type": "indicator",
  "indicator": "RSI",
  "parameters": {"period": 14},
  "condition": {"operator": "<", "value": 30},
  "alert_action": "BUY",
  "timeframe": "15m"
}
```

**Bollinger Bands Alert:**
```json
{
  "type": "indicator",
  "indicator": "BollingerBands",
  "parameters": {"period": 20, "deviation": 2},
  "condition": {"operator": "below_lower_band"},
  "alert_action": "BUY",
  "timeframe": "1h"
}
```

**MACD Crossover Alert:**
```json
{
  "type": "indicator",
  "indicator": "MACD",
  "parameters": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
  "condition": {"operator": "crossover"},
  "alert_action": "BUY",
  "timeframe": "4h"
}
```

**Complex AND Logic Alert:**
```json
{
  "type": "indicator",
  "logic": "AND",
  "conditions": [
    {
      "indicator": "RSI",
      "parameters": {"period": 14},
      "condition": {"operator": "<", "value": 30}
    },
    {
      "indicator": "BollingerBands",
      "parameters": {"period": 20, "deviation": 2},
      "condition": {"operator": "below_lower_band"}
    }
  ],
  "alert_action": "BUY",
  "timeframe": "15m"
}
```

**Price Alert with JSON:**
```json
{
  "type": "price",
  "threshold": 150.00,
  "condition": "below",
  "alert_action": "notify",
  "timeframe": "15m"
}
```

**Supported Condition Operators:**
- **Comparison**: `<`, `<=`, `>`, `>=`, `==`, `!=`
- **Bollinger Bands**: `above_upper_band`, `below_lower_band`, `between_bands`
- **MACD**: `crossover`, `crossunder`, `above_signal`, `below_signal`
- **Moving Averages**: `above_ma`, `below_ma`, `ma_crossover`, `ma_crossunder`

### Scheduled Reports

| Command                                 | Description                                               | Example Usage                  |
|-----------------------------------------|-----------------------------------------------------------|-------------------------------|
| `/schedules`                            | List all your scheduled reports.                          | `/schedules`                  |
| `/schedules add TICKER TIME [flags]`    | Schedule a report at a specific time (UTC, 24h).          | `/schedules add AAPL 09:00 -email` |
| `/schedules add_json CONFIG_JSON`       | Schedule with advanced JSON configuration.                | `/schedules add_json '{"type":"report","ticker":"AAPL","scheduled_time":"09:00","period":"1y","interval":"1d","email":true}'` |
| `/schedules edit SCHEDULE_ID [params]`  | Edit a scheduled report.                                  | `/schedules edit 1 10:00`     |
| `/schedules delete SCHEDULE_ID`         | Delete a scheduled report by ID.                          | `/schedules delete 1`         |
| `/schedules pause SCHEDULE_ID`          | Pause a scheduled report.                                 | `/schedules pause 1`          |
| `/schedules resume SCHEDULE_ID`         | Resume a paused scheduled report.                         | `/schedules resume 1`         |
| `/schedules add_json CONFIG_JSON`       | Schedule with advanced JSON configuration (single/multiple tickers).| `/schedules add_json '{"type":"report","tickers":["AAPL","MSFT"],"scheduled_time":"09:00","period":"1y","interval":"1d","indicators":"RSI,MACD","email":true}'` |

**Schedule Flags:**
- `-email`: Send report to email.
- `-indicators=...`: Comma-separated indicators (e.g., RSI,MACD,MA50,PE,EPS).
- `-period=...`: Data period (e.g., 3mo, 1y, 2y). Default: 2y.
- `-interval=...`: Data interval (e.g., 1d, 15m). Default: 1d.
- `-provider=...`: Data provider (e.g., yf for Yahoo, bnc for Binance).

**JSON Configuration for Schedules:**
The `/schedules add_json` command supports advanced JSON configuration for complex scheduling requirements, including both single and multiple ticker reports:

**Single Ticker Report:**
```json
{
  "type": "report",
  "ticker": "AAPL",
  "scheduled_time": "09:00",
  "period": "1y",
  "interval": "1d",
  "indicators": ["RSI", "MACD", "BollingerBands"],
  "email": true
}
```

**Multiple Ticker Report:**
```json
{
  "type": "report",
  "tickers": ["AAPL", "MSFT", "GOOGL"],
  "scheduled_time": "09:00",
  "period": "1y",
  "interval": "1d",
  "indicators": "RSI,MACD,BollingerBands",
  "provider": "yf",
  "email": true
}
```

**Supported JSON Fields for Report Schedules:**
- `type`: "report" (required)
- `ticker`: Single ticker symbol (use either `ticker` OR `tickers`, not both)
- `tickers`: Array of ticker symbols (use either `ticker` OR `tickers`, not both)
- `scheduled_time`: Time in HH:MM format (24h UTC) (required)
- `period`: Data period ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
- `interval`: Data interval ("1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo")
- `indicators`: Comma-separated technical indicators or array
- `provider`: Data provider ("yf", "alpha_vantage", "polygon")
- `email`: Boolean to send to email

**Supported JSON Fields:**
- `type`: "report" or "screener"
- `ticker`: Ticker symbol (for reports) or list_type (for screeners)
- `scheduled_time`: Time in HH:MM format (24h UTC)
- `period`: Data period ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
- `interval`: Data interval ("1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo")
- `indicators`: Array of indicators or comma-separated string
- `email`: Boolean to send to email
- `provider`: Data provider ("yf", "alpha_vantage", "polygon")

**Schedule Examples:**

**Single Ticker Report Schedule:**
```json
{
  "type": "report",
  "ticker": "AAPL",
  "scheduled_time": "09:00",
  "period": "1y",
  "interval": "1d",
  "email": true
}
```

**Multiple Ticker Report Schedule:**
```json
{
  "type": "report",
  "tickers": ["AAPL", "MSFT", "GOOGL"],
  "scheduled_time": "09:00",
  "period": "1y",
  "interval": "1d",
  "indicators": "RSI,MACD,BollingerBands",
  "email": true
}
```

**Advanced Report Schedule:**
```json
{
  "type": "report",
  "ticker": "TSLA",
  "scheduled_time": "16:30",
  "period": "6mo",
  "interval": "1h",
  "indicators": ["RSI", "MACD", "BollingerBands"],
  "email": true
}
```

**Crypto Report Schedule:**
```json
{
  "type": "report",
  "ticker": "BTCUSDT",
  "scheduled_time": "08:00",
  "period": "3mo",
  "interval": "4h",
  "indicators": ["RSI", "MACD", "BollingerBands"],
  "provider": "bnc",
  "email": true
}
```

**Fundamental Screener Schedule:**
```json
{
  "type": "screener",
  "list_type": "us_small_cap",
  "scheduled_time": "08:00",
  "period": "1y",
  "interval": "1d",
  "indicators": "PE,PB,ROE",
  "email": true
}
```

**Enhanced Screener Schedule (Hybrid):**
```json
{
  "screener_type": "hybrid",
  "list_type": "us_medium_cap",
  "fundamental_criteria": [
    {
      "indicator": "PE",
      "operator": "max",
      "value": 15,
      "weight": 1.0,
      "required": true
    },
    {
      "indicator": "ROE",
      "operator": "min",
      "value": 12,
      "weight": 0.8,
      "required": false
    }
  ],
  "technical_criteria": [
    {
      "indicator": "RSI",
      "parameters": {"period": 14},
      "condition": {"operator": "<", "value": 70},
      "weight": 0.6,
      "required": false
    },
    {
      "indicator": "BollingerBands",
      "parameters": {"period": 20, "deviation": 2},
      "condition": {"operator": "not_above_upper_band"},
      "weight": 0.5,
      "required": false
    }
  ],
  "period": "6mo",
  "interval": "1d",
  "max_results": 15,
  "min_score": 6.5,
  "include_technical_analysis": true,
  "include_fundamental_analysis": true,
  "email": true
}
```

**Technical-Only Screener:**
```json
{
  "screener_type": "technical",
  "list_type": "us_large_cap",
  "technical_criteria": [
    {
      "indicator": "RSI",
      "parameters": {"period": 14},
      "condition": {"operator": "<", "value": 30},
      "weight": 1.0,
      "required": true
    },
    {
      "indicator": "MACD",
      "parameters": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
      "condition": {"operator": "above_signal"},
      "weight": 0.8,
      "required": false
    }
  ],
  "period": "3mo",
  "interval": "1d",
  "max_results": 8,
  "min_score": 7.5,
  "include_technical_analysis": true,
  "include_fundamental_analysis": false,
  "email": false
}
```

**Custom List Screener Schedule:**
```json
{
  "type": "screener",
  "list_type": "custom_list",
  "scheduled_time": "09:30",
  "period": "1y",
  "interval": "1d",
  "indicators": "PE,PB,ROE,ROA,Dividend_Yield",
  "email": true
}
```

**Multiple Ticker Report Schedule:**
```json
{
  "type": "report",
  "tickers": ["AAPL", "MSFT", "GOOGL"],
  "scheduled_time": "10:00",
  "period": "1y",
  "interval": "1d",
  "indicators": ["RSI", "MACD"],
  "email": true
}
```

### Admin Commands (restricted)

| Command                                 | Description                                               | Example Usage                  |
|-----------------------------------------|-----------------------------------------------------------|-------------------------------|
| `/admin users`                          | List all registered users and emails.                     | `/admin users`                |
| `/admin listusers`                      | List all users as telegram_user_id - email pairs.         | `/admin listusers`            |
| `/admin resetemail TELEGRAM_USER_ID`    | Reset a user's email.                                     | `/admin resetemail 123456789` |
| `/admin verify TELEGRAM_USER_ID`        | Manually verify a user's email.                           | `/admin verify 123456789`     |
| `/admin approve TELEGRAM_USER_ID`       | Approve a user for access to restricted features.         | `/admin approve 123456789`    |
| `/admin reject TELEGRAM_USER_ID`        | Reject a user's approval request.                         | `/admin reject 123456789`     |
| `/admin pending`                        | List users waiting for approval.                          | `/admin pending`              |
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

## Command Processing Notes

**Important:** All commands are case-insensitive for improved user experience:
- **Commands**: `/REPORT`, `/Report`, `/report` all work the same
- **Tickers**: Automatically converted to uppercase (e.g., `aapl` → `AAPL`)
- **Actions**: Converted to lowercase for consistency (e.g., `SCREENER` → `screener`)
- **Parameters**: Preserved in original case for flexibility

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

## Enhanced Screener Examples

### 1. Conservative Value Screener (Fundamental-Only)
**Purpose**: Find undervalued stocks with strong fundamentals and low debt
```bash
/schedules enhanced_screener '{
  "screener_type": "fundamental",
  "list_type": "us_medium_cap",
  "fundamental_criteria": [
    {
      "indicator": "PE",
      "operator": "max",
      "value": 15,
      "weight": 1.0,
      "required": true
    },
    {
      "indicator": "PB",
      "operator": "max",
      "value": 1.5,
      "weight": 0.9,
      "required": true
    },
    {
      "indicator": "PS",
      "operator": "max",
      "value": 1.0,
      "weight": 0.8,
      "required": false
    },
    {
      "indicator": "ROE",
      "operator": "min",
      "value": 15,
      "weight": 0.9,
      "required": false
    },
    {
      "indicator": "ROA",
      "operator": "min",
      "value": 8,
      "weight": 0.7,
      "required": false
    },
    {
      "indicator": "Debt_Equity",
      "operator": "max",
      "value": 0.5,
      "weight": 0.8,
      "required": false
    },
    {
      "indicator": "Current_Ratio",
      "operator": "min",
      "value": 1.5,
      "weight": 0.6,
      "required": false
    },
    {
      "indicator": "Free_Cash_Flow",
      "operator": "min",
      "value": 0,
      "weight": 0.8,
      "required": true
    },
    {
      "indicator": "Operating_Margin",
      "operator": "min",
      "value": 10,
      "weight": 0.7,
      "required": false
    },
    {
      "indicator": "Revenue_Growth",
      "operator": "min",
      "value": 5,
      "weight": 0.6,
      "required": false
    }
  ],
  "period": "2y",
  "interval": "1d",
  "provider": "yf",
  "max_results": 20,
  "min_score": 7.5,
  "include_fundamental_analysis": true,
  "include_technical_analysis": false,
  "email": true
}'
```

### 2. Growth + Momentum Screener (Hybrid)
**Purpose**: Find growth stocks with positive technical momentum
```bash
/schedules enhanced_screener '{
  "screener_type": "hybrid",
  "list_type": "us_large_cap",
  "fundamental_criteria": [
    {
      "indicator": "PEG",
      "operator": "max",
      "value": 1.5,
      "weight": 1.0,
      "required": true
    },
    {
      "indicator": "Revenue_Growth",
      "operator": "min",
      "value": 15,
      "weight": 0.9,
      "required": true
    },
    {
      "indicator": "Net_Income_Growth",
      "operator": "min",
      "value": 10,
      "weight": 0.8,
      "required": false
    },
    {
      "indicator": "ROE",
      "operator": "min",
      "value": 12,
      "weight": 0.7,
      "required": false
    },
    {
      "indicator": "Operating_Margin",
      "operator": "min",
      "value": 8,
      "weight": 0.6,
      "required": false
    },
    {
      "indicator": "Free_Cash_Flow",
      "operator": "min",
      "value": 0,
      "weight": 0.8,
      "required": true
    }
  ],
  "technical_criteria": [
    {
      "indicator": "RSI",
      "parameters": {"period": 14},
      "condition": {"operator": "range", "min": 40, "max": 80},
      "weight": 0.7,
      "required": false
    },
    {
      "indicator": "MACD",
      "parameters": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
      "condition": {"operator": "above_signal"},
      "weight": 0.8,
      "required": false
    },
    {
      "indicator": "SMA",
      "parameters": {"period": 20},
      "condition": {"operator": "above", "value": "close"},
      "weight": 0.6,
      "required": false
    },
    {
      "indicator": "EMA",
      "parameters": {"period": 50},
      "condition": {"operator": "above", "value": "close"},
      "weight": 0.5,
      "required": false
    },
    {
      "indicator": "BollingerBands",
      "parameters": {"period": 20, "deviation": 2},
      "condition": {"operator": "not_above_upper_band"},
      "weight": 0.6,
      "required": false
    }
  ],
  "period": "1y",
  "interval": "1d",
  "provider": "yf",
  "max_results": 15,
  "min_score": 7.0,
  "include_fundamental_analysis": true,
  "include_technical_analysis": true,
  "email": true
}'
```

### 3. Dividend Aristocrat Screener (Fundamental-Only)
**Purpose**: Find high-quality dividend-paying stocks
```bash
/schedules enhanced_screener '{
  "screener_type": "fundamental",
  "list_type": "us_large_cap",
  "fundamental_criteria": [
    {
      "indicator": "Dividend_Yield",
      "operator": "min",
      "value": 3.0,
      "weight": 1.0,
      "required": true
    },
    {
      "indicator": "Payout_Ratio",
      "operator": "max",
      "value": 60,
      "weight": 0.9,
      "required": true
    },
    {
      "indicator": "ROE",
      "operator": "min",
      "value": 12,
      "weight": 0.8,
      "required": false
    },
    {
      "indicator": "ROA",
      "operator": "min",
      "value": 6,
      "weight": 0.7,
      "required": false
    },
    {
      "indicator": "Debt_Equity",
      "operator": "max",
      "value": 0.6,
      "weight": 0.8,
      "required": false
    },
    {
      "indicator": "Current_Ratio",
      "operator": "min",
      "value": 1.2,
      "weight": 0.6,
      "required": false
    },
    {
      "indicator": "Operating_Margin",
      "operator": "min",
      "value": 12,
      "weight": 0.7,
      "required": false
    },
    {
      "indicator": "Profit_Margin",
      "operator": "min",
      "value": 8,
      "weight": 0.6,
      "required": false
    },
    {
      "indicator": "Free_Cash_Flow",
      "operator": "min",
      "value": 0,
      "weight": 0.9,
      "required": true
    },
    {
      "indicator": "Revenue_Growth",
      "operator": "min",
      "value": 3,
      "weight": 0.5,
      "required": false
    }
  ],
  "period": "2y",
  "interval": "1d",
  "provider": "yf",
  "max_results": 25,
  "min_score": 8.0,
  "include_fundamental_analysis": true,
  "include_technical_analysis": false,
  "email": true
}'
```

### 4. Technical Momentum Screener (Technical-Only)
**Purpose**: Find stocks with strong technical momentum signals
```bash
/schedules enhanced_screener '{
  "screener_type": "technical",
  "list_type": "us_medium_cap",
  "technical_criteria": [
    {
      "indicator": "RSI",
      "parameters": {"period": 14},
      "condition": {"operator": "range", "min": 50, "max": 75},
      "weight": 1.0,
      "required": true
    },
    {
      "indicator": "MACD",
      "parameters": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
      "condition": {"operator": "above_signal"},
      "weight": 0.9,
      "required": true
    },
    {
      "indicator": "SMA",
      "parameters": {"period": 20},
      "condition": {"operator": "above", "value": "close"},
      "weight": 0.8,
      "required": false
    },
    {
      "indicator": "EMA",
      "parameters": {"period": 50},
      "condition": {"operator": "above", "value": "close"},
      "weight": 0.7,
      "required": false
    },
    {
      "indicator": "BollingerBands",
      "parameters": {"period": 20, "deviation": 2},
      "condition": {"operator": "between_bands"},
      "weight": 0.6,
      "required": false
    },
    {
      "indicator": "ADX",
      "parameters": {"period": 14},
      "condition": {"operator": ">", "value": 25},
      "weight": 0.5,
      "required": false
    },
    {
      "indicator": "ATR",
      "parameters": {"period": 14},
      "condition": {"operator": ">", "value": 0},
      "weight": 0.4,
      "required": false
    },
    {
      "indicator": "Stochastic",
      "parameters": {"k_period": 14, "d_period": 3},
      "condition": {"operator": "range", "min": 20, "max": 80},
      "weight": 0.5,
      "required": false
    },
    {
      "indicator": "WilliamsR",
      "parameters": {"period": 14},
      "condition": {"operator": "range", "min": -80, "max": -20},
      "weight": 0.4,
      "required": false
    },
    {
      "indicator": "CCI",
      "parameters": {"period": 20},
      "condition": {"operator": "range", "min": -100, "max": 100},
      "weight": 0.3,
      "required": false
    }
  ],
  "period": "6mo",
  "interval": "1d",
  "provider": "yf",
  "max_results": 30,
  "min_score": 7.5,
  "include_fundamental_analysis": false,
  "include_technical_analysis": true,
  "email": false
}'
```

### 5. Deep Value + Oversold Screener (Hybrid)
**Purpose**: Find deeply undervalued stocks that are technically oversold
```bash
/schedules enhanced_screener '{
  "screener_type": "hybrid",
  "list_type": "us_small_cap",
  "fundamental_criteria": [
    {
      "indicator": "PE",
      "operator": "max",
      "value": 10,
      "weight": 1.0,
      "required": true
    },
    {
      "indicator": "PB",
      "operator": "max",
      "value": 1.0,
      "weight": 1.0,
      "required": true
    },
    {
      "indicator": "PS",
      "operator": "max",
      "value": 0.8,
      "weight": 0.9,
      "required": false
    },
    {
      "indicator": "PEG",
      "operator": "max",
      "value": 1.0,
      "weight": 0.8,
      "required": false
    },
    {
      "indicator": "ROE",
      "operator": "min",
      "value": 10,
      "weight": 0.7,
      "required": false
    },
    {
      "indicator": "ROA",
      "operator": "min",
      "value": 5,
      "weight": 0.6,
      "required": false
    },
    {
      "indicator": "Debt_Equity",
      "operator": "max",
      "value": 0.4,
      "weight": 0.8,
      "required": false
    },
    {
      "indicator": "Current_Ratio",
      "operator": "min",
      "value": 1.3,
      "weight": 0.6,
      "required": false
    },
    {
      "indicator": "Quick_Ratio",
      "operator": "min",
      "value": 1.0,
      "weight": 0.5,
      "required": false
    },
    {
      "indicator": "Operating_Margin",
      "operator": "min",
      "value": 8,
      "weight": 0.7,
      "required": false
    },
    {
      "indicator": "Profit_Margin",
      "operator": "min",
      "value": 5,
      "weight": 0.6,
      "required": false
    },
    {
      "indicator": "Free_Cash_Flow",
      "operator": "min",
      "value": 0,
      "weight": 0.9,
      "required": true
    },
    {
      "indicator": "Revenue_Growth",
      "operator": "min",
      "value": 3,
      "weight": 0.5,
      "required": false
    },
    {
      "indicator": "Net_Income_Growth",
      "operator": "min",
      "value": 2,
      "weight": 0.5,
      "required": false
    }
  ],
  "technical_criteria": [
    {
      "indicator": "RSI",
      "parameters": {"period": 14},
      "condition": {"operator": "<", "value": 35},
      "weight": 0.8,
      "required": false
    },
    {
      "indicator": "BollingerBands",
      "parameters": {"period": 20, "deviation": 2},
      "condition": {"operator": "below_lower_band"},
      "weight": 0.7,
      "required": false
    },
    {
      "indicator": "Stochastic",
      "parameters": {"k_period": 14, "d_period": 3},
      "condition": {"operator": "<", "value": 20},
      "weight": 0.6,
      "required": false
    },
    {
      "indicator": "WilliamsR",
      "parameters": {"period": 14},
      "condition": {"operator": "<", "value": -80},
      "weight": 0.5,
      "required": false
    },
    {
      "indicator": "CCI",
      "parameters": {"period": 20},
      "condition": {"operator": "<", "value": -100},
      "weight": 0.4,
      "required": false
    },
    {
      "indicator": "MACD",
      "parameters": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
      "condition": {"operator": "below_signal"},
      "weight": 0.3,
      "required": false
    }
  ],
  "period": "1y",
  "interval": "1d",
  "provider": "yf",
  "max_results": 20,
  "min_score": 6.5,
  "include_fundamental_analysis": true,
  "include_technical_analysis": true,
  "email": true
}'
```

---

## Enhanced Screener Reference Guide

### JSON Configuration Parameters

#### Core Parameters
- **`screener_type`**: `"fundamental"`, `"technical"`, or `"hybrid"`
- **`list_type`**: `"us_small_cap"`, `"us_medium_cap"`, `"us_large_cap"`, `"swiss_shares"`, `"custom_list"`
- **`period`**: `"1d"`, `"5d"`, `"1mo"`, `"3mo"`, `"6mo"`, `"1y"`, `"2y"`, `"5y"`, `"10y"`, `"ytd"`, `"max"`
- **`interval`**: `"1m"`, `"2m"`, `"5m"`, `"15m"`, `"30m"`, `"60m"`, `"90m"`, `"1h"`, `"1d"`, `"5d"`, `"1wk"`, `"1mo"`, `"3mo"`
- **`provider`**: `"yf"`, `"alpha_vantage"`, `"polygon"`
- **`max_results`**: Number of stocks to return (1-50)
- **`min_score`**: Minimum composite score (0-10)
- **`email`**: `true`/`false` to send results via email

#### Fundamental Criteria Parameters
- **`indicator`**: Fundamental metric to evaluate
- **`operator`**: `"min"`, `"max"`, or `"range"`
- **`value`**: Single value or range object `{"min": x, "max": y}`
- **`weight`**: Importance weight (0.0-1.0)
- **`required`**: `true`/`false` - if true, stock must meet this criterion

#### Technical Criteria Parameters
- **`indicator`**: Technical indicator to evaluate
- **`parameters`**: Indicator-specific parameters (periods, etc.)
- **`condition`**: Evaluation condition with operator and values
- **`weight`**: Importance weight (0.0-1.0)
- **`required`**: `true`/`false` - if true, stock must meet this criterion

### Supported Fundamental Indicators

#### Valuation Metrics
- **`PE`**: Price-to-Earnings ratio
- **`Forward_PE`**: Forward P/E ratio
- **`PB`**: Price-to-Book ratio
- **`PS`**: Price-to-Sales ratio
- **`PEG`**: Price/Earnings-to-Growth ratio

#### Financial Health
- **`Debt_Equity`**: Debt-to-Equity ratio
- **`Current_Ratio`**: Current ratio
- **`Quick_Ratio`**: Quick ratio

#### Profitability
- **`ROE`**: Return on Equity
- **`ROA`**: Return on Assets
- **`Operating_Margin`**: Operating margin
- **`Profit_Margin`**: Profit margin

#### Growth
- **`Revenue_Growth`**: Revenue growth rate
- **`Net_Income_Growth`**: Net income growth rate

#### Cash Flow & Dividends
- **`Free_Cash_Flow`**: Free cash flow
- **`Dividend_Yield`**: Dividend yield
- **`Payout_Ratio`**: Dividend payout ratio

### Supported Technical Indicators

#### Momentum Indicators
- **`RSI`**: Relative Strength Index
  - Parameters: `{"period": 14}`
  - Conditions: `<`, `>`, `range`
- **`MACD`**: Moving Average Convergence Divergence
  - Parameters: `{"fast_period": 12, "slow_period": 26, "signal_period": 9}`
  - Conditions: `above_signal`, `below_signal`
- **`Stochastic`**: Stochastic Oscillator
  - Parameters: `{"k_period": 14, "d_period": 3}`
  - Conditions: `<`, `>`, `range`

#### Moving Averages
- **`SMA`**: Simple Moving Average
  - Parameters: `{"period": 20}`
  - Conditions: `above`, `below`
- **`EMA`**: Exponential Moving Average
  - Parameters: `{"period": 50}`
  - Conditions: `above`, `below`

#### Volatility Indicators
- **`BollingerBands`**: Bollinger Bands
  - Parameters: `{"period": 20, "deviation": 2}`
  - Conditions: `below_lower_band`, `not_above_upper_band`, `between_bands`
- **`ATR`**: Average True Range
  - Parameters: `{"period": 14}`
  - Conditions: `<`, `>`

#### Trend Indicators
- **`ADX`**: Average Directional Index
  - Parameters: `{"period": 14}`
  - Conditions: `<`, `>`
- **`WilliamsR`**: Williams %R
  - Parameters: `{"period": 14}`
  - Conditions: `<`, `>`, `range`
- **`CCI`**: Commodity Channel Index
  - Parameters: `{"period": 20}`
  - Conditions: `<`, `>`, `range`
- **`ROC`**: Rate of Change
  - Parameters: `{"period": 10}`
  - Conditions: `<`, `>`, `range`
- **`MFI`**: Money Flow Index
  - Parameters: `{"period": 14}`
  - Conditions: `<`, `>`, `range`

### Scoring System

#### Fundamental Scoring
- Each criterion is evaluated on a 0-1 scale
- Scores are weighted by the `weight` parameter
- Required criteria must be met (score > 0)
- Final score is normalized to 0-10 scale

#### Technical Scoring
- Each indicator condition is evaluated as pass/fail (1.0/0.0)
- Scores are weighted by the `weight` parameter
- Required criteria must be met (score > 0)
- Final score is normalized to 0-10 scale

#### Composite Scoring (Hybrid)
- Fundamental weight: 70% (default)
- Technical weight: 30% (default)
- Weights can be adjusted via `include_fundamental_analysis` and `include_technical_analysis`

### Recommendations
- **STRONG_BUY**: Score ≥ 8.0
- **BUY**: Score ≥ 7.0
- **HOLD**: Score ≥ 6.0
- **WEAK_HOLD**: Score ≥ 5.0
- **SELL**: Score < 5.0

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

## Command Processing & User Experience

### Case-Insensitive Commands
All bot commands work regardless of case for improved user experience:
- **Commands**: `/REPORT`, `/Report`, `/report` all work the same
- **Tickers**: Automatically converted to uppercase (e.g., `aapl` → `AAPL`)
- **Actions**: Converted to lowercase for consistency (e.g., `SCREENER` → `screener`)
- **Parameters**: Preserved in original case for flexibility

### Smart Command Parsing
The command parser intelligently handles different argument types:
- **Tickers**: Automatically converted to uppercase for consistency
- **Actions**: Converted to lowercase for internal processing
- **Flags**: Preserved in original case for flexibility
- **Validation**: Comprehensive parameter validation and error handling

### User Approval System
Secure multi-tier access control system:
- **Public Commands**: Available to all users (`/start`, `/help`, `/register`, `/verify`)
- **Restricted Commands**: Require email verification + admin approval (`/report`, `/alerts`, `/schedules`)
- **Admin Commands**: Require admin role (`/admin` commands)
- **Approval Workflow**: Users must register, verify email, and be approved by admin

### Error Handling & User Feedback
- **Friendly Messages**: Clear, actionable error messages for users
- **Access Denied**: Clear messaging for unauthorized command attempts
- **Logging**: All errors logged for admin review and debugging
- **Recovery**: Graceful error handling with fallback mechanisms

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

### Service Layer Architecture

The Telegram bot follows clean architecture principles with proper service layer separation:

#### Service Components
- **Telegram Service (`src/data/db/services/telegram_service.py`):** Handles all database operations for users, alerts, schedules, and settings
- **Indicator Service (`src/indicators/service.py`):** Handles all technical and fundamental indicator calculations
- **Business Logic Layer (`business_logic.py`):** Uses dependency injection to access service layers
- **Background Services:** Alert monitoring and schedule processing with service integration

#### Integration Benefits
- **Clean Separation:** Business logic separated from data access and calculations
- **Testability:** Easy to mock service dependencies for unit testing
- **Maintainability:** Changes to database or calculations don't affect business logic
- **Consistency:** All modules use the same service interfaces
- **Error Handling:** Centralized error handling and logging

### System Components
- **Telegram Bot API:** Receives commands, sends messages, manages user interaction.
- **Backend Server:** Handles business logic, command parsing, user state, and orchestrates service requests.
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

### Web-Based Admin Interface
- **URL:** `http://localhost:5000` (when running `admin_panel.py`)
- **Authentication:** Admin login with username/password (set via environment variables)
- **Session Management:** Secure login/logout with session tracking

### User Management
- **User List:** View all users with Telegram ID, email, verification status, approval status
- **User Approval:** Approve/reject users who have verified their email but need access to restricted features
- **Manual Verification:** Verify users manually if needed
- **Email Reset:** Reset user emails and revoke verification
- **User Limits:** View and manage user limits for alerts and schedules

### Dashboard Features
- **Statistics Cards:** Total users, verified users, approved users, pending approvals, active alerts, active schedules, open feedback
- **Pending Approvals Section:** Quick view of users waiting for approval with direct approve/reject buttons
- **Recent Activity:** Overview of system activity

### System Management
- **Alerts Management:** View, edit, pause, resume, or delete all user alerts
- **Schedules Management:** View, edit, pause, resume, or delete all scheduled reports
- **Broadcast Messaging:** Send messages to all users or specific segments
- **Feedback Management:** View and respond to user feedback and feature requests
- **Logs & Monitoring:** View system logs, API errors, email delivery status, admin actions
- **Settings:** Configure global limits, API keys, and system settings

### Command Audit System
- **Complete Tracking:** Every command logged with full context (registered and non-registered users)
- **Performance Monitoring:** Response times measured and stored for all commands
- **Error Analysis:** Detailed error tracking and reporting for failed commands
- **User Classification:** Distinguishes between registered and non-registered users
- **Statistics Dashboard:** Comprehensive analytics and reporting on system usage
- **Filtering Capabilities:** Time-based, user-based, command-based, and status-based filtering
- **User History:** Individual user command timelines and analysis

### Enhanced Navigation
- **Dashboard Navigation:** Direct access to filtered views from dashboard stat cards
- **Quick Filters:** One-click access to common filter combinations
- **Smart Filtering:** Automatic application of relevant filters based on navigation
- **Visual Indicators:** Clear indication of current filter status and navigation state
- **Non-Registered Users:** Dedicated view for monitoring unregistered user activity

### JSON Generator Tool
- **Purpose:** Interactive web-based tool for creating complex JSON configurations for alerts, schedules, reports, and screeners
- **Access:** Available as a tab in the admin panel and as a standalone shareable file
- **Features:**
  - **Dedicated Indicator Configuration Snippet:** Reusable component for all tabs with consistent UI/UX
  - **Multiple Indicators Support:** Combine multiple technical indicators with AND/OR logic
  - **Duplicate Prevention:** Cannot add the same indicator twice to prevent conflicts
  - **Dynamic Parameter Configuration:** Real-time parameter adjustment for all indicators
  - **Visual Indicator Management:** Clear list view of added indicators with remove functionality
  - **Template System:** Quick templates for common configurations
  - **Real-time JSON Generation:** Live preview of generated JSON as you configure
  - **Command Generation:** Automatic generation of bot commands from JSON
  - **Validation:** Built-in JSON validation and error checking
  - **Copy-to-Clipboard:** Easy copying of generated configurations

#### JSON Generator Tabs

**🚨 Alerts Tab:**
- **Price Alerts:** Simple price-based alerts with above/below conditions
- **Indicator Alerts:** Single or multiple technical indicators with custom parameters
- **Logic Options:** Single indicator, Multiple indicators (AND), Multiple indicators (OR)
- **Supported Indicators:** RSI, MACD, Bollinger Bands, SMA, EMA, ADX, ATR, Stochastic, Williams %R, CCI, ROC, MFI
- **Parameter Configuration:** Customizable periods, deviations, and thresholds for each indicator
- **Quick Templates:** Pre-configured templates for common alert scenarios

**📅 Schedules Tab:**
- **Report Schedules:** Daily reports with multiple indicators and custom parameters
- **Screener Schedules:** Automated screening with fundamental and technical criteria
- **Enhanced Screener:** Advanced screening with multiple indicators and scoring
- **Multiple Indicators:** Support for combining multiple indicators in scheduled reports
- **Time Configuration:** Flexible scheduling with 24-hour format
- **Period Selection:** Various time periods from 1 day to 2 years

**📊 Reports Tab:**
- **Multi-Ticker Reports:** Generate reports for multiple tickers simultaneously
- **Technical Analysis:** Include various technical indicators with custom parameters
- **Fundamental Analysis:** Include fundamental indicators and ratios
- **Data Provider Selection:** Choose between Yahoo Finance, Binance, or auto-detect
- **Multiple Indicators:** Combine multiple indicators for comprehensive analysis
- **Email Integration:** Direct email delivery configuration

**🔍 Screeners Tab:**
- **Screener Types:** Fundamental, Technical, or Hybrid screening
- **List Types:** Various market cap categories and custom lists
- **Fundamental Criteria:** Market cap, P/E ratio, ROE, and other fundamental metrics
- **Technical Criteria:** RSI ranges, volume requirements, and primary indicators
- **Multiple Indicators:** Support for multiple technical indicators in screening
- **Result Limits:** Configurable maximum results and scoring thresholds

#### Multiple Indicators Feature

The JSON Generator supports combining multiple indicators across all tabs:

**Logic Options:**
- **Single Indicator:** Traditional single indicator configuration
- **Multiple Indicators (AND):** All conditions must be true simultaneously
- **Multiple Indicators (OR):** Any condition can be true

**Dynamic Management:**
- **Add/Remove Indicators:** Dynamically add or remove indicators from configurations
- **Parameter Persistence:** All indicator parameters are preserved when adding to lists
- **Visual Feedback:** Clear display of configured indicators with their parameters
- **Real-time Updates:** JSON and commands update automatically as you modify configurations

**Supported Indicators with Parameters:**
- **RSI:** Period (1-100, default: 14)
- **MACD:** Fast period (1-100, default: 12), Slow period (1-100, default: 26), Signal period (1-100, default: 9)
- **Bollinger Bands:** Period (1-100, default: 20), Standard deviation (0.1-5.0, default: 2.0)
- **SMA/EMA:** Period (1-200, default: 20)
- **ADX:** Period (1-100, default: 14)
- **ATR:** Period (1-100, default: 14)
- **Stochastic:** K period (1-100, default: 14), D period (1-20, default: 3)
- **Williams %R:** Period (1-100, default: 14)

#### Usage Examples

**Multiple Indicator Alert:**
```json
{
  "type": "indicator",
  "ticker": "AAPL",
  "conditions": [
    {
      "indicator": "RSI",
      "parameters": {"period": 14},
      "condition": {"operator": "<", "value": 30}
    },
    {
      "indicator": "MACD",
      "parameters": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
      "condition": {"operator": "crossover", "value": null}
    }
  ],
  "logic": "AND",
  "timeframe": "15m",
  "alert_action": "notify",
  "email": true
}
```

**Scheduled Report with Multiple Indicators:**
```json
{
  "type": "report",
  "tickers": ["AAPL", "MSFT", "GOOGL"],
  "scheduled_time": "09:00",
  "period": "1y",
  "interval": "1d",
  "indicators": ["RSI", "MACD", "BollingerBands"],
  "email": true
}
```

**Enhanced Screener with Technical Criteria:**
```json
{
  "screener_type": "hybrid",
  "list_type": "us_medium_cap",
  "fmp_criteria": {
    "marketCapMoreThan": 2000000000,
    "peRatioLessThan": 20,
    "returnOnEquityMoreThan": 0.12
  },
  "technical_criteria": {
    "primary_indicator": "RSI",
    "rsi_min": 30,
    "rsi_max": 70,
    "volume_min": 1.5
  },
  "max_results": 10,
  "email": true
}
```

### Security & Audit
- **Access Control:** All admin routes protected with login authentication
- **Audit Trail:** All admin actions logged with timestamps and user information
- **User Approval Workflow:** Secure process for granting access to restricted features
- **Complete Visibility:** Track all bot interactions for security and compliance

---

## Localization (i18n)

- All user-facing text uses translation keys (not raw text).
- User language stored in DB; `/language LANG` to update.
- Use `gettext`/`Babel` for message catalogs.
- Fallback to English if translation missing.
- Directory: `src/telegram_screener/locales/` with `.po`/`.mo` files.

---

## Technical Notes & Code Reuse

### Command Processing
- **Case-Insensitive Commands:** All bot commands work regardless of case (e.g., `/REPORT`, `/Report`, `/report`)
- **Smart Parsing:** Command parser intelligently handles different argument types:
  - **Tickers:** Automatically converted to uppercase (e.g., `aapl` → `AAPL`)
  - **Actions:** Converted to lowercase for consistency (e.g., `SCREENER` → `screener`)
  - **Parameters:** Preserved in original case for flexibility
- **Command Handler Registry:** Dynamic registration for extensibility

### Architecture
- **Reuse existing modules:** Data downloaders, emailer, logger, screener_db, notification manager
- **Separation of concerns:** Business logic separate from Telegram API handling
- **Unit tests:** Required for business logic and command parsing
- **Error handling:** All exceptions logged, user-friendly messages
- **Caching:** API responses cached per ticker/provider/interval
- **Downloader interface:** All providers implement a common interface
- **Notification logging:** All outgoing notifications logged

### User Approval System
- **Secure Workflow:** Users must register, verify email, and be approved by admin
- **Access Control:** Restricted commands require approval (`/report`, `/alerts`, `/schedules`)
- **Admin Interface:** Web-based approval system with real-time notifications
- **Audit Trail:** All approval actions logged with timestamps

### Command Audit System
- **Automatic Logging:** All commands automatically logged via wrapper function in `bot.py`
- **Database Schema:** `command_audit` table with comprehensive tracking fields
- **Performance Tracking:** Response times measured and stored for all commands
- **Error Handling:** Failed commands logged with detailed error messages
- **User Classification:** Distinguishes between registered and non-registered users
- **Admin Integration:** Complete audit dashboard with filtering and statistics

### Enhanced Navigation System
- **Dashboard Links:** Each stat card includes navigation links to relevant filtered views
- **Filter Support:** All management pages support URL-based filtering
- **Quick Filters:** Common filter combinations available as one-click buttons
- **Visual Feedback:** Clear indication of current filter status and navigation state
- **Non-Registered Users:** Dedicated audit page for monitoring unregistered user activity

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
- **[SERVICE_LAYER_INTEGRATION.md](SERVICE_LAYER_INTEGRATION.md)** - Service layer integration patterns, best practices, and migration guide

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

### Enhanced Screener (Immediate)

| Command                                 | Description                                               | Example Usage                  |
|-----------------------------------------|-----------------------------------------------------------|-------------------------------|
| `/screener JSON_CONFIG [-email]`        | Run enhanced screener immediately with JSON configuration.| `/screener '{"screener_type":"hybrid","list_type":"us_medium_cap","fmp_criteria":{"marketCapMoreThan":200000000,"peRatioLessThan":20},"fundamental_criteria":[{"indicator":"Revenue_Growth","operator":"min","value":0.05}],"max_results":5,"min_score":2.0}'` |
|                                         |                                                           | `/screener '{"screener_type":"fundamental","list_type":"us_small_cap","fmp_strategy":"conservative_value","max_results":10}' -email` |

**Supported Flags:**
- `-email`: Send results to your registered email

**JSON Configuration:**
The `/screener` command uses the same JSON configuration format as `/schedules enhanced_screener`. See the "Enhanced Screener with FMP Integration" section below for detailed examples.

### Fundamental Screener

**Fundamental Screener Schedule:**
```json
{
  "type": "screener",
  "list_type": "us_small_cap",
  "scheduled_time": "08:00",
  "period": "1y",
  "interval": "1d",
  "indicators": "PE,PB,ROE",
  "email": true
}
```

**Report Schedule Examples:**

**Basic Multi-Ticker Report:**
```json
{
  "tickers": ["AAPL", "MSFT"],
  "scheduled_time": "09:00",
  "period": "1y",
  "interval": "1d",
  "email": true
}
```

**Advanced Technical Report:**
```json
{
  "tickers": ["TSLA", "NVDA"],
  "scheduled_time": "16:30",
  "period": "6mo",
  "interval": "1h",
  "indicators": "RSI,MACD,BollingerBands",
  "email": true
}
```

**Daily Market Update:**
```json
{
  "tickers": ["SPY", "QQQ", "IWM"],
  "scheduled_time": "08:00",
  "period": "5d",
  "interval": "1d",
  "indicators": "RSI,SMA",
  "email": false
}
```
