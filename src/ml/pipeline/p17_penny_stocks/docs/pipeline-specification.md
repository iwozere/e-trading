````md
# Explosive Penny Stock Screener — Agent Pipeline Specification

Version: 1.0  
Target Market: NASDAQ penny stocks  
Execution Frequency: Daily (pre-market + post-market optional)  
Goal: Detect early-stage explosive growth candidates before broad retail attention

---

# 1. High-Level Objective

Build an automated agent that:

1. Loads NASDAQ penny stock universe
2. Enriches tickers with market/fundamental/news data
3. Scores each ticker across:
   - momentum
   - volume expansion
   - accumulation
   - catalyst quality
   - dilution risk
   - financial quality
4. Produces:
   - ranked candidate list
   - detailed report
   - alert notifications
5. Stores historical snapshots for backtesting and model improvement

---

# 2. Definitions

## Penny Stock Definition

Default:
- Price between $0.50 and $10.00

Configurable:
```yaml
MIN_PRICE: 0.5
MAX_PRICE: 10
````

---

# 3. Directory Structure

```text
PROJECT_ROOT/
│
├── DATA_CACHE_DIR/
│   ├── nasdaq/
│   │   ├── tickers.csv
│   │   └── ...
│   │
│   ├── market_data/
│   ├── fundamentals/
│   ├── news/
│   ├── scores/
│   └── reports/
│
├── config/
│   ├── screener.yaml
│   └── weights.yaml
│
├── agents/
│   ├── universe_agent.py
│   ├── market_agent.py
│   ├── fundamentals_agent.py
│   ├── catalyst_agent.py
│   ├── scoring_agent.py
│   ├── reporting_agent.py
│   └── notification_agent.py
│
├── models/
│
├── logs/
│
└── run_daily.py
```

---

# 4. Pipeline Architecture

```text
NASDAQ LIST
    ↓
Universe Filter
    ↓
Market Data Enrichment
    ↓
Fundamental Enrichment
    ↓
Catalyst / News Analysis
    ↓
Dilution Risk Detection
    ↓
Technical Pattern Detection
    ↓
Composite Scoring Engine
    ↓
Ranking
    ↓
Report Generation
    ↓
Notification Delivery
```

---

# 5. Universe Selection

## Input

Source:

```text
DATA_CACHE_DIR/nasdaq/
```

Supported formats:

* CSV
* parquet
* JSON

Required fields:

```text
ticker
company_name
exchange
sector
industry
```

---

# 6. Data Providers

## Market Data

Recommended:

* Polygon
* AlphaVantage
* Finnhub
* TwelveData
* Yahoo Finance fallback

Required:

```text
price
market_cap
volume
avg_volume
float
short_interest
daily_ohlcv
```

---

## Fundamental Data

Required:

```text
revenue_growth
gross_margin
cash
debt
shares_outstanding
cash_flow
eps_growth
institutional_ownership
insider_transactions
```

---

## News / Catalyst Sources

Recommended:

* Benzinga
* Finnhub News
* SEC filings
* RSS feeds
* Reddit API
* X/Twitter optional

---

# 7. Hard Filters

Reject immediately if:

## Exchange

```yaml
ALLOWED_EXCHANGES:
  - NASDAQ
```

---

## Liquidity

```yaml
MIN_DAILY_VOLUME: 500000
MIN_AVG_DOLLAR_VOLUME: 1000000
```

Formula:

```text
avg_dollar_volume = avg_volume_30d * price
```

---

## Market Cap

```yaml
MIN_MARKET_CAP: 30000000
MAX_MARKET_CAP: 2000000000
```

---

## Float

```yaml
MIN_FLOAT: 5000000
MAX_FLOAT: 50000000
```

---

## Financial Survival

Reject if:

* cash runway < 6 months
* debt/cash > 5
* bankruptcy warnings
* active delisting notices

---

# 8. Feature Engineering

# 8.1 Momentum Features

## Relative Volume

```text
relative_volume =
today_volume / avg_volume_30d
```

Ideal:

```text
> 3.0
```

---

## Price Momentum

Calculate:

```text
5d return
20d return
60d return
```

Strong setup:

```text
20d_return > 20%
```

but:

```text
NOT > 300%
```

Avoid late-stage euphoric spikes.

---

## Volatility Compression

Detect:

* Bollinger Band squeeze
* ATR contraction
* Tight consolidation ranges

Explosive moves often start after compression.

---

# 8.2 Technical Breakout Features

Detect:

* breakout above 20d high
* breakout above 50d high
* volume expansion breakout
* base breakout

Patterns:

* cup and handle
* flat base
* ascending triangle
* volatility contraction pattern

---

# 8.3 Accumulation Features

Bullish:

* multiple green high-volume days
* close near daily highs
* OBV rising
* accumulation/distribution improving

---

# 8.4 Fundamental Acceleration

## Revenue Growth Score

Strong:

```text
revenue_growth_yoy > 25%
```

Elite:

```text
> 50%
```

---

## Revenue Acceleration

Example:

```text
Q1: 20%
Q2: 40%
Q3: 70%
```

Acceleration is more important than raw growth.

---

## Profitability Transition

Huge signal:

* turning cash-flow positive
* first profitable quarter
* margin expansion

---

# 8.5 Dilution Risk Detection

CRITICAL MODULE.

Detect:

* shelf offerings
* ATM offerings
* convertible debt
* warrant issuance
* reverse splits

Penalty examples:

```yaml
ATM_OFFERING_PENALTY: -20
CONVERTIBLE_DEBT_PENALTY: -30
RECENT_REVERSE_SPLIT_PENALTY: -40
```

---

# 8.6 Catalyst Detection

Extract from:

* SEC filings
* news headlines
* earnings calls

Bullish catalyst categories:

* FDA
* AI
* defense
* nuclear
* rare earths
* contracts
* partnerships
* guidance raise
* insider buying

---

# 8.7 Social / Sentiment

Optional.

Track:

* Reddit mentions
* Stocktwits acceleration
* X/Twitter mentions

Important:
Social sentiment alone must NEVER trigger candidate selection.

Only additive.

---

# 9. Composite Scoring System

## Final Score

```text
FINAL_SCORE =
0.25 * momentum_score +
0.20 * volume_score +
0.20 * technical_score +
0.15 * fundamentals_score +
0.10 * catalyst_score +
0.05 * accumulation_score -
0.05 * dilution_risk
```

Configurable.

---

# 10. Explosive Candidate Criteria

A ticker becomes:

```text
EXPLOSIVE_CANDIDATE = TRUE
```

if:

## Mandatory Conditions

```yaml
relative_volume > 3
price_above_50dma == true
breakout_detected == true
dilution_risk < threshold
```

AND

## One of:

```yaml
revenue_growth > 30%
OR
strong_catalyst == true
OR
institutional_accumulation == true
```

---

# 11. Ranking Tiers

## Tier A — Elite

Characteristics:

* strong fundamentals
* strong technicals
* real catalyst
* low dilution risk

Potential:

* swing position
* multi-week runner

---

## Tier B — Momentum

Strong technical momentum but weaker fundamentals.

Potential:

* short-term explosive move

---

## Tier C — Speculative

Catalyst-driven only.
High risk.

---

# 12. Output Report

Generate:

## JSON

```json
{
  "ticker": "XYZ",
  "score": 87.5,
  "tier": "A",
  "signals": [
    "breakout",
    "high_relative_volume",
    "revenue_acceleration"
  ]
}
```

---

## Markdown Report

Daily:

```text
reports/YYYY-MM-DD.md
```

Include:

* top candidates
* charts
* catalyst summaries
* risk warnings

---

## CSV Export

```text
ticker,score,tier,price,rvol,float,revenue_growth
```

---

# 13. Notifications

Send:

* Telegram
* Discord
* Slack
* email

Alert only:

```yaml
MIN_ALERT_SCORE: 75
```

---

# 14. Historical Storage

Store:

* daily scores
* raw features
* alerts
* outcomes

Needed for:

* backtesting
* ML improvements
* parameter tuning

---

# 15. Backtesting Engine

Measure:

## Metrics

* win rate
* average return
* max drawdown
* Sharpe ratio
* false breakout rate

---

## Holding Period Tests

Test:

* 1d
* 5d
* 10d
* 20d

---

# 16. Optional ML Layer

Future upgrade.

## Features

* volume profile
* NLP embeddings from news
* earnings sentiment
* historical breakout success

Models:

* XGBoost
* LightGBM
* RandomForest

Goal:
Predict probability of:

```text
+50%
+100%
+200%
```

within:

```text
5-30 trading days
```

---

# 17. Risk Controls

Avoid:

* low liquidity traps
* halt-prone stocks
* serial diluters
* pump-and-dumps

Hard exclusions:

```yaml
MAX_INTRADAY_SPREAD: 8%
MAX_OFFERINGS_LAST_12M: 3
```

---

# 18. Scheduling

## Recommended Times

### Pre-market

```text
08:00 ET
```

### Midday update

```text
12:00 ET
```

### Post-market

```text
17:00 ET
```

---

# 19. Suggested Technology Stack

## Language

Python 3.11+

---

## Core Libraries

```text
pandas
numpy
scikit-learn
TA-Lib
yfinance
sqlalchemy
requests
beautifulsoup4
```

---

## Storage

Recommended:

* PostgreSQL
* DuckDB
* parquet

---

## Scheduler

Recommended:

* cron
* Airflow
* Prefect

---

# 20. Example Daily Workflow

## Step 1

Load NASDAQ universe

## Step 2

Filter penny stocks

## Step 3

Download market data

## Step 4

Compute indicators

## Step 5

Pull fundamentals

## Step 6

Analyze news + SEC filings

## Step 7

Compute dilution risk

## Step 8

Generate composite scores

## Step 9

Rank candidates

## Step 10

Send report + alerts

---

# 21. Example Tier-A Candidate

```text
Ticker: ABCD
Price: $3.20
Float: 18M
Relative Volume: 5.8x
Revenue Growth: +72%
Catalyst: Defense contract
Breakout: Yes
Dilution Risk: Low

FINAL SCORE: 91
```

---

# 22. Long-Term Improvements

Future enhancements:

* options flow
* dark pool data
* insider cluster analysis
* sector rotation engine
* LLM-based earnings call analysis
* anomaly detection
* adaptive scoring weights

---

# 23. Primary Design Philosophy

The system should prioritize:

```text
REAL EARLY-STAGE MOMENTUM
```

NOT:

* random hype
* social media pumps
* illiquid garbage stocks

Core principle:

```text
Momentum + Volume + Catalyst + Survivability
```

is where explosive asymmetric opportunities usually emerge.

```
```
