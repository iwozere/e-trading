# US Market Signal Research — Data Pipeline Specification

> Agent specification for replicating and extending the weak-signal discovery pipeline
> described in the Signal Mind project (habr.com/ru/articles/1030160), adapted for US markets.

---

## Overview

This document specifies **what to download, from where, how, and in what schema** for each data source.
The pipeline feeds a DuckDB database that an LLM agent queries via SQL to discover lagged cross-asset correlations.

Already available (skip):
- `yfinance` price downloads
- VIX data

Needs implementation:
1. SEC EDGAR (corporate filings)
2. GDELT / NewsAPI (news sentiment)
3. FRED API (macro indicators)
4. Sector ETF + macro price universe

---

## 1. SEC EDGAR

### Purpose
EDGAR filings serve two roles:
- **RAG context**: before generating a hypothesis about financials/energy/healthcare, the agent reads relevant 10-Q/10-K/8-K passages to understand what management said about macro conditions.
- **Quantitative signals**: sentiment scores from filing text (positive/negative tone shifts in MD&A sections) as lagged predictors.

### What to download

| Filing Type | Content | Update Frequency | Priority |
|-------------|---------|-----------------|----------|
| 10-K | Annual report — full business overview, risk factors, MD&A | Annually | HIGH |
| 10-Q | Quarterly report — MD&A, financial statements | Quarterly | HIGH |
| 8-K | Material events (earnings, M&A, guidance changes, executive departures) | Event-driven | HIGH |
| DEF 14A | Proxy — executive compensation, governance | Annually | LOW |

### Which companies

Focus on **sector representatives** that mirror the ETFs in your price universe:

```python
EDGAR_UNIVERSE = {
    "XLF": ["JPM", "BAC", "WFC", "GS", "MS", "BRK-B", "C", "AXP"],
    "XLE": ["XOM", "CVX", "COP", "EOG", "SLB", "PSX", "VLO"],
    "XLK": ["AAPL", "MSFT", "NVDA", "AVGO", "META", "GOOGL"],
    "XLV": ["UNH", "LLY", "JNJ", "ABBV", "MRK", "TMO", "ABT"],
    "XLI": ["GE", "RTX", "CAT", "HON", "UPS", "LMT", "DE"],
    "XLB": ["LIN", "APD", "SHW", "FCX", "NEM", "NUE"],
    "XLU": ["NEE", "SO", "DUK", "AEP", "EXC", "SRE"],
    "XLP": ["PG", "KO", "PEP", "COST", "WMT", "PM", "MO"],
    "XLRE": ["PLD", "AMT", "EQIX", "SPG", "O"],
    "XLY": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW"],
    "XLC": ["META", "GOOGL", "NFLX", "DIS", "CMCSA", "VZ"],
}
```

### How to download — `sec-edgar-downloader`

```python
pip install sec-edgar-downloader
```

```python
from sec_edgar_downloader import Downloader
import os

dl = Downloader(
    company_name="YourResearchProject",
    email_address="your@email.com",   # SEC requires identification
    save_path="./data/edgar_raw"
)

# Download 10-K filings since 2010
for ticker in TICKERS:
    dl.get("10-K", ticker, after="2010-01-01", before="2025-12-31")
    dl.get("10-Q", ticker, after="2010-01-01", before="2025-12-31")
    dl.get("8-K",  ticker, after="2010-01-01", before="2025-12-31")
```

**Rate limit:** SEC enforces max **10 requests/second**. Add `time.sleep(0.11)` between tickers.

### Alternative — SEC EDGAR Full-Text Search API (no library needed)

```python
import requests, time

BASE = "https://efts.sec.gov/LATEST/search-index"

def search_edgar(query: str, date_range: tuple, form_type: str = "10-K"):
    params = {
        "q": f'"{query}"',
        "dateRange": "custom",
        "startdt": date_range[0],   # "2010-01-01"
        "enddt":   date_range[1],   # "2025-12-31"
        "forms":   form_type,
    }
    r = requests.get("https://efts.sec.gov/LATEST/search-index", params=params)
    return r.json()
```

EDGAR XBRL API for structured financial data (no text parsing needed):
```python
# Get all 10-K facts for a CIK
CIK = "0000320193"  # Apple
url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{CIK}.json"
facts = requests.get(url).json()
# facts["facts"]["us-gaap"]["NetIncomeLoss"]["units"]["USD"]
```

### Text extraction & sentiment pipeline

```python
pip install beautifulsoup4 lxml finbert-embedding transformers torch
```

```python
from transformers import pipeline

finbert = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    tokenizer="ProsusAI/finbert",
    device=0  # GPU; use -1 for CPU
)

def score_filing_section(text: str, section: str = "MD&A") -> dict:
    """
    Extract MD&A section, chunk into 512-token windows,
    score each with FinBERT, return aggregate sentiment.
    """
    chunks = [text[i:i+400] for i in range(0, len(text), 400)]
    results = finbert(chunks[:20])  # limit to first 20 chunks
    scores = {"positive": 0, "negative": 0, "neutral": 0}
    for r in results:
        scores[r["label"].lower()] += r["score"]
    n = len(results)
    return {k: v/n for k, v in scores.items()}
```

### DuckDB schema

```sql
CREATE TABLE edgar_filings (
    ticker          VARCHAR,
    cik             VARCHAR,
    form_type       VARCHAR,         -- '10-K', '10-Q', '8-K'
    filed_date      DATE,
    period_of_report DATE,
    fiscal_quarter  INTEGER,         -- 1-4
    fiscal_year     INTEGER,
    mda_text        VARCHAR,         -- raw MD&A text (truncated to 50K chars)
    sentiment_pos   DOUBLE,          -- FinBERT aggregate
    sentiment_neg   DOUBLE,
    sentiment_neu   DOUBLE,
    sentiment_score DOUBLE,          -- pos - neg (composite signal)
    word_count      INTEGER,
    filing_url      VARCHAR
);

-- Quarterly sentiment changes (the signal, not the level)
CREATE VIEW v_edgar_sentiment_delta AS
SELECT
    ticker,
    period_of_report,
    sentiment_score,
    sentiment_score - LAG(sentiment_score) OVER (
        PARTITION BY ticker ORDER BY period_of_report
    ) AS sentiment_delta,
    form_type
FROM edgar_filings
WHERE form_type IN ('10-K', '10-Q');
```

### Agent usage pattern

Before generating a hypothesis about sector XLF or ticker JPM, the agent should:
1. Query `edgar_filings` for recent 10-Q text from XLF constituents
2. Use ChromaDB vector search to find relevant passages: *"interest rate margin"*, *"credit losses"*, *"macro outlook"*
3. Include top-3 passages as RAG context in the hypothesis-generation prompt

---

## 2. News Sentiment — GDELT & NewsAPI

### 2.1 GDELT Project (Historical — Free, Massive)

GDELT is the closest equivalent to the 2.5M-article corpus in the original paper.
It covers global news from 2013-present with daily updates, structured event codes, and tone scores.

**Two relevant datasets:**

| Dataset | What it contains | Update |
|---------|-----------------|--------|
| GDELT Events 2.0 | Structured events (actor, action, location, tone) | Every 15 min |
| GDELT GKG 2.0 | Knowledge graph — themes, persons, orgs, tone per article | Every 15 min |

**Use GKG (Global Knowledge Graph) — it has the themes you need.**

#### Option A — Google BigQuery (recommended for historical)

```sql
-- Free tier: 1TB/month queries
-- Historical data back to 2015 available

SELECT
  DATE(PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING))) AS date,
  AVG(CAST(TONE AS FLOAT64)) AS avg_tone,
  COUNT(*) AS article_count
FROM `gdelt-bq.gdeltv2.gkg`
WHERE
  DATE >= 20200101
  AND DATE <= 20251231
  AND THEMES LIKE '%ECON_INTEREST_RATE%'
GROUP BY date
ORDER BY date
```

Key GKG theme codes for US market research:
```
ECON_INTEREST_RATE        — Fed rate discussions
ECON_INFLATION            — inflation news
ECON_RECESSION            — recession fears
ECON_OILPRICE             — oil price news
ECON_STOCKMARKET          — stock market news
ECON_BANKRUPTCY           — bankruptcy events
FED_RESERVE               — Federal Reserve actions
BANKING                   — banking sector
ENERGY                    — energy sector
EMPLOYMENT                — jobs/unemployment
```

#### Option B — Direct HTTP download (no BigQuery account)

```python
import requests, io, zipfile, pandas as pd
from datetime import datetime, timedelta

GDELT_BASE = "http://data.gdeltproject.org/gdeltv2/"

def download_gdelt_gkg(date: datetime) -> pd.DataFrame:
    """Download one day of GKG data."""
    # Files are named YYYYMMDDHHMMSS.gkg.csv.zip, every 15 min
    # For daily aggregation, download all 96 files or use the master list
    date_str = date.strftime("%Y%m%d")
    # Use the daily summary endpoint
    url = f"{GDELT_BASE}{date_str}000000.gkg.csv.zip"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        return pd.DataFrame()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        fname = z.namelist()[0]
        return pd.read_csv(
            z.open(fname),
            sep="\t",
            header=None,
            names=GKG_COLS,      # see column list below
            on_bad_lines="skip",
            low_memory=False
        )

GKG_COLS = [
    "GKGRECORDID", "DATE", "SourceCollectionIdentifier",
    "SourceCommonName", "DocumentIdentifier", "Counts",
    "V2Counts", "Themes", "V2Themes", "Locations",
    "V2Locations", "Persons", "V2Persons", "Organizations",
    "V2Organizations", "V2Tone", "Dates", "GCAM",
    "SharingImage", "RelatedImages", "SocialImageEmbeds",
    "SocialVideoEmbeds", "Quotations", "AllNames",
    "Amounts", "TranslationInfo", "Extras"
]
```

**V2Tone field** is comma-separated: `Tone, Positive, Negative, Polarity, ActivityRefDensity, SelfGroupDensity, WordCount`

```python
def parse_tone(tone_str: str) -> dict:
    parts = tone_str.split(",")
    if len(parts) < 7:
        return {}
    return {
        "tone":       float(parts[0]),   # positive - negative
        "positive":   float(parts[1]),
        "negative":   float(parts[2]),
        "polarity":   float(parts[3]),
        "word_count": int(float(parts[6]))
    }
```

#### GDELT DuckDB schema

```sql
CREATE TABLE gdelt_daily (
    date         DATE,
    theme        VARCHAR,          -- e.g. 'ECON_INTEREST_RATE'
    avg_tone     DOUBLE,           -- mean tone across articles
    article_count INTEGER,
    positive_avg DOUBLE,
    negative_avg DOUBLE,
    polarity_avg DOUBLE
);

-- Pre-aggregate by week for faster hypothesis testing
CREATE VIEW v_gdelt_weekly AS
SELECT
    DATE_TRUNC('week', date) AS week,
    theme,
    AVG(avg_tone)       AS tone,
    SUM(article_count)  AS articles,
    AVG(negative_avg)   AS negativity
FROM gdelt_daily
GROUP BY 1, 2;
```

### 2.2 NewsAPI (recent + higher quality)

- **Free tier:** 100 requests/day, articles from past **30 days only**
- **Developer plan ($449/mo):** full historical archive
- **Recommendation:** use GDELT for history (2013–2023), NewsAPI for live pipeline

```python
pip install newsapi-python
```

```python
from newsapi import NewsApiClient

newsapi = NewsApiClient(api_key="YOUR_KEY")

# Fetch articles about Fed decisions
articles = newsapi.get_everything(
    q="Federal Reserve interest rate",
    from_param="2024-01-01",
    to="2024-12-31",
    language="en",
    sort_by="publishedAt",
    page_size=100
)
```

### 2.3 Alternative free sources for historical news sentiment

| Source | Coverage | Access |
|--------|----------|--------|
| **Benzinga** (via Alpaca) | 2010–present, financial only | Alpaca free tier |
| **Reddit PRAW** (r/wallstreetbets, r/investing) | 2008–present | Free API |
| **Federal Reserve press releases** | All FOMC statements | FRED / Fed website |
| **Refinitiv/LSEG (formerly Reuters)** | Professional, expensive | Paid |
| **Alpaca News API** | 2015–present, financial | Free with brokerage account |

```python
# Alpaca news — financial-specific, free
import alpaca_trade_api as tradeapi

api = tradeapi.REST(key_id="KEY", secret_key="SECRET", base_url="https://paper-api.alpaca.markets")
news = api.get_news(symbol="XLF", start="2020-01-01", end="2020-12-31", limit=50)
```

---

## 3. FRED API — Macro Indicators

### Setup

```python
pip install fredapi pandas
```

```python
from fredapi import Fred
fred = Fred(api_key="YOUR_FRED_API_KEY")  # free at fred.stlouisfed.org/docs/api/api_key.html
```

No rate limit for reasonable usage (<120 requests/minute).

### Full download specification

#### 3.1 Monetary Policy

| Series ID | Name | Frequency | Start |
|-----------|------|-----------|-------|
| `FEDFUNDS` | Effective Federal Funds Rate | Daily | 1954 |
| `DFF` | Fed Funds Rate (daily) | Daily | 1954 |
| `DFEDTARL` | Fed Funds Target Rate Lower Bound | Daily | 2008 |
| `DFEDTARU` | Fed Funds Target Rate Upper Bound | Daily | 2008 |
| `WALCL` | Fed Balance Sheet (Total Assets) | Weekly | 2002 |
| `WRESBAL` | Reserve Balances at Fed | Weekly | 1959 |

#### 3.2 Yield Curve (most important for US bank sector signals)

| Series ID | Name | Frequency |
|-----------|------|-----------|
| `DGS1MO` | 1-Month Treasury | Daily |
| `DGS3MO` | 3-Month Treasury | Daily |
| `DGS6MO` | 6-Month Treasury | Daily |
| `DGS1` | 1-Year Treasury | Daily |
| `DGS2` | 2-Year Treasury | Daily |
| `DGS5` | 5-Year Treasury | Daily |
| `DGS10` | 10-Year Treasury | Daily |
| `DGS30` | 30-Year Treasury | Daily |
| `T10Y2Y` | 10Y-2Y Spread (inverted yield curve) | Daily |
| `T10Y3M` | 10Y-3M Spread | Daily |

> ⚠️ **T10Y2Y is arguably the single most important US macro signal.**
> Historically inverts 12–18 months before recessions. Test lags of 180–540 days against XLF, XLI, XLY.

#### 3.3 Inflation

| Series ID | Name | Frequency |
|-----------|------|-----------|
| `CPIAUCSL` | CPI All Urban Consumers | Monthly |
| `CPILFESL` | Core CPI (ex food & energy) | Monthly |
| `PPIACO` | PPI All Commodities | Monthly |
| `PCEPILFE` | Core PCE (Fed's preferred measure) | Monthly |
| `T5YIE` | 5-Year Breakeven Inflation | Daily |
| `T10YIE` | 10-Year Breakeven Inflation | Daily |
| `MICH` | U Michigan Inflation Expectations | Monthly |

#### 3.4 Labor Market

| Series ID | Name | Frequency |
|-----------|------|-----------|
| `UNRATE` | Unemployment Rate | Monthly |
| `PAYEMS` | Nonfarm Payrolls | Monthly |
| `ICSA` | Initial Jobless Claims | Weekly |
| `CCSA` | Continued Claims | Weekly |
| `JTSJOL` | JOLTS Job Openings | Monthly |
| `LNS12300060` | Prime-Age Employment Rate | Monthly |

#### 3.5 Credit & Financial Conditions

| Series ID | Name | Why it matters |
|-----------|------|----------------|
| `BAMLH0A0HYM2` | HY Credit Spread (ICE BofA) | Risk-off signal, leads equities |
| `BAMLC0A0CM` | IG Credit Spread | Investment grade stress |
| `DRTSCILM` | C&I Loan Tightening Standards | Bank lending → economic activity |
| `DRTSCLCC` | Credit Card Lending Standards | Consumer stress |
| `TOTCI` | Commercial & Industrial Loans | Credit expansion/contraction |
| `CONSUMER` | Consumer Credit | Household debt |

#### 3.6 Housing

| Series ID | Name | Frequency |
|-----------|------|-----------|
| `MORTGAGE30US` | 30-Year Fixed Mortgage Rate | Weekly |
| `HOUST` | Housing Starts | Monthly |
| `EXHOSLUSM495S` | Existing Home Sales | Monthly |
| `CSUSHPISA` | Case-Shiller Home Price Index | Monthly |
| `RHORUSQ156N` | Homeownership Rate | Quarterly |

#### 3.7 Activity & Sentiment

| Series ID | Name | Frequency |
|-----------|------|-----------|
| `INDPRO` | Industrial Production Index | Monthly |
| `TCU` | Capacity Utilization | Monthly |
| `UMCSENT` | U Michigan Consumer Sentiment | Monthly |
| `ISRATIO` | Inventory/Sales Ratio | Monthly |
| `RETAILSMSA` | Retail Sales | Monthly |
| `DSPIC96` | Real Disposable Personal Income | Monthly |

#### 3.8 Money Supply

| Series ID | Name |
|-----------|------|
| `M2SL` | M2 Money Supply |
| `M2V` | Velocity of M2 |
| `BOGMBASE` | Monetary Base |

### Download script

```python
import pandas as pd
from fredapi import Fred
from datetime import date

fred = Fred(api_key="YOUR_KEY")

FRED_SERIES = {
    # Monetary policy
    "fed_funds_rate":     "FEDFUNDS",
    "fed_balance_sheet":  "WALCL",
    # Yield curve
    "yield_2y":           "DGS2",
    "yield_10y":          "DGS10",
    "yield_spread_10_2":  "T10Y2Y",
    "yield_spread_10_3m": "T10Y3M",
    # Inflation
    "cpi":                "CPIAUCSL",
    "core_cpi":           "CPILFESL",
    "core_pce":           "PCEPILFE",
    "breakeven_5y":       "T5YIE",
    "breakeven_10y":      "T10YIE",
    # Labor
    "unemployment":       "UNRATE",
    "nonfarm_payrolls":   "PAYEMS",
    "jobless_claims":     "ICSA",
    # Credit
    "hy_spread":          "BAMLH0A0HYM2",
    "ig_spread":          "BAMLC0A0CM",
    "loan_standards_ci":  "DRTSCILM",
    # Housing
    "mortgage_rate_30y":  "MORTGAGE30US",
    "housing_starts":     "HOUST",
    # Activity
    "industrial_prod":    "INDPRO",
    "consumer_sentiment": "UMCSENT",
    "retail_sales":       "RETAILSMSA",
    # Money
    "m2":                 "M2SL",
}

def download_all_fred(start: str = "2010-01-01") -> pd.DataFrame:
    frames = {}
    for name, series_id in FRED_SERIES.items():
        try:
            s = fred.get_series(series_id, observation_start=start)
            s.name = name
            frames[name] = s
            print(f"✓ {name} ({series_id}): {len(s)} observations")
        except Exception as e:
            print(f"✗ {name}: {e}")

    df = pd.DataFrame(frames)
    df.index.name = "date"
    df.index = pd.to_datetime(df.index)
    return df.sort_index()
```

### DuckDB schema

```sql
CREATE TABLE fred_macro (
    date                DATE PRIMARY KEY,
    -- Monetary policy
    fed_funds_rate      DOUBLE,
    fed_balance_sheet   DOUBLE,
    -- Yield curve
    yield_2y            DOUBLE,
    yield_10y           DOUBLE,
    yield_spread_10_2   DOUBLE,    -- KEY SIGNAL: negative = inverted
    yield_spread_10_3m  DOUBLE,
    -- Inflation
    cpi                 DOUBLE,
    core_cpi            DOUBLE,
    core_pce            DOUBLE,
    breakeven_5y        DOUBLE,
    breakeven_10y       DOUBLE,
    -- Labor
    unemployment        DOUBLE,
    nonfarm_payrolls    DOUBLE,
    jobless_claims      DOUBLE,
    -- Credit
    hy_spread           DOUBLE,    -- KEY SIGNAL for risk appetite
    ig_spread           DOUBLE,
    loan_standards_ci   DOUBLE,
    -- Housing
    mortgage_rate_30y   DOUBLE,
    housing_starts      DOUBLE,
    -- Activity
    industrial_prod     DOUBLE,
    consumer_sentiment  DOUBLE,
    retail_sales        DOUBLE,
    -- Money
    m2                  DOUBLE
);

-- Forward-fill monthly/weekly series to daily for join convenience
CREATE VIEW v_fred_daily AS
SELECT
    d.date,
    LAST_VALUE(f.fed_funds_rate IGNORE NULLS) OVER w AS fed_funds_rate,
    LAST_VALUE(f.yield_spread_10_2 IGNORE NULLS) OVER w AS yield_spread_10_2,
    LAST_VALUE(f.hy_spread IGNORE NULLS) OVER w AS hy_spread,
    LAST_VALUE(f.unemployment IGNORE NULLS) OVER w AS unemployment,
    LAST_VALUE(f.cpi IGNORE NULLS) OVER w AS cpi,
    LAST_VALUE(f.core_pce IGNORE NULLS) OVER w AS core_pce,
    LAST_VALUE(f.jobless_claims IGNORE NULLS) OVER w AS jobless_claims
FROM (SELECT UNNEST(generate_series(DATE '2010-01-01', DATE '2025-12-31', INTERVAL '1 day'))::DATE AS date) d
LEFT JOIN fred_macro f ON f.date = d.date
WINDOW w AS (ORDER BY d.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW);
```

---

## 4. Price Universe — Sector ETFs + Macro Assets

### 4.1 Sector ETFs (SPDR Select Sectors)

| Ticker | Sector | Key constituents |
|--------|--------|-----------------|
| `XLF` | Financials | JPM, BAC, WFC, GS, MS, BRK-B |
| `XLE` | Energy | XOM, CVX, COP, EOG, SLB |
| `XLK` | Technology | AAPL, MSFT, NVDA, AVGO, META |
| `XLV` | Health Care | UNH, LLY, JNJ, ABBV, MRK |
| `XLI` | Industrials | GE, RTX, CAT, HON, UPS, LMT |
| `XLB` | Materials | LIN, APD, SHW, FCX, NEM |
| `XLU` | Utilities | NEE, SO, DUK, AEP, EXC |
| `XLP` | Consumer Staples | PG, KO, PEP, COST, WMT |
| `XLRE` | Real Estate | PLD, AMT, EQIX, SPG, O |
| `XLY` | Consumer Discretionary | AMZN, TSLA, HD, MCD, NKE |
| `XLC` | Communication Services | META, GOOGL, NFLX, DIS, VZ |

### 4.2 Sub-sector ETFs (for finer signals)

| Ticker | Sub-sector | Why useful |
|--------|-----------|-----------|
| `KRE` | Regional Banks | More rate-sensitive than XLF |
| `KBE` | Bank Index | Broader banking |
| `IAT` | Regional Banks (iShares) | Alternative to KRE |
| `XBI` | Biotech | High-beta health care |
| `IBB` | Biotech (iShares) | Biotech alternative |
| `ITB` | Home Builders | Housing cycle signal |
| `XHB` | Home Builders (SPDR) | Housing alternative |
| `JETS` | Airlines | Oil price sensitivity |
| `XOP` | Oil & Gas E&P | More leveraged to oil than XLE |
| `OIH` | Oil Services | Capex cycle |
| `SOXX` | Semiconductors | Tech cycle, China risk |
| `SMH` | Semiconductors (VanEck) | Alternative semis |
| `IYR` | Real Estate (iShares) | Mortgage rate sensitivity |

### 4.3 Broad Market Indices

| Ticker | Index | Notes |
|--------|-------|-------|
| `SPY` | S&P 500 | Main benchmark |
| `QQQ` | NASDAQ 100 | Tech-heavy |
| `IWM` | Russell 2000 | Small caps — more rate sensitive |
| `DIA` | Dow Jones | Large cap industrials |
| `MDY` | S&P 400 Mid Cap | Mid tier |

### 4.4 Commodities

| Ticker | Asset | Notes |
|--------|-------|-------|
| `USO` | Oil (WTI) ETF | Or use `CL=F` futures |
| `BNO` | Brent Oil ETF | Or use `BZ=F` futures |
| `GLD` | Gold ETF | Or use `GC=F` futures |
| `SLV` | Silver | Industrial/safe haven hybrid |
| `PDBC` | Diversified Commodities | Broad commodity signal |
| `DBA` | Agricultural Commodities | Food inflation |
| `CPER` | Copper ETF | Economic activity proxy |
| `UNG` | Natural Gas | Or use `NG=F` futures |
| `WEAT` | Wheat | Geopolitical/inflation signal |

**Note:** For commodities, use continuous futures (`CL=F`, `BZ=F`, `GC=F`, `NG=F`) via yfinance when possible — they have longer history than ETFs.

### 4.5 Rates & Bonds

| Ticker | Asset | Duration |
|--------|-------|---------|
| `TLT` | 20+ Year Treasury | Long |
| `IEF` | 7-10 Year Treasury | Medium |
| `SHY` | 1-3 Year Treasury | Short |
| `TIP` | TIPS (Inflation-Protected) | Real rates |
| `HYG` | High Yield Corporate | Credit risk |
| `LQD` | Investment Grade Corporate | IG credit |
| `EMB` | EM Bonds (USD) | EM risk appetite |
| `MBB` | Mortgage-Backed Securities | Housing credit |

### 4.6 Currencies & International

| Ticker | Asset | Signal for |
|--------|-------|-----------|
| `UUP` | US Dollar Index ETF | DXY proxy |
| `FXE` | Euro | EUR/USD |
| `FXY` | Yen | Risk-off/carry trade |
| `EEM` | Emerging Markets Equities | Global risk appetite |
| `EFA` | Developed Markets ex-US | Global growth |
| `FXI` | China Large Cap | China risk |
| `EWJ` | Japan | Yen carry trade signal |
| `EWG` | Germany | European industrial cycle |

### 4.7 Volatility & Sentiment

| Ticker | Asset | Notes |
|--------|-------|-------|
| `^VIX` | CBOE VIX | Already have |
| `VIXY` | VIX Futures Short-Term | Tradeable VIX proxy |
| `^VXN` | NASDAQ Volatility | Tech-specific vol |
| `^MOVE` | Bond Market Volatility | Rate uncertainty |
| `^SKEW` | CBOE SKEW Index | Tail risk |
| `^PCR` | Put/Call Ratio | Sentiment |

### 4.8 yfinance download script

```python
import yfinance as yf
import pandas as pd
import duckdb

TICKERS = {
    "sector_etf": [
        "XLF", "XLE", "XLK", "XLV", "XLI", "XLB",
        "XLU", "XLP", "XLRE", "XLY", "XLC"
    ],
    "subsector_etf": [
        "KRE", "KBE", "XBI", "IBB", "ITB", "JETS",
        "XOP", "OIH", "SOXX", "SMH", "IYR", "XHB"
    ],
    "broad_market": ["SPY", "QQQ", "IWM", "DIA", "MDY"],
    "commodities": [
        "GLD", "SLV", "USO", "BNO", "PDBC", "DBA",
        "CPER", "UNG", "WEAT",
        "CL=F", "BZ=F", "GC=F", "NG=F"
    ],
    "bonds": ["TLT", "IEF", "SHY", "TIP", "HYG", "LQD", "EMB", "MBB"],
    "currencies_intl": [
        "UUP", "FXE", "FXY",
        "EEM", "EFA", "FXI", "EWJ", "EWG",
        "EURUSD=X", "USDJPY=X", "GBPUSD=X"
    ],
    "volatility": ["^VIX", "^VXN", "^SKEW", "VIXY"],
}

def download_all_prices(
    start: str = "2010-01-01",
    end: str = "2025-12-31"
) -> pd.DataFrame:
    all_tickers = [t for group in TICKERS.values() for t in group]
    df = yf.download(
        tickers=all_tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=True,
    )
    # Flatten multi-level columns
    close = df["Close"].copy()
    close.index.name = "date"
    return close

def save_to_duckdb(df: pd.DataFrame, db_path: str = "market.duckdb"):
    con = duckdb.connect(db_path)
    df_reset = df.reset_index()
    df_reset["date"] = pd.to_datetime(df_reset["date"]).dt.date
    con.execute("DROP TABLE IF EXISTS prices")
    con.execute("CREATE TABLE prices AS SELECT * FROM df_reset")
    con.close()
    print(f"Saved {len(df_reset)} rows, {len(df.columns)} tickers to {db_path}")
```

### DuckDB schema for prices

```sql
-- Wide format (one column per ticker) — good for cross-asset queries
CREATE TABLE prices (
    date   DATE PRIMARY KEY,
    -- Sector ETFs
    XLF DOUBLE, XLE DOUBLE, XLK DOUBLE, XLV DOUBLE, XLI DOUBLE,
    XLB DOUBLE, XLU DOUBLE, XLP DOUBLE, XLRE DOUBLE, XLY DOUBLE, XLC DOUBLE,
    -- Sub-sector
    KRE DOUBLE, KBE DOUBLE, XBI DOUBLE, SOXX DOUBLE, ITB DOUBLE,
    -- Broad market
    SPY DOUBLE, QQQ DOUBLE, IWM DOUBLE,
    -- Commodities
    GLD DOUBLE, USO DOUBLE, "CL=F" DOUBLE, "BZ=F" DOUBLE, "GC=F" DOUBLE,
    -- Bonds
    TLT DOUBLE, HYG DOUBLE, LQD DOUBLE, TIP DOUBLE,
    -- Currencies/Intl
    EEM DOUBLE, "EURUSD=X" DOUBLE, "USDJPY=X" DOUBLE, UUP DOUBLE,
    -- Volatility
    "^VIX" DOUBLE
);

-- Returns table (log returns, forward-filled NULLs)
CREATE VIEW v_returns AS
SELECT
    date,
    LN(XLF / LAG(XLF) OVER (ORDER BY date)) AS r_xlf,
    LN(XLE / LAG(XLE) OVER (ORDER BY date)) AS r_xle,
    LN(XLK / LAG(XLK) OVER (ORDER BY date)) AS r_xlk,
    LN(XLF / LAG(XLF,5)  OVER (ORDER BY date)) AS r_xlf_1w,
    LN(XLF / LAG(XLF,21) OVER (ORDER BY date)) AS r_xlf_1m,
    LN(XLF / LAG(XLF,63) OVER (ORDER BY date)) AS r_xlf_3m,
    -- ... repeat for all sector ETFs
    LN("CL=F" / LAG("CL=F") OVER (ORDER BY date)) AS r_oil_1d,
    LN("CL=F" / LAG("CL=F",21) OVER (ORDER BY date)) AS r_oil_1m
FROM prices;
```

---

## 5. Unified Context View for Agent

The agent should use a single denormalized view combining all sources for hypothesis testing:

```sql
CREATE VIEW v_market_context AS
SELECT
    p.date,

    -- Prices (levels)
    p.XLF, p.XLE, p.XLK, p.XLV, p.XLI, p.XLB, p.XLU, p.XLP,
    p."CL=F" AS oil_wti, p."BZ=F" AS oil_brent,
    p.GLD AS gold, p.TLT AS bonds_long, p.HYG AS hy_bonds,
    p.EEM AS em_equities, p.UUP AS usd_index,
    p."^VIX" AS vix,

    -- FRED macro (forward-filled)
    m.fed_funds_rate,
    m.yield_spread_10_2,       -- CRITICAL: inverted yield curve
    m.hy_spread,               -- CRITICAL: risk appetite
    m.unemployment,
    m.cpi,
    m.core_pce,
    m.jobless_claims,

    -- Regime classification (pre-computed)
    CASE
        WHEN m.fed_funds_rate < 1.0 THEN 'ZIRP'
        WHEN m.fed_funds_rate < 3.0 THEN 'LOW_RATE'
        WHEN m.fed_funds_rate < 5.0 THEN 'NORMAL_RATE'
        ELSE 'HIGH_RATE'
    END AS rate_regime,

    CASE
        WHEN m.yield_spread_10_2 < 0 THEN 'INVERTED'
        WHEN m.yield_spread_10_2 < 0.5 THEN 'FLAT'
        ELSE 'NORMAL'
    END AS yield_curve_regime,

    CASE
        WHEN p."^VIX" > 30 THEN 'HIGH_STRESS'
        WHEN p."^VIX" > 20 THEN 'ELEVATED'
        ELSE 'CALM'
    END AS vol_regime,

    -- News sentiment (if available for date)
    n_oil.avg_tone    AS oil_news_tone,
    n_fed.avg_tone    AS fed_news_tone,
    n_bank.avg_tone   AS bank_news_tone,
    n_econ.avg_tone   AS econ_news_tone

FROM prices p
LEFT JOIN v_fred_daily m ON m.date = p.date
LEFT JOIN gdelt_daily n_oil  ON n_oil.date  = p.date AND n_oil.theme  = 'ECON_OILPRICE'
LEFT JOIN gdelt_daily n_fed  ON n_fed.date  = p.date AND n_fed.theme  = 'FED_RESERVE'
LEFT JOIN gdelt_daily n_bank ON n_bank.date = p.date AND n_bank.theme = 'BANKING'
LEFT JOIN gdelt_daily n_econ ON n_econ.date = p.date AND n_econ.theme = 'ECON_RECESSION'

WHERE p.date BETWEEN '2010-01-01' AND '2025-12-31';
```

---

## 6. Hypothesis Templates for the Agent

These are example hypotheses the agent should generate and test, organized by chain type:

### 6.1 Oil price chains (analogous to the Brent→MOEXFN signal from the original paper)

```
oil_wti ↑ (+20% over 3 months)
  → inflation expectations rise (T10YIE)
  → Fed signals hawkishness
  → yield_spread_10_2 compresses
  → KRE (regional banks) falls [lag: 60-120 days]

oil_wti ↑
  → JETS (airlines cost structure deteriorates) [lag: 0-30 days]
  → XLY (consumer discretionary hurt by energy costs) [lag: 30-90 days]
```

### 6.2 Yield curve chains (US-specific, no analog in Russian market)

```
yield_spread_10_2 < 0 (inversion)
  → KRE (regional banks: NIM compression) [lag: 0-90 days]
  → XLI (recession signal, capex cuts) [lag: 180-540 days]
  → HOUST (housing starts drop) [lag: 60-180 days]

yield_spread_10_2 normalizes after inversion
  → XLF outperforms (re-rating) [lag: 0-60 days]
```

### 6.3 Credit spread chains

```
hy_spread spikes (>500bp)
  → XLB (materials, commodity producers: financing stress) [lag: 0-30 days]
  → XLI (capex freeze) [lag: 30-90 days]
  → unemployment lags equities [lag: 60-180 days]
```

### 6.4 Dollar strength chains

```
UUP / DXY ↑ (strong dollar)
  → EEM falls (EM debt burden) [lag: 0-30 days]
  → XLB falls (commodities priced in USD) [lag: 0-60 days]
  → XLE mixed (oil down but US producers partially hedged) [lag: 14-60 days]
  → QQQ falls (multinational revenue headwind) [lag: 30-90 days]
```

### 6.5 VIX regime chains

```
VIX spikes >30
  → XLP outperforms XLK [lag: 0-30 days]  (defensive rotation)
  → GLD outperforms EEM [lag: 0-14 days]
  → 3 months after spike: mean-reversion trade (XLK recovery)
```

---

## 7. Regime Analysis Framework

**Critical lesson from the original paper:** a signal that shows r=0.70 in aggregate may show r=0.10 in two of the four sub-regimes. Always segment:

```sql
-- Template: test signal X → target Y in each regime
WITH lagged AS (
  SELECT
    date,
    oil_wti,
    LEAD(XLF, 90) OVER (ORDER BY date) AS xlf_90d_forward,
    rate_regime,
    yield_curve_regime,
    vol_regime
  FROM v_market_context
  WHERE oil_wti IS NOT NULL AND XLF IS NOT NULL
)
SELECT
    rate_regime,
    yield_curve_regime,
    CORR(oil_wti, xlf_90d_forward) AS corr_oil_xlf_90d,
    COUNT(*) AS n
FROM lagged
GROUP BY rate_regime, yield_curve_regime
HAVING n > 100
ORDER BY corr_oil_xlf_90d;
```

Regime dimensions to always segment by:
1. **Rate regime:** ZIRP / LOW / NORMAL / HIGH
2. **Yield curve:** INVERTED / FLAT / NORMAL
3. **Vol regime:** HIGH_STRESS / ELEVATED / CALM
4. **Economic cycle:** EXPANSION / CONTRACTION (use NBER dates from FRED series `USREC`)

---

## 8. Build Order (Recommended)

```
Week 1  — Prices + FRED
          yfinance download → DuckDB prices table
          FRED download → fred_macro table
          Build v_market_context view
          Run first correlation sweep manually (no LLM)

Week 2  — Validate & first signals
          Implement lag correlation sweep over all (factor × sector × lag) combinations
          Find 5-10 candidate signals
          Add regime segmentation — eliminate false positives

Week 3  — News sentiment
          GDELT via BigQuery (fastest path) or direct HTTP download
          Populate gdelt_daily table
          Add news tone columns to v_market_context

Week 4  — EDGAR
          Download 10-Q for XLF/XLE/XLK constituents
          Run FinBERT → edgar_filings table
          Add as RAG context source for agent

Week 5+ — Agent loop
          Implement hypothesis generator (LLM prompt)
          SQL executor with error recovery (Level 1 from paper)
          Result evaluator
          Knowledge accumulator (forbidden_patterns.md equivalent)
```

---

## 9. Known Pitfalls (Lessons from the Original Paper)

| Pitfall | Description | Prevention |
|---------|-------------|-----------|
| **Ticker substitution** | Agent uses SPY data but labels it as "XLK" | Enforce `WHERE ticker = 'XLK'` pattern; no column renaming |
| **Survivorship bias** | ETF composition changes over time | Use point-in-time constituent data or accept limitation |
| **Look-ahead bias** | FRED data is revised; published value differs from real-time | Use vintage data from Alfred (FRED's archival database) for backtesting |
| **Spurious correlation** | 2+ time series with trends will correlate | Always use returns, not levels, for correlation testing |
| **Regime aggregation** | Strong signal in one year drives aggregate r | Always report per-regime breakdown alongside aggregate |
| **Multiple testing** | 50+ factors × 11 sectors × 10 lags = 5500 tests | Apply Bonferroni or FDR correction; require r > 0.4 as minimum threshold |

---

*Sources: SEC EDGAR (edgar.sec.gov), FRED (fred.stlouisfed.org), GDELT Project (gdeltproject.org), SPDR ETFs (ssga.com), yfinance documentation.*
