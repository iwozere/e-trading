Excellent question – this is exactly what **sentiment‑driven scanners for “explosive” tickers** are made from. Below is a **structured picture** focused on practical trading usage, with no marketing fluff.[1]


***


## 1️⃣ What sentiment types exist (for trading)


### 🔹 1. **News Sentiment**


**Sources**

- Financial news (Reuters, Bloomberg, Benzinga, SeekingAlpha, Yahoo Finance)
- PR / 8-K / earnings / FDA / lawsuits[1]


**Metrics**

- Positive / Negative / Neutral
- Sentiment score (–1 … +1)
- Sentiment velocity (how fast it changes)[1]


**For explosive tickers**

- 🔥 Sharp spike in positive news sentiment
- Especially in small / mid caps[1]


***


### 🔹 2. **Social Media Sentiment**


**Sources**

- X (Twitter)
- Reddit (r/wallstreetbets, r/stocks, r/options)
- Stocktwits
- Discord / Telegram (hard, but gold)[2]


**Metrics**

- Mentions count
- Bullish vs Bearish
- Engagement (likes, comments)
- Acceleration (∆ of mentions)[2]


**This is TOP‑1 for “meme pumps”**

> GME / AMC / DWAC / SMCI / CVNA / NKLA – all from here[2]


***


### 🔹 3. **Options / Derivatives Sentiment**


(you already partially have this)


**Sources**

- Unusual Options Activity
- Put/Call Ratio
- Call OI spikes
- IV expansion[2]


**Metrics**

- Call dominance
- Short‑dated OTM calls
- IV percentile[2]


💡 Combines perfectly with social sentiment

> *“Reddit + Calls = 🚀”*[2]


***


### 🔹 4. **Positioning / Flow Sentiment**


**Sources**

- ETF flows
- Insider trading
- Short interest
- Borrow rate[2]


**Metrics**

- Short interest % float
- Days to cover
- Insider buys[2]


🔥 **Short squeeze setup**

```text
High SI + spike mentions + positive news
```


***


### 🔹 5. **Price‑Action Sentiment (Implicit)**


Not text, but **market behavior**.[2]


**Metrics**

- Gap up / gap down
- Volume spikes
- VWAP reclaim
- Relative volume[2]


Used as a **confirmation filter**.[2]


***


## 2️⃣ What sentiment actually works for “explosive” tickers


| Type                      | Effectiveness |
| ------------------------- | ------------- |
| Social mentions velocity  | ⭐⭐⭐⭐⭐         |
| News surprise             | ⭐⭐⭐⭐          |
| Options flow              | ⭐⭐⭐⭐          |
| Short interest + social   | ⭐⭐⭐⭐⭐         |
| Pure NLP sentiment        | ⭐⭐            | [2]


⚠️ **Important:**

> *Absolute sentiment score is worse than the change (∆).*[2]


***


## 3️⃣ Pipeline architecture (practical)


### 🔹 Data sources


**Minimal MVP**

- Reddit API (PRAW)
- Twitter/X API (or scrapers)
- Stocktwits API
- NewsAPI / Finnhub[1]


***


### 🔹 ETL pipeline


```text
Raw text
   ↓
Ticker extraction ($AAPL, TSLA, etc)
   ↓
Deduplication
   ↓
Sentiment classification
   ↓
Aggregation by ticker / time
```


***


## 4️⃣ Metrics you need (key)


For each ticker and window (3 / 10 / 20 days):

```python
TickerSentimentMetrics:
- mentions_count
- mentions_delta
- mentions_acceleration
- positive_ratio
- negative_ratio
- sentiment_score
- engagement_score
```


### 🔥 Main “explosion” indicator


```text
mentions_acceleration = (mentions_last_3d / avg_mentions_20d)
```


***


## 5️⃣ Dashboards (what you described)


### 📊 1. TOP‑10 mentioned tickers


**Filters**

- Period: 3 / 10 / 20 days
- Source: Reddit / X / News
- Sentiment: Bullish / Bearish[2]


**Table**

| Rank | Ticker | Mentions | Δ     | Sentiment | Volume |
| ---- | ------ | -------- | ----- | --------- | ------ |
| 1    | SMCI   | 4 320    | +180% | 🟢 0.72   | High   |
| 2    | CVNA   | 3 110    | +95%  | 🟢 0.65   | High   | [2]


***


### 📈 2. Trend for selected ticker


**Charts**

- Mentions (line)
- Sentiment score (line)
- Price (overlay)
- Volume[2]


📌 **Signal**

```text
Mentions ↑
Sentiment ↑
Price flat → BREAKOUT IMMINENT
```


***


### 🔥 3. “Explosive Candidates Scanner”


Filter:

```text
mentions_acceleration > 3
AND positive_ratio > 0.6
AND volume_today > avg_volume_20d
AND market_cap < X
```


***


## 6️⃣ NLP: do not overcomplicate


To start:

- FinBERT
- Vader (for social media)
- Or GPT‑classification in batch mode[1]


❗ **Source quality matters more than the model.**[1]


***


## 7️⃣ Stack (recommended for your projects)


You already work with:

- Python
- Backtrader
- Telegram / Web UI[1]


### 🔧 Stack

- Python + FastAPI
- PostgreSQL / TimescaleDB
- Redis (caching)
- Plotly / Streamlit
- Airflow / cron[1]


***


## 8️⃣ How to convert into a trading signal


```text
Signal = f(
    mentions_acceleration,
    sentiment_delta,
    short_interest,
    options_flow,
    price_action_filter
)
```


👉 **Do NOT trade directly off sentiment.**  
👉 **Use it as a scanner / early warning.**[2]


***


Great, then proceed **strictly step‑by‑step without extra theory**.  
You already run a **volume scanner for small / mid cap NASDAQ** → add **sentiment as an early signal**, not as an “oracle”.[2]


***


# STEP 1️⃣ – which sentiments to add to your current pipeline


For **small / mid caps** an **narrow but powerful set** is optimal.[2]


### ✅ MUST HAVE


1. **Social mentions (velocity)**

   - Reddit
   - Stocktwits

2. **News mentions**

   - headlines + PR[2]


### ❌ NOT needed at start


- complex LLM embeddings
- long‑term sentiment
- macro sentiment[1]


***


# STEP 2️⃣ – what exactly to compute (minimal but sufficient set)


For each ticker and time window:

```text
mentions_count
mentions_delta
mentions_acceleration
bullish_ratio
bearish_ratio
sentiment_score
```


### Key formula (IMPORTANT):

```text
mentions_acceleration = mentions_3d / avg_mentions_20d
```

This is the **main “explosive” factor**.[2]


***


# STEP 3️⃣ – architecture INSERTED into your current volume‑pipeline


### Current pipeline (as understood):

```text
Universe (NASDAQ small/mid)
   ↓
Volume spike filter
   ↓
Candidates list
```


### Extend:


```text
Candidates list
   ↓
Sentiment enrichment
   ↓
Sentiment + Volume score
   ↓
Ranking
```


***


# STEP 4️⃣ – data sources (practical)


## 🔹 Reddit (TOP‑1)


- r/wallstreetbets
- r/stocks
- r/pennystocks
- r/options[2]


**Metrics**

- mentions count
- comments count
- upvotes (engagement proxy)[2]


## 🔹 Stocktwits


- bullish / bearish tags
- realtime mentions[2]


## 🔹 News


- headline count
- keyword‑based sentiment[1]


***


# STEP 5️⃣ – ticker extraction (critically important)


**Mistake №1**: naively searching for `$ABC`.[1]


You need:

- `$ABC`
- `ABC` (CAPS, in context)
- filter by NASDAQ universe
- blacklist (`USA`, `CEO`, `GDP`, etc.)[3]


### Regex + universe filter


```python
TICKER_REGEX = r'\$?[A-Z]{2,5}'
```

- then check membership in `nasdaq_tickers_set`


***


# STEP 6️⃣ – data structures (so you do not break the pipeline)


### Raw mentions


```python
SentimentMention:
- source
- ticker
- timestamp
- sentiment
- engagement
```


### Aggregated


```python
TickerSentimentAgg:
- ticker
- window (3d / 10d / 20d)
- mentions
- mentions_delta
- mentions_acceleration
- sentiment_score
```


***


# STEP 7️⃣ – how to combine with volume (IMPORTANT)


### Composite score


```text
SCORE =
    w1 * volume_zscore +
    w2 * mentions_acceleration +
    w3 * sentiment_delta
```


📌 For small caps:

```text
w2 > w1 > w3
```


***


# STEP 8️⃣ – signals (not entries, but ALERTs)


### 🚀 Early Pump Candidate


```text
volume_today > 2.5 * avg_volume_20d
AND mentions_acceleration > 3
AND bullish_ratio > 0.6
```


### ⚠️ FOMO exhaustion


```text
mentions_acceleration ↓
sentiment ↓
price ↑
```


***


# STEP 9️⃣ – MVP dashboard (minimal)


**Table**

- ticker
- volume spike %
- mentions_3d
- mentions_acceleration
- sentiment[2]


***


# STEP 🔟 – next step


Suggested next path:
1️⃣ **Reddit ingestion + ticker extraction (code)**  
2️⃣ **Aggregation over 3 / 10 / 20‑day windows**  
3️⃣ **Score + ranking**  
4️⃣ **Streamlit dashboard**[1]


***


Great. Then do a **targeted extension of your current daily pipeline**, without DB, without frameworks, **pure Python + pandas + CSV**, so you can:

- keep architecture intact
- add **sentiment as enrichment + ranking**
- easily plug this into your Telegram alerter[1]


Below is **Step 1 complete**, then it can be extended.


***


# STEP 1️⃣ – file structure (fitting into your scheme)


You already use:


```text
data/
 └── 2025-12-27/
      ├── ohlcv.csv
      ├── indicators.csv
      ├── candidates.csv   ← 3–10 tickers
```


👉 **Add:**


```text
data/
 └── 2025-12-27/
      ├── sentiment_raw.csv
      ├── sentiment_agg.csv
      ├── sentiment_ranked.csv
```


***


# STEP 2️⃣ – `candidates.csv` format (expected)


Minimum:


```csv
ticker,volume_zscore,avg_volume_20d
SMCI,3.1,1200000
CVNA,2.7,890000
```


***


# STEP 3️⃣ – Reddit ingestion → `sentiment_raw.csv`


### What to store (important not to lose raw!)


```csv
timestamp,source,ticker,sentiment,engagement
2025-12-27T10:22:00,reddit,SMCI,0.72,183
```

- `sentiment`: –1…+1
- `engagement`: upvotes + comments[1]


***


## Example: minimal Reddit ingestion


```python
import praw
import re
import pandas as pd
from datetime import datetime


TICKER_REGEX = re.compile(r'\$?[A-Z]{2,5}')
BLACKLIST = {"USA", "CEO", "GDP"}


def extract_tickers(text, universe):
    candidates = set(TICKER_REGEX.findall(text))
    return [
        t.replace('$', '')
        for t in candidates
        if t.replace('$', '') in universe and t not in BLACKLIST
    ]


def ingest_reddit(universe):
    reddit = praw.Reddit(
        client_id="XXX",
        client_secret="XXX",
        user_agent="sentiment_scanner"
    )

    rows = []

    for sub in ["wallstreetbets", "stocks", "pennystocks"]:
        for post in reddit.subreddit(sub).new(limit=300):
            tickers = extract_tickers(
                f"{post.title} {post.selftext}", universe
            )

            for t in tickers:
                rows.append({
                    "timestamp": datetime.utcfromtimestamp(post.created_utc),
                    "source": "reddit",
                    "ticker": t,
                    "sentiment": None,  # will add later
                    "engagement": post.score + post.num_comments
                })

    return pd.DataFrame(rows)
```


***


# STEP 4️⃣ – sentiment classification (simple and effective)


### MVP:

- Vader (fast)
- FinBERT later[1]


```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


analyzer = SentimentIntensityAnalyzer()


def classify_sentiment(df):
    df["sentiment"] = df["text"].apply(
        lambda x: analyzer.polarity_scores(x)["compound"]
    )
    return df
```


⚠️ **Important**: sentiment is secondary – mentions are primary.[2]


***


# STEP 5️⃣ – aggregation over windows (3 / 10 / 20 days)


```python
def aggregate_sentiment(df, as_of_date):
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date

    agg_rows = []

    for ticker in df["ticker"].unique():
        df_t = df[df["ticker"] == ticker]

        for window in [3, 10, 20]:
            cutoff = as_of_date - pd.Timedelta(days=window)
            slice_ = df_t[df_t["date"] >= cutoff]

            if slice_.empty:
                continue

            agg_rows.append({
                "ticker": ticker,
                "window": window,
                "mentions": len(slice_),
                "sentiment_avg": slice_["sentiment"].mean(),
                "engagement": slice_["engagement"].sum()
            })

    return pd.DataFrame(agg_rows)
```


***


# STEP 6️⃣ – acceleration (key metric)


```python
def compute_acceleration(df):
    pivot = df.pivot(index="ticker", columns="window", values="mentions")

    pivot["mentions_acceleration"] = (
        pivot[3] / pivot[20]
    ).replace([float("inf"), -float("inf")], 0)

    return pivot.reset_index()
```


***


# STEP 7️⃣ – merge with volume candidates


```python
def merge_with_volume(sentiment_df, candidates_df):
    return candidates_df.merge(
        sentiment_df,
        on="ticker",
        how="left"
    ).fillna(0)
```


***


# STEP 8️⃣ – ranking (what goes to Telegram)


```python
def rank(df):
    df["score"] = (
        0.5 * df["volume_zscore"] +
        0.4 * df["mentions_acceleration"] +
        0.1 * df["sentiment_avg"]
    )

    return df.sort_values("score", ascending=False)
```


***


# STEP 9️⃣ – what you get each day


📁 `sentiment_ranked.csv`


```csv
ticker,volume_zscore,mentions_acceleration,sentiment_avg,score
SMCI,3.1,4.2,0.61,3.48
CVNA,2.7,2.9,0.55,2.94
```


👉 This is **ready input for Telegram alerts**.[1]


***


# STEP 🔟 – next


Proposed next steps:
1️⃣ **Ticker extraction hardening** (false positives)  
2️⃣ **Historical baseline (rolling 20d CSV)**  
3️⃣ **Alert rules (early / exhaustion)**[1]


***


Great. **Step 2 – “hardening” ticker extraction**.  
This is critical for **small/mid caps**, otherwise noise kills the signal.[2]


Goal:
👉 **extract ONLY real tickers from your universe**,  
👉 **minimize false positives**,  
👉 **keep real early mentions**.[1]


***


# STEP 2️⃣ – HARDENED TICKER EXTRACTION


## Problems of the naïve approach (what is fixed)


❌ `$USA`, `CEO`, `FDA`, `AI`, `GDP`  
❌ Normal words in CAPS  
❌ Company names without `$`  
❌ Tickers inside words  
❌ Spam posts with dozens of tickers[3]


***


## Extraction architecture (proper)


```text
Raw text
  ↓
Regex candidates
  ↓
Context filters
  ↓
Universe validation
  ↓
Frequency sanity check
  ↓
Final tickers
```


***


## 1️⃣ Universe as main filter


You already work with NASDAQ small/mid caps → that is the anchor.[2]


```python
def load_universe(path="nasdaq_universe.csv"):
    return set(pd.read_csv(path)["ticker"].str.upper())
```


📌 **Rule**

> If the ticker is not in the universe – it does not exist.[2]


***


## 2️⃣ Regex: more careful than `$?[A-Z]{2,5}`


### Use **word boundaries**


```python
TICKER_REGEX = re.compile(
    r'(?<![A-Z])\$?[A-Z]{2,5}(?![A-Z])'
)
```


This removes:

- `AIPOWER`
- `USAID`
- `FDAAPPROVAL`[3]


***


## 3️⃣ Hard blacklist (mandatory)


```python
BLACKLIST = {
    "USA", "CEO", "CFO", "GDP", "FED", "SEC",
    "AI", "IPO", "ETF", "FDA", "COVID",
    "IMO", "YOLO", "DD", "ATH"
}
```

💡 Better store blacklist in `blacklist.txt`.[3]


***


## 4️⃣ Context filter (key trick)


### Ticker is valid if:

- there is a `$` nearby
- **or** trading words nearby[2]


```python
CONTEXT_WORDS = {
    "buy", "sell", "long", "short",
    "calls", "puts", "bullish", "bearish",
    "breakout", "pump", "squeeze"
}
```


### Context check


```python
def has_trading_context(text, ticker):
    text = text.lower()
    ticker = ticker.lower()

    window = 20
    idx = text.find(ticker)
    if idx == -1:
        return False

    start = max(0, idx - window)
    end = idx + window

    context = text[start:end]
    return any(w in context for w in CONTEXT_WORDS)
```


***


## 5️⃣ Limit “tickers per post”


🚨 Spam posts often contain **10–30 tickers** → not a signal.[2]


```python
MAX_TICKERS_PER_POST = 5
```


***


## 6️⃣ Final extraction function


```python
def extract_tickers(text, universe):
    candidates = set(TICKER_REGEX.findall(text))
    tickers = []

    for raw in candidates:
        t = raw.replace("$", "").upper()

        if t in BLACKLIST:
            continue
        if t not in universe:
            continue
        if not has_trading_context(text, t) and "$" not in raw:
            continue

        tickers.append(t)

    if len(tickers) > MAX_TICKERS_PER_POST:
        return []

    return tickers
```


***


## 7️⃣ Anti‑bias protection (important)


❗ **Do NOT**:

- filter by market cap at this stage
- filter by price
- filter by “fame”[2]


Otherwise early pumps are killed.[2]


***


## 8️⃣ What to write into `sentiment_raw.csv` now


```csv
timestamp,source,ticker,engagement
2025-12-27T14:03:00,reddit,SMCI,214
```


➡ sentiment will be added later  
➡ raw must always be “clean”.[1]


***


## 9️⃣ Quick self‑check (recommended)


Once per day:


```python
df["ticker"].value_counts().head(20)
```


If you see:

```text
USA
CEO
FDA
```

→ blacklist has holes.[3]


---


## 1️⃣0️⃣ What you get after Step 2


✅ Minimal noise  
✅ Correct mentions count  
✅ Real acceleration spikes  
✅ Sentiment starts to **work**, not hurt.[2]


***


## Next step (Step 3)


👉 **Historical baseline (rolling 20d CSV)** so acceleration is computed **relative to history**, not “from thin air”.[1]


***


Great. **Step 3 – Historical baseline without DB**, fully compatible with your scheme (**Python + pandas + CSV, daily run**).[1]


Goal:
👉 keep **real mentions history**  
👉 correctly compute **acceleration (3 / 10 / 20 days)**  
👉 avoid storing “yesterday’s world” in process memory.[1]


***


# STEP 3️⃣ – HISTORICAL BASELINE (CSV‑only)


## Key idea


❗ Do not recompute the past.  
❗ Each day **add one row per ticker**.  
❗ History = simple CSV file.[1]


***


## 1️⃣ Where to store history


Next to daily folders:


```text
sentiment_history/
 └── mentions_daily.csv
```


***


## 2️⃣ `mentions_daily.csv` format


```csv
date,ticker,mentions,engagement
2025-12-05,SMCI,12,340
2025-12-06,SMCI,9,210
2025-12-07,SMCI,0,0
```


📌 **IMPORTANT**  
If a ticker was not mentioned → **explicitly write `0`**.  
Otherwise rolling is broken.[1]


***


## 3️⃣ Daily append (after ingestion)


```python
from pathlib import Path
import pandas as pd


HISTORY_PATH = Path("sentiment_history/mentions_daily.csv")


def update_mentions_history(daily_mentions_df, date):
    """
    daily_mentions_df:
      ticker | mentions | engagement
    """

    daily_mentions_df["date"] = date

    if HISTORY_PATH.exists():
        hist = pd.read_csv(HISTORY_PATH)
        hist = pd.concat([hist, daily_mentions_df], ignore_index=True)
    else:
        hist = daily_mentions_df.copy()

    hist.to_csv(HISTORY_PATH, index=False)
```


***


## 4️⃣ Mandatory zero‑fill


This is **critical**.[1]


```python
def fill_missing_tickers(hist, universe, date):
    existing = set(hist[hist["date"] == date]["ticker"])

    missing = universe - existing

    rows = [{
        "date": date,
        "ticker": t,
        "mentions": 0,
        "engagement": 0
    } for t in missing]

    return pd.concat([hist, pd.DataFrame(rows)], ignore_index=True)
```


💡 Use **only small/mid cap universe**, not all NASDAQ.[2]


***


## 5️⃣ Rolling windows (3 / 10 / 20 days)


```python
def compute_rolling(hist):
    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist.sort_values(["ticker", "date"])

    out = []

    for ticker, df_t in hist.groupby("ticker"):
        df_t = df_t.set_index("date")

        for w in [3, 10, 20]:
            roll = df_t["mentions"].rolling(w).sum().iloc[-1]

            out.append({
                "ticker": ticker,
                "window": w,
                "mentions": roll
            })

    return pd.DataFrame(out)
```


***


## 6️⃣ Acceleration (correct formula)


```python
def compute_acceleration(rolling_df):
    pivot = rolling_df.pivot(
        index="ticker",
        columns="window",
        values="mentions"
    ).fillna(0)

    pivot["mentions_acceleration"] = (
        pivot[3] / pivot[20].replace(0, 1)
    )

    return pivot.reset_index()
```


📌 **Why `replace(0,1)`?**  
To avoid `inf` and not drop new tickers.[1]


---


## 7️⃣ Join with current candidates


```python
def enrich_candidates(candidates_df, accel_df):
    return candidates_df.merge(
        accel_df[["ticker", "mentions_acceleration"]],
        on="ticker",
        how="left"
    ).fillna(0)
```


***


## 8️⃣ Quality checks (mandatory)


Log daily:


```python
accel_df.sort_values(
    "mentions_acceleration",
    ascending=False
).head(10)
```


Expected:

- 1–5 tickers with acceleration > 3
- >10 → extraction is noisy
- 0 → data sources are dead[1]


***


## 9️⃣ What you have after Step 3


✅ Real baseline  
✅ Acceleration without “first‑day effect”  
✅ Works for years without DB  
✅ Easy backtests (CSV = history)[1]


***


## 🔜 Step 4 (next)


👉 **Scoring + ranking + alert rules** (early pump / exhaustion / ignore).[2]


***


Moving on. **Step 4 – Scoring, ranking and alert logic**.  
This is the **last layer before Telegram**, everything above prepares the signal.[2]


You already have:

- volume filter (small / mid caps)
- candidates.csv (3–10 tickers)
- mentions acceleration (3 / 10 / 20)
- daily pipeline[1]


Now add:
👉 **clear score**  
👉 **deterministic rules**  
👉 **CSV ready for Telegram**.[1]


***


# STEP 4️⃣ – SCORING + ALERT RULES


## 1️⃣ What goes into the score (minimal that works)


For each ticker:


| Factor                    | Why                              |
| ------------------------- | -------------------------------- |
| `volume_zscore`          | confirms real interest           |
| `mentions_acceleration`  | early pump                       |
| `mentions_3d`            | avoid “1 mention = x10”          |
| `sentiment_avg` (optional) | noise filter                   | [2]


❗ **Price / RSI / MACD are NOT needed here** – they live in your indicators layer.[2]


***


## 2️⃣ Normalization (important)


Do not let one extreme break ranking.[2]


```python
def clip(series, low=0, high=5):
    return series.clip(lower=low, upper=high)
```


***


## 3️⃣ Score formula (battle‑tested logic)


```python
def compute_score(df):
    df["accel_c"] = clip(df["mentions_acceleration"], 0, 5)
    df["vol_c"] = clip(df["volume_zscore"], 0, 5)

    df["score"] = (
        0.55 * df["accel_c"] +
        0.35 * df["vol_c"] +
        0.10 * df.get("sentiment_avg", 0)
    )

    return df.sort_values("score", ascending=False)
```


📌 **Why like this:**

- mentions > volume (earlier signal)
- sentiment gets light weight[2]


***


## 4️⃣ Alert classes (key part)


Do **NOT** send everything.  
Each ticker gets an **alert type**.[2]


***


### 🚀 ALERT: EARLY PUMP CANDIDATE


```python
def is_early_pump(row):
    return (
        row["mentions_acceleration"] >= 3 and
        row["mentions_3d"] >= 5 and
        row["volume_zscore"] >= 2
    )
```


📌 This is your **main Telegram alert**.[2]


***


### ⚠️ ALERT: FOMO / LATE STAGE


```python
def is_fomo(row):
    return (
        row["mentions_acceleration"] < 1.2 and
        row["mentions_10d"] > row["mentions_20d"] * 0.8 and
        row["volume_zscore"] >= 3
    )
```


📌 Use as:

- “do not enter”
- or “scalp only”[2]


***


### ❌ IGNORE


```python
def is_noise(row):
    return (
        row["mentions_3d"] <= 1 and
        row["volume_zscore"] < 2
    )
```


***


## 5️⃣ Assigning alert type


```python
def assign_alert_type(df):
    def classify(row):
        if is_early_pump(row):
            return "EARLY_PUMP"
        if is_fomo(row):
            return "FOMO"
        if is_noise(row):
            return "IGNORE"
        return "WATCH"

    df["alert_type"] = df.apply(classify, axis=1)
    return df
```


***


## 6️⃣ Final CSV (input for Telegram)


📁 `sentiment_ranked.csv`


```csv
ticker,score,alert_type,volume_zscore,mentions_3d,mentions_acceleration
SMCI,3.82,EARLY_PUMP,3.1,14,4.6
CVNA,2.41,WATCH,2.7,6,2.1
XYZ,1.12,FOMO,3.8,18,0.9
```


👉 Your Telegram service:

- filters `alert_type == EARLY_PUMP`
- sorts by `score`[1]


***


## 7️⃣ Sanity checks (mandatory)


Log daily:


```python
df["alert_type"].value_counts()
```


Expected:

- EARLY_PUMP: 0–2
- WATCH: 2–6
- FOMO: 0–3
- IGNORE: the rest[2]


If:

- many EARLY → extraction is noisy
- always 0 → sources are weak[2]


***


## 8️⃣ How this fits your pipeline


```text
Volume filter
   ↓
Candidates.csv
   ↓
Sentiment enrichment
   ↓
Rolling baseline
   ↓
Scoring
   ↓
Alert classification
   ↓
Telegram
```


***


## What next (optional but powerful)


### 🔥 Step 5 (if desired)

- correlation: sentiment → price lag
- auto‑tune thresholds
- backtest “sentiment‑before‑volume”[2]


***


Short answer: **for a start – yes, Reddit alone is enough**.  
Long and important answer – **why exactly, and when it stops being enough**.[2]


***


## Why Reddit alone is enough (initially)


### 1️⃣ Reddit = **earliest source**


For **small / mid cap NASDAQ**:

- first “seeds” of a pump almost always appear **on Reddit**
- Twitter/X usually **lags**
- News usually **confirms, not initiates**[2]


Historically:

```text
Reddit → Volume → Price → News → Twitter
```

You **already catch volume** → Reddit closes the **early phase**.[2]


***


### 2️⃣ You have few candidates (3–10 per day)


This is critical.[2]


If you scanned **the whole market** → Reddit alone would be weak.  
But your flow is:

```text
Volume filter → 3–10 tickers → sentiment check
```

For this flow:

- Reddit gives **enough differentiation**
- extra sources increase noise[2]


***


### 3️⃣ Mentions acceleration > coverage


You measure:

- not absolute count
- but **sharp change**[2]


For that you need:

- **stable source**
- Reddit has been stable for years[2]


***


## When Reddit becomes insufficient


Clear signs 👇


### 🚨 1. You see a pump, but acceleration = 0


Price + volume move,  
but Reddit **is silent** → signal missed.[2]


👉 That means:

- ticker is “institutional / news‑driven”
- or pump started **on Twitter / Discord**[2]


***


### 🚨 2. Too many false Reddit spikes


Many:

- mentions
- acceleration  

but:

- price does not react[2]


👉 That is meme noise → needs another source as filter.[2]


***


### 🚨 3. You want **even earlier**


Earlier than Reddit = only:

- Discord
- private Telegram
- options flow[2]


***


## Best extension order (NOT all at once)


### 🥇 #1 – Stocktwits (add first)


Why:

- pre‑labeled bullish / bearish
- simple API
- ticker‑centric[2]


Adds:

- confidence to Reddit
- removes trash[2]


📌 Use as **confirmation**, not primary driver.[2]


***


### 🥈 #2 – Twitter/X (careful)


Good for:

- biotech
- news‑driven names[2]


But:

- very noisy
- tricky rate limits
- many bots[2]


***


### 🥉 #3 – News (last)


Use:

- only headline count
- without complex NLP[1]


***


## Practical recommendation specifically for you


### Now:

✅ Keep Reddit as **only source**.  
✅ Collect **30–60 days of history**.  
✅ Check:

- how many EARLY_PUMP actually “fire”
- lag to price move.[2]


***


### Later (in a month):

Add **Stocktwits** as binary filter:


```text
Reddit accel > 3
AND Stocktwits mentions > 0
```


This sharply boosts precision.[2]


***


## Bottom line


| Question                           | Answer        |
| ---------------------------------- | ------------- |
| Is Reddit enough?                  | ✅ Yes, for MVP |
| Will you miss some pumps?          | ❌ Yes         |
| Will 2nd source help a lot?        | ⭐⭐⭐⭐          |
| Do you need everything at once?    | ❌ No          | [2]

[1](https://www.deepl.com/en/translator/l/ru/en)
[2](https://www.ig.com/en/trading-strategies/28-trading-slang-expressions-every-trader-should-know-210727)
[3](https://files.consumerfinance.gov/f/documents/cfpb_adult-fin-ed_russian-style-guide-glossary.pdf)
[4](https://products.groupdocs.app/translation/markdown/russian-english)
[5](https://openl.io/translate/markdown)
[6](https://simplelocalize.io/markdown-translations/)
[7](https://linnk.ai/tools/markdown-translator/)
[8](https://github.com/playcanvas/markdown-translator)
[9](https://www.online-translator.com/translation/english-russian/sentiment%20analysis)
[10](https://dev.to/ryuya/translate-markdown-to-any-language-github-bot-3pde)
[11](https://pdfs.semanticscholar.org/cbb8/1522a18e0146832dae954a19a39bd9555549.pdf)
[12](https://www.smartcat.com/picture-translator/translate-russian-image-to-english/)
[13](https://www.youtube.com/watch?v=X76Q5bTL0Ac)
[14](https://techcommunity.microsoft.com/blog/educatordeveloperblog/automate-markdown-and-image-translations-using-co-op-translator-phi-3-cookbook-c/4263474)
[15](https://www.sciencedirect.com/science/article/pii/S2214845025000146)
[16](https://www.npmjs.com/package/@diplodoc%2Fmarkdown-translation)
[17](https://journals.rudn.ru/linguistics/article/view/17848/15443)
[18](https://documents.worldbank.org/en/publication/documents-reports/documentdetail/655641468740656535)
[19](https://en.bab.la/dictionary/english-russian/bullish-sentiment)