````md
# Explosive Penny Stock Screener — Agent Pipeline Specification

Version: 1.1  
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
│   ├── data_quality_agent.py
│   ├── market_agent.py
│   ├── fundamentals_agent.py
│   ├── catalyst_agent.py
│   ├── dilution_agent.py
│   ├── short_squeeze_agent.py
│   ├── technical_agent.py
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
NASDAQ / NYSE-AMERICAN LIST
    ↓
Universe Filter
    ↓
Data Quality Check  ← data freshness, corporate actions, missing fields
    ↓
Market Data Enrichment  ← OHLCV, float, SI, pre-market, VWAP
    ↓
Fundamental Enrichment
    ↓
Catalyst / News Analysis
    ↓
Dilution Risk Detection  ← hard penalty deduction
    ↓
Short Squeeze Detection  ← SI%, days-to-cover, borrow rate
    ↓
Technical Pattern Detection  ← breakout, compression, accumulation
    ↓
Composite Scoring Engine  ← normalized sub-scores, weighted sum
    ↓
Ranking + Tier Assignment  ← A / B / C / W (Watchlist)
    ↓
Report Generation
    ↓
Notification Delivery
    ↓
Historical Snapshot Storage  ← for backtesting and ML
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

## Market Data — Additional Required Fields

The following fields are essential and must be added to the data model:

```text
52_week_high
52_week_low
vwap_intraday         # critical for intraday run detection
premarket_volume      # leading indicator for gap-up setups
premarket_price_chg   # % change vs prior close
short_interest_shares
short_interest_pct_float
days_to_cover
```

---

## Data Quality and Freshness

All ingested data must carry a `data_as_of` timestamp.

### Staleness Policy

```yaml
MARKET_DATA_MAX_AGE_HOURS: 2
FUNDAMENTAL_DATA_MAX_AGE_DAYS: 95   # roughly one quarter; flag if older
NEWS_DATA_MAX_AGE_HOURS: 24
```

### Handling Missing Data

```text
If market data missing      → skip ticker, log warning
If fundamentals missing     → score fundamentals_score = 0, flag as "fundamentals unknown"
If float missing            → skip ticker (float is required for liquidity assessment)
If short_interest missing   → score short_squeeze_score = 0, flag as "SI unknown"
```

Do NOT impute or estimate missing fundamental values.
A ticker with unknown fundamentals can still qualify via momentum + catalyst path.

### Corporate Actions Handling

Price history must be adjusted for:

* forward/reverse stock splits
* ticker symbol changes (carry historical data to new ticker)
* delistings (remove from universe immediately)

Sources for corporate action data:

* Polygon corporate actions API
* SEC filings (8-K for reverse splits)

Flag any ticker that executed a **reverse split in the past 12 months** for enhanced
dilution scrutiny — this is a strong negative signal independent of current price.

---

# 7. Hard Filters

Reject immediately if:

## Exchange

```yaml
ALLOWED_EXCHANGES:
  - NASDAQ
  - NYSE_AMERICAN     # AMEX — hosts many legitimate explosive small-caps

EXCLUDED_EXCHANGES:
  - OTC
  - PINK
  - GREY              # OTC/Pink/Grey Sheet stocks: insufficient regulatory oversight,
                      # no minimum listing standards, highest pump-and-dump exposure
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

## 8.1 Momentum Features

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

## 8.2 Technical Breakout Features

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

## 8.3 Accumulation Features

Bullish:

* multiple green high-volume days
* close near daily highs
* OBV rising
* accumulation/distribution improving

---

## 8.4 Fundamental Acceleration

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

## 8.5 Dilution Risk Detection

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

## 8.6 Catalyst Detection

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

## 8.7 Social / Sentiment

Optional.

Track:

* Reddit mentions
* Stocktwits acceleration
* X/Twitter mentions

Important:
Social sentiment alone must NEVER trigger candidate selection.

Only additive.

---

## 8.8 Short Squeeze Detection

HIGH PRIORITY MODULE.

Short squeezes are among the most explosive and reliable penny stock move catalysts.
The `short_interest` field from market data must feed a dedicated feature set.

### Required Fields

```text
short_interest_shares
short_interest_pct_float
days_to_cover            = short_interest_shares / avg_volume_30d
borrow_rate              (if available via data provider)
```

### Scoring Signals

Strong squeeze setup:

```yaml
SHORT_INTEREST_THRESHOLD: 20%    # of float
DAYS_TO_COVER_THRESHOLD: 3       # high squeeze potential above this
```

Score escalation:

```text
SI < 10% float   → base score
SI 10–20% float  → moderate squeeze potential
SI > 20% float   → high squeeze potential
SI > 30% float   → extreme squeeze (adds to score but flags halt risk)
```

### Squeeze Trigger Confirmation

A short squeeze setup alone is NOT sufficient.
It must coincide with:

* relative_volume > 2.5
* price moving against the short (upward momentum)
* catalyst or volume breakout

### Risk Note

High short interest + low liquidity = halt risk.
Apply additional liquidity check when `short_interest_pct_float > 25%`:

```yaml
MIN_DAILY_VOLUME_SQUEEZE: 1000000
```

---

# 9. Composite Scoring System

## Final Score

```text
FINAL_SCORE =
0.25 * momentum_score +
0.20 * volume_score +
0.15 * technical_score +
0.15 * fundamentals_score +
0.10 * catalyst_score +
0.10 * short_squeeze_score +
0.05 * accumulation_score
```

Weights sum to 1.00. Configurable via `config/weights.yaml`.

### Dilution Penalty

Applied as a hard point deduction AFTER the weighted sum:

```text
FINAL_SCORE -= dilution_penalty
```

Where `dilution_penalty` is the sum of applicable penalties from section 8.5.
Dilution risk is treated as a hard deduction — not a weighted component — because
a confirmed serial diluter must be demoted regardless of other strong signals.

---

## Sub-Score Normalization

Each sub-score MUST be normalized to [0, 100] before the weighted sum.

Use min-max normalization across all candidates on each run day:

```text
score_normalized = 100 * (raw - min_raw) / (max_raw - min_raw)
```

Or use fixed thresholds (preferred for interpretability):

| Sub-score          | 0 (min)         | 50 (mid)      | 100 (max)          |
|--------------------|-----------------|---------------|--------------------|
| momentum_score     | ≤ -20% 20d ret  | flat          | ≥ +50% 20d ret     |
| volume_score       | rvol < 1.0      | rvol = 2.0    | rvol ≥ 5.0         |
| technical_score    | no signals      | 1 pattern     | breakout + pattern |
| fundamentals_score | declining rev   | stable        | rev accel > 50%    |
| catalyst_score     | no catalyst     | minor news    | tier-1 catalyst    |
| short_squeeze_score| SI < 5% float   | SI ~15% float | SI > 25% float     |
| accumulation_score | distribution    | neutral       | strong accumulation|

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

## Tier W — Watchlist (Pre-Alert)

Tickers that are ONE condition away from qualifying as Tier A or B.

Examples:

* strong fundamentals + breakout setup but relative_volume currently 2.1 (threshold 3.0)
* excellent catalyst but no technical breakout yet
* high short interest but no volume trigger yet

Purpose:

* surface setups to watch before they trigger
* allow intraday monitoring for condition completion
* reduce false-negative rate from rigid thresholding

Tier-W tickers appear in the daily report's "Watching" section but do NOT trigger alerts.

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

## Entry Rules

Entry price must be defined before computing any metric:

```yaml
ENTRY_METHOD: next_open          # default: open price of the trading day after alert
ENTRY_ALTERNATIVES:
  - breakout_print               # price at which volume breakout confirmed intraday
  - vwap_cross                   # entry at first VWAP cross after alert
```

Use `next_open` as the primary method — it is the most conservative and realistic.

---

## Exit Rules

```yaml
EXIT_METHODS:
  trailing_stop:
    initial_stop_pct: 8          # 8% trailing stop from entry
  time_based:
    days: [1, 5, 10, 20]        # forced exit at end of Nth trading day
  target_based:
    targets: [20, 50, 100]      # % gain targets — measure how often hit before stop
```

All backtests should run under ALL exit methods and report separately.

---

## Metrics

* win rate (per exit method)
* average return (per exit method)
* median return
* max drawdown (per candidate, portfolio-level)
* Sharpe ratio (annualized, using daily returns)
* false breakout rate (breakout detected but -10% within 5 days)
* tier accuracy: % of Tier-A alerts that beat Tier-B and Tier-C

---

## Holding Period Tests

Test:

* 1d
* 5d
* 10d
* 20d

---

## Concentration Risk in Backtests

Flag days where >3 Tier-A candidates come from the same sector.
Sector concentration is a common source of correlated drawdowns.

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

Scan pre-market movers using `premarket_volume` and `premarket_price_chg`.
Surface gap-up + high pre-market volume as early Tier-W candidates.

### Market Open Watch (09:30–10:30 ET)

```text
09:30 ET  — first 5-min bar
09:45 ET  — volume trend confirmation
10:00 ET  — breakout confirmation scan
```

This window is optional but high-value: the first hour produces the majority
of intraday explosive moves. Run a lightweight scan at each interval checking
Tier-W tickers from the pre-market run for condition completion.

### Midday update

```text
12:00 ET
```

### Post-market

```text
17:00 ET
```

---

## Intraday Volume Spike Trigger

For Tier-W tickers already in the watchlist, implement a real-time volume
spike monitor:

```yaml
INTRADAY_RVOL_TRIGGER: 4.0      # if intraday rvol crosses this, re-score immediately
INTRADAY_CHECK_INTERVAL: 5min   # polling interval during market hours
```

When triggered: re-run scoring for that ticker and send alert if it now qualifies.

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

---

# 24. Scoring Calibration Backlog (open tuning items)

Concrete, evidence-backed tuning items identified after the **CatalystAgent (8-K
detection, §8.6) went live**. Until then `catalyst_score` was a hardcoded `0`
placeholder; with real catalyst signal now flowing, the composite scoring (§9)
and ranking tiers (§11) need re-calibration. These are near-term, data-driven
knobs — distinct from the long-term roadmap in §22.

## 24.1 Observed problem

For several consecutive sessions the screener produced **A=0, B=0** — every
candidate stuck at tier C or below — even though the catalyst stage is healthy
(e.g. run of 2026-06-26: 66/160 candidates flagged via the 8-K index cache).
A 24-day analysis (§24.4) **resolved this**: Tier B is healthy (fires ~42% of
days); the real, structural issue is that **Tier A is unreachable** under the
current scoring. It is *not* a data gap (catalyst signal now flows).

Reference case — **ILLR (2026-06-26)**, the top-ranked name, `final_score=54.0`,
exactly 1 point below `tier_b_min_score=55`, flagged `explosive_candidate=True`:

| sub-score      | value | × weight | points |
| -------------- | ----- | -------- | ------ |
| momentum       | **0.0** | 0.25   | **0.00** |
| volume         | 100   | 0.20     | 20.00  |
| technical      | 80    | 0.15     | 12.00  |
| fundamentals   | 50    | 0.15     | 7.50   |
| catalyst       | 100   | 0.10     | 10.00  |
| short_squeeze  | 10    | 0.10     | 1.00   |
| accumulation   | 71    | 0.05     | 3.55   |

> ⚠️ **ILLR is a misleading poster child — read this before using it as
> evidence.** Its `momentum_score=0` is **correct, not a defect**. P17 return
> fields are *fractions* (`TechnicalAgent._return` → `(end-start)/start`, where
> `1.0 = +100%`). ILLR's stored `price_5d_return=19.33` means **+1,933% in 5
> days**, and `price_20d_return=3.0` is **+300% clamped at `momentum_20d_max`**.
> `_momentum_score` deliberately returns `0` at that cap to penalise a *late
> euphoric spike* (a name already up ~19× in a week is a chase, not an
> early-stage entry — see §23). So ILLR landing at tier C is the anti-chase
> logic **working as designed**, not the tier wall clipping a good setup.

## 24.2 Tuning levers (priority order)

1. **Rebalance composite weights** (`P17ScoringConfig`, §9): now that catalyst is
   a *real* signal (was a hardcoded `0`), `weight_catalyst=0.10` may be too low
   for an explosive-penny thesis. **Note:** the §24.4 sweep found that reweighting
   alone only lifts the score ceiling from ~67.5 to ~70 and trades away Tier C
   breadth — it does **not** make Tier A (`>=75`) reachable. Treat reweighting as
   secondary to the §24.5 decision, and do not tune to make ILLR-type
   already-exploded names tier up.

2. **Smooth the momentum cap discontinuity** (minor): `_momentum_score` cliffs
   from ~100 (at +299%) to 0 (at the +300% cap). A monotone taper down from the
   cap would be less brittle than the hard cliff. Low priority — it only affects
   already-blown-off names that *should* score low anyway, so it does not explain
   A=0/B=0.

3. **Lower `tier_b_min_score` (last resort)**: least principled lever. Only
   revisit if the §24.3 sweep shows a real population of *early-stage* setups
   sitting just under 55.

> **Rejected idea — explosive → tier-B floor.** An earlier draft proposed
> flooring `explosive_candidate=True` names to tier B. ILLR shows why that is
> wrong: it satisfies every mandatory explosive gate (§10) yet is already up ~19×
> — flooring it would promote exactly the blow-off chases the screener is meant
> to avoid. If anything, the **explosive criteria (§10) need a "not already at
> the euphoric momentum cap" guard**, not a tier floor.

## 24.3 Validation methodology (avoid single-day overfit)

* **Do not calibrate against one day.** Tuning to a single date overfits.
* There is **no backtester engine** yet (§15 is design-only), so a forward-return
  P&L backtest is not runnable. The analysis below instead uses the **stored daily
  candidate CSVs** (`results/p17_penny_stocks/{date}/{date}_candidates.csv`) plus
  an **offline re-score** that credits the now-live 8-K catalyst (the
  `_job_edgar_8k_index` backfill covers the window). This recomputes
  `final_score`/tier from the persisted sub-scores without re-running the full
  pipeline, which makes weight/threshold sweeps fast.
* Re-check that the **catalyst flag rate is sane** — the 2026-06-26 run flagged
  ~41% (66/160), which is high; verify tier-2 keyword matches (`" ai "`,
  `partnership`, earnings items) are not over-flagging and inflating
  `catalyst_score`.

## 24.4 Analysis results (24-day window, 2026-05-15 → 06-26)

Ran 2026-06-28 over 24 days of stored output (3,609 candidate-rows), with an
offline catalyst-live re-score on the 16 fully-covered days (05-28 → 06-26).

**Finding 1 — Tier B is healthy.** B fired on **10 / 24 days** (~42%). The
A=0/B=0 that triggered this investigation was just a recent dry stretch, not a
systemic failure. No calibration action needed for B.

**Finding 2 — Tier A is structurally unreachable (the real issue).**

* Tier A fired **0 / 24 days**, ever.
* Global max `final_score` = **66.8** (stored) / **67.5** (catalyst-live
  re-score) vs `tier_a_min_score=75` — a **~7.5-point gap even in the best case**.
* Going catalyst-live barely moved tiers (`0/7/104 → 0/6/103`); the 0.10 catalyst
  weight cannot bridge that gap.
* *Why:* no name in 24 days scored high on momentum (.25) **and** volume (.20)
  **and** technical (.15) **and** catalyst (.10) at once. There is a structural
  trade-off — names with `momentum_score>=80` averaged `volume_score` **17**;
  names with `volume_score>=80` averaged `momentum_score` **55** (19% were `0` =
  already-exploded, ILLR-style). The best name ever (CPOP, 06-11: momentum 100,
  volume 100, fundamentals 100) still only reached 66.8 because `technical=20`,
  `short_squeeze=10`, `catalyst=0`.

**Sweep — totals across the 16 covered days** (offline re-score):

| variant | A | B | C | best max |
| ------- | - | - | - | -------- |
| baseline (`tier_a_min=75`, current weights) | 0 | 6 | 103 | 67.5 |
| `tier_a_min=70` (current weights) | 0 | 6 | 103 | 67.5 |
| `tier_a_min=68` (current weights) | 0 | 6 | 103 | 67.5 |
| catalyst-heavy weights¹ (`tier_a_min=75`) | 0 | 6 | 56 | 70.1 |
| catalyst-heavy weights¹ + `tier_a_min=70` | 1 | 5 | 56 | 70.1 |
| catalyst-heavy weights¹ + `tier_a_min=68` | 1 | 5 | 56 | 70.1 |

¹ catalyst-heavy = momentum .25→.15, catalyst .10→.20, fundamentals .15→.10,
accumulation .05→.10 (volume/technical/short-squeeze unchanged).

**Read-out:**

* **Lowering `tier_a_min` alone does nothing** until it drops *below the ~67.5
  ceiling* — 70 and 68 still yield 0 A. (My earlier "~68" suggestion was wrong;
  it sits just above the ceiling.)
* **Catalyst-heavy weights raise the ceiling only to ~70** and yield a *trickle*
  of A (1 in 16 days at `tier_a_min<=70`) — but collapse Tier C (103→56) by
  down-weighting momentum, i.e. they make the whole screen far more selective,
  not obviously better.
* **No modest reweight/threshold change cleanly "fixes" Tier A.** The ceiling is
  ~67–70 regardless.

## 24.5 Decision required (product call, not a bug)

Per §11, **Tier A = "Elite."** So 0 A's may be *correct by design*. Choose one:

1. **Tier A as a regular daily top-pick** → lower `tier_a_min_score` to **~65–67**
   (just below the empirical ceiling). Most direct; expect a handful of A's per
   month. Re-run the sweep harness to confirm the exact count before committing.
2. **Tier A as genuinely rare/elite** → keep `75`, accept A fires a few times a
   *year*, and treat **Tier B as the actionable daily list** (it already works).
3. **Loosen sub-score normalisation** (deeper change) — the real reason the
   ceiling is ~67 is that even great names leave `technical` / `short_squeeze` /
   `catalyst` sub-scores low. Revisit the §9 normalisation caps if Tier A should
   be reachable *without* lowering its threshold. Larger scope; do last.

Do **not** rebalance weights chasing Tier A before deciding what Tier A is *for*
— the sweep shows it trades away Tier C breadth for ~1 extra A.
