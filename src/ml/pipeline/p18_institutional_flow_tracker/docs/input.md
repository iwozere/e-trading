# P18 — Institutional Flow Tracker: Input & Ideas

## Problem Statement

Large financial institutions (hedge funds, mutual funds, banks, insurance companies) periodically
liquidate equity positions to raise cash — for upcoming IPO commitments, fund redemptions,
rebalancing, or regulatory requirements. These forced or strategic sell-offs create predictable
price dislocations that can be detected and traded.

**Core question:** Can we identify when "smart money" is exiting a position, before or during the
price drop, and use that signal to avoid losses or enter contrarian positions?

---

## Signal Sources

### 1. SEC Regulatory Filings (Quarterly, Free)

| Filing | Content | Lag | Source |
|--------|---------|-----|--------|
| **Form 13F** | All equity positions of institutions with >$100M AUM | ~45 days after quarter-end | SEC EDGAR |
| **Form 4** | Insider / director transactions | Within 2 business days | SEC EDGAR |
| **Schedule 13D/G** | Ownership >5% stake changes | Within 10 days of threshold cross | SEC EDGAR |

**Key insight:** 13F data is the backbone — compare Q-to-Q position deltas to detect exits.
CUSIPs must be mapped to tickers via the OpenFIGI API (free).

### 2. Market Microstructure Signals (Intraday)

| Signal | Threshold / Rule | Interpretation |
|--------|-----------------|----------------|
| **Volume spike** | Daily volume > 3–5× 20-day average | Large participant active |
| **Block trades** | Single trade > 10,000 shares or >$1M | Institutional order |
| **Dark pool print %** | ATS volume share > 40% on a given day | Institutions routing away from lit markets |
| **Price-volume divergence** | Price flat or rising, volume 2× normal | Distribution (selling into strength) |
| **Bid-ask spread widening** | Spread > 2× 5-day average | Informed sellers present |

### 3. Derivative Market Signals

- **Unusual put volume:** Put/Call ratio spike on a stock → protection buying before large unwind
- **Deep OTM put buying:** Hedging a large long position before exit
- **Volatility skew steepening:** Market pricing in downside risk

### 4. IPO Calendar Correlation

When a major underwriter (Goldman, Morgan Stanley, JPMorgan) has a large IPO in the pipeline:
1. Pull their latest 13F
2. Check if they trimmed positions in liquid large-caps (freeing up capital)
3. The trimmed stock may be temporarily depressed due to forced selling
4. Post-IPO lockup expiry (~180 days) is another sell-off trigger

### 5. Fund Redemption Seasonal Patterns

| Period | Typical Behaviour |
|--------|-----------------|
| Late October – mid-December | Tax-loss harvesting; hedge fund redemption gates (Dec 31) |
| March 31 | Fiscal year-end for many funds → rebalancing |
| September 30 | Q3 close → window dressing |
| Any large market drawdown | Redemption waves → forced selling across holdings |

---

## Data Pipeline Architecture

```
SEC EDGAR (13F XML)
        │
        ▼
 13F Downloader          OpenFIGI API
        │                     │
        ▼                     ▼
 CUSIP → Ticker Mapper ───────┘
        │
        ▼
 Position Delta Calculator
 (Q vs Q-1 holdings per institution)
        │
        ├──► Exit Screener (position reduced >30% or fully closed)
        │
        ├──► Concentration Screener (position now <1% of portfolio)
        │
        └──► Multi-Inst Consensus (same stock exited by 3+ institutions)

Market Data Feed (Yahoo / IBKR)
        │
        ▼
 Volume Anomaly Detector
        │
        ├──► Spike detector (>N × rolling average)
        └──► Dark pool % monitor (FINRA ATS data)

IPO Calendar Feed
        │
        ▼
 Underwriter → 13F Exit Correlator

        All signals
            │
            ▼
     Composite Score
     (weighted sum of active signals)
            │
            ▼
     Alert Engine → Telegram notification
```

---

## Output Signals & Use Cases

### A. Avoidance Signal (Risk Management)
- **Trigger:** 2+ institutions exited last quarter + volume anomaly in current week
- **Action:** Flag stock as "institutional distribution — avoid or reduce"
- **Use:** Pre-filter for stock screeners; remove from buy watchlist

### B. Contrarian Entry Signal
- **Trigger:** High-quality stock (strong fundamentals) + institutional exit driven by
  forced/structural reasons (fund redemption, IPO capital raise) + price down >15% from 52w high
- **Action:** Flag as potential dislocation opportunity
- **Use:** Add to accumulation watchlist with price targets

### C. Sector Rotation Signal
- **Trigger:** Multiple institutions simultaneously reducing sector A while adding sector B
  (visible in 13F deltas across institutions)
- **Action:** Flag macro rotation → align portfolio sector weights
- **Use:** Tactical allocation adjustments

### D. IPO Arbitrage Signal
- **Trigger:** Underwriter trimmed stock X in Q before IPO Y (same underwriter)
- **Action:** Monitor stock X post-IPO for mean reversion
- **Use:** Short-term trade idea; 2–6 week horizon

---

## Implementation Priorities

### Phase 1 — Foundation (Data Layer)
- [ ] SEC EDGAR 13F downloader (XML parser → structured DataFrame)
- [ ] OpenFIGI CUSIP-to-ticker mapper with local cache
- [ ] Q-to-Q position delta calculator
- [ ] Store historical 13F snapshots (DuckDB or SQLite)

### Phase 2 — Signal Generation
- [ ] Exit screener (position reduction > configurable threshold)
- [ ] Multi-institution consensus detector
- [ ] Volume anomaly detector (integrate with existing market data feed)
- [ ] Composite signal scorer

### Phase 3 — Integration
- [ ] Wire into existing scheduler as `data_processing` job (quarterly on 13F filing dates)
- [ ] Telegram alerts for high-score signals
- [ ] Stock screener pre-filter (exclude stocks under institutional distribution)

### Phase 4 — Advanced (Deferred)
- [ ] Dark pool / ATS data ingestion (FINRA ATS feed)
- [ ] Options flow integration (put volume anomaly)
- [ ] IPO calendar scraper + underwriter correlation
- [ ] ML model: predict post-exit price trajectory using historical 13F exit patterns

---

## Key External Dependencies

| Dependency | Purpose | Cost |
|-----------|---------|------|
| SEC EDGAR full-text search | 13F XML bulk download | Free |
| OpenFIGI API | CUSIP → ticker mapping | Free (rate-limited) |
| Yahoo Finance / yfinance | Historical + intraday volume | Free |
| FINRA ATS transparency data | Dark pool volume by stock | Free (monthly CSV) |
| Unusual Whales API | Options flow, dark pool real-time | Paid (~$50/mo) |
| Quandl / Nasdaq Data Link | Institutional ownership data | Paid (or free tier) |

---

## Related Pipelines / Modules
- `src/data/` — market data feed (volume, price)
- `src/notification/` — Telegram alert delivery
- `p15_gdelt_sentiment` — news sentiment (complement: news often follows institutional moves)
- Existing stock screeners — output of this pipeline feeds their pre-filter

---

## Open Questions
1. Should 13F data be stored per-institution or aggregated per-stock?
2. How to handle restatements / amended 13F filings?
3. Minimum AUM threshold for institutions to track (>$1B? >$5B?)?
4. Should the composite score be rule-based or trained (classification model)?
5. How to account for passive index funds whose sells are mechanical, not informational?
