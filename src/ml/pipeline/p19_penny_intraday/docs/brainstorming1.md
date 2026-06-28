# Penny Stocks Predictive Pipeline: Architecture and Specification

This document outlines the architecture and data requirements for an automated pipeline designed to detect potential explosive movements ("shots") in the penny stocks market (stocks traded under $5). It is specifically optimized to utilize **only free data sources** and includes a specification for backtesting and parameter optimization.

---

## 1. Core Data Categories & Free Sources

[cite_start]Unlike large-cap stocks driven by macroeconomics, penny stocks move due to supply/demand imbalances, hype, and corporate catalysts[cite: 5, 6]. 

### A. Market Data (Technical Explosive Signals)
* [cite_start]**Low Float (< 20M shares):** The most critical metric[cite: 10, 11]. [cite_start]Low float means less supply; when demand spikes, the price skyrockets[cite: 12].
    * *Free Source:* **Finviz (Free Tier)** API/Scraper or **Yahoo Finance** (via `yfinance` library in Python) to extract share statistics.
* [cite_start]**Relative Volume (RVOL > 3-5x):** Compares current pre-market or early-session volume with the 30-to-60-day average[cite: 13, 14]. 
    * *Free Source:* Calculate manually via `yfinance` by fetching historical daily data and comparing it with current intraday volume.
* [cite_start]**Premarket Gappers (> 3-5% growth):** Scanning stocks before the market opens[cite: 15].
    * *Free Source:* **TradingView (Free Web Screener)** using pre-market filters or **Alpha Vantage (Free Tier with rate limits)**.
* [cite_start]**Short Interest (> 15-20% Short Float):** Fuel for a potential short squeeze[cite: 16].
    * *Free Source:* **Finviz** or **Shortsqueeze.com** (manual/scraped data).

### B. Corporate Catalysts (Fundamental Triggers)
[cite_start]Penny stocks rarely spike without an informational trigger[cite: 17, 18].
* [cite_start]**SEC Filings:** * *Form 8-K:* Major corporate events, mergers, contracts[cite: 20].
    * [cite_start]*Form S-1/S-3/424B:* Capital dilution tracking[cite: 21]. [cite_start]If a company constantly issues new shares to survive, the price will stall[cite: 22]. [cite_start]Filter these out[cite: 23].
    * [cite_start]*Form 4:* Insider buying (strong bullish signal)[cite: 24].
    * *Free Source:* **SEC EDGAR RSS Feeds** (completely free, real-time access provided by the US government).
* [cite_start]**Press Releases & Biotech Drivers:** * FDA approvals or clinical phase results (1, 2, 3) can boost biotech stocks by 300% in a single day[cite: 25, 26].
    * *Free Source:* **Yahoo Finance News RSS**, **BiopharmCatalyst** (free calendar tier), or **Google News RSS** filtered by stock tickers.

### C. Alternative Data (Retail Sentiment & Hype)
* [cite_start]**Social Platforms:** Reddit (`r/pennystocks`, `r/wallstreetbets`), X (Twitter), and StockTwits[cite: 29, 30]. [cite_start]Spikes in ticker mentions with a `$` sign (e.g., `$CELZ`) often precede price action[cite: 31].
    * *Free Source:* **Reddit API (PRAW)** (free tier available), **StockTwits Public API** (scraping public streams), and **Twikit** (for scraping X without a paid API).
* [cite_start]**Google Trends:** Sharp increases in search queries for a company name[cite: 32].
    * *Free Source:* **PyTrends** library (unofficial Google Trends API for Python).

---

## 2. Pipeline Execution Steps

[Free Data Ingestion] ──> [Filter Stage] ──> [Momentum Stage] ──> [Catalyst & Sentiment Check] ──> [Alert]

yfinance (Market)       - Price < $5      - Pre-market RVOL > 3   - SEC EDGAR / Yahoo RSS     - Discord / Telegram

SEC RSS (Filings)       - Float < 25M     - Gap > 3%              - Social Media Sentiment

Reddit/Praw (Social)    - Vol > 500k

1.  **Step 1 (Hard Filtering):** Price < $5, US market, Float < 25M, Daily Volume > 500k shares (to avoid absolute illiquidity)[cite: 39].
2.  **Step 2 (Momentum Search):** Every 5 minutes during pre-market, calculate: `current_volume / 5_day_average_volume`[cite: 40]. If the ratio is > 3, pass the ticker to the next stage[cite: 41].
3.  **Step 3 (Catalyst & Sentiment Check):** Run a free NLP model (e.g., a local HuggingFace sentiment model like `FinBERT`) over news/filings from the last 24 hours[cite: 42]. If a positive catalyst (FDA, new contract) or a major social sentiment spike is detected, the ticker becomes Priority #1[cite: 43].

---

## 3. Backtesting, Historical Data Collection & Optimization

Since high-quality historical intraday data for penny stocks is expensive, the pipeline must implement a **Forward Testing & Local History Collection** model.

### A. Phase 1: 3-to-6-Month Data Accumulation (The Sandbox)
Because free APIs do not provide granular historical pre-market/order-book data for low-cap stocks, you must build your own database from day one.

* **Continuous Logging:** Run the pipeline daily in "shadow mode". Every day, capture and save the following into a local SQLite/PostgreSQL database:
    * Every ticker that passes the initial Step 1 filter[cite: 39].
    * Its exact pre-market volume, RVOL, and gap size at 9:15 AM EST[cite: 13, 14, 15, 40].
    * Social media mention metrics at that exact moment[cite: 29, 31].
    * The end-of-day (EOD) maximum price reached, open price, and close price.

### B. Phase 2: Backtesting Rules
Once 3–6 months of data are collected, simulate trading strategies on this dataset using these strict rules:

1.  **The "No-Fill" Realism Rule:** Assume you can never buy at the absolute lowest pre-market price. Simulate entries at the Market Open price or 1-2% above the trigger price to account for slippage.
2.  **Dilution & Pump-and-Dump Filtering:** Eliminate any historical trades where an SEC filing containing keywords like *“At-the-market offering”*, *“Warrants”*, or *“Prospectus Supplement”* was released within 48 hours[cite: 21, 22].
3.  **Strict Risk-Management Simulation:** Penny stocks fall as fast as they rise[cite: 45]. Test two specific exit strategies:
    * *Trailing Stop:* Trailing the price by 5–10%.
    * *Time-based/Hard Target:* Selling 50% of the position at a +20% gain, and setting a hard stop-loss if the price drops below the pre-market support level[cite: 45].

### C. Parameter Optimization
Using the logged data, run optimization loops (e.g., Grid Search) to fine-tune your variables:
* **Optimize RVOL Threshold:** Test if an RVOL of > 3x yields better results than > 5x[cite: 14, 41]. (Often, > 5x means you are already too late).
* **Optimize Float Caps:** Compare the success rate of stocks with a < 10M float versus a 10M–25M float[cite: 11, 39].
* **Sentiment Correlation:** Find the mathematical correlation between the number of Reddit/X mentions and the maximum intraday percentage move to determine the optimal "hype threshold"[cite: 29, 31].