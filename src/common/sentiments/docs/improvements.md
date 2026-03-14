This Technical Requirements Document (TRD) is designed for an AI agent to build a sentiment analysis tool. It prioritizes **cross-platform readiness** (clean separation of logic and UI) to ensure easy future porting between environments.

---

# Technical Requirements Document: Multi-Source Sentiment Aggregator

## 1. Project Overview

**Objective:** Develop a robust backend module (Agent-ready) to fetch and normalize sentiment data for specific stock tickers from multiple free/freemium sources.
**Design Philosophy:** Decoupled architecture. The core logic must be independent of the delivery platform (CLI, Web, or Mobile/iOS).

---

## 2. Data Source Specifications

### A. Alpha Vantage (News & Sentiment API)

* **Type:** AI-driven News Sentiment.
* **Endpoint:** `function=NEWS_SENTIMENT&tickers=[TICKER]`
* **Free Tier Limits (2026):** 25 requests per day / 5 requests per minute.
* **Data Points:** `overall_sentiment_score` (-1.0 to 1.0), `overall_sentiment_label`.
* **Implementation Note:** Must handle "Rate Limit Reached" gracefully without crashing the agent.

### B. Finnhub.io (Social & News Sentiment)

* **Type:** Social Media (Reddit) + News Sentiment.
* **Endpoint:** `/api/v1/news-sentiment?symbol=[TICKER]` and `/api/v1/stock/social-sentiment?symbol=[TICKER]`
* **Free Tier Limits:** 30–60 requests per minute (varies by API key).
* **Data Points:** `buzz`, `sentiment` (bullish/bearish percentages), and Reddit mention volume.

### C. StockTwits (Social Pulse)

* **Type:** Direct Retail Sentiment.
* **Source:** Public Ticker Stream or RapidAPI StockTwits wrapper.
* **Free Tier:** Free public access to basic ticker streams.
* **Logic:** The agent must parse the `sentiment` field from the latest 30 messages in the stream and calculate a weighted average.

### D. Finviz (Web Scraping - Fallback)

* **Type:** Visual News Mood.
* **Method:** Scrape news table for `[TICKER]` and detect CSS classes for color-coded sentiment (Green = Positive, Red = Negative).
* **Constraint:** Use a standard User-Agent to avoid immediate blocks.

---

## 3. Architecture & Cross-Platform Readiness

To ensure the code is **"iOS/Cross-platform ready"** (as per project notes), the agent must follow these rules:

1. **Strict Interface Separation:** All sentiment fetchers must inherit from a base `SentimentProvider` class.
2. **No Platform-Specific Dependencies:** Avoid libraries like `win32gui` or specialized Linux shells. Use `aiohttp` or `requests` for networking.
3. **Data Normalization:** Every source must return a standardized JSON object:
```json
{
  "ticker": "AAPL",
  "source": "AlphaVantage",
  "score": 0.45,  // Scale -1 to 1
  "label": "Bullish",
  "timestamp": "2026-03-13T..."
}

```



---

## 4. Agent Task Instructions

**The Agent must:**

1. Create a `SentimentEngine` class that initializes all available providers.
2. Implement a `fetch_all(ticker)` method that runs providers in parallel (using `asyncio`).
3. If a provider fails (rate limit/API key error), skip it and return data from successful sources.
4. Calculate a **Composite Sentiment Index** ($CSI$) using the formula:

$$CSI = \frac{\sum (Score_i \times Weight_i)}{\sum Weight_i}$$



*(Default weights: News 40%, Social 60%)*

---

## 5. Deployment & Testing

* **Environment:** Python 3.10+.
* **Output Format:** Clean Markdown table and a raw JSON object for the calling application.

---
