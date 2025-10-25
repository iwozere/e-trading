# Short Squeeze Detection Pipeline — Final Requirements

## 1. Overview

The Short Squeeze Detection Pipeline identifies publicly traded companies with a high probability of an upcoming short squeeze event. It operates using free-of-charge data providers (FMP, Finnhub) and optionally others later. The system follows a **hybrid scheduling design**:

* **Weekly Screener** — broad structural scan for high short-interest candidates.
* **Daily Deep Scan** — focused analysis on previously found candidates and ad-hoc signals.

The system stores data persistently, computes dynamic risk metrics, and triggers alerts when squeeze conditions are met.

---

## 2. Data Providers

* **Primary:**

  * [FMP](https://financialmodelingprep.com/) — fundamentals, short interest, float, volume, market cap.
  * [Finnhub](https://finnhub.io/) — borrow rates, sentiment, news, option data (where free tier allows).
* **Optional (future):** Binance, Alpaca, yfinance, Reddit, Stocktwits, etc.

---

## 3. Scheduling and Frequency

| Module                | Frequency  | Example Time (Europe/Zurich) | Purpose                                                       |
| --------------------- | ---------- | ---------------------------- | ------------------------------------------------------------- |
| **Screener**          | Weekly     | Monday 08:00                 | Full-universe structural scan (SI, float, etc.)               |
| **Deep Scan**         | Daily      | 10:00                        | Focused scan on candidates from last Screener and ad-hoc list |
| **Ad-hoc candidates** | Continuous | As needed                    | Add tickers showing rapid sentiment/volume spikes             |

---

## 4. Pipeline Architecture

### 4.1 Modules

1. **Universe Loader** — fetches and filters universe from FMP.
2. **Screener** — computes and filters based on structural metrics (short interest, float, etc.).
3. **Candidate Store** — saves weekly screener snapshot and ad-hoc candidates.
4. **Deep Scan** — recalculates dynamic metrics (volume, sentiment, options) daily.
5. **Scoring Engine** — computes `squeeze_score` based on structural and transient metrics.
6. **Alerting Engine** — triggers alerts based on scoring thresholds and cooldown logic.
7. **Reporting** — generates HTML/CSV weekly summaries and daily highlights.
8. **Scheduler** — orchestrates runs (cron or APScheduler) according to YAML config.

---

## 5. Data Storage and Model

### 5.1 Tables

* **`ss_snapshot`** — append-only table with weekly scan results:

  * ticker, run_date, short_interest_pct, days_to_cover, float_shares, avg_volume_14d, market_cap, screener_score, raw_payload, data_quality
* **`ss_deep_metrics`** — daily computed metrics:

  * ticker, date, volume_spike, call_put_ratio, sentiment_24h, squeeze_score, alert_level
* **`ss_alerts`** — fired alerts with metadata:

  * ticker, alert_level, reason, timestamp, sent (bool), cooldown_expires
* **`ss_ad_hoc_candidates`** — transient table for dynamically added tickers:

  * ticker, reason, first_seen, expires_at, active (bool)

---

## 6. Configuration

All parameters are stored in `config.yml` or `config.json`.

### 6.1 Example YAML

```yaml
scheduling:
  screener:
    frequency: weekly
    day: monday
    time: '08:00'
  deep_scan:
    frequency: daily
    time: '10:00'
  timezone: Europe/Zurich

screener:
  si_percent_cutoff: 0.15
  days_to_cover_cutoff: 5
  avg_volume_14d_min: 200000
  float_max: 100_000_000
  top_k_for_deep_scan: 50

deep_scan:
  batch_size: 10
  ad_hoc_candidate_ttl_days: 7
  high_confidence_alert:
    volume_spike: 4
    sentiment_24h: 0.6
    min_screener_SI: 0.25
  medium_alert:
    volume_spike: 3
    sentiment_24h: 0.5
    min_screener_SI: 0.20

alerting:
  cooldown_days: 7
  channels:
    telegram: true
    email: true
```

---

## 7. Thresholds (default values)

| Metric           | Description              | Threshold                            |
| ---------------- | ------------------------ | ------------------------------------ |
| Short Interest % | Portion of float shorted | ≥ 15% (candidate), ≥ 25% (high risk) |
| Days to Cover    | SI / Avg Volume          | ≥ 5 days                             |
| Float Shares     | Liquidity                | ≤ 100M                               |
| Volume Spike     | Today’s volume / 14d avg | ≥ 3x (alert), ≥ 4x (high)            |
| Sentiment (24h)  | Positive ratio           | ≥ 0.5 medium, ≥ 0.6 high             |
| Call/Put Ratio   | Option bullishness       | ≥ 1.5                                |

---

## 8. Workflow

### 8.1 Weekly Screener

1. Load universe from FMP.
2. Fetch SI%, float, avg volume, market cap.
3. Filter using thresholds.
4. Save snapshot to DB (append-only).
5. Select top-K tickers for deep scan.

### 8.2 Daily Deep Scan

1. Load latest screener snapshot + active ad-hoc candidates.
2. Fetch updated metrics (volume, sentiment, options, borrow fee).
3. Compute daily features:

   * volume_spike = today_vol / avg_vol_14d
   * call_put_ratio
   * sentiment_24h
   * squeeze_score = weighted average of normalized features
4. Evaluate alert rules.
5. Trigger alerts & record in DB.
6. Update daily report.

### 8.3 Ad-hoc Candidates

* Added automatically when transient signals (volume spike or sentiment surge) detected for non-candidate tickers.
* Each ad-hoc entry expires after `ttl_days` (default 7) unless promoted by next Screener.

---

## 9. Alerts

### 9.1 Conditions

| Alert Level | Criteria                                         |
| ----------- | ------------------------------------------------ |
| **High**    | SI ≥ 25%, volume_spike ≥ 4, sentiment ≥ 0.6      |
| **Medium**  | SI ≥ 20%, volume_spike ≥ 3, sentiment ≥ 0.5      |
| **Low**     | SI ≥ 15%, transient signal moderate but trending |

### 9.2 Cooldown

* Once triggered, the same ticker cannot re-alert until cooldown expires (default: 7 days), unless stronger alert appears.

### 9.3 Channels

* Telegram bot integration via Bot API.
* Optional email (future phase).

---

## 10. Reporting

* **Weekly Summary:** Top candidates by `screener_score` with SI and float details.
* **Daily Report:** Top squeeze scores with trend visualization (volume, sentiment, options).
* Formats: HTML, CSV, optional Plotly PNG charts.

---

## 11. Logging & Monitoring

* Centralized structured logs (`run_id`, `module`, `status`, `runtime_s`).
* Metrics emitted per run:

  * `run_duration_seconds`, `candidates_count`, `alerts_count`, `api_calls`, `errors`
* Alert on pipeline failure or prolonged runtime.

---

## 12. Non-Functional Requirements

| Area              | Requirement                                                   |
| ----------------- | ------------------------------------------------------------- |
| **Runtime**       | Weekly full run ≤ 3h; daily deep scan ≤ 30min                 |
| **Data Quality**  | ≥ 99% valid JSON payloads; ≥ 95% key fields non-null          |
| **Resilience**    | Retry/backoff on API rate limits; continue on partial failure |
| **Security**      | API keys via env vars; no secrets in logs                     |
| **Extensibility** | Pluggable providers via interface adapters                    |
| **Storage**       | SQLite or PostgreSQL supported                                |
| **Config-driven** | All thresholds and schedules defined in YAML/JSON             |

---

## 13. Future Extensions

* Integrate Reddit / Stocktwits for richer sentiment.
* Add Telegram dashboard or web UI (Streamlit / FastAPI).
* Enable backtesting and alert precision tracking.
* Include option-flow anomaly detection.
* Integrate advanced ML-based sentiment scoring.

---

## 14. Acceptance Criteria Summary

1. Weekly run completes with valid screener snapshot.
2. Daily deep scan runs automatically, updates DB, and triggers alerts.
3. Alerts respect cooldown and are logged.
4. Reports generated successfully and contain top results.
5. Logs show API usage, errors, and performance metrics.
6. Secrets stored securely; config changes reflected without code changes.

---

**This version is the final consolidated requirements specification** for the Short Squeeze Detection Pipeline, including hybrid scheduling (weekly + daily), persistence model, and alerting rules.
