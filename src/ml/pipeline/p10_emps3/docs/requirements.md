Вот обновленная спецификация. Я убрал все упоминания кроссплатформенности и iOS, сосредоточившись исключительно на технической реализации логики поиска «сжатой пружины» внутри вашего текущего стека.

---

```markdown
# Technical Specification: EMPS2-Precursor Pipeline (Accumulation Phase Detection)

## 1. Objective
Implement a specialized sub-pipeline for the EMPS2 system to identify stocks in a "coiled spring" state. The goal is to detect institutional accumulation *before* the price breakout occurs, focusing on volume/price divergence.

## 2. Core Strategy (The "Spring" Formula)
The pipeline must identify tickers where liquidity is high but volatility is low. 
**Required Criteria:**
1.  **Volume Presence:** $Volume\ Z\text{-}Score > 1.5$ (Elevated interest).
2.  **Price Compression:** $Price\ Range\ (1d) < 2.5\%$ AND $ATR\ Ratio < 2\%$.
3.  **High Absorption Ratio:** $Vol/RV\ Ratio > 2.0$ (High volume absorbed without significant price movement).

## 3. Pipeline Stages & Filters

### Stage A: Universe & Fundamentals
* **Source:** NASDAQ Complete (~8,000 tickers).
* **Filters:**
    * Market Cap: $50M - $5B (Mid/Small cap focus).
    * Float: < 60M shares (Scarcity factor).
    * Price: > $1.00.

### Stage B: Dark Pool (TRF) Integration
* **Data:** Utilize `trf_correction_factor` from FINRA TRF API.
* **Accumulation Signal:** Flag tickers where `trf_correction_factor` > 1.6 and the net price change is neutral (between -1% and +1%). This indicates massive off-exchange buying.

### Stage C: Pre-Breakout Filter Logic
Implement a `PreBreakoutFilter` class to ensure we are not catching "already risen" stocks:
* **Exclusion Rule 1:** Discard if $Price\ Change\ (Current\ vs\ T-1) > 3.5\%$.
* **Exclusion Rule 2:** Discard if $Current\ Price$ is more than $10\%$ away from the $20$-day SMA (avoids overextended stocks).
* **Inclusion Rule:** Price must be within $3\%$ of the $52$-week High (identifies stocks "pressing" against resistance).

### Stage D: Rolling Memory Enhancement
* **Phase 1.5 (Early Warning):** Identify tickers appearing 3+ times in the last 5 days where:
    * $ATR$ is trending *downwards*.
    * $Volume\ Z\text{-}Score$ is trending *upwards*.

## 4. Technical Implementation & Outputs

### Integration
* Core logic must be integrated into the existing `src/ml/pipeline/p06_emps2/` structure.
* Use existing `DataDownloader` and `SentimentFilter` modules.

### Output Files
* `07_prebreakout_watchlist.csv`: High-priority list of stocks ready to move.
* `08_absorption_diagnostics.csv`: Log of $Vol/RV$ and $TRF$ divergence for manual review.

## 5. Execution Parameters (Config)
```json
{
  "mode": "precursor",
  "min_vol_zscore": 1.5,
  "max_price_impact": 0.025,
  "min_vol_rv_ratio": 2.0,
  "lookback_days": 10,
  "require_dark_pool_surge": true,
  "max_distance_from_resistance": 0.03
}

```

## 6. Definition of Done

1. Pipeline reduces the universe to <15 high-conviction candidates per scan.
2. All candidates in the final output show a price change of < 3% for the target day.
3. Automated alerts are triggered for Phase 1.5/Pre-Breakout candidates via Telegram.

```

---
Here is the detailed algorithmic specification for the developer agent, written in professional technical English. You can append this directly to your `.md` file.

---

### Appendix A: Detailed Accumulation Detection Algorithm

The developer must implement the following logic within a new class: `AccumulationAnalyzer`. This module identifies the "Coiled Spring" effect where high-volume buying is hidden by low price volatility.

#### 1. Absorption Ratio Calculation ($AR$)

The goal is to find anomalous volume that fails to move the price, signifying institutional absorption.

1. **Volume Z-Score ($V_z$):** Calculate the standard score of the current total volume (including TRF correction) against a 20-day moving average.

$$V_z = \frac{Vol_{current} - \mu(Vol_{20})}{\sigma(Vol_{20})}$$


2. **Realized Volatility ($RV$):** Calculate the annualized standard deviation of log returns using 15-minute or 1-hour intervals over the last 5 sessions.
3. **Final Absorption Ratio ($AR$):**

$$AR = \frac{V_z}{RV}$$


* **Threshold:** $AR > 2.0$ indicates that volume intensity is double the "fair" price volatility. The stock is absorbing supply without price appreciation.



#### 2. Price Compression Logic ("The Squeeze")

To ensure the ticker is ready for an imminent breakout, the agent must validate three compression states:

* **Condition 1 (Inside Day):** Current $High < Yesterday's High$ AND $Current Low > Yesterday's Low$.
* **Condition 2 (BB Squeeze):** Bollinger Band Width ($Upper - Lower$) must be at a 12-month relative minimum.
* **Condition 3 (V/R Divergence):** 
$$\frac{Volume_{current}}{\text{AvgVolume}_{20}} > 1.5 \quad \text{AND} \quad \frac{Range_{current}}{\text{AvgRange}_{20}} < 0.8$$



*(This confirms volume is expanding while the daily trading range is shrinking.)*

#### 3. Dark Pool (TRF) Surge Validation

Use the `trf_correction_factor` as the primary confirmation tool for the Pre-Breakout phase:

* **Signal:** If the `trf_correction_factor` increases by $>20\%$ over a 3-day rolling window while the price remains within a $\pm 1.5\%$ band, assign the **Ultra-High Priority** flag.
* **Rationale:** This identifies accelerated off-exchange accumulation immediately preceding a public move.

#### 4. Pre-Breakout Scoring System

The agent must generate a `prebreakout_score` (0-100) based on the following weights:

* **+30 Points:** Absorption Ratio ($AR$) > 2.5.
* **+20 Points:** "Squeeze" state (Bollinger Band Width < 5%).
* **+30 Points:** Resistance Pressing (Price within 2% of 20-day High).
* **+20 Points:** Social Virality Index > 1.2 (Early crowd awareness).

**Final Output Trigger:** Any ticker with a `prebreakout_score > 70` must be immediately dispatched to the `07_prebreakout_watchlist.csv`.

---