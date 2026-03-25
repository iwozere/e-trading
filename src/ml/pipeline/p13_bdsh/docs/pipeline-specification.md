## 1. pipeline-specification.md

### Project Overview
A Python-based backtesting engine to evaluate a **VIX-Threshold Scaling Strategy**. The core logic uses the VIX Z-Score to trigger multi-stage entries ("Scaling In") and exits ("Scaling Out") on equity tickers over a 20-year horizon.

### Core Logic: The Scaling Mechanism
Instead of a binary Buy/Sell, the pipeline will manage a "Heat Map" of exposure:
* **Tier 1 Entry:** VIX Z-Score > 1.5 (Invest 33% of allocated capital)
* **Tier 2 Entry:** VIX Z-Score > 2.5 (Invest additional 33%)
* **Tier 3 Entry:** VIX Z-Score > 3.5 (Invest final 34%)
* **Exit Logic:** Liquidate all tiers when VIX Z-Score crosses below 0.0 (Mean Reversion).

### Pipeline Stages
1.  **Ingestion:** Batch download of `^VIX` and user-defined tickers via `yfinance`.
2.  **Feature Engineering:** * Compute 20-day and 50-day rolling VIX means/stds.
    * Calculate $Z = \frac{VIX_{t} - \mu_{rolling}}{\sigma_{rolling}}$.
3.  **Signal Generation:** Vectorized generation of "Target Exposure" (0.0, 0.33, 0.66, or 1.0).
4.  **Backtest Engine:** * Account for daily compounding.
    * Calculate **Maximum Drawdown (MDD)** and **Sharpe Ratio**.
5.  **Analytics:** Comparison of "Scaled VIX Strategy" vs. "Buy & Hold" per ticker.

### Mathematical Definitions
The primary signal is the Rolling Z-Score:
$$Z_t = \frac{VIX_t - \text{MA}(VIX, n)}{\text{StdDev}(VIX, n)}$$
Where $n$ is the lookback period (default 30 days).

