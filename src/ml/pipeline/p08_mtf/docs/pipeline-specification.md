# P08 Multi-Timeframe (MTF) Pipeline Specification

## 1. Overview
The P08 MTF Pipeline extends the capabilities of research framework by introducing **trend-aware execution**. It leverages higher-timeframe "Anchor" data to provide macro context for lower-timeframe "Execution" strategies. This approach aims to reduce noise and align trade signals with the dominant market direction.

---

## 2. Multi-Timeframe Architecture

### Anchor vs. Execution
- **Execution TF**: The primary timeframe where trade entries and exits occur (e.g., 5m, 15m, 1h).
- **Anchor TF**: A higher timeframe used to define the broader trend and regime (e.g., 1h for 5m, 1d for 1h).

### Mapping Table
| Execution TF | Anchor TF |
|--------------|-----------|
| 5m           | 1h        |
| 15m          | 4h        |
| 30m          | 4h        |
| 1h           | 1d        |
| 4h           | 1d        |

---

## 3. Data Integrity & Look-Ahead Protection

A critical challenge in MTF research is avoiding "Look-Ahead Bias" (using data that wouldn't have been known at the time of the trade). P08 implements two layers of protection:

1.  **Anchor Shifting**: The Anchor dataset is shifted by 1 bar (`shift(1)`) BEFORE joining. This ensures that a 15m bar at 10:15 only sees the Anchor bar that CLOSED at or before 10:00.
2.  **Point-in-Time Join**: Uses `pd.merge_asof` with `direction='backward'`. This matches each execution timestamp with the most recent *completed* anchor timestamp.

---

## 4. Feature Engineering (MTF Capabilities)

The `P08FeatureEngine` generates features that bridge the two timeframes:

*   **Anchor Trend**: Log-return of the Anchor's EMA to define the macro slope.
*   **Anchor Regime**: (Categorical) Bull/Bear/Neutral states based on Anchor trend strength.
*   **MTF Divergence**: Distance between Execution price and Anchor EMA.
*   **Anchor RSI**: Relative strength on the higher timeframe.
*   **Anchor BB Position**: Identifying if the macro trend is overextended.

---

## 5. Workflow Execution

1.  **Preparation**: Ensure both Execution and Anchor CSV files are available in the `data/` directory.
2.  **Dataset Construction**: The `P08DataLoader` automatically identifies, loads, and merges the appropriate Anchor file for any given Execution file.
3.  **Optimization**: Optuna searches for parameters that perform best *within the context of the higher timeframe trend*.
4.  **Validation**: A holdout year (OOS) is used to verify that the MTF patterns generalize.

---

## 6. Result Interpretation

*   **Trend Alignment**: Robust strategies should show higher win rates when the execution signal aligns with the `anchor_trend`.
*   **Regime Sensitivity**: The model's performance should be analyzed across different `anchor_regime` states to identify where its edge is strongest.
