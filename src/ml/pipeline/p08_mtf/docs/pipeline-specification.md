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

## 5. Workflow Execution & Testing Periods
P08 uses a multi-stage validation process to ensure strategy robustness:

### Data Partitioning
1.  **Training Set (IS + OOS)**: Formed by the first $N-1$ files in a batch (sorted by date).
    - **In-Sample (IS - 70%)**: Used to fit the XGBoost model.
    - **Out-of-Sample (OOS - 30%)**: Used to calculate the "Adjusted Sharpe" used for Optuna optimization.
2.  **Validation Set (Holdout)**: The final file in the batch (e.g., the most recent year). This data is never seen during optimization.

### Execution Flow
1.  **Preparation**: Ensure both Execution and Anchor CSV files are in the `data/` directory.
2.  **Dataset Construction**: The `P08DataLoader` automatically merges the appropriate Anchor file.
3.  **Optimization**: Optuna searches for parameters that maximize performance on the OOS segment.
4.  **Automatic Post-Optimization Suite**:
    - **Step 1: Selection**: `select_candidates.py` identifies top unique ticker/tf pairs from aggregated results.
    - **Step 2: Robustness**: `run_robustness_checks.py` runs MC and WFA for all selected candidates.
    - **Step 3: Generalization**: `run_generalization_test.py` validates candidates against all other market regimes/tickers.
5.  **Winner Analysis**: `run_final_winners.py` identifies the single most robust strategy for high-resolution validation.

---

## 6. Real-World Simulation Best Practices

To bridge the gap between backtest and production, the P08 pipeline recommends:

| Parameter | Backtest | Recommended Simulation | Rationale |
|-----------|----------|------------------------|-----------|
| **Slippage** | 0.05% | **0.1% - 0.2%** | Accounts for thin order books and high volatility fills. |
| **Fees** | 0.1% | **0.075% - 0.1%** | Reflects Binance VIP/BNB discounts. |
| **Latency**| 0ms | **50ms - 200ms** | Simulates signal-to-order execution delay. |

---

## 7. Input Parameters (`pipeline.py`)

When running the pipeline via CLI (`python src/ml/pipeline/p08_mtf/pipeline.py`), the following parameters are available:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--ticker` | String | None | Specific ticker to run (e.g., `ETHUSDT`). If omitted, processes all matching files. |
| `--tf`     | String | None | Specific execution timeframe (e.g., `30m`). |
| `--years`  | String | None | Comma-separated list of years to include in the training batch. |
| `db_url`   | String (Init) | `test_optuna.db` | SQLite URL for Optuna study storage. |
| `result_root` | Path (Init) | `results/p08_mtf` | Directory where artifacts and logs are stored. |

---

## 8. Result Interpretation

*   **Trend Alignment**: Robust strategies should show higher win rates when the execution signal aligns with the `anchor_trend`.
*   **Regime Sensitivity**: The model's performance should be analyzed across different `anchor_regime` states to identify where its edge is strongest.
*   **Generalization Pass Rate**: A robust strategy should have a >60% PASS rate in cross-ticker/cross-timeframe tests.
