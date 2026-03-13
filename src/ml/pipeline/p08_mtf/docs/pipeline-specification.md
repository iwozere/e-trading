# P08 Multi-Timeframe (MTF) Pipeline Specification

## 1. Overview
The P08 MTF Pipeline extends the research framework by introducing **trend-aware execution**. It leverages higher-timeframe "Anchor" data to provide macro context for lower-timeframe "Execution" strategies.

---

## 2. Multi-Timeframe Architecture

### Anchor vs. Execution Mapping
| Execution TF | Anchor TF |
|--------------|-----------|
| 5m           | 1h        |
| 15m          | 4h        |
| 30m          | 4h        |
| 1h           | 1d        |
| 4h           | 1d        |

---

## 3. Automated Post-Optimization Suite

After the initial Optuna optimization and result aggregation, the pipeline automatically executes the following sequence:

### Step 1: Candidate Selection (`select_candidates.py`)
*   **Logic**: Filters aggregated results from all optimization runs to identify high-quality unique (ticker, timeframe) pairs.
*   **Filters**:
    *   Total Trades >= 15
    *   Profit Factor > 1.2
    *   Total Return [%] > 0
*   **Inputs**: `results/p08_mtf/p08_mtf_aggregated_results.csv`
*   **Outputs**: `results/p08_mtf/p08_robustness_candidates.csv`

### Step 2: Robustness Checks (`run_robustness_checks.py`)
*   **Logic**: Performs high-fidelity stress tests on selected candidates.
*   **Checks**:
    *   **Monte Carlo (Randomized OHLCV)**: Identifies sensitivity to specific price path variations.
    *   **Walk-Forward Analysis (WFA)**: Validates performance on rolling OOS segments.
    *   **Parameter Sensitivity**: Analyzes how small changes in parameters affect the Sharpe ratio.
*   **Inputs**: `p08_robustness_candidates.csv`, Optuna DB.
*   **Outputs**: Robustness artifacts (JSON/PNG) in each candidate's result directory.

### Step 3: Generalization Tests (`run_generalization_test.py`)
*   **Logic**: Tests a frozen model against all other available data segments and tickers to detect overfitting to a specific market period.
*   **Criteria**: Pass if Sharpe > 0.5 and Trades > 5 in cross-market tests.
*   **Inputs**: `p08_robustness_candidates.csv`, `data/*.csv`.
*   **Outputs**: `results/p08_mtf/p08_generalization_{ticker}_{tf}_segments.csv`.

### Step 4: Winner Analysis (`run_final_winners.py`)
*   **Logic**: Aggregates all generalization results to find the "Clear Winner" (highest avg Sharpe with consistent PASS rate).
*   **Inputs**: All generalization segment CSVs.
*   **Outputs**: High-resolution (500 trials) final validation profile for the winner.

### Step 5: Backtrader Simulation (`run_backtrader.py`)
*   **Logic**: Event-driven simulation on the full 2020-2025 dataset with realistic brokers.
*   **Realism Layer**:
    *   0.1% Commission, 0.1% Slippage.
    *   Next-Bar Open execution (signals at Bar T close are filled at Bar T+1 open).
*   **Inputs**: `p08_robustness_candidates.csv`, Full MTF Datasets.
*   **Outputs**: `results/p08_mtf/backtrader_winners_summary.csv`.

---

## 4. Input Parameters (`pipeline.py`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--ticker` | String | None | Specific ticker to run (e.g., `ETHUSDT`). |
| `--tf`     | String | None | Specific execution timeframe (e.g., `30m`). |
| `--years`  | String | None | Comma-separated list of training years (e.g., `2022,2023`). |

---

## 5. Implementation Notes
*   **Look-Ahead Protection**: Anchor data is shifted by 1 bar before joining via `pd.merge_asof`.
*   **Memory Efficiency**: Merged MTF datasets are cached in memory during batch runs to avoid redundant I/O.
*   **Environment**: All scripts must be executed within the `.venv` to ensure `vectorbt` and `xgboost` compatibility.
