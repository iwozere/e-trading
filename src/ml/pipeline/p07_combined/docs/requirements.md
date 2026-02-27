A unified pipeline that uses vectorbt + Optuna for fast research/optimization, and Backtrader only for “short‑listed” strategies and execution, will cover both crypto/stocks and all your timeframes with one architecture. Below is a concrete approach, then a .md spec you can hand to a dev agent. [vectorbt](https://vectorbt.dev)

***

## 1. High‑level approach

- Use a **single feature + model API** that works for any (ticker, timeframe, market).  
- Train **separate models per (ticker, timeframe)**, but reuse the same pipeline code.  
- Research/optimize in **vectorbt + Optuna** (fast, vectorized), then export best params/models. [quantnomad](https://quantnomad.com/running-grid-optimization-for-backtests-in-python-using-vectorbt/)
- Validate and forward‑test in **Backtrader** for realistic fills, costs, slippage. [greyhoundanalytics](https://greyhoundanalytics.com/blog/vectorbt-vs-backtrader/)
- Inputs: CSV OHLCV for crypto and stocks (unified format).  
- Outputs: BUY/SELL/HOLD signals + performance metrics per model and per market/timeframe.

***

## 2. Core design decisions

- **Data model:**  
  - Standardized OHLCV CSV schema: `timestamp, open, high, low, close, volume`.  
  - Convert to timezone‑aware pandas Series/DataFrames; resample if needed.  
- **Features:**  
  - Price‑based: returns over multiple lags, rolling volatility, volume ratios.  
  - Technical: a small set (RSI, ATR, Bollinger Bands, maybe Ichimoku state flags) computed via TA‑Lib or custom. [vectorbt](https://vectorbt.dev)
- **Model type:**  
  - First iteration: tree‑based (XGBoost/LightGBM) classifier for 3‑class (BUY/SELL/HOLD) using tabular features (easier and fast).  
  - Later: optional plug‑in for sequence models (LSTM/CNN) behind same interface.  
- **Targets:**  
  - Label each bar as BUY / SELL / HOLD by future N‑bar return with thresholds, consistent across assets/timeframes.  
- **Optimization:**  
  - Use Optuna to tune: indicator params, lookback window lengths, model hyperparams, maybe thresholds. [youtube](https://www.youtube.com/watch?v=osoDo8r956Y)
- **Backtesting:**  
  - vectorbt: large grid of params / assets / timeframes (fast, vectorized). [pineify](https://pineify.app/resources/blog/backtrader-vs-vectorbt-vs-pineify-python-trading-framework-comparison-guide)
  - Backtrader: final candidates only, with commission, slippage, partial fills, etc. [autotradelab](https://autotradelab.com/blog/backtrader-vs-nautilusttrader-vs-vectorbt-vs-zipline-reloaded)

***

## 3. Development spec for the agent (.md)

```markdown
# Trading ML Pipeline – Development Requirements

## 1. Scope

Build a unified Python pipeline that:

- Ingests OHLCV CSV data for both crypto and stocks.
- Trains and evaluates ML models **per (ticker, timeframe)**.
- Uses **vectorbt + Optuna** for fast research and hyperparameter optimization.
- Uses **Backtrader** for realistic backtests of selected strategy configurations.
- Produces actionable **BUY / SELL / HOLD** signals and performance reports.
- Is easily extensible to new models, features, and markets.

Target timeframes:

- Crypto: 5m, 15m, 30m, 1h, 4h
- Stocks: 15m, 1h, 4h, 1d

## 2. Tech Stack

- Python 3.11+ (virtualenv/poetry)
- Core libs:
  - `pandas`, `numpy`
  - `vectorbt` (community) for vectorized backtests
  - `backtrader` for event-driven realistic backtests
  - `optuna` for hyperparameter optimization
  - `ta-lib` or a TA library for indicators
  - `xgboost` or `lightgbm` for tabular ML
  - `scikit-learn` for preprocessing/metrics
- Project structure: standard package layout (`src/` with modules).

## 3. Data Layer

### 3.1. CSV Format

- Input CSV schema (both crypto and stocks):

  ```text
  timestamp,open,high,low,close,volume
  ```

- Requirements:
  - `timestamp` in ISO 8601 (UTC); handle parsing robustly.
  - No missing OHLCV rows within trading hours after resampling.
  - Use separate directories, e.g.:
    - `data/crypto/{symbol}_{timeframe}.csv`
    - `data/stocks/{symbol}_{timeframe}.csv`

### 3.2. Loader API

Implement a module `data_loader.py` exposing:

```python
def load_ohlcv(
    symbol: str,
    timeframe: str,
    market: str  # "crypto" or "stock"
) -> pd.DataFrame:
    """Return OHLCV with DatetimeIndex (UTC), sorted, no duplicates."""
```

- Automatically infers file path from (symbol, timeframe, market).
- Performs:
  - Parsing of timestamps to tz-aware index.
  - Sorting, de-duplication.
  - Optional resampling to exact timeframe grid.
  - Basic QA checks (no huge gaps, no negative prices).

## 4. Feature Engineering

Implement `features.py` with a **single feature interface** that works for all markets/timeframes.

### 4.1. Core Features

Given an OHLCV DataFrame:

- Price/return features:
  - Log/percentage returns for 1, 2, 4, 8, 16 bars.
  - Rolling volatility over configurable windows (e.g. 14, 28 bars).
  - High-low range, body size, volume z-scores.

- Technical indicators (via TA-Lib or equivalent):
  - RSI with configurable period.
  - ATR with configurable period.
  - Bollinger Bands (period, stddev).
  - (Optional) Ichimoku – encode as categorical flags:
    - price above/below cloud, Tenkan >/< Kijun, etc.

### 4.2. Feature Function

```python
def build_features(
    ohlcv: pd.DataFrame,
    config: dict
) -> pd.DataFrame:
    """Return feature matrix X indexed by timestamp.

    config includes:
      - indicator parameters (e.g. rsi_period, bb_period, bb_mult, atr_period)
      - lookback windows for returns/volatility
    """
```

- Ensure any NaNs due to indicator warm-up are dropped consistently later.

## 5. Labeling / Target Definition

Implement `labeling.py`:

- Goal: 3-class classification `y ∈ {BUY, SELL, HOLD}` for each bar using the **Triple Barrier Method**.

### 5.1. Triple Barrier Labeling

Configurable parameters per trial:

- `pt_mult` – Profit Take multiplier of rolling ATR.
- `sl_mult` – Stop Loss multiplier of rolling ATR.
- `tpl_bars` – Time Path Limit (max bars to hold).

Algorithm:
1. For each bar, set upper barrier at `close + pt_mult * ATR` and lower at `close - sl_mult * ATR`.
2. Vertical barrier at `t + tpl_bars`.
3. Label:
   - **BUY (1)** if upper barrier hit first.
   - **SELL (-1)** if lower barrier hit first.
   - **HOLD (0)** if vertical barrier (time) hit first.

Function:

```python
def make_triple_barrier_labels(
    ohlcv: pd.DataFrame,
    pt_mult: float,
    sl_mult: float,
    tpl_bars: int
) -> pd.Series:
    """Return integer labels {1, -1, 0} based on the first barrier touched."""
```

- Ensure alignment with features (drop rows that have NaN in X or y).

## 6. Model Layer

First iteration: gradient boosting classifier (XGBoost or LightGBM).

Implement `models.py`:

```python
class TabularSignalModel:
    def __init__(self, model_params: dict):
        ...

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        ...

    def predict_proba(self, X) -> np.ndarray:
        """Returns class probabilities for [-1, 0, 1] in a fixed order."""

    def predict_signal(self, X, config: dict) -> pd.Series:
        """Maps probabilities to discrete BUY/SELL/HOLD using thresholds and hysteresis."""
```

- Config options for `predict_signal`:
  - `buy_prob_min`, `sell_prob_min` (min prob for BUY/SELL).
  - Optional hysteresis / min time between signal flips.
- **Regime Feature**: Must include a `global_regime` feature derived from an external HMM (VIX/SPY for stocks, BTC/TotalMCap for crypto).
- Model params and thresholds must be Optuna-optimizable.

## 7. Backtesting Layer – vectorbt

Implement `research_vectorbt.py`:

### 7.1. Signal Generation

- Given OHLCV and predicted signals Series, map to `entries` / `exits` for vectorbt:

  ```python
  # pseudo
  entries = signals == 1
  exits = signals == -1
  ```

- Use `vectorbt.Portfolio.from_signals` (or equivalent) to backtest:

  - Configurable:
    - Initial capital.
    - Fees, slippage (simple model).
    - Position sizing (fixed fraction, 1x leverage, etc.).

### 7.2. Metrics

Compute and store for each trial:

- CAGR, total return.
- Sharpe ratio, Sortino.
- Max drawdown.
- Win rate, average trade return.
- Turnover / number of trades.

## 8. Hyperparameter Optimization – Optuna

Implement `optimize.py` with:

```python
def objective(trial, symbol, timeframe, market, config_global) -> float:
    """Single Optuna objective for one (symbol, timeframe, market)."""
```

Search space:

- Indicator parameters: e.g.
  - `rsi_period` ∈ [5, 30]
  - `bb_period` ∈ [10, 40]
  - `bb_mult` ∈ [1.0, 3.0]
  - `atr_period` ∈ [7, 30]

- Labeling:
  - `horizon_bars` ∈ [4, 24] (depending on timeframe).
  - `up_threshold`, `down_threshold`.

- Model hyperparams:
  - Booster depth, learning rate, n_estimators, subsample, etc.
  - `tree_method='hist'` (CPU-optimized) or `'gpu_hist'` (GPU-enabled).

- Signal thresholds:
  - `buy_prob_min`, `sell_prob_min`.

Objective:
- Optimize a composite metric, e.g.:
  ```text
  objective = Sharpe_ratio - penalty_for_large_drawdown - penalty_for_low_trades
  ```
- Use `n_jobs=-1` to maximize CPU utilization on local/Pi environments.

- Run Optuna with:
  - Pruners for early stopping of bad trials.
  - Storage (SQLite) to resume studies.

CLI/entry function to run optimization for:

- One (symbol, timeframe, market).
- A batch of tickers/timeframes in parallel.

## 9. Backtesting Layer – Backtrader

Implement `backtesting_bt.py`:

- Strategy class that:

  - Loads OHLCV from CSV (or uses in-memory DataFrame).
  - Uses pre-trained model and feature config:
    - At each bar:
      - Build feature vector from recent OHLCV.
      - Call model’s `predict_signal`.
      - Issue orders (BUY/SELL/HOLD) accordingly.

- Add realistic settings:
  - Commission per trade (stock vs crypto).
  - Slippage model: Simulate **100-500ms execution latency** between signal and fill.
  - Cash and position sizing (fixed fraction, max exposure).

- Live Readiness:
  - Use Backtrader `Store` abstractions (e.g., BinanceStore, IBStore) to ensure the strategy code is identical for backtest and live execution.
  - For stocks, ensure the use of **Trade-Adjusted Data** (splits/dividends).

## 10. Experiment Management & Persistence

Implement:

- `config/` folder with YAML/JSON per study:
  - Global defaults (fees, capital).
  - Per-market overrides (crypto vs stocks).
- `artifacts/` folder for:
  - Best Optuna trials (JSON of params).
  - Trained model binaries (`.json` for XGBoost, etc.).
  - Backtest results (CSV metrics, equity curves).

### 10.1. Reproducibility

- Fix random seeds where possible.
- Store:
  - Git commit hash.
  - Library versions.
  - Study name and datetime.

## 11. Modular API for (ticker, timeframe) runs

High-level function in `pipeline.py`:

```python
def run_experiment(
    symbol: str,
    timeframe: str,
    market: str,
    n_trials: int,
    mode: str  # "optimize", "backtest_best"
):
    """Orchestrates data loading, optimization, and backtesting."""
```

Behavior:

- `mode="optimize"`:
  - Load data.
  - Run Optuna study with vectorbt backtests.
  - Save best params + metrics.

- `mode="backtest_best"`:
  - Load best params/model.
  - Run Backtrader backtest over in-sample or out-of-sample period.
  - Save equity curve and trade log.

## 12. Extensibility

- Design `TabularSignalModel` as a base class.
- Later add:
  - `SequenceSignalModel` (LSTM/CNN) that uses same `fit/predict_signal` interface.
- Allow plugging alternative labelers and features via config without code changes.

## 13. Deliverables

- Well-structured repo with:
  - `src/` modules described above.
  - Example Jupyter notebook demonstrating:
    - One crypto symbol (e.g., BTCUSDT, 15m).
    - One stock symbol (e.g., AAPL, 1h).
    - Full flow: load → features → labels → Optuna → vectorbt metrics → Backtrader backtest.
- Basic CLI scripts:
  - `python -m pipeline.run --symbol BTCUSDT --timeframe 15m --market crypto --mode optimize`
  - `python -m pipeline.run --symbol AAPL --timeframe 1h --market stock --mode backtest_best`

## 14. Visualization Suite
The pipeline must include a `visualizer.py` module using `matplotlib` or `plotly`.

### 14.1. Automated Plotting
Upon completion of `mode="backtest_best"`, generate:
- **Signal Integrity:** `predictions_scatter.png` (Prob vs. Outcome) and `feature_importance.png`.
- **Error Analysis:** `error_distribution.png` and `cumulative_error.png`.
- **Market Context:** `predictions_by_regime.png` using a simple ATR-based regime filter.
- **Equity Dashboard:** `backtest_summary.png` containing the equity curve, drawdown, and rolling Sharpe.

## 15. Export & Portability
- **Model Format:** Save best models as `.onnx` for cross-platform (iOS) compatibility.
- **Metadata:** Every artifact must include a `metadata.json` with the full feature config, scaling parameters, and versioning.

## 16. Raspberry Pi (ARM) Deployment
- **Architecture:** Target ARM64 (aarch64) for headless Ubuntu.
- **Model Export:** Export best XGBoost model to `.onnx` for deployment on the Pi using `onnxruntime-openvino` or standard `onnxruntime`.
- **Pre-calculation:** Perform all TA-Lib calculations using vectorized pandas-ta or talib wrappers to minimize per-bar processing time.
- **Robust Paths:** Use `pathlib` for all file operations to ensure Windows (\) and Linux (/) compatibility.

## 17. Data & Storage Requirements

### 17.1. Dynamic Input Parsing

The system must dynamically infer metadata from the filename to eliminate redundant configuration.

* **Filename Schema:** `{ticker}_{timeframe}_{start_date}_{end_date}.csv` (e.g., `XRPUSDT_5m_20200101_20260201.csv`).
* **Logic:** The `data_loader.py` must use regex to extract the **ticker** and **timeframe** automatically from the filename.

### 17.2. Result Persistence Path

To maintain a clean project structure, all outputs (models, logs, and plots) must be stored using the following hierarchy:

* **Base Path:** `results/p07_combine/{ticker}/{timeframe}/`
* **Contents:** * `model.onnx`: The trained inference engine.
* `metadata.json`: Feature scaling parameters, Optuna params, and TA-Lib configurations.
* `plots/`: All diagnostic images from the Visualization Chapter.



---

## 18. Robustness: Recovery & State Management

To handle potential crashes during long optimization runs on Windows 11, the main execution loop must implement a **Stateful Recovery Logic**.

* **Checkpointing:** Before processing a file, the script checks for the existence of the `model.onnx` or a `completed.flag` in the corresponding `results/p07_combine/{ticker}/{timeframe}/` directory.
* **Resumption:** If the output directory for a specific file already contains a successful result, the pipeline skips that file and moves to the next.
* **Optuna Persistence:** Use a local **SQLite** database (`optuna_study.db`) for the Optuna storage backend. This allows the optimization trials themselves to be resumed even if the Python process is killed.

---

## 19. Visualization Chapter (The Diagnostic Dashboard)

For every successful "Best Trial," the `visualizer.py` module will generate a suite of diagrams to reflect model quality and strategy robustness.

### 19.1. Model Diagnostics

* **predictions_scatter.png:** Plots predicted probability against actual forward returns to check signal correlation.
* **error_distribution.png:** A histogram of residuals to identify bias or extreme outliers.
* **error_over_time.png:** A time-series plot showing if the model's accuracy is drifting or decaying.
* **cumulative_error.png:** Helps identify specific market periods where the model systematically failed.

### 19.2. Performance & Regime Analysis

* **predictions_by_regime.png:** Accuracy metrics (Precision/Recall) segmented by market state (Bull, Bear, Sideways).
* **rolling_performance.png:** A plot of Rolling Sharpe Ratio and Rolling Max Drawdown to ensure consistency.
* **tbm_barrier_hits.png:** A bar chart showing the frequency of Take-Profit hits vs. Stop-Loss hits vs. Time-Outs.

### 19.3. The Master Overlay (`strategy_overlay.png`)

A comprehensive multi-pane visualization:

1. **Price Pane:** OHLCV candles with clear Entry (Green) and Exit (Red) marks.
2. **Probability Pane:** A heatmap showing the model's confidence over time.
3. **Equity Pane:** The strategy's equity curve compared against a Buy & Hold benchmark, with shaded drawdown areas.

## 20. Portability: Windows to Raspberry Pi

### 20.1. The ONNX Bridge

* **Training (Windows):** Save the XGBoost model as an ONNX graph. Ensure all feature scaling (e.g., Mean/StdDev from `RobustScaler`) is saved in `metadata.json`.
* **Execution (Pi):** Use `onnxruntime` on the headless Pi. The Pi script should only require `pandas`, `ta-lib`, and `onnxruntime`—avoiding the heavy `xgboost` or `scikit-learn` dependencies.

### 20.2. Path Management

Use the `pathlib` library exclusively for all file operations to ensure that Windows-style paths (`\`) on your dev machine don't break when the code is moved to the Ubuntu/Pi environment (`/`).

## 1. Proposed Logical Improvements

### A. Walk-Forward Validation (Crucial)

The current spec mentions Optuna and Backtrader but doesn't explicitly define a **Walk-Forward Analysis (WFA)** or **Anchored/Rolling Cross-Validation**.

* **Improvement:** Add a `validation_strategy` to the Model Layer. Instead of a single Train/Test split, implement a rolling window (e.g., train on 6 months, test on 1 month, slide forward). This prevents overfitting to specific market regimes.

### B. Feature Scaling & Stationarity

ML models (especially if you later add LSTMs) struggle with raw price data.

* **Improvement:** Ensure `features.py` includes a **Stationarity Check**. Use fractional differentiation or log-returns instead of raw prices. Add a `PreProcessor` class to handle scaling (e.g., `RobustScaler`) fitted *only* on training data to prevent data leakage.

### C. Cross-Platform Readiness (iOS/Mobile)

To ensure the code remains portable for potential future iOS integration (as noted in your requirements), the pipeline should maintain a strict separation between the **Inference Engine** and the **Research Framework**.

* **Improvement:** Add a requirement to export the final trained models to **ONNX** format. This allows the model to run natively on iOS (via CoreML) or other environments without needing the full Python ML stack.

### D. Regime-Aware Labeling

Global thresholds (e.g., 0.5%) are dangerous because 0.5% in a low-volatility period is "huge," while in a high-volatility period, it's "noise."

* **Improvement:** Use **Volatility-Adjusted Thresholds**. Set `up_threshold` and `down_threshold` as multiples of the rolling ATR (e.g., ) rather than fixed percentages.

## 2. Visualization Chapter (Model Quality & Performance)

For the "Best Trial," the system must generate a comprehensive diagnostic dashboard. These charts should be saved in `artifacts/{symbol}_{timeframe}/plots/`.

### 2.1. Model Diagnostics (ML Focus)

* **predictions_scatter.png:** Predicted probability vs. Actual forward return. Helps visualize if high-confidence signals actually correlate with larger moves.
* **error_distribution.png:** A histogram of residuals. Look for "fat tails" or bias (e.g., the model consistently misses big upside moves).
* **error_over_time.png:** A time-series of prediction errors. Useful for spotting **model drift** where accuracy degrades as market conditions change.
* **cumulative_error.png:** A running sum of errors to identify specific periods (regimes) where the model failed systematically.
* **feature_importance.png:** A bar chart showing which indicators (RSI, Vol, etc.) drove the model's decisions.

### 2.2. Strategy Performance (Quant Focus)

* **predictions_by_regime.png:** Accuracy metrics segmented by market state (e.g., Bull, Bear, Sideways).
> * **rolling_performance.png:** Rolling Sharpe ratio and Rolling Drawdown. A strategy that looks great over 3 years but has a "flat" 18-month period is a red flag.
> 
> 


* **trade_edge_plot.png:** A "Mean Reversion" or "Trend Following" check—plots the average price path  bars before and after a BUY signal.

### 2.3. The "Master" Chart: `strategy_overlay.png`

This is a high-resolution, multi-pane interactive plot (or large PNG):

1. **Top Pane:** OHLCV candles with **Green Triangles** (Buy) and **Red Triangles** (Sell).
2. **Mid Pane:** Model confidence (probabilities) as a heatmap or line.
3. **Bottom Pane:** Equity curve vs. Benchmark (Buy & Hold) with shaded drawdown regions.

