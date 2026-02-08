# Requirements: e-trading Research & Production Pipeline

## 1. Project Folder Structure

* **Root:** `e-trading/`
* **Research:** `src/vectorbt/`
* **Production Logic:** Shared module for signal calculation to ensure consistency.

## 2. Technical Architecture

### 2.1 Dual-Database Strategy

* **Optimization DB (SQLite):** * Used by **Optuna** for trial persistence.
* Location: `src/vectorbt/db/optimization_study.db`.
* Requirement: Use **WAL (Write-Ahead Logging)** mode in SQLite to handle high-concurrency writes from multiple CPU cores (`n_jobs=-1`).


* **Production DB (PostgreSQL):**
* Stores the final "Golden Configs".
* Requirement: A `PromoteStrategy` script that reads the best trial from SQLite and writes it to the Postgres `active_configs` table.



### 2.2 Vectorbt Signal Factory

* **Decoupled Logic:** Create a `Signals` class that uses `vbt.IndicatorFactory.from_talib()`.
* **Broadcasting:** The factory must be able to accept `vbt.Param` lists for all hyperparameters (e.g., `rsi_window=[14, 21]`, `ema_long=[50, 100, 200]`).
* **Multi-Asset Support:** Logic must accept a multi-index DataFrame containing all 4 tickers (BTC, ETH, XRP, LTC) to calculate signals across the entire portfolio in one pass.

### 2.3 Optimization & WFO (Walk-Forward)

* **Engine:** Implement `vbt.Portfolio.from_signals` inside the Optuna `objective` function.
* **Walk-Forward Splits:** Use `vbt.ArrayWrapper.rolling_split` to create In-Sample (IS) and Out-of-Sample (OOS) windows.
* **Metric:** Maximize the **Calmar Ratio** (Total Return / Max Drawdown) to ensure the 50%+ APR goal doesn't come with account-destroying risks.

## 3. Implementation Tasks for Agent

### Task 1: Data Pre-processor

* Load Binance OHLCV from CSV files located in `data/` (2020-2025 data).
* Ensure a strict `DateTime` index.
* Merge all 4 tickers into a single wide DataFrame for vectorized operations.

### Task 2: Optuna Study Manager

* Configure the SQLite storage: `sqlite:///src/vectorbt/db/optimization_study.db`.
* Implement `n_jobs=-1` multi-processing.
* **Safety Check:** Add a memory monitor to prevent OOM on Raspberry Pi if optimization is ever run locally (though primarily targeted for the PC).

### Task 3: JSON & Trade Export

* Function to extract `pf.trades.records_readable` and convert it to a structured JSON.
* Save Plotly HTML reports to `src/vectorbt/reports/{strategy_id}.html`.

---

## 4. Discussion Point for the Agent

> **Logic Parity:** Since the live bot uses Backtrader and the researcher uses Vectorbt, the agent must ensure that `TA-Lib` is the underlying engine for both. Any custom "Numba-optimized" indicators in Vectorbt must be strictly validated against their Backtrader counterparts.

### Summary Table for Agent

| Component | Technology | Role |
| --- | --- | --- |
| **Backtest Engine** | Vectorbt | Fast matrix-based simulations |
| **Optimizer** | Optuna | Hyperparameter search |
| **Local Storage** | SQLite | High-speed trial logging |
| **Prod Storage** | PostgreSQL | Active bot configurations |
| **Signals** | TA-Lib | Consistent calculation engine |


Adding **Futures** and **Leverage optimization** significantly changes the math. You are no longer just measuring price change; you are managing **Margin**, **Liquidation Risks**, and **Funding Rates**.

Vectorbt handles this via the `leverage` and `leverage_mode` parameters in the `Portfolio` class, but it requires strict configuration to be realistic for Binance Futures.

---

# Updated Requirements: e-trading Research & Production

## 1. Project Context Extension

* **Target Market:** Binance USDT-M Futures.
* **Instruments:** BTCUSDT, ETHUSDT, XRPUSDT, LTCUSDT.
* **Core Goal:** Maximize APR (>50%) by optimizing both entry logic and **dynamic leverage (1x to 20x)**.

---

## 2. Technical Architecture: Futures & Leverage

### 2.1 Leverage Optimization (Optuna)

* **Search Space:** Include `leverage` as a hyperparameter in the Optuna `trial.suggest_float("leverage", 1.0, 10.0, step=1.0)`.
* **Risk Filtering:** The objective function must penalize or return a "failure" value if a trial hits a **Liquidation Event**.
* **Vectorbt Setting:** Use `vbt.Portfolio.from_signals(..., leverage=leverage, leverage_mode='lazy')`.

### 2.2 Futures Mechanics

* **Slippage & Fees:** Set realistic Binance Futures fees (e.g., `fees=0.0004` for VIP0 taker) and `slippage` to account for market impact.
* **Shorting:** Enable `direction='both'` in `from_signals` to allow the strategy to take advantage of downtrends.
* **Margin & Funding (Advanced):** For 50%+ targets, the agent should simulate **Funding Rates**.
> *Note: Funding rates can be passed to Vectorbt as a `cash_sharing` adjustment or custom fee to simulate the cost of holding long/short positions in futures.*



---

## 3. Revised Functional Requirements

### 3.1 Objective Function: The "Risk-Adjusted Return"

To avoid "lucky" 1000% gains that actually have a 99% liquidation risk, the Optuna objective must use:

$$Score = \frac{Total\ Return}{Max\ Drawdown \times Leverage\ Factor}$$

This formula ensures that as leverage increases, the "punishment" for drawdowns scales proportionally.

### 3.2 Liquidation Monitoring

* The agent must implement a check for `pf.is_liquidated()`.
* If a strategy is liquidated even once during the test period, it must be discarded immediately, regardless of its end profit.

### 3.3 Multi-Asset Margin Simulation

* Since you trade 4 tickers, the research module should support **Cross-Margin** vs **Isolated-Margin** simulation.
* Use `vbt.Portfolio.from_signals(..., cash_sharing=True)` to simulate a single USDT wallet powering multiple futures positions.

---

## 4. Modified Pipeline (The "e-trading" Flow)

| Step | Action | Tool |
| --- | --- | --- |
| **Data** | Download 15m/1h Futures OHLCV + Funding Rates | Binance API / CSV |
| **Search** | Suggest Signals + **Leverage Value** | Optuna |
| **Simulate** | Run Long/Short backtest with Margin settings | Vectorbt |
| **Log** | Store all trials (including liquidations) | **SQLite** |
| **Deploy** | Push Best Params + Optimized Leverage to Prod | **PostgreSQL** |

---

## 5. Deployment Requirement for Antigravity Agent

The agent must ensure the `active_configs` table in PostgreSQL is updated to include:

* `leverage`: (float) The optimized leverage value.
* `margin_mode`: (string) "isolated" or "cross".
* `max_position_size`: (float) To prevent the bot from using 100% of the wallet on a high-leverage trade.
