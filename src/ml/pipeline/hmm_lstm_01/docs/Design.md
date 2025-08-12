Got it — I can draft **Design.md** and **Tasks.md** from your updated `Requirements.md` with Optuna included.

Here’s what they could look like:

---

## **Design.md**

````markdown
# Design Document: Automated Trading Pipeline with HMM + LSTM + Optuna

## 1. System Overview
The pipeline integrates **Hidden Markov Models (HMM)** for market regime detection, a **single LSTM model** for next-step forecasting, and **Optuna** for hyperparameter optimization of both **technical indicators** and **LSTM architecture**.

Data flows sequentially through the following components:

1. **Data Loader** → Raw OHLCV historical data acquisition.
2. **Preprocessing** → Normalization, feature engineering, log returns.
3. **HMM Training** → Learn market regimes.
4. **HMM Application** → Label each candle with regime state.
5. **Optuna Indicator Optimization** → Tune technical indicator parameters.
6. **Optuna LSTM Optimization** → Tune model hyperparameters.
7. **LSTM Training** → Train final model with optimized parameters.
8. **Validation & Testing** → Compare against naive baseline, generate reports.

---

## 2. Architecture Diagram

```text
 ┌────────────┐      ┌───────────────┐      ┌─────────────┐
 │ Data Loader│──▶──▶│ Preprocessing │──▶──▶│ Train HMM   │
 └────────────┘      └───────────────┘      └─────┬───────┘
                                                    │
                                                    ▼
                                            ┌─────────────┐
                                            │ Apply HMM   │
                                            └─────┬───────┘
                                                  │
                                                  ▼
     ┌───────────────────────┐     ┌─────────────────────────┐
     │ Optuna Indicators     │     │ Optuna LSTM              │
     │ (Feature parameters)  │     │ (Model parameters)       │
     └──────────┬────────────┘     └───────────────┬──────────┘
                │                                  │
                ▼                                  ▼
         ┌───────────────┐                 ┌───────────────┐
         │ Train LSTM    │                 │ Validate/Test │
         └───────────────┘                 └───────────────┘
````

---

## 3. Data Flow

1. **Input**

   * OHLCV from exchange APIs (Binance, etc.) for multiple timeframes.
   * Stored in `data/`.

2. **Preprocessing**

   * Normalization (MinMax or StandardScaler).
   * Rolling statistics.
   * Technical indicators (parameters later tuned via Optuna).
   * Saved in `data/processed/`.

3. **HMM**

   * Trained with selected features.
   * Produces `regime` labels.
   * Output stored in `data/labeled/`.

4. **Optimization**

   * Indicators: Sharpe ratio, profit factor, and drawdown as objective metrics.
   * LSTM: MSE, directional accuracy, and Sharpe ratio as objective metrics.

5. **Model Training**

   * Inputs: `close`, `log_return`, `regime` (categorical), optimized indicators.
   * Target: Next-step `close` or `log_return`.

6. **Evaluation**

   * Metrics: MSE, Directional Accuracy, Sharpe Ratio, Max Drawdown, Profit Factor.
   * Output: Reports, charts, and JSON parameter files.

---

## 4. Key Components

* **HMM**

  * `n_components = 3`
  * Uses rolling training window.

* **Indicators** (to optimize with Optuna)

  * RSI period
  * Bollinger Bands period & std dev
  * SMA/EMA lengths

* **LSTM Hyperparameters** (to optimize with Optuna)

  * Sequence length
  * Hidden size
  * Batch size
  * Learning rate
  * Dropout
  * Number of layers

---

## 5. Persistence & Outputs

* Models stored in:

  * `src/ml/hmm/model/`
  * `src/ml/lstm/model/`
* Parameters stored in:

  * `results/indicators_<symbol>_<tf>_<timestamp>.json`
  * `results/lstm_params_<symbol>_<tf>_<timestamp>.json`
* Reports stored in `results/` and `reports/`.

---

## 6. Retraining Policy

| Component            | Retraining Frequency | Data Window          |
| -------------------- | -------------------- | -------------------- |
| HMM                  | Monthly / Quarterly  | 1–3 years            |
| Indicator Parameters | Weekly / Bi-weekly   | 1–6 months           |
| LSTM                 | Weekly / Bi-weekly   | 1–3 months (rolling) |

````

