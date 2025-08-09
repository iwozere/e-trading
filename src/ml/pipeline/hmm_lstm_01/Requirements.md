Here’s the updated **Requirements.md** with **Optuna optimization** for both indicators and model parameters included:

---

````markdown
# Technical Requirements: Automated Trading Pipeline with HMM + Single LSTM + Optuna Optimization

## 1. General Idea
Goal — to build a trading pipeline where:

- **Hidden Markov Model (HMM)** determines the market regime (`regime`).
- **A single LSTM** is used to predict the next price (or log return).
- **LSTM receives the regime as an input feature**.
- **Optuna** is used for:
  - Hyperparameter optimization of indicators (RSI, Bollinger Bands, moving averages, etc.).
  - Hyperparameter optimization of the LSTM model.
- The result is used to generate trading signals.
- The model’s performance is compared with a naive prediction (`previous close`).
- As an example you can check `src/ml/lstm/lstm_optuna_log_return_from_csv.py`.

---

## 2. Pipeline Structure
### Steps:

1. **Data Loading (`x_01_data_loader.py`)**
   - Load OHLCV data for required timeframes: `5m`, `15m`, `1h`, `4h`.
   - Save to: `data/<symbol>_<tf>_<start_date>_<end_date>.csv`.
   - As an example you can take `src/util/data_downloader.py` or directly call it just by changing TEST_SCENARIOS object.

2. **Preprocessing (`x_02_preprocess.py`)**
   - Add `log_return`, normalization, and rolling statistics.
   - Save to: `data/processed/`.

3. **HMM Training (`x_03_train_hmm.py`)**
   - Train on the last 1–3 years of data (depending on timeframe).
   - Number of hidden states fixed (`n_components = 3`).
   - Save model to: `src/ml/hmm/model/hmm_<symbol>_<tf>_<timestamp>.pt`.

4. **HMM Application (`x_04_apply_hmm.py`)**
   - Generate `regime` column for each candle.
   - Save updated CSV to: `data/labeled/`.

5. **Optuna Optimization for Indicators (`x_05_optuna_indicators.py`)**
   - Optimize parameters for technical indicators (e.g., RSI length, Bollinger Bands period, SMA/EMA periods).
   - Objective function uses backtest performance metrics (Sharpe ratio, profit factor, drawdown).
   - Save best indicator parameters to: `results/indicators_<symbol>_<tf>_<timestamp>.json`.

6. **Optuna Optimization for LSTM (`x_06_optuna_lstm.py`)**
   - Search optimal LSTM parameters:
     - Sequence length
     - Hidden size
     - Batch size
     - Learning rate
     - Dropout
     - Number of layers
   - Use Optuna's pruning and TPE sampler for efficiency.
   - Save best model parameters to: `results/lstm_params_<symbol>_<tf>_<timestamp>.json`.

7. **LSTM Training (`x_07_train_lstm.py`)**
   - Use columns: `close`, `log_return`, `regime`, and additional features.
   - Add `regime` as a categorical feature (one-hot or embedding).
   - LSTM predicts `close[t+1]` or `log_return[t+1]`.
   - Save model to: `src/ml/lstm/model/lstm_<symbol>_<tf>_<timestamp>.pt`.

8. **Validation & Testing (`x_08_validate_lstm.py`)**
   - Use a hold-out set (last 10–20% of data).
   - Compare:
     - `MSE(LSTM)` vs `MSE(naive_pred = close[t])`
   - Generate chart + PDF report.
   - Save to: `results/lstm_<symbol>_<tf>_<timestamp>.pdf`, `.png` and `.json`.
   - Save best parameters to JSON.

---

## 3. Special Notes
- Only **one LSTM model** is used (not separate models for each regime).
- The `regime` is included as a feature in `X`, it does not define the architecture.
- The strategy uses:
  - Optimized indicator parameters for feature generation.
  - LSTM forecast for trading signal generation.

---

## 4. Configuration & Parameters
Example config (`config/pipeline/x01.yaml`):

```yaml
symbols: [BTCUSDT, ETHUSDT, LTCUSDT]
timeframes: [5m, 15m, 1h, 4h]

hmm:
  n_components: 3
  train_window_days: 730 # 2 years

lstm:
  sequence_length: 60
  hidden_size: 64
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  validation_split: 0.2
  dropout: 0.2
  num_layers: 2

optuna:
  n_trials: 50
  sampler: tpe
  pruning: true

evaluation:
  test_split: 0.1
  baseline_model: naive
  metrics: [mse, directional_accuracy, sharpe_ratio]
````

---

## 5. Quality Evaluation

Main metrics:

* **MSE(LSTM)** vs **MSE(Naive)**
* **Directional Accuracy**
* **Sharpe Ratio**
* **Max Drawdown**
* **Profit Factor**

Charts:

* Market regimes overlaid on prices
* Predictions vs Actual values
* Error over time
* Backtest equity curve

---

## 6. File Structure

```
project/
├── data/
│   ├── processed/
│   └── labeled/
├── src/
│   ├── ml/
│   │   ├── hmm/
│   │   └── lstm/
├── results/
├── reports/
├── scripts/
├── config/
└── retrain_pipeline.sh
```

---

## 7. Update Frequency

| Component            | Retraining Frequency | Data Window          |
| -------------------- | -------------------- | -------------------- |
| HMM                  | Monthly / Quarterly  | 1–3 years            |
| Indicator Parameters | Weekly / Bi-weekly   | 1–6 months           |
| LSTM                 | Weekly / Bi-weekly   | 1–3 months (rolling) |

```

---
