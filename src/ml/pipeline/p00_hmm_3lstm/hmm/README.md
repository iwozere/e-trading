Here is the updated `README.md` content, reflecting your current project structure, feature/backend options, and example usage:

```markdown
# Market Regime Detection with Hidden Markov Models (HMM)

This project builds a multi-timeframe, unsupervised regime detection system using **Hidden Markov Models**. It identifies **bullish**, **bearish**, and **sideways** market conditions based on log returns, volatility, and technical indicators.

---

## 📁 Directory Structure

```

├── data/               # Input CSV files (OHLCV)
├── results/            # Outputs (.csv, .json, .png)
├── models/             # Saved HMM models (.pkl)
├── scripts/            # Main processing scripts

````

---

## 🔧 Requirements

* Python 3.8+
* Install dependencies:

```bash
pip install numpy pandas matplotlib hmmlearn optuna joblib TA-Lib pomegranate
````

> **Note:** TA-Lib requires system dependencies. On Debian/Ubuntu:
>
> ```bash
> sudo apt-get install -y libta-lib0 libta-lib-dev
> pip install TA-Lib
> ```

---

## 📥 Input Format

Files must be stored in the `data/` folder, named like:

```
SYMBOL_TIMEFRAME_START_END.csv
Example: LTCUSDT_1h_20220101_20250707.csv
```

Each CSV should contain:

* `timestamp` (datetime)
* `open`, `high`, `low`, `close`, `volume`

---

## 🚀 Usage

### 1. Train Single File (with optimization)

```bash
python scripts/train_hmm.py \
  --csv data/LTCUSDT_1h_20220101_20250707.csv \
  --timeframe LTCUSDT_1h_20220101_20250707 \
  --optimize --n_trials 50 \
  --features log_return volatility rsi macd boll \
  --backend pomegranate
```

### 2. Evaluate and Plot

```bash
python scripts/evaluate_hmm.py \
  --csv results/LTCUSDT_1h_20220101_20250707.csv
```

### 3. Run All Files in `data/`

```bash
python scripts/run_all.py
```

This runs optimization and training on all CSV files in `data/` using the full feature set and `pomegranate` backend by default.

---

## 🧠 Model Details

* Model: `GaussianHMM` from `hmmlearn` (default) or `HiddenMarkovModel` from `pomegranate` (Bayesian support)
* Features:

  * Mandatory: `log_return`
  * Optional: `volatility` (rolling std), `RSI`, `MACD`, `Bollinger Band Width`
* Regime count: 2–4 (optimized)
* Volatility window and indicator parameters are optimized via Optuna when `--optimize` is set

### Output Files

* `results/LTCUSDT_1h_....csv`: with regime labels
* `results/LTCUSDT_1h_....json`: best params from Optuna
* `results/LTCUSDT_1h_....png`: visualization of regimes
* `models/LTCUSDT_1h_....pkl`: saved HMM model

---

## 🎨 Visualization

* Bullish regime = **green**
* Bearish regime = **red**
* Sideways regime = **black**

Regimes are mapped based on mean return per state.

---

## 🧪 Optimization Objective

* Uses Optuna to minimize **regime similarity**:

  * Objective = **negative std of regime means** (higher separation is better)
* Optionally optimizes:

  * Number of regimes
  * Volatility rolling window
  * RSI period
  * MACD fast, slow, signal periods
  * Bollinger Band window

---

## 📌 Next Steps

* ✅ Add more features: RSI, MACD, Bollinger Bands
* ✅ Enable feature selection via CLI
* ✅ Support backend switching (`hmmlearn` or `pomegranate`)
* ⏳ Implement Dirichlet Process HMM for automatic regime count
* ⏳ Support online/incremental re-fitting
* ⏳ Strategy integration with live signals and alerts

---

## 📫 Author

Mark Kosyrev — 2025

```

---

Would you like me to save this as `README.md` file or help you with further code improvements next?
```
