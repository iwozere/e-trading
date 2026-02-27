# How-To: p07_combined Pipeline usage

This guide outlines the steps to start the `p07_combined` pipeline and the workflow for progressing from raw data to an optimized, production-ready model on Raspberry Pi.

---

## 🏗️ 1. Setup & Ingestion

The pipeline relies on a unified data structure and global macro features (VIX and BTC Market Cap).

1.  **Prepare Data**: Ensure your ticker files are in the `data/` directory following the naming convention: `{TICKER}_{TIMEFRAME}_{START}_{END}.csv`.
2.  **Macro Sync**: Run the pipeline for the first time or manually check for `data/vix/vix.csv` and `data/btc_mc/btc_mc.csv`. The `P07DataLoader` will attempt to auto-download missing macro data via `CoinGeckoDataDownloader`.

## 📈 2. Starting the Pipeline

The main entry point is `pipeline.py`.

```bash
python src/ml/pipeline/p07_combined/pipeline.py
```

**What it does initially:**
- Scans `data/` for valid CSV files.
- Trains/Loads the **Global Regime Model** (GaussianHMM).
- Checks for existing results in `results/p07_combined/` using `completed.flag` files to resume or skip.

## 🔬 3. Research & Optimization Loop

To progress your research, modify the following modules:

1.  **Labeling (`labeling.py`)**: Adjust ATR multipliers (`pt_mult`, `sl_mult`) or the time-path limit (`tpl_bars`) to change the strategy's "personality" (e.g., scalping vs. trend following).
2.  **Features (`features.py`)**: Add new TA-Lib indicators or stationary feature transformations.
3.  **Optimization (`optimize.py`)**: 
    - Adjust the Optuna search space in `objective()`.
    - Change the objective metric (e.g., Sharpe Ratio, Sortino, or Profit Factor).
    - Run the search to find the best hyper-parameters.

## 📊 4. Validation & Diagnostics

After an optimization run:
1.  Check the `results/p07_combined/{ticker}/{timeframe}/` directory.
2.  Inspect `strategy_overlay.png`: Look for signal clustering and equity curve stability.
3.  Inspect `prediction_diagnostics.png`: Ensure the model isn't overfitting and has well-calibrated confidence.

## ⚔️ 5. Realism Check (Backtrader)

Once you have a candidate model:
1.  Run `backtesting_bt.py` using the best parameters.
2.  Review performance with simulated **100-500ms latency**. If the Sharpe ratio drops significantly, your strategy is too sensitive to execution timing.

## 🚀 6. Export to Production (Pi)

1.  **Export**: Run `export.py` to convert your `XGBoost` model to `model.onnx` and generate `metadata.json`.
2.  **Deploy**: Transfer the `p07_combined` folder, `index.onnx`, and `metadata.json` to your Raspberry Pi.
3.  **Run**: Use `pi_inference.py` for headless execution.

---

## 🛠️ Progression Checklist

- [ ] Successful macro data download (VIX/BTC MC).
- [ ] Trained HMM regime model (`models/regime/macro_hmm.joblib`).
- [ ] First Optuna optimization run (`optuna_study.db`).
- [ ] Visual inspection of master overlay signals.
- [ ] Backtrader test with >200ms latency simulation.
- [ ] ONNX Export successful.
- [ ] Parity check between Research (Python) and Production (ONNX).
