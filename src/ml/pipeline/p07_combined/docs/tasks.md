# Implementation Guideline: p07_combined Roadmap (Submodule Integration)

This document serves as the granular checklist for the Senior Python Developer to implement the `p07_combined` trading pipeline, integrated as a submodule within the `e-trading` project.

---

## Phase 0: System-Level Extensions (src/data/downloader)
- [ ] **BTC Market Cap Downloader**
  - [ ] Create `src/data/downloader/btc_marketcap_downloader.py` following the `BaseDataDownloader` pattern.
  - [ ] Use CoinGecko API (`/coins/bitcoin/market_chart`) to fetch historical market cap data.
  - [ ] Register `"btc_mc"` provider in `src/data/downloader/data_downloader_factory.py`.
- [ ] **VIX Downloader Review**
  - [ ] Ensure `VIXDataDownloader.update_vix()` produces consistent data in `data/vix/vix.csv` for pipeline consumption.

---

## Phase 1: Pipeline Foundation & Data Integration
- [x] **Module Initialization**
  - [x] Initialize `src/ml/pipeline/p07_combined/` structure (no redundant downloaders).
- [x] **Unified Data Loader (`data_loader.py`)**
  - [x] Implement regex-based filename parser for ticker files: `{ticker}_{timeframe}_{start}_{end}.csv`.
  - [x] **Feature Merge Logic**:
    - [x] Interface with `VIXDataDownloader` (via `data/vix/vix.csv`).
    - [x] Interface with new `BTCMarketCapDownloader` (via `data/btc_mc/`).
    - [x] Join global macro features (VIX, BTC MC) to local ticker OHLCV based on timestamps.
- [x] **Stateful Recovery**
  - [x] Setup Optuna SQLite storage (`src/ml/pipeline/p07_combined/optuna_study.db`).
  - [x] Implement resumption check via `completed.flag` in result directories.

---

## Phase 2: Macro Intelligence (Global HMM)
- [x] **Global Feature Engine**
  - [x] Process macro features (VIX, BTC MC) into stationarity-adjusted inputs.
- [x] **Global HMM Training (`regime_model.py`)**
  - [x] Implement Unsupervised HMM (GaussianHMM) training.
  - [x] Save trained HMM state mapping (Heuristic state mapping simplified).
- [x] **Context Enrichment**
  - [x] Inject `global_regime_id` into the training dataset for the main XGBoost model.

---

## Phase 3: Research Pipeline (VectorBT + Optuna)
- [x] **Triple Barrier Labeling (`labeling.py`)**
  - [x] Implement ATR-based PT/SL and Time barriers.
- [x] **Feature Engineering (`features.py`)**
  - [x] Standardize on `ta-lib` (C-wrappers) for consistent indicator calculation.
  - [x] Stationarity handling (Log-returns, fractional differentiation).
- [x] **Tabular Model (`models.py`)**
  - [x] XGBoost Classifier with `tree_method='hist'` for CPU efficiency.
- [x] **Optimization Pipeline (`optimize.py`)**
  - [x] Optuna objective utilizing `vectorbt` for fast backtesting.
  - [x] **Zero-Trade Fix**: Expanded `tpl_bars` and introduced dynamic probability thresholds for multi-timeframe stability.

---

## Phase 4: Diagnostic Dashboard (`visualizer.py`)
- [x] **Suite Generation**
  - [x] `predictions_scatter.png`, `error_distribution.png`, `tbm_barrier_hits.png`.
  - [x] **Master Overlay**: OHLCV + TBM Barriers + Prob Heatmap + Equity Curve (Simplified Overlay).
  - [x] **Equity Curve Fix**: Masked terminal forced-exits to prevent artificial realized PnL jumps.

---

## Phase 5: Realism Layer (Backtrader)
- [/] **Live-Ready Backtest (`backtesting_bt.py`)** [IN PROGRESS]
  - [/] Port best models to Backtrader `Strategy` (Infrastructure built, logic pending).
  - [ ] Simulate **100-500ms latency**.
  - [ ] Use `Store` abstractions (Binance/IB) for future live execution compatibility.
  - [ ] *Note: Currently a standalone module, NOT integrated into main pipeline.py.*

---

## Phase 6: ONNX Export & Pi Deployment
- [x] **Export Engine**
  - [x] XGBoost to `.onnx` export (Implemented in export.py).
  - [x] `metadata.json` for feature scaling and parity checks.
- [x] **Minimal Pi Client**
  - [x] Headless runner for ARM64 using only `onnxruntime`, `pandas`, and `ta-lib` (Implemented in pi_inference.py).
