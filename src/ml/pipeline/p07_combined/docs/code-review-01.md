# Code Review Analysis: P07 Combined Pipeline

**Reviewer**: Senior Python Developer
**Date**: 2026-02-27
**Component**: `src/ml/pipeline/p07_combined`

---

## 🛑 Critical Issues (Must Fix)

### 1. Data Leakage: Look-Ahead Bias in Regime Model
The `P07RegimeModel.train()` method in `regime_model.py` is called on the entire macro dataset in `pipeline.py` (Line 44-55).
- **Issue**: When `enrich_data` is used for backtesting (e.g., year 2020), the HMM has already "seen" the volatility and market cap trends for 2024. This informs the regime classification for historical dates using future info.
- **Impact**: Significant inflation of backtest results.
- **Recommendation**: Implement a temporal split for the HMM or use an Expanding Window approach where the HMM is only trained on data prior to the current prediction date.

### 2. Duplicated Logic & Maintenance Risk
There is significant overlap between `pipeline.py:save_artifacts` and `optimize.py:objective`.
- **Issue**: Both functions independently implement the sequence: `build_features` -> `get_triple_barrier_labels` -> data splitting -> `P07XGBModel` fit -> `vbt.Portfolio.from_signals`.
- **Impact**: A change in feature engineering or labeling logic must be applied in two places. If they diverge, the "best params" from optimization won't match the "saved artifacts".
- **Recommendation**: Refactor this sequence into a shared `P07Trainer` or `P07Evaluator` class.

### 3. Missing Exit Logic in Portfolio Evaluation
In `pipeline.py`, the signal mapping for `actual_signals` only considers entries and simple exits.
- **Issue**: If the model is trained with `direction='both'` (allowed in `optimize.py`), the signal mapping in `pipeline.py` (Lines 158-165) does not correctly differentiate between Closing Long vs. Opening Short.
- **Impact**: Visualization of trade flow may be misleading compared to the actual Portfolio performance.

---

## ⚠️ High/Medium Issues

### 4. Placeholder Inference Client (`pi_inference.py`)
- **Issue**: The `preprocess` method is a skeleton returning zeros.
- **Impact**: The "developed code" for edge inference is non-functional.
- **Recommendation**: Implement exact parity logic for TA-Lib features and scaling based on `metadata.json`.

### 5. Inefficient Labeling Loop (`labeling.py`)
- **Issue**: `get_triple_barrier_labels` uses a nested Python loop.
- **Impact**: For high-frequency data (e.g., 1m or 5m over years), this will become a massive bottleneck.
- **Recommendation**: Use `numba.jit` to decorate the search loop or use VectorBT's built-in labeling utilities if available.

### 6. Hardcoded Schema Assumptions (`data_loader.py`)
- **Issue**: The regex in `parse_filename` is strictly tied to a specific underscore-delimited format.
- **Impact**: Slight changes in data sourcing (e.g., extra underscores in ticker names) will break the batch processor.

---

## ℹ️ Style & Technical Debt

### 7. Typo in Pipeline Logic (`pipeline.py`)
- Line 114: `feature_config` is defined manually, missing potential hyperparams from the best model.

### 8. Optuna Integration
- Use `trial.report()` and `trial.should_prune()` inside the objective function to allow Optuna to stop poor-performing trials earlier, saving significant time.

### 9. Type Hint Inconsistencies
- `export.py` and `pi_inference.py` have missing or generic `Any` type hints for core objects like models and dataframes.

---

## Summary Verdict
The pipeline architecture is well-structured for modularity, but the **look-ahead bias** in the regime model makes current backtest results untrustworthy. Prioritize refactoring the HMM training and consolidating the training/evaluation logic.
