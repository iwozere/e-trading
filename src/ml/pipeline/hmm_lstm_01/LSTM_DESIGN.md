---

# Design Document: Two-Stage Optimization for HMM + LSTM Trading System

---

## Overview

The trading system has two main components:

* **Stage 1:** Market regime detection via HMM, optimized on technical indicator parameters.
* **Stage 2:** Predictive modeling via LSTM, optimized on model architecture and training parameters, taking regime info as additional input.

---

## Goals

* Achieve reliable regime classification with HMM by optimizing indicator parameters (RSI, ATR, Bollinger Bands, volume SMA, etc.).
* Improve LSTM predictive accuracy by tuning model parameters, with regime info included.
* Maintain modularity to optimize indicators and LSTM hyperparameters separately but allow integration.
* Ensure reproducibility, prevent data leakage, and enable walk-forward testing.

---

## Step-by-step Optimization Workflow

---

### Step 1: Data Preparation

* **Input:** OHLCV data + precomputed log returns.
* **Split data** into training, validation, and test sets by chronological order (e.g., 60/20/20).
* Ensure **no look-ahead bias** during regime detection or LSTM training.

---

### Step 2: Indicator Parameter Optimization (HMM Regime Detection)

#### 2.1 Define Indicator Parameter Grid/Space

* Parameters to tune: RSI period, ATR period, Bollinger Bands period, volume SMA period.
* Parameter ranges: define small discrete sets (e.g., RSI=\[7,14,21]) or continuous intervals for Bayesian optimization.

#### 2.2 For Each Parameter Set:

* Compute indicators on training data only.
* Fit HMM with several states (2 to 6 states).
* Evaluate models via BIC/AIC criteria, predictive stability, and regime interpretability.

#### 2.3 Select Best Indicator + HMM Parameters:

* Based on minimum BIC or combined metric.
* Save trained HMM model and regime assignments for the training period.

---

### Step 3: Assign Market Regimes to Data

* Use trained HMM model to assign regime labels on **training + validation sets**.
* Store regimes as categorical variables aligned with OHLCV and indicator features.

---

### Step 4: LSTM Input Feature Construction

* Features:

  * Technical indicators used in HMM step (with best parameters).
  * Market regime labels encoded as one-hot vectors or embeddings.
  * Log returns or price changes.
  * Optional lagged features for temporal context.

* Normalize features (rolling window z-score or min-max scaling) to avoid leakage.

---

### Step 5: LSTM Model and Training Parameter Optimization

#### 5.1 Define LSTM Hyperparameter Search Space

* Number of LSTM layers (1-3)
* Hidden units per layer (32-128)
* Dropout rate (0.0-0.5)
* Learning rate (1e-4 to 1e-2)
* Batch size (32-256)
* Sequence length (e.g., 30-60 timesteps)
* Optimizer type (Adam, RMSProp)
* Epoch count with early stopping patience

#### 5.2 Define Objective Function

* Train LSTM on training set with given hyperparameters.
* Validate on validation set.
* Objective metric: validation loss (MSE or cross-entropy), or trading metric (Sharpe ratio, accuracy).
* Incorporate early stopping.

#### 5.3 Run Hyperparameter Optimization

* Use Bayesian optimization (Optuna recommended) for efficient search.
* Use pruning to stop poor trials early.
* Save best model checkpoint.

---

### Step 6: Testing and Final Evaluation

* Use best HMM indicator parameters + trained HMM model to assign regimes to test set.
* Use best LSTM model to predict on test set with regime-aware features.
* Evaluate trading strategy metrics: returns, Sharpe ratio, drawdown, trade frequency, etc.

---

## Implementation & Modularization

* Separate scripts/modules for:

  * Indicator computation & HMM training + optimization
  * Data processing + regime labeling
  * LSTM model definition, training, and optimization
  * Evaluation & plotting

* Configuration files or YAML/JSON to store hyperparameter ranges, paths, and constants.

* Logging and checkpoints to resume interrupted runs.

---

## Notes & Best Practices

* Always train HMM only on training data to avoid data leakage.
* For regime assignment on validation/test sets, use the trained HMM model without refitting.
* Normalize features separately for train/validation/test sets using training set stats.
* Consider retraining HMM and LSTM periodically for live trading or new data.
* Monitor overfitting with walk-forward or rolling-window validation.

---
