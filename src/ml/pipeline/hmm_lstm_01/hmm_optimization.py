"""
Hidden Markov Model (HMM) Regime Detection with Parameter Optimization
======================================================================

This script trains a Gaussian Hidden Markov Model (HMM) to detect 
market regimes (bull, bear, sideways) based on OHLCV price data and 
technical indicators from TA-Lib.

Features:
---------
1. Reads multiple CSV files from the input folder. Each file should contain:
   - Columns: "open", "high", "low", "close", "volume", "log_return"
   - log_return must already be precomputed and included in the file.

2. Computes technical indicators using TA-Lib:
   - RSI (Relative Strength Index)
   - ATR (Average True Range)
   - Bollinger Bands (Upper, Middle, Lower)
   - SMA of Volume

3. Parameter Optimization:
   - Supports two modes:
       * Grid Search  : Exhaustively tests all parameter combinations.
       * Optuna Search: Uses Bayesian optimization to minimize BIC.
   - Parameters optimized:
       * RSI period
       * ATR period
       * Bollinger Bands period
       * Volume SMA period
       * Number of HMM states

4. Model Evaluation:
   - For each parameter set, computes AIC and BIC.
   - Stores all results in `full_optimization_report.csv`.

5. Regime Mapping:
   - If number of HMM states <= 3 → direct mapping: bear, sideways, bull
   - If number of states > 3 → clusters states into 3 groups via KMeans.

6. Visualization:
   - Subplot 1: Price with Bollinger Bands (points colored by regime)
   - Subplot 2: RSI with overbought/oversold lines
   - Subplot 3: Histogram of log returns

Outputs:
--------
- Annotated CSV per input file with predicted HMM states and mapped regimes.
- PNG plot with price, indicators, and return distribution.
- Global CSV report with all tested parameter combinations and AIC/BIC.

Usage:
------
1. Place CSV files into `input_folder` (configured at top).
2. Ensure TA-Lib is installed (`pip install TA-Lib`).
3. Run script. Select optimization mode via SEARCH_MODE ("grid" or "optuna").
4. Review results in `output_folder`.

Author:
-------
GPT-5 — tailored for adaptive HMM regime classification in trading research.

"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans
from itertools import product
import optuna

# -------------------------
# CONFIGURATION
# -------------------------
SEARCH_MODE = "optuna"  # "grid" or "optuna"
N_OPTUNA_TRIALS = 50    # Only used if SEARCH_MODE == "optuna"

input_folder = "data"
output_folder = "results"
os.makedirs(output_folder, exist_ok=True)

# -------------------------
# Indicator calculation
# -------------------------
def compute_indicators(df, rsi_period, atr_period, bb_period, vol_period):
    """Compute TA-Lib indicators for RSI, ATR, Bollinger Bands, and SMA Volume."""
    df['rsi'] = talib.RSI(df['close'], timeperiod=rsi_period)
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=atr_period)
    upper, middle, lower = talib.BBANDS(df['close'], timeperiod=bb_period, nbdevup=2, nbdevdn=2, matype=0)
    df['bb_upper'] = upper
    df['bb_middle'] = middle
    df['bb_lower'] = lower
    df['vol_sma'] = talib.SMA(df['volume'], timeperiod=vol_period)
    return df

# -------------------------
# Evaluate HMM model
# -------------------------
def evaluate_hmm_info_criteria(X, n_states):
    """Fit GaussianHMM and return AIC, BIC, and fitted model."""
    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100, random_state=42)
    model.fit(X)
    logL = model.score(X)
    k = n_states * (n_states - 1) + n_states * X.shape[1] * 2
    aic = -2 * logL + 2 * k
    bic = -2 * logL + k * np.log(len(X))
    return aic, bic, model

# -------------------------
# Map HMM states into regimes
# -------------------------
def map_states_to_regimes(df, state_col='state', ret_col='log_return', atr_col='atr'):
    """Map HMM states into bull, bear, and sideways regimes."""
    stats = []
    states = sorted(df[state_col].unique())
    for s in states:
        sub = df[df[state_col] == s]
        mean_r = sub[ret_col].mean()
        std_r = sub[ret_col].std()
        mean_atr = sub[atr_col].mean() if atr_col in sub.columns else np.nan
        stats.append((s, mean_r, std_r, mean_atr, len(sub)))

    stats_df = pd.DataFrame(stats, columns=['state','mean_r','std_r','mean_atr','count'])

    if len(states) <= 3:
        stats_df = stats_df.sort_values('mean_r').reset_index(drop=True)
        labels = ['bear', 'sideways', 'bull']
        mapping = {int(row['state']): labels[i] if i < 3 else labels[-1]
                   for i, row in stats_df.iterrows()}
        df['regime'] = df[state_col].map(mapping)
        return mapping, df

    X = stats_df[['mean_r','std_r']].fillna(0).values
    kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
    stats_df['cluster'] = kmeans.labels_

    cluster_mean = stats_df.groupby('cluster')['mean_r'].mean().sort_values()
    cluster_to_label = {cluster_idx: label
                        for cluster_idx, label in zip(cluster_mean.index, ['bear', 'sideways', 'bull'])}

    mapping = {int(row['state']): cluster_to_label[row['cluster']]
               for _, row in stats_df.iterrows()}
    df['regime'] = df[state_col].map(mapping)
    return mapping, df

# -------------------------
# Plot results
# -------------------------
def plot_results(df, file_name):
    """Plot price with Bollinger Bands, RSI, and log return distribution."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    color_map = {'bull': 'green', 'bear': 'red', 'sideways': 'blue'}
    colors = df['regime'].map(color_map)

    axes[0].scatter(df.index, df['close'], c=colors, s=10)
    axes[0].plot(df['bb_upper'], color='orange', alpha=0.5)
    axes[0].plot(df['bb_middle'], color='grey', alpha=0.5)
    axes[0].plot(df['bb_lower'], color='orange', alpha=0.5)
    axes[0].set_title('Price with Bollinger Bands')

    axes[1].plot(df['rsi'], color='purple')
    axes[1].axhline(70, color='red', linestyle='--', alpha=0.5)
    axes[1].axhline(30, color='green', linestyle='--', alpha=0.5)
    axes[1].set_title('RSI')

    axes[2].hist(df['log_return'].dropna(), bins=50, color='grey', alpha=0.7)
    axes[2].set_title('Log Return Distribution')

    plt.tight_layout()
    plt.savefig(file_name, dpi=150)
    plt.close()

# -------------------------
# GRID SEARCH
# -------------------------
def run_grid_search(df, file_base):
    """Run exhaustive grid search over all parameter combinations."""
    rsi_grid = [7, 14, 21]
    atr_grid = [7, 14, 21]
    bb_grid = [14, 20, 30]
    vol_grid = [14, 20, 30]
    n_states_grid = [2, 3, 4, 5]

    best_bic = np.inf
    best_aic = None
    best_params = None
    best_model = None
    best_df = None
    report_rows = []

    for rsi_p, atr_p, bb_p, vol_p, n_states in product(rsi_grid, atr_grid, bb_grid, vol_grid, n_states_grid):
        try:
            temp_df = compute_indicators(df.copy(), rsi_p, atr_p, bb_p, vol_p).dropna()
            features = temp_df[['rsi', 'atr', 'bb_upper', 'bb_middle', 'bb_lower', 'vol_sma', 'log_return']].values
            aic, bic, model = evaluate_hmm_info_criteria(features, n_states)

            report_rows.append({'file': file_base, 'rsi': rsi_p, 'atr': atr_p, 'bb': bb_p,
                                'vol_sma': vol_p, 'n_states': n_states, 'aic': aic, 'bic': bic})

            if bic < best_bic:
                best_bic = bic
                best_aic = aic
                best_params = (rsi_p, atr_p, bb_p, vol_p, n_states)
                best_model = model
                best_df = temp_df.copy()
        except Exception:
            continue

    return best_df, best_model, best_params, report_rows

# -------------------------
# OPTUNA SEARCH
# -------------------------
def run_optuna_search(df, file_base):
    """Run Optuna optimization to minimize BIC over parameter space."""
    best_result = {'bic': np.inf}
    report_rows = []

    def objective(trial):
        rsi_p = trial.suggest_int('rsi', 5, 25)
        atr_p = trial.suggest_int('atr', 5, 25)
        bb_p = trial.suggest_int('bb', 10, 30)
        vol_p = trial.suggest_int('vol_sma', 10, 30)
        n_states = trial.suggest_int('n_states', 2, 6)

        try:
            temp_df = compute_indicators(df.copy(), rsi_p, atr_p, bb_p, vol_p).dropna()
            features = temp_df[['rsi', 'atr', 'bb_upper', 'bb_middle', 'bb_lower', 'vol_sma', 'log_return']].values

            # Initialize model with warm_start=True for incremental fitting
            model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100, random_state=42, warm_start=True)
            
            best_logL = -np.inf
            for i in range(1, 101, 10):  # Fit in increments of 10 iterations
                model.n_iter = i
                model.fit(features)
                logL = model.score(features)
                trial.report(-logL, i)  # Report negative log likelihood as "loss"
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                if logL > best_logL:
                    best_logL = logL

            k = n_states * (n_states - 1) + n_states * features.shape[1] * 2
            bic = -2 * best_logL + k * np.log(len(features))
            aic = -2 * best_logL + 2 * k

            report_rows.append({'file': file_base, 'rsi': rsi_p, 'atr': atr_p, 'bb': bb_p,
                                'vol_sma': vol_p, 'n_states': n_states, 'aic': aic, 'bic': bic})

            if bic < best_result['bic']:
                best_result.update({'bic': bic, 'aic': aic, 'params': (rsi_p, atr_p, bb_p, vol_p, n_states),
                                    'model': model, 'df': temp_df.copy()})

            return bic

        except optuna.exceptions.TrialPruned:
            raise

        except Exception:
            return np.inf

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS)

    return best_result['df'], best_result['model'], best_result['params'], report_rows

# -------------------------
# MAIN LOOP
# -------------------------
full_report_rows = []
for file in glob.glob(os.path.join(input_folder, "*.csv")):
    df = pd.read_csv(file).dropna().reset_index(drop=True)
    file_base = os.path.splitext(os.path.basename(file))[0]

    if SEARCH_MODE == "grid":
        best_df, best_model, best_params, report_rows = run_grid_search(df, file_base)
    else:
        best_df, best_model, best_params, report_rows = run_optuna_search(df, file_base)

    full_report_rows.extend(report_rows)

    if best_df is None:
        continue

    features_best = best_df[['rsi', 'atr', 'bb_upper', 'bb_middle', 'bb_lower', 'vol_sma', 'log_return']].values
    best_df['state'] = best_model.predict(features_best)

    mapping, best_df = map_states_to_regimes(best_df)
    best_df.to_csv(os.path.join(output_folder, f"{file_base}_annotated.csv"), index=False)
    plot_results(best_df, os.path.join(output_folder, f"{file_base}.png"))

pd.DataFrame(full_report_rows).to_csv(os.path.join(output_folder, "full_hmm_optimization_report.csv"), index=False)
print("Done. Report saved to full_optimization_report.csv")
