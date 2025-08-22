"""
Automated HMM-based Market Regime Detection for Cryptocurrency Time Series.

This script automates the process of training Hidden Markov Models (HMMs) to
identify distinct market regimes (Bull, Bear, Sideways) from financial time
series data. It is designed to be run as a pipeline, processing multiple
CSV files and generating corresponding model and plot files.

Workflow:
1.  Scans the `data/` directory for input files matching the format
    `symbol_timeframe.csv` (e.g., 'btcusdt_15m.csv').
2.  For each file, it loads the data and calculates the 'log_return' from the
    'close' price. It then uses 'log_return' and 'volume' as the primary
    features for the HMM.
3.  It leverages the Optuna library to perform hyperparameter optimization for a
    3-state Gaussian HMM, maximizing the silhouette score to ensure a good
    separation between the identified states.
4.  A final HMM is trained using the best hyperparameters found.
5.  The model's hidden states (0, 1, 2) are interpreted and mapped to meaningful
    market regimes based on the statistical properties of their log returns.
6.  Three output files are saved to the `src/strategy/ml/hmm/` directory:
    - A trained model file: `hmm_symbol_timeframe.joblib`
    - A visualization plot: `hmm_symbol_timeframe.png`
    - A JSON file with the best hyperparameters: `hmm_symbol_timeframe.json`

Regime Interpretation (Values 0, 1, 2):
The HMM identifies three distinct, unnamed states (0, 1, 2). This script assigns
meaning to them as follows:

-   **Bull Regime (Green):** This is the state that exhibits the highest
    positive average `log_return`. It represents periods of sustained upward
    price momentum.

-   **Bear Regime (Red):** This is the state that exhibits the most negative
    average `log_return`. It represents periods of sustained downward price
    pressure.

-   **Sideways/Consolidation Regime (Black):** This is the state whose average
    `log_return` is closest to zero. It represents periods of low volatility
    and lack of a clear directional trend.

Required Input Data Format:
-   CSV files located in the `data/` directory.
-   Filename format: `symbol_timeframe.csv` (e.g., 'ethusdt_1h.csv').
-   Required columns: 'timestamp', 'close', and 'volume'. The 'log_return'
    column is calculated automatically by the script.
"""

import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import joblib
import optuna
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import logging
import warnings
import json

# --- Suppress verbose logging and warnings for a cleaner output ---
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_preprocess_data(file_path: Path) -> tuple[np.ndarray | None, pd.DataFrame | None]:
    """
    Loads data, calculates log returns, validates columns, and prepares features for HMM.
    It calculates the 'log_return' column from the 'close' price.
    Features used for HMM: 'log_return' and 'volume'.
    """
    try:
        df = pd.read_csv(file_path)
        # The script now only requires these basic columns.
        required_cols = ['timestamp', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            logging.warning(f"Skipping {file_path.name}: Missing one of required columns: {required_cols}.")
            return None, None

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        # --- KEY CHANGE: Calculate log_return directly in the script ---
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))

        # Select features for the HMM
        features_df = df[['log_return', 'volume']].copy()
        # Drop rows with NaN values (specifically the first row due to log_return calculation)
        features_df.dropna(inplace=True)

        if len(features_df) < 100:
            logging.warning(f"Skipping {file_path.name}: Not enough data points ({len(features_df)}).")
            return None, None

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_df)

        # Return scaled features and the original dataframe (now with log_return)
        # aligned to the features_df index.
        return scaled_features, df.loc[features_df.index]
    except Exception as e:
        logging.error(f"Failed to load or preprocess {file_path.name}: {e}")
        return None, None

def find_best_hyperparameters(trial: optuna.trial.Trial, X_train: np.ndarray) -> float:
    """Optuna objective function for a 3-state HMM."""
    params = {
        'covariance_type': trial.suggest_categorical("covariance_type", ["full", "diag", "tied", "spherical"]),
        'n_iter': trial.suggest_int("n_iter", 50, 300),
        'tol': trial.suggest_float("tol", 1e-4, 1e-2, log=True),
        'params': trial.suggest_categorical("params", ["stmc", "tmc", "mc"])
    }

    model = GaussianHMM(n_components=3, random_state=42, **params)

    try:
        model.fit(X_train)
        preds = model.predict(X_train)
        if len(np.unique(preds)) < 2: return -1.0
        return silhouette_score(X_train, preds)
    except (ValueError, np.linalg.LinAlgError):
        return float('-inf')

def map_regimes_to_market_conditions(df: pd.DataFrame) -> dict:
    """Interprets HMM regimes by analyzing the mean 'log_return' for each state."""
    mean_log_returns = df.groupby('hmm_regime')['log_return'].mean()

    bull_regime = mean_log_returns.idxmax()
    bear_regime = mean_log_returns.idxmin()
    sideways_regime = (mean_log_returns - 0).abs().idxmin()

    # Ensure all three regimes are unique, falling back to simple sorting if needed.
    if len({bull_regime, bear_regime, sideways_regime}) < 3:
        sorted_regimes = mean_log_returns.sort_values().index
        bear_regime, sideways_regime, bull_regime = sorted_regimes[0], sorted_regimes[1], sorted_regimes[2]

    return {
        bull_regime: ('Bull', 'green'),
        bear_regime: ('Bear', 'red'),
        sideways_regime: ('Sideways', 'black')
    }

def plot_colored_prices_and_states(df: pd.DataFrame, regime_map: dict, title: str, save_path: Path):
    """Plots and saves the closing price colored by market regime."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(60, 9), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    ax1.set_title(title)
    for regime, (label, color) in regime_map.items():
        subset = df[df['hmm_regime'] == regime]
        ax1.scatter(subset.index, subset['close'], color=color, label=f'{label} Market', s=5, alpha=0.7)
    ax1.set_ylabel("Close Price")
    ax1.legend()
    ax1.grid(True, which='major', linestyle='--', alpha=0.6)

    ax2.set_title("HMM Regime Sequence")
    for regime, (_, color) in regime_map.items():
        subset = df[df['hmm_regime'] == regime]
        ax2.scatter(subset.index, subset['hmm_regime'], color=color, s=15)
    ax2.set_ylabel("HMM Regime")
    ax2.set_xlabel("Date")
    ax2.set_yticks(list(regime_map.keys()))
    ax2.grid(True, which='major', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def process_file(file_path: Path, output_dir: Path):
    """Runs the full HMM pipeline for a single data file."""
    try:
        filename_stem = file_path.stem
        tokens = filename_stem.split('_')
        symbol = tokens[0]
        timeframe = tokens[1]
    except ValueError:
        logging.warning(f"Skipping {file_path.name}: Filename must be 'symbol_timeframe.csv'.")
        return

    logging.info(f"Processing {symbol.upper()} ({timeframe})...")

    X_scaled, df = load_and_preprocess_data(file_path)
    if X_scaled is None or df is None: return

    logging.info("Starting hyperparameter optimization...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: find_best_hyperparameters(trial, X_scaled), n_trials=75, show_progress_bar=False)

    best_params = study.best_params
    logging.info(f"Best hyperparameters found for {symbol.upper()}: {best_params}")

    logging.info("Training final model and saving artifacts...")
    final_model = GaussianHMM(n_components=3, random_state=42, **best_params)
    final_model.fit(X_scaled)

    base_filename = f"hmm_{symbol}_{timeframe}"

    # 1. Save the model
    model_path = output_dir / f"{base_filename}.joblib"
    joblib.dump(final_model, model_path)
    logging.info(f"Saved model to {model_path}")

    # 2. Save the best hyperparameters to a JSON file
    json_path = output_dir / f"{base_filename}.json"
    with open(json_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    logging.info(f"Saved hyperparameters to {json_path}")

    # 3. Predict, interpret, and save the plot
    df['hmm_regime'] = final_model.predict(X_scaled)
    regime_map = map_regimes_to_market_conditions(df)

    plot_title = f"{symbol.upper()} ({timeframe}) Price Colored by HMM-Identified Market Regime"
    plot_path = output_dir / f"{base_filename}.png"
    plot_colored_prices_and_states(df, regime_map, plot_title, plot_path)
    logging.info(f"Saved plot to {plot_path}")


if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[3]
    data_dir = project_root / "data"
    output_dir = project_root / "src" / "strategy" / "ml" / "hmm"

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list(data_dir.glob("*.csv"))

    if not csv_files:
        logging.error(f"No CSV files found in {data_dir}. Please add data files.")
    else:
        for file in tqdm(csv_files, desc="Processing HMM Models"):
            process_file(file, output_dir)
        logging.info("--- All files processed. ---")
