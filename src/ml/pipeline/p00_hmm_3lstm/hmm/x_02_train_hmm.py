"""
HMM Market Regime Training and Optimization Pipeline.

This script provides a command-line interface to train a Hidden Markov Model (HMM)
for identifying market regimes from financial time series data. It supports
feature engineering, optional hyperparameter optimization using Optuna, and two
different HMM library backends.

Key Features:
- **Configurable Features**: Uses a separate `feature_engineering.py` module to
  allow for a flexible set of technical indicators to be used as model inputs.
- **Hyperparameter Optimization**: Includes an `--optimize` flag to run an Optuna
  study. This tunes both the number of HMM states and the parameters for the
  selected features (e.g., RSI period, volatility window).
- **Dual Backend Support**: Allows training with either the standard `hmmlearn`
  library (`GaussianHMM`) or the modern `pomegranate` library.
- **Artifact Generation**: Saves three key outputs for each run:
  1. A CSV file with the original data plus the predicted 'regime' column.
  2. A pickled model file (`.pkl`) for later use.
  3. A JSON file containing the exact hyperparameters used for training.

Workflow:
1.  Parse command-line arguments to determine input file, features, backend, etc.
2.  If optimization is enabled, run an Optuna study to find the best set of
    hyperparameters based on the objective function.
3.  If optimization is disabled, use a predefined set of default parameters.
4.  Train a final HMM using the selected parameters and backend.
5.  Save the trained model, the labeled DataFrame, and the parameters to disk.

Usage:
  # Train with default settings (3 states, GaussianHMM)
  python your_script_name.py --csv data/btcusdt.csv --timeframe btcusdt_15m

  # Run Optuna optimization to find the best number of states and feature params
  python your_script_name.py --csv data/btcusdt.csv --timeframe btcusdt_15m_opt --optimize --n_trials 50

  # Use the pomegranate backend with specific features
  python your_script_name.py --csv data/ethusdt.csv --timeframe ethusdt_1h_pome --backend pomegranate --features rsi macd
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3] # Go up 3 levels from 'src/ml/hmm'
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import os
import argparse
import json
import joblib
import optuna

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

# Assumes feature_engineering.py is in the same directory or accessible via PYTHONPATH
from x_01_feature_engineering import generate_features

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

# For optional import of pomegranate
HiddenMarkovModel = None
NormalDistribution = None
DEFAULT_FEATURES = ["log_return", "volatility"]


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the HMM training script.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train an HMM for market regime detection.")
    parser.add_argument("--csv", required=True, help="Path to the input OHLCV CSV file.")
    parser.add_argument("--timeframe", required=True, help="A base name for output files (e.g., 'btcusdt_1h').")
    parser.add_argument("--optimize", action="store_true", help="Enable hyperparameter optimization with Optuna.")
    parser.add_argument("--n_trials", type=int, default=30, help="Number of Optuna trials to run if --optimize is set.")
    parser.add_argument("--features", nargs="*", default=DEFAULT_FEATURES, help="A list of features to use for the model.")
    parser.add_argument("--backend", choices=["gaussian", "pomegranate"], default="gaussian", help="The HMM library to use.")
    return parser.parse_args()


def objective(trial: optuna.trial.Trial, df: pd.DataFrame, features: list[str]) -> float:
    """
    Defines a single Optuna trial to find the best hyperparameters.

    This function suggests hyperparameters for both the HMM (number of states)
    and the feature engineering process. It then generates features, trains a
    temporary HMM, and computes a score. The scoring metric is the negative
    standard deviation of the mean log returns of each regime. Optuna aims to
    maximize this score, effectively minimizing the standard deviation.

    Args:
        trial (optuna.trial.Trial): The Optuna trial object.
        df (pd.DataFrame): The raw input DataFrame.
        features (list[str]): The list of features being used in this study.

    Returns:
        float: The score for this set of hyperparameters. A higher score is better.
    """
    # Suggest HMM and feature engineering parameters
    params = {
        "n_components": 3,
        "vol_window": trial.suggest_int("vol_window", 5, 50)
    }
    if "rsi" in features:
        params["rsi_period"] = trial.suggest_int("rsi_period", 10, 30)
    if "macd" in features:
        params["macd_fast"] = trial.suggest_int("macd_fast", 8, 16)
        params["macd_slow"] = trial.suggest_int("macd_slow", 20, 30)
        params["macd_signal"] = trial.suggest_int("macd_signal", 5, 15)
    if "boll" in features:
        params["boll_window"] = trial.suggest_int("boll_window", 10, 30)

    # Generate features and scale them
    X, _ = generate_features(df.copy(), features, params)
    X = StandardScaler().fit_transform(X)

    # Fit a model with the trial parameters
    if args.backend == "pomegranate":
        model = HiddenMarkovModel.from_samples(NormalDistribution, n_components=params["n_components"], X=[X])
        hidden_states = model.predict(X)
    else: # Default to gaussian
        model = GaussianHMM(n_components=params["n_components"], covariance_type="full", n_iter=100)
        model.fit(X)
        hidden_states = model.predict(X)

    # Calculate the score based on regime mean separation
    regime_means = [X[hidden_states == state][:, 0].mean() for state in np.unique(hidden_states)]

    if len(regime_means) < params["n_components"]:
        return -1e9 # Penalize if model collapses states

    score = -np.std(regime_means)
    return score


def train_model(df: pd.DataFrame, features: list[str], params: dict, backend: str = "gaussian") -> tuple[object, pd.DataFrame]:
    """
    Trains a final HMM using the specified parameters and backend.

    This function generates features, scales the data, fits the HMM, predicts
    the state sequence, and appends it to the DataFrame.

    Args:
        df (pd.DataFrame): The raw input DataFrame.
        features (list[str]): The list of feature names to generate.
        params (dict): A dictionary of the final hyperparameters for features and the model.
        backend (str): The HMM library to use, either "gaussian" or "pomegranate".

    Raises:
        ImportError: If the 'pomegranate' backend is chosen but the library is not installed.
        ValueError: If an unsupported backend is specified.

    Returns:
        tuple[object, pd.DataFrame]: A tuple containing:
            - The trained model object.
            - A DataFrame with the 'regime' column added and NaN rows dropped.
    """
    X, df_full = generate_features(df.copy(), features, params)
    X = StandardScaler().fit_transform(X)

    if backend == "gaussian":
        model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000)
        model.fit(X)
        states = model.predict(X)
    elif backend == "pomegranate":
        if HiddenMarkovModel is None:
            raise ImportError("pomegranate is not installed. Run 'pip install pomegranate' to use this backend.")
        model = HiddenMarkovModel.from_samples(NormalDistribution, n_components=params["n_components"], X=[X])
        states = model.predict(X)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    # Align the states with the feature DataFrame (which had NaNs dropped)
    df_full = df_full.iloc[-len(states):].copy()
    df_full["regime"] = states
    return model, df_full


def main():
    """
    Main execution function for the HMM training pipeline.

    Orchestrates the process of parsing arguments, running optimization (or using
    defaults), training the final model, and saving all resulting artifacts.
    """
    global args, HiddenMarkovModel, NormalDistribution
    args = parse_args()

    # Lazily import pomegranate only if needed to avoid mandatory installation
    if args.backend == "pomegranate":
        try:
            from pomegranate.hmm import HiddenMarkovModel
            from pomegranate.distributions import MultivariateGaussianDistribution as NormalDistribution
        except ImportError:
            _logger.error("'pomegranate' backend was chosen but the library is not installed.")
            _logger.error("Please run: pip install pomegranate")
            return

    df = pd.read_csv(args.csv, parse_dates=["timestamp"])
    features = args.features

    if args.optimize:
        _logger.info("Running Optuna optimization...")
        study = optuna.create_study()
        study.optimize(lambda trial: objective(trial, df, features), n_trials=args.n_trials)
        best_params = study.best_params
        _logger.info("Optimization complete. Best parameters found: %s", best_params)
    else:
        _logger.info("Using default parameters...")
        best_params = {
            "n_components": 3,
            "vol_window": 20,
            "rsi_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "boll_window": 20
        }

    # Train the final model with the determined parameters
    _logger.info("Training final model...")
    model, labeled_df = train_model(df, features, best_params, backend=args.backend)
    _logger.info("Model training complete.")

    # Create directories and save artifacts
    os.makedirs("results", exist_ok=True)

    base = args.timeframe
    csv_out_path = f"results/HMM_{base}.csv"
    model_out_path = f"results/HMM_{base}.pkl"
    json_out_path = f"results/HMM_{base}.json"

    labeled_df.to_csv(csv_out_path, index=False)
    joblib.dump(model, model_out_path)
    with open(json_out_path, "w") as f:
        json.dump(best_params, f, indent=2)

    _logger.info("\nArtifacts saved successfully:")
    _logger.info("  - Labeled Data: %s")
    _logger.info("  - Model: %s")
    _logger.info("  - Parameters: %s")


if __name__ == "__main__":
    main()
