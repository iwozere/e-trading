# File: scripts/train_hmm.py

import pandas as pd
import numpy as np
import os
import argparse
import json
import joblib
import optuna

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from feature_engineering import generate_features

HiddenMarkovModel = None
NormalDistribution = None
DEFAULT_FEATURES = ["log_return", "volatility"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Input OHLCV CSV file")
    parser.add_argument("--timeframe", required=True, help="Base name for output")
    parser.add_argument("--optimize", action="store_true", help="Use Optuna to optimize")
    parser.add_argument("--n_trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument("--features", nargs="*", default=DEFAULT_FEATURES, help="Features to include")
    parser.add_argument("--backend", choices=["gaussian", "pomegranate"], default="gaussian")
    return parser.parse_args()


def objective(trial, df, features):
    params = {
        "n_components": trial.suggest_int("n_components", 2, 4),
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

    X, _ = generate_features(df.copy(), features, params)
    X = StandardScaler().fit_transform(X)

    if args.backend == "pomegranate":
        model = HiddenMarkovModel.from_samples(NormalDistribution, n_components=params["n_components"], X=X)
        hidden_states = model.predict(X)
    else:
        model = GaussianHMM(n_components=params["n_components"], covariance_type="full", n_iter=1000)
        model.fit(X)
        hidden_states = model.predict(X)

    regime_means = []
    for state in np.unique(hidden_states):
        regime_means.append(X[hidden_states == state][:, 0].mean())

    score = -np.std(regime_means)
    return score


def train_model(df, features, params, backend="gaussian"):
    X, df_full = generate_features(df.copy(), features, params)
    X = StandardScaler().fit_transform(X)

    if backend == "gaussian":
        model = GaussianHMM(n_components=params["n_components"], covariance_type="full", n_iter=1000)
        model.fit(X)
        states = model.predict(X)
    elif backend == "pomegranate":
        if HiddenMarkovModel is None:
            raise ImportError("pomegranate is not installed. Use 'pip install pomegranate'")
        model = HiddenMarkovModel.from_samples(NormalDistribution, n_components=params["n_components"], X=X)
        states = model.predict(X)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    df_full = df_full.iloc[-len(states):].copy()
    df_full["regime"] = states
    return model, df_full


def main():
    global args
    args = parse_args()

    df = pd.read_csv(args.csv, parse_dates=["timestamp"])
    features = args.features

    if args.optimize:
        study = optuna.create_study()
        study.optimize(lambda trial: objective(trial, df, features), n_trials=args.n_trials)
        best_params = study.best_params
    else:
        best_params = {
            "n_components": 3,
            "vol_window": 20,
            "rsi_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "boll_window": 20
        }

    model, labeled_df = train_model(df, features, best_params, backend=args.backend)

    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    base = args.timeframe
    labeled_df.to_csv(f"results/{base}.csv", index=False)
    joblib.dump(model, f"models/{base}.pkl")

    with open(f"results/{base}.json", "w") as f:
        json.dump(best_params, f, indent=2)


if __name__ == "__main__":
    main()
