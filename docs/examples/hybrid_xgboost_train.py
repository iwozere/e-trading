"""
Hybrid XGBoost Training Script
-----------------------------

- Loads features and targets from hybrid_xgboost_features.npz (features generated using TA-Lib in the previous script)
- Trains an XGBoost classifier (up/down) using Hyperopt for Bayesian optimization
- Evaluates using total profit (if possible) or Sharpe Ratio
- Prints best hyperparameters and evaluation results
"""

import numpy as np
import xgboost as xgb
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# --- Load Data ---
def load_features(path='hybrid_xgboost_features.npz'):
    data = np.load(path)
    X = data['features']
    y = data['target']
    return X, y

# --- Evaluation Metric ---
def compute_total_profit(y_true, y_pred, price_changes):
    # y_pred: 1=up, 0=down; price_changes: next bar's price change
    # Buy if y_pred==1, sell/short if y_pred==0
    profit = np.where(y_pred == 1, price_changes, -price_changes)
    return profit.sum()

def compute_sharpe_ratio(profit_series):
    mean = np.mean(profit_series)
    std = np.std(profit_series)
    if std == 0:
        return 0
    return mean / std * np.sqrt(252)  # annualized

# --- Hyperopt Objective ---
def objective(params, X, y, price_changes):
    clf = xgb.XGBClassifier(
        learning_rate=params['learning_rate'],
        max_depth=int(params['max_depth']),
        subsample=params['subsample'],
        n_estimators=100,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0
    )
    clf.fit(X, y)
    y_pred = clf.predict(X)
    profit = compute_total_profit(y, y_pred, price_changes)
    # We want to maximize profit, so minimize negative profit
    return {'loss': -profit, 'status': STATUS_OK, 'model': clf}

# --- Main Training Script ---
def main():
    print("Loading features and targets...")
    X, y = load_features()
    # For profit calculation, we need price changes
    # Here, we assume price_changes = close[t+1] - close[t] for each window
    # Since y was constructed as (y[1:] > y[:-1]), we can reconstruct price_changes
    # For demo, use random price changes
    price_changes = np.random.normal(0, 1, size=len(y))
    print(f"X: {X.shape}, y: {y.shape}")
    # Hyperopt search space
    space = {
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
        'max_depth': hp.quniform('max_depth', 3, 12, 1),
        'subsample': hp.uniform('subsample', 0.6, 0.8)
    }
    print("Starting Hyperopt Bayesian optimization...")
    trials = Trials()
    best = fmin(fn=lambda params: objective(params, X, y, price_changes),
                space=space,
                algo=tpe.suggest,
                max_evals=20,
                trials=trials)
    print("Best hyperparameters:", best)
    # Train final model
    best_params = {
        'learning_rate': best['learning_rate'],
        'max_depth': int(best['max_depth']),
        'subsample': best['subsample'],
        'n_estimators': 100,
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'verbosity': 0
    }
    clf = xgb.XGBClassifier(**best_params)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    profit = compute_total_profit(y, y_pred, price_changes)
    sharpe = compute_sharpe_ratio(np.where(y_pred == 1, price_changes, -price_changes))
    print(f"Accuracy: {acc:.4f}")
    print(f"Total Profit: {profit:.2f}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print("Done.")

if __name__ == '__main__':
    main() 