import pandas as pd
import optuna
from src.ml.pipeline.p07_combined.evaluator import P07Evaluator
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

def objective(trial, ohlcv_clean: pd.DataFrame):
    """
    Optuna objective function for p07_combined.
    Matches features, labels, and backtest metrics.
    """
    # 1. Sugges Parameters
    params = {
        'rsi_period': trial.suggest_int('rsi_period', 7, 21),
        'bb_period': trial.suggest_int('bb_period', 10, 30),
        'pt_mult': trial.suggest_float('pt_mult', 1.0, 3.0),
        'sl_mult': trial.suggest_float('sl_mult', 0.5, 2.0),
        'tpl_bars': trial.suggest_int('tpl_bars', 5, 24),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200)
    }

    # 2. Run Centralized Evaluation
    res = P07Evaluator.run_evaluation(ohlcv_clean, params)

    if "error" in res:
        return -1.0

    # 3. Objective Metric (Sharpe Ratio)
    sharpe = res["pf"].sharpe_ratio()
    if pd.isna(sharpe):
        return -1.0

    return float(sharpe)
