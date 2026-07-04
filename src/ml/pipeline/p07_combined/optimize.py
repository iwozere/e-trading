from typing import Any

import numpy as np
import pandas as pd

from src.ml.pipeline.p07_combined.evaluator import P07Evaluator
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def objective(trial, ohlcv_clean: Any, timeframe: str = "15m", enable_mtf: bool = False):
    """
    Optuna objective function for p07_combined.
    Matches features, labels, and backtest metrics.

    Args:
        enable_mtf: When True, adds anchor-timeframe hyper-parameters to the search space.
    """
    # 1. Suggest Parameters
    params = {
        "rsi_period": trial.suggest_int("rsi_period", 7, 21),
        "bb_period": trial.suggest_int("bb_period", 10, 30),
        "bb_std": trial.suggest_float("bb_std", 1.5, 3.0),
        "atr_period": trial.suggest_int("atr_period", 10, 20),
        "vol_lookback": trial.suggest_int("vol_lookback", 10, 40),
        "pt_mult": trial.suggest_float("pt_mult", 0.5, 4.0),
        "sl_mult": trial.suggest_float("sl_mult", 0.25, 3.0),
        "tpl_hours": trial.suggest_float("tpl_hours", 1.0, 96.0),
        "buy_prob_min": trial.suggest_float("buy_prob_min", 0.35, 0.65),
        "sell_prob_min": trial.suggest_float("sell_prob_min", 0.35, 0.65),
        "max_depth": trial.suggest_int("max_depth", 3, 5),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "enable_mtf": enable_mtf,
    }

    if enable_mtf:
        params["anchor_ema_period"] = trial.suggest_int("anchor_ema_period", 10, 50)
        params["anchor_rsi_period"] = trial.suggest_int("anchor_rsi_period", 7, 21)
        params["anchor_bb_period"] = trial.suggest_int("anchor_bb_period", 10, 30)
        params["anchor_atr_period"] = trial.suggest_int("anchor_atr_period", 10, 20)
        params["regime_threshold"] = trial.suggest_float("regime_threshold", 0.00005, 0.001, log=True)

    # 2. Run Centralized Evaluation with timeframe awareness
    res = P07Evaluator.run_evaluation(ohlcv_clean, params, timeframe=timeframe)

    if "error" in res:
        return -1.0

    # 3. Objective Metric — score on val set ONLY; test set is never touched by Optuna
    sharpe = res["pf_val"].sharpe_ratio()
    total_trades = res["pf_val"].trades.count().sum()

    if pd.isna(sharpe) or total_trades < 20:
        return -1.0

    adjusted_score = sharpe * np.log10(max(total_trades, 2))

    return float(adjusted_score)
