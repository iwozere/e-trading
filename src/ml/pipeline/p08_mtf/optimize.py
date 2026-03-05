import numpy as np
import pandas as pd
import optuna
from typing import Any
from src.ml.pipeline.p08_mtf.evaluator import P08Evaluator
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

def objective(trial, ohlcv_clean: Any, timeframe: str = "15m"):
    """
    Optuna objective function for p08_mtf.
    Supports MTF features and robust scoring.
    """
    # 1. Suggest Parameters (Local + Anchor)
    params = {
        # Local indicators
        'rsi_period': trial.suggest_int('rsi_period', 7, 21),
        'bb_period': trial.suggest_int('bb_period', 10, 30),

        # Anchor indicators
        'anchor_ema_period': trial.suggest_int('anchor_ema_period', 10, 50),
        'anchor_rsi_period': trial.suggest_int('anchor_rsi_period', 7, 21),

        # Barriers / Labels
        'pt_mult': trial.suggest_float('pt_mult', 0.5, 4.0),
        'sl_mult': trial.suggest_float('sl_mult', 0.25, 3.0),
        'tpl_hours': trial.suggest_float('tpl_hours', 1.0, 96.0),

        # Confidence thresholds
        'buy_prob_min': trial.suggest_float('buy_prob_min', 0.35, 0.65),
        'sell_prob_min': trial.suggest_float('sell_prob_min', 0.35, 0.65),

        # Model Hyperparams
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200)
    }

    # 2. Run Centralized Evaluation
    res = P08Evaluator.run_evaluation(ohlcv_clean, params, timeframe=timeframe)

    if "error" in res:
        return -1.0

    # 3. Objective Metric (Adjusted Sharpe)
    sharpe = res["pf"].sharpe_ratio()
    total_trades = res["pf"].trades.count().sum()

    if pd.isna(sharpe) or total_trades < 10:
        return -1.0

    # Reward more trades (up to a point) using log multiplier
    adjusted_score = sharpe * np.log10(total_trades)

    return float(adjusted_score)
