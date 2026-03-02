import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict, Any, Tuple
from pathlib import Path

from src.ml.pipeline.p07_combined.features import build_features
from src.ml.pipeline.p07_combined.labeling import get_triple_barrier_labels
from src.ml.pipeline.p07_combined.models import P07XGBModel
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

class P07Evaluator:
    """
    Shared logic for model training, signal generation, and backtesting.
    Ensures parity between Optimization and Final Artifact generation.
    """

    @staticmethod
    def prepare_data(ohlcv: pd.DataFrame, params: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]:
        """Align features and labels."""
        feature_config = {
            'rsi_period': params.get('rsi_period', 14),
            'bb_period': params.get('bb_period', 20),
            'bb_std': params.get('bb_std', 2.0),
            'atr_period': params.get('atr_period', 14)
        }

        X = build_features(ohlcv, feature_config)
        y = get_triple_barrier_labels(
            ohlcv,
            pt_mult=params.get('pt_mult', 2.0),
            sl_mult=params.get('sl_mult', 1.0),
            tpl_bars=params.get('tpl_bars', 12),
            atr_period=params.get('atr_period', 14)
        )

        common_idx = X.index.intersection(y.index)
        return X.loc[common_idx], y.loc[common_idx]

    @staticmethod
    def hours_to_bars(hours: float, timeframe: str) -> int:
        """Convert hours to number of bars based on timeframe."""
        # Normalize timeframe (e.g., '1h' -> '60m', '4h' -> '240m')
        tf_map = {
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '4h': 240,
            'd': 1440
        }
        minutes_per_bar = tf_map.get(timeframe.lower(), 15)
        bars = int((hours * 60) / minutes_per_bar)
        return max(1, bars)

    @classmethod
    def run_evaluation(cls, ohlcv: pd.DataFrame, params: Dict[str, Any], timeframe: str = "15m") -> Dict[str, Any]:
        """
        Full pipeline: Features -> Train -> Predict -> Backtest.
        Adaptive Window: Uses tpl_hours converted to bars.
        """
        # 1. Adaptive Window Calculation
        tpl_hours = params.get('tpl_hours', params.get('tpl_bars', 12) * 15 / 60) # Fallback for old trials
        tpl_bars = cls.hours_to_bars(tpl_hours, timeframe)

        # Inject tpl_bars for prepare_data
        params_with_bars = params.copy()
        params_with_bars['tpl_bars'] = tpl_bars

        X_f, y_f = cls.prepare_data(ohlcv, params_with_bars)

        if len(X_f) < 100:
            return {"error": "Insufficient data samples"}

        # 70/30 Split
        split_idx = int(len(X_f) * 0.7)
        X_train, y_train = X_f[:split_idx], y_f[:split_idx]
        X_test, y_test = X_f[split_idx:], y_f[split_idx:]

        # Train Model
        xgb_params = {
            'max_depth': params.get('max_depth', 6),
            'learning_rate': params.get('learning_rate', 0.1),
            'n_estimators': params.get('n_estimators', 100)
        }
        model = P07XGBModel(params=xgb_params)
        model.fit(X_train, y_train)

        # Generate Signals
        thresholds = {
            'buy_prob_min': params.get('buy_prob_min', 0.5),
            'sell_prob_min': params.get('sell_prob_min', 0.5)
        }
        signals = model.predict_signal(X_test, thresholds=thresholds)

        # Backtest - Dynamic Frequency
        # Map p07 timeframe to VectorBT frequency
        vbt_freq = timeframe if timeframe != "d" else "1D"

        ohlcv_test = ohlcv.loc[X_test.index]
        pf = vbt.Portfolio.from_signals(
            ohlcv_test['close'],
            signals == 1,
            signals == -1,
            fees=0.001,
            slippage=0.0005,
            freq=vbt_freq,
            direction='both'
        )

        return {
            "model": model,
            "pf": pf,
            "signals": signals,
            "X_test": X_test,
            "y_test": y_test,
            "X_train": X_train,
            "y_train": y_train,
            "y_f": y_f,
            "ohlcv_test": ohlcv_test,
            "metrics": pf.stats(),
            "trades": pf.trades.records_readable
        }
