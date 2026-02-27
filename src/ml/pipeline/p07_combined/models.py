import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Dict, Any

class P07XGBModel:
    """
    XGBoost Classifier wrapper for p07_combined.
    Optimized for CPU and ONNX compatibility.
    """

    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'objective': 'multi:softprob',
            'num_class': 3, # [-1, 0, 1] mapped to [0, 1, 2]
            'tree_method': 'hist', # CPU optimized
            'eval_metric': 'mlogloss',
            'random_state': 42
        }
        if params:
            default_params.update(params)
        self.params = default_params
        self.model = None

    def _map_labels(self, y: pd.Series) -> pd.Series:
        """Map [-1, 0, 1] to [0, 1, 2]."""
        return y.map({-1: 0, 0: 1, 1: 2})

    def fit(self, X: pd.DataFrame, y: pd.Series):
        y_mapped = self._map_labels(y)
        dtrain = xgb.DMatrix(X, label=y_mapped)

        # xgb.train uses num_boost_round, not n_estimators in params dict
        params = self.params.copy()
        n_estimators = params.pop('n_estimators', 100)

        self.model = xgb.train(params, dtrain, num_boost_round=n_estimators)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

    def predict_signal(self, X: pd.DataFrame, thresholds: Dict[str, float] = None) -> pd.Series:
        """
        Produce discrete signals [-1, 0, 1] with confidence thresholds.
        """
        probs = self.predict_proba(X)
        # probs order: [0=Sell, 1=Hold, 2=Buy]

        buy_threshold = thresholds.get('buy_prob_min', 0.5) if thresholds else 0.5
        sell_threshold = thresholds.get('sell_prob_min', 0.5) if thresholds else 0.5

        signals = pd.Series(0, index=X.index)

        # Buy: prob[2] > threshold
        signals[probs[:, 2] > buy_threshold] = 1
        # Sell: prob[0] > threshold
        signals[probs[:, 0] > sell_threshold] = -1

        return signals

    def save_model(self, path: str):
        self.model.save_model(path)
