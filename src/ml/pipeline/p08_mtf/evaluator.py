import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path

from src.ml.pipeline.p08_mtf.features import P08FeatureEngine
from src.ml.pipeline.p07_combined.labeling import get_triple_barrier_labels
from src.ml.pipeline.p07_combined.models import P07XGBModel
from src.ml.pipeline.p07_combined.evaluator import P07Evaluator
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

class P08Evaluator(P07Evaluator):
    """
    MTF-Aware Evaluator for P08.
    Uses P08FeatureEngine for trend-aware signals.
    """

    @staticmethod
    def prepare_data(ohlcv: Any, params: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Calculates MTF features and labels.
        Supports both single DF and list of DFs (for gap-aware processing).
        """
        # If ohlcv is a list, process each segment independently to prevent gap leakage
        if isinstance(ohlcv, list):
            all_X = []
            all_y = []
            for segment in ohlcv:
                if segment.empty: continue
                X_seg = P08FeatureEngine.build_features(segment, params)
                y_seg = get_triple_barrier_labels(segment, params)

                # Align X and y
                common_idx = X_seg.index.intersection(y_seg.index)
                all_X.append(X_seg.loc[common_idx])
                all_y.append(y_seg.loc[common_idx])

            if not all_X: return pd.DataFrame(), pd.Series()

            X_f = pd.concat(all_X).sort_index()
            y_f = pd.concat(all_y).sort_index()
            # Drop duplicates if any overlap
            X_f = X_f.loc[~X_f.index.duplicated(keep='last')]
            y_f = y_f.loc[~y_f.index.duplicated(keep='last')]
            return X_f, y_f
        else:
            # Single segment processing
            X_f = P08FeatureEngine.build_features(ohlcv, params)
            y_f = get_triple_barrier_labels(ohlcv, params)
            common_idx = X_f.index.intersection(y_f.index)
            return X_f.loc[common_idx], y_f.loc[common_idx]

    @classmethod
    def run_evaluation(cls, ohlcv: Any, params: Dict[str, Any], timeframe: str = "15m") -> Dict[str, Any]:
        """Runs the full MTF training and backtest loop."""
        try:
            X_f, y_f = cls.prepare_data(ohlcv, params)
            if X_f.empty or len(X_f) < 50:
                return {"error": "Insufficient data after feature/label alignment"}

            # 70/30 Time-series split
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

            # Backtest
            vbt_freq = timeframe if timeframe != "d" else "1D"

            # Indexer must be from the full concatenated dataset
            if isinstance(ohlcv, list):
                ohlcv_full = pd.concat(ohlcv).sort_index()
                ohlcv_full = ohlcv_full.loc[~ohlcv_full.index.duplicated(keep='last')]
            else:
                ohlcv_full = ohlcv

            ohlcv_test = ohlcv_full.loc[X_test.index]

            pf = vbt.Portfolio.from_signals(
                ohlcv_test['close'],
                signals == 1,
                signals == -1,
                freq=vbt_freq,
                fees=0.001,
                init_cash=10000
            )

            return {
                "model": model,
                "pf": pf,
                "metrics": pf.stats(),
                "trades": pf.trades.records_readable,
                "X_test": X_test,
                "y_test": y_test,
                "y_f": y_f,
                "ohlcv_test": ohlcv_test
            }

        except Exception as e:
            _logger.error("Evaluation failed: %s", str(e), exc_info=True)
            return {"error": str(e)}
