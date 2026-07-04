from typing import Any, Dict, Tuple

import pandas as pd

from src.ml.pipeline.p07_combined.evaluator import P07Evaluator
from src.ml.pipeline.p07_combined.labeling import get_triple_barrier_labels
from src.ml.pipeline.p08_mtf.features import P08FeatureEngine
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class P08Evaluator(P07Evaluator):
    """
    MTF-Aware Evaluator for P08.
    Uses P08FeatureEngine for trend-aware signals.
    Inherits the 3-way split logic from P07Evaluator.run_evaluation().
    """

    @staticmethod
    def prepare_data(ohlcv: Any, params: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Calculates MTF features and labels.
        Supports both single DF and list of DFs (for gap-aware processing).
        """
        if isinstance(ohlcv, list):
            all_X = []
            all_y = []
            for segment in ohlcv:
                if segment.empty:
                    continue
                X_seg = P08FeatureEngine.build_features(segment, params)
                y_seg = get_triple_barrier_labels(
                    segment,
                    pt_mult=params.get("pt_mult", 2.0),
                    sl_mult=params.get("sl_mult", 1.0),
                    tpl_bars=params.get("tpl_bars", 12),
                    atr_period=params.get("atr_period", 14),
                )
                common_idx = X_seg.index.intersection(y_seg.index)
                all_X.append(X_seg.loc[common_idx])
                all_y.append(y_seg.loc[common_idx])

            if not all_X:
                return pd.DataFrame(), pd.Series()

            X_f = pd.concat(all_X).sort_index()
            y_f = pd.concat(all_y).sort_index()
            X_f = X_f.loc[~X_f.index.duplicated(keep="last")]
            y_f = y_f.loc[~y_f.index.duplicated(keep="last")]
            return X_f, y_f
        else:
            X_f = P08FeatureEngine.build_features(ohlcv, params)
            y_seg = get_triple_barrier_labels(
                ohlcv,
                pt_mult=params.get("pt_mult", 2.0),
                sl_mult=params.get("sl_mult", 1.0),
                tpl_bars=params.get("tpl_bars", 12),
                atr_period=params.get("atr_period", 14),
            )
            common_idx = X_f.index.intersection(y_seg.index)
            return X_f.loc[common_idx], y_seg.loc[common_idx]
