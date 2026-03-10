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
                y_seg = get_triple_barrier_labels(
                    segment,
                    pt_mult=params.get('pt_mult', 2.0),
                    sl_mult=params.get('sl_mult', 1.0),
                    tpl_bars=params.get('tpl_bars', 12),
                    atr_period=params.get('atr_period', 14)
                )

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
            y_seg = get_triple_barrier_labels(
                ohlcv,
                pt_mult=params.get('pt_mult', 2.0),
                sl_mult=params.get('sl_mult', 1.0),
                tpl_bars=params.get('tpl_bars', 12),
                atr_period=params.get('atr_period', 14)
            )
            common_idx = X_f.index.intersection(y_seg.index)
            return X_f.loc[common_idx], y_seg.loc[common_idx]

