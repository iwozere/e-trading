from typing import Any, Dict, Tuple

import pandas as pd

import vectorbt as vbt
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
    def prepare_data(ohlcv: Any, params: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]:
        """Align features and labels across potentially multiple continuous segments."""
        feature_config = {
            "rsi_period": params.get("rsi_period", 14),
            "bb_period": params.get("bb_period", 20),
            "bb_std": params.get("bb_std", 2.0),
            "atr_period": params.get("atr_period", 14),
            "vol_lookback": params.get("vol_lookback", 20),
            "anchor_ema_period": params.get("anchor_ema_period", 20),
            "anchor_rsi_period": params.get("anchor_rsi_period", 14),
            "anchor_bb_period": params.get("anchor_bb_period", 20),
            "anchor_atr_period": params.get("anchor_atr_period", 14),
            "regime_threshold": params.get("regime_threshold", 0.0001),
        }
        enable_mtf = params.get("enable_mtf", False)

        # If ohlcv is a single DF, make it a list for uniform processing
        segments = [ohlcv] if isinstance(ohlcv, pd.DataFrame) else ohlcv

        Xs, ys = [], []
        for seg in segments:
            X_seg = build_features(seg, feature_config, enable_mtf=enable_mtf)
            y_seg = get_triple_barrier_labels(
                seg,
                pt_mult=params.get("pt_mult", 2.0),
                sl_mult=params.get("sl_mult", 1.0),
                tpl_bars=params.get("tpl_bars", 12),
                atr_period=params.get("atr_period", 14),
            )

            common_idx = X_seg.index.intersection(y_seg.index)
            Xs.append(X_seg.loc[common_idx])
            ys.append(y_seg.loc[common_idx])

        if not Xs:
            return pd.DataFrame(), pd.Series()

        return pd.concat(Xs).sort_index(), pd.concat(ys).sort_index()

    @staticmethod
    def hours_to_bars(hours: float, timeframe: str) -> int:
        """Convert hours to number of bars based on timeframe."""
        # Normalize timeframe (e.g., '1h' -> '60m', '4h' -> '240m')
        tf_map = {"5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "d": 1440}
        minutes_per_bar = tf_map.get(timeframe.lower(), 15)
        bars = int((hours * 60) / minutes_per_bar)
        return max(1, bars)

    @classmethod
    def run_evaluation(
        cls, ohlcv: "pd.DataFrame | list[pd.DataFrame]", params: Dict[str, Any], timeframe: str = "15m"
    ) -> Dict[str, Any]:
        """
        Full pipeline: Features -> Train -> Predict -> Backtest.

        Splits raw OHLCV FIRST into 60/20/20 train/val/test segments with a
        tpl_bars-wide buffer between each boundary to eliminate triple-barrier
        label leakage.  Optuna callers must use pf_val; save_artifacts must
        use pf_test for all reported metrics.
        """
        # 1. Adaptive Window Calculation
        tpl_hours = params.get("tpl_hours", params.get("tpl_bars", 12) * 15 / 60)
        tpl_bars = cls.hours_to_bars(tpl_hours, timeframe)

        params_with_bars = params.copy()
        params_with_bars["tpl_bars"] = tpl_bars

        # 2. Resolve raw OHLCV to a single sorted DataFrame
        if isinstance(ohlcv, list):
            ohlcv_full = pd.concat(ohlcv).sort_index()
            ohlcv_full = ohlcv_full.loc[~ohlcv_full.index.duplicated(keep="last")]
        else:
            ohlcv_full = ohlcv

        n = len(ohlcv_full)
        train_end = int(n * 0.60) - tpl_bars
        val_end = int(n * 0.80) - tpl_bars

        if train_end < 50 or val_end <= train_end or val_end >= n:
            return {"error": "Insufficient data for 3-way split"}

        ohlcv_train = ohlcv_full.iloc[:train_end]
        ohlcv_val = ohlcv_full.iloc[train_end + tpl_bars : val_end]
        ohlcv_test = ohlcv_full.iloc[val_end + tpl_bars :]

        # 3. Compute features + labels independently per segment (no leakage)
        X_train, y_train = cls.prepare_data(ohlcv_train, params_with_bars)
        X_val, y_val = cls.prepare_data(ohlcv_val, params_with_bars)
        X_test, y_test = cls.prepare_data(ohlcv_test, params_with_bars)

        if len(X_train) < 50 or len(X_val) < 10:
            return {"error": "Insufficient data samples after split"}

        # 4. Train Model on train segment only
        xgb_params = {
            "max_depth": params.get("max_depth", 6),
            "learning_rate": params.get("learning_rate", 0.1),
            "n_estimators": params.get("n_estimators", 100),
        }
        model = P07XGBModel(params=xgb_params)
        model.fit(X_train, y_train)

        thresholds = {
            "buy_prob_min": params.get("buy_prob_min", 0.5),
            "sell_prob_min": params.get("sell_prob_min", 0.5),
        }

        vbt_freq = timeframe if timeframe != "d" else "1D"

        def _backtest(X: pd.DataFrame, ohlcv_seg: pd.DataFrame) -> Tuple[vbt.Portfolio, pd.Series]:
            sigs = model.predict_signal(X, thresholds=thresholds)
            prices = ohlcv_seg.loc[X.index, "close"]
            return vbt.Portfolio.from_signals(
                prices, sigs == 1, sigs == -1, fees=0.001, slippage=0.0005, freq=vbt_freq, direction="both"
            ), sigs

        pf_val, signals_val = _backtest(X_val, ohlcv_val)
        pf_test, signals_test = _backtest(X_test, ohlcv_test)

        return {
            "model": model,
            # val portfolio — for Optuna objective scoring
            "pf_val": pf_val,
            "signals_val": signals_val,
            "X_val": X_val,
            "y_val": y_val,
            # test portfolio — true OOS, for save_artifacts only
            "pf_test": pf_test,
            "signals": signals_test,
            "X_test": X_test,
            "y_test": y_test,
            # train segment kept for diagnostics
            "X_train": X_train,
            "y_train": y_train,
            "ohlcv_test": ohlcv_test,
            # legacy alias so callers that used "pf" still get the test portfolio
            "pf": pf_test,
            "metrics": pf_test.stats(),
            "trades": pf_test.trades.records_readable,
        }

    @classmethod
    def evaluate_model(
        cls,
        model: P07XGBModel,
        ohlcv: "pd.DataFrame | list[pd.DataFrame]",
        params: Dict[str, Any],
        timeframe: str = "15m",
        init_cash: float = 100.0,
    ) -> Dict[str, Any]:
        """
        Evaluates a pre-trained model on a dataset without re-training.
        """
        # 1. Adaptive Window Calculation
        tpl_hours = params.get("tpl_hours", params.get("tpl_bars", 12) * 15 / 60)
        tpl_bars = cls.hours_to_bars(tpl_hours, timeframe)

        params_with_bars = params.copy()
        params_with_bars["tpl_bars"] = tpl_bars

        # For generalization, we might use the whole OHLCV or just a segment.
        # We'll use the whole thing as "test" data.
        X_test, y_test = cls.prepare_data(ohlcv, params_with_bars)

        if len(X_test) < 10:
            return {"error": "Insufficient data samples for generalization"}

        # Generate Signals
        thresholds = {
            "buy_prob_min": params.get("buy_prob_min", 0.5),
            "sell_prob_min": params.get("sell_prob_min", 0.5),
        }
        signals = model.predict_signal(X_test, thresholds=thresholds)

        # Backtest
        vbt_freq = timeframe if timeframe != "d" else "1D"

        # Handle list of segments for indexing
        if isinstance(ohlcv, list):
            ohlcv_full = pd.concat(ohlcv).sort_index()
            # Drop duplicates if any across files
            ohlcv_full = ohlcv_full.loc[~ohlcv_full.index.duplicated(keep="last")]
        else:
            ohlcv_full = ohlcv

        ohlcv_test = ohlcv_full.loc[X_test.index]

        pf = vbt.Portfolio.from_signals(
            ohlcv_test["close"],
            signals == 1,
            signals == -1,
            init_cash=init_cash,
            fees=0.001,
            slippage=0.0005,
            freq=vbt_freq,
            direction="both",
        )

        return {
            "pf": pf,
            "metrics": pf.stats(),
            "test_start": str(ohlcv_test.index.min()),
            "test_end": str(ohlcv_test.index.max()),
            "num_trades": pf.trades.count().sum(),
        }
