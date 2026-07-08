# pyright: reportAttributeAccessIssue=false, reportCallIssue=false
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


from src.ml.pipeline.p07_combined.models import P07XGBModel
from src.ml.pipeline.p08_mtf.features import P08FeatureEngine
from src.notification.logger import setup_logger
from src.strategy.base_strategy import BaseStrategy

_logger = setup_logger(__name__)


class P08XGBStrategy(BaseStrategy):
    """
    P08 MTF Strategy (XGBoost).

    This strategy loads a pre-trained XGBoost model from the p08_mtf pipeline,
    computes multi-timeframe features bar-by-bar, and issues BUY/SELL
    signals depending on configured thresholds.
    """

    def __init__(self):
        super().__init__()

        # Strategy config
        self.model_path = self.config.get("model_path")
        self.buy_prob_min = self.config.get("buy_prob_min", 0.5)
        self.sell_prob_min = self.config.get("sell_prob_min", 0.5)
        self.feature_lookback = self.config.get("feature_lookback", 100)
        self.feature_cols = self.config.get("feature_columns", [])

        self.model = None
        self._initialize_strategy()

        # Buffer for calculating features dynamically
        self.df_buffer = []

    def _initialize_strategy(self):
        """Load the pre-trained P08 model model."""
        try:
            if not self.model_path:
                _logger.warning("No model_path provided for P08XGBStrategy")
                return

            model_file = Path(self.model_path)
            if not model_file.exists():
                _logger.error(f"Model file not found: {model_file}")
                return

            # Initialize XGB model wrapper
            self.model = P07XGBModel()
            self.model.load(str(model_file))
            _logger.info(f"Successfully loaded P08 XGBoost model from {model_file}")

            # If feature cols weren't explicitly supplied, try to extract from model if possible
            if not self.feature_cols and hasattr(self.model, "booster"):
                try:
                    self.feature_cols = self.model.booster.feature_names
                    _logger.info(f"Extracted {len(self.feature_cols)} feature columns from model")
                except Exception as e:
                    _logger.warning(f"Could not extract feature names from XGB model: {e}")

        except Exception:
            _logger.exception("Error loading P08 XGBoost model:")
            raise

    def next(self):
        # BaseStrategy.next() isn't heavily used if we just override this,
        # but we need to do basic equity tracking from base strategy if needed.
        super().next()

    def _execute_strategy_logic(self):
        """Execute the bar-by-bar ML predictions and entry/exit."""
        if not self.model:
            return  # Model not loaded yet

        # 1. Build the current bar's data tuple
        # P08 expects 'open', 'high', 'low', 'close', 'volume'.
        current_bar = {
            "datetime": self.data.datetime.datetime(0),
            "open": self.data.open[0],
            "high": self.data.high[0],
            "low": self.data.low[0],
            "close": self.data.close[0],
            "volume": self.data.volume[0],
        }

        # In a real multi-timeframe scenario, we might also get anchor_close, vix, btc_mc
        # from other data lines. We try to read them if added to cerebro.
        # However, paper trading bot might just run on a single timeframe with standard OHLCV
        # If macro indicators are unavailable, the FeatureEngine falls back gracefully.
        for attr in ["anchor_open", "anchor_high", "anchor_low", "anchor_close", "anchor_volume", "vix", "btc_mc"]:
            if hasattr(self.data, attr):
                current_bar[attr] = getattr(self.data, attr)[0]

        self.df_buffer.append(current_bar)

        # Keep buffer length manageable
        if len(self.df_buffer) > self.feature_lookback:
            self.df_buffer.pop(0)

        # We need a decent amount of bars to calculate indicators like EMA/BBANDS
        if len(self.df_buffer) < max(20, self.config.get("anchor_ema_period", 20)):
            return

        # 2. Build Features using the engine
        df = pd.DataFrame(self.df_buffer).set_index("datetime")
        try:
            # We pass the strategy config into the feature engine
            features = P08FeatureEngine.build_features(df, self.config)
        except Exception as e:
            _logger.debug(f"Feature calculation failed (likely not enough bars): {e}")
            return

        if len(features) == 0:
            return

        # 3. Extract latest feature row
        latest_features = features.iloc[-1:]

        # Ensure all required columns are present. Missing columns get 0
        if self.feature_cols:
            for col in self.feature_cols:
                if col not in latest_features.columns:
                    latest_features[col] = 0.0

            # Reorder
            latest_features = latest_features[self.feature_cols]

        # 4. Predict
        try:
            # Returns proba format: [Sell=0, Hold=1, Buy=2]
            probs = self.model.predict_proba(latest_features)[0]
        except Exception as e:
            _logger.error(f"Prediction error: {e}")
            return

        # 5. Trading Logic
        buy_prob = probs[2]
        sell_prob = probs[0]

        if buy_prob > self.buy_prob_min:
            if self.position.size <= 0:  # Allow reversal or fresh long
                if self.position.size < 0:
                    self._exit_position(reason=f"P08_BUY_REVERSAL_prob={buy_prob:.2f}")

                # Confidence scales between threshold and 1.0 (capped)
                confidence = min(1.0, buy_prob / max(0.01, self.buy_prob_min))
                self._enter_position("long", confidence=confidence, reason=f"P08_BUY_prob={buy_prob:.2f}")

        elif sell_prob > self.sell_prob_min:
            if self.position.size > 0:  # Exit current long
                self._exit_position(reason=f"P08_SELL_prob={sell_prob:.2f}")
                # Note: Currently pure long/short setup. Let's just exit long positions.
                # If shorting is desired: self._enter_position("short", ...)
