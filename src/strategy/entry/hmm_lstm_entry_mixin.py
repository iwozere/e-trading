"""
HMM-LSTM Entry Mixin

This mixin provides entry signals based on HMM regime detection and LSTM predictions.
It integrates with the existing strategy framework and can be combined with various
exit strategies.

Usage:
    Can be used with the CustomStrategy class by specifying:
    entry_logic: {
        "name": "HMMLSTMEntryMixin",
        "params": {
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "prediction_threshold": 0.001,
            "regime_confidence_threshold": 0.6,
            "models_dir": "src/ml/pipeline/p01_hmm_lstm/models"
        }
    }
"""

import numpy as np
import torch
import torch.nn as nn
import pickle
import json
import talib
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional, Tuple
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

class LSTMModel(nn.Module):
    """LSTM model architecture matching the pipeline implementation."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 dropout: float = 0.2, output_size: int = 1, n_regimes: int = 3):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_regimes = n_regimes

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Output layer with regime conditioning
        self.linear = nn.Linear(hidden_size + n_regimes, output_size)

    def forward(self, x, regime_onehot):
        # Initialize hidden state
        device = x.device
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Take the output from the last time step
        last_output = lstm_out[:, -1, :]

        # Apply dropout
        dropped = self.dropout(last_output)

        # Concatenate with regime information
        combined = torch.cat([dropped, regime_onehot], dim=1)

        # Final prediction
        output = self.linear(combined)

        return output

class HMMLSTMEntryMixin:
    """
    Entry mixin that uses HMM regime detection and LSTM predictions.

    Parameters:
    -----------
    symbol : str
        Trading symbol (e.g., 'BTCUSDT')
    timeframe : str
        Timeframe (e.g., '1h')
    prediction_threshold : float
        Minimum prediction magnitude for entry signal
    regime_confidence_threshold : float
        Minimum regime confidence for entry signal
    models_dir : str
        Directory containing trained models
    results_dir : str
        Directory containing optimization results
    """

    def __init__(self, params: Dict):
        self.params = params
        self.symbol = params.get('symbol', 'BTCUSDT')
        self.timeframe = params.get('timeframe', '1h')
        self.prediction_threshold = params.get('prediction_threshold', 0.001)
        self.regime_confidence_threshold = params.get('regime_confidence_threshold', 0.6)
        self.models_dir = Path(params.get('models_dir', 'src/ml/pipeline/p01_hmm_lstm/models'))
        self.results_dir = Path(params.get('results_dir', 'results'))

        # Model components
        self.hmm_model = None
        self.lstm_model = None
        self.hmm_scaler = None
        self.lstm_scalers = None
        self.optimized_indicators = None
        self.lstm_features = None
        self.hmm_features = None

        # Buffers
        self.feature_buffer = deque(maxlen=100)
        self.lstm_sequence_buffer = None
        self.sequence_length = 60  # Default, will be updated when model loads

        # State
        self.current_regime = None
        self.regime_confidence = 0.0
        self.current_prediction = 0.0
        self.strategy = None

        _logger.info("Initialized HMM-LSTM Entry Mixin for %s %s", self.symbol, self.timeframe)

    def init_entry(self, strategy):
        """Initialize the entry mixin with strategy reference."""
        self.strategy = strategy

        try:
            self._load_models()
            _logger.info("HMM-LSTM models loaded successfully")
        except Exception as e:
            _logger.exception("Failed to load models")
            # Continue without models - mixin will return False for should_enter()

    def _load_models(self):
        """Load trained HMM and LSTM models with their parameters."""
        try:
            # Load HMM model
            hmm_pattern = f"hmm_{self.symbol}_{self.timeframe}_*.pkl"
            hmm_files = list((self.models_dir / 'hmm').glob(hmm_pattern))

            if hmm_files:
                hmm_file = sorted(hmm_files)[-1]  # Latest model
                with open(hmm_file, 'rb') as f:
                    hmm_package = pickle.load(f)

                self.hmm_model = hmm_package['model']
                self.hmm_scaler = hmm_package['scaler']
                self.hmm_features = hmm_package['features']
                _logger.info("Loaded HMM model from %s", hmm_file)
            else:
                _logger.warning("No HMM model found for %s %s", self.symbol, self.timeframe)

            # Load LSTM model
            lstm_pattern = f"lstm_{self.symbol}_{self.timeframe}_*.pkl"
            lstm_files = list((self.models_dir / 'lstm').glob(lstm_pattern))

            if lstm_files:
                lstm_file = sorted(lstm_files)[-1]  # Latest model
                with open(lstm_file, 'rb') as f:
                    lstm_package = pickle.load(f)

                # Recreate LSTM model
                arch = lstm_package['model_architecture']
                self.lstm_model = LSTMModel(
                    input_size=arch['input_size'],
                    hidden_size=arch['hidden_size'],
                    num_layers=arch['num_layers'],
                    n_regimes=arch['n_regimes']
                )
                self.lstm_model.load_state_dict(lstm_package['model_state_dict'])
                self.lstm_model.eval()

                self.lstm_scalers = lstm_package['scalers']
                self.lstm_features = lstm_package['features']
                self.sequence_length = lstm_package['hyperparameters']['sequence_length']

                # Initialize LSTM sequence buffer
                self.lstm_sequence_buffer = deque(maxlen=self.sequence_length)

                _logger.info("Loaded LSTM model from %s", lstm_file)
            else:
                _logger.warning("No LSTM model found for %s %s", self.symbol, self.timeframe)

            # Load optimized indicator parameters
            indicator_pattern = f"indicators_{self.symbol}_{self.timeframe}_*.json"
            indicator_files = list(self.results_dir.glob(indicator_pattern))

            if indicator_files:
                indicator_file = sorted(indicator_files)[-1]  # Latest optimization
                with open(indicator_file, 'r') as f:
                    indicator_results = json.load(f)

                self.optimized_indicators = indicator_results['best_params']
                _logger.info("Loaded optimized indicators from %s", indicator_file)
            else:
                _logger.warning("No optimized indicators found for %s %s", self.symbol, self.timeframe)
                self.optimized_indicators = {}

        except Exception as e:
            _logger.exception("Error loading models")
            raise

    def _calculate_indicators(self) -> Dict[str, float]:
        """Calculate technical indicators for the current bar."""
        try:
            # Get recent OHLCV data
            lookback = 100  # Enough for indicator calculations

            if len(self.strategy.data) < lookback:
                return {}

            # Extract OHLCV arrays
            high = np.array([self.strategy.data.high[-i] for i in range(lookback, 0, -1)])
            low = np.array([self.strategy.data.low[-i] for i in range(lookback, 0, -1)])
            close = np.array([self.strategy.data.close[-i] for i in range(lookback, 0, -1)])
            volume = np.array([self.strategy.data.volume[-i] for i in range(lookback, 0, -1)])

            indicators = {}
            params = self.optimized_indicators

            # RSI
            rsi_period = params.get('rsi_period', 14)
            rsi = talib.RSI(close, timeperiod=rsi_period)
            indicators['rsi_optimized'] = rsi[-1] if not np.isnan(rsi[-1]) else 50.0

            # Bollinger Bands
            bb_period = params.get('bb_period', 20)
            bb_std = params.get('bb_std', 2.0)
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=bb_period,
                                                        nbdevup=bb_std, nbdevdn=bb_std)

            if not np.isnan(bb_upper[-1]):
                indicators['bb_upper_opt'] = bb_upper[-1]
                indicators['bb_middle_opt'] = bb_middle[-1]
                indicators['bb_lower_opt'] = bb_lower[-1]
                indicators['bb_position_opt'] = (close[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
                indicators['bb_width_opt'] = (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1]

            # MACD
            macd_fast = params.get('macd_fast', 12)
            macd_slow = params.get('macd_slow', 26)
            macd_signal = params.get('macd_signal', 9)

            macd, macd_signal_line, macd_hist = talib.MACD(close, fastperiod=macd_fast,
                                                          slowperiod=macd_slow, signalperiod=macd_signal)

            if not np.isnan(macd[-1]):
                indicators['macd_opt'] = macd[-1]
                indicators['macd_signal_opt'] = macd_signal_line[-1]
                indicators['macd_histogram_opt'] = macd_hist[-1]

            # EMA
            ema_fast_period = params.get('ema_fast', 12)
            ema_slow_period = params.get('ema_slow', 26)

            ema_fast = talib.EMA(close, timeperiod=ema_fast_period)
            ema_slow = talib.EMA(close, timeperiod=ema_slow_period)

            if not np.isnan(ema_fast[-1]) and not np.isnan(ema_slow[-1]):
                indicators['ema_fast_opt'] = ema_fast[-1]
                indicators['ema_slow_opt'] = ema_slow[-1]
                indicators['ema_spread_opt'] = (ema_fast[-1] - ema_slow[-1]) / close[-1]

            # ATR
            atr_period = params.get('atr_period', 14)
            atr = talib.ATR(high, low, close, timeperiod=atr_period)
            if not np.isnan(atr[-1]):
                indicators['atr_opt'] = atr[-1]

            return indicators

        except Exception as e:
            _logger.exception("Error calculating indicators")
            return {}

    def _predict_regime(self, features: Dict[str, float]) -> Tuple[int, float]:
        """Predict market regime using HMM model."""
        try:
            if not self.hmm_model or not self.hmm_features:
                return 1, 0.5  # Default to neutral regime

            # Prepare features for HMM
            feature_values = []
            for feature in self.hmm_features:
                if feature in features:
                    feature_values.append(features[feature])
                else:
                    feature_values.append(0.0)  # Default value

            # Add to buffer and check if we have enough data
            self.feature_buffer.append(feature_values)

            if len(self.feature_buffer) < 10:  # Need some history
                return 1, 0.5

            # Scale features
            recent_features = np.array(list(self.feature_buffer)[-10:])  # Last 10 observations
            scaled_features = self.hmm_scaler.transform(recent_features)

            # Predict regime
            regime = self.hmm_model.predict(scaled_features[-1:])  # Predict for last observation

            # Get posterior probabilities for confidence
            try:
                posteriors = self.hmm_model.predict_proba(scaled_features[-1:])
                confidence = np.max(posteriors[0])
            except:
                confidence = 0.7  # Default confidence

            return int(regime[0]), float(confidence)

        except Exception as e:
            _logger.exception("Error predicting regime")
            return 1, 0.5  # Default to neutral regime

    def _predict_price(self, features: Dict[str, float], regime: int) -> float:
        """Predict next price change using LSTM model."""
        try:
            if not self.lstm_model or not self.lstm_features:
                return 0.0

            # Prepare features for LSTM
            feature_values = []
            for feature in self.lstm_features:
                if feature in features:
                    feature_values.append(features[feature])
                else:
                    feature_values.append(0.0)  # Default value

            # Add to sequence buffer
            if len(feature_values) != len(self.lstm_features):
                return 0.0

            self.lstm_sequence_buffer.append(feature_values)

            if len(self.lstm_sequence_buffer) < self.sequence_length:
                return 0.0

            # Scale features
            sequence_data = np.array(list(self.lstm_sequence_buffer))
            sequence_scaled = self.lstm_scalers['feature_scaler'].transform(
                sequence_data.reshape(-1, len(self.lstm_features))
            ).reshape(1, self.sequence_length, len(self.lstm_features))

            # Create regime one-hot encoding
            regime_onehot = np.zeros((1, 3))  # Assuming 3 regimes
            if 0 <= regime < 3:
                regime_onehot[0, regime] = 1

            # Predict
            with torch.no_grad():
                sequence_tensor = torch.tensor(sequence_scaled, dtype=torch.float32)
                regime_tensor = torch.tensor(regime_onehot, dtype=torch.float32)

                prediction_scaled = self.lstm_model(sequence_tensor, regime_tensor).item()

                # Inverse transform to get actual prediction
                prediction = self.lstm_scalers['target_scaler'].inverse_transform(
                    [[prediction_scaled]]
                )[0][0]

            return float(prediction)

        except Exception as e:
            _logger.exception("Error predicting price")
            return 0.0

    def next(self):
        """Called on each bar to update internal state."""
        # This method is called by the strategy framework
        pass

    def should_enter(self) -> bool:
        """Determine if entry signal is present."""
        try:
            # Return False if models are not loaded
            if not self.hmm_model or not self.lstm_model:
                return False

            # Need sufficient data
            if len(self.strategy.data) < 100:
                return False

            # Calculate current features
            current_features = {
                'open': self.strategy.data.open[0],
                'high': self.strategy.data.high[0],
                'low': self.strategy.data.low[0],
                'close': self.strategy.data.close[0],
                'volume': self.strategy.data.volume[0],
                'log_return': np.log(self.strategy.data.close[0] / self.strategy.data.close[-1])
                             if self.strategy.data.close[-1] > 0 else 0.0
            }

            # Add technical indicators
            indicators = self._calculate_indicators()
            current_features.update(indicators)

            # Predict market regime
            regime, regime_confidence = self._predict_regime(current_features)

            # Predict price change
            prediction = self._predict_price(current_features, regime)

            # Update state
            self.current_regime = regime
            self.regime_confidence = regime_confidence
            self.current_prediction = prediction

            # Entry logic
            entry_signal = (
                regime_confidence >= self.regime_confidence_threshold and
                abs(prediction) >= self.prediction_threshold and
                prediction > 0  # Only long signals for now
            )

            if entry_signal:
                _logger.info("Entry signal: Regime=%d (conf=%.3f), Prediction=%.6f",
                           regime or 0, regime_confidence or 0.0, prediction or 0.0)

            return entry_signal

        except Exception as e:
            _logger.exception("Error in should_enter")
            return False

    def notify_trade(self, trade):
        """Handle trade notifications."""
        # This method is called by the strategy framework when trades occur
        pass

    def get_current_state(self) -> Dict:
        """Get current state for debugging/monitoring."""
        return {
            'regime': self.current_regime,
            'regime_confidence': self.regime_confidence,
            'prediction': self.current_prediction,
            'models_loaded': self.hmm_model is not None and self.lstm_model is not None
        }
