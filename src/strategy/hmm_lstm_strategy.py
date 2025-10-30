"""
Refactored HMM-LSTM Pipeline Trading Strategy

This module demonstrates how HMMLSTMPipelineStrategy can be simplified by inheriting
from BaseBacktraderStrategy, which provides common functionality like trade tracking,
position management, and performance monitoring.

The strategy focuses on:
- HMM model loading and regime detection
- LSTM model loading and price prediction
- Technical indicator calculation
- Strategy-specific entry/exit logic
"""

import numpy as np
import torch
import torch.nn as nn
import pickle
import json
import talib
from pathlib import Path
from collections import deque
from typing import Dict, Tuple, Any
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.notification.logger import setup_logger
from src.strategy.base_strategy import BaseStrategy

_logger = setup_logger(__name__)


class LSTMModel(nn.Module):
    """Enhanced LSTM model architecture matching the pipeline implementation."""

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
            batch_first=True,
            bidirectional=False
        )

        # Additional dense layers for better feature extraction
        self.dense1 = nn.Linear(hidden_size + n_regimes, hidden_size // 2)
        self.dense2 = nn.Linear(hidden_size // 2, hidden_size // 4)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout * 0.5)

        # Output layer
        self.linear = nn.Linear(hidden_size // 4, output_size)

        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x, regime_onehot):
        # Initialize hidden state
        device = x.device
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Take the output from the last time step
        last_output = lstm_out[:, -1, :]

        # Concatenate with regime information early
        combined = torch.cat([last_output, regime_onehot], dim=1)

        # Apply dense layers with activation and dropout
        dense1_out = self.dropout1(self.relu(self.dense1(combined)))
        dense2_out = self.dropout2(self.relu(self.dense2(dense1_out)))

        # Final prediction
        output = self.linear(dense2_out)

        return output


class HMMLSTMStrategy(BaseStrategy):
    """
    Refactored HMM-LSTM Pipeline Trading Strategy.

    This version inherits from BaseStrategy, which provides:
    - Trade tracking and management
    - Position sizing
    - Performance monitoring
    - Configuration management
    - Error handling

    The strategy focuses on:
    - HMM model loading and regime detection
    - LSTM model loading and log return prediction
    - Technical indicator calculation
    - Strategy-specific entry/exit logic

    **Important: LSTM Predictions**
    The LSTM model predicts log returns: log(price[t+1] / price[t])
    - Positive log return = predicted price increase
    - Negative log return = predicted price decrease
    - Typical log returns range from -0.003 to +0.003 (-0.3% to +0.3%)
    - Thresholds are set for log return scale (e.g., 0.001 = 0.1% predicted change)

    Can be used for both backtesting (with pre-loaded models) and live trading (loading from files).
    """

    def __init__(self,
                 hmm_model=None,
                 hmm_scaler=None,
                 hmm_features=None,
                 lstm_model=None,
                 lstm_scalers=None,
                 lstm_features=None,
                 sequence_length=None,
                 **kwargs):
        """
        Initialize the HMM-LSTM strategy.

        Args:
            hmm_model: Pre-loaded HMM model (for backtesting)
            hmm_scaler: HMM feature scaler (for backtesting)
            hmm_features: List of HMM feature names (for backtesting)
            lstm_model: Pre-loaded LSTM model (for backtesting)
            lstm_scalers: LSTM scalers (for backtesting)
            lstm_features: List of LSTM feature names (for backtesting)
            sequence_length: LSTM sequence length (for backtesting)
            **kwargs: Additional arguments passed to BaseStrategy
        """
        # Store pre-loaded models if provided (for backtesting)
        self.hmm_model_param = hmm_model
        self.hmm_scaler_param = hmm_scaler
        self.hmm_features_param = hmm_features
        self.lstm_model_param = lstm_model
        self.lstm_scalers_param = lstm_scalers
        self.lstm_features_param = lstm_features
        self.sequence_length_param = sequence_length

        # Initialize base strategy
        super().__init__(**kwargs)

    def _initialize_strategy(self):
        """Initialize HMM and LSTM models and indicators."""
        try:
            _logger.info("Initializing HMM-LSTM Pipeline Strategy")

            # Model components
            self.hmm_model = None
            self.lstm_model = None
            self.hmm_scaler = None
            self.lstm_scalers = None
            self.optimized_indicators = None
            self.lstm_features = None

            # Trading state
            self.current_regime = None
            self.regime_confidence = 0.0
            self.prediction_buffer = deque(maxlen=10)
            self.feature_buffer = deque(maxlen=100)

            # Trading parameters (configured for log return predictions)
            self.prediction_threshold = self.config.get('prediction_threshold', 0.001)  # 0.1% log return
            self.regime_confidence_threshold = self.config.get('regime_confidence_threshold', 0.4)  # 40% confidence

            # Exit parameters (configured for log return scale)
            self.profit_target = self.config.get('profit_target', 0.01)  # 1% profit target
            self.stop_loss = self.config.get('stop_loss', 0.005)  # 0.5% stop loss
            self.trailing_stop = self.config.get('trailing_stop', 0.005)  # 0.5% trailing stop

            # Check if models are passed as parameters (for backtesting)
            if hasattr(self, 'hmm_model_param') and self.hmm_model_param is not None:
                # Use pre-loaded models from backtesting
                self.hmm_model = self.hmm_model_param
                self.hmm_scaler = self.hmm_scaler_param
                self.hmm_features = self.hmm_features_param
                self.lstm_model = self.lstm_model_param
                self.lstm_scalers = self.lstm_scalers_param
                self.lstm_features = self.lstm_features_param
                self.sequence_length = self.sequence_length_param
                _logger.info("Using pre-loaded models for backtesting")
            else:
                # Load models from files (for live trading)
                self._load_models()

            # Initialize indicators
            self._init_indicators()

            # Trade tracking
            self.current_trade_regime = None

        except Exception as e:
            _logger.exception("Error in _initialize_strategy")
            raise

    def _load_models(self):
        """Load trained HMM and LSTM models with their parameters."""
        try:
            models_dir = Path(self.config.get('models_dir', 'src/ml/pipeline/p01_hmm_lstm/models'))
            results_dir = Path(self.config.get('results_dir', 'results'))

            # Load HMM model
            hmm_pattern = f"hmm_{self.symbol}_{self.timeframe}_*.pkl"
            hmm_files = list((models_dir / 'hmm').glob(hmm_pattern))

            if hmm_files:
                hmm_file = sorted(hmm_files)[-1]  # Latest model
                with open(hmm_file, 'rb') as f:
                    hmm_package = pickle.load(f)

                self.hmm_model = hmm_package['model']
                self.hmm_scaler = hmm_package['scaler']
                self.hmm_features = hmm_package['features']
                self._logger.info("Loaded HMM model from %s", hmm_file)
            else:
                self._logger.warning("No HMM model found for %s %s", self.symbol, self.timeframe)

            # Load LSTM model
            lstm_pattern = f"lstm_{self.symbol}_{self.timeframe}_*.pkl"
            lstm_files = list((models_dir / 'lstm').glob(lstm_pattern))

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

                self._logger.info("Loaded LSTM model from %s", lstm_file)
            else:
                self._logger.warning("No LSTM model found for %s %s", self.symbol, self.timeframe)

            # Load optimized indicator parameters
            indicator_pattern = f"indicators_{self.symbol}_{self.timeframe}_*.json"
            indicator_files = list(results_dir.glob(indicator_pattern))

            if indicator_files:
                indicator_file = sorted(indicator_files)[-1]  # Latest optimization
                with open(indicator_file, 'r') as f:
                    indicator_results = json.load(f)

                self.optimized_indicators = indicator_results['best_params']
                self._logger.info("Loaded optimized indicators from %s", indicator_file)
            else:
                self._logger.warning("No optimized indicators found for %s %s", self.symbol, self.timeframe)
                self.optimized_indicators = {}

        except Exception as e:
            self._logger.exception("Error loading models")
            raise

    def _init_indicators(self):
        """Initialize technical indicators using optimized parameters."""
        try:
            # Use optimized parameters if available, otherwise use defaults
            params = self.optimized_indicators or {}

            # Set default parameters if optimized_indicators is None
            if not params:
                params = {
                    'rsi_period': 14,
                    'bb_period': 20,
                    'bb_std': 2.0,
                    'macd_fast': 12,
                    'macd_slow': 26,
                    'macd_signal': 9,
                    'ema_fast': 12,
                    'ema_slow': 26,
                    'atr_period': 14,
                    'stoch_k': 14,
                    'stoch_d': 3
                }
                self.optimized_indicators = params

            # RSI
            rsi_period = params.get('rsi_period', 14)

            # Bollinger Bands
            bb_period = params.get('bb_period', 20)
            bb_std = params.get('bb_std', 2.0)

            # MACD
            macd_fast = params.get('macd_fast', 12)
            macd_slow = params.get('macd_slow', 26)
            macd_signal = params.get('macd_signal', 9)

            # EMA
            ema_fast = params.get('ema_fast', 12)
            ema_slow = params.get('ema_slow', 26)

            # ATR
            atr_period = params.get('atr_period', 14)

            self._logger.info("Initialized indicators with optimized parameters: %s", params)

        except Exception as e:
            self._logger.exception("Error initializing indicators")
            raise

    def _calculate_indicators(self) -> Dict[str, float]:
        """Calculate technical indicators for the current bar."""
        try:
            # Get recent OHLCV data
            lookback = 100  # Enough for indicator calculations

            if len(self.data) < lookback:
                return {}

            # Extract OHLCV arrays
            high = np.array([self.data.high[-i] for i in range(lookback, 0, -1)])
            low = np.array([self.data.low[-i] for i in range(lookback, 0, -1)])
            close = np.array([self.data.close[-i] for i in range(lookback, 0, -1)])
            volume = np.array([self.data.volume[-i] for i in range(lookback, 0, -1)])

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

            # Additional indicators
            stoch_k_period = params.get('stoch_k', 14)
            stoch_d_period = params.get('stoch_d', 3)

            stoch_k, stoch_d = talib.STOCH(high, low, close, fastk_period=stoch_k_period,
                                          slowk_period=stoch_d_period, slowd_period=stoch_d_period)

            if not np.isnan(stoch_k[-1]):
                indicators['stoch_k_opt'] = stoch_k[-1]
                indicators['stoch_d_opt'] = stoch_d[-1]

            williams_period = params.get('williams_period', 14)
            williams_r = talib.WILLR(high, low, close, timeperiod=williams_period)
            if not np.isnan(williams_r[-1]):
                indicators['williams_r_opt'] = williams_r[-1]

            mfi_period = params.get('mfi_period', 14)
            mfi = talib.MFI(high, low, close, volume, timeperiod=mfi_period)
            if not np.isnan(mfi[-1]):
                indicators['mfi_opt'] = mfi[-1]

            return indicators

        except Exception as e:
            self._logger.exception("Error calculating indicators")
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

                # Debug: Log regime prediction details occasionally
                if len(self.data) % 100 == 0:  # Log every 100 bars
                    self._logger.debug("HMM Regime Debug - Regime: %d, Posteriors: %s, Max Confidence: %.3f",
                                     int(regime[0]), posteriors[0].tolist(), confidence)

            except Exception as e:
                self._logger.warning("Error getting HMM posteriors: %s", e)
                confidence = 0.7  # Default confidence

            return int(regime[0]), float(confidence)

        except Exception as e:
            self._logger.exception("Error predicting regime")
            return 1, 0.5  # Default to neutral regime

    def _predict_price(self, features: Dict[str, float], regime: int, regime_confidence: float) -> float:
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

            # Add to buffer and check if we have enough sequence data
            if len(feature_values) != len(self.lstm_features):
                return 0.0

            # For this example, we'll use a simplified approach
            # In a real implementation, you'd maintain a proper sequence buffer
            if not hasattr(self, 'lstm_sequence_buffer'):
                self.lstm_sequence_buffer = deque(maxlen=self.sequence_length)

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
            self._logger.exception("Error predicting price")
            return 0.0

    def _execute_strategy_logic(self):
        """Execute HMM-LSTM specific strategy logic."""
        try:
            if len(self.data) < 100:  # Need sufficient data for indicators
                return

            # Calculate current features
            current_features = {
                'open': self.data.open[0],
                'high': self.data.high[0],
                'low': self.data.low[0],
                'close': self.data.close[0],
                'volume': self.data.volume[0],
                'log_return': np.log(self.data.close[0] / self.data.close[-1]) if self.data.close[-1] > 0 else 0.0
            }

            # Add technical indicators
            indicators = self._calculate_indicators()
            current_features.update(indicators)

            # Predict market regime
            regime, regime_confidence = self._predict_regime(current_features)

            # Predict price change
            prediction = self._predict_price(current_features, regime, regime_confidence)

            # Store predictions for analysis
            self.prediction_buffer.append({
                'regime': regime,
                'confidence': regime_confidence,
                'prediction': prediction,
                'price': self.data.close[0]
            })

            # Update current state
            self.current_regime = regime
            self.regime_confidence = regime_confidence

            # Convert log return to percentage for logging
            percentage_change = self._log_return_to_percentage(prediction)
            self._logger.debug("Regime: %d (conf: %.3f), Log Return: %.6f (%.4f%%), Price: %.4f",
                             regime, regime_confidence, prediction, percentage_change * 100, self.data.close[0])

            # Trading logic
            current_position = self.position.size

            # Entry logic
            if current_position == 0:
                self._check_entry_signals(prediction, regime, regime_confidence)

            # Exit logic
            elif current_position != 0:
                self._check_exit_signals(prediction, regime, regime_confidence)

        except Exception as e:
            self._logger.exception("Error in _execute_strategy_logic")

    def _log_return_to_percentage(self, log_return: float) -> float:
        """Convert log return to percentage change."""
        return np.exp(log_return) - 1

    def _percentage_to_log_return(self, percentage: float) -> float:
        """Convert percentage change to log return."""
        return np.log(1 + percentage)

    def _check_entry_signals(self, prediction: float, regime: int, regime_confidence: float):
        """Check for entry signals and execute trades."""
        try:
            # Debug: Log signal check details occasionally
            if len(self.data) % 50 == 0:  # Log every 50 bars
                self._logger.debug("Signal Check - Prediction: %.6f, Threshold: %.6f, Regime Conf: %.3f, Threshold: %.3f",
                                 prediction, self.prediction_threshold, regime_confidence, self.regime_confidence_threshold)

            # Only trade if we have sufficient confidence in regime prediction
            if regime_confidence < self.regime_confidence_threshold:
                if len(self.data) % 50 == 0:  # Log occasionally
                    self._logger.debug("No trade: Regime confidence %.3f < threshold %.3f",
                                     regime_confidence, self.regime_confidence_threshold)
                return

            # Only trade if prediction is strong enough (prediction is log return)
            if abs(prediction) < self.prediction_threshold:
                if len(self.data) % 50 == 0:  # Log occasionally
                    self._logger.debug("No trade: Prediction %.6f < threshold %.6f",
                                     abs(prediction), self.prediction_threshold)
                return

            # Convert log return prediction to percentage change for signal strength calculation
            percentage_change = self._log_return_to_percentage(prediction)

            # Calculate confidence and risk multiplier based on percentage change strength
            prediction_strength = abs(percentage_change)
            confidence = min(1.0, prediction_strength * 1000)  # Scale for percentage (0.001 = 0.1%)
            risk_multiplier = min(2.0, 1.0 + prediction_strength * 1000)  # Scale for percentage

            # Long signal (positive log return = price increase)
            if prediction > self.prediction_threshold:
                self._logger.info("LONG signal - Log Return: %.6f (%.4f%%), Regime: %d, Confidence: %.3f",
                               prediction, percentage_change * 100, regime, regime_confidence)

                self._enter_position(
                    direction='long',
                    confidence=confidence,
                    risk_multiplier=risk_multiplier,
                    reason=f"HMM-LSTM log return prediction: {prediction:.6f} ({percentage_change*100:.4f}%)"
                )

            # Short signal (negative log return = price decrease)
            elif prediction < -self.prediction_threshold and self.config.get('allow_short', False):
                self._logger.info("SHORT signal - Log Return: %.6f (%.4f%%), Regime: %d, Confidence: %.3f",
                               prediction, percentage_change * 100, regime, regime_confidence)

                self._enter_position(
                    direction='short',
                    confidence=confidence,
                    risk_multiplier=risk_multiplier,
                    reason=f"HMM-LSTM prediction: {prediction:.6f}"
                )

        except Exception as e:
            self._logger.exception("Error checking entry signals")

    def _check_exit_signals(self, prediction: float, regime: int, regime_confidence: float):
        """Check for exit signals and close positions."""
        try:
            if not self.entry_price:
                return

            current_price = self.data.close[0]
            position_size = self.position.size

            # Calculate current profit/loss
            if position_size > 0:  # Long position
                pnl_pct = (current_price - self.entry_price) / self.entry_price
            else:  # Short position
                pnl_pct = (self.entry_price - current_price) / self.entry_price

            # Update highest profit for trailing stop
            if pnl_pct > self.highest_profit:
                self.highest_profit = pnl_pct

            # Exit conditions
            exit_signal = False
            exit_reason = ""

            # 1. Profit target
            if pnl_pct >= self.profit_target:
                exit_signal = True
                exit_reason = "profit_target"

            # 2. Stop loss
            elif pnl_pct <= -self.stop_loss:
                exit_signal = True
                exit_reason = "stop_loss"

            # 3. Trailing stop
            elif self.highest_profit > 0 and (self.highest_profit - pnl_pct) >= self.trailing_stop:
                exit_signal = True
                exit_reason = "trailing_stop"

            # 4. Regime change with low confidence
            elif regime != self.current_trade_regime and regime_confidence > 0.7:
                exit_signal = True
                exit_reason = "regime_change"

            # 5. Prediction reversal (log return predictions)
            elif position_size > 0 and prediction < -self.prediction_threshold:
                exit_signal = True
                exit_reason = "prediction_reversal"
            elif position_size < 0 and prediction > self.prediction_threshold:
                exit_signal = True
                exit_reason = "prediction_reversal"

            # 6. Low regime confidence
            elif regime_confidence < 0.3:
                exit_signal = True
                exit_reason = "low_confidence"

            # Execute exit
            if exit_signal:
                self._logger.info("EXIT signal (%s) - PnL: %.4f, Prediction: %.6f, Regime: %d",
                               exit_reason, pnl_pct, prediction, regime)

                self._exit_position(reason=exit_reason)

                # Reset trade tracking
                self.current_trade_regime = None

        except Exception as e:
            self._logger.exception("Error checking exit signals")

    def stop(self):
        """Called when strategy stops."""
        try:
            # Log performance summary
            if hasattr(self, 'prediction_buffer') and self.prediction_buffer:
                predictions = [p['prediction'] for p in self.prediction_buffer]
                regimes = [p['regime'] for p in self.prediction_buffer]
                confidences = [p['confidence'] for p in self.prediction_buffer]

                self._logger.info("HMM-LSTM Performance Summary:")
                self._logger.info("  Total predictions: %s", len(predictions))
                self._logger.info("  Avg prediction magnitude: %s", np.mean([abs(p) for p in predictions]))
                self._logger.info("  Avg regime confidence: %s", np.mean(confidences))
                self._logger.info("  Regime distribution: %s", np.bincount(regimes))

            # Call base class stop method
            super().stop()

        except Exception as e:
            self._logger.exception("Error in stop")
