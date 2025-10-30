"""
CNN-XGBoost Trading Strategy.

This strategy combines CNN feature extraction with XGBoost classification for trading decisions.
It uses the models trained in pipeline p03_cnn_xgboost to make predictions on multiple targets:
- target_direction: Price direction (up/down)
- target_volatility: Volatility regime (high/low)
- target_trend: Trend strength (strong/weak)
- target_magnitude: Price movement magnitude (large/small)

The strategy integrates these predictions to make trading decisions with proper risk management.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import backtrader as bt
import json
import pickle

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.strategy.base_strategy import BaseStrategy
from src.notification.logger import setup_logger

warnings.filterwarnings('ignore')

_logger = setup_logger(__name__)


class CNN1D(nn.Module):
    """
    1D Convolutional Neural Network for time series feature extraction.

    This matches the CNN architecture used in pipeline p03_cnn_xgboost.
    """

    def __init__(self,
                 input_channels: int = 5,
                 sequence_length: int = 120,
                 num_filters: List[int] = [32, 64, 128],
                 kernel_sizes: List[int] = [3, 5, 7],
                 dropout_rate: float = 0.3) -> None:
        """
        Initialize the 1D CNN architecture.

        Args:
            input_channels: Number of input features (OHLCV = 5)
            sequence_length: Length of time series sequence
            num_filters: List of filter counts for each convolutional layer
            kernel_sizes: List of kernel sizes for each convolutional layer
            dropout_rate: Dropout rate for regularization
        """
        super(CNN1D, self).__init__()

        self.input_channels = input_channels
        self.sequence_length = sequence_length

        # Build convolutional layers
        layers = []
        in_channels = input_channels

        for i, (filters, kernel_size) in enumerate(zip(num_filters, kernel_sizes)):
            layers.extend([
                nn.Conv1d(in_channels, filters, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            in_channels = filters

        self.conv_layers = nn.Sequential(*layers)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Final classification layer (needed to load trained models)
        self.classification_layer = nn.Linear(num_filters[-1], 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN.

        Args:
            x: Input tensor of shape (batch_size, input_channels, sequence_length)

        Returns:
            Features tensor of shape (batch_size, num_filters[-1]) before classification
        """
        # Apply convolutional layers
        x = self.conv_layers(x)

        # Global average pooling
        x = self.global_pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Return features before classification (for feature extraction)
        return x

    def forward_with_classification(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN with classification.

        Args:
            x: Input tensor of shape (batch_size, input_channels, sequence_length)

        Returns:
            Logits tensor of shape (batch_size, 1) for binary classification
        """
        # Get features
        features = self.forward(x)

        # Apply classification layer
        logits = self.classification_layer(features)

        return logits


class CNNXGBoostStrategy(BaseStrategy):
    """
    CNN-XGBoost Trading Strategy.

    This strategy combines CNN feature extraction with XGBoost classification for trading decisions.
    It uses the models trained in pipeline p03_cnn_xgboost to make predictions on multiple targets.

    The strategy integrates these predictions to make trading decisions with proper risk management.
    """

    params = (
        ('prediction_threshold', 0.6),  # Minimum confidence for trading signals
        ('direction_weight', 0.4),      # Weight for direction prediction
        ('volatility_weight', 0.2),     # Weight for volatility prediction
        ('trend_weight', 0.2),          # Weight for trend prediction
        ('magnitude_weight', 0.2),      # Weight for magnitude prediction
        ('sequence_length', 120),       # CNN sequence length
        ('cnn_input_channels', 5),      # OHLCV features
        ('cnn_num_filters', [32, 64, 128]),
        ('cnn_kernel_sizes', [3, 5, 7]),
        ('cnn_dropout_rate', 0.3),
    )

    def __init__(self):
        """Initialize the CNN-XGBoost strategy."""
        super().__init__()

        # Strategy parameters
        self.prediction_threshold = self.config.get('prediction_threshold', 0.6)
        self.direction_weight = self.config.get('direction_weight', 0.4)
        self.volatility_weight = self.config.get('volatility_weight', 0.2)
        self.trend_weight = self.config.get('trend_weight', 0.2)
        self.magnitude_weight = self.config.get('magnitude_weight', 0.2)

        # Model parameters
        self.sequence_length = self.config.get('sequence_length', 120)
        self.cnn_input_channels = self.config.get('cnn_input_channels', 5)
        self.cnn_num_filters = self.config.get('cnn_num_filters', [32, 64, 128])
        self.cnn_kernel_sizes = self.config.get('cnn_kernel_sizes', [3, 5, 7])
        self.cnn_dropout_rate = self.config.get('cnn_dropout_rate', 0.3)

        # Exit parameters (configured for classification scale)
        self.profit_target = self.config.get('profit_target', 0.02)  # 2% profit target
        self.stop_loss = self.config.get('stop_loss', 0.01)  # 1% stop loss
        self.trailing_stop = self.config.get('trailing_stop', 0.005)  # 0.5% trailing stop

        # Model components
        self.cnn_model = None
        self.cnn_scaler = None
        self.xgb_models = {}
        self.xgb_scalers = {}

        # Data buffers
        self.ohlcv_buffer = []
        self.feature_buffer = []

        # Target variables
        self.targets = ["target_direction", "target_volatility", "target_trend", "target_magnitude"]

        # Initialize models
        self._load_models()
        self._init_indicators()

        _logger.info("CNN-XGBoost strategy initialized successfully")

    def _load_models(self):
        """Load CNN and XGBoost models from pipeline p03_cnn_xgboost."""
        try:
            # Load CNN model
            cnn_model_path = self.config.get('cnn_model_path')
            if cnn_model_path and Path(cnn_model_path).exists():
                self._load_cnn_model(cnn_model_path)

            # Load XGBoost models
            xgb_models_dir = self.config.get('xgb_models_dir')
            if xgb_models_dir and Path(xgb_models_dir).exists():
                self._load_xgb_models(xgb_models_dir)

            _logger.info("Models loaded successfully")

        except Exception as e:
            _logger.exception("Error loading models:")
            raise

    def _load_cnn_model(self, model_path: str):
        """Load CNN model from saved checkpoint."""
        try:
            # Load model configuration
            config_path = model_path.replace('.pth', '_config.json')
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)

                # Create model with saved configuration
                self.cnn_model = CNN1D(
                    input_channels=config.get('input_channels', 5),
                    sequence_length=config.get('sequence_length', 120),
                    num_filters=config.get('num_filters', [32, 64, 128]),
                    kernel_sizes=config.get('kernel_sizes', [3, 5, 7]),
                    dropout_rate=config.get('dropout_rate', 0.3)
                )

                # Load model weights
                self.cnn_model.load_state_dict(torch.load(model_path, map_location='cpu'))
                self.cnn_model.eval()

                # Load scaler
                scaler_path = model_path.replace('.pth', '_scaler.pkl')
                if Path(scaler_path).exists():
                    with open(scaler_path, 'rb') as f:
                        self.cnn_scaler = pickle.load(f)

                _logger.info("CNN model loaded from %s", model_path)

        except Exception as e:
            _logger.exception("Error loading CNN model:")
            raise

    def _load_xgb_models(self, models_dir: str):
        """Load XGBoost models for all targets."""
        try:
            models_dir = Path(models_dir)

            for target in self.targets:
                model_path = models_dir / f"{target}_model.pkl"
                if model_path.exists():
                    with open(model_path, 'rb') as f:
                        self.xgb_models[target] = pickle.load(f)

                    # Load scaler if available
                    scaler_path = models_dir / f"{target}_scaler.pkl"
                    if scaler_path.exists():
                        with open(scaler_path, 'rb') as f:
                            self.xgb_scalers[target] = pickle.load(f)

                    _logger.info("XGBoost model loaded for %s", target)

        except Exception as e:
            _logger.exception("Error loading XGBoost models:")
            raise

    def _init_indicators(self):
        """Initialize technical indicators."""
        try:
            # Use optimized parameters if available, otherwise use defaults
            params = self.config.get('optimized_indicators', {})

            # Set default parameters if optimized_indicators is None
            if not params:
                params = {
                    'rsi_period': 14,
                    'bb_period': 20,
                    'bb_std': 2.0,
                    'macd_fast': 12,
                    'macd_slow': 26,
                    'macd_signal': 9
                }

            # RSI
            self.rsi = bt.indicators.RSI(
                self.data.close,
                period=params.get('rsi_period', 14)
            )

            # Bollinger Bands
            self.bb = bt.indicators.BollingerBands(
                self.data.close,
                period=params.get('bb_period', 20),
                devfactor=params.get('bb_std', 2.0)
            )

            # MACD
            self.macd = bt.indicators.MACD(
                self.data.close,
                period_me1=params.get('macd_fast', 12),
                period_me2=params.get('macd_slow', 26),
                period_signal=params.get('macd_signal', 9)
            )

            # Moving averages
            self.sma_short = bt.indicators.SMA(self.data.close, period=10)
            self.sma_long = bt.indicators.SMA(self.data.close, period=30)

            # ATR for volatility
            self.atr = bt.indicators.ATR(self.data, period=14)

            # Additional indicators for feature matching
            self.ema_fast = bt.indicators.EMA(self.data.close, period=12)
            self.sma_20 = bt.indicators.SMA(self.data.close, period=20)

            _logger.debug("Technical indicators initialized")

        except Exception as e:
            _logger.exception("Error initializing indicators:")
            raise

    def _prepare_ohlcv_data(self) -> np.ndarray:
        """Prepare OHLCV data for CNN input."""
        try:
            # Get recent OHLCV data
            ohlcv_data = []
            for i in range(-self.sequence_length + 1, 1):
                ohlcv_data.append([
                    self.data.open[i],
                    self.data.high[i],
                    self.data.low[i],
                    self.data.close[i],
                    self.data.volume[i]
                ])

            # Convert to numpy array and reshape for CNN
            ohlcv_array = np.array(ohlcv_data, dtype=np.float32)
            ohlcv_array = ohlcv_array.T  # Transpose to (channels, sequence_length)

            # Normalize if scaler is available
            if self.cnn_scaler is not None:
                ohlcv_array = self.cnn_scaler.transform(ohlcv_array.T).T

            return ohlcv_array

        except Exception as e:
            _logger.exception("Error preparing OHLCV data:")
            return None

    def _extract_cnn_features(self, ohlcv_data: np.ndarray) -> np.ndarray:
        """Extract features using CNN model."""
        try:
            if self.cnn_model is None:
                return None

            # Convert to tensor
            ohlcv_tensor = torch.FloatTensor(ohlcv_data).unsqueeze(0)  # Add batch dimension

            # Get CNN features (before classification layer)
            with torch.no_grad():
                # Get features from global pooling layer
                x = self.cnn_model.conv_layers(ohlcv_tensor)
                x = self.cnn_model.global_pool(x)
                features = x.view(x.size(0), -1).numpy()

            return features.flatten()

        except Exception as e:
            _logger.exception("Error extracting CNN features:")
            return None

    def _prepare_xgb_features(self) -> np.ndarray:
        """Prepare features for XGBoost models."""
        try:
            # Technical indicators - match the pipeline's feature set
            features = []

            # RSI
            features.append(self.rsi[0] if not np.isnan(self.rsi[0]) else 50)

            # MACD
            features.append(self.macd.macd[0] if not np.isnan(self.macd.macd[0]) else 0)
            features.append(self.macd.signal[0] if not np.isnan(self.macd.signal[0]) else 0)
            features.append(self.macd.macd[0] - self.macd.signal[0] if not np.isnan(self.macd.macd[0] - self.macd.signal[0]) else 0)

            # Bollinger Bands
            features.append(self.bb.lines.top[0] if not np.isnan(self.bb.lines.top[0]) else self.data.close[0])
            features.append(self.bb.lines.mid[0] if not np.isnan(self.bb.lines.mid[0]) else self.data.close[0])
            features.append(self.bb.lines.bot[0] if not np.isnan(self.bb.lines.bot[0]) else self.data.close[0])

            # Moving averages
            features.append(self.sma_short[0] if not np.isnan(self.sma_short[0]) else self.data.close[0])
            features.append(self.sma_long[0] if not np.isnan(self.sma_long[0]) else self.data.close[0])

            # Price vs moving averages
            features.append(self.data.close[0] / self.sma_short[0] if not np.isnan(self.sma_short[0]) else 1.0)
            features.append(self.data.close[0] / self.sma_long[0] if not np.isnan(self.sma_long[0]) else 1.0)

            # Stochastic (simplified)
            features.append(50.0)  # Placeholder for stoch_k
            features.append(50.0)  # Placeholder for stoch_d

            # ADX (simplified)
            features.append(25.0)  # Placeholder for adx

            # OBV (simplified)
            features.append(self.data.volume[0] if self.data.volume[0] > 0 else 1)

            # ATR
            features.append(self.atr[0] if not np.isnan(self.atr[0]) else 0)

            # CCI (simplified)
            features.append(0.0)  # Placeholder for cci

            # ROC (simplified)
            features.append(0.0)  # Placeholder for roc

            # MFI (simplified)
            features.append(50.0)  # Placeholder for mfi

            # Additional price-based features (matching pipeline)
            features.append(self.data.close[0] / self.data.open[0] - 1)  # Current bar return
            features.append(self.data.high[0] / self.data.low[0] - 1)    # High-low ratio
            features.append(self.data.close[0] / self.data.open[0])      # Close-open ratio

            # Momentum features (simplified)
            features.append(0.0)  # Placeholder for momentum

            # Base OHLCV features (only close and volume to match pipeline)
            features.append(self.data.close[0])
            features.append(self.data.volume[0] if self.data.volume[0] > 0 else 1)

            return np.array(features, dtype=np.float32).reshape(1, -1)

        except Exception as e:
            _logger.exception("Error preparing XGB features:")
            return None

    def _get_predictions(self) -> Dict[str, float]:
        """Get predictions from CNN and XGBoost models."""
        try:
            predictions = {}

            # Get CNN features
            ohlcv_data = self._prepare_ohlcv_data()
            if ohlcv_data is not None:
                cnn_features = self._extract_cnn_features(ohlcv_data)

                # Get XGBoost features
                xgb_features = self._prepare_xgb_features()

                if cnn_features is not None and xgb_features is not None:
                    # Combine features
                    combined_features = np.concatenate([cnn_features, xgb_features.flatten()])

                    # Get predictions from XGBoost models
                    for target in self.targets:
                        if target in self.xgb_models:
                            model = self.xgb_models[target]

                            # Scale features if scaler is available
                            if target in self.xgb_scalers:
                                features_scaled = self.xgb_scalers[target].transform(combined_features.reshape(1, -1))
                            else:
                                features_scaled = combined_features.reshape(1, -1)

                            # Get prediction probability
                            pred_proba = model.predict_proba(features_scaled)[0]
                            predictions[target] = pred_proba[1]  # Probability of positive class

            return predictions

        except Exception as e:
            _logger.exception("Error getting predictions:")
            return {}

    def _calculate_signal_strength(self, predictions: Dict[str, float]) -> float:
        """Calculate overall signal strength from all predictions."""
        try:
            if not predictions:
                return 0.0

            # Weighted combination of predictions
            signal_strength = 0.0

            # Direction prediction (most important)
            if 'target_direction' in predictions:
                direction_signal = predictions['target_direction'] - 0.5  # Center around 0
                signal_strength += self.direction_weight * direction_signal

            # Volatility prediction
            if 'target_volatility' in predictions:
                volatility_signal = predictions['target_volatility'] - 0.5
                signal_strength += self.volatility_weight * volatility_signal

            # Trend prediction
            if 'target_trend' in predictions:
                trend_signal = predictions['target_trend'] - 0.5
                signal_strength += self.trend_weight * trend_signal

            # Magnitude prediction
            if 'target_magnitude' in predictions:
                magnitude_signal = predictions['target_magnitude'] - 0.5
                signal_strength += self.magnitude_weight * magnitude_signal

            return signal_strength

        except Exception as e:
            _logger.exception("Error calculating signal strength:")
            return 0.0

    def _check_entry_signals(self, predictions: Dict[str, float], signal_strength: float):
        """Check for entry signals and execute trades."""
        try:
            # Only trade if signal strength is above threshold
            if abs(signal_strength) < self.prediction_threshold:
                return

            # Calculate confidence and risk multiplier based on signal strength
            confidence = min(1.0, abs(signal_strength) * 2)  # Scale to 0-1
            risk_multiplier = min(2.0, 1.0 + abs(signal_strength))  # Cap at 2x

            # Long signal
            if signal_strength > self.prediction_threshold:
                _logger.info("LONG signal - Signal Strength: %.3f, Confidence: %.3f, Price: %.4f",
                            signal_strength, confidence, self.data.close[0])

                # Calculate position size
                position_size = self._calculate_position_size(confidence, risk_multiplier)

                # Execute buy order
                self.buy(size=position_size)

                # Set stop loss and take profit
                self._set_exit_orders(position_size, 'long')

            # Short signal
            elif signal_strength < -self.prediction_threshold:
                _logger.info("SHORT signal - Signal Strength: %.3f, Confidence: %.3f, Price: %.4f",
                            signal_strength, confidence, self.data.close[0])

                # Calculate position size
                position_size = self._calculate_position_size(confidence, risk_multiplier)

                # Execute sell order
                self.sell(size=position_size)

                # Set stop loss and take profit
                self._set_exit_orders(position_size, 'short')

        except Exception as e:
            _logger.exception("Error in entry signal check:")

    def _check_exit_signals(self, predictions: Dict[str, float], signal_strength: float):
        """Check for exit signals."""
        try:
            position_size = self.position.size

            if position_size == 0:
                return

            exit_signal = False
            exit_reason = ""

            # Check various exit conditions
            if position_size > 0:  # Long position
                # 1. Signal reversal
                if signal_strength < -self.prediction_threshold:
                    exit_signal = True
                    exit_reason = "signal_reversal"

                # 2. Direction prediction changes
                elif 'target_direction' in predictions and predictions['target_direction'] < 0.3:
                    exit_signal = True
                    exit_reason = "direction_change"

                # 3. Low confidence in all predictions
                elif all(pred < 0.4 for pred in predictions.values()):
                    exit_signal = True
                    exit_reason = "low_confidence"

            elif position_size < 0:  # Short position
                # 1. Signal reversal
                if signal_strength > self.prediction_threshold:
                    exit_signal = True
                    exit_reason = "signal_reversal"

                # 2. Direction prediction changes
                elif 'target_direction' in predictions and predictions['target_direction'] > 0.7:
                    exit_signal = True
                    exit_reason = "direction_change"

                # 3. Low confidence in all predictions
                elif all(pred < 0.4 for pred in predictions.values()):
                    exit_signal = True
                    exit_reason = "low_confidence"

            # Execute exit if signal detected
            if exit_signal:
                _logger.info("EXIT signal - Reason: %s, Signal Strength: %.3f, Price: %.4f",
                            exit_reason, signal_strength, self.data.close[0])
                self.close()

        except Exception as e:
            _logger.exception("Error in exit signal check:")

    def next(self):
        """Main strategy logic executed on each bar."""
        try:
            # Update equity curve
            self._update_equity_curve()

            # Get predictions from models
            predictions = self._get_predictions()

            if predictions:
                # Calculate overall signal strength
                signal_strength = self._calculate_signal_strength(predictions)

                # Log predictions occasionally
                if len(self.data) % 50 == 0:
                                    _logger.debug("Predictions - Direction: %.3f, Volatility: %.3f, Trend: %.3f, Magnitude: %.3f, Signal: %.3f",
                             predictions.get('target_direction', 0), predictions.get('target_volatility', 0),
                             predictions.get('target_trend', 0), predictions.get('target_magnitude', 0), signal_strength)

                # Check for exit signals first
                self._check_exit_signals(predictions, signal_strength)

                # Check for entry signals
                self._check_entry_signals(predictions, signal_strength)

        except Exception as e:
            _logger.exception("Error in strategy next():")

    def _set_exit_orders(self, position_size: float, position_type: str):
        """Set stop loss and take profit orders."""
        try:
            current_price = self.data.close[0]

            if position_type == 'long':
                # Stop loss
                stop_price = current_price * (1 - self.stop_loss)
                self.sell(exectype=bt.Order.Stop, price=stop_price, size=position_size)

                # Take profit
                profit_price = current_price * (1 + self.profit_target)
                self.sell(exectype=bt.Order.Limit, price=profit_price, size=position_size)

            elif position_type == 'short':
                # Stop loss
                stop_price = current_price * (1 + self.stop_loss)
                self.buy(exectype=bt.Order.Stop, price=stop_price, size=position_size)

                # Take profit
                profit_price = current_price * (1 - self.profit_target)
                self.buy(exectype=bt.Order.Limit, price=profit_price, size=position_size)

        except Exception as e:
            _logger.exception("Error setting exit orders:")

    def _calculate_position_size(self, confidence: float, risk_multiplier: float) -> float:
        """Calculate position size based on confidence and risk."""
        try:
            # Base position size
            base_size = self.config.get('base_position_size', 0.1)

            # Adjust based on confidence and risk multiplier
            adjusted_size = base_size * confidence * risk_multiplier

            # Apply position size limits
            min_size = self.config.get('min_position_size', 0.01)
            max_size = self.config.get('max_position_size', 0.5)

            return max(min_size, min(max_size, adjusted_size))

        except Exception as e:
            _logger.exception("Error calculating position size:")
            return self.config.get('base_position_size', 0.1)
