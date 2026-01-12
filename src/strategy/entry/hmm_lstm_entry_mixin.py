"""
HMM-LSTM Entry Mixin

This mixin provides entry signals based on HMM regime detection and LSTM predictions.
"""

import numpy as np
import torch
import torch.nn as nn
import pickle
import json
import talib
from pathlib import Path
from collections import deque
from typing import Dict, Tuple, Any, Optional, List
import sys
import os

from src.strategy.entry.base_entry_mixin import BaseEntryMixin
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

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size + n_regimes, output_size)

    def forward(self, x, regime_onehot):
        device = x.device
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        lstm_out, _ = self.lstm(x, (h0, c0))
        last_output = lstm_out[:, -1, :]
        dropped = self.dropout(last_output)
        combined = torch.cat([dropped, regime_onehot], dim=1)
        output = self.linear(combined)
        return output


class HMMLSTMEntryMixin(BaseEntryMixin):
    """
    Entry mixin that uses HMM regime detection and LSTM predictions.
    New Architecture only.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.symbol = self.get_param('symbol', 'BTCUSDT')
        self.timeframe = self.get_param('timeframe', '1h')
        self.prediction_threshold = self.get_param('prediction_threshold', 0.001)
        self.regime_confidence_threshold = self.get_param('regime_confidence_threshold', 0.6)
        self.models_dir = Path(self.get_param('models_dir', 'src/ml/pipeline/p01_hmm_lstm/models'))
        self.results_dir = Path(self.get_param('results_dir', 'results'))

        self.hmm_model = None
        self.lstm_model = None
        self.hmm_scaler = None
        self.lstm_scalers = None
        self.optimized_indicators = None
        self.lstm_features = None
        self.hmm_features = None

        self.feature_buffer = deque(maxlen=100)
        self.lstm_sequence_buffer = None
        self.sequence_length = 60

        self.current_regime = None
        self.regime_confidence = 0.0
        self.current_prediction = 0.0
        self.is_initialized = False

    def get_required_params(self) -> list:
        return []

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "prediction_threshold": 0.001,
            "regime_confidence_threshold": 0.6
        }

    @classmethod
    def get_indicator_config(cls, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        HMM/LSTM uses custom models, not standard TALib indicators.
        Returns empty list to signal no auto-provisioning needed.
        """
        return []

    def _init_indicators(self):
        """No-op for new architecture."""
        pass

    def init_entry(self, strategy, additional_params=None):
        super().init_entry(strategy, additional_params)
        try:
            self._load_models()
            self.is_initialized = True
        except Exception:
            _logger.exception("Failed to load HMM-LSTM models")
            self.is_initialized = False

    def _load_models(self):
        """Load trained HMM and LSTM models."""
        # Existing loading logic remains same
        try:
            hmm_pattern = f"hmm_{self.symbol}_{self.timeframe}_*.pkl"
            hmm_files = list((self.models_dir / 'hmm').glob(hmm_pattern))
            if hmm_files:
                hmm_file = sorted(hmm_files)[-1]
                with open(hmm_file, 'rb') as f:
                    hmm_package = pickle.load(f)
                self.hmm_model = hmm_package['model']
                self.hmm_scaler = hmm_package['scaler']
                self.hmm_features = hmm_package['features']

            lstm_pattern = f"lstm_{self.symbol}_{self.timeframe}_*.pkl"
            lstm_files = list((self.models_dir / 'lstm').glob(lstm_pattern))
            if lstm_files:
                lstm_file = sorted(lstm_files)[-1]
                with open(lstm_file, 'rb') as f:
                    lstm_package = pickle.load(f)
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
                self.lstm_sequence_buffer = deque(maxlen=self.sequence_length)

            indicator_pattern = f"indicators_{self.symbol}_{self.timeframe}_*.json"
            indicator_files = list(self.results_dir.glob(indicator_pattern))
            if indicator_files:
                indicator_file = sorted(indicator_files)[-1]
                with open(indicator_file, 'r') as f:
                    indicator_results = json.load(f)
                self.optimized_indicators = indicator_results['best_params']
            else:
                self.optimized_indicators = {}
        except Exception:
            raise

    def get_minimum_lookback(self) -> int:
        """Returns the minimum number of bars required (100 for heavy indicator calculation)."""
        return 100

    def _calculate_indicators(self) -> Dict[str, float]:
        """Calculate technical indicators for the current bar using TALib."""
        # Existing calculation logic remains same
        try:
            lookback = 100
            if len(self.strategy.data) < lookback: return {}
            high = np.array([self.strategy.data.high[-i] for i in range(lookback, 0, -1)])
            low = np.array([self.strategy.data.low[-i] for i in range(lookback, 0, -1)])
            close = np.array([self.strategy.data.close[-i] for i in range(lookback, 0, -1)])

            indicators = {}
            params = self.optimized_indicators

            rsi = talib.RSI(close, timeperiod=params.get('rsi_period', 14))
            indicators['rsi_optimized'] = rsi[-1] if not np.isnan(rsi[-1]) else 50.0

            bb_period, bb_std = params.get('bb_period', 20), params.get('bb_std', 2.0)
            u, m, l = talib.BBANDS(close, timeperiod=bb_period, nbdevup=bb_std, nbdevdn=bb_std)
            if not np.isnan(u[-1]):
                indicators['bb_position_opt'] = (close[-1] - l[-1]) / (u[-1] - l[-1]) if (u[-1] - l[-1]) != 0 else 0.5

            return indicators
        except Exception:
            return {}

    def are_indicators_ready(self) -> bool:
        """Check if models and data are ready for predictions."""
        if not self.is_initialized or not self.hmm_model or not self.lstm_model:
            return False
        if not self.strategy or len(self.strategy.data) < 100:
            return False
        if self.lstm_sequence_buffer and len(self.lstm_sequence_buffer) < self.sequence_length:
            return False
        return True

    def should_enter(self) -> bool:
        """Determine if entry signal is present."""
        if not self.are_indicators_ready():
            return False
        try:
            current_features = {
                'open': self.strategy.data.open[0],
                'high': self.strategy.data.high[0],
                'low': self.strategy.data.low[0],
                'close': self.strategy.data.close[0],
                'volume': self.strategy.data.volume[0],
                'log_return': np.log(self.strategy.data.close[0] / self.strategy.data.close[-1]) if self.strategy.data.close[-1] > 0 else 0.0
            }
            indicators = self._calculate_indicators()
            current_features.update(indicators)

            regime, conf = self._predict_regime(current_features)
            prediction = self._predict_price(current_features, regime)

            self.current_regime, self.regime_confidence, self.current_prediction = regime, conf, prediction

            return conf >= self.regime_confidence_threshold and prediction >= self.prediction_threshold
        except Exception:
            return False

    def _predict_regime(self, features: Dict[str, float]) -> Tuple[int, float]:
        """Predict market regime using HMM."""
        try:
            if not self.hmm_model: return 1, 0.5
            vals = [features.get(f, 0.0) for f in self.hmm_features]
            self.feature_buffer.append(vals)
            if len(self.feature_buffer) < 10: return 1, 0.5
            scaled = self.hmm_scaler.transform([vals])
            regime = self.hmm_model.predict(scaled)[0]
            posteriors = self.hmm_model.predict_proba(scaled)[0]
            return int(regime), float(np.max(posteriors))
        except Exception:
            return 1, 0.5

    def _predict_price(self, features: Dict[str, float], regime: int) -> float:
        """Predict next price change using LSTM."""
        try:
            if not self.lstm_model: return 0.0
            vals = [features.get(f, 0.0) for f in self.lstm_features]
            self.lstm_sequence_buffer.append(vals)
            if len(self.lstm_sequence_buffer) < self.sequence_length: return 0.0
            seq = np.array(list(self.lstm_sequence_buffer))
            seq_scaled = self.lstm_scalers['feature_scaler'].transform(seq).reshape(1, self.sequence_length, -1)
            r_onehot = np.zeros((1, 3))
            if 0 <= regime < 3: r_onehot[0, regime] = 1
            with torch.no_grad():
                pred = self.lstm_model(torch.tensor(seq_scaled, dtype=torch.float32), torch.tensor(r_onehot, dtype=torch.float32)).item()
                return float(self.lstm_scalers['target_scaler'].inverse_transform([[pred]])[0][0])
        except Exception:
            return 0.0

    def get_current_state(self) -> Dict:
        return {
            'regime': self.current_regime,
            'prediction': self.current_prediction,
            'ready': self.are_indicators_ready()
        }
