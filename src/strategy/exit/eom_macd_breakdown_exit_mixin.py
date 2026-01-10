"""
EOM MACD Breakdown Exit Mixin

This module implements SELL #3: MACD Bearish Turn + Breakdown (Trend + Structure Confirm)

Exit Signal Logic:
------------------
Combine strong trend shift + structural breakdown.

Conditions (all must be true):
1. MACD bearish cross: MACD line crosses below Signal line AND Histogram falling
2. Breakdown confirmation: Close < Support * (1 - support_threshold)
3. EOM negative: EOM < 0
4. Volume > SMA (breakdowns on high volume are more reliable)

Configuration Example (New architecture):
    {
        "exit_logic": {
            "name": "EOMMAcdBreakdownExitMixin",
            "indicators": [
                {
                    "type": "SupportResistance",
                    "params": {"lookback_bars": 2},
                    "fields_mapping": {
                        "resistance": "exit_resistance",
                        "support": "exit_support"
                    }
                },
                {
                    "type": "MACD",
                    "params": {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
                    "fields_mapping": {
                        "macd": "exit_macd",
                        "signal": "exit_macd_signal",
                        "hist": "exit_macd_hist"
                    }
                },
                {
                    "type": "EOM",
                    "params": {"timeperiod": 14},
                    "fields_mapping": {"eom": "exit_eom"}
                },
                {
                    "type": "SMA",
                    "params": {"timeperiod": 20},
                    "data_field": "volume",
                    "fields_mapping": {"sma": "exit_volume_sma"}
                }
            ],
            "logic_params": {
                "support_threshold": 0.002
            }
        }
    }
"""

from typing import Any, Dict, Optional
import math

from src.strategy.exit.base_exit_mixin import BaseExitMixin
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class EOMMAcdBreakdownExitMixin(BaseExitMixin):
    """
    Exit mixin for SELL #3: MACD Bearish Turn + Breakdown.

    Purpose: Combine strong trend shift + structural breakdown.
    Supports both new architecture and legacy configurations.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the mixin with parameters"""
        super().__init__(params)
        self.use_new_architecture = False
        self._exit_reason = ""

    def get_required_params(self) -> list:
        """There are no required parameters - all have default values"""
        return []

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """Default parameters"""
        return {
            "x_support_threshold": 0.002,
            "macd_fast_period": 12,
            "macd_slow_period": 26,
            "macd_signal_period": 9,
            "eom_period": 14,
            "vol_ma_period": 20,
            "resistance_lookback": 2,
        }

    def init_exit(self, strategy, additional_params: Optional[Dict[str, Any]] = None):
        """Standardize architecture detection."""
        if hasattr(strategy, 'indicators') and strategy.indicators:
            self.use_new_architecture = True
        else:
            self.use_new_architecture = False
        super().init_exit(strategy, additional_params)

    def _init_indicators(self):
        """Indicators managed by strategy in unified architecture."""
        pass

    def get_minimum_lookback(self) -> int:
        """Returns the minimum number of bars required."""
        periods = [
            self.get_param("macd_slow_period", 26) + self.get_param("macd_signal_period", 9),
            self.get_param("eom_period", 14),
            self.get_param("vol_ma_period", 20),
            self.get_param("resistance_lookback", 2) * 5
        ]
        return max(periods)

    def are_indicators_ready(self) -> bool:
        """Check if indicators are initialized."""
        required = [
            'exit_support', 'exit_macd', 'exit_macd_signal', 'exit_macd_hist', 'exit_eom', 'exit_volume_sma'
        ]
        if hasattr(self.strategy, 'indicators') and self.strategy.indicators:
            return all(alias in self.strategy.indicators for alias in required)
        return False

    def should_exit(self) -> bool:
        """
        Determines if the mixin should exit a position.

        Returns:
            bool: True if all exit conditions are met
        """
        if not self.strategy.position:
            return False

        if not self.are_indicators_ready():
            return False

        try:
            # Get current price data
            close = self.strategy.data.close[0]
            volume = self.strategy.data.volume[0]

            # Unified Indicator Access
            support = self.get_indicator('exit_support')
            macd = self.get_indicator('exit_macd')
            macd_signal = self.get_indicator('exit_macd_signal')
            macd_hist = self.get_indicator('exit_macd_hist')

            macd_prev = self.get_indicator_prev('exit_macd', 1)
            macd_signal_prev = self.get_indicator_prev('exit_macd_signal', 1)
            macd_hist_prev = self.get_indicator_prev('exit_macd_hist', 1)

            eom = self.get_indicator('exit_eom')
            volume_sma = self.get_indicator('exit_volume_sma')

            # Get parameters
            support_threshold = self.get_param("support_threshold") or self.get_param("x_support_threshold", 0.002)

            # Check if support is valid (not NaN)
            if math.isnan(support):
                return False

            # 1. MACD bearish cross: MACD line crosses below Signal line AND histogram falling
            is_macd_crossunder = macd < macd_signal and macd_prev >= macd_signal_prev
            is_hist_falling = macd_hist < macd_hist_prev

            if not (is_macd_crossunder and is_hist_falling):
                return False

            # 2. Breakdown confirmation: Close < Support * (1 - threshold)
            breakdown_level = support * (1 - support_threshold)
            is_breakdown = close < breakdown_level

            if not is_breakdown:
                return False

            # 3. EOM negative: EOM < 0
            is_eom_negative = eom < 0

            if not is_eom_negative:
                return False

            # 4. Volume > SMA
            is_volume_confirmed = volume > volume_sma

            if not is_volume_confirmed:
                return False

            # All conditions met
            self._exit_reason = (
                f"macd_breakdown: close={close:.2f}, support={support:.2f}, "
                f"macd={macd:.4f}, signal={macd_signal:.4f}, eom={eom:.4f}, vol={volume:.0f}"
            )
            _logger.info("EOM MACD Breakdown Exit signal: %s", self._exit_reason)
            return True

        except KeyError as e:
            _logger.error("Required indicator not found: %s", e)
            return False
        except Exception:
            _logger.exception("Error in EOMMAcdBreakdownExitMixin.should_exit")
            return False

    def get_exit_reason(self) -> str:
        """Get the reason for exit"""
        return self._exit_reason if self._exit_reason else "macd_breakdown"
