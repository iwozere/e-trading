"""
EOM Rejection Exit Mixin

This module implements SELL #2: Resistance Reject + EOM Turns Negative (Momentum Reversal Down)

Exit Signal Logic:
------------------
Fade failed breakout / mean reversion down.

Conditions (all must be true):
1. Price rejection at resistance: High >= Resistance * resistance_threshold AND Close < Open (bearish rejection candle)
2. EOM crosses below 0: bearish EOM momentum reversal
3. RSI overbought → falling: RSI > rsi_overbought AND RSI falling (RSI[0] < RSI[-1])

Configuration Example (New architecture):
    {
        "exit_logic": {
            "name": "EOMRejectionExitMixin",
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
                    "type": "EOM",
                    "params": {"timeperiod": 14},
                    "fields_mapping": {"eom": "exit_eom"}
                },
                {
                    "type": "RSI",
                    "params": {"timeperiod": 14},
                    "fields_mapping": {"rsi": "exit_rsi"}
                }
            ],
            "logic_params": {
                "resistance_threshold": 0.995,
                "rsi_overbought": 60
            }
        }
    }
"""

from typing import Any, Dict, Optional, List
import math

from src.strategy.exit.base_exit_mixin import BaseExitMixin
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class EOMRejectionExitMixin(BaseExitMixin):
    """
    Exit mixin for SELL #2: Resistance Reject + EOM Turns Negative.
    New Architecture only.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the mixin with parameters"""
        super().__init__(params)
        self._exit_reason = ""

    def get_required_params(self) -> list:
        """There are no required parameters - all have default values"""
        return []

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """Default parameters"""
        return {
            "resistance_threshold": 0.995,
            "rsi_overbought": 60,
            "eom_period": 14,
            "rsi_period": 14,
            "resistance_lookback": 2,
        }

    @classmethod
    def get_indicator_config(cls, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Define indicators required by this mixin."""
        res_lookback = params.get("resistance_lookback", 2)
        eom_period = params.get("eom_period", 14)
        rsi_period = params.get("rsi_period", 14)

        return [
            {
                "type": "SupportResistance",
                "params": {"lookback_bars": res_lookback},
                "fields_mapping": {"resistance": "exit_resistance", "support": "exit_support"}
            },
            {
                "type": "EOM",
                "params": {"timeperiod": eom_period},
                "fields_mapping": {"eom": "exit_eom"}
            },
            {
                "type": "RSI",
                "params": {"timeperiod": rsi_period},
                "fields_mapping": {"rsi": "exit_rsi"}
            }
        ]

    def _init_indicators(self):
        """No-op for new architecture."""
        pass

    def get_minimum_lookback(self) -> int:
        """Returns the minimum number of bars required."""
        periods = [
            self.get_param("eom_period", 14),
            self.get_param("rsi_period", 14),
            self.get_param("resistance_lookback", 2) * 5
        ]
        return max(periods)

    def are_indicators_ready(self) -> bool:
        """Check if required indicators exist in the strategy registry."""
        required = [
            'exit_resistance', 'exit_eom', 'exit_rsi'
        ]
        return all(alias in getattr(self.strategy, 'indicators', {}) for alias in required)

    def should_exit(self) -> bool:
        """Determines if the mixin should exit a position."""
        if not self.strategy.position:
            return False

        if not self.are_indicators_ready():
            return False

        try:
            # Get current price data
            close = self.strategy.data.close[0]
            open_price = self.strategy.data.open[0]
            high = self.strategy.data.high[0]

            # Unified Indicator Access
            resistance = self.get_indicator('exit_resistance')
            eom = self.get_indicator('exit_eom')
            eom_prev = self.get_indicator_prev('exit_eom', 1)
            rsi = self.get_indicator('exit_rsi')
            rsi_prev = self.get_indicator_prev('exit_rsi', 1)

            # Get parameters
            resistance_threshold = self.get_param("resistance_threshold") or self.get_param("x_resistance_threshold", 0.995)
            rsi_overbought = self.get_param("rsi_overbought") or self.get_param("x_rsi_overbought", 60)

            # Check if resistance is valid (not NaN)
            if math.isnan(resistance):
                return False

            # 1. Price rejection at resistance
            resistance_level = resistance * resistance_threshold
            is_at_resistance = high >= resistance_level
            is_rejection_candle = close < open_price

            if not (is_at_resistance and is_rejection_candle):
                return False

            # 2. EOM crosses below 0
            is_eom_crossing_down = eom < 0 and eom_prev >= 0

            if not is_eom_crossing_down:
                return False

            # 3. RSI overbought → falling
            is_rsi_overbought = rsi > rsi_overbought
            is_rsi_falling = rsi < rsi_prev

            if not (is_rsi_overbought and is_rsi_falling):
                return False

            # All conditions met
            self._exit_reason = (
                f"resistance_rejection: close={close:.2f}, high={high:.2f}, "
                f"resistance={resistance:.2f}, rsi={rsi:.2f}, eom={eom:.4f}"
            )
            _logger.info("EOM Rejection Exit signal: %s", self._exit_reason)
            return True

        except Exception:
            _logger.exception("Error in EOMRejectionExitMixin.should_exit")
            return False

    def get_exit_reason(self) -> str:
        """Get the reason for exit"""
        return self._exit_reason if self._exit_reason else "resistance_rejection"

    def get_exit_reason(self) -> str:
        """Get the reason for exit"""
        return self._exit_reason if self._exit_reason else "resistance_rejection"
