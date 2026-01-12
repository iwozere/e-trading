"""
EOM Breakdown Exit Mixin

This module implements SELL #1: Breakdown + EOM Negative (Momentum Breakdown)

Exit Signal Logic:
------------------
Opposite of Buy #1 — strong bearish momentum.

Conditions (all must be true):
1. Breakdown: Close < Support * (1 - breakdown_threshold)
2. EOM bearish: EOM < 0 AND EOM falling (EOM[0] < EOM[-1])
3. Volume confirmation: Volume > Volume_SMA
4. ATR confirmation: ATR rising vs yesterday

Configuration Example (New architecture):
    {
        "exit_logic": {
            "name": "EOMBreakdownExitMixin",
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
                    "type": "SMA",
                    "params": {"timeperiod": 20},
                    "data_field": "volume",
                    "fields_mapping": {"sma": "exit_volume_sma"}
                },
                {
                    "type": "ATR",
                    "params": {"timeperiod": 14},
                    "fields_mapping": {"atr": "exit_atr"}
                }
            ],
            "logic_params": {
                "breakdown_threshold": 0.002
            }
        }
    }
"""

from typing import Any, Dict, Optional, List
import math

from src.strategy.exit.base_exit_mixin import BaseExitMixin
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class EOMBreakdownExitMixin(BaseExitMixin):
    """
    Exit mixin for SELL #1: Breakdown + EOM Negative.
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
            "breakdown_threshold": 0.002,
            "eom_period": 14,
            "vol_ma_period": 20,
            "atr_period": 14,
            "resistance_lookback": 2,
        }

    @classmethod
    def get_indicator_config(cls, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Define indicators required by this mixin."""
        res_lookback = params.get("resistance_lookback", 2)
        eom_period = params.get("eom_period", 14)
        vol_ma_period = params.get("vol_ma_period", 20)
        atr_period = params.get("atr_period", 14)

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
                "type": "SMA",
                "params": {"timeperiod": vol_ma_period},
                "data_field": "volume",
                "fields_mapping": {"sma": "exit_volume_sma"}
            },
            {
                "type": "ATR",
                "params": {"timeperiod": atr_period},
                "fields_mapping": {"atr": "exit_atr"}
            }
        ]

    def _init_indicators(self):
        """No-op for new architecture."""
        pass

    def get_minimum_lookback(self) -> int:
        """Returns the minimum number of bars required."""
        periods = [
            self.get_param("eom_period", 14),
            self.get_param("vol_ma_period", 20),
            self.get_param("atr_period", 14),
            self.get_param("resistance_lookback", 2) * 5
        ]
        return max(periods)

    def are_indicators_ready(self) -> bool:
        """Check if required indicators exist in the strategy registry."""
        required = [
            'exit_support', 'exit_eom', 'exit_volume_sma', 'exit_atr'
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
            volume = self.strategy.data.volume[0]

            # Unified Indicator Access
            support = self.get_indicator('exit_support')
            eom = self.get_indicator('exit_eom')
            eom_prev = self.get_indicator_prev('exit_eom', 1)
            volume_sma = self.get_indicator('exit_volume_sma')
            atr = self.get_indicator('exit_atr')
            atr_prev = self.get_indicator_prev('exit_atr', 1)

            # Get parameters
            breakdown_threshold = self.get_param("breakdown_threshold") or self.get_param("x_breakdown_threshold", 0.002)

            # Check if support is valid (not NaN)
            if math.isnan(support):
                return False

            # 1. Breakdown: Close < Support * (1 - threshold)
            breakdown_level = support * (1 - breakdown_threshold)
            is_breakdown = close < breakdown_level

            if not is_breakdown:
                return False

            # 2. EOM bearish: EOM < 0 AND EOM falling
            is_eom_bearish = eom < 0 and eom < eom_prev

            if not is_eom_bearish:
                return False

            # 3. Volume confirmation: Volume > Volume_SMA
            is_volume_confirmed = volume > volume_sma

            if not is_volume_confirmed:
                return False

            # 4. ATR confirmation: ATR rising
            is_atr_rising = atr > atr_prev

            if not is_atr_rising:
                return False

            # All conditions met
            self._exit_reason = (
                f"breakdown_momentum: close={close:.2f}, support={support:.2f}, "
                f"eom={eom:.4f}, volume={volume:.0f}, vol_sma={volume_sma:.0f}"
            )
            _logger.info("EOM Breakdown Exit signal: %s", self._exit_reason)
            return True

        except Exception:
            _logger.exception("Error in EOMBreakdownExitMixin.should_exit")
            return False

    def get_exit_reason(self) -> str:
        """Get the reason for exit"""
        return self._exit_reason if self._exit_reason else "breakdown_momentum"

    def get_exit_reason(self) -> str:
        """Get the reason for exit"""
        return self._exit_reason if self._exit_reason else "breakdown_momentum"
