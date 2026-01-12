"""
MA Crossover Exit Mixin

This module implements an exit strategy based on Moving Average crossovers.
The strategy exits a position when:
1. Fast MA crosses below Slow MA for long positions
2. Fast MA crosses above Slow MA for short positions

Configuration Example (New TALib Architecture):
    {
        "exit_logic": {
            "name": "MACrossoverExitMixin",
            "indicators": [
                {
                    "type": "SMA",
                    "params": {"timeperiod": 10},
                    "fields_mapping": {"sma": "exit_fast_ma"}
                },
                {
                    "type": "SMA",
                    "params": {"timeperiod": 20},
                    "fields_mapping": {"sma": "exit_slow_ma"}
                }
            ],
            "logic_params": {
                "ma_type": "SMA"
            }
        }
    }
"""

from typing import Any, Dict, Optional, List

import backtrader as bt
from src.strategy.exit.base_exit_mixin import BaseExitMixin
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


class MACrossoverExitMixin(BaseExitMixin):
    """Exit mixin based on Moving Average crossovers.
    New Architecture only.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the mixin with parameters"""
        super().__init__(params)

    def get_required_params(self) -> list:
        """There are no required parameters - all have default values"""
        return []

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """Default parameters"""
        return {
            "fast_period": 10,
            "slow_period": 20,
            "ma_type": "SMA",
        }

    @classmethod
    def get_indicator_config(cls, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Define indicators required by this mixin."""
        fast_period = params.get("fast_period") or params.get("x_fast_period", 10)
        slow_period = params.get("slow_period") or params.get("x_slow_period", 20)
        ma_type = (params.get("ma_type") or params.get("x_ma_type", "SMA")).upper()

        return [
            {
                "type": ma_type,
                "params": {"timeperiod": fast_period},
                "fields_mapping": {ma_type.lower(): "exit_fast_ma"}
            },
            {
                "type": ma_type,
                "params": {"timeperiod": slow_period},
                "fields_mapping": {ma_type.lower(): "exit_slow_ma"}
            }
        ]

    def _init_indicators(self):
        """No-op for new architecture."""
        pass

    def get_minimum_lookback(self) -> int:
        """Returns the minimum number of bars required."""
        return max(
            self.get_param("fast_period", 10),
            self.get_param("slow_period", 20)
        )

    def are_indicators_ready(self) -> bool:
        """Check if required indicators exist in the strategy registry."""
        required = ['exit_fast_ma', 'exit_slow_ma']
        return all(alias in getattr(self.strategy, 'indicators', {}) for alias in required)

    def should_exit(self) -> bool:
        """Check if we should exit a position."""
        if not self.strategy.position:
            return False

        if not self.are_indicators_ready():
            return False

        try:
            # Unified Indicator Access
            fast_ma_current = self.get_indicator('exit_fast_ma')
            slow_ma_current = self.get_indicator('exit_slow_ma')
            fast_ma_prev = self.get_indicator_prev('exit_fast_ma', 1)
            slow_ma_prev = self.get_indicator_prev('exit_slow_ma', 1)

            # Check for crossover based on position
            if self.strategy.position.size > 0:  # Long position
                return_value = (fast_ma_prev > slow_ma_prev and fast_ma_current < slow_ma_current)
            else:  # Short position
                return_value = (fast_ma_prev < slow_ma_prev and fast_ma_current > slow_ma_current)

            if return_value:
                logger.debug(
                    f"EXIT Crossover - Fast: {fast_ma_current:.2f}, Slow: {slow_ma_current:.2f}, "
                    f"Prev Fast: {fast_ma_prev:.2f}, Prev Slow: {slow_ma_prev:.2f}"
                )
                ma_type = self.get_param("ma_type") or self.get_param("x_ma_type", "SMA")
                self.strategy.current_exit_reason = f"{ma_type.lower()}_crossover"
            return return_value

        except Exception:
            logger.exception("Error in MACrossoverExitMixin.should_exit")
            return False

    def get_exit_reason(self) -> str:
        """Get the reason for exit"""
        return getattr(self.strategy, 'current_exit_reason', 'ma_crossover')

    def get_exit_reason(self) -> str:
        """Get the reason for exit"""
        return getattr(self.strategy, 'current_exit_reason', 'ma_crossover')