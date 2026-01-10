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

from typing import Any, Dict, Optional

import backtrader as bt
from src.strategy.exit.base_exit_mixin import BaseExitMixin
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


class MACrossoverExitMixin(BaseExitMixin):
    """Exit mixin based on Moving Average crossovers.

    Supports both new TALib-based architecture and legacy configurations.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the mixin with parameters"""
        super().__init__(params)

        # Legacy architecture support
        self.fast_ma_name = "exit_fast_ma"
        self.slow_ma_name = "exit_slow_ma"
        self.fast_ma = None
        self.slow_ma = None

        # Detect architecture mode
        self.use_new_architecture = False  # Will be set in init_exit()

    def get_required_params(self) -> list:
        """There are no required parameters - all have default values"""
        return []

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """Default parameters"""
        return {
            "x_fast_period": 10,
            "x_slow_period": 20,
            "x_ma_type": "sma",
        }

    def init_exit(self, strategy, additional_params: Optional[Dict[str, Any]] = None):
        """Override to detect architecture mode before calling parent."""
        if hasattr(strategy, 'indicators') and strategy.indicators:
            self.use_new_architecture = True
        else:
            self.use_new_architecture = False

        super().init_exit(strategy, additional_params)

    def _init_indicators(self):
        """Initialize indicators (legacy architecture only)."""
        if self.use_new_architecture:
            return

        if not hasattr(self, "strategy"):
            logger.error("No strategy available in _init_indicators")
            return

        try:
            fast_period = self.get_param("x_fast_period")
            slow_period = self.get_param("x_slow_period")

            if self.strategy.use_talib:
                self.fast_ma = bt.talib.SMA(self.strategy.data.close, fast_period)
                self.slow_ma = bt.talib.SMA(self.strategy.data.close, slow_period)
            else:
                self.fast_ma = bt.indicators.SMA(self.strategy.data.close, period=fast_period)
                self.slow_ma = bt.indicators.SMA(self.strategy.data.close, period=slow_period)

            self.register_indicator(self.fast_ma_name, self.fast_ma)
            self.register_indicator(self.slow_ma_name, self.slow_ma)

        except Exception:
            logger.exception("Error initializing indicators: ")
            raise

    def get_minimum_lookback(self) -> int:
        """Returns the minimum number of bars required."""
        if self.use_new_architecture:
            return max(
                self.get_param("fast_period", 10),
                self.get_param("slow_period", 20)
            )
        else:
            return max(
                self.get_param("x_fast_period", 10),
                self.get_param("x_slow_period", 20)
            )

    def are_indicators_ready(self) -> bool:
        """Check if indicators are initialized."""
        if self.use_new_architecture:
            required = ['exit_fast_ma', 'exit_slow_ma']
            return all(alias in getattr(self.strategy, 'indicators', {}) for alias in required)
        else:
            return self.fast_ma_name in self.indicators and self.slow_ma_name in self.indicators

    def should_exit(self) -> bool:
        """Check if we should exit a position."""
        if not self.strategy.position:
            return False

        if not self.are_indicators_ready():
            return False

        try:
            # Unified Indicator Access
            if self.use_new_architecture:
                fast_ma_current = self.get_indicator('exit_fast_ma')
                slow_ma_current = self.get_indicator('exit_slow_ma')
                fast_ma_prev = self.get_indicator_prev('exit_fast_ma', 1)
                slow_ma_prev = self.get_indicator_prev('exit_slow_ma', 1)
            else:
                fast_ma = self.indicators[self.fast_ma_name]
                slow_ma = self.indicators[self.slow_ma_name]
                fast_ma_current = fast_ma[0]
                slow_ma_current = slow_ma[0]
                fast_ma_prev = fast_ma[-1]
                slow_ma_prev = slow_ma[-1]

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
                ma_type = self.get_param("ma_type") or self.get_param("x_ma_type", "sma")
                self.strategy.current_exit_reason = f"{ma_type.lower()}_crossover"
            return return_value

        except Exception:
            logger.exception("Error in MACrossoverExitMixin.should_exit")
            return False

    def get_exit_reason(self) -> str:
        """Get the reason for exit"""
        return getattr(self.strategy, 'current_exit_reason', 'ma_crossover')