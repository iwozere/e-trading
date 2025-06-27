"""
MA Crossover Exit Mixin

This module implements an exit strategy based on Moving Average crossovers.
The strategy exits a position when:
1. Fast MA crosses below Slow MA for long positions
2. Fast MA crosses above Slow MA for short positions

Parameters:
    fast_period (int): Period for fast MA (default: 10)
    slow_period (int): Period for slow MA (default: 20)
    ma_type (str): Type of MA to use ('SMA', 'EMA', 'WMA', 'DEMA', 'TEMA', 'TRIMA', 'KAMA', 'MAMA', 'T3') (default: 'SMA')
    require_confirmation (bool): Whether to require confirmation of crossover (default: False)
    use_talib (bool): Whether to use TA-Lib for calculations (default: True)

This strategy is particularly effective for:
1. Trend following exit signals
2. Reducing false signals with confirmation
3. Adapting to different market conditions with different MA types
"""

from typing import Any, Dict, Optional

import backtrader as bt
import numpy as np
from src.exit.base_exit_mixin import BaseExitMixin
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


class MACrossoverExitMixin(BaseExitMixin):
    """Exit mixin based on Moving Average crossovers"""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the mixin with parameters"""
        super().__init__(params)
        self.fast_ma_name = "exit_fast_ma"
        self.slow_ma_name = "exit_slow_ma"
        self.fast_ma = None
        self.slow_ma = None

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
            "x_require_confirmation": False,
            "x_use_talib": True,
        }

    def _init_indicators(self):
        """Initialize indicators"""
        logger.debug("MACrossoverExitMixin._init_indicators called")
        if not hasattr(self, "strategy"):
            logger.error("No strategy available in _init_indicators")
            return

        try:
            fast_period = self.get_param("x_fast_period")
            slow_period = self.get_param("x_slow_period")

            if self.strategy.use_talib:
                self.fast_ma = bt.talib.SMA(self.strategy.data.volume, fast_period)
                self.slow_ma = bt.talib.SMA(self.strategy.data.volume, slow_period)
            else:
                self.fast_ma = bt.indicators.SMA(
                    self.strategy.data.volume, period=fast_period
                )
                self.slow_ma = bt.indicators.SMA(
                    self.strategy.data.volume, period=slow_period
                )

            self.register_indicator(self.fast_ma_name, self.fast_ma)
            self.register_indicator(self.slow_ma_name, self.slow_ma)

        except Exception as e:
            logger.error(f"Error initializing indicators: {e}", exc_info=e)
            raise

    def should_exit(self) -> bool:
        """Check if we should exit a position"""
        if not self.strategy.position:
            return False

        if (
            self.fast_ma_name not in self.indicators
            or self.slow_ma_name not in self.indicators
        ):
            return False

        try:
            # Get indicators from mixin's indicators dictionary
            fast_ma = self.indicators[self.fast_ma_name]
            slow_ma = self.indicators[self.slow_ma_name]

            # Get current and previous values
            fast_ma_current = fast_ma[0]
            slow_ma_current = slow_ma[0]
            fast_ma_prev = fast_ma[-1]
            slow_ma_prev = slow_ma[-1]

            # Check for crossover based on position
            if self.strategy.position.size > 0:  # Long position
                return_value = (
                    fast_ma_prev > slow_ma_prev and fast_ma_current < slow_ma_current
                )
            else:  # Short position
                return_value = (
                    fast_ma_prev < slow_ma_prev and fast_ma_current > slow_ma_current
                )

            if return_value:
                logger.debug(
                    f"EXIT: Price: {self.strategy.data.close[0]}, "
                    f"Fast MA: {fast_ma_current}, Slow MA: {slow_ma_current}, "
                    f"Position: {'long' if self.strategy.position.size > 0 else 'short'}"
                )
                ma_type = self.get_param("x_ma_type", "sma").lower()
                self.strategy.current_exit_reason = f"{ma_type}_crossover"
            return return_value
        except Exception as e:
            logger.error(f"Error in should_exit: {e}", exc_info=e)
            return False
