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
from src.strategy.exit.base_exit_mixin import BaseExitMixin
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
            "x_require_confirmation": False,
            "x_use_talib": True,
        }

    def init_exit(self, strategy, additional_params: Optional[Dict[str, Any]] = None):
        """Override to detect architecture mode before calling parent."""
        # Detect architecture: new if strategy has indicators dict with entries
        if hasattr(strategy, 'indicators') and strategy.indicators:
            self.use_new_architecture = True
            logger.debug("Using new TALib-based architecture")
        else:
            self.use_new_architecture = False
            logger.debug("Using legacy architecture")

        # Call parent init_exit which will call _init_indicators
        super().init_exit(strategy, additional_params)

    def _init_indicators(self):
        """Initialize indicators (legacy architecture only).

        In new architecture, indicators are created by the strategy
        and accessed via get_indicator().
        """
        if self.use_new_architecture:
            # New architecture: indicators already created by strategy
            return

        # Legacy architecture: create indicators in mixin
        logger.debug("MACrossoverExitMixin._init_indicators called (legacy architecture)")
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
                self.fast_ma = bt.indicators.SMA(
                    self.strategy.data.close, period=fast_period
                )
                self.slow_ma = bt.indicators.SMA(
                    self.strategy.data.close, period=slow_period
                )

            self.register_indicator(self.fast_ma_name, self.fast_ma)
            self.register_indicator(self.slow_ma_name, self.slow_ma)

            logger.debug("Legacy indicators initialized: Fast MA(period=%d), Slow MA(period=%d)",
                        fast_period, slow_period)

        except Exception:
            logger.exception("Error initializing indicators: ")
            raise

    def are_indicators_ready(self) -> bool:
        """Check if indicators are ready to be used"""
        if self.use_new_architecture:
            # New architecture: check strategy's indicators
            if not hasattr(self.strategy, 'indicators') or not self.strategy.indicators:
                return False

            # Check if required indicators exist
            required_indicators = ['exit_fast_ma', 'exit_slow_ma']
            for ind_alias in required_indicators:
                if ind_alias not in self.strategy.indicators:
                    return False

            # Check if we can access values
            try:
                _ = self.get_indicator('exit_fast_ma')
                _ = self.get_indicator('exit_slow_ma')
                return True
            except (IndexError, KeyError, AttributeError):
                return False

        else:
            # Legacy architecture: check mixin's indicators
            return super().are_indicators_ready()

    def should_exit(self) -> bool:
        """Check if we should exit a position.

        Works with both new and legacy architectures.
        """
        if not self.strategy.position:
            return False

        if not self.are_indicators_ready():
            return False

        try:
            # Get indicator values based on architecture
            if self.use_new_architecture:
                # New architecture: access via get_indicator()
                fast_ma_current = self.get_indicator('exit_fast_ma')
                slow_ma_current = self.get_indicator('exit_slow_ma')
                fast_ma_prev = self.get_indicator_prev('exit_fast_ma', 1)
                slow_ma_prev = self.get_indicator_prev('exit_slow_ma', 1)

            else:
                # Legacy architecture: access via mixin's indicators dict
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
        except Exception:
            logger.exception("Error in should_exit: ")
            return False

    def get_exit_reason(self) -> str:
        """Get the reason for exit"""
        return getattr(self.strategy, 'current_exit_reason', 'ma_crossover')