"""
Trailing Stop Exit Mixin

This module implements a trailing stop exit strategy that dynamically adjusts the stop loss
level as the price moves in favor of the position. The strategy exits when:
1. Price falls below the trailing stop level
2. The trailing stop level is calculated as: highest_price * (1 - trail_pct)
3. Optionally, the trailing stop can be based on ATR for dynamic adjustment

Parameters:
    trail_pct (float): Percentage for trailing stop calculation (default: 0.02)
    activation_pct (float): Minimum profit percentage before trailing stop activates (default: 0.0)
    use_atr (bool): Whether to use ATR for dynamic trailing stop (default: False)
    atr_multiplier (float): Multiplier for ATR-based trailing stop (default: 2.0)
    use_talib (bool): Whether to use TA-Lib for ATR calculation (default: False)

This strategy is particularly effective for:
1. Capturing trends while protecting profits
2. Letting winners run while managing risk
3. Adapting to market volatility when using ATR-based stops
4. Preventing premature exits in strong trends
"""

from typing import Any, Dict, Optional

import backtrader as bt
from src.exit.base_exit_mixin import BaseExitMixin
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


class TrailingStopExitMixin(BaseExitMixin):
    """Exit mixin based on trailing stop"""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the mixin with parameters"""
        super().__init__(params)
        self.highest_price = 0
        self.atr_name = "exit_atr"
        self.atr = None

    def get_required_params(self) -> list:
        """There are no required parameters - all have default values"""
        return []

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """Default parameters"""
        return {
            "x_trail_pct": 0.02,
            "x_activation_pct": 0.0,
            "x_use_atr": False,
            "x_atr_multiplier": 2.0,
            "x_atr_period": 14,
            "x_use_talib": False,
        }

    def _init_indicators(self):
        """Initialize indicators"""
        logger.debug("TrailingStopExitMixin._init_indicators called")
        if not hasattr(self, "strategy"):
            logger.error("No strategy available in _init_indicators")
            return

        try:
            atr_period = self.get_param("x_atr_period")
            if self.strategy.use_talib:
                self.atr = bt.talib.ATR(
                    self.strategy.data.high,
                    self.strategy.data.low,
                    self.strategy.data.close,
                    timeperiod=atr_period,
                )
            else:
                self.atr = bt.indicators.AverageTrueRange(
                    self.strategy.data, period=atr_period
                )
            self.register_indicator(self.atr_name, self.atr)
        except Exception as e:
            logger.error(f"Error initializing indicators: {e}", exc_info=e)
            raise

    def should_exit(self) -> bool:
        """Check if we should exit a position"""
        if not self.strategy.position:
            return False

        try:
            price = self.strategy.data.close[0]
            entry_price = self.strategy.position.price

            # Update highest price if current price is higher
            if price > self.highest_price:
                self.highest_price = price

            # Calculate trailing stop level
            if self.get_param("x_use_atr", False):
                if self.atr_name not in self.indicators:
                    return False
                atr = self.indicators[self.atr_name]
                atr_val = atr[0] if hasattr(atr, "__getitem__") else atr.lines.atr[0]
                stop_level = self.highest_price - (
                    atr_val * self.get_param("x_atr_multiplier")
                )
            else:
                stop_level = self.highest_price * (1 - self.get_param("x_trail_pct"))

            # Check if trailing stop should be activated
            if self.get_param("x_activation_pct", 0.0) > 0:
                profit_pct = (price - entry_price) / entry_price
                if profit_pct < self.get_param("x_activation_pct"):
                    return False

            # Exit if price falls below trailing stop
            return_value = price < stop_level
            if return_value:
                if self.get_param("x_use_atr", False):
                    logger.debug(
                        f"EXIT: Price: {price}, Entry: {entry_price}, "
                        f"Highest: {self.highest_price}, Stop: {stop_level}, "
                        f"ATR: {atr_val}, ATR Multiplier: {self.get_param('x_atr_multiplier')}"
                    )
                else:
                    logger.debug(
                        f"EXIT: Price: {price}, Entry: {entry_price}, "
                        f"Highest: {self.highest_price}, Stop: {stop_level}, "
                        f"Trail %: {self.get_param('x_trail_pct')}"
                    )
                self.strategy.current_exit_reason = "trailing_stop"
            return return_value
        except Exception as e:
            logger.error(f"Error in should_exit: {e}", exc_info=e)
            return False

    def get_exit_reason(self) -> str:
        """Get the reason for exiting the position"""
        if not self.strategy.position:
            return "unknown"

        price = self.strategy.data.close[0]
        entry_price = self.strategy.position.price
        profit_pct = (price - entry_price) / entry_price

        # If trailing stop is not activated yet
        if self.get_param("x_activation_pct", 0.0) > 0 and profit_pct < self.get_param(
            "x_activation_pct"
        ):
            return "unknown"

        return "trailing_stop"
