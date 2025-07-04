"""
ATR Exit Mixin
-------------

This module implements an exit strategy based on Average True Range (ATR) with both stop loss and take profit levels.
The strategy uses ATR to dynamically adjust stop loss and take profit levels based on market volatility.

Stop Loss Logic:
--------------
- Stop loss is calculated as: current_high - (ATR * sl_multiplier)
- Stop loss is trailing, meaning it only moves up, never down
- Once set, stop loss remains at its highest level until triggered

Take Profit Logic:
---------------
- Take profit is calculated as: entry_price + (entry_price * tp_multiplier)
- Take profit is fixed at entry and doesn't trail
- Example: If tp_multiplier is 2.0, take profit is set at 200% of entry price

Parameters:
-----------
atr_period : int
    Period for ATR calculation (default: 14)
sl_multiplier : float
    Multiplier for ATR to determine stop loss distance (default: 1.0)
tp_multiplier : float
    Multiplier for entry price to determine take profit level (default: 2.0)
use_talib : bool
    Whether to use TA-Lib for calculations (default: True)

Exit Reasons:
-----------
- "stop_loss": When price falls below the trailing stop loss
- "take_profit": When price rises above the take profit level
"""

from typing import Any, Dict, Optional

import backtrader as bt
from src.strategy.exit.base_exit_mixin import BaseExitMixin
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


class ATRExitMixin(BaseExitMixin):
    """
    Exit mixin that implements a trailing stop loss and fixed take profit strategy using ATR.

    The strategy uses ATR to dynamically adjust the stop loss level based on market volatility,
    while maintaining a fixed take profit level based on the entry price.

    Stop Loss:
    - Trailing stop loss that only moves up
    - Distance from current high is determined by ATR * sl_multiplier
    - Once triggered, resets for next trade

    Take Profit:
    - Fixed level set at entry
    - Level is entry_price * (1 + tp_multiplier)
    - Once triggered, resets for next trade

    Both stop loss and take profit levels are reset when a position is closed.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the mixin with parameters"""
        super().__init__(params)
        self.atr_name = "exit_atr"
        self.atr = None
        self.stop_loss = None
        self.take_profit = None

    def get_required_params(self) -> list:
        """There are no required parameters - all have default values"""
        return []

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """Default parameters"""
        return {
            "x_atr_period": 14,
            "x_tp_multiplier": 2.0,
            "x_sl_multiplier": 1.0,
        }

    def _init_indicators(self):
        """Initialize indicators"""
        logger.debug("ATRExitMixin._init_indicators called")
        if not hasattr(self, "strategy"):
            logger.error("No strategy available in _init_indicators")
            return

        try:
            atr_period = self.get_param("x_atr_period")

            # Validate ATR period to prevent division by zero
            if atr_period is None or atr_period <= 0:
                logger.warning("Invalid ATR period: %s, using default value 14", atr_period)
                atr_period = 14
            elif atr_period < 2:
                logger.warning("ATR period too small: %s, using minimum value 2", atr_period)
                atr_period = 2

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
        if not self.are_indicators_ready():
            return False

        try:
            # Get indicator from mixin's indicators dictionary
            atr = self.indicators[self.atr_name]
            atr_val = atr[0] if hasattr(atr, "__getitem__") else atr.lines.atr[0]
            current_price = self.strategy.data.close[0]
            current_high = self.strategy.data.high[0]

            # Defensive check: Ensure ATR value is valid and not zero
            if atr_val is None or atr_val <= 0:
                logger.warning("Invalid ATR value: %s, skipping exit check", atr_val)
                return False

            # Calculate stop loss from highest price
            sl_multiplier = self.get_param("x_sl_multiplier")
            stop_loss = current_high - (atr_val * sl_multiplier)

            if self.stop_loss is None:
                self.stop_loss = stop_loss
            elif self.stop_loss < stop_loss:
                self.stop_loss = stop_loss

            if current_price < stop_loss:
                logger.debug(
                    f"EXIT: Current Price: {current_price}, Stop Loss: {stop_loss}, ATR: {atr_val}, ATR Multiplier: {sl_multiplier}"
                )
                # Set the exit reason in the strategy
                self.strategy.current_exit_reason = "stop_loss"
                self.stop_loss = None
                self.take_profit = None
                return True

            # Check take profit logic
            if self.take_profit is None and self.strategy.current_trade is not None:
                entry_price = self.strategy.current_trade["entry_price"]
                tp_multiplier = self.get_param("x_tp_multiplier")
                take_profit = entry_price + (entry_price * tp_multiplier)
                self.take_profit = take_profit

            # Check if take profit is set and current price exceeds it
            if self.take_profit is not None and current_price > self.take_profit:
                logger.debug(
                    f"EXIT: Current Price: {current_price}, Take Profit: {self.take_profit}, ATR: {atr_val}, TP Multiplier: {self.get_param('x_tp_multiplier')}"
                )
                # Set the exit reason in the strategy
                self.strategy.current_exit_reason = "take_profit"
                self.stop_loss = None
                self.take_profit = None
                return True

            return False
        except Exception as e:
            logger.error(f"Error in should_exit: {e}", exc_info=e)
            return False

    def next(self):
        """Called for each new bar"""
        super().next()
        # Reset highest price when position is closed
        if not self.strategy.position:
            self.highest_price = None
