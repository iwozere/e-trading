"""
ATR Exit Mixin
-------------

This module implements a trailing stop loss strategy based on Average True Range (ATR).
The strategy uses ATR to dynamically adjust the stop loss level based on market volatility.
The stop loss trails the price upward, never moving down, allowing profits to run while protecting against reversals.

Trailing Stop Logic:
------------------
- Tracks the highest price since entry
- Stop loss is calculated as: highest_price - (ATR * sl_multiplier)
- Stop loss is trailing, meaning it only moves up, never down
- Stop loss follows the price with an ATR-based gap for protection
- Example: If sl_multiplier is 2.0, stop loss trails 2 ATR below the highest price

Parameters:
-----------
x_atr_period : int
    Period for ATR calculation (default: 14)
x_sl_multiplier : float
    Multiplier for ATR to determine stop loss distance (default: 2.0)

Exit Reasons:
-----------
- "stop_loss": When price falls below the trailing stop loss
"""

from typing import Any, Dict, Optional

import backtrader as bt
from src.strategy.exit.base_exit_mixin import BaseExitMixin
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


class ATRExitMixin(BaseExitMixin):
    """
    Exit mixin that implements a trailing stop loss strategy using ATR.

    The strategy uses ATR to dynamically adjust the stop loss level based on market volatility.
    The stop loss trails the price upward, never moving down, allowing profits to run while protecting against reversals.

    Trailing Stop Loss:
    - Tracks the highest price since entry
    - Distance from highest price is determined by ATR * sl_multiplier
    - Stop loss only moves up, never down
    - Once triggered, resets for next trade

    All variables are reset when a position is closed.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the mixin with parameters"""
        super().__init__(params)
        self.atr_name = "exit_atr"
        self.atr = None
        self.stop_loss = None
        self.highest_price = None

    def get_required_params(self) -> list:
        """There are no required parameters - all have default values"""
        return []

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """Default parameters"""
        return {
            "x_atr_period": 14,
            "x_sl_multiplier": 2.0,
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
            logger.exception("Error initializing indicators: ")
            raise

    def on_entry(self, entry_price: float, entry_time, position_size: float, direction: str):
        """Called when a position is entered"""
        logger.debug("ATRExitMixin.on_entry: price=%s, direction=%s, size=%s", entry_price, direction, position_size)

        # Reset variables for new position
        self.highest_price = entry_price  # Start with entry price as highest
        self.stop_loss = None

        # Validate entry price
        import math
        if math.isnan(entry_price) or math.isinf(entry_price):
            logger.warning("Invalid entry price: %s, skipping ATR exit setup", entry_price)
            return

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

            # Track the highest price since entry
            if self.highest_price is None:
                self.highest_price = current_high
            else:
                self.highest_price = max(self.highest_price, current_high)

            # Calculate trailing stop loss from highest price
            sl_multiplier = self.get_param("x_sl_multiplier")
            new_stop_loss = self.highest_price - (atr_val * sl_multiplier)

            # Initialize or update trailing stop loss (only moves up, never down)
            if self.stop_loss is None:
                self.stop_loss = new_stop_loss
                logger.debug("ATR Stop Loss initialized: %s (Highest: %s, ATR: %s, Multiplier: %s)", self.stop_loss, self.highest_price, atr_val, sl_multiplier)
            else:
                # Only update if the new stop loss is higher (trailing up)
                old_stop_loss = self.stop_loss
                self.stop_loss = max(self.stop_loss, new_stop_loss)
                if self.stop_loss > old_stop_loss:
                    logger.debug("ATR Stop Loss updated: %s -> %s (Highest: %s)", old_stop_loss, self.stop_loss, self.highest_price)

            # Check if current price has fallen below the trailing stop loss
            if current_price <= self.stop_loss:
                logger.debug(
                    f"EXIT: Current Price: {current_price}, Stop Loss: {self.stop_loss}, "
                    f"Highest Price: {self.highest_price}, ATR: {atr_val}, ATR Multiplier: {sl_multiplier}"
                )
                # Set the exit reason in the strategy
                self.strategy.current_exit_reason = "stop_loss"
                return True

            return False
        except Exception as e:
            logger.exception("Error in should_exit: ")
            return False

    def next(self):
        """Called for each new bar"""
        super().next()
        # Reset variables when position is closed
        if not self.strategy.position:
            self.highest_price = None
            self.stop_loss = None

    def get_exit_reason(self) -> str:
        """Get the reason for exit"""
        return getattr(self.strategy, 'current_exit_reason', 'atr_exit')