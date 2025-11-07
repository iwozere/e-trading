"""
ATR Exit Mixin
-------------

This module implements a trailing stop loss strategy based on Average True Range (ATR).
The ATR value is locked at position entry to maintain consistent stop loss distance.
The stop loss trails the price upward, never moving down, allowing profits to run while protecting against reversals.

Trailing Stop Logic:
------------------
- ATR value is fixed at position entry (prevents stop loss distance from widening)
- Tracks the highest price since entry
- Stop loss is calculated as: highest_price - (entry_ATR * sl_multiplier)
- Stop loss is trailing, meaning it only moves up, never down
- Stop loss follows the price with a fixed ATR-based gap for protection
- Example: If entry ATR is 500 and sl_multiplier is 2.0, stop loss trails 1000 points below the highest price

Configuration Example (New TALib Architecture):
    {
        "exit_logic": {
            "name": "ATRExitMixin",
            "indicators": [
                {
                    "type": "ATR",
                    "params": {"timeperiod": 14},
                    "fields_mapping": {"atr": "exit_atr"}
                }
            ],
            "logic_params": {
                "sl_multiplier": 2.0
            }
        }
    }

Legacy Configuration (Backward Compatible):
    {
        "exit_logic": {
            "name": "ATRExitMixin",
            "params": {
                "x_atr_period": 14,
                "x_sl_multiplier": 2.0
            }
        }
    }

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

    The ATR value is locked at position entry to maintain consistent stop loss distance.
    The stop loss trails the price upward, never moving down, allowing profits to run while protecting against reversals.

    Trailing Stop Loss:
    - ATR value is fixed at entry (entry_atr)
    - Tracks the highest price since entry
    - Distance from highest price is determined by entry_ATR * sl_multiplier
    - Stop loss only moves up, never down
    - Once triggered, resets for next trade

    Supports both new TALib-based architecture (indicators created by strategy)
    and legacy architecture (indicators created by mixin).

    All variables (entry_atr, highest_price, stop_loss) are reset when a position is closed.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the mixin with parameters"""
        super().__init__(params)
        self.stop_loss = None
        self.highest_price = None
        self.entry_atr = None  # Fixed ATR value at position entry

        # Legacy architecture support
        self.atr_name = "exit_atr"
        self.atr = None

        # Detect architecture mode
        self.use_new_architecture = False  # Will be set in init_exit()

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
            #logger.debug("Skipping indicator initialization (new architecture)")
            return

        # Legacy architecture: create indicators in mixin
        logger.debug("ATRExitMixin._init_indicators called (legacy architecture)")
        if not hasattr(self, "strategy"):
            logger.error("No strategy available in _init_indicators")
            return

        try:
            atr_period = self.get_param("x_atr_period")

            # Validate ATR period
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

            logger.debug("Legacy indicators initialized: ATR(period=%d)", atr_period)

        except Exception as e:
            logger.exception("Error initializing indicators: ")
            raise

    def are_indicators_ready(self) -> bool:
        """Check if indicators are ready to be used"""
        if self.use_new_architecture:
            # New architecture: check strategy's indicators
            if not hasattr(self.strategy, 'indicators') or not self.strategy.indicators:
                return False

            # Check if exit_atr exists
            if 'exit_atr' not in self.strategy.indicators:
                return False

            # Check if we can access value
            try:
                _ = self.get_indicator('exit_atr')
                return True
            except (IndexError, KeyError, AttributeError):
                return False

        else:
            # Legacy architecture: use base class implementation
            return super().are_indicators_ready()

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

        # Lock ATR value at entry (Option 1: Fixed ATR)
        try:
            if self.use_new_architecture:
                self.entry_atr = self.get_indicator('exit_atr')
            else:
                atr = self.indicators[self.atr_name]
                self.entry_atr = atr[0] if hasattr(atr, "__getitem__") else atr.lines.atr[0]

            logger.debug("ATR locked at entry: %.2f", self.entry_atr)
        except Exception as e:
            logger.warning("Failed to lock ATR at entry: %s", e)
            self.entry_atr = None

    def should_exit(self) -> bool:
        """Check if we should exit a position.

        Works with both new and legacy architectures.
        """
        if not self.are_indicators_ready():
            return False

        try:
            current_price = self.strategy.data.close[0]
            current_high = self.strategy.data.high[0]

            # Use fixed ATR from entry (Option 1: Fixed ATR)
            # If entry_atr is not set, fall back to current ATR
            if self.entry_atr is not None and self.entry_atr > 0:
                atr_val = self.entry_atr
            else:
                # Fallback to current ATR if entry_atr not available
                if self.use_new_architecture:
                    atr_val = self.get_indicator('exit_atr')
                else:
                    atr = self.indicators[self.atr_name]
                    atr_val = atr[0] if hasattr(atr, "__getitem__") else atr.lines.atr[0]

            # Get sl_multiplier based on architecture
            if self.use_new_architecture:
                sl_multiplier = self.get_param("sl_multiplier") or self.get_param("x_sl_multiplier", 2.0)
            else:
                sl_multiplier = self.get_param("x_sl_multiplier", 2.0)

            # Validate ATR value
            if atr_val is None or atr_val <= 0:
                logger.warning("Invalid ATR value: %s, skipping exit check", atr_val)
                return False

            # Track the highest price since entry
            if self.highest_price is None:
                self.highest_price = current_high
            else:
                self.highest_price = max(self.highest_price, current_high)

            # Calculate trailing stop loss from highest price
            new_stop_loss = self.highest_price - (atr_val * sl_multiplier)

            # Initialize or update trailing stop loss (only moves up, never down)
            if self.stop_loss is None:
                self.stop_loss = new_stop_loss
                logger.debug(
                    "ATR Stop Loss initialized: %.2f (Highest: %.2f, ATR: %.2f, Multiplier: %.2f)",
                    self.stop_loss, self.highest_price, atr_val, sl_multiplier
                )
            else:
                # Only update if the new stop loss is higher (trailing up)
                old_stop_loss = self.stop_loss
                self.stop_loss = max(self.stop_loss, new_stop_loss)
                if self.stop_loss > old_stop_loss:
                    logger.debug(
                        "ATR Stop Loss updated: %.2f -> %.2f (Highest: %.2f)",
                        old_stop_loss, self.stop_loss, self.highest_price
                    )

            # Check if current price has fallen below the trailing stop loss
            if current_price <= self.stop_loss:
                logger.debug(
                    f"EXIT SIGNAL - Current Price: {current_price:.2f}, Stop Loss: {self.stop_loss:.2f}, "
                    f"Highest Price: {self.highest_price:.2f}, ATR: {atr_val:.2f}, ATR Multiplier: {sl_multiplier:.2f}"
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
            self.entry_atr = None

    def get_exit_reason(self) -> str:
        """Get the reason for exit"""
        return getattr(self.strategy, 'current_exit_reason', 'atr_exit')