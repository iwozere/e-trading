"""
Simple ATR-Based Exit Mixin

This module implements a simplified, effective ATR-based trailing stop exit strategy.
It focuses on the core functionality without the complexity of the AdvancedATRExitMixin.

Key features:
- Simple ATR-based trailing stop
- Break-even functionality
- Configurable ATR multiplier
- Clean and efficient implementation

Parameters:
    x_atr_period (int): Period for ATR calculation (default: 14)
    x_atr_multiplier (float): ATR multiplier for stop distance (default: 2.0)
    x_breakeven_atr (float): ATR multiple to move stop to breakeven (default: 1.0)
    x_use_breakeven (bool): Whether to use breakeven functionality (default: True)
"""

from typing import Any, Dict, Optional
import backtrader as bt
from src.strategy.exit.base_exit_mixin import BaseExitMixin
from src.indicators.adapters.backtrader_wrappers import UnifiedATRIndicator
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


class SimpleATRExitMixin(BaseExitMixin):
    """Simple ATR-based trailing stop exit strategy."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the mixin with parameters"""
        super().__init__(params)
        self.atr_name = "exit_atr"
        self.atr = None
        self.initial_stop = None
        self.current_stop = None
        self.breakeven_triggered = False

    def get_required_params(self) -> list:
        """There are no required parameters - all have default values"""
        return []

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """Default parameters"""
        return {
            "x_atr_period": 14,
            "x_atr_multiplier": 2.0,
            "x_breakeven_atr": 1.0,
            "x_use_breakeven": True,
        }

    def _init_indicators(self):
        """Initialize indicators"""
        logger.debug("SimpleATRExitMixin._init_indicators called")
        if not hasattr(self, "strategy"):
            logger.error("No strategy available in _init_indicators")
            return

        try:
            atr_period = self.get_param("x_atr_period")

            # Create unified ATR indicator directly
            backend = "bt-talib" if self.strategy.use_talib else "bt"

            self.atr = UnifiedATRIndicator(
                self.strategy.data,
                period=atr_period,
                backend=backend
            )
            self.register_indicator(self.atr_name, self.atr)

        except Exception as e:
            logger.exception("Error initializing indicators: ")
            raise

    def on_entry(self, entry_price: float, direction: str, size: float, entry_reason: str):
        """Called when a position is entered"""
        logger.debug("SimpleATRExitMixin.on_entry: price=%s, direction=%s, size=%s", entry_price, direction, size)

        # Validate entry price
        import math
        if math.isnan(entry_price) or math.isinf(entry_price):
            logger.warning("Invalid entry price: %s, skipping ATR exit setup", entry_price)
            return

        self.breakeven_triggered = False

        # Calculate initial stop based on ATR
        if self.atr_name in self.indicators:
            atr_value = self.atr.atr[0]
            if atr_value is not None and atr_value > 0:
                atr_multiplier = self.get_param("x_atr_multiplier")

                if direction.lower() == "long":
                    self.initial_stop = entry_price - (atr_value * atr_multiplier)
                    self.current_stop = self.initial_stop
                else:  # short
                    self.initial_stop = entry_price + (atr_value * atr_multiplier)
                    self.current_stop = self.initial_stop

                logger.debug("Initial ATR stop set: %s (ATR: %s, multiplier: %s)", self.current_stop, atr_value, atr_multiplier)
            else:
                logger.warning("ATR value is None or zero, cannot set initial stop")
        else:
            logger.warning("ATR indicator not available for stop calculation")

    def should_exit(self) -> bool:
        """Check if we should exit a position"""
        if not self.strategy.position:
            return False

        if self.atr_name not in self.indicators:
            return False

        try:
            current_price = self.strategy.data.close[0]
            atr_value = self.atr.atr[0]

            if current_price is None or atr_value is None or atr_value <= 0:
                return False

            # Check if we should move to breakeven
            if (self.get_param("x_use_breakeven") and
                not self.breakeven_triggered and
                self.initial_stop is not None):

                entry_price = getattr(self.strategy, 'entry_price', None)
                if entry_price is not None:
                    breakeven_atr = self.get_param("x_breakeven_atr")
                    profit_threshold = atr_value * breakeven_atr

                    if self.strategy.position.size > 0:  # long position
                        if current_price >= entry_price + profit_threshold:
                            self.current_stop = entry_price
                            self.breakeven_triggered = True
                            logger.debug("Moved to breakeven: %s", self.current_stop)
                    else:  # short position
                        if current_price <= entry_price - profit_threshold:
                            self.current_stop = entry_price
                            self.breakeven_triggered = True
                            logger.debug("Moved to breakeven: %s", self.current_stop)

            # Update trailing stop
            if self.current_stop is not None:
                atr_multiplier = self.get_param("x_atr_multiplier")

                if self.strategy.position.size > 0:  # long position
                    new_stop = current_price - (atr_value * atr_multiplier)
                    if new_stop > self.current_stop:
                        self.current_stop = new_stop
                        logger.debug("Updated long stop: %s", self.current_stop)
                else:  # short position
                    new_stop = current_price + (atr_value * atr_multiplier)
                    if new_stop < self.current_stop:
                        self.current_stop = new_stop
                        logger.debug("Updated short stop: %s", self.current_stop)

            # Check exit condition
            if self.current_stop is not None:
                if self.strategy.position.size > 0:  # long position
                    should_exit = current_price <= self.current_stop
                else:  # short position
                    should_exit = current_price >= self.current_stop

                if should_exit:
                    logger.debug("ATR exit triggered: price=%s, stop=%s", current_price, self.current_stop)
                    self.strategy.current_exit_reason = "atr_trailing_stop"

                return should_exit

            return False

        except Exception as e:
            logger.exception("Error in should_exit: ")
            return False

    def get_exit_reason(self) -> str:
        """Get the reason for exit"""
        return getattr(self.strategy, 'current_exit_reason', 'atr_trailing_stop')
