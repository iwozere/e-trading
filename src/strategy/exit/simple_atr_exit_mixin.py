"""
Simple ATR-Based Exit Mixin

This module implements a simplified, effective ATR-based trailing stop exit strategy.

Configuration Example (New TALib Architecture):
    {
        "exit_logic": {
            "name": "SimpleATRExitMixin",
            "indicators": [
                {
                    "type": "ATR",
                    "params": {"timeperiod": 14},
                    "fields_mapping": {"atr": "exit_atr"}
                }
            ],
            "logic_params": {
                "atr_multiplier": 2.0,
                "use_breakeven": true,
                "breakeven_atr": 1.0
            }
        }
    }
"""

from typing import Any, Dict, Optional

import backtrader as bt
from src.strategy.exit.base_exit_mixin import BaseExitMixin
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


class SimpleATRExitMixin(BaseExitMixin):
    """Simple ATR-based trailing stop exit strategy.

    Supports both new TALib-based architecture and legacy configurations.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the mixin with parameters"""
        super().__init__(params)
        self.atr_name = "exit_atr"
        self.atr = None
        self.initial_stop = None
        self.current_stop = None
        self.breakeven_triggered = False

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
            "x_atr_multiplier": 2.0,
            "x_breakeven_atr": 1.0,
            "x_use_breakeven": True,
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
            from src.indicators.adapters.backtrader_wrappers import UnifiedATRIndicator

            atr_period = self.get_param("x_atr_period")
            backend = "bt-talib" if self.strategy.use_talib else "bt"

            self.atr = UnifiedATRIndicator(
                self.strategy.data,
                period=atr_period,
                backend=backend
            )
            self.register_indicator(self.atr_name, self.atr)

        except Exception:
            logger.exception("Error initializing indicators: ")
            raise

    def get_minimum_lookback(self) -> int:
        """Returns the minimum number of bars required."""
        if self.use_new_architecture:
            return self.get_param("atr_period", 14)
        else:
            return self.get_param("x_atr_period", 14)

    def are_indicators_ready(self) -> bool:
        """Check if indicators are initialized."""
        if self.use_new_architecture:
            return 'exit_atr' in getattr(self.strategy, 'indicators', {})
        else:
            return self.atr_name in self.indicators

    def on_entry(self, entry_price: float, entry_time, position_size: float, direction: str):
        """Called when a position is entered"""
        self.breakeven_triggered = False

        if self.are_indicators_ready():
            if self.use_new_architecture:
                atr_value = self.get_indicator('exit_atr')
                atr_multiplier = self.get_param("atr_multiplier") or self.get_param("x_atr_multiplier", 2.0)
            else:
                atr_value = self.indicators[self.atr_name].atr[0]
                atr_multiplier = self.get_param("x_atr_multiplier", 2.0)

            if atr_value is not None and atr_value > 0:
                if direction.lower() == "long":
                    self.initial_stop = entry_price - (atr_value * atr_multiplier)
                else:
                    self.initial_stop = entry_price + (atr_value * atr_multiplier)
                self.current_stop = self.initial_stop
                logger.debug(f"Initial ATR stop: {self.current_stop:.2f}")

    def should_exit(self) -> bool:
        """Check if we should exit a position."""
        if not self.strategy.position:
            return False

        if not self.are_indicators_ready():
            return False

        try:
            current_price = self.strategy.data.close[0]

            if self.use_new_architecture:
                atr_value = self.get_indicator('exit_atr')
                use_breakeven = self.get_param("use_breakeven") or self.get_param("x_use_breakeven", True)
                breakeven_atr = self.get_param("breakeven_atr") or self.get_param("x_breakeven_atr", 1.0)
                atr_multiplier = self.get_param("atr_multiplier") or self.get_param("x_atr_multiplier", 2.0)
            else:
                atr_value = self.indicators[self.atr_name].atr[0]
                use_breakeven = self.get_param("x_use_breakeven", True)
                breakeven_atr = self.get_param("x_breakeven_atr", 1.0)
                atr_multiplier = self.get_param("x_atr_multiplier", 2.0)

            if atr_value is None or atr_value <= 0:
                return False

            # Breakeven logic
            if use_breakeven and not self.breakeven_triggered and self.initial_stop is not None:
                entry_price = getattr(self.strategy, 'entry_price', None)
                if entry_price is not None:
                    profit_threshold = atr_value * breakeven_atr
                    if self.strategy.position.size > 0:
                        if current_price >= entry_price + profit_threshold:
                            self.current_stop = entry_price
                            self.breakeven_triggered = True
                    else:
                        if current_price <= entry_price - profit_threshold:
                            self.current_stop = entry_price
                            self.breakeven_triggered = True

            # Trailing stop update
            if self.current_stop is not None:
                if self.strategy.position.size > 0:
                    new_stop = current_price - (atr_value * atr_multiplier)
                    if new_stop > self.current_stop:
                        self.current_stop = new_stop
                else:
                    new_stop = current_price + (atr_value * atr_multiplier)
                    if new_stop < self.current_stop:
                        self.current_stop = new_stop

            # Check exit
            if self.current_stop is not None:
                should_exit_trade = False
                if self.strategy.position.size > 0:
                    should_exit_trade = current_price <= self.current_stop
                else:
                    should_exit_trade = current_price >= self.current_stop

                if should_exit_trade:
                    logger.debug(f"Simple ATR Exit - Price: {current_price:.2f}, Stop: {self.current_stop:.2f}")
                    self.strategy.current_exit_reason = "atr_trailing_stop"
                    return True

            return False

        except Exception:
            logger.exception("Error in SimpleATRExitMixin.should_exit")
            return False

    def get_exit_reason(self) -> str:
        """Get the reason for exit"""
        return getattr(self.strategy, 'current_exit_reason', 'atr_trailing_stop')
