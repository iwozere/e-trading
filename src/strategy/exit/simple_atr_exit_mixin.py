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

from typing import Any, Dict, Optional, List

import backtrader as bt
from src.strategy.exit.base_exit_mixin import BaseExitMixin
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


class SimpleATRExitMixin(BaseExitMixin):
    """Simple ATR-based trailing stop exit strategy.
    New Architecture only.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the mixin with parameters"""
        super().__init__(params)
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
            "atr_period": 14,
            "atr_multiplier": 2.0,
            "breakeven_atr": 1.0,
            "use_breakeven": True,
        }

    @classmethod
    def get_indicator_config(cls, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Define indicators required by this mixin."""
        atr_period = params.get("atr_period") or params.get("x_atr_period", 14)
        return [
            {
                "type": "ATR",
                "params": {"timeperiod": atr_period},
                "fields_mapping": {"atr": "exit_atr"}
            }
        ]

    def _init_indicators(self):
        """No-op for new architecture."""
        pass

    def get_minimum_lookback(self) -> int:
        """Returns the minimum number of bars required."""
        return self.get_param("atr_period") or self.get_param("x_atr_period", 14)

    def are_indicators_ready(self) -> bool:
        """Check if required indicators exist in the strategy registry."""
        return 'exit_atr' in getattr(self.strategy, 'indicators', {})

    def on_entry(self, entry_price: float, entry_time, position_size: float, direction: str):
        """Called when a position is entered"""
        self.breakeven_triggered = False

        if self.are_indicators_ready():
            atr_value = self.get_indicator('exit_atr')
            atr_multiplier = self.get_param("atr_multiplier") or self.get_param("x_atr_multiplier", 2.0)

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

            atr_value = self.get_indicator('exit_atr')
            use_breakeven = self.get_param("use_breakeven") or self.get_param("x_use_breakeven", True)
            breakeven_atr = self.get_param("breakeven_atr") or self.get_param("x_breakeven_atr", 1.0)
            atr_multiplier = self.get_param("atr_multiplier") or self.get_param("x_atr_multiplier", 2.0)

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
