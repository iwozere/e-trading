"""
ATR Exit Mixin

This module implements a trailing stop loss strategy based on Average True Range (ATR).
The ATR value is locked at position entry to maintain consistent stop loss distance.
The stop loss trails the price upward, never moving down, allowing profits to run while protecting against reversals.

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

Exit Reasons:
-----------
- "stop_loss": When price falls below the trailing stop loss
"""

from typing import Any, Dict, Optional, List
import math

import backtrader as bt
from src.strategy.exit.base_exit_mixin import BaseExitMixin
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


class ATRExitMixin(BaseExitMixin):
    """
    Exit mixin that implements a trailing stop loss strategy using ATR.
    New Architecture only.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the mixin with parameters"""
        super().__init__(params)
        self.stop_loss = None
        self.highest_price = None
        self.entry_atr = None  # Fixed ATR value at position entry

    def get_required_params(self) -> list:
        """There are no required parameters - all have default values"""
        return []

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """Default parameters"""
        return {
            "atr_period": 14,
            "sl_multiplier": 2.0,
        }

    @classmethod
    def get_indicator_config(cls, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Define indicators required by this mixin."""
        # Support both new and legacy parameter names for the blueprint
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
        # Reset variables for new position
        self.highest_price = entry_price
        self.stop_loss = None

        if math.isnan(entry_price) or math.isinf(entry_price):
            logger.warning("Invalid entry price: %s", entry_price)
            return

        # Lock ATR value at entry
        try:
            self.entry_atr = self.get_indicator('exit_atr')
            logger.debug("ATR locked at entry: %.2f", self.entry_atr)
        except Exception as e:
            logger.warning("Failed to lock ATR at entry: %s", e)
            self.entry_atr = None

    def should_exit(self) -> bool:
        """Check if we should exit a position."""
        if not self.are_indicators_ready():
            return False

        try:
            current_price = self.strategy.data.close[0]
            current_high = self.strategy.data.high[0]

            # Use fixed ATR from entry if available
            if self.entry_atr is not None and self.entry_atr > 0:
                atr_val = self.entry_atr
            else:
                atr_val = self.get_indicator('exit_atr')

            sl_multiplier = self.get_param("sl_multiplier") or self.get_param("x_sl_multiplier", 2.0)

            if atr_val is None or atr_val <= 0:
                return False

            # Track highest price
            self.highest_price = max(self.highest_price or current_high, current_high)

            # Calculate and update trailing stop loss
            new_stop_loss = self.highest_price - (atr_val * sl_multiplier)

            if self.stop_loss is None:
                self.stop_loss = new_stop_loss
            else:
                self.stop_loss = max(self.stop_loss, new_stop_loss)

            # Check trigger
            if current_price <= self.stop_loss:
                logger.debug(f"ATR EXIT - Price: {current_price:.2f}, SL: {self.stop_loss:.2f}")
                self.strategy.current_exit_reason = "stop_loss"
                return True

            return False

        except Exception:
            logger.exception("Error in ATRExitMixin.should_exit")
            return False

    def next(self):
        """Called for each new bar"""
        super().next()
        if not self.strategy.position:
            self.highest_price = None
            self.stop_loss = None
            self.entry_atr = None

    def get_exit_reason(self) -> str:
        """Get the reason for exit"""
        return getattr(self.strategy, 'current_exit_reason', 'atr_exit')
