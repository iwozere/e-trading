"""
Trailing Stop Exit Mixin

This module implements a trailing stop exit strategy that dynamically adjusts the stop loss
level as the price moves in favor of the position.

Configuration Example (New TALib Architecture):
    {
        "exit_logic": {
            "name": "TrailingStopExitMixin",
            "indicators": [
                {
                    "type": "ATR",
                    "params": {"timeperiod": 14},
                    "fields_mapping": {"atr": "exit_atr"}
                }
            ],
            "logic_params": {
                "use_atr": true,
                "atr_multiplier": 2.0,
                "trail_pct": 0.02,
                "activation_pct": 0.0
            }
        }
    }
"""

from typing import Any, Dict, Optional

import backtrader as bt
from src.strategy.exit.base_exit_mixin import BaseExitMixin
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


class TrailingStopExitMixin(BaseExitMixin):
    """Exit mixin based on trailing stop.

    Supports both new TALib-based architecture and legacy configurations.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the mixin with parameters"""
        super().__init__(params)
        self.highest_price = 0
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
            "x_trail_pct": 0.02,
            "x_activation_pct": 0.0,
            "x_use_atr": False,
            "x_atr_multiplier": 2.0,
            "x_atr_period": 14,
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
            atr_period = self.get_param("x_atr_period")
            if self.strategy.use_talib:
                self.atr = bt.talib.ATR(
                    self.strategy.data.high,
                    self.strategy.data.low,
                    self.strategy.data.close,
                    timeperiod=atr_period,
                )
            else:
                self.atr = bt.indicators.AverageTrueRange(self.strategy.data, period=atr_period)

            self.register_indicator(self.atr_name, self.atr)

        except Exception:
            logger.exception("Error initializing indicators: ")
            raise

    def get_minimum_lookback(self) -> int:
        """Returns the minimum number of bars required."""
        use_atr = self.get_param("use_atr") or self.get_param("x_use_atr", False)
        if not use_atr:
            return 0

        if self.use_new_architecture:
            return self.get_param("atr_period", 14)
        else:
            return self.get_param("x_atr_period", 14)

    def are_indicators_ready(self) -> bool:
        """Check if indicators are initialized."""
        use_atr = self.get_param("use_atr") or self.get_param("x_use_atr", False)
        if not use_atr:
            return True

        if self.use_new_architecture:
            return 'exit_atr' in getattr(self.strategy, 'indicators', {})
        else:
            return self.atr_name in self.indicators

    def should_exit(self) -> bool:
        """Check if we should exit a position."""
        if not self.strategy.position:
            return False

        if not self.are_indicators_ready():
            return False

        try:
            price = self.strategy.data.close[0]
            entry_price = self.strategy.position.price

            # Update highest price
            self.highest_price = max(self.highest_price, price)

            # Get parameters
            use_atr = self.get_param("use_atr") or self.get_param("x_use_atr", False)
            atr_multiplier = self.get_param("atr_multiplier") or self.get_param("x_atr_multiplier", 2.0)
            trail_pct = self.get_param("trail_pct") or self.get_param("x_trail_pct", 0.02)
            activation_pct = self.get_param("activation_pct") or self.get_param("x_activation_pct", 0.0)

            # Calculate stop level
            if use_atr:
                if self.use_new_architecture:
                    atr_val = self.get_indicator('exit_atr')
                else:
                    atr = self.indicators[self.atr_name]
                    atr_val = atr[0] if hasattr(atr, "__getitem__") else atr.lines.atr[0]
                stop_level = self.highest_price - (atr_val * atr_multiplier)
            else:
                stop_level = self.highest_price * (1 - trail_pct)

            # Activation check
            if activation_pct > 0:
                profit_pct = (price - entry_price) / entry_price
                if profit_pct < activation_pct:
                    return False

            # Trigger check
            if price < stop_level:
                logger.debug(f"Trailing Stop EXIT - Price: {price:.2f}, SL: {stop_level:.2f}")
                self.strategy.current_exit_reason = "trailing_stop"
                return True

            return False

        except Exception:
            logger.exception("Error in TrailingStopExitMixin.should_exit")
            return False

    def get_exit_reason(self) -> str:
        """Get the reason for exit"""
        return "trailing_stop"
