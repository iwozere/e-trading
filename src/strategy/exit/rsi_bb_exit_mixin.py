"""
RSI and Bollinger Bands Exit Mixin

This module implements an exit strategy based on the combination of:
1. RSI (Relative Strength Index)
2. Bollinger Bands

The strategy exits a position when:
1. RSI is overbought (or crosses below it)
2. Price touches or crosses above the upper Bollinger Band

Configuration Example (New TALib Architecture):
    {
        "exit_logic": {
            "name": "RSIBBExitMixin",
            "indicators": [
                {
                    "type": "RSI",
                    "params": {"timeperiod": 14},
                    "fields_mapping": {"rsi": "exit_rsi"}
                },
                {
                    "type": "BBANDS",
                    "params": {"timeperiod": 20, "nbdevup": 2.0, "nbdevdn": 2.0},
                    "fields_mapping": {
                        "upperband": "exit_bb_upper"
                    }
                }
            ],
            "logic_params": {
                "rsi_overbought": 70
            }
        }
    }
"""

from typing import Any, Dict, Optional, List

import backtrader as bt
from src.strategy.exit.base_exit_mixin import BaseExitMixin
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


class RSIBBExitMixin(BaseExitMixin):
    """Exit mixin that combines RSI and Bollinger Bands for exit signals.
    New Architecture only.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the mixin with parameters"""
        super().__init__(params)

    def get_required_params(self) -> list:
        """There are no required parameters - all have default values"""
        return []

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """Default parameters"""
        return {
            "rsi_period": 14,
            "rsi_overbought": 70,
            "bb_period": 20,
            "bb_dev": 2.0,
        }

    @classmethod
    def get_indicator_config(cls, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Define indicators required by this mixin."""
        rsi_period = params.get("rsi_period", 14)
        bb_period = params.get("bb_period", 20)
        bb_dev = params.get("bb_dev", 2.0)

        return [
            {
                "type": "RSI",
                "params": {"timeperiod": rsi_period},
                "fields_mapping": {"rsi": "exit_rsi"}
            },
            {
                "type": "BBANDS",
                "params": {"timeperiod": bb_period, "nbdevup": bb_dev, "nbdevdn": bb_dev},
                "fields_mapping": {
                    "upperband": "exit_bb_upper"
                }
            }
        ]

    def _init_indicators(self):
        """No-op for new architecture."""
        pass

    def get_minimum_lookback(self) -> int:
        """Returns the minimum number of bars required."""
        return max(
            self.get_param("rsi_period", 14),
            self.get_param("bb_period", 20)
        )

    def are_indicators_ready(self) -> bool:
        """Check if required indicators exist in the strategy registry."""
        required = ['exit_rsi', 'exit_bb_upper']
        return all(alias in getattr(self.strategy, 'indicators', {}) for alias in required)

    def should_exit(self) -> bool:
        """Check if we should exit a position."""
        if not self.strategy.position:
            return False

        if not self.are_indicators_ready():
            return False

        try:
            # Standardized parameter retrieval
            rsi_overbought = self.get_param("rsi_overbought") or self.get_param("x_rsi_overbought", 70)

            # Unified Indicator Access
            current_rsi = self.get_indicator('exit_rsi')
            previous_rsi = self.get_indicator_prev('exit_rsi', 1)
            current_bb_top = self.get_indicator('exit_bb_upper')
            previous_bb_top = self.get_indicator_prev('exit_bb_upper', 1)

            # Get current and previous prices
            current_price = self.strategy.data.close[0]
            previous_price = self.strategy.data.close[-1]

            # RSI cross condition
            rsi_cross_condition = (previous_rsi >= rsi_overbought and current_rsi < rsi_overbought)

            # BB cross condition
            # Note: This checks if price was above upper band and closed back below it
            bb_cross_condition = (previous_price >= previous_bb_top and current_price < current_bb_top)

            should_exit_pos = rsi_cross_condition and bb_cross_condition

            if should_exit_pos:
                logger.debug(
                    f"EXIT: RSI cross from {previous_rsi:.2f} to {current_rsi:.2f} (overbought: {rsi_overbought}), "
                    f"Price cross from {previous_price:.2f} to {current_price:.2f} (BB top: {current_bb_top:.2f})"
                )
                self.strategy.current_exit_reason = "rsi_bb_cross_exit"

            return should_exit_pos
        except Exception:
            logger.exception("Error in RSIBBExitMixin.should_exit")
            return False

    def get_exit_reason(self) -> str:
        """Get the reason for exit"""
        return getattr(self.strategy, 'current_exit_reason', 'rsi_bb_cross_exit')
