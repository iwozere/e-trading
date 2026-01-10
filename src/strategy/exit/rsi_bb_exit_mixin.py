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

from typing import Any, Dict, Optional

import backtrader as bt
from src.strategy.exit.base_exit_mixin import BaseExitMixin
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


class RSIBBExitMixin(BaseExitMixin):
    """Exit mixin that combines RSI and Bollinger Bands for exit signals.

    Supports both new TALib-based architecture and legacy configurations.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the mixin with parameters"""
        super().__init__(params)

        # Legacy architecture support
        self.rsi_name = "exit_rsi"
        self.bb_name = "exit_bb"
        self.rsi = None
        self.bb = None

        # Detect architecture mode
        self.use_new_architecture = False  # Will be set in init_exit()

    def get_required_params(self) -> list:
        """There are no required parameters - all have default values"""
        return []

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """Default parameters"""
        return {
            "x_rsi_period": 14,
            "x_rsi_overbought": 70,
            "x_bb_period": 20,
            "x_bb_dev": 2.0,
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
            rsi_period = self.get_param("x_rsi_period")
            bb_period = self.get_param("x_bb_period")
            bb_dev_factor = self.get_param("x_bb_dev")

            if self.strategy.use_talib:
                self.rsi = bt.talib.RSI(self.strategy.data.close, timeperiod=rsi_period)
                self.bb = bt.talib.BBANDS(
                    self.strategy.data.close,
                    timeperiod=bb_period,
                    nbdevup=bb_dev_factor,
                    nbdevdn=bb_dev_factor,
                )
            else:
                self.rsi = bt.indicators.RSI(self.strategy.data.close, period=rsi_period)
                self.bb = bt.indicators.BollingerBands(
                    self.strategy.data.close, period=bb_period, devfactor=bb_dev_factor
                )

            self.register_indicator(self.rsi_name, self.rsi)
            self.register_indicator(self.bb_name, self.bb)

        except Exception:
            logger.exception("Error initializing indicators: ")
            raise

    def get_minimum_lookback(self) -> int:
        """Returns the minimum number of bars required."""
        if self.use_new_architecture:
            return max(
                self.get_param("rsi_period", 14),
                self.get_param("bb_period", 20)
            )
        else:
            return max(
                self.get_param("x_rsi_period", 14),
                self.get_param("x_bb_period", 20)
            )

    def are_indicators_ready(self) -> bool:
        """Check if indicators are initialized."""
        if self.use_new_architecture:
            required = ['exit_rsi', 'exit_bb_upper']
            return all(alias in getattr(self.strategy, 'indicators', {}) for alias in required)
        else:
            return self.rsi_name in self.indicators and self.bb_name in self.indicators

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
            if self.use_new_architecture:
                current_rsi = self.get_indicator('exit_rsi')
                previous_rsi = self.get_indicator_prev('exit_rsi', 1)
                current_bb_top = self.get_indicator('exit_bb_upper')
                previous_bb_top = self.get_indicator_prev('exit_bb_upper', 1)
            else:
                current_rsi = self.rsi[0]
                previous_rsi = self.rsi[-1]

                if self.strategy.use_talib:
                    current_bb_top = self.bb.upperband[0]
                    previous_bb_top = self.bb.upperband[-1]
                else:
                    current_bb_top = self.bb.top[0]
                    previous_bb_top = self.bb.top[-1]

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