"""
RSI and Bollinger Bands Exit Mixin

This module implements an exit strategy based on the combination of:
1. RSI (Relative Strength Index)
2. Bollinger Bands

The strategy exits a position when:
1. RSI is overbought
2. Price touches or crosses above the upper Bollinger Band

Parameters:
    rsi_period (int): Period for RSI calculation (default: 14)
    rsi_overbought (float): Overbought threshold for RSI (default: 70)
    bb_period (int): Period for Bollinger Bands calculation (default: 20)
    bb_stddev (float): Standard deviation multiplier for Bollinger Bands (default: 2.0)
    use_bb_touch (bool): Whether to require price touching the upper band (default: True)
    use_talib (bool): Whether to use TA-Lib for calculations (default: True)

This strategy combines mean reversion (RSI + BB) to identify potential reversal points.
"""

from typing import Any, Dict, Optional

import backtrader as bt
from src.exit.base_exit_mixin import BaseExitMixin
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


class RSIBBExitMixin(BaseExitMixin):
    """Exit mixin that combines RSI and Bollinger Bands for exit signals."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the mixin with parameters"""
        super().__init__(params)
        self.rsi_name = "exit_rsi"
        self.bb_name = "exit_bb"
        self.rsi = None
        self.bb = None
        self.bb_bot = None
        self.bb_mid = None
        self.bb_top = None

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
            "x_use_bb_touch": True,
        }

    def _init_indicators(self):
        """Initialize indicators"""
        logger.debug("RSIBBExitMixin._init_indicators called")
        if not hasattr(self, "strategy"):
            logger.error("No strategy available in _init_indicators")
            return

        try:
            rsi_period = self.get_param("x_rsi_period")
            bb_period = self.get_param("x_bb_period")
            bb_dev_factor = self.get_param("x_bb_dev")

            if self.strategy.use_talib:
                self.rsi = bt.talib.RSI(self.strategy.data.close, period=rsi_period)
                self.bb = bt.talib.BBANDS(
                    self.strategy.data.close,
                    timeperiod=bb_period,
                    nbdevup=bb_dev_factor,
                    nbdevdn=bb_dev_factor,
                )
                self.bb_top = self.bb.upperband
                self.bb_mid = self.bb.middleband
                self.bb_bot = self.bb.lowerband
            else:
                self.rsi = bt.indicators.RSI(
                    self.strategy.data.close, period=rsi_period
                )
                self.bb = bt.indicators.BollingerBands(
                    self.strategy.data.close, period=bb_period, devfactor=bb_dev_factor
                )
                self.bb_top = self.bb.top
                self.bb_mid = self.bb.mid
                self.bb_bot = self.bb.bot

            self.register_indicator(self.rsi_name, self.rsi)
            self.register_indicator(self.bb_name, self.bb)
        except Exception as e:
            logger.error(f"Error initializing indicators: {e}", exc_info=e)
            raise

    def should_exit(self) -> bool:
        """Check if we should exit a position"""
        if not self.strategy.position:
            return False

        if self.rsi_name not in self.indicators or self.bb_name not in self.indicators:
            return False

        try:
            # Get indicators from mixin's indicators dictionary
            current_price = self.strategy.data.close[0]

            # Check RSI condition
            rsi_condition = self.rsi[0] >= self.get_param("x_rsi_overbought")

            # Check Bollinger Bands condition if enabled
            bb_condition = False
            if self.get_param("x_use_bb_touch"):
                bb_condition = current_price >= self.bb_top[0] * 0.99
            else:
                bb_condition = current_price >= self.bb_top[0]

            return_value = rsi_condition or bb_condition
            if return_value:
                logger.debug(
                    f"EXIT: Price: {current_price}, RSI: {self.rsi[0]}, BB Upper: {self.bb_top[0]}, RSI Overbought: {self.get_param('x_rsi_overbought')}"
                )
                self.strategy.current_exit_reason = "rsi_bb_overbought"
            return return_value
        except Exception as e:
            logger.error(f"Error in should_exit: {e}", exc_info=e)
            return False
