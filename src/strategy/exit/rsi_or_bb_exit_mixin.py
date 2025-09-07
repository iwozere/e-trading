"""
RSI or Bollinger Bands Exit Mixin

This module implements an exit strategy based on either:
1. RSI (Relative Strength Index) crossing overbought line, OR
2. Price crossing Bollinger Bands top line

The strategy exits a position when either condition is met:
1. RSI crosses overbought line (previous value above, current below the overbought line)
2. Price crosses BB top line (previous price above, current price below BB top)

Parameters:
    x_rsi_period (int): Period for RSI calculation (default: 14)
    x_rsi_overbought (float): Overbought threshold for RSI (default: 70)
    x_bb_period (int): Period for Bollinger Bands calculation (default: 20)
    x_bb_stddev (float): Standard deviation multiplier for Bollinger Bands (default: 2.0)

This strategy provides more flexible exit conditions by allowing either RSI or BB signals.
"""

from typing import Any, Dict, Optional

import backtrader as bt
from src.strategy.exit.base_exit_mixin import BaseExitMixin
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


class RSIOrBBExitMixin(BaseExitMixin):
    """Exit mixin that uses either RSI or Bollinger Bands for exit signals."""

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
        }

    def _init_indicators(self):
        """Initialize indicators"""
        logger.debug("RSIOrBBExitMixin._init_indicators called")
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
            logger.exception("Error initializing indicators: ")
            raise

    def should_exit(self) -> bool:
        """Check if we should exit a position"""
        if not self.strategy.position:
            return False

        if self.rsi_name not in self.indicators or self.bb_name not in self.indicators:
            return False

        try:
            # Get current and previous values
            current_price = self.strategy.data.close[0]
            previous_price = self.strategy.data.close[-1]
            current_rsi = self.rsi[0]
            previous_rsi = self.rsi[-1]
            current_bb_top = self.bb_top[0]
            previous_bb_top = self.bb_top[-1]

            # Check if we have enough data
            if (current_rsi is None or previous_rsi is None or
                current_bb_top is None or previous_bb_top is None or
                current_price is None or previous_price is None):
                return False

            # RSI cross condition: previous RSI above overbought, current RSI below overbought
            rsi_cross_condition = (previous_rsi >= self.get_param("x_rsi_overbought") and
                                 current_rsi < self.get_param("x_rsi_overbought"))

            # BB cross condition: previous price above BB top, current price below BB top
            bb_cross_condition = (previous_price >= previous_bb_top and
                                current_price < current_bb_top)

            # Either condition can trigger exit
            should_exit = rsi_cross_condition or bb_cross_condition

            if should_exit:
                exit_reason = []
                if rsi_cross_condition:
                    exit_reason.append("rsi_cross")
                if bb_cross_condition:
                    exit_reason.append("bb_cross")

                reason_text = "_".join(exit_reason)
                self.strategy.current_exit_reason = f"rsi_or_bb_{reason_text}"

                logger.debug(
                    f"EXIT: RSI cross: {rsi_cross_condition} (from {previous_rsi:.2f} to {current_rsi:.2f}), "
                    f"BB cross: {bb_cross_condition} (price from {previous_price:.2f} to {current_price:.2f}, BB top: {current_bb_top:.2f})"
                )

            return should_exit
        except Exception as e:
            logger.exception("Error in should_exit: ")
            return False

    def get_exit_reason(self) -> str:
        """Get the reason for exit"""
        return getattr(self.strategy, 'current_exit_reason', 'rsi_or_bb_exit')
