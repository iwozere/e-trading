"""
RSI and Bollinger Bands Entry Mixin

This module implements an entry strategy based on the combination of:
1. RSI (Relative Strength Index)
2. Bollinger Bands

The strategy enters a position when:
1. RSI is oversold
2. Price is below the lower Bollinger Band

Parameters:
    rsi_period (int): Period for RSI calculation (default: 14)
    rsi_oversold (float): Oversold threshold for RSI (default: 30)
    bb_period (int): Period for Bollinger Bands calculation (default: 20)
    bb_stddev (float): Standard deviation multiplier for Bollinger Bands (default: 2.0)
    use_bb_touch (bool): Whether to require price touching the lower band (default: True)

This strategy combines mean reversion (RSI + BB) to identify potential reversal points.
"""

from typing import Any, Dict, Optional

import backtrader as bt
from src.entry.base_entry_mixin import BaseEntryMixin
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


class RSIBBEntryMixin(BaseEntryMixin):
    """Entry mixin that combines RSI and Bollinger Bands for entry signals."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the mixin with parameters"""
        super().__init__(params)
        self.rsi_name = "entry_rsi"
        self.bb_name = "entry_bb"
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
            "e_rsi_period": 14,
            "e_rsi_oversold": 30,
            "e_bb_period": 20,
            "e_bb_dev": 2.0,
            "e_use_bb_touch": True,
        }

    def _init_indicators(self):
        """Initialize indicators"""
        logger.debug("RSIBBEntryMixin._init_indicators called")
        if not hasattr(self, "strategy"):
            logger.error("No strategy available in _init_indicators")
            return

        try:
            rsi_period = self.get_param("e_rsi_period")
            bb_period = self.get_param("e_bb_period")
            bb_dev_factor = self.get_param("e_bb_dev")

            # Validate parameters to prevent issues
            if rsi_period is None or rsi_period <= 0:
                logger.warning("Invalid RSI period: %s, using default value 14", rsi_period)
                rsi_period = 14
            elif rsi_period < 2:
                logger.warning("RSI period too small: %s, using minimum value 2", rsi_period)
                rsi_period = 2

            if bb_period is None or bb_period <= 0:
                logger.warning("Invalid BB period: %s, using default value 20", bb_period)
                bb_period = 20
            elif bb_period < 2:
                logger.warning("BB period too small: %s, using minimum value 2", bb_period)
                bb_period = 2

            if bb_dev_factor is None or bb_dev_factor <= 0:
                logger.warning("Invalid BB deviation factor: %s, using default value 2.0", bb_dev_factor)
                bb_dev_factor = 2.0

            if self.strategy.use_talib:
                self.rsi = bt.talib.RSI(self.strategy.data.close, timeperiod=rsi_period)
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

            # Register indicators after they are created
            if self.rsi is not None:
                self.register_indicator(self.rsi_name, self.rsi)
            if self.bb is not None:
                self.register_indicator(self.bb_name, self.bb)

        except Exception as e:
            logger.error(f"Error initializing indicators: {e}", exc_info=e)
            raise

    def are_indicators_ready(self) -> bool:
        """Check if indicators are ready to be used"""
        if not hasattr(self, "indicators"):
            return False

        try:
            # Check if we have enough data points
            if len(self.strategy.data) < max(
                self.get_param("e_rsi_period"), self.get_param("e_bb_period")
            ):
                return False

            # Check if indicators are registered and have values
            if (
                self.rsi_name not in self.indicators
                or self.bb_name not in self.indicators
            ):
                return False

            # Check if we can access the first value of each indicator
            rsi = self.indicators[self.rsi_name]
            bb = self.indicators[self.bb_name]

            # Try to access the first value of each indicator
            _ = rsi[0]
            _ = bb.lines.top[0]
            _ = bb.lines.mid[0]
            _ = bb.lines.bot[0]

            return True
        except (IndexError, AttributeError):
            return False

    def should_enter(self) -> bool:
        """Check if we should enter a position"""
        if not self.are_indicators_ready():
            return False

        try:
            # Get indicators from mixin's indicators dictionary
            rsi = self.indicators[self.rsi_name]
            current_price = self.strategy.data.close[0]

            # Defensive check: Ensure RSI value is valid
            rsi_value = rsi[0]
            if rsi_value is None:
                logger.warning("RSI value is None, skipping entry check")
                return False

            # Check RSI
            rsi_condition = rsi_value <= self.get_param("e_rsi_oversold")

            # Defensive check: Ensure Bollinger Bands values are valid
            bb_bot_value = self.bb_bot[0]
            if bb_bot_value is None:
                logger.warning("Bollinger Bands lower value is None, skipping entry check")
                return False

            if self.get_param("e_use_bb_touch"):
                bb_condition = current_price <= bb_bot_value
            else:
                bb_condition = current_price < bb_bot_value

            return_value = rsi_condition and bb_condition
            if return_value:
                logger.debug(
                    f"ENTRY: Price: {current_price}, RSI: {rsi_value}, BB Lower: {bb_bot_value}"
                )
            return return_value
        except Exception as e:
            logger.error(f"Error in should_enter: {e}", exc_info=e)
            return False
