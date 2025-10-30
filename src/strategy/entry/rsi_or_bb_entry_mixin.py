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
from src.strategy.entry.base_entry_mixin import BaseEntryMixin
from src.indicators.adapters.backtrader_wrappers import UnifiedRSIIndicator, UnifiedBollingerBandsIndicator
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


class RSIOrBBEntryMixin(BaseEntryMixin):
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
        self.last_entry_bar = None

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
            "e_rsi_cross": False,
            "e_bb_reentry": False,
            "e_cooldown_bars": 0,
        }

    def _init_indicators(self):
        """Initialize indicators"""
        logger.debug("RSIOrBBEntryMixin._init_indicators called")
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

            # Create unified indicators directly
            backend = "bt-talib" if self.strategy.use_talib else "bt"

            self.rsi = UnifiedRSIIndicator(
                self.strategy.data,
                period=rsi_period,
                backend=backend
            )

            self.bb = UnifiedBollingerBandsIndicator(
                self.strategy.data,
                period=bb_period,
                devfactor=bb_dev_factor,
                backend=backend
            )

            # Register wrapped indicators
            self.register_indicator(self.rsi_name, self.rsi)
            self.register_indicator(self.bb_name, self.bb)

        except Exception as e:
            logger.exception("Error initializing indicators: ")
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

            # Try to access the first value of each indicator using unified access
            _ = rsi.rsi[0]
            _ = bb.upper[0]  # Use unified access - works for both TALib and standard
            _ = bb.middle[0]
            _ = bb.lower[0]

            return True
        except (IndexError, AttributeError):
            return False

    def should_enter(self) -> bool:
        """Check if we should enter a position"""
        if not self.are_indicators_ready():
            return False

        try:
            # Check cooldown period
            if self.get_param("e_cooldown_bars", 0) > 0:
                current_bar = len(self.strategy.data)
                if (self.last_entry_bar is not None and
                    current_bar - self.last_entry_bar < self.get_param("e_cooldown_bars")):
                    return False

            # Get indicators from mixin's indicators dictionary
            rsi = self.indicators[self.rsi_name]
            bb = self.indicators[self.bb_name]
            current_price = self.strategy.data.close[0]

            # Defensive check: Ensure RSI value is valid
            rsi_value = rsi.rsi[0]
            rsi_prev = rsi.rsi[-1] if len(rsi.rsi) > 1 else rsi_value
            if rsi_value is None or rsi_prev is None:
                logger.warning("RSI value is None, skipping entry check")
                return False

            # Defensive check: Ensure Bollinger Bands values are valid
            bb_bot_value = bb.lower[0]  # Use unified access
            bb_mid_value = bb.middle[0]
            if bb_bot_value is None or bb_mid_value is None:
                logger.warning("Bollinger Bands values are None, skipping entry check")
                return False


            # RSI condition with optional cross confirmation
            if self.get_param("e_rsi_cross", False):
                # RSI cross upward: require RSI to cross back above oversold threshold
                rsi_condition = (rsi_prev <= self.get_param("e_rsi_oversold") and
                               rsi_value > self.get_param("e_rsi_oversold"))
            else:
                # Original RSI condition
                rsi_condition = rsi_value <= self.get_param("e_rsi_oversold")

            # Bollinger Bands condition with optional re-entry confirmation
            if self.get_param("e_bb_reentry", False):
                # BB re-entry: require close > lower BB after touching below
                bb_condition = current_price > bb_bot_value
            else:
                # Original BB condition
                if self.get_param("e_use_bb_touch"):
                    bb_condition = current_price <= bb_bot_value
                else:
                    bb_condition = current_price < bb_bot_value

            # OR logic for RSI or BB condition
            return_value = rsi_condition or bb_condition
            if return_value:
                self.last_entry_bar = len(self.strategy.data)
                logger.debug(
                    f"ENTRY: Price: {current_price}, RSI: {rsi_value}, BB Lower: {bb_bot_value}, "
                    f"RSI Cross: {self.get_param('e_rsi_cross')}, BB Reentry: {self.get_param('e_bb_reentry')}"
                )
            return return_value
        except Exception as e:
            logger.exception("Error in should_enter: ")
            return False
