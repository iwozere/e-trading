"""
Bollinger Bands, Volume, and Supertrend Entry Mixin

This module implements an entry strategy based on the combination of:
1. Bollinger Bands
2. Volume analysis
3. Supertrend indicator

The strategy enters a position when:
1. Price is below the lower Bollinger Band
2. Volume is above its moving average
3. Supertrend indicates a bullish trend

Parameters:
    bb_period (int): Period for Bollinger Bands calculation (default: 20)
    bb_stddev (float): Standard deviation multiplier for Bollinger Bands (default: 2.0)
    volume_ma_period (int): Period for volume moving average (default: 20)
    supertrend_period (int): Period for Supertrend calculation (default: 10)
    supertrend_multiplier (float): Multiplier for Supertrend ATR (default: 3.0)
    use_bb_touch (bool): Whether to require price touching the lower band (default: True)
    use_talib (bool): Whether to use TA-Lib for calculations (default: True)

This strategy combines mean reversion (BB), volume confirmation, and trend following (Supertrend).
"""

from typing import Any, Dict, Optional

import backtrader as bt
from src.strategy.entry.base_entry_mixin import BaseEntryMixin
from src.indicators.adapters.backtrader_wrappers import UnifiedSuperTrendIndicator
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


class BBVolumeSupertrendEntryMixin(BaseEntryMixin):
    """Entry mixin that combines Bollinger Bands, Volume, and Supertrend for entry signals."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the mixin with parameters"""
        super().__init__(params)
        self.bb_name = "entry_bb"
        self.volume_ma_name = "entry_volume_ma"
        self.supertrend_name = "entry_supertrend"

        self.bb = None
        self.bb_bot = None
        self.bb_mid = None
        self.bb_top = None

        self.sma = None
        self.super_trend = None

    def get_required_params(self) -> list:
        """There are no required parameters - all have default values"""
        return []

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """Default parameters"""
        return {
            "e_bb_period": 20,
            "e_bb_dev": 2.0,
            "e_vol_ma_period": 20,
            "e_min_volume_ratio": 1.1,
            "e_st_period": 10,
            "e_st_multiplier": 3.0,
            "e_use_bb_touch": True,
        }

    def _init_indicators(self):
        """Initialize indicators"""
        logger.debug("BBVolumeSupertrendEntryMixin._init_indicators called")
        if not hasattr(self, "strategy"):
            logger.error("No strategy available in _init_indicators")
            return

        try:
            bb_period = self.get_param("e_bb_period")
            bb_dev_factor = self.get_param("e_bb_dev")
            sma_period = self.get_param("e_vol_ma_period")

            if self.strategy.use_talib:
                self.bb = bt.talib.BBANDS(
                    self.strategy.data.close,
                    timeperiod=bb_period,
                    nbdevup=bb_dev_factor,
                    nbdevdn=bb_dev_factor,
                )
                self.bb_top = self.bb.upperband
                self.bb_mid = self.bb.middleband
                self.bb_bot = self.bb.lowerband
                self.sma = bt.talib.SMA(
                    self.strategy.data.volume, timeperiod=sma_period
                )
            else:
                self.bb = bt.indicators.BollingerBands(
                    self.strategy.data.close, period=bb_period, devfactor=bb_dev_factor
                )
                self.bb_top = self.bb.top
                self.bb_mid = self.bb.mid
                self.bb_bot = self.bb.bot
                self.sma = bt.indicators.SMA(
                    self.strategy.data.volume, period=sma_period
                )

            self.register_indicator(self.bb_name, self.bb)
            self.register_indicator(self.volume_ma_name, self.sma)

            # Create Supertrend indicator using unified service
            supertrend = UnifiedSuperTrendIndicator(
                self.strategy.data,
                length=self.get_param("e_st_period"),
                multiplier=self.get_param("e_st_multiplier"),
            )
            self.register_indicator(self.supertrend_name, supertrend)
        except Exception as e:
            logger.exception("Error initializing indicators: ")
            raise

    def should_enter(self) -> bool:
        """Check if we should enter a position"""
        if (
            self.bb_name not in self.indicators
            or self.volume_ma_name not in self.indicators
            or self.supertrend_name not in self.indicators
        ):
            return False

        try:
            # Get indicators from mixin's indicators dictionary
            bb = self.indicators[self.bb_name]
            vol_ma = self.indicators[self.volume_ma_name]
            supertrend = self.indicators[self.supertrend_name]
            current_price = self.strategy.data.close[0]
            current_volume = self.strategy.data.volume[0]

            # Check Bollinger Bands
            if self.strategy.use_talib:
                # For TA-Lib BB, use bb_lower
                if self.get_param("e_use_bb_touch"):
                    bb_condition = current_price <= bb.bb_lower[0]
                else:
                    bb_condition = current_price < bb.bb_lower[0]
            else:
                # For Backtrader's native BB, use lines.bot
                if self.get_param("e_use_bb_touch"):
                    bb_condition = current_price <= bb.lines.bot[0]
                else:
                    bb_condition = current_price < bb.lines.bot[0]

            # Check Volume
            volume_condition = current_volume > vol_ma[0] * self.get_param(
                "e_min_volume_ratio"
            )

            # Check Supertrend
            supertrend_condition = supertrend.direction[0] == 1  # 1 means uptrend

            return_value = bb_condition and volume_condition and supertrend_condition
            if return_value:
                logger.debug(
                    f"ENTRY: Price: {current_price}, BB Lower: {bb.bb_lower[0] if self.strategy.use_talib else bb.lines.bot[0]}, Volume: {current_volume}, Volume MA: {vol_ma[0]}, Supertrend Direction: {supertrend.direction[0]}"
                )
            return return_value
        except Exception as e:
            logger.exception("Error in should_enter: ")
            return False
