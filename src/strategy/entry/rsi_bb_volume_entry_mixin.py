"""
RSI, Bollinger Bands, and Volume Entry Mixin

This module implements an entry strategy based on the combination of:
1. RSI (Relative Strength Index)
2. Bollinger Bands
3. Volume analysis

The strategy enters a position when:
1. RSI is oversold
2. Price is below the lower Bollinger Band
3. Volume is above its moving average

Parameters:
    rsi_period (int): Period for RSI calculation (default: 14)
    rsi_oversold (float): Oversold threshold for RSI (default: 30)
    bb_period (int): Period for Bollinger Bands calculation (default: 20)
    bb_stddev (float): Standard deviation multiplier for Bollinger Bands (default: 2.0)
    volume_ma_period (int): Period for volume moving average (default: 20)
    use_bb_touch (bool): Whether to use Bollinger Band touch for entry (default: True)

This strategy combines mean reversion (RSI and BB) with volume confirmation
to identify potential reversal points with strong momentum.
"""

from typing import Any, Dict, Optional

import backtrader as bt
from src.entry.base_entry_mixin import BaseEntryMixin
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


class RSIBBVolumeEntryMixin(BaseEntryMixin):
    """Entry mixin based on RSI, Bollinger Bands, and Volume"""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the mixin with parameters"""
        super().__init__(params)
        self.rsi_name = "entry_rsi"
        self.bb_name = "entry_bb"
        self.vol_ma_name = "entry_volume_ma"

        self.rsi = None
        self.bb = None
        self.bb_bot = None
        self.bb_mid = None
        self.bb_top = None
        self.sma = None

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
            "e_vol_ma_period": 20,
            "e_use_bb_touch": True,
            "e_min_volume_ratio": 1.1,
        }

    def _init_indicators(self):
        """Initialize indicators"""
        logger.debug("RSIBBVolumeEntryMixin._init_indicators called")
        if not hasattr(self, "strategy"):
            logger.error("No strategy available in _init_indicators")
            return

        try:
            rsi_period = self.get_param("e_rsi_period")
            bb_period = self.get_param("e_bb_period")
            bb_dev_factor = self.get_param("e_bb_dev")
            sma_period = self.get_param("e_vol_ma_period")

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
                self.sma = bt.talib.SMA(self.strategy.data.volume, sma_period)
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
                self.sma = bt.indicators.SMA(
                    self.strategy.data.volume, period=sma_period
                )

            self.register_indicator(self.rsi_name, self.rsi)
            self.register_indicator(self.bb_name, self.bb)
            self.register_indicator(self.vol_ma_name, self.sma)
        except Exception as e:
            logger.error(f"Error initializing indicators: {e}", exc_info=e)
            raise

    def should_enter(self) -> bool:
        """Check if we should enter a position"""
        if (
            self.rsi_name not in self.indicators
            or self.bb_name not in self.indicators
            or self.vol_ma_name not in self.indicators
        ):
            return False

        try:
            # Get indicators from mixin's indicators dictionary
            rsi = self.indicators[self.rsi_name]
            bb = self.indicators[self.bb_name]
            vol_ma = self.indicators[self.vol_ma_name]
            current_price = self.strategy.data.close[0]
            current_volume = self.strategy.data.volume[0]

            # Check RSI
            rsi_condition = rsi[0] <= self.get_param("e_rsi_oversold")

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

            return_value = rsi_condition and bb_condition and volume_condition
            if return_value:
                logger.debug(
                    f"ENTRY: Price: {current_price}, RSI: {rsi[0]}, BB Lower: {bb.bb_lower[0] if self.strategy.use_talib else bb.lines.bot[0]}, Volume: {current_volume}, Volume MA: {vol_ma[0]}"
                )
            return return_value
        except Exception as e:
            logger.error(f"Error in should_enter: {e}", exc_info=e)
            return False
