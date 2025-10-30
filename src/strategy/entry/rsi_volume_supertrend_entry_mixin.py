"""
RSI, Volume, and Supertrend Entry Mixin

This module implements an entry strategy based on the combination of:
1. RSI (Relative Strength Index)
2. Volume analysis
3. Supertrend indicator

The strategy enters a position when:
1. RSI is oversold
2. Volume is above its moving average
3. Supertrend indicates a bullish trend

Parameters:
    rsi_period (int): Period for RSI calculation (default: 14)
    rsi_oversold (float): Oversold threshold for RSI (default: 30)
    volume_ma_period (int): Period for volume moving average (default: 20)
    supertrend_period (int): Period for Supertrend calculation (default: 10)
    supertrend_multiplier (float): Multiplier for Supertrend ATR (default: 3.0)

This strategy combines mean reversion (RSI) with volume confirmation and trend following (Supertrend)
to identify potential reversal points with strong momentum.
"""

from typing import Any, Dict, Optional

import backtrader as bt
from src.strategy.entry.base_entry_mixin import BaseEntryMixin
from src.indicators.adapters.backtrader_wrappers import UnifiedSuperTrendIndicator
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


class RSIVolumeSupertrendEntryMixin(BaseEntryMixin):
    """Entry mixin based on RSI, Volume, and Supertrend"""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the mixin with parameters"""
        super().__init__(params)
        self.rsi_name = "entry_rsi"
        self.vol_ma_name = "entry_volume_ma"
        self.supertrend_name = "entry_supertrend"
        self.direction_name = "entry_direction"

        self.rsi = None
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
            "e_vol_ma_period": 20,
            "e_min_volume_ratio": 1.5,
            "e_st_period": 10,
            "e_st_multiplier": 3.0,
        }

    def _init_indicators(self):
        """Initialize indicators"""
        logger.debug("RSIVolumeSupertrendEntryMixin._init_indicators called")
        if not hasattr(self, "strategy"):
            logger.error("No strategy available in _init_indicators")
            return

        try:
            rsi_period = self.get_param("e_rsi_period")
            sma_period = self.get_param("e_vol_ma_period")

            if self.strategy.use_talib:
                self.rsi = bt.talib.RSI(self.strategy.data.close, timeperiod=rsi_period)
                self.sma = bt.talib.SMA(
                    self.strategy.data.volume, timeperiod=sma_period
                )
            else:
                self.rsi = bt.indicators.RSI(
                    self.strategy.data.close, period=rsi_period
                )
                self.sma = bt.indicators.SMA(
                    self.strategy.data.volume, period=sma_period
                )

            self.register_indicator(self.rsi_name, self.rsi)
            self.register_indicator(self.vol_ma_name, self.sma)

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
            self.rsi_name not in self.indicators
            or self.vol_ma_name not in self.indicators
            or self.supertrend_name not in self.indicators
        ):
            return False

        try:
            # Get indicators from mixin's indicators dictionary
            rsi = self.indicators[self.rsi_name]
            vol_ma = self.indicators[self.vol_ma_name]
            supertrend = self.indicators[self.supertrend_name]
            current_price = self.strategy.data.close[0]
            current_volume = self.strategy.data.volume[0]

            # Check RSI
            rsi_condition = rsi[0] <= self.get_param("e_rsi_oversold")

            # Check Volume
            volume_condition = current_volume > vol_ma[0] * self.get_param("e_min_volume_ratio")

            # Check Supertrend
            supertrend_condition = supertrend.direction[0] == 1  # 1 means uptrend

            return_value = rsi_condition and volume_condition and supertrend_condition
            if return_value:
                logger.debug("ENTRY: Price: %s, RSI: %s, Volume: %s, Volume MA: %s, Supertrend Direction: %s", current_price, rsi[0], current_volume, vol_ma[0], supertrend.direction[0])
            return return_value
        except Exception as e:
            logger.exception("Error in should_enter: ")
            return False
