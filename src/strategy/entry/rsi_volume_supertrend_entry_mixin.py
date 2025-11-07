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

        # Detect architecture mode
        self.use_new_architecture = False  # Will be set in init_entry()

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

    def init_entry(self, strategy, additional_params: Optional[Dict[str, Any]] = None):
        """Override to detect architecture mode before calling parent."""
        # Detect architecture: new if strategy has indicators dict with entries
        if hasattr(strategy, 'indicators') and strategy.indicators:
            self.use_new_architecture = True
            logger.debug("Using new TALib-based architecture")
        else:
            self.use_new_architecture = False
            logger.debug("Using legacy architecture")

        # Call parent init_entry which will call _init_indicators
        super().init_entry(strategy, additional_params)

    def _init_indicators(self):
        """Initialize indicators (legacy architecture only).

        In new architecture, indicators are created by the strategy
        and accessed via get_indicator().
        """
        if self.use_new_architecture:
            # New architecture: indicators already created by strategy
            return

        # Legacy architecture: create indicators in mixin
        logger.debug("RSIVolumeSupertrendEntryMixin._init_indicators called (legacy architecture)")
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

            logger.debug("Legacy indicators initialized: entry_rsi, entry_volume_ma, entry_supertrend")
        except Exception as e:
            logger.exception("Error initializing indicators: ")
            raise

    def are_indicators_ready(self) -> bool:
        """Check if indicators are ready to be used"""
        if self.use_new_architecture:
            # New architecture: check strategy's indicators
            if not hasattr(self.strategy, 'indicators') or not self.strategy.indicators:
                return False

            # Check if required indicators exist
            required_indicators = ['entry_rsi', 'entry_volume_ma', 'entry_supertrend_direction']
            for ind_alias in required_indicators:
                if ind_alias not in self.strategy.indicators:
                    return False

            # Check if we can access values
            try:
                _ = self.get_indicator('entry_rsi')
                _ = self.get_indicator('entry_volume_ma')
                _ = self.get_indicator('entry_supertrend_direction')
                return True
            except (IndexError, KeyError, AttributeError):
                return False

        else:
            # Legacy architecture: check mixin's indicators
            return (self.rsi_name in self.indicators and
                    self.vol_ma_name in self.indicators and
                    self.supertrend_name in self.indicators)

    def should_enter(self) -> bool:
        """Check if we should enter a position.

        Works with both new and legacy architectures.
        """
        if not self.are_indicators_ready():
            return False

        try:
            current_price = self.strategy.data.close[0]
            current_volume = self.strategy.data.volume[0]

            # Get indicator values and params based on architecture
            if self.use_new_architecture:
                # New architecture: access via get_indicator()
                current_rsi = self.get_indicator('entry_rsi')
                vol_ma = self.get_indicator('entry_volume_ma')
                supertrend_direction = self.get_indicator('entry_supertrend_direction')

                # Get params from logic_params (new) or fallback to legacy params
                rsi_oversold = self.get_param("rsi_oversold") or self.get_param("e_rsi_oversold", 30)
                min_volume_ratio = self.get_param("min_volume_ratio") or self.get_param("e_min_volume_ratio", 1.5)
            else:
                # Legacy architecture: access via mixin's indicators dict
                rsi = self.indicators[self.rsi_name]
                vol_ma_ind = self.indicators[self.vol_ma_name]
                supertrend = self.indicators[self.supertrend_name]

                current_rsi = rsi[0]
                vol_ma = vol_ma_ind[0]
                supertrend_direction = supertrend.direction[0]

                # Get params from legacy params
                rsi_oversold = self.get_param("e_rsi_oversold", 30)
                min_volume_ratio = self.get_param("e_min_volume_ratio", 1.5)

            # Check RSI
            rsi_condition = current_rsi <= rsi_oversold

            # Check Volume
            volume_condition = current_volume > vol_ma * min_volume_ratio

            # Check Supertrend
            supertrend_condition = supertrend_direction == 1  # 1 means uptrend

            return_value = rsi_condition and volume_condition and supertrend_condition
            if return_value:
                logger.debug(
                    "ENTRY: Price: %s, RSI: %s, Volume: %s, Volume MA: %s, Supertrend Direction: %s",
                    current_price, current_rsi, current_volume, vol_ma, supertrend_direction
                )
            return return_value
        except Exception as e:
            logger.exception("Error in should_enter: ")
            return False
