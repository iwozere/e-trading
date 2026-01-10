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

Configuration Example (New TALib Architecture):
    {
        "entry_logic": {
            "name": "BBVolumeSupertrendEntryMixin",
            "indicators": [
                {
                    "type": "BBANDS",
                    "params": {"timeperiod": 20, "nbdevup": 2.0, "nbdevdn": 2.0},
                    "fields_mapping": {
                        "upperband": "entry_bb_upper",
                        "middleband": "entry_bb_middle",
                        "lowerband": "entry_bb_lower"
                    }
                },
                {
                    "type": "SMA",
                    "data_inputs": ["volume"],
                    "params": {"timeperiod": 20},
                    "fields_mapping": {"sma": "entry_volume_ma"}
                },
                {
                    "type": "SUPERTREND",
                    "params": {"length": 10, "multiplier": 3.0},
                    "fields_mapping": {
                        "super_trend": "entry_supertrend",
                        "direction": "entry_supertrend_direction"
                    }
                }
            ],
            "logic_params": {
                "use_bb_touch": true,
                "min_volume_ratio": 1.1
            }
        }
    }
"""

from typing import Any, Dict, Optional

import backtrader as bt
from src.strategy.entry.base_entry_mixin import BaseEntryMixin
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


class BBVolumeSupertrendEntryMixin(BaseEntryMixin):
    """Entry mixin that combines Bollinger Bands, Volume, and Supertrend for entry signals.

    Supports both new TALib-based architecture and legacy configurations.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the mixin with parameters"""
        super().__init__(params)

        # Legacy architecture support
        self.bb_name = "entry_bb"
        self.volume_ma_name = "entry_volume_ma"
        self.supertrend_name = "entry_supertrend"
        self.bb = None
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
            "e_bb_period": 20,
            "e_bb_dev": 2.0,
            "e_vol_ma_period": 20,
            "e_min_volume_ratio": 1.1,
            "e_st_period": 10,
            "e_st_multiplier": 3.0,
            "e_use_bb_touch": True,
        }

    def init_entry(self, strategy, additional_params: Optional[Dict[str, Any]] = None):
        """Override to detect architecture mode before calling parent."""
        if hasattr(strategy, 'indicators') and strategy.indicators:
            self.use_new_architecture = True
        else:
            self.use_new_architecture = False

        super().init_entry(strategy, additional_params)

    def _init_indicators(self):
        """Initialize indicators (legacy architecture only)."""
        if self.use_new_architecture:
            return

        if not hasattr(self, "strategy"):
            logger.error("No strategy available in _init_indicators")
            return

        try:
            from src.indicators.adapters.backtrader_wrappers import UnifiedSuperTrendIndicator

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
                self.sma = bt.talib.SMA(self.strategy.data.volume, timeperiod=sma_period)
            else:
                self.bb = bt.indicators.BollingerBands(
                    self.strategy.data.close, period=bb_period, devfactor=bb_dev_factor
                )
                self.sma = bt.indicators.SMA(self.strategy.data.volume, period=sma_period)

            supertrend = UnifiedSuperTrendIndicator(
                self.strategy.data,
                length=self.get_param("e_st_period"),
                multiplier=self.get_param("e_st_multiplier"),
            )

            self.register_indicator(self.bb_name, self.bb)
            self.register_indicator(self.volume_ma_name, self.sma)
            self.register_indicator(self.supertrend_name, supertrend)

        except Exception:
            logger.exception("Error initializing indicators: ")
            raise

    def get_minimum_lookback(self) -> int:
        """Returns the minimum number of bars required."""
        if self.use_new_architecture:
            return max(
                self.get_param("bb_period", 20),
                self.get_param("vol_ma_period", 20),
                self.get_param("st_period", 10)
            )
        else:
            return max(
                self.get_param("e_bb_period", 14),
                self.get_param("e_vol_ma_period", 20),
                self.get_param("e_st_period", 10)
            )

    def are_indicators_ready(self) -> bool:
        """Check if indicators are initialized."""
        required = ['entry_volume_ma']
        if self.use_new_architecture:
            required.extend(['entry_bb_lower', 'entry_supertrend_direction'])
            return all(alias in getattr(self.strategy, 'indicators', {}) for alias in required)
        else:
            required.extend([self.bb_name, self.supertrend_name])
            return all(name in self.indicators for name in required)

    def should_enter(self) -> bool:
        """Check if we should enter a position."""
        if not self.are_indicators_ready():
            return False

        try:
            current_price = self.strategy.data.close[0]
            current_volume = self.strategy.data.volume[0]

            # Standardized parameter retrieval
            use_bb_touch = self.get_param("use_bb_touch") or self.get_param("e_use_bb_touch", True)
            min_volume_ratio = self.get_param("min_volume_ratio") or self.get_param("e_min_volume_ratio", 1.1)

            # Unified Indicator Access
            if self.use_new_architecture:
                bb_lower = self.get_indicator('entry_bb_lower')
                vol_ma = self.get_indicator('entry_volume_ma')
                supertrend_direction = self.get_indicator('entry_supertrend_direction')
            else:
                bb = self.indicators[self.bb_name]
                vol_ma_ind = self.indicators[self.volume_ma_name]
                supertrend = self.indicators[self.supertrend_name]

                if self.strategy.use_talib:
                    bb_lower = bb.lowerband[0]
                else:
                    bb_lower = bb.bot[0]

                vol_ma = vol_ma_ind[0]
                supertrend_direction = supertrend.direction[0]

            # Check Bollinger Bands
            if use_bb_touch:
                bb_condition = current_price <= bb_lower
            else:
                bb_condition = current_price < bb_lower

            # Check Volume
            volume_condition = current_volume > vol_ma * min_volume_ratio

            # Check Supertrend
            supertrend_condition = supertrend_direction == 1

            entry_signal = bb_condition and volume_condition and supertrend_condition

            if entry_signal:
                logger.debug(
                    f"ENTRY SIGNAL - Price: {current_price:.2f}, BB Lower: {bb_lower:.2f}, "
                    f"Volume Ratio: {current_volume/vol_ma:.2f} (> {min_volume_ratio})"
                )
            return entry_signal

        except Exception:
            logger.exception("Error in BBVolumeSupertrendEntryMixin.should_enter")
            return False
