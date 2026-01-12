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

from typing import Any, Dict, Optional, List

import backtrader as bt
from src.strategy.entry.base_entry_mixin import BaseEntryMixin
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


class BBVolumeSupertrendEntryMixin(BaseEntryMixin):
    """Entry mixin that combines Bollinger Bands, Volume, and Supertrend for entry signals.
    New Architecture only.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the mixin with parameters"""
        super().__init__(params)

    def get_required_params(self) -> list:
        """There are no required parameters - all have default values"""
        return []

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """Default parameters"""
        return {
            "bb_period": 20,
            "bb_dev": 2.0,
            "vol_ma_period": 20,
            "min_volume_ratio": 1.1,
            "st_period": 10,
            "st_multiplier": 3.0,
            "use_bb_touch": True,
        }

    @classmethod
    def get_indicator_config(cls, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Define indicators required by this mixin."""
        bb_period = params.get("bb_period", 20)
        bb_dev = params.get("bb_dev", 2.0)
        vol_ma_period = params.get("vol_ma_period", 20)
        st_period = params.get("st_period", 10)
        st_multiplier = params.get("st_multiplier", 3.0)

        return [
            {
                "type": "BBANDS",
                "params": {"timeperiod": bb_period, "nbdevup": bb_dev, "nbdevdn": bb_dev},
                "fields_mapping": {
                    "upperband": "entry_bb_upper",
                    "middleband": "entry_bb_middle",
                    "lowerband": "entry_bb_lower"
                }
            },
            {
                "type": "SMA",
                "data_inputs": ["volume"],
                "params": {"timeperiod": vol_ma_period},
                "fields_mapping": {"sma": "entry_volume_ma"}
            },
            {
                "type": "SUPERTREND",
                "params": {"length": st_period, "multiplier": st_multiplier},
                "fields_mapping": {
                    "super_trend": "entry_supertrend",
                    "direction": "entry_supertrend_direction"
                }
            }
        ]

    def _init_indicators(self):
        """No-op for new architecture."""
        pass

    def get_minimum_lookback(self) -> int:
        """Returns the minimum number of bars required."""
        return max(
            self.get_param("bb_period", 20),
            self.get_param("vol_ma_period", 20),
            self.get_param("st_period", 10)
        )

    def are_indicators_ready(self) -> bool:
        """Check if required indicators exist in the strategy registry."""
        required = ['entry_volume_ma', 'entry_bb_lower', 'entry_supertrend_direction']
        return all(alias in getattr(self.strategy, 'indicators', {}) for alias in required)

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
            bb_lower = self.get_indicator('entry_bb_lower')
            vol_ma = self.get_indicator('entry_volume_ma')
            supertrend_direction = self.get_indicator('entry_supertrend_direction')

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
