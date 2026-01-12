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

Configuration Example (New TALib Architecture):
    {
        "entry_logic": {
            "name": "RSIVolumeSupertrendEntryMixin",
            "indicators": [
                {
                    "type": "RSI",
                    "params": {"timeperiod": 14},
                    "fields_mapping": {"rsi": "entry_rsi"}
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
                "rsi_oversold": 30,
                "min_volume_ratio": 1.5
            }
        }
    }
"""

from typing import Any, Dict, Optional, List

import backtrader as bt
from src.strategy.entry.base_entry_mixin import BaseEntryMixin
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


class RSIVolumeSupertrendEntryMixin(BaseEntryMixin):
    """Entry mixin based on RSI, Volume, and Supertrend.
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
            "rsi_period": 14,
            "rsi_oversold": 30,
            "vol_ma_period": 20,
            "min_volume_ratio": 1.5,
            "st_period": 10,
            "st_multiplier": 3.0,
        }

    @classmethod
    def get_indicator_config(cls, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Define indicators required by this mixin."""
        rsi_period = params.get("rsi_period", 14)
        vol_ma_period = params.get("vol_ma_period", 20)
        st_period = params.get("st_period", 10)
        st_multiplier = params.get("st_multiplier", 3.0)

        return [
            {
                "type": "RSI",
                "params": {"timeperiod": rsi_period},
                "fields_mapping": {"rsi": "entry_rsi"}
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
            self.get_param("rsi_period", 14),
            self.get_param("vol_ma_period", 20),
            self.get_param("st_period", 10)
        )

    def are_indicators_ready(self) -> bool:
        """Check if required indicators exist in the strategy registry."""
        required = ['entry_rsi', 'entry_volume_ma', 'entry_supertrend_direction']
        return all(alias in getattr(self.strategy, 'indicators', {}) for alias in required)

    def should_enter(self) -> bool:
        """Check if we should enter a position."""
        if not self.are_indicators_ready():
            return False

        try:
            current_volume = self.strategy.data.volume[0]

            # Standardized parameter retrieval
            rsi_oversold = self.get_param("rsi_oversold") or self.get_param("e_rsi_oversold", 30)
            min_volume_ratio = self.get_param("min_volume_ratio") or self.get_param("e_min_volume_ratio", 1.5)

            # Unified Indicator Access
            current_rsi = self.get_indicator('entry_rsi')
            vol_ma = self.get_indicator('entry_volume_ma')
            supertrend_direction = self.get_indicator('entry_supertrend_direction')

            # Check conditions
            rsi_condition = current_rsi <= rsi_oversold
            volume_condition = current_volume > vol_ma * min_volume_ratio
            supertrend_condition = supertrend_direction == 1  # 1 means uptrend

            entry_signal = rsi_condition and volume_condition and supertrend_condition

            if entry_signal:
                logger.debug(
                    f"ENTRY SIGNAL - RSI: {current_rsi:.2f} (<= {rsi_oversold}), "
                    f"Volume Ratio: {current_volume/vol_ma:.2f} (> {min_volume_ratio}), "
                    f"Supertrend: {supertrend_direction}"
                )
            return entry_signal

        except Exception:
            logger.exception("Error in RSIVolumeSupertrendEntryMixin.should_enter")
            return False
