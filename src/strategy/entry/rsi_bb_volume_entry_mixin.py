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

Configuration Example (New TALib Architecture):
    {
        "entry_logic": {
            "name": "RSIBBVolumeEntryMixin",
            "indicators": [
                {
                    "type": "RSI",
                    "params": {"timeperiod": 14},
                    "fields_mapping": {"rsi": "entry_rsi"}
                },
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
                }
            ],
            "logic_params": {
                "oversold": 30,
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


class RSIBBVolumeEntryMixin(BaseEntryMixin):
    """Entry mixin based on RSI, Bollinger Bands, and Volume.
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
            "bb_period": 20,
            "bb_dev": 2.0,
            "vol_ma_period": 20,
            "use_bb_touch": True,
            "min_volume_ratio": 1.1,
        }

    @classmethod
    def get_indicator_config(cls, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Define indicators required by this mixin."""
        rsi_period = params.get("rsi_period", 14)
        bb_period = params.get("bb_period", 20)
        bb_dev = params.get("bb_dev", 2.0)
        vol_ma_period = params.get("vol_ma_period", 20)

        return [
            {
                "type": "RSI",
                "params": {"timeperiod": rsi_period},
                "fields_mapping": {"rsi": "entry_rsi"}
            },
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
            }
        ]

    def _init_indicators(self):
        """No-op for new architecture."""
        pass

    def get_minimum_lookback(self) -> int:
        """Returns the minimum number of bars required."""
        return max(
            self.get_param("rsi_period", 14),
            self.get_param("bb_period", 20),
            self.get_param("vol_ma_period", 20)
        )

    def are_indicators_ready(self) -> bool:
        """Check if required indicators exist in the strategy registry."""
        required = ['entry_rsi', 'entry_bb_lower', 'entry_volume_ma']
        return all(alias in getattr(self.strategy, 'indicators', {}) for alias in required)

    def should_enter(self) -> bool:
        """Check if we should enter a position."""
        if not self.are_indicators_ready():
            return False

        try:
            current_price = self.strategy.data.close[0]
            current_volume = self.strategy.data.volume[0]

            # Standardized parameter retrieval
            oversold = self.get_param("oversold") or self.get_param("rsi_oversold") or self.get_param("e_rsi_oversold", 30)
            use_bb_touch = self.get_param("use_bb_touch", self.get_param("e_use_bb_touch", True))
            min_volume_ratio = self.get_param("min_volume_ratio", self.get_param("e_min_volume_ratio", 1.1))

            # Unified Indicator Access
            rsi_value = self.get_indicator('entry_rsi')
            bb_lower = self.get_indicator('entry_bb_lower')
            vol_ma = self.get_indicator('entry_volume_ma')

            # Check RSI
            rsi_condition = rsi_value <= oversold

            # Check Bollinger Bands
            if use_bb_touch:
                bb_condition = current_price <= bb_lower
            else:
                bb_condition = current_price < bb_lower

            # Check Volume
            volume_condition = current_volume > vol_ma * min_volume_ratio

            entry_signal = rsi_condition and bb_condition and volume_condition
            if entry_signal:
                logger.debug(
                    f"ENTRY SIGNAL - Price: {current_price:.2f}, RSI: {rsi_value:.2f} (<= {oversold}), "
                    f"BB Lower: {bb_lower:.2f}, Volume: {current_volume:.0f}, Volume MA: {vol_ma:.0f} "
                    f"(Ratio: {current_volume/vol_ma:.2f} > {min_volume_ratio})"
                )
            return entry_signal
        except Exception:
            logger.exception("Error in should_enter: ")
            return False
