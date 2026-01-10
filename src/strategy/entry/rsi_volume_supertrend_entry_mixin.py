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

from typing import Any, Dict, Optional

import backtrader as bt
from src.strategy.entry.base_entry_mixin import BaseEntryMixin
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


class RSIVolumeSupertrendEntryMixin(BaseEntryMixin):
    """Entry mixin based on RSI, Volume, and Supertrend.

    Supports both new TALib-based architecture and legacy configurations.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the mixin with parameters"""
        super().__init__(params)

        # Legacy architecture support
        self.rsi_name = "entry_rsi"
        self.vol_ma_name = "entry_volume_ma"
        self.supertrend_name = "entry_supertrend"
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

            rsi_period = self.get_param("e_rsi_period")
            sma_period = self.get_param("e_vol_ma_period")

            if self.strategy.use_talib:
                self.rsi = bt.talib.RSI(self.strategy.data.close, timeperiod=rsi_period)
                self.sma = bt.talib.SMA(self.strategy.data.volume, timeperiod=sma_period)
            else:
                self.rsi = bt.indicators.RSI(self.strategy.data.close, period=rsi_period)
                self.sma = bt.indicators.SMA(self.strategy.data.volume, period=sma_period)

            supertrend = UnifiedSuperTrendIndicator(
                self.strategy.data,
                length=self.get_param("e_st_period"),
                multiplier=self.get_param("e_st_multiplier"),
            )

            self.register_indicator(self.rsi_name, self.rsi)
            self.register_indicator(self.vol_ma_name, self.sma)
            self.register_indicator(self.supertrend_name, supertrend)

        except Exception:
            logger.exception("Error initializing indicators: ")
            raise

    def get_minimum_lookback(self) -> int:
        """Returns the minimum number of bars required."""
        if self.use_new_architecture:
            return max(
                self.get_param("rsi_period", 14),
                self.get_param("vol_ma_period", 20),
                self.get_param("st_period", 10)
            )
        else:
            return max(
                self.get_param("e_rsi_period", 14),
                self.get_param("e_vol_ma_period", 20),
                self.get_param("e_st_period", 10)
            )

    def are_indicators_ready(self) -> bool:
        """Check if indicators are initialized."""
        required = ['entry_rsi', 'entry_volume_ma']
        if self.use_new_architecture:
            required.append('entry_supertrend_direction')
            return all(alias in getattr(self.strategy, 'indicators', {}) for alias in required)
        else:
            required.append(self.supertrend_name)
            return all(name in self.indicators for name in [self.rsi_name, self.vol_ma_name, self.supertrend_name])

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
            if self.use_new_architecture:
                current_rsi = self.get_indicator('entry_rsi')
                vol_ma = self.get_indicator('entry_volume_ma')
                supertrend_direction = self.get_indicator('entry_supertrend_direction')
            else:
                rsi = self.indicators[self.rsi_name]
                vol_ma_ind = self.indicators[self.vol_ma_name]
                supertrend = self.indicators[self.supertrend_name]
                current_rsi = rsi[0]
                vol_ma = vol_ma_ind[0]
                supertrend_direction = supertrend.direction[0]

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
