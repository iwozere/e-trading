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

from typing import Any, Dict, Optional

import backtrader as bt
from src.strategy.entry.base_entry_mixin import BaseEntryMixin
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


class RSIBBVolumeEntryMixin(BaseEntryMixin):
    """Entry mixin based on RSI, Bollinger Bands, and Volume.

    Supports both new TALib-based architecture (indicators created by strategy)
    and legacy architecture (indicators created by mixin).
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the mixin with parameters"""
        super().__init__(params)

        # Legacy architecture support
        self.rsi_name = "entry_rsi"
        self.bb_name = "entry_bb"
        self.vol_ma_name = "entry_volume_ma"
        self.rsi = None
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
            "e_rsi_period": 14,
            "e_rsi_oversold": 30,
            "e_bb_period": 20,
            "e_bb_dev": 2.0,
            "e_vol_ma_period": 20,
            "e_use_bb_touch": True,
            "e_min_volume_ratio": 1.1,
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
        """Initialize indicators (legacy architecture only)."""
        if self.use_new_architecture:
            return

        if not hasattr(self, "strategy"):
            logger.error("No strategy available in _init_indicators")
            return

        try:
            from src.indicators.adapters.backtrader_wrappers import (
                UnifiedRSIIndicator,
                UnifiedBollingerBandsIndicator
            )

            rsi_period = self.get_param("e_rsi_period")
            bb_period = self.get_param("e_bb_period")
            bb_dev_factor = self.get_param("e_bb_dev")
            sma_period = self.get_param("e_vol_ma_period")

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

            # Create volume SMA using standard Backtrader
            if self.strategy.use_talib:
                self.sma = bt.talib.SMA(self.strategy.data.volume, sma_period)
            else:
                self.sma = bt.indicators.SMA(
                    self.strategy.data.volume, period=sma_period
                )

            self.register_indicator(self.rsi_name, self.rsi)
            self.register_indicator(self.bb_name, self.bb)
            self.register_indicator(self.vol_ma_name, self.sma)

        except Exception:
            logger.exception("Error initializing indicators: ")
            raise

    def get_minimum_lookback(self) -> int:
        """Returns the minimum number of bars required."""
        if self.use_new_architecture:
            return max(
                self.get_param("rsi_period", 14),
                self.get_param("bb_period", 20),
                self.get_param("vol_ma_period", 20)
            )
        else:
            return max(
                self.get_param("e_rsi_period", 14),
                self.get_param("e_bb_period", 20),
                self.get_param("e_vol_ma_period", 20)
            )

    def are_indicators_ready(self) -> bool:
        """Check if indicators are initialized."""
        required = ['entry_rsi', 'entry_bb_lower', 'entry_volume_ma']
        if self.use_new_architecture:
            return all(alias in getattr(self.strategy, 'indicators', {}) for alias in required)
        else:
            return all(name in self.indicators for name in [self.rsi_name, self.bb_name, self.vol_ma_name])

    def should_enter(self) -> bool:
        """Check if we should enter a position."""
        if not self.are_indicators_ready():
            return False

        try:
            current_price = self.strategy.data.close[0]
            current_volume = self.strategy.data.volume[0]

            # Standardized parameter retrieval
            oversold = self.get_param("oversold") or self.get_param("e_rsi_oversold", 30)
            use_bb_touch = self.get_param("use_bb_touch", self.get_param("e_use_bb_touch", True))
            min_volume_ratio = self.get_param("min_volume_ratio", self.get_param("e_min_volume_ratio", 1.1))

            # Unified Indicator Access
            if self.use_new_architecture:
                rsi_value = self.get_indicator('entry_rsi')
                bb_lower = self.get_indicator('entry_bb_lower')
                vol_ma = self.get_indicator('entry_volume_ma')
            else:
                rsi = self.indicators[self.rsi_name]
                bb = self.indicators[self.bb_name]
                vol_ma_ind = self.indicators[self.vol_ma_name]
                rsi_value = rsi.rsi[0]
                bb_lower = bb.lower[0]
                vol_ma = vol_ma_ind[0]

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
