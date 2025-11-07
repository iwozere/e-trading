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
from src.strategy.entry.base_entry_mixin import BaseEntryMixin
from src.indicators.adapters.backtrader_wrappers import UnifiedRSIIndicator, UnifiedBollingerBandsIndicator
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
        """Initialize indicators (legacy architecture only).

        In new architecture, indicators are created by the strategy
        and accessed via get_indicator().
        """
        if self.use_new_architecture:
            # New architecture: indicators already created by strategy
            return

        # Legacy architecture: create indicators in mixin
        logger.debug("RSIBBVolumeEntryMixin._init_indicators called (legacy architecture)")
        if not hasattr(self, "strategy"):
            logger.error("No strategy available in _init_indicators")
            return

        try:
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

            logger.debug("Legacy indicators initialized: RSI(period=%d), BB(period=%d, dev=%s), Volume MA(period=%d)",
                        rsi_period, bb_period, bb_dev_factor, sma_period)

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
            required_indicators = ['entry_rsi', 'entry_bb_lower', 'entry_bb_middle', 'entry_volume_ma']
            for ind_alias in required_indicators:
                if ind_alias not in self.strategy.indicators:
                    return False

            # Check if we can access values
            try:
                _ = self.get_indicator('entry_rsi')
                _ = self.get_indicator('entry_bb_lower')
                _ = self.get_indicator('entry_bb_middle')
                _ = self.get_indicator('entry_volume_ma')
                return True
            except (IndexError, KeyError, AttributeError):
                return False

        else:
            # Legacy architecture: check mixin's indicators
            if not super().are_indicators_ready():
                return False

            try:
                # Check if we have enough data points
                data_length = len(self.strategy.data)
                required_length = max(
                    self.get_param("e_rsi_period"),
                    self.get_param("e_bb_period"),
                    self.get_param("e_vol_ma_period")
                )
                if data_length < required_length:
                    return False

                # Check if indicators are registered
                if (
                    self.rsi_name not in self.indicators
                    or self.bb_name not in self.indicators
                    or self.vol_ma_name not in self.indicators
                ):
                    return False

                # Check if we can access the first value of each indicator
                rsi = self.indicators[self.rsi_name]
                bb = self.indicators[self.bb_name]
                vol_ma = self.indicators[self.vol_ma_name]

                # Try to access the first value of each indicator using unified access
                _ = rsi.rsi[0]
                _ = bb.lower[0]  # Use unified access
                _ = vol_ma[0]

                return True
            except (IndexError, AttributeError):
                return False

    def should_enter(self) -> bool:
        """Check if we should enter a position.

        Works with both new and legacy architectures.
        """
        if not self.are_indicators_ready():
            return False

        try:
            current_price = self.strategy.data.close[0]
            current_volume = self.strategy.data.volume[0]

            # Get indicator values based on architecture
            if self.use_new_architecture:
                # New architecture: access via get_indicator()
                rsi_value = self.get_indicator('entry_rsi')
                bb_lower = self.get_indicator('entry_bb_lower')
                vol_ma = self.get_indicator('entry_volume_ma')

                # Get thresholds from logic_params (new) or fallback to legacy params
                oversold = self.get_param("oversold") or self.get_param("e_rsi_oversold", 30)
                use_bb_touch = self.get_param("use_bb_touch", self.get_param("e_use_bb_touch", True))
                min_volume_ratio = self.get_param("min_volume_ratio", self.get_param("e_min_volume_ratio", 1.1))

            else:
                # Legacy architecture: access via mixin's indicators dict
                rsi = self.indicators[self.rsi_name]
                bb = self.indicators[self.bb_name]
                vol_ma_ind = self.indicators[self.vol_ma_name]

                rsi_value = rsi.rsi[0]
                bb_lower = bb.lower[0]
                vol_ma = vol_ma_ind[0]

                # Get thresholds from legacy params
                oversold = self.get_param("e_rsi_oversold", 30)
                use_bb_touch = self.get_param("e_use_bb_touch", True)
                min_volume_ratio = self.get_param("e_min_volume_ratio", 1.1)

            # Check RSI
            rsi_condition = rsi_value <= oversold

            # Check Bollinger Bands
            if use_bb_touch:
                bb_condition = current_price <= bb_lower
            else:
                bb_condition = current_price < bb_lower

            # Check Volume
            volume_condition = current_volume > vol_ma * min_volume_ratio

            return_value = rsi_condition and bb_condition and volume_condition
            if return_value:
                logger.debug(
                    f"ENTRY SIGNAL - Price: {current_price:.2f}, RSI: {rsi_value:.2f} (<= {oversold}), "
                    f"BB Lower: {bb_lower:.2f}, Volume: {current_volume:.0f}, Volume MA: {vol_ma:.0f} "
                    f"(Ratio: {current_volume/vol_ma:.2f} > {min_volume_ratio})"
                )
            return return_value
        except Exception as e:
            logger.exception("Error in should_enter: ")
            return False
