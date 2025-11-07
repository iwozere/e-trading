"""
RSI and Bollinger Bands Entry Mixin

This module implements an entry strategy based on the combination of:
1. RSI (Relative Strength Index)
2. Bollinger Bands

The strategy enters a position when:
1. RSI is oversold
2. Price is below the lower Bollinger Band

Configuration Example (New TALib Architecture):
    {
        "entry_logic": {
            "name": "RSIBBEntryMixin",
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
                }
            ],
            "logic_params": {
                "oversold": 30,
                "use_bb_touch": true,
                "rsi_cross": false,
                "bb_reentry": false,
                "cooldown_bars": 0
            }
        }
    }

Legacy Configuration (Backward Compatible):
    {
        "entry_logic": {
            "name": "RSIBBEntryMixin",
            "params": {
                "e_rsi_period": 14,
                "e_rsi_oversold": 30,
                "e_bb_period": 20,
                "e_bb_dev": 2.0,
                "e_use_bb_touch": true
            }
        }
    }

This strategy combines mean reversion (RSI + BB) to identify potential reversal points.
"""

from typing import Any, Dict, Optional

import backtrader as bt
from src.strategy.entry.base_entry_mixin import BaseEntryMixin
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


class RSIBBEntryMixin(BaseEntryMixin):
    """Entry mixin that combines RSI and Bollinger Bands for entry signals.

    Supports both new TALib-based architecture (indicators created by strategy)
    and legacy architecture (indicators created by mixin).
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the mixin with parameters"""
        super().__init__(params)
        self.last_entry_bar = None

        # Legacy architecture support
        self.rsi_name = "entry_rsi"
        self.bb_name = "entry_bb"
        self.rsi = None
        self.bb = None

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
            "e_use_bb_touch": True,
            "e_rsi_cross": False,
            "e_bb_reentry": False,
            "e_cooldown_bars": 0,
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
            #logger.debug("Skipping indicator initialization (new architecture)")
            return

        # Legacy architecture: create indicators in mixin
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

            # Validate parameters
            if rsi_period is None or rsi_period <= 0:
                logger.warning("Invalid RSI period: %s, using default value 14", rsi_period)
                rsi_period = 14
            elif rsi_period < 2:
                logger.warning("RSI period too small: %s, using minimum value 2", rsi_period)
                rsi_period = 2

            if bb_period is None or bb_period <= 0:
                logger.warning("Invalid BB period: %s, using default value 20", bb_period)
                bb_period = 20
            elif bb_period < 2:
                logger.warning("BB period too small: %s, using minimum value 2", bb_period)
                bb_period = 2

            if bb_dev_factor is None or bb_dev_factor <= 0:
                logger.warning("Invalid BB deviation factor: %s, using default value 2.0", bb_dev_factor)
                bb_dev_factor = 2.0

            # Create unified indicators
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

            # Register wrapped indicators
            self.register_indicator(self.rsi_name, self.rsi)
            self.register_indicator(self.bb_name, self.bb)

            logger.debug("Legacy indicators initialized: RSI(period=%d), BB(period=%d, dev=%s)",
                        rsi_period, bb_period, bb_dev_factor)

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
            required_indicators = ['entry_rsi', 'entry_bb_lower', 'entry_bb_middle']
            for ind_alias in required_indicators:
                if ind_alias not in self.strategy.indicators:
                    return False

            # Check if we can access values
            try:
                _ = self.get_indicator('entry_rsi')
                _ = self.get_indicator('entry_bb_lower')
                _ = self.get_indicator('entry_bb_middle')
                return True
            except (IndexError, KeyError, AttributeError):
                return False

        else:
            # Legacy architecture: check mixin's indicators
            # Use base class implementation first
            if not super().are_indicators_ready():
                return False

            try:
                # Check if we have enough data points
                data_length = len(self.strategy.data)
                required_length = max(self.get_param("e_rsi_period"), self.get_param("e_bb_period"))
                if data_length < required_length:
                    return False

                # Check if indicators are registered
                if (
                    self.rsi_name not in self.indicators
                    or self.bb_name not in self.indicators
                ):
                    return False

                # Check if we can access the first value of each indicator
                rsi = self.indicators[self.rsi_name]
                bb = self.indicators[self.bb_name]

                # Try to access the first value of each indicator using unified access
                _ = rsi.rsi[0]
                _ = bb.lower[0]  # Use unified access - works for both TALib and standard
                _ = bb.middle[0]
                _ = bb.upper[0]

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
            # Check cooldown period
            cooldown_bars = self.get_param("e_cooldown_bars") or self.get_param("cooldown_bars", 0)
            if cooldown_bars > 0:
                current_bar = len(self.strategy.data)
                if (self.last_entry_bar is not None and
                    current_bar - self.last_entry_bar < cooldown_bars):
                    return False

            current_price = self.strategy.data.close[0]

            # Get indicator values based on architecture
            if self.use_new_architecture:
                # New architecture: access via get_indicator()
                rsi_value = self.get_indicator('entry_rsi')
                rsi_prev = self.get_indicator_prev('entry_rsi', 1)
                bb_lower = self.get_indicator('entry_bb_lower')
                bb_middle = self.get_indicator('entry_bb_middle')

                # Get thresholds from logic_params (new) or fallback to legacy params
                oversold = self.get_param("oversold") or self.get_param("e_rsi_oversold", 30)
                use_bb_touch = self.get_param("use_bb_touch", self.get_param("e_use_bb_touch", True))
                rsi_cross = self.get_param("rsi_cross", self.get_param("e_rsi_cross", False))
                bb_reentry = self.get_param("bb_reentry", self.get_param("e_bb_reentry", False))

            else:
                # Legacy architecture: access via mixin's indicators dict
                rsi = self.indicators[self.rsi_name]
                bb = self.indicators[self.bb_name]

                rsi_value = rsi.rsi[0]
                rsi_prev = rsi.rsi[-1] if len(rsi.rsi) > 1 else rsi_value
                bb_lower = bb.lower[0]
                bb_middle = bb.middle[0]

                # Get thresholds from legacy params
                oversold = self.get_param("e_rsi_oversold", 30)
                use_bb_touch = self.get_param("e_use_bb_touch", True)
                rsi_cross = self.get_param("e_rsi_cross", False)
                bb_reentry = self.get_param("e_bb_reentry", False)

            # Validate indicator values
            if rsi_value is None or rsi_prev is None:
                logger.warning("RSI value is None, skipping entry check")
                return False

            if bb_lower is None or bb_middle is None:
                logger.warning("Bollinger Bands values are None, skipping entry check")
                return False

            # RSI condition with optional cross confirmation
            if rsi_cross:
                # RSI cross upward: require RSI to cross back above oversold threshold
                rsi_condition = (rsi_prev <= oversold and rsi_value > oversold)
            else:
                # Original RSI condition
                rsi_condition = rsi_value <= oversold

            # Bollinger Bands condition with optional re-entry confirmation
            if bb_reentry:
                # BB re-entry: require close > lower BB after touching below
                bb_condition = current_price > bb_lower
            else:
                # Original BB condition
                if use_bb_touch:
                    bb_condition = current_price <= bb_lower
                else:
                    bb_condition = current_price < bb_lower

            entry_signal = rsi_condition and bb_condition

            if entry_signal:
                self.last_entry_bar = len(self.strategy.data)
                logger.debug(
                    f"ENTRY SIGNAL - Price: {current_price:.2f}, RSI: {rsi_value:.2f} (<= {oversold}), "
                    f"BB Lower: {bb_lower:.2f}, RSI Cross: {rsi_cross}, BB Reentry: {bb_reentry}"
                )

            return entry_signal

        except Exception as e:
            logger.exception("Error in should_enter: ")
            return False
