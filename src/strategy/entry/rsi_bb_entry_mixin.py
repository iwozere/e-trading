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

from typing import Any, Dict, Optional, List

from src.strategy.entry.base_entry_mixin import BaseEntryMixin
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


class RSIBBEntryMixin(BaseEntryMixin):
    """Entry mixin that combines RSI and Bollinger Bands for entry signals.
    New Architecture only.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the mixin with parameters"""
        super().__init__(params)
        self.last_entry_bar = None

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
            "use_bb_touch": True,
            "rsi_cross": False,
            "bb_reentry": False,
            "cooldown_bars": 0,
        }

    @classmethod
    def get_indicator_config(cls, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Define indicators required by this mixin."""
        rsi_period = params.get("rsi_period") or params.get("e_rsi_period", 14)
        bb_period = params.get("bb_period") or params.get("e_bb_period", 20)
        bb_dev = params.get("bb_dev") or params.get("e_bb_dev", 2.0)

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
            }
        ]

    def _init_indicators(self):
        """No-op for new architecture."""
        pass

    def get_minimum_lookback(self) -> int:
        """Returns the minimum number of bars required."""
        return max(
            self.get_param("rsi_period") or self.get_param("e_rsi_period", 14),
            self.get_param("bb_period") or self.get_param("e_bb_period", 20)
        )

    def are_indicators_ready(self) -> bool:
        """Check if required indicators exist in the strategy registry."""
        required = ['entry_rsi', 'entry_bb_lower', 'entry_bb_middle']

        indicators = getattr(self.strategy, 'indicators', {})
        missing = [alias for alias in required if alias not in indicators]

        if missing:
            logger.debug(f"Indicators not ready for {type(self).__name__}: missing {missing}")
            return False

        return True

    def should_enter(self) -> bool:
        """Check if we should enter a position."""
        if not self.are_indicators_ready():
            return False

        try:
            # Check cooldown period
            cooldown_bars = self.get_param("cooldown_bars") or self.get_param("e_cooldown_bars", 0)
            if cooldown_bars > 0:
                current_bar = len(self.strategy.data)
                if (self.last_entry_bar is not None and
                    current_bar - self.last_entry_bar < cooldown_bars):
                    return False

            current_price = self.strategy.data.close[0]

            # Standardized parameter retrieval
            oversold = self.get_param("rsi_oversold") or self.get_param("e_rsi_oversold", 30)
            use_bb_touch = self.get_param("use_bb_touch") or self.get_param("e_use_bb_touch", True)
            rsi_cross = self.get_param("rsi_cross") or self.get_param("e_rsi_cross", False)
            bb_reentry = self.get_param("bb_reentry") or self.get_param("e_bb_reentry", False)

            # Unified Indicator Access
            rsi_value = self.get_indicator('entry_rsi')
            rsi_prev = self.get_indicator_prev('entry_rsi', 1)
            bb_lower = self.get_indicator('entry_bb_lower')
            bb_middle = self.get_indicator('entry_bb_middle')

            # Validate indicator values
            if rsi_value is None or rsi_prev is None:
                return False

            if bb_lower is None or bb_middle is None:
                return False

            # RSI condition with optional cross confirmation
            if rsi_cross:
                rsi_condition = (rsi_prev <= oversold and rsi_value > oversold)
            else:
                rsi_condition = rsi_value <= oversold

            # Bollinger Bands condition
            if bb_reentry:
                bb_condition = current_price > bb_lower
            else:
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

        except Exception:
            logger.exception("Error in should_enter: ")
            return False

    def get_entry_reason(self) -> str:
        return "RSI Oversold + BB Lower Touch"
