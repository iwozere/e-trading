"""
EOM MACD Breakout Entry Mixin

This module implements BUY #3: MACD Bullish Turn + S/R Break (Momentum + Trend Confirmation)

Entry Signal Logic:
-------------------
Combine trend structure (MACD) + breakout (S/R).

Conditions (all must be true):
1. MACD bullish: MACD line crosses above Signal line AND MACD histogram rising
2. Resistance pre-breakout: Close in range (Resistance * resistance_range_low ... Resistance * resistance_range_high)
   (tight consolidation near breakout level)
3. EOM positive: EOM > 0
4. Volume expansion: Volume >= volume_threshold * Volume_SMA
"""

from typing import Any, Dict, Optional
import math

from src.strategy.entry.base_entry_mixin import BaseEntryMixin
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class EOMMAcdBreakoutEntryMixin(BaseEntryMixin):
    """
    Entry mixin for BUY #3: MACD Bullish Turn + S/R Break.

    Supports both new TALib-based architecture and legacy configurations.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the mixin with parameters"""
        super().__init__(params)
        self.use_new_architecture = False

    def get_required_params(self) -> list:
        """There are no required parameters - all have default values"""
        return []

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """Default parameters"""
        return {
            "e_resistance_range_low": 0.995,
            "e_resistance_range_high": 1.002,
            "e_volume_threshold": 0.8,
            "resistance_lookback": 2,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "eom_period": 14,
            "vol_sma_period": 20,
        }

    def init_entry(self, strategy, additional_params: Optional[Dict[str, Any]] = None):
        """Standardize architecture detection."""
        if hasattr(strategy, 'indicators') and strategy.indicators:
            self.use_new_architecture = True
        else:
            self.use_new_architecture = False
        super().init_entry(strategy, additional_params)

    def _init_indicators(self):
        """Indicators managed by strategy in unified architecture."""
        pass

    def get_minimum_lookback(self) -> int:
        """Returns the minimum number of bars required."""
        periods = [
            self.get_param("macd_slow", 26) + self.get_param("macd_signal", 9),
            self.get_param("eom_period", 14),
            self.get_param("vol_sma_period", 20),
            self.get_param("resistance_lookback", 2) * 5
        ]
        return max(periods)

    def are_indicators_ready(self) -> bool:
        """Check if indicators are initialized."""
        required = [
            'entry_resistance', 'entry_macd', 'entry_macd_signal',
            'entry_macd_hist', 'entry_eom', 'entry_volume_sma'
        ]
        if hasattr(self.strategy, 'indicators') and self.strategy.indicators:
            return all(alias in self.strategy.indicators for alias in required)
        return len(self.indicators) > 0 or self.use_new_architecture

    def should_enter(self) -> bool:
        """
        Determines if the mixin should enter a position.

        Returns:
            bool: True if all entry conditions are met
        """
        if not self.are_indicators_ready():
            return False

        try:
            # Get current price data
            close = self.strategy.data.close[0]
            volume = self.strategy.data.volume[0]

            # Unified Indicator Access
            resistance = self.get_indicator('entry_resistance')
            macd = self.get_indicator('entry_macd')
            macd_prev = self.get_indicator_prev('entry_macd', 1)
            macd_signal = self.get_indicator('entry_macd_signal')
            macd_signal_prev = self.get_indicator_prev('entry_macd_signal', 1)
            macd_hist = self.get_indicator('entry_macd_hist')
            macd_hist_prev = self.get_indicator_prev('entry_macd_hist', 1)
            eom = self.get_indicator('entry_eom')
            volume_sma = self.get_indicator('entry_volume_sma')

            # Get parameters
            resistance_range_low = self.get_param("e_resistance_range_low") or self.get_param("resistance_range_low", 0.995)
            resistance_range_high = self.get_param("e_resistance_range_high") or self.get_param("resistance_range_high", 1.002)
            volume_threshold = self.get_param("e_volume_threshold") or self.get_param("volume_threshold", 0.8)

            # Check if resistance is valid (not NaN)
            if math.isnan(resistance):
                return False

            # 1. MACD bullish: MACD line crosses above Signal line AND histogram rising
            is_macd_crossover = macd > macd_signal and macd_prev <= macd_signal_prev
            is_hist_rising = macd_hist > macd_hist_prev

            if not (is_macd_crossover and is_hist_rising):
                return False

            # 2. Resistance pre-breakout: Close in range
            lower_bound = resistance * resistance_range_low
            upper_bound = resistance * resistance_range_high
            is_near_resistance = lower_bound <= close <= upper_bound

            if not is_near_resistance:
                return False

            # 3. EOM positive: EOM > 0
            is_eom_positive = eom > 0

            if not is_eom_positive:
                return False

            # 4. Volume expansion
            volume_floor = volume_threshold * volume_sma
            is_volume_sufficient = volume >= volume_floor

            if not is_volume_sufficient:
                return False

            # All conditions met
            _logger.info(
                f"EOM MACD Breakout Entry signal: close={close:.2f}, resistance={resistance:.2f}, "
                f"macd={macd:.4f}, macd_signal={macd_signal:.4f}, eom={eom:.4f}, volume={volume:.0f}"
            )
            return True

        except KeyError as e:
            _logger.warning(f"Required indicator not found: {e}")
            return False
        except Exception:
            _logger.exception("Error in EOMMAcdBreakoutEntryMixin.should_enter")
            return False
