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

Configuration Example:
    {
        "entry_logic": {
            "name": "EOMMAcdBreakoutEntryMixin",
            "indicators": [
                {
                    "type": "SupportResistance",
                    "params": {"lookback_bars": 2},
                    "fields_mapping": {
                        "resistance": "entry_resistance",
                        "support": "entry_support"
                    }
                },
                {
                    "type": "MACD",
                    "params": {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
                    "fields_mapping": {
                        "macd": "entry_macd",
                        "signal": "entry_macd_signal",
                        "hist": "entry_macd_hist"
                    }
                },
                {
                    "type": "EOM",
                    "params": {"timeperiod": 14},
                    "fields_mapping": {"eom": "entry_eom"}
                },
                {
                    "type": "SMA",
                    "params": {"timeperiod": 20},
                    "data_field": "volume",
                    "fields_mapping": {"sma": "entry_volume_sma"}
                }
            ],
            "logic_params": {
                "e_resistance_range_low": 0.995,
                "e_resistance_range_high": 1.002,
                "e_volume_threshold": 0.8
            }
        }
    }
"""

from typing import Any, Dict, Optional
import math

from src.strategy.entry.base_entry_mixin import BaseEntryMixin
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class EOMMAcdBreakoutEntryMixin(BaseEntryMixin):
    """
    Entry mixin for BUY #3: MACD Bullish Turn + S/R Break

    Purpose: Combine trend structure (MACD) + breakout (S/R).

    Indicators required (provided by strategy):
        - entry_resistance: Resistance level from SupportResistance indicator
        - entry_macd: MACD line
        - entry_macd_signal: MACD signal line
        - entry_macd_hist: MACD histogram
        - entry_eom: EOM value
        - entry_volume_sma: Volume SMA for expansion check

    Parameters:
        e_resistance_range_low: Lower bound multiplier for resistance range (default: 0.995)
        e_resistance_range_high: Upper bound multiplier for resistance range (default: 1.002)
        e_volume_threshold: Volume threshold multiplier (default: 0.8)
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
            "e_resistance_range_low": 0.995,
            "e_resistance_range_high": 1.002,
            "e_volume_threshold": 0.8,
        }

    def _init_indicators(self):
        """
        Initialize indicators (new architecture only).

        In new architecture, indicators are created by the strategy
        and accessed via get_indicator().
        """
        # New architecture: indicators already created by strategy
        #_logger.debug("EOMMAcdBreakoutEntryMixin: indicators provided by strategy")
        pass

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

            # Get indicator values
            resistance = self.get_indicator('entry_resistance')
            macd = self.get_indicator('entry_macd')
            macd_signal = self.get_indicator('entry_macd_signal')
            macd_hist = self.get_indicator('entry_macd_hist')
            macd_signal_prev = self.get_indicator_prev('entry_macd_signal')
            macd_hist_prev = self.get_indicator_prev('entry_macd_hist')
            eom = self.get_indicator('entry_eom')
            volume_sma = self.get_indicator('entry_volume_sma')

            # Get parameters
            resistance_range_low = self.get_param("e_resistance_range_low")
            resistance_range_high = self.get_param("e_resistance_range_high")
            volume_threshold = self.get_param("e_volume_threshold")

            # Check if resistance is valid (not NaN)
            if math.isnan(resistance):
                _logger.debug("No resistance level found, cannot check pre-breakout")
                return False

            # 1. MACD bullish: MACD line crosses above Signal line AND histogram rising
            is_macd_crossover = macd > macd_signal and macd_signal_prev >= macd_signal_prev
            is_hist_rising = macd_hist > macd_hist_prev

            if not (is_macd_crossover and is_hist_rising):
                _logger.debug(
                    "MACD not bullish: macd=%s, signal=%s, hist=%s, hist_prev=%s",
                    macd,
                    macd_signal,
                    macd_hist,
                    macd_hist_prev
                )
                return False

            # 2. Resistance pre-breakout: Close in range (Resistance * low ... Resistance * high)
            # Tight consolidation near breakout level
            lower_bound = resistance * resistance_range_low
            upper_bound = resistance * resistance_range_high
            is_near_resistance = lower_bound <= close <= upper_bound

            if not is_near_resistance:
                _logger.debug(
                    "Not near resistance: close=%s, range=[%s, %s]",
                    close,
                    lower_bound,
                    upper_bound
                )
                return False

            # 3. EOM positive: EOM > 0
            is_eom_positive = eom > 0

            if not is_eom_positive:
                _logger.debug("EOM not positive: eom=%s", eom)
                return False

            # 4. Volume expansion: Volume >= threshold * Volume_SMA
            volume_floor = volume_threshold * volume_sma
            is_volume_sufficient = volume >= volume_floor

            if not is_volume_sufficient:
                _logger.debug(
                    "Volume insufficient: volume=%s, floor=%s",
                    volume,
                    volume_floor
                )
                return False

            # All conditions met
            _logger.info(
                "EOM MACD Breakout Entry signal: close=%s, resistance=%s, "
                "macd=%s, macd_signal=%s, macd_hist=%s, eom=%s, volume=%s, volume_sma=%s",
                close, resistance, macd, macd_signal, macd_hist, eom, volume, volume_sma
            )
            return True

        except KeyError as e:
            _logger.error("Required indicator not found: %s", e)
            return False
        except Exception as e:
            _logger.error("Error in should_enter: %s", e, exc_info=True)
            return False
