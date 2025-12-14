"""
EOM MACD Breakdown Exit Mixin

This module implements SELL #3: MACD Bearish Turn + Breakdown (Trend + Structure Confirm)

Exit Signal Logic:
------------------
Combine strong trend shift + structural breakdown.

Conditions (all must be true):
1. MACD bearish cross: MACD line crosses below Signal line AND Histogram falling
2. Breakdown confirmation: Close < Support * (1 - support_threshold)
3. EOM negative: EOM < 0
4. Volume > SMA (breakdowns on high volume are more reliable)

Configuration Example:
    {
        "exit_logic": {
            "name": "EOMMAcdBreakdownExitMixin",
            "indicators": [
                {
                    "type": "SupportResistance",
                    "params": {"lookback_bars": 2},
                    "fields_mapping": {
                        "resistance": "exit_resistance",
                        "support": "exit_support"
                    }
                },
                {
                    "type": "MACD",
                    "params": {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
                    "fields_mapping": {
                        "macd": "exit_macd",
                        "signal": "exit_macd_signal",
                        "hist": "exit_macd_hist"
                    }
                },
                {
                    "type": "EOM",
                    "params": {"timeperiod": 14},
                    "fields_mapping": {"eom": "exit_eom"}
                },
                {
                    "type": "SMA",
                    "params": {"timeperiod": 20},
                    "data_field": "volume",
                    "fields_mapping": {"sma": "exit_volume_sma"}
                }
            ],
            "logic_params": {
                "x_support_threshold": 0.002
            }
        }
    }
"""

from typing import Any, Dict, Optional
import math

from src.strategy.exit.base_exit_mixin import BaseExitMixin
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class EOMMAcdBreakdownExitMixin(BaseExitMixin):
    """
    Exit mixin for SELL #3: MACD Bearish Turn + Breakdown

    Purpose: Combine strong trend shift + structural breakdown.

    Indicators required (provided by strategy):
        - exit_support: Support level from SupportResistance indicator
        - exit_macd: MACD line
        - exit_macd_signal: MACD signal line
        - exit_macd_hist: MACD histogram
        - exit_eom: EOM value
        - exit_volume_sma: Volume SMA for confirmation

    Parameters:
        x_support_threshold: Support breakdown threshold (default: 0.002 = 0.2%)
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the mixin with parameters"""
        super().__init__(params)
        self._exit_reason = ""

    def get_required_params(self) -> list:
        """There are no required parameters - all have default values"""
        return []

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """Default parameters"""
        return {
            "x_support_threshold": 0.002,  # 0.2% support breakdown threshold
        }

    def _init_indicators(self):
        """
        Initialize indicators (new architecture only).

        In new architecture, indicators are created by the strategy
        and accessed via get_indicator().
        """
        # New architecture: indicators already created by strategy
        #_logger.debug("EOMMAcdBreakdownExitMixin: indicators provided by strategy")
        pass

    def should_exit(self) -> bool:
        """
        Determines if the mixin should exit a position.

        Returns:
            bool: True if all exit conditions are met
        """
        if not self.are_indicators_ready():
            return False

        try:
            # Get current price data
            close = self.strategy.data.close[0]
            volume = self.strategy.data.volume[0]

            # Get indicator values
            support = self.get_indicator('exit_support')
            macd = self.get_indicator('exit_macd')
            macd_signal = self.get_indicator('exit_macd_signal')
            macd_hist = self.get_indicator('exit_macd_hist')
            macd_prev = self.get_indicator_prev('exit_macd')
            macd_signal_prev = self.get_indicator_prev('exit_macd_signal')
            macd_hist_prev = self.get_indicator_prev('exit_macd_hist')
            eom = self.get_indicator('exit_eom')
            volume_sma = self.get_indicator('exit_volume_sma')

            # Get parameters
            support_threshold = self.get_param("x_support_threshold")

            # Check if support is valid (not NaN)
            if math.isnan(support):
                _logger.debug("No support level found, cannot check breakdown")
                return False

            # 1. MACD bearish cross: MACD line crosses below Signal line AND histogram falling
            is_macd_crossunder = macd < macd_signal and macd_prev >= macd_signal_prev
            is_hist_falling = macd_hist < macd_hist_prev

            if not (is_macd_crossunder and is_hist_falling):
                _logger.debug(
                    "MACD not bearish: macd=%s, signal=%s, hist=%s, hist_prev=%s",
                    macd,
                    macd_signal,
                    macd_hist,
                    macd_hist_prev
                )
                return False

            # 2. Breakdown confirmation: Close < Support * (1 - threshold)
            breakdown_level = support * (1 - support_threshold)
            is_breakdown = close < breakdown_level

            if not is_breakdown:
                _logger.debug(
                    "No breakdown: close=%s, breakdown_level=%s",
                    close,
                    breakdown_level
                )
                return False

            # 3. EOM negative: EOM < 0
            is_eom_negative = eom < 0

            if not is_eom_negative:
                _logger.debug("EOM not negative: eom=%s", eom)
                return False

            # 4. Volume > SMA (breakdowns on high volume are more reliable)
            is_volume_confirmed = volume > volume_sma

            if not is_volume_confirmed:
                _logger.debug(
                    "Volume not confirmed: volume=%s, volume_sma=%s",
                    volume,
                    volume_sma
                )
                return False

            # All conditions met
            self._exit_reason = (
                f"macd_breakdown: close={close:.2f}, support={support:.2f}, "
                f"breakdown_level={breakdown_level:.2f}, macd={macd:.4f}, "
                f"macd_signal={macd_signal:.4f}, eom={eom:.2f}, volume={volume:.0f}"
            )
            _logger.info("EOM MACD Breakdown Exit signal: %s", self._exit_reason)
            return True

        except KeyError as e:
            _logger.error("Required indicator not found: %s", e)
            return False
        except Exception as e:
            _logger.error("Error in should_exit: %s", e, exc_info=True)
            return False

    def get_exit_reason(self) -> str:
        """Get the reason for exit (called after should_exit returns True)."""
        return self._exit_reason if self._exit_reason else "macd_breakdown"
