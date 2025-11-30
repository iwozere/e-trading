"""
EOM Breakdown Exit Mixin

This module implements SELL #1: Breakdown + EOM Negative (Momentum Breakdown)

Exit Signal Logic:
------------------
Opposite of Buy #1 — strong bearish momentum.

Conditions (all must be true):
1. Breakdown: Close < Support * (1 - breakdown_threshold)
2. EOM bearish: EOM < 0 AND EOM falling (EOM[0] < EOM[-1])
3. Volume confirmation: Volume > Volume_SMA
4. ATR confirmation: ATR rising vs yesterday

Configuration Example:
    {
        "exit_logic": {
            "name": "EOMBreakdownExitMixin",
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
                    "type": "EOM",
                    "params": {"timeperiod": 14},
                    "fields_mapping": {"eom": "exit_eom"}
                },
                {
                    "type": "SMA",
                    "params": {"timeperiod": 20},
                    "data_field": "volume",
                    "fields_mapping": {"sma": "exit_volume_sma"}
                },
                {
                    "type": "ATR",
                    "params": {"timeperiod": 14},
                    "fields_mapping": {"atr": "exit_atr"}
                }
            ],
            "logic_params": {
                "x_breakdown_threshold": 0.002
            }
        }
    }
"""

from typing import Any, Dict, Optional
import math

from src.strategy.exit.base_exit_mixin import BaseExitMixin
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class EOMBreakdownExitMixin(BaseExitMixin):
    """
    Exit mixin for SELL #1: Breakdown + EOM Negative

    Purpose: Exit on strong bearish momentum breakdown.

    Indicators required (provided by strategy):
        - exit_support: Support level from SupportResistance indicator
        - exit_eom: EOM value
        - exit_volume_sma: Volume SMA for confirmation
        - exit_atr: ATR for volatility confirmation

    Parameters:
        x_breakdown_threshold: Breakdown threshold below support (default: 0.002 = 0.2%)
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
            "x_breakdown_threshold": 0.002,  # 0.2% breakdown threshold
        }

    def _init_indicators(self):
        """
        Initialize indicators (new architecture only).

        In new architecture, indicators are created by the strategy
        and accessed via get_indicator().
        """
        # New architecture: indicators already created by strategy
        _logger.debug("EOMBreakdownExitMixin: indicators provided by strategy")

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
            eom = self.get_indicator('exit_eom')
            eom_prev = self.get_indicator_prev('exit_eom')
            volume_sma = self.get_indicator('exit_volume_sma')
            atr = self.get_indicator('exit_atr')
            atr_prev = self.get_indicator_prev('exit_atr')

            # Get parameters
            breakdown_threshold = self.get_param("x_breakdown_threshold")

            # Check if support is valid (not NaN)
            if math.isnan(support):
                _logger.debug("No support level found, cannot check breakdown")
                return False

            # 1. Breakdown: Close < Support * (1 - threshold)
            breakdown_level = support * (1 - breakdown_threshold)
            is_breakdown = close < breakdown_level

            if not is_breakdown:
                return False

            # 2. EOM bearish: EOM < 0 AND EOM falling
            is_eom_bearish = eom < 0 and eom < eom_prev

            if not is_eom_bearish:
                _logger.debug(
                    "EOM not bearish: eom=%s, eom_prev=%s",
                    eom,
                    eom_prev
                )
                return False

            # 3. Volume confirmation: Volume > Volume_SMA
            is_volume_confirmed = volume > volume_sma

            if not is_volume_confirmed:
                _logger.debug(
                    "Volume not confirmed: volume=%s, volume_sma=%s",
                    volume,
                    volume_sma
                )
                return False

            # 4. ATR confirmation: ATR rising
            is_atr_rising = atr > atr_prev

            if not is_atr_rising:
                _logger.debug(
                    "ATR not rising: atr=%s, atr_prev=%s",
                    atr,
                    atr_prev
                )
                return False

            # All conditions met
            self._exit_reason = (
                f"breakdown_momentum: close={close:.2f}, support={support:.2f}, "
                f"breakdown_level={breakdown_level:.2f}, eom={eom:.2f}, "
                f"volume={volume:.0f}, volume_sma={volume_sma:.0f}"
            )
            _logger.info("EOM Breakdown Exit signal: %s", self._exit_reason)
            return True

        except KeyError as e:
            _logger.error("Required indicator not found: %s", e)
            return False
        except Exception as e:
            _logger.error("Error in should_exit: %s", e, exc_info=True)
            return False

    def get_exit_reason(self) -> str:
        """Get the reason for exit (called after should_exit returns True)."""
        return self._exit_reason if self._exit_reason else "breakdown_momentum"
