"""
EOM Rejection Exit Mixin

This module implements SELL #2: Resistance Reject + EOM Turns Negative (Momentum Reversal Down)

Exit Signal Logic:
------------------
Fade failed breakout / mean reversion down.

Conditions (all must be true):
1. Price rejection at resistance: High >= Resistance * resistance_threshold AND Close < Open (bearish rejection candle)
2. EOM crosses below 0: bearish EOM momentum reversal
3. RSI overbought → falling: RSI > rsi_overbought AND RSI falling (RSI[0] < RSI[-1])

Configuration Example:
    {
        "exit_logic": {
            "name": "EOMRejectionExitMixin",
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
                    "type": "RSI",
                    "params": {"timeperiod": 14},
                    "fields_mapping": {"rsi": "exit_rsi"}
                }
            ],
            "logic_params": {
                "x_resistance_threshold": 0.995,
                "x_rsi_overbought": 60
            }
        }
    }
"""

from typing import Any, Dict, Optional
import math

from src.strategy.exit.base_exit_mixin import BaseExitMixin
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class EOMRejectionExitMixin(BaseExitMixin):
    """
    Exit mixin for SELL #2: Resistance Reject + EOM Turns Negative

    Purpose: Fade failed breakout / mean reversion down.

    Indicators required (provided by strategy):
        - exit_resistance: Resistance level from SupportResistance indicator
        - exit_eom: EOM value
        - exit_rsi: RSI for overbought check

    Parameters:
        x_resistance_threshold: Resistance rejection threshold (default: 0.995)
        x_rsi_overbought: RSI overbought threshold (default: 60)
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
            "x_resistance_threshold": 0.995,
            "x_rsi_overbought": 60,
        }

    def _init_indicators(self):
        """
        Initialize indicators (new architecture only).

        In new architecture, indicators are created by the strategy
        and accessed via get_indicator().
        """
        # New architecture: indicators already created by strategy
        #_logger.debug("EOMRejectionExitMixin: indicators provided by strategy")
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
            open_price = self.strategy.data.open[0]
            high = self.strategy.data.high[0]

            # Get indicator values
            resistance = self.get_indicator('exit_resistance')
            eom = self.get_indicator('exit_eom')
            eom_prev = self.get_indicator_prev('exit_eom')
            rsi = self.get_indicator('exit_rsi')
            rsi_prev = self.get_indicator_prev('exit_rsi')

            # Get parameters
            resistance_threshold = self.get_param("x_resistance_threshold")
            rsi_overbought = self.get_param("x_rsi_overbought")

            # Check if resistance is valid (not NaN)
            if math.isnan(resistance):
                _logger.debug("No resistance level found, cannot check rejection")
                return False

            # 1. Price rejection at resistance
            # High >= Resistance * threshold AND Close < Open (bearish rejection candle)
            resistance_level = resistance * resistance_threshold
            is_at_resistance = high >= resistance_level
            is_rejection_candle = close < open_price

            if not (is_at_resistance and is_rejection_candle):
                return False

            # 2. EOM crosses below 0: bearish EOM momentum reversal
            is_eom_crossing_down = eom < 0 and eom_prev >= 0

            if not is_eom_crossing_down:
                _logger.debug(
                    "EOM not crossing down: eom=%s, eom_prev=%s",
                    eom,
                    eom_prev
                )
                return False

            # 3. RSI overbought → falling: RSI > overbought AND RSI falling
            is_rsi_overbought = rsi > rsi_overbought
            is_rsi_falling = rsi < rsi_prev

            if not (is_rsi_overbought and is_rsi_falling):
                _logger.debug(
                    "RSI conditions not met: rsi=%s (overbought=%s), rsi_prev=%s",
                    rsi,
                    rsi_overbought,
                    rsi_prev
                )
                return False

            # All conditions met
            self._exit_reason = (
                f"resistance_rejection: close={close:.2f}, high={high:.2f}, "
                f"resistance={resistance:.2f}, resistance_level={resistance_level:.2f}, "
                f"eom={eom:.2f}, rsi={rsi:.2f}"
            )
            _logger.info("EOM Rejection Exit signal: %s", self._exit_reason)
            return True

        except KeyError as e:
            _logger.error("Required indicator not found: %s", e)
            return False
        except Exception as e:
            _logger.error("Error in should_exit: %s", e, exc_info=True)
            return False

    def get_exit_reason(self) -> str:
        """Get the reason for exit (called after should_exit returns True)."""
        return self._exit_reason if self._exit_reason else "resistance_rejection"
