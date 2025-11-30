"""
EOM Pullback Entry Mixin

This module implements BUY #2: Pullback to Support + EOM Reversal (Mean-Reversion Trend-Continuation)

Entry Signal Logic:
-------------------
Trend-following entry after pullback respecting support.

Conditions (all must be true):
1. Price bounces from support: Low <= Support * (1 + support_threshold) AND Close > Open (reversal candle)
2. EOM reversal: EOM crosses above 0 (momentum turning positive)
3. RSI oversold → recovery: RSI < rsi_oversold AND RSI rising (RSI[0] > RSI[-1])
4. ATR volatility floor: ATR > ATR_SMA * atr_floor_multiplier

Configuration Example:
    {
        "entry_logic": {
            "name": "EOMPullbackEntryMixin",
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
                    "type": "EOM",
                    "params": {"timeperiod": 14},
                    "fields_mapping": {"eom": "entry_eom"}
                },
                {
                    "type": "RSI",
                    "params": {"timeperiod": 14},
                    "fields_mapping": {"rsi": "entry_rsi"}
                },
                {
                    "type": "ATR",
                    "params": {"timeperiod": 14},
                    "fields_mapping": {"atr": "entry_atr"}
                },
                {
                    "type": "SMA",
                    "params": {"timeperiod": 100},
                    "data_field": "atr",
                    "fields_mapping": {"sma": "entry_atr_sma"}
                }
            ],
            "logic_params": {
                "e_support_threshold": 0.005,
                "e_rsi_oversold": 40,
                "e_atr_floor_multiplier": 0.9
            }
        }
    }
"""

from typing import Any, Dict, Optional
import math

from src.strategy.entry.base_entry_mixin import BaseEntryMixin
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class EOMPullbackEntryMixin(BaseEntryMixin):
    """
    Entry mixin for BUY #2: Pullback to Support + EOM Reversal

    Purpose: Trend-following entry after pullback respecting support.

    Indicators required (provided by strategy):
        - entry_support: Support level from SupportResistance indicator
        - entry_eom: EOM value
        - entry_rsi: RSI for oversold check
        - entry_atr: ATR for volatility floor
        - entry_atr_sma: ATR SMA for floor calculation

    Parameters:
        e_support_threshold: Support bounce threshold (default: 0.005 = 0.5%)
        e_rsi_oversold: RSI oversold threshold (default: 40)
        e_atr_floor_multiplier: ATR floor multiplier (default: 0.9)
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
            "e_support_threshold": 0.005,  # 0.5% support bounce threshold
            "e_rsi_oversold": 40,
            "e_atr_floor_multiplier": 0.9,
        }

    def _init_indicators(self):
        """
        Initialize indicators (new architecture only).

        In new architecture, indicators are created by the strategy
        and accessed via get_indicator().
        """
        # New architecture: indicators already created by strategy
        _logger.debug("EOMPullbackEntryMixin: indicators provided by strategy")

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
            open_price = self.strategy.data.open[0]
            low = self.strategy.data.low[0]

            # Get indicator values
            support = self.get_indicator('entry_support')
            eom = self.get_indicator('entry_eom')
            eom_prev = self.get_indicator_prev('entry_eom')
            rsi = self.get_indicator('entry_rsi')
            rsi_prev = self.get_indicator_prev('entry_rsi')
            atr = self.get_indicator('entry_atr')
            atr_sma = self.get_indicator('entry_atr_sma')

            # Get parameters
            support_threshold = self.get_param("e_support_threshold")
            rsi_oversold = self.get_param("e_rsi_oversold")
            atr_floor_multiplier = self.get_param("e_atr_floor_multiplier")

            # Check if support is valid (not NaN)
            if math.isnan(support):
                _logger.debug("No support level found, cannot check pullback")
                return False

            # 1. Price bounces from support
            # Low <= Support * (1 + threshold) AND Close > Open (reversal candle)
            support_level = support * (1 + support_threshold)
            is_at_support = low <= support_level
            is_reversal_candle = close > open_price

            if not (is_at_support and is_reversal_candle):
                return False

            # 2. EOM reversal: EOM crosses above 0
            is_eom_crossing_up = eom > 0 and eom_prev <= 0

            if not is_eom_crossing_up:
                _logger.debug(
                    "EOM not crossing up: eom=%s, eom_prev=%s",
                    eom,
                    eom_prev
                )
                return False

            # 3. RSI oversold → recovery: RSI < oversold AND RSI rising
            is_rsi_oversold = rsi < rsi_oversold
            is_rsi_rising = rsi > rsi_prev

            if not (is_rsi_oversold and is_rsi_rising):
                _logger.debug(
                    "RSI conditions not met: rsi=%s (oversold=%s), rsi_prev=%s",
                    rsi,
                    rsi_oversold,
                    rsi_prev
                )
                return False

            # 4. ATR volatility floor: ATR > ATR_SMA * floor_multiplier
            atr_floor = atr_sma * atr_floor_multiplier
            is_atr_sufficient = atr > atr_floor

            if not is_atr_sufficient:
                _logger.debug(
                    "ATR below floor: atr=%s, floor=%s",
                    atr,
                    atr_floor
                )
                return False

            # All conditions met
            _logger.info(
                "EOM Pullback Entry signal: close=%s, low=%s, support=%s, support_level=%s, "
                "eom=%s, rsi=%s, atr=%s",
                close, low, support, support_level, eom, rsi, atr
            )
            return True

        except KeyError as e:
            _logger.error("Required indicator not found: %s", e)
            return False
        except Exception as e:
            _logger.error("Error in should_enter: %s", e, exc_info=True)
            return False
