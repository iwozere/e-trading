"""
EOM Breakout Entry Mixin

This module implements BUY #1: Breakout + EOM Confirmation (Momentum Breakout Entry)

Entry Signal Logic:
-------------------
Enter after a strong breakout confirmed by EOM, volume, and volatility.

Conditions (all must be true):
1. Breakout: Close > Resistance * (1 + breakout_threshold)
2. EOM bullish: EOM > 0 AND EOM rising (EOM[0] > EOM[-1])
3. Volume confirmation: Volume > Volume_SMA
4. ATR trend filter (optional): ATR > ATR_SMA to avoid low-volatility zones
5. No overbought RSI: RSI < rsi_overbought threshold

Configuration Example:
    {
        "entry_logic": {
            "name": "EOMBreakoutEntryMixin",
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
                    "type": "SMA",
                    "params": {"timeperiod": 20},
                    "data_field": "volume",
                    "fields_mapping": {"sma": "entry_volume_sma"}
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
                },
                {
                    "type": "RSI",
                    "params": {"timeperiod": 14},
                    "fields_mapping": {"rsi": "entry_rsi"}
                }
            ],
            "logic_params": {
                "e_breakout_threshold": 0.002,
                "e_use_atr_filter": true,
                "e_rsi_overbought": 70
            }
        }
    }
"""

from typing import Any, Dict, Optional
import math

from src.strategy.entry.base_entry_mixin import BaseEntryMixin
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class EOMBreakoutEntryMixin(BaseEntryMixin):
    """
    Entry mixin for BUY #1: Breakout + EOM Confirmation

    Purpose: Enter after a strong breakout confirmed by EOM, volume, and volatility.

    Indicators required (provided by strategy):
        - entry_resistance: Resistance level from SupportResistance indicator
        - entry_eom: EOM value
        - entry_volume_sma: Volume SMA for confirmation
        - entry_atr: ATR for volatility filter
        - entry_atr_sma: ATR SMA for trend filter
        - entry_rsi: RSI for overbought check

    Parameters:
        e_breakout_threshold: Breakout threshold above resistance (default: 0.002 = 0.2%)
        e_use_atr_filter: Use ATR trend filter to avoid low-volatility zones (default: True)
        e_rsi_overbought: RSI overbought threshold (default: 70)
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
            "e_breakout_threshold": 0.002,  # 0.2% breakout threshold
            "e_use_atr_filter": True,
            "e_rsi_overbought": 70,
        }

    def _init_indicators(self):
        """
        Initialize indicators (new architecture only).

        In new architecture, indicators are created by the strategy
        and accessed via get_indicator().
        """
        # New architecture: indicators already created by strategy
        _logger.debug("EOMBreakoutEntryMixin: indicators provided by strategy")

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
            eom = self.get_indicator('entry_eom')
            eom_prev = self.get_indicator_prev('entry_eom')
            volume_sma = self.get_indicator('entry_volume_sma')
            rsi = self.get_indicator('entry_rsi')

            # Get parameters
            breakout_threshold = self.get_param("e_breakout_threshold")
            use_atr_filter = self.get_param("e_use_atr_filter")
            rsi_overbought = self.get_param("e_rsi_overbought")

            # Check if resistance is valid (not NaN)
            if math.isnan(resistance):
                _logger.debug("No resistance level found, cannot check breakout")
                return False

            # 1. Breakout: Close > Resistance * (1 + threshold)
            breakout_level = resistance * (1 + breakout_threshold)
            is_breakout = close > breakout_level

            if not is_breakout:
                return False

            # 2. EOM bullish: EOM > 0 AND EOM rising
            is_eom_bullish = eom > 0 and eom > eom_prev

            if not is_eom_bullish:
                _logger.debug(
                    "EOM not bullish: eom=%s, eom_prev=%s",
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

            # 4. ATR trend filter (optional): ATR > ATR_SMA
            if use_atr_filter:
                atr = self.get_indicator('entry_atr')
                atr_sma = self.get_indicator('entry_atr_sma')

                if atr <= atr_sma:
                    _logger.debug(
                        "ATR filter failed: atr=%s, atr_sma=%s",
                        atr,
                        atr_sma
                    )
                    return False

            # 5. No overbought RSI: RSI < rsi_overbought
            is_not_overbought = rsi < rsi_overbought

            if not is_not_overbought:
                _logger.debug(
                    "RSI overbought: rsi=%s, threshold=%s",
                    rsi,
                    rsi_overbought
                )
                return False

            # All conditions met
            _logger.info(
                "EOM Breakout Entry signal: close=%s, resistance=%s, breakout_level=%s, "
                "eom=%s, volume=%s, volume_sma=%s, rsi=%s",
                close, resistance, breakout_level, eom, volume, volume_sma, rsi
            )
            return True

        except KeyError as e:
            _logger.error("Required indicator not found: %s", e)
            return False
        except Exception as e:
            _logger.error("Error in should_enter: %s", e, exc_info=True)
            return False
