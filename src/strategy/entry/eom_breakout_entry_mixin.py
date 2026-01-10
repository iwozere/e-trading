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
"""

from typing import Any, Dict, Optional
import math

from src.strategy.entry.base_entry_mixin import BaseEntryMixin
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class EOMBreakoutEntryMixin(BaseEntryMixin):
    """
    Entry mixin for BUY #1: Breakout + EOM Confirmation.

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
            "e_breakout_threshold": 0.002,
            "e_use_atr_filter": True,
            "e_rsi_overbought": 70,
            "resistance_lookback": 2,
            "eom_period": 14,
            "volume_sma_period": 20,
            "atr_period": 14,
            "atr_sma_period": 100,
            "rsi_period": 14,
        }

    def init_entry(self, strategy, additional_params: Optional[Dict[str, Any]] = None):
        """Standardize architecture detection."""
        if hasattr(strategy, 'indicators') and strategy.indicators:
            self.use_new_architecture = True
        else:
            self.use_new_architecture = False
        super().init_entry(strategy, additional_params)

    def _init_indicators(self):
        """In legacy mode, this would initialize indicators. For now, we assume strategy provides them."""
        # If we wanted to support full legacy initialization (where mixin creates its own indicators),
        # we would do it here. For EOM mixins, they currently rely on the new architecture.
        pass

    def get_minimum_lookback(self) -> int:
        """Returns the minimum number of bars required."""
        periods = [
            self.get_param("eom_period", 14),
            self.get_param("volume_sma_period", 20),
            self.get_param("atr_period", 14),
            self.get_param("atr_sma_period", 100),
            self.get_param("rsi_period", 14),
            self.get_param("resistance_lookback", 2) * 5 # S/R needs some data
        ]
        return max(periods)

    def are_indicators_ready(self) -> bool:
        """Check if indicators are initialized."""
        required = [
            'entry_resistance', 'entry_eom', 'entry_volume_sma', 'entry_rsi'
        ]
        if self.get_param("e_use_atr_filter", True):
            required.extend(['entry_atr', 'entry_atr_sma'])

        if hasattr(self.strategy, 'indicators') and self.strategy.indicators:
            return all(alias in self.strategy.indicators for alias in required)

        # If legacy indicators were registered in self.indicators, check them here.
        # But EOM mixins currently mostly use the new architecture.
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

            # Standardized Indicator Access
            resistance = self.get_indicator('entry_resistance')
            eom = self.get_indicator('entry_eom')
            eom_prev = self.get_indicator_prev('entry_eom', 1)
            volume_sma = self.get_indicator('entry_volume_sma')
            rsi = self.get_indicator('entry_rsi')

            # Get parameters
            breakout_threshold = self.get_param("e_breakout_threshold") or self.get_param("breakout_threshold", 0.002)
            use_atr_filter = self.get_param("e_use_atr_filter") or self.get_param("use_atr_filter", True)
            rsi_overbought = self.get_param("e_rsi_overbought") or self.get_param("rsi_overbought", 70)

            # Check if resistance is valid (not NaN)
            if math.isnan(resistance):
                return False

            # 1. Breakout: Close > Resistance * (1 + threshold)
            breakout_level = resistance * (1 + breakout_threshold)
            is_breakout = close > breakout_level

            if not is_breakout:
                return False

            # 2. EOM bullish: EOM > 0 AND EOM rising
            is_eom_bullish = eom > 0 and eom > eom_prev

            if not is_eom_bullish:
                return False

            # 3. Volume confirmation: Volume > Volume_SMA
            is_volume_confirmed = volume > volume_sma

            if not is_volume_confirmed:
                return False

            # 4. ATR trend filter (optional): ATR > ATR_SMA
            if use_atr_filter:
                atr = self.get_indicator('entry_atr')
                atr_sma = self.get_indicator('entry_atr_sma')

                if atr <= atr_sma:
                    return False

            # 5. No overbought RSI: RSI < rsi_overbought
            is_not_overbought = rsi < rsi_overbought

            if not is_not_overbought:
                return False

            # All conditions met
            _logger.info(
                f"EOM Breakout Entry signal: close={close:.2f}, resistance={resistance:.2f}, "
                f"eom={eom:.4f}, volume={volume:.0f}, volume_sma={volume_sma:.0f}, rsi={rsi:.2f}"
            )
            return True

        except KeyError as e:
            _logger.warning(f"Required indicator not found: {e}")
            return False
        except Exception:
            _logger.exception("Error in EOMBreakoutEntryMixin.should_enter")
            return False
