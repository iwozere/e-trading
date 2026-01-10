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
"""

from typing import Any, Dict, Optional
import math

from src.strategy.entry.base_entry_mixin import BaseEntryMixin
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class EOMPullbackEntryMixin(BaseEntryMixin):
    """
    Entry mixin for BUY #2: Pullback to Support + EOM Reversal.

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
            "e_support_threshold": 0.005,
            "e_rsi_oversold": 40,
            "e_atr_floor_multiplier": 0.9,
            "resistance_lookback": 2,
            "eom_period": 14,
            "rsi_period": 14,
            "atr_period": 14,
            "atr_sma_period": 100,
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
            self.get_param("eom_period", 14),
            self.get_param("rsi_period", 14),
            self.get_param("atr_period", 14),
            self.get_param("atr_sma_period", 100),
            self.get_param("resistance_lookback", 2) * 5
        ]
        return max(periods)

    def are_indicators_ready(self) -> bool:
        """Check if indicators are initialized."""
        required = [
            'entry_support', 'entry_eom', 'entry_rsi', 'entry_atr', 'entry_atr_sma'
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
            open_price = self.strategy.data.open[0]
            low = self.strategy.data.low[0]

            # Unified Indicator Access
            support = self.get_indicator('entry_support')
            eom = self.get_indicator('entry_eom')
            eom_prev = self.get_indicator_prev('entry_eom', 1)
            rsi = self.get_indicator('entry_rsi')
            rsi_prev = self.get_indicator_prev('entry_rsi', 1)
            atr = self.get_indicator('entry_atr')
            atr_sma = self.get_indicator('entry_atr_sma')

            # Get parameters
            support_threshold = self.get_param("e_support_threshold") or self.get_param("support_threshold", 0.005)
            rsi_oversold = self.get_param("e_rsi_oversold") or self.get_param("rsi_oversold", 40)
            atr_floor_multiplier = self.get_param("e_atr_floor_multiplier") or self.get_param("atr_floor_multiplier", 0.9)

            # Check if support is valid (not NaN)
            if math.isnan(support):
                return False

            # 1. Price bounces from support
            support_level = support * (1 + support_threshold)
            is_at_support = low <= support_level
            is_reversal_candle = close > open_price

            if not (is_at_support and is_reversal_candle):
                return False

            # 2. EOM reversal: EOM crosses above 0
            is_eom_crossing_up = eom > 0 and eom_prev <= 0

            if not is_eom_crossing_up:
                return False

            # 3. RSI oversold → recovery: RSI < oversold AND RSI rising
            is_rsi_oversold = rsi < rsi_oversold
            is_rsi_rising = rsi > rsi_prev

            if not (is_rsi_oversold and is_rsi_rising):
                return False

            # 4. ATR volatility floor
            atr_floor = atr_sma * atr_floor_multiplier
            is_atr_sufficient = atr > atr_floor

            if not is_atr_sufficient:
                return False

            # All conditions met
            _logger.info(
                f"EOM Pullback Entry signal: close={close:.2f}, low={low:.2f}, support={support:.2f}, "
                f"eom={eom:.4f}, rsi={rsi:.2f}, atr={atr:.2f}"
            )
            return True

        except KeyError as e:
            _logger.warning(f"Required indicator not found: {e}")
            return False
        except Exception:
            _logger.exception("Error in EOMPullbackEntryMixin.should_enter")
            return False
