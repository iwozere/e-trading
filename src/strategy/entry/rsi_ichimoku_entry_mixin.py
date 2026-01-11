"""
RSI and Ichimoku Cloud Entry Mixin

This module implements an entry strategy based on the combination of:
1. RSI (Relative Strength Index)
2. Ichimoku Cloud

The strategy enters a position when:
1. RSI is oversold
2. Price is above the Ichimoku Cloud (Bullish trend)
3. Price crosses above the Tenkan-sen (Conversion Line)

Configuration Example (New Unified Architecture):
    {
        "entry_logic": {
            "name": "RSIIchimokuEntryMixin",
            "indicators": [
                {
                    "type": "RSI",
                    "params": {"timeperiod": 14},
                    "fields_mapping": {"rsi": "entry_rsi"}
                },
                {
                    "type": "ICHIMOKU",
                    "params": {"tenkan": 9, "kijun": 26, "senkou": 52},
                    "fields_mapping": {
                        "tenkan_sen": "entry_ichimoku_tenkan",
                        "senkou_span_a": "entry_ichimoku_senkou_a",
                        "senkou_span_b": "entry_ichimoku_senkou_b"
                    }
                }
            ],
            "logic_params": {
                "rsi_oversold": 30
            }
        }
    }

Note: This mixin no longer supports legacy internal indicator initialization.
"""

from typing import Any, Dict, Optional

import backtrader as bt
from src.strategy.entry.base_entry_mixin import BaseEntryMixin
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


class RSIIchimokuEntryMixin(BaseEntryMixin):
    """Entry mixin based on RSI and Ichimoku Cloud"""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)

    def get_required_params(self) -> list:
        """There are no required parameters - all have default values"""
        return []

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """Default parameters"""
        return {
            "rsi_oversold": 30
        }

    def _init_indicators(self):
        """
        No-op for new architecture. Indicators are created by the strategy
        via the IndicatorFactory and defined in the JSON configuration.
        """
        pass

    def get_minimum_lookback(self) -> int:
        """
        Returns the minimum number of bars required (Kijun period for Ichimoku cloud).
        """
        # Get kijun period from params
        return self.get_param("kijun") or self.get_param("e_kijun", 26)

    def are_indicators_ready(self) -> bool:
        """
        Check if required indicators exist in the strategy registry.
        """
        required = [
            'entry_rsi',
            'entry_ichimoku_tenkan',
            'entry_ichimoku_senkou_a',
            'entry_ichimoku_senkou_b'
        ]
        return all(alias in getattr(self.strategy, 'indicators', {}) for alias in required)

    def should_enter(self) -> bool:
        """Check if we should enter a position."""
        if not self.are_indicators_ready():
            return False

        try:
            current_price = self.strategy.data.close[0]

            # Standardized parameter retrieval
            rsi_oversold = self.get_param("rsi_oversold", 30)
            kijun_period = 26  # Standard lookback for Ichimoku Span A/B

            # Access indicators via the unified registry
            current_rsi = self.get_indicator('entry_rsi')

            span_a = self.get_indicator_prev('entry_ichimoku_senkou_a', kijun_period)
            span_b = self.get_indicator_prev('entry_ichimoku_senkou_b', kijun_period)

            tenkan = self.get_indicator('entry_ichimoku_tenkan')
            prev_tenkan = self.get_indicator_prev('entry_ichimoku_tenkan', 1)
            prev_price = self.strategy.data.close[-1]

            # Cross-over logic
            cross_over_tenkan = (prev_price <= prev_tenkan and current_price > tenkan)
            kumo_top = max(span_a, span_b)

            # Strategy conditions
            entry_signal = (
                current_price > kumo_top and
                current_rsi <= rsi_oversold and
                cross_over_tenkan
            )

            if entry_signal:
                logger.debug(
                    f"ENTRY: Price: {current_price}, RSI: {current_rsi}, "
                    f"span_a: {span_a}, span_b: {span_b}, kumo_top: {kumo_top}"
                )

            return entry_signal

        except Exception:
            logger.exception("Error in should_enter: ")
            return False
