"""
RSI and Ichimoku Cloud Entry Mixin

This module implements an entry strategy based on the combination of:
1. RSI (Relative Strength Index)
2. Ichimoku Cloud

The strategy enters a position when:
1. RSI is oversold
2. Price is below the Ichimoku Cloud
3. Tenkan-sen (Conversion Line) crosses above Kijun-sen (Base Line)

Parameters:
    rsi_period (int): Period for RSI calculation (default: 14)
    rsi_oversold (float): Oversold threshold for RSI (default: 30)
    tenkan_period (int): Period for Tenkan-sen calculation (default: 9)
    kijun_period (int): Period for Kijun-sen calculation (default: 26)
    senkou_span_b_period (int): Period for Senkou Span B calculation (default: 52)

This strategy combines mean reversion (RSI) with trend following (Ichimoku) to identify potential reversal points.
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
        self.rsi_name = "entry_rsi"
        self.ichimoku_name = "entry_ichimoku"
        self.rsi = None
        self.ichimoku = None

    def get_required_params(self) -> list:
        """There are no required parameters - all have default values"""
        return []

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """Default parameters"""
        return {
            "e_rsi_period": 14,
            "e_rsi_oversold": 30,
            "e_tenkan": 9,
            "e_kijun": 26,
            "e_senkou": 52,
            "e_senkou_lead": 26,
            "e_chikou": 26,
        }

    def _init_indicators(self):
        """Initialize indicators"""
        logger.debug("RSIIchimokuEntryMixin._init_indicators called")
        if not hasattr(self, "strategy"):
            logger.error("No strategy available in _init_indicators")
            return

        try:
            rsi_period = self.get_param("e_rsi_period")
            if self.strategy.use_talib:
                self.rsi = bt.talib.RSI(self.strategy.data.close, timeperiod=rsi_period)
            else:
                self.rsi = bt.indicators.RSI(
                    self.strategy.data.close, period=rsi_period
                )

            self.register_indicator(self.rsi_name, self.rsi)

            self.ichimoku = bt.indicators.Ichimoku(
                self.strategy.data,
                tenkan=self.get_param("e_tenkan"),
                kijun=self.get_param("e_kijun"),
                senkou=self.get_param("e_senkou"),
                senkou_lead=self.get_param("e_senkou_lead"),
                chikou=self.get_param("e_chikou"),
            )
            self.register_indicator(self.ichimoku_name, self.ichimoku)

            self.tenkan_sen = self.ichimoku.tenkan_sen
            self.kijun_sen = self.ichimoku.kijun_sen
            self.senkou_span_a = self.ichimoku.senkou_span_a
            self.senkou_span_b = self.ichimoku.senkou_span_b

            self.cross_over_tenkan = bt.indicators.CrossOver(self.strategy.data.close, self.tenkan_sen)
            self.cross_below_kijun = bt.indicators.CrossDown(self.strategy.data.close, self.kijun_sen)

        except Exception as e:
            logger.exception("Error initializing indicators: ")
            raise

    def should_enter(self) -> bool:
        """Check if we should enter a position"""
        if (
            self.rsi_name not in self.indicators
            or self.ichimoku_name not in self.indicators
        ):
            return False

        try:
            # Get indicators from mixin's indicators dictionary
            rsi = self.indicators[self.rsi_name]
            ichimoku = self.indicators[self.ichimoku_name]
            current_price = self.strategy.data.close[0]

            # NEW try: 27.06.2025
            """
            Entry signal:
            Price is above the Kumo cloud (bullish signal from Ichimoku)
            RSI is below 30 (oversold condition, possible rebound)
            Price crosses above Tenkan-sen (momentum turning up)
            Exit signal:
            Price crosses below Kijun-sen (weakness)
            Or RSI is above 70 (overbought, potential reversal)
            """

            span_a = self.senkou_span_a[-self.get_param("e_kijun")]  # shift back to current candle
            span_b = self.senkou_span_b[-self.get_param("e_kijun")]

            kumo_top = max(span_a, span_b)

            # Price must be above the cloud
            # RSI oversold and bullish price crossover above Tenkan-sen
            return_value = current_price > kumo_top and self.rsi[0] <= self.get_param("e_rsi_oversold") and self.cross_over_tenkan[0] > 0
            if return_value:
                logger.debug(
                    f"ENTRY: Price: {current_price}, RSI: {rsi[0]}, "
                    f"Tenkan-sen: {ichimoku.tenkan_sen[0]}, Kijun-sen: {ichimoku.kijun_sen[0]}, "
                    f"span_a: {span_a}, span_b: {span_b}"
                )

            # Exit condition: cross below Kijun-sen, RSI overbought, inside Kumo cloud (optional)
            #kumo_bottom = min(span_a, span_b)
            #return_value = self.cross_below_kijun[0] or self.rsi[0] > self.p.rsi_overbought or (kumo_bottom <= current_price <= kumo_top)

            return return_value
        except Exception as e:
            logger.exception("Error in should_enter: ")
            return False
