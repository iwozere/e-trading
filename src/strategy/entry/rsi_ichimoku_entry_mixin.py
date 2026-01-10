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

        # Detect architecture mode
        self.use_new_architecture = False  # Will be set in init_entry()

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

    def init_entry(self, strategy, additional_params: Optional[Dict[str, Any]] = None):
        """Override to detect architecture mode before calling parent."""
        # Detect architecture: new if strategy has indicators dict with entries
        if hasattr(strategy, 'indicators') and strategy.indicators:
            self.use_new_architecture = True
            logger.debug("Using new TALib-based architecture")
        else:
            self.use_new_architecture = False
            logger.debug("Using legacy architecture")

        # Call parent init_entry which will call _init_indicators
        super().init_entry(strategy, additional_params)

    def _init_indicators(self):
        """Initialize indicators (legacy architecture only).

        In new architecture, indicators are created by the strategy
        and accessed via get_indicator().
        """
        if self.use_new_architecture:
            # New architecture: indicators already created by strategy
            return

        # Legacy architecture: create indicators in mixin
        logger.debug("RSIIchimokuEntryMixin._init_indicators called (legacy architecture)")
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

            logger.debug("Legacy indicators initialized: entry_rsi, entry_ichimoku")
        except Exception:
            logger.exception("Error initializing indicators: ")
            raise

    def get_minimum_lookback(self) -> int:
        """
        Returns the minimum number of bars required (Kijun period for Ichimoku cloud).
        """
        # Get kijun period from params
        return self.get_param("kijun") or self.get_param("e_kijun", 26)

    def are_indicators_ready(self) -> bool:
        """
        Check if indicators are initialized.
        History/Lookback is now handled by BaseStrategy.
        """
        if self.use_new_architecture:
            # Check if required indicators exist in strategy
            required = ['entry_rsi', 'entry_ichimoku_tenkan', 'entry_ichimoku_senkou_a', 'entry_ichimoku_senkou_b']
            return all(alias in getattr(self.strategy, 'indicators', {}) for alias in required)
        else:
            # Legacy: check mixin's indicators dict
            return self.rsi_name in self.indicators and self.ichimoku_name in self.indicators

    def should_enter(self) -> bool:
        """Check if we should enter a position."""
        if not self.are_indicators_ready():
            return False

        try:
            current_price = self.strategy.data.close[0]

            # Standardized parameter retrieval (supports both config styles)
            rsi_oversold = self.get_param("rsi_oversold") or self.get_param("e_rsi_oversold", 30)
            kijun_period = self.get_param("kijun") or self.get_param("e_kijun", 26)

            # Centralized indicator access via get_indicator/get_indicator_prev
            # This works for both legacy (mixin-registered) and new (factory-created) indicators
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
