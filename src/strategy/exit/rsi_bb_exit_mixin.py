"""
RSI and Bollinger Bands Exit Mixin

This module implements an exit strategy based on the combination of:
1. RSI (Relative Strength Index)
2. Bollinger Bands

The strategy exits a position when:
1. RSI is overbought
2. Price touches or crosses above the upper Bollinger Band

Parameters:
    rsi_period (int): Period for RSI calculation (default: 14)
    rsi_overbought (float): Overbought threshold for RSI (default: 70)
    bb_period (int): Period for Bollinger Bands calculation (default: 20)
    bb_stddev (float): Standard deviation multiplier for Bollinger Bands (default: 2.0)
    use_bb_touch (bool): Whether to require price touching the upper band (default: True)
    use_talib (bool): Whether to use TA-Lib for calculations (default: True)

This strategy combines mean reversion (RSI + BB) to identify potential reversal points.
"""

from typing import Any, Dict, Optional

import backtrader as bt
from src.strategy.exit.base_exit_mixin import BaseExitMixin
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


class RSIBBExitMixin(BaseExitMixin):
    """Exit mixin that combines RSI and Bollinger Bands for exit signals."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the mixin with parameters"""
        super().__init__(params)
        self.rsi_name = "exit_rsi"
        self.bb_name = "exit_bb"
        self.rsi = None
        self.bb = None
        self.bb_bot = None
        self.bb_mid = None
        self.bb_top = None

        # Detect architecture mode
        self.use_new_architecture = False  # Will be set in init_exit()

    def get_required_params(self) -> list:
        """There are no required parameters - all have default values"""
        return []

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """Default parameters"""
        return {
            "x_rsi_period": 14,
            "x_rsi_overbought": 70,
            "x_bb_period": 20,
            "x_bb_dev": 2.0,
        }

    def init_exit(self, strategy, additional_params: Optional[Dict[str, Any]] = None):
        """Override to detect architecture mode before calling parent."""
        # Detect architecture: new if strategy has indicators dict with entries
        if hasattr(strategy, 'indicators') and strategy.indicators:
            self.use_new_architecture = True
            logger.debug("Using new TALib-based architecture")
        else:
            self.use_new_architecture = False
            logger.debug("Using legacy architecture")

        # Call parent init_exit which will call _init_indicators
        super().init_exit(strategy, additional_params)

    def _init_indicators(self):
        """Initialize indicators (legacy architecture only).

        In new architecture, indicators are created by the strategy
        and accessed via get_indicator().
        """
        if self.use_new_architecture:
            # New architecture: indicators already created by strategy
            return

        # Legacy architecture: create indicators in mixin
        logger.debug("RSIBBExitMixin._init_indicators called (legacy architecture)")
        if not hasattr(self, "strategy"):
            logger.error("No strategy available in _init_indicators")
            return

        try:
            rsi_period = self.get_param("x_rsi_period")
            bb_period = self.get_param("x_bb_period")
            bb_dev_factor = self.get_param("x_bb_dev")

            if self.strategy.use_talib:
                self.rsi = bt.talib.RSI(self.strategy.data.close, period=rsi_period)
                self.bb = bt.talib.BBANDS(
                    self.strategy.data.close,
                    timeperiod=bb_period,
                    nbdevup=bb_dev_factor,
                    nbdevdn=bb_dev_factor,
                )
                self.bb_top = self.bb.upperband
                self.bb_mid = self.bb.middleband
                self.bb_bot = self.bb.lowerband
            else:
                self.rsi = bt.indicators.RSI(
                    self.strategy.data.close, period=rsi_period
                )
                self.bb = bt.indicators.BollingerBands(
                    self.strategy.data.close, period=bb_period, devfactor=bb_dev_factor
                )
                self.bb_top = self.bb.top
                self.bb_mid = self.bb.mid
                self.bb_bot = self.bb.bot

            self.register_indicator(self.rsi_name, self.rsi)
            self.register_indicator(self.bb_name, self.bb)

            logger.debug("Legacy indicators initialized: exit_rsi, exit_bb")
        except Exception as e:
            logger.exception("Error initializing indicators: ")
            raise

    def are_indicators_ready(self) -> bool:
        """Check if indicators are ready to be used"""
        if self.use_new_architecture:
            # New architecture: check strategy's indicators
            if not hasattr(self.strategy, 'indicators') or not self.strategy.indicators:
                return False

            # Check if required indicators exist
            required_indicators = ['exit_rsi', 'exit_bb_upper']
            for ind_alias in required_indicators:
                if ind_alias not in self.strategy.indicators:
                    return False

            # Check if we can access values
            try:
                _ = self.get_indicator('exit_rsi')
                _ = self.get_indicator('exit_bb_upper')
                return True
            except (IndexError, KeyError, AttributeError):
                return False

        else:
            # Legacy architecture: check mixin's indicators
            return self.rsi_name in self.indicators and self.bb_name in self.indicators

    def should_exit(self) -> bool:
        """Check if we should exit a position.

        Works with both new and legacy architectures.
        """
        if not self.strategy.position:
            return False

        if not self.are_indicators_ready():
            return False

        try:
            # Get indicator values based on architecture
            if self.use_new_architecture:
                # New architecture: access via get_indicator()
                current_rsi = self.get_indicator('exit_rsi')
                previous_rsi = self.get_indicator('exit_rsi', -1)
                current_bb_top = self.get_indicator('exit_bb_upper')
                previous_bb_top = self.get_indicator('exit_bb_upper', -1)

                # Get params from logic_params (new) or fallback to legacy params
                rsi_overbought = self.get_param("rsi_overbought") or self.get_param("x_rsi_overbought", 70)
            else:
                # Legacy architecture: access via mixin's indicators dict
                current_rsi = self.rsi[0]
                previous_rsi = self.rsi[-1]
                current_bb_top = self.bb_top[0]
                previous_bb_top = self.bb_top[-1]

                # Get params from legacy params
                rsi_overbought = self.get_param("x_rsi_overbought", 70)

            # Get current and previous prices
            current_price = self.strategy.data.close[0]
            previous_price = self.strategy.data.close[-1]

            # Check if we have enough data
            if (current_rsi is None or previous_rsi is None or
                current_bb_top is None or previous_bb_top is None or
                current_price is None or previous_price is None):
                return False

            # RSI cross condition: previous RSI above overbought, current RSI below overbought
            rsi_cross_condition = (previous_rsi >= rsi_overbought and
                                 current_rsi < rsi_overbought)

            # BB cross condition: previous price above BB top, current price below BB top
            bb_cross_condition = (previous_price >= previous_bb_top and
                                current_price < current_bb_top)

            # Both conditions must be met for exit
            should_exit = rsi_cross_condition and bb_cross_condition

            if should_exit:
                logger.debug(
                    f"EXIT: RSI cross from {previous_rsi:.2f} to {current_rsi:.2f} (overbought: {rsi_overbought}), "
                    f"Price cross from {previous_price:.2f} to {current_price:.2f} (BB top: {current_bb_top:.2f})"
                )
                self.strategy.current_exit_reason = "rsi_bb_cross_exit"

            return should_exit
        except Exception as e:
            logger.exception("Error in should_exit: ")
            return False

    def get_exit_reason(self) -> str:
        """Get the reason for exit"""
        return getattr(self.strategy, 'current_exit_reason', 'rsi_bb_cross_exit')