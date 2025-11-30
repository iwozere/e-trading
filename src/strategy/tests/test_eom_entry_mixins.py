"""
Unit tests for EOM Entry Mixins

Tests the three EOM-based entry strategies:
- EOMBreakoutEntryMixin
- EOMPullbackEntryMixin
- EOMMAcdBreakoutEntryMixin
"""

import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from unittest.mock import Mock, MagicMock
import math

from src.strategy.entry.eom_breakout_entry_mixin import EOMBreakoutEntryMixin
from src.strategy.entry.eom_pullback_entry_mixin import EOMPullbackEntryMixin
from src.strategy.entry.eom_macd_breakout_entry_mixin import EOMMAcdBreakoutEntryMixin
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class TestEOMBreakoutEntryMixin(unittest.TestCase):
    """Test cases for EOM Breakout Entry Mixin (BUY #1)"""

    def setUp(self):
        """Set up test fixtures"""
        self.mixin = EOMBreakoutEntryMixin()
        self.mixin.strategy = self._create_mock_strategy()
        self.mixin.indicators = {}

    def test_initialization(self):
        """Test mixin initialization"""
        mixin = EOMBreakoutEntryMixin()
        self.assertIsNotNone(mixin)
        self.assertEqual(mixin.get_param("e_breakout_threshold"), 0.002)
        self.assertEqual(mixin.get_param("e_use_atr_filter"), True)
        self.assertEqual(mixin.get_param("e_rsi_overbought"), 70)

    def test_custom_params(self):
        """Test mixin with custom parameters"""
        params = {
            "e_breakout_threshold": 0.005,
            "e_use_atr_filter": False,
            "e_rsi_overbought": 75
        }
        mixin = EOMBreakoutEntryMixin(params=params)
        self.assertEqual(mixin.get_param("e_breakout_threshold"), 0.005)
        self.assertEqual(mixin.get_param("e_use_atr_filter"), False)
        self.assertEqual(mixin.get_param("e_rsi_overbought"), 75)

    def test_breakout_entry_signal(self):
        """Test entry signal on breakout with EOM confirmation"""
        # Setup mock indicators for bullish breakout
        self.mixin.strategy.data.close = [105.0]  # Current close
        self.mixin.strategy.data.volume = [15000000]  # High volume

        self.mixin.get_indicator = Mock(side_effect=lambda x: {
            'entry_resistance': 100.0,  # Resistance at 100
            'entry_eom': 5.0,  # Positive EOM
            'entry_volume_sma': 10000000,  # Volume SMA
            'entry_atr': 3.0,  # ATR
            'entry_atr_sma': 2.5,  # ATR SMA (ATR > ATR_SMA)
            'entry_rsi': 65  # Not overbought
        }[x])

        self.mixin.get_indicator_prev = Mock(side_effect=lambda x, offset=1: {
            'entry_eom': 4.0  # EOM rising
        }[x])

        self.mixin.are_indicators_ready = Mock(return_value=True)

        # Should generate entry signal
        result = self.mixin.should_enter()
        self.assertTrue(result, "Should enter on breakout with EOM confirmation")

    def test_no_breakout_no_signal(self):
        """Test no signal when price hasn't broken resistance"""
        self.mixin.strategy.data.close = [99.0]  # Below resistance
        self.mixin.strategy.data.volume = [15000000]

        self.mixin.get_indicator = Mock(side_effect=lambda x: {
            'entry_resistance': 100.0,
            'entry_eom': 5.0,
            'entry_volume_sma': 10000000,
            'entry_atr': 3.0,
            'entry_atr_sma': 2.5,
            'entry_rsi': 65
        }[x])

        self.mixin.get_indicator_prev = Mock(return_value=4.0)
        self.mixin.are_indicators_ready = Mock(return_value=True)

        result = self.mixin.should_enter()
        self.assertFalse(result, "Should not enter without breakout")

    def test_bearish_eom_no_signal(self):
        """Test no signal when EOM is bearish"""
        self.mixin.strategy.data.close = [105.0]
        self.mixin.strategy.data.volume = [15000000]

        self.mixin.get_indicator = Mock(side_effect=lambda x: {
            'entry_resistance': 100.0,
            'entry_eom': -2.0,  # Negative EOM
            'entry_volume_sma': 10000000,
            'entry_atr': 3.0,
            'entry_atr_sma': 2.5,
            'entry_rsi': 65
        }[x])

        self.mixin.get_indicator_prev = Mock(return_value=-1.0)
        self.mixin.are_indicators_ready = Mock(return_value=True)

        result = self.mixin.should_enter()
        self.assertFalse(result, "Should not enter with bearish EOM")

    def test_overbought_rsi_no_signal(self):
        """Test no signal when RSI is overbought"""
        self.mixin.strategy.data.close = [105.0]
        self.mixin.strategy.data.volume = [15000000]

        self.mixin.get_indicator = Mock(side_effect=lambda x: {
            'entry_resistance': 100.0,
            'entry_eom': 5.0,
            'entry_volume_sma': 10000000,
            'entry_atr': 3.0,
            'entry_atr_sma': 2.5,
            'entry_rsi': 75  # Overbought
        }[x])

        self.mixin.get_indicator_prev = Mock(return_value=4.0)
        self.mixin.are_indicators_ready = Mock(return_value=True)

        result = self.mixin.should_enter()
        self.assertFalse(result, "Should not enter when RSI is overbought")

    def _create_mock_strategy(self):
        """Create a mock strategy object"""
        strategy = Mock()
        strategy.data = Mock()
        strategy.data.close = Mock(__getitem__=lambda self, x: 100.0)
        strategy.data.volume = Mock(__getitem__=lambda self, x: 10000000)
        strategy.indicators = {}
        return strategy


class TestEOMPullbackEntryMixin(unittest.TestCase):
    """Test cases for EOM Pullback Entry Mixin (BUY #2)"""

    def setUp(self):
        """Set up test fixtures"""
        self.mixin = EOMPullbackEntryMixin()
        self.mixin.strategy = self._create_mock_strategy()
        self.mixin.indicators = {}

    def test_initialization(self):
        """Test mixin initialization"""
        mixin = EOMPullbackEntryMixin()
        self.assertIsNotNone(mixin)
        self.assertEqual(mixin.get_param("e_support_threshold"), 0.005)
        self.assertEqual(mixin.get_param("e_rsi_oversold"), 40)
        self.assertEqual(mixin.get_param("e_atr_floor_multiplier"), 0.9)

    def test_pullback_entry_signal(self):
        """Test entry signal on pullback with EOM reversal"""
        # Setup for pullback at support
        self.mixin.strategy.data.close = [95.5]  # Reversal candle
        self.mixin.strategy.data.open = [95.0]  # Close > Open
        self.mixin.strategy.data.low = [94.8]  # Touched support

        self.mixin.get_indicator = Mock(side_effect=lambda x: {
            'entry_support': 95.0,  # Support at 95
            'entry_eom': 0.5,  # EOM crosses above 0
            'entry_rsi': 38,  # Oversold
            'entry_atr': 2.5,
            'entry_atr_sma': 2.0  # ATR sufficient
        }[x])

        self.mixin.get_indicator_prev = Mock(side_effect=lambda x, offset=1: {
            'entry_eom': -0.5,  # Was negative
            'entry_rsi': 35  # RSI rising
        }[x])

        self.mixin.are_indicators_ready = Mock(return_value=True)

        result = self.mixin.should_enter()
        self.assertTrue(result, "Should enter on pullback with EOM reversal")

    def test_no_support_bounce_no_signal(self):
        """Test no signal when price doesn't bounce from support"""
        self.mixin.strategy.data.close = [98.0]  # Too far from support
        self.mixin.strategy.data.open = [97.5]
        self.mixin.strategy.data.low = [97.0]

        self.mixin.get_indicator = Mock(side_effect=lambda x: {
            'entry_support': 95.0,
            'entry_eom': 0.5,
            'entry_rsi': 38,
            'entry_atr': 2.5,
            'entry_atr_sma': 2.0
        }[x])

        self.mixin.get_indicator_prev = Mock(return_value=-0.5)
        self.mixin.are_indicators_ready = Mock(return_value=True)

        result = self.mixin.should_enter()
        self.assertFalse(result, "Should not enter without support bounce")

    def test_eom_not_crossing_up_no_signal(self):
        """Test no signal when EOM doesn't cross above 0"""
        self.mixin.strategy.data.close = [95.5]
        self.mixin.strategy.data.open = [95.0]
        self.mixin.strategy.data.low = [94.8]

        self.mixin.get_indicator = Mock(side_effect=lambda x: {
            'entry_support': 95.0,
            'entry_eom': -1.0,  # Still negative
            'entry_rsi': 38,
            'entry_atr': 2.5,
            'entry_atr_sma': 2.0
        }[x])

        self.mixin.get_indicator_prev = Mock(return_value=-2.0)
        self.mixin.are_indicators_ready = Mock(return_value=True)

        result = self.mixin.should_enter()
        self.assertFalse(result, "Should not enter without EOM crossing up")

    def _create_mock_strategy(self):
        """Create a mock strategy object"""
        strategy = Mock()
        strategy.data = Mock()
        strategy.data.close = Mock(__getitem__=lambda self, x: 100.0)
        strategy.data.open = Mock(__getitem__=lambda self, x: 99.5)
        strategy.data.low = Mock(__getitem__=lambda self, x: 99.0)
        strategy.indicators = {}
        return strategy


class TestEOMMAcdBreakoutEntryMixin(unittest.TestCase):
    """Test cases for EOM MACD Breakout Entry Mixin (BUY #3)"""

    def setUp(self):
        """Set up test fixtures"""
        self.mixin = EOMMAcdBreakoutEntryMixin()
        self.mixin.strategy = self._create_mock_strategy()
        self.mixin.indicators = {}

    def test_initialization(self):
        """Test mixin initialization"""
        mixin = EOMMAcdBreakoutEntryMixin()
        self.assertIsNotNone(mixin)
        self.assertEqual(mixin.get_param("e_resistance_range_low"), 0.995)
        self.assertEqual(mixin.get_param("e_resistance_range_high"), 1.002)
        self.assertEqual(mixin.get_param("e_volume_threshold"), 0.8)

    def test_macd_breakout_entry_signal(self):
        """Test entry signal on MACD bullish with price near resistance"""
        self.mixin.strategy.data.close = [99.8]  # Near resistance
        self.mixin.strategy.data.volume = [12000000]  # Good volume

        self.mixin.get_indicator = Mock(side_effect=lambda x: {
            'entry_resistance': 100.0,  # Resistance at 100
            'entry_macd': 0.5,  # MACD above signal
            'entry_macd_signal': 0.3,
            'entry_macd_hist': 0.2,  # Positive histogram
            'entry_eom': 2.0,  # Positive EOM
            'entry_volume_sma': 10000000
        }[x])

        self.mixin.get_indicator_prev = Mock(side_effect=lambda x, offset=1: {
            'entry_macd_signal': 0.4,  # Was below signal
            'entry_macd_hist': 0.1  # Histogram rising
        }[x])

        self.mixin.are_indicators_ready = Mock(return_value=True)

        result = self.mixin.should_enter()
        self.assertTrue(result, "Should enter on MACD bullish near resistance")

    def test_not_near_resistance_no_signal(self):
        """Test no signal when price not near resistance"""
        self.mixin.strategy.data.close = [95.0]  # Too far from resistance
        self.mixin.strategy.data.volume = [12000000]

        self.mixin.get_indicator = Mock(side_effect=lambda x: {
            'entry_resistance': 100.0,
            'entry_macd': 0.5,
            'entry_macd_signal': 0.3,
            'entry_macd_hist': 0.2,
            'entry_eom': 2.0,
            'entry_volume_sma': 10000000
        }[x])

        self.mixin.get_indicator_prev = Mock(return_value=0.1)
        self.mixin.are_indicators_ready = Mock(return_value=True)

        result = self.mixin.should_enter()
        self.assertFalse(result, "Should not enter when not near resistance")

    def test_negative_eom_no_signal(self):
        """Test no signal when EOM is negative"""
        self.mixin.strategy.data.close = [99.8]
        self.mixin.strategy.data.volume = [12000000]

        self.mixin.get_indicator = Mock(side_effect=lambda x: {
            'entry_resistance': 100.0,
            'entry_macd': 0.5,
            'entry_macd_signal': 0.3,
            'entry_macd_hist': 0.2,
            'entry_eom': -1.0,  # Negative EOM
            'entry_volume_sma': 10000000
        }[x])

        self.mixin.get_indicator_prev = Mock(return_value=0.1)
        self.mixin.are_indicators_ready = Mock(return_value=True)

        result = self.mixin.should_enter()
        self.assertFalse(result, "Should not enter with negative EOM")

    def _create_mock_strategy(self):
        """Create a mock strategy object"""
        strategy = Mock()
        strategy.data = Mock()
        strategy.data.close = Mock(__getitem__=lambda self, x: 100.0)
        strategy.data.volume = Mock(__getitem__=lambda self, x: 10000000)
        strategy.indicators = {}
        return strategy


if __name__ == '__main__':
    unittest.main()
