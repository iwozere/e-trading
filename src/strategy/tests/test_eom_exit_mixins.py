"""
Unit tests for EOM Exit Mixins

Tests the three EOM-based exit strategies:
- EOMBreakdownExitMixin
- EOMRejectionExitMixin
- EOMMAcdBreakdownExitMixin
"""

import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from unittest.mock import Mock, MagicMock
import math

from src.strategy.exit.eom_breakdown_exit_mixin import EOMBreakdownExitMixin
from src.strategy.exit.eom_rejection_exit_mixin import EOMRejectionExitMixin
from src.strategy.exit.eom_macd_breakdown_exit_mixin import EOMMAcdBreakdownExitMixin
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class TestEOMBreakdownExitMixin(unittest.TestCase):
    """Test cases for EOM Breakdown Exit Mixin (SELL #1)"""

    def setUp(self):
        """Set up test fixtures"""
        self.mixin = EOMBreakdownExitMixin()
        self.mixin.strategy = self._create_mock_strategy()
        self.mixin.indicators = {}

    def test_initialization(self):
        """Test mixin initialization"""
        mixin = EOMBreakdownExitMixin()
        self.assertIsNotNone(mixin)
        self.assertEqual(mixin.get_param("x_breakdown_threshold"), 0.002)

    def test_custom_params(self):
        """Test mixin with custom parameters"""
        params = {"x_breakdown_threshold": 0.005}
        mixin = EOMBreakdownExitMixin(params=params)
        self.assertEqual(mixin.get_param("x_breakdown_threshold"), 0.005)

    def test_breakdown_exit_signal(self):
        """Test exit signal on breakdown with EOM negative"""
        # Setup mock indicators for bearish breakdown
        self.mixin.strategy.data.close = [94.5]  # Below support
        self.mixin.strategy.data.volume = [15000000]  # High volume

        self.mixin.get_indicator = Mock(side_effect=lambda x: {
            'exit_support': 95.0,  # Support at 95
            'exit_eom': -5.0,  # Negative EOM
            'exit_volume_sma': 10000000,
            'exit_atr': 3.0  # ATR rising
        }[x])

        self.mixin.get_indicator_prev = Mock(side_effect=lambda x, offset=1: {
            'exit_eom': -3.0,  # EOM falling
            'exit_atr': 2.5  # ATR was lower
        }[x])

        self.mixin.are_indicators_ready = Mock(return_value=True)

        # Should generate exit signal
        result = self.mixin.should_exit()
        self.assertTrue(result, "Should exit on breakdown with EOM negative")

        # Check exit reason
        reason = self.mixin.get_exit_reason()
        self.assertIn("breakdown_momentum", reason)

    def test_no_breakdown_no_signal(self):
        """Test no signal when price hasn't broken support"""
        self.mixin.strategy.data.close = [96.0]  # Above support
        self.mixin.strategy.data.volume = [15000000]

        self.mixin.get_indicator = Mock(side_effect=lambda x: {
            'exit_support': 95.0,
            'exit_eom': -5.0,
            'exit_volume_sma': 10000000,
            'exit_atr': 3.0
        }[x])

        self.mixin.get_indicator_prev = Mock(return_value=-3.0)
        self.mixin.are_indicators_ready = Mock(return_value=True)

        result = self.mixin.should_exit()
        self.assertFalse(result, "Should not exit without breakdown")

    def test_bullish_eom_no_signal(self):
        """Test no signal when EOM is bullish"""
        self.mixin.strategy.data.close = [94.5]
        self.mixin.strategy.data.volume = [15000000]

        self.mixin.get_indicator = Mock(side_effect=lambda x: {
            'exit_support': 95.0,
            'exit_eom': 2.0,  # Positive EOM
            'exit_volume_sma': 10000000,
            'exit_atr': 3.0
        }[x])

        self.mixin.get_indicator_prev = Mock(return_value=1.0)
        self.mixin.are_indicators_ready = Mock(return_value=True)

        result = self.mixin.should_exit()
        self.assertFalse(result, "Should not exit with bullish EOM")

    def test_low_volume_no_signal(self):
        """Test no signal when volume is low"""
        self.mixin.strategy.data.close = [94.5]
        self.mixin.strategy.data.volume = [5000000]  # Low volume

        self.mixin.get_indicator = Mock(side_effect=lambda x: {
            'exit_support': 95.0,
            'exit_eom': -5.0,
            'exit_volume_sma': 10000000,
            'exit_atr': 3.0
        }[x])

        self.mixin.get_indicator_prev = Mock(return_value=-3.0)
        self.mixin.are_indicators_ready = Mock(return_value=True)

        result = self.mixin.should_exit()
        self.assertFalse(result, "Should not exit without volume confirmation")

    def _create_mock_strategy(self):
        """Create a mock strategy object"""
        strategy = Mock()
        strategy.data = Mock()
        strategy.data.close = Mock(__getitem__=lambda self, x: 100.0)
        strategy.data.volume = Mock(__getitem__=lambda self, x: 10000000)
        strategy.indicators = {}
        return strategy


class TestEOMRejectionExitMixin(unittest.TestCase):
    """Test cases for EOM Rejection Exit Mixin (SELL #2)"""

    def setUp(self):
        """Set up test fixtures"""
        self.mixin = EOMRejectionExitMixin()
        self.mixin.strategy = self._create_mock_strategy()
        self.mixin.indicators = {}

    def test_initialization(self):
        """Test mixin initialization"""
        mixin = EOMRejectionExitMixin()
        self.assertIsNotNone(mixin)
        self.assertEqual(mixin.get_param("x_resistance_threshold"), 0.995)
        self.assertEqual(mixin.get_param("x_rsi_overbought"), 60)

    def test_rejection_exit_signal(self):
        """Test exit signal on resistance rejection with EOM reversal"""
        # Setup for rejection at resistance
        self.mixin.strategy.data.close = [99.5]  # Rejection candle
        self.mixin.strategy.data.open = [100.2]  # Close < Open
        self.mixin.strategy.data.high = [100.5]  # Hit resistance

        self.mixin.get_indicator = Mock(side_effect=lambda x: {
            'exit_resistance': 100.0,  # Resistance at 100
            'exit_eom': -0.5,  # EOM crosses below 0
            'exit_rsi': 65  # Overbought
        }[x])

        self.mixin.get_indicator_prev = Mock(side_effect=lambda x, offset=1: {
            'exit_eom': 0.5,  # Was positive
            'exit_rsi': 68  # RSI falling
        }[x])

        self.mixin.are_indicators_ready = Mock(return_value=True)

        result = self.mixin.should_exit()
        self.assertTrue(result, "Should exit on resistance rejection with EOM reversal")

        # Check exit reason
        reason = self.mixin.get_exit_reason()
        self.assertIn("resistance_rejection", reason)

    def test_no_resistance_touch_no_signal(self):
        """Test no signal when price doesn't touch resistance"""
        self.mixin.strategy.data.close = [98.0]  # Below resistance
        self.mixin.strategy.data.open = [98.5]
        self.mixin.strategy.data.high = [98.8]

        self.mixin.get_indicator = Mock(side_effect=lambda x: {
            'exit_resistance': 100.0,
            'exit_eom': -0.5,
            'exit_rsi': 65
        }[x])

        self.mixin.get_indicator_prev = Mock(return_value=0.5)
        self.mixin.are_indicators_ready = Mock(return_value=True)

        result = self.mixin.should_exit()
        self.assertFalse(result, "Should not exit without touching resistance")

    def test_eom_not_crossing_down_no_signal(self):
        """Test no signal when EOM doesn't cross below 0"""
        self.mixin.strategy.data.close = [99.5]
        self.mixin.strategy.data.open = [100.2]
        self.mixin.strategy.data.high = [100.5]

        self.mixin.get_indicator = Mock(side_effect=lambda x: {
            'exit_resistance': 100.0,
            'exit_eom': 1.0,  # Still positive
            'exit_rsi': 65
        }[x])

        self.mixin.get_indicator_prev = Mock(return_value=2.0)
        self.mixin.are_indicators_ready = Mock(return_value=True)

        result = self.mixin.should_exit()
        self.assertFalse(result, "Should not exit without EOM crossing down")

    def _create_mock_strategy(self):
        """Create a mock strategy object"""
        strategy = Mock()
        strategy.data = Mock()
        strategy.data.close = Mock(__getitem__=lambda self, x: 100.0)
        strategy.data.open = Mock(__getitem__=lambda self, x: 100.5)
        strategy.data.high = Mock(__getitem__=lambda self, x: 101.0)
        strategy.indicators = {}
        return strategy


class TestEOMMAcdBreakdownExitMixin(unittest.TestCase):
    """Test cases for EOM MACD Breakdown Exit Mixin (SELL #3)"""

    def setUp(self):
        """Set up test fixtures"""
        self.mixin = EOMMAcdBreakdownExitMixin()
        self.mixin.strategy = self._create_mock_strategy()
        self.mixin.indicators = {}

    def test_initialization(self):
        """Test mixin initialization"""
        mixin = EOMMAcdBreakdownExitMixin()
        self.assertIsNotNone(mixin)
        self.assertEqual(mixin.get_param("x_support_threshold"), 0.002)

    def test_macd_breakdown_exit_signal(self):
        """Test exit signal on MACD bearish with breakdown"""
        self.mixin.strategy.data.close = [94.5]  # Below support
        self.mixin.strategy.data.volume = [12000000]  # Good volume

        self.mixin.get_indicator = Mock(side_effect=lambda x: {
            'exit_support': 95.0,  # Support at 95
            'exit_macd': -0.5,  # MACD below signal
            'exit_macd_signal': -0.3,
            'exit_macd_hist': -0.2,  # Negative histogram
            'exit_eom': -2.0,  # Negative EOM
            'exit_volume_sma': 10000000
        }[x])

        self.mixin.get_indicator_prev = Mock(side_effect=lambda x, offset=1: {
            'exit_macd': -0.2,  # Was above signal
            'exit_macd_signal': -0.4,
            'exit_macd_hist': -0.1  # Histogram falling
        }[x])

        self.mixin.are_indicators_ready = Mock(return_value=True)

        result = self.mixin.should_exit()
        self.assertTrue(result, "Should exit on MACD bearish with breakdown")

        # Check exit reason
        reason = self.mixin.get_exit_reason()
        self.assertIn("macd_breakdown", reason)

    def test_no_breakdown_no_signal(self):
        """Test no signal when price doesn't break support"""
        self.mixin.strategy.data.close = [96.0]  # Above support
        self.mixin.strategy.data.volume = [12000000]

        self.mixin.get_indicator = Mock(side_effect=lambda x: {
            'exit_support': 95.0,
            'exit_macd': -0.5,
            'exit_macd_signal': -0.3,
            'exit_macd_hist': -0.2,
            'exit_eom': -2.0,
            'exit_volume_sma': 10000000
        }[x])

        self.mixin.get_indicator_prev = Mock(return_value=-0.1)
        self.mixin.are_indicators_ready = Mock(return_value=True)

        result = self.mixin.should_exit()
        self.assertFalse(result, "Should not exit without breakdown")

    def test_positive_eom_no_signal(self):
        """Test no signal when EOM is positive"""
        self.mixin.strategy.data.close = [94.5]
        self.mixin.strategy.data.volume = [12000000]

        self.mixin.get_indicator = Mock(side_effect=lambda x: {
            'exit_support': 95.0,
            'exit_macd': -0.5,
            'exit_macd_signal': -0.3,
            'exit_macd_hist': -0.2,
            'exit_eom': 1.0,  # Positive EOM
            'exit_volume_sma': 10000000
        }[x])

        self.mixin.get_indicator_prev = Mock(return_value=-0.1)
        self.mixin.are_indicators_ready = Mock(return_value=True)

        result = self.mixin.should_exit()
        self.assertFalse(result, "Should not exit with positive EOM")

    def test_macd_not_bearish_no_signal(self):
        """Test no signal when MACD is not bearish"""
        self.mixin.strategy.data.close = [94.5]
        self.mixin.strategy.data.volume = [12000000]

        self.mixin.get_indicator = Mock(side_effect=lambda x: {
            'exit_support': 95.0,
            'exit_macd': 0.5,  # MACD above signal
            'exit_macd_signal': 0.3,
            'exit_macd_hist': 0.2,
            'exit_eom': -2.0,
            'exit_volume_sma': 10000000
        }[x])

        self.mixin.get_indicator_prev = Mock(return_value=0.1)
        self.mixin.are_indicators_ready = Mock(return_value=True)

        result = self.mixin.should_exit()
        self.assertFalse(result, "Should not exit without MACD bearish")

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
