#!/usr/bin/env python3
"""
Test module for backtrader integration with BaseBroker.

This module tests the conditional inheritance mechanism and backtrader compatibility
of the BaseBroker class.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.trading.broker.base_broker import (
    BaseBroker,
    BACKTRADER_AVAILABLE,
    check_backtrader_availability,
    require_backtrader,
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
    TradingMode
)


class TestConditionalInheritance(unittest.TestCase):
    """Test conditional inheritance mechanism."""

    def test_backtrader_availability_check(self):
        """Test backtrader availability check function."""
        # This should return the actual availability status
        availability = check_backtrader_availability()
        self.assertIsInstance(availability, bool)
        self.assertEqual(availability, BACKTRADER_AVAILABLE)

    def test_require_backtrader_when_available(self):
        """Test require_backtrader when backtrader is available."""
        if BACKTRADER_AVAILABLE:
            # Should not raise an exception
            try:
                require_backtrader("test feature")
            except ImportError:
                self.fail("require_backtrader raised ImportError when backtrader is available")
        else:
            # Should raise ImportError
            with self.assertRaises(ImportError) as context:
                require_backtrader("test feature")
            self.assertIn("backtrader", str(context.exception).lower())

    def test_enhanced_broker_inheritance_with_backtrader(self):
        """Test BaseBroker inheritance when backtrader is available."""
        config = {
            'name': 'test_broker',
            'trading_mode': 'paper'
        }

        broker = BaseBroker(config)

        if BACKTRADER_AVAILABLE:
            # Should inherit from bt.broker.BrokerBase
            import backtrader as bt
            self.assertTrue(isinstance(broker, bt.broker.BrokerBase))
            self.assertTrue(broker.is_backtrader_mode())
        else:
            # Should inherit from ABC
            from abc import ABC
            self.assertTrue(isinstance(broker, ABC))
            self.assertFalse(broker.is_backtrader_mode())

    def test_enhanced_broker_initialization(self):
        """Test BaseBroker initialization in both modes."""
        config = {
            'name': 'test_broker',
            'trading_mode': 'paper',
            'paper_trading_config': {
                'initial_balance': 5000.0,
                'commission_rate': 0.002
            }
        }

        broker = BaseBroker(config)

        # Test common properties
        self.assertEqual(broker.get_name(), 'test_broker')
        self.assertEqual(broker.get_trading_mode(), TradingMode.PAPER)
        self.assertTrue(broker.is_paper_trading())

        # Test backtrader mode detection
        self.assertEqual(broker.is_backtrader_mode(), BACKTRADER_AVAILABLE)

        # Test paper trading config
        self.assertEqual(broker.paper_trading_config.initial_balance, 5000.0)
        self.assertEqual(broker.paper_trading_config.commission_rate, 0.002)


class TestBacktraderInterfaceCompatibility(unittest.TestCase):
    """Test backtrader interface compatibility."""

    def setUp(self):
        """Set up test broker."""
        self.config = {
            'name': 'test_broker',
            'trading_mode': 'paper',
            'paper_trading_config': {
                'initial_balance': 10000.0
            }
        }
        self.broker = BaseBroker(self.config)

    @unittest.skipUnless(BACKTRADER_AVAILABLE, "Backtrader not available")
    def test_backtrader_buy_method(self):
        """Test backtrader buy method."""
        # Mock data feed
        mock_data = Mock()
        mock_data._name = 'AAPL'

        # Test market buy order
        order = self.broker.buy(
            owner=None,
            data=mock_data,
            size=100,
            exectype=None  # Market order
        )

        self.assertIsInstance(order, Order)
        self.assertEqual(order.symbol, 'AAPL')
        self.assertEqual(order.side, OrderSide.BUY)
        self.assertEqual(order.quantity, 100.0)
        self.assertEqual(order.order_type, OrderType.MARKET)

    @unittest.skipUnless(BACKTRADER_AVAILABLE, "Backtrader not available")
    def test_backtrader_sell_method(self):
        """Test backtrader sell method."""
        # Mock data feed
        mock_data = Mock()
        mock_data._name = 'AAPL'

        # Test market sell order
        order = self.broker.sell(
            owner=None,
            data=mock_data,
            size=50,
            exectype=None  # Market order
        )

        self.assertIsInstance(order, Order)
        self.assertEqual(order.symbol, 'AAPL')
        self.assertEqual(order.side, OrderSide.SELL)
        self.assertEqual(order.quantity, 50.0)
        self.assertEqual(order.order_type, OrderType.MARKET)

    @unittest.skipUnless(BACKTRADER_AVAILABLE, "Backtrader not available")
    def test_backtrader_limit_order(self):
        """Test backtrader limit order."""
        import backtrader as bt

        # Mock data feed
        mock_data = Mock()
        mock_data._name = 'AAPL'

        # Test limit buy order
        order = self.broker.buy(
            owner=None,
            data=mock_data,
            size=100,
            price=150.0,
            exectype=bt.Order.Limit
        )

        self.assertEqual(order.order_type, OrderType.LIMIT)
        self.assertEqual(order.price, 150.0)

    def test_backtrader_methods_without_backtrader_mode(self):
        """Test that backtrader methods raise errors when not in backtrader mode."""
        # Force non-backtrader mode
        self.broker._backtrader_mode = False

        with self.assertRaises(RuntimeError) as context:
            self.broker.buy(size=100)
        self.assertIn("backtrader mode", str(context.exception))

        with self.assertRaises(RuntimeError) as context:
            self.broker.sell(size=100)
        self.assertIn("backtrader mode", str(context.exception))

        with self.assertRaises(RuntimeError) as context:
            self.broker.cancel(Mock())
        self.assertIn("backtrader mode", str(context.exception))

    def test_parameter_validation(self):
        """Test parameter validation for backtrader methods."""
        if not BACKTRADER_AVAILABLE:
            self.skipTest("Backtrader not available")

        # Mock data feed
        mock_data = Mock()
        mock_data._name = 'AAPL'

        # Test invalid size
        order = self.broker.buy(
            data=mock_data,
            size=0  # Invalid size
        )
        self.assertEqual(order.status, OrderStatus.REJECTED)
        self.assertIn('error', order.metadata)

        # Test negative size
        order = self.broker.buy(
            data=mock_data,
            size=-100  # Invalid size
        )
        self.assertEqual(order.status, OrderStatus.REJECTED)


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with existing broker functionality."""

    def setUp(self):
        """Set up test broker."""
        self.config = {
            'name': 'test_broker',
            'trading_mode': 'paper'
        }
        self.broker = BaseBroker(self.config)

    def test_existing_broker_methods(self):
        """Test that existing broker methods still work."""
        # Test basic properties
        self.assertEqual(self.broker.get_name(), 'test_broker')
        self.assertTrue(self.broker.is_paper_trading())
        self.assertEqual(self.broker.get_trading_mode(), TradingMode.PAPER)

        # Test paper trading config access
        config = self.broker.get_paper_trading_config()
        self.assertIsNotNone(config)

        # Test status method
        status = self.broker.get_status()
        self.assertIsInstance(status, dict)
        self.assertIn('broker_name', status)
        self.assertIn('trading_mode', status)

    def test_paper_trading_functionality(self):
        """Test that paper trading functionality is preserved."""
        # Test paper portfolio initialization
        if self.broker.paper_trading_enabled:
            self.assertIsNotNone(self.broker.paper_portfolio)
            self.assertEqual(self.broker.paper_portfolio.initial_balance,
                           self.broker.paper_trading_config.initial_balance)

    def test_notification_manager(self):
        """Test that notification manager is properly initialized."""
        self.assertIsNotNone(self.broker.notification_manager)

    def test_execution_metrics_tracking(self):
        """Test that execution metrics tracking is preserved."""
        self.assertIsInstance(self.broker.execution_metrics, list)
        self.assertEqual(self.broker.total_executions, 0)
        self.assertIsInstance(self.broker.execution_quality_stats, dict)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)