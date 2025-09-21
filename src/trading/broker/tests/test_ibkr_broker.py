#!/usr/bin/env python3
"""
Test Suite for Enhanced IBKR Broker
-----------------------------------

Comprehensive test suite for the enhanced IBKR broker implementation,
covering both paper trading and live trading modes, multi-asset support,
order validation, and notification systems.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timezone

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.trading.broker.ibkr_broker import IBKRBroker
from src.trading.broker.base_broker import (
    Order, OrderType, OrderSide, OrderStatus, TradingMode
)
from src.trading.broker.ibkr_utils import (
    IBKRContractManager, IBKRCommissionCalculator,
    create_ibkr_config_template
)


class TestIBKRBroker:
    """Test cases for IBKRBroker class."""

    @pytest.fixture
    def paper_config(self):
        """Paper trading configuration fixture."""
        return {
            'type': 'ibkr',
            'trading_mode': 'paper',
            'cash': 25000.0,
            'connection': {
                'host': '127.0.0.1',
                'port': 7497,
                'client_id': 1
            },
            'paper_trading_config': {
                'mode': 'realistic',
                'initial_balance': 25000.0,
                'commission_rate': 0.0005,
                'slippage_model': 'sqrt',
                'base_slippage': 0.0003,
                'latency_simulation': True,
                'min_latency_ms': 5,
                'max_latency_ms': 50
            },
            'notifications': {
                'position_opened': True,
                'position_closed': True,
                'email_enabled': False,  # Disable for tests
                'telegram_enabled': False
            }
        }

    @pytest.fixture
    def live_config(self):
        """Live trading configuration fixture."""
        return {
            'type': 'ibkr',
            'trading_mode': 'live',
            'cash': 50000.0,
            'connection': {
                'host': '127.0.0.1',
                'port': 4001,
                'client_id': 1
            },
            'live_trading_confirmed': True,
            'risk_management': {
                'max_position_size': 5000.0,
                'max_daily_loss': 2000.0
            },
            'notifications': {
                'position_opened': True,
                'position_closed': True,
                'email_enabled': False,  # Disable for tests
                'telegram_enabled': False
            }
        }

    @pytest.fixture
    def mock_ib_client(self):
        """Mock IB client fixture."""
        mock_ib = Mock()
        mock_ib.connect.return_value = None
        mock_ib.isConnected.return_value = True
        mock_ib.disconnect.return_value = None
        mock_ib.accountSummary.return_value = []
        mock_ib.portfolio.return_value = []
        mock_ib.positions.return_value = []
        mock_ib.reqMktData.return_value = Mock()
        mock_ib.cancelMktData.return_value = None
        return mock_ib

    @pytest.fixture
    def paper_broker(self, paper_config, mock_ib_client):
        """Paper trading broker fixture."""
        with patch('src.trading.broker.ibkr_broker.IB', return_value=mock_ib_client):
            broker = IBKRBroker(
                host='127.0.0.1',
                port=7497,
                client_id=1,
                cash=25000.0,
                config=paper_config
            )
            broker._load_account_info = AsyncMock()
            broker._start_market_data_updates = AsyncMock()
            return broker

    @pytest.fixture
    def live_broker(self, live_config, mock_ib_client):
        """Live trading broker fixture."""
        with patch('src.trading.broker.ibkr_broker.IB', return_value=mock_ib_client):
            broker = IBKRBroker(
                host='127.0.0.1',
                port=4001,
                client_id=1,
                cash=50000.0,
                config=live_config
            )
            broker._load_account_info = AsyncMock()
            return broker

    def test_broker_initialization_paper_mode(self, paper_broker):
        """Test broker initialization in paper mode."""
        assert paper_broker.trading_mode == TradingMode.PAPER
        assert paper_broker.paper_trading_enabled is True
        assert paper_broker.port == 7497  # Paper trading port
        assert paper_broker.paper_portfolio is not None
        assert paper_broker.paper_portfolio.initial_balance == 25000.0

    def test_broker_initialization_live_mode(self, live_broker):
        """Test broker initialization in live mode."""
        assert live_broker.trading_mode == TradingMode.LIVE
        assert live_broker.paper_trading_enabled is False
        assert live_broker.port == 4001  # Live trading port

    def test_automatic_port_selection(self):
        """Test automatic port selection based on trading mode."""
        # Paper mode should default to port 7497
        paper_config = {'trading_mode': 'paper', 'type': 'ibkr'}
        with patch('src.trading.broker.ibkr_broker.IB'):
            paper_broker = IBKRBroker(config=paper_config)
            assert paper_broker.port == 7497

        # Live mode should default to port 4001
        live_config = {'trading_mode': 'live', 'type': 'ibkr', 'live_trading_confirmed': True}
        with patch('src.trading.broker.ibkr_broker.IB'):
            live_broker = IBKRBroker(config=live_config)
            assert live_broker.port == 4001

    @pytest.mark.asyncio
    async def test_connection_paper_mode(self, paper_broker):
        """Test connection in paper mode."""
        result = await paper_broker.connect()
        assert result is True
        assert paper_broker.is_connected is True
        paper_broker.ib.connect.assert_called_once_with('127.0.0.1', 7497, clientId=1)

    @pytest.mark.asyncio
    async def test_connection_live_mode(self, live_broker):
        """Test connection in live mode."""
        result = await live_broker.connect()
        assert result is True
        assert live_broker.is_connected is True
        live_broker.ib.connect.assert_called_once_with('127.0.0.1', 4001, clientId=1)

    def test_contract_creation(self, paper_broker):
        """Test IBKR contract creation."""
        # Test stock contract
        stock_contract = paper_broker._create_contract('AAPL', 'STK')
        assert stock_contract.symbol == 'AAPL'
        assert stock_contract.secType == 'STK'

        # Test forex contract
        forex_contract = paper_broker._create_contract('EURUSD', 'CASH')
        assert forex_contract.symbol == 'EURUSD'
        assert forex_contract.secType == 'CASH'

    @pytest.mark.asyncio
    async def test_paper_order_placement_stock(self, paper_broker):
        """Test paper trading stock order placement."""
        # Mock market price
        paper_broker.update_market_data_cache('AAPL', 150.0)

        order = Order(
            symbol='AAPL',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )

        order_id = await paper_broker.place_order(order)

        assert order_id == order.order_id
        assert order.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]

    @pytest.mark.asyncio
    async def test_paper_order_placement_limit(self, paper_broker):
        """Test paper trading limit order placement."""
        # Mock market price
        paper_broker.update_market_data_cache('AAPL', 150.0)

        order = Order(
            symbol='AAPL',
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=149.0  # Below market price
        )

        order_id = await paper_broker.place_order(order)

        assert order_id == order.order_id
        assert order.status == OrderStatus.PENDING  # Should remain pending

    @pytest.mark.asyncio
    async def test_live_order_placement(self, live_broker):
        """Test live order placement."""
        order = Order(
            symbol='AAPL',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )

        # Mock IBKR order creation and placement
        mock_ibkr_order = Mock()
        mock_ibkr_order.orderId = 12345
        mock_trade = Mock()
        mock_trade.order = mock_ibkr_order

        with patch.object(live_broker, '_create_ibkr_order', return_value=mock_ibkr_order):
            with patch.object(live_broker.ib, 'placeOrder', return_value=mock_trade):
                order_id = await live_broker.place_order(order)

                assert order_id == order.order_id
                assert order.status == OrderStatus.PENDING
                assert order.metadata['ibkr_order_id'] == 12345

    @pytest.mark.asyncio
    async def test_order_validation_ibkr_rules(self, paper_broker):
        """Test order validation against IBKR-specific rules."""
        # Test minimum quantity validation
        order = Order(
            symbol='AAPL',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.5  # Below minimum for stocks
        )

        is_valid, message = await paper_broker._validate_ibkr_order(order)
        assert is_valid is False
        assert 'minimum' in message.lower()

    @pytest.mark.asyncio
    async def test_multi_asset_support(self, paper_broker):
        """Test multi-asset trading support."""
        # Test different asset classes
        assets = [
            ('AAPL', 'STK'),
            ('EURUSD', 'CASH'),
        ]

        for symbol, asset_class in assets:
            contract = paper_broker._create_contract(symbol, asset_class)
            assert contract is not None
            assert contract.symbol == symbol

    @pytest.mark.asyncio
    async def test_market_data_subscription(self, paper_broker):
        """Test market data subscription."""
        # Mock ticker
        mock_ticker = Mock()
        mock_ticker.last = 150.0
        paper_broker.ib.reqMktData.return_value = mock_ticker

        result = paper_broker._subscribe_market_data('AAPL')
        assert result is True
        assert 'AAPL' in paper_broker.market_data_subscriptions
        assert 'AAPL' in paper_broker.contracts

    @pytest.mark.asyncio
    async def test_paper_portfolio_tracking(self, paper_broker):
        """Test paper trading portfolio tracking."""
        # Place a buy order
        paper_broker.update_market_data_cache('AAPL', 150.0)

        buy_order = Order(
            symbol='AAPL',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )

        await paper_broker.place_order(buy_order)

        # Check portfolio
        portfolio = await paper_broker.get_paper_portfolio()
        assert portfolio.cash < 25000.0  # Cash should decrease

        positions = await paper_broker.get_paper_positions()
        assert 'AAPL' in positions
        assert positions['AAPL'].quantity == 100

    @pytest.mark.asyncio
    async def test_pending_order_processing(self, paper_broker):
        """Test processing of pending orders against market data."""
        # Place a limit order below market
        order = Order(
            symbol='AAPL',
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=149.0
        )

        paper_broker.update_market_data_cache('AAPL', 150.0)
        await paper_broker.place_order(order)

        # Simulate price drop to trigger order
        market_data = {'AAPL': 148.0}
        await paper_broker.process_pending_paper_orders(market_data)

        # Order should now be filled
        updated_order = paper_broker.get_paper_order_status(order.order_id)
        assert updated_order.status == OrderStatus.FILLED

    def test_supported_order_types(self, paper_broker):
        """Test supported order types."""
        supported_types = paper_broker.get_supported_order_types()

        assert OrderType.MARKET in supported_types
        assert OrderType.LIMIT in supported_types
        assert OrderType.STOP in supported_types
        assert OrderType.STOP_LIMIT in supported_types
        assert OrderType.TRAILING_STOP in supported_types
        assert OrderType.BRACKET in supported_types

    def test_supported_asset_classes(self, paper_broker):
        """Test supported asset classes."""
        asset_classes = paper_broker.get_supported_asset_classes()

        assert 'STK' in asset_classes
        assert 'OPT' in asset_classes
        assert 'FUT' in asset_classes
        assert 'CASH' in asset_classes

    @pytest.mark.asyncio
    async def test_account_info_paper_mode(self, paper_broker):
        """Test account info in paper mode."""
        account_info = await paper_broker.get_account_info()

        assert account_info['account_type'] == 'paper'
        assert account_info['trading_mode'] == 'paper'
        assert 'total_value' in account_info
        assert 'paper_trading_config' in account_info
        assert 'connection_info' in account_info

    @pytest.mark.asyncio
    async def test_account_info_live_mode(self, live_broker):
        """Test account info in live mode."""
        account_info = await live_broker.get_account_info()

        assert account_info['account_type'] == 'live'
        assert account_info['trading_mode'] == 'live'
        assert 'connection_info' in account_info
        assert 'account_values' in account_info

    @pytest.mark.asyncio
    async def test_ibkr_specific_info(self, paper_broker):
        """Test IBKR-specific broker information."""
        info = await paper_broker.get_ibkr_specific_info()

        assert info['broker_type'] == 'ibkr'
        assert info['trading_mode'] == 'paper'
        assert info['paper_trading'] is True
        assert 'supported_order_types' in info
        assert 'supported_asset_classes' in info
        assert 'connection_info' in info


class TestIBKRUtils:
    """Test cases for IBKR utility functions."""

    def test_contract_manager_stock(self):
        """Test IBKR contract manager for stocks."""
        manager = IBKRContractManager()

        contract = manager.create_stock_contract('AAPL')
        assert contract.symbol == 'AAPL'
        assert contract.secType == 'STK'
        assert contract.exchange == 'SMART'
        assert contract.currency == 'USD'

    def test_contract_manager_forex(self):
        """Test IBKR contract manager for forex."""
        manager = IBKRContractManager()

        contract = manager.create_forex_contract('EUR')
        assert contract.symbol == 'EURUSD'
        assert contract.secType == 'CASH'

    def test_contract_manager_caching(self):
        """Test contract caching functionality."""
        manager = IBKRContractManager()

        # First call should create contract
        contract1 = manager.get_contract('AAPL', 'STK')
        assert contract1 is not None

        # Second call should return cached contract
        contract2 = manager.get_contract('AAPL', 'STK')
        assert contract2 is contract1  # Same object reference

    def test_symbol_validation(self):
        """Test symbol format validation."""
        manager = IBKRContractManager()

        # Valid stock symbols
        assert manager.validate_symbol_format('AAPL', 'STK')[0] is True
        assert manager.validate_symbol_format('MSFT', 'STK')[0] is True

        # Invalid stock symbols
        assert manager.validate_symbol_format('TOOLONG', 'STK')[0] is False
        assert manager.validate_symbol_format('', 'STK')[0] is False

        # Valid forex symbols
        assert manager.validate_symbol_format('EURUSD', 'CASH')[0] is True
        assert manager.validate_symbol_format('GBPUSD', 'CASH')[0] is True

        # Invalid forex symbols
        assert manager.validate_symbol_format('EUR', 'CASH')[0] is False
        assert manager.validate_symbol_format('EURUSD123', 'CASH')[0] is False

    def test_commission_calculator_stocks(self):
        """Test stock commission calculation."""
        calculator = IBKRCommissionCalculator()

        # Test small order
        result = calculator.calculate_stock_commission(100, 150.0)
        assert result['commission'] >= result['min_commission']
        assert 'rate_per_share' in result

        # Test large order
        result = calculator.calculate_stock_commission(1000, 150.0)
        assert result['commission'] > 0
        assert result['commission'] <= result['max_commission']

    def test_commission_calculator_options(self):
        """Test option commission calculation."""
        calculator = IBKRCommissionCalculator()

        result = calculator.calculate_option_commission(10, 5.0)
        assert result['commission'] >= result['min_commission']
        assert result['commission'] <= result['max_commission']
        assert 'rate_per_contract' in result

    def test_commission_calculator_forex(self):
        """Test forex cost calculation."""
        calculator = IBKRCommissionCalculator()

        result = calculator.calculate_forex_cost('EURUSD', 100000.0)
        assert result['cost'] > 0
        assert 'spread_pips' in result
        assert 'spread_decimal' in result

    def test_config_template_creation(self):
        """Test IBKR configuration template creation."""
        paper_config = create_ibkr_config_template('paper')
        live_config = create_ibkr_config_template('live')

        assert paper_config['type'] == 'ibkr'
        assert paper_config['trading_mode'] == 'paper'
        assert paper_config['connection']['port'] == 7497
        assert 'paper_trading_config' in paper_config
        assert 'ibkr_config' in paper_config

        assert live_config['trading_mode'] == 'live'
        assert live_config['connection']['port'] == 4001
        assert live_config['live_trading_confirmed'] is False
        assert '_WARNING' in live_config


class TestIBKRIntegration:
    """Integration tests for IBKR broker."""

    @pytest.mark.asyncio
    async def test_full_paper_trading_workflow(self):
        """Test complete paper trading workflow."""
        config = create_ibkr_config_template('paper')
        config['notifications']['email_enabled'] = False
        config['notifications']['telegram_enabled'] = False

        with patch('src.trading.broker.ibkr_broker.IB') as mock_ib_class:
            mock_ib = Mock()
            mock_ib.connect.return_value = None
            mock_ib.isConnected.return_value = True
            mock_ib.accountSummary.return_value = []
            mock_ib.portfolio.return_value = []
            mock_ib.positions.return_value = []
            mock_ib.reqMktData.return_value = Mock()
            mock_ib_class.return_value = mock_ib

            broker = IBKRBroker(
                host='127.0.0.1',
                port=7497,
                client_id=1,
                cash=25000.0,
                config=config
            )

            # Connect
            await broker.connect()
            assert broker.is_connected

            # Update market data
            broker.update_market_data_cache('AAPL', 150.0)

            # Place buy order
            buy_order = Order(
                symbol='AAPL',
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=100
            )

            buy_order_id = await broker.place_order(buy_order)
            assert buy_order_id == buy_order.order_id

            # Check positions
            positions = await broker.get_positions()
            assert 'AAPL' in positions

            # Place sell order
            sell_order = Order(
                symbol='AAPL',
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=100
            )

            broker.update_market_data_cache('AAPL', 155.0)  # Price increase
            sell_order_id = await broker.place_order(sell_order)
            assert sell_order_id == sell_order.order_id

            # Check portfolio performance
            portfolio = await broker.get_portfolio()
            assert portfolio.realized_pnl > 0  # Should have profit

            # Get performance report
            if hasattr(broker, 'get_paper_trading_performance_report'):
                report = broker.get_paper_trading_performance_report()
                assert 'portfolio_metrics' in report
                assert 'trading_statistics' in report

            # Disconnect
            await broker.disconnect()
            assert not broker.is_connected


if __name__ == '__main__':
    pytest.main([__file__, '-v'])