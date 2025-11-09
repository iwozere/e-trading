#!/usr/bin/env python3
"""
Test Suite for Enhanced Binance Broker
-------------------------------------

Comprehensive test suite for the enhanced Binance broker implementation,
covering both paper trading and live trading modes, order validation,
WebSocket integration, and notification systems.
"""

import pytest
from unittest.mock import patch, AsyncMock

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.trading.broker.binance_broker import BinanceBroker
from src.trading.broker.base_broker import (
    Order, OrderType, OrderSide, OrderStatus, TradingMode
)
from src.trading.broker.binance_utils import (
    BinanceSymbolValidator, BinanceCommissionCalculator,
    create_binance_config_template
)


class TestBinanceBroker:
    """Test cases for BinanceBroker class."""

    @pytest.fixture
    def paper_config(self):
        """Paper trading configuration fixture."""
        return {
            'type': 'binance',
            'trading_mode': 'paper',
            'cash': 10000.0,
            'paper_trading_config': {
                'mode': 'realistic',
                'initial_balance': 10000.0,
                'commission_rate': 0.001,
                'slippage_model': 'linear',
                'base_slippage': 0.0005,
                'latency_simulation': True,
                'min_latency_ms': 20,
                'max_latency_ms': 100
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
            'type': 'binance',
            'trading_mode': 'live',
            'cash': 5000.0,
            'live_trading_confirmed': True,
            'risk_management': {
                'max_position_size': 500.0,
                'max_daily_loss': 250.0
            },
            'notifications': {
                'position_opened': True,
                'position_closed': True,
                'email_enabled': False,  # Disable for tests
                'telegram_enabled': False
            }
        }

    @pytest.fixture
    def mock_exchange_info(self):
        """Mock Binance exchange info."""
        return {
            'symbols': [
                {
                    'symbol': 'BTCUSDT',
                    'status': 'TRADING',
                    'baseAsset': 'BTC',
                    'quoteAsset': 'USDT',
                    'baseAssetPrecision': 8,
                    'quoteAssetPrecision': 8,
                    'orderTypes': ['LIMIT', 'MARKET', 'STOP_LOSS', 'STOP_LOSS_LIMIT'],
                    'icebergAllowed': True,
                    'ocoAllowed': True,
                    'filters': [
                        {
                            'filterType': 'PRICE_FILTER',
                            'minPrice': '0.01000000',
                            'maxPrice': '1000000.00000000',
                            'tickSize': '0.01000000'
                        },
                        {
                            'filterType': 'LOT_SIZE',
                            'minQty': '0.00001000',
                            'maxQty': '9000.00000000',
                            'stepSize': '0.00001000'
                        },
                        {
                            'filterType': 'MIN_NOTIONAL',
                            'minNotional': '10.00000000'
                        }
                    ]
                }
            ]
        }

    @pytest.fixture
    def paper_broker(self, paper_config, mock_exchange_info):
        """Paper trading broker fixture."""
        with patch('src.trading.broker.binance_broker.Client') as mock_client:
            mock_client.return_value.get_account.return_value = {'accountType': 'SPOT'}
            mock_client.return_value.get_exchange_info.return_value = mock_exchange_info

            broker = BinanceBroker(
                api_key='test_key',
                api_secret='test_secret',
                cash=10000.0,
                config=paper_config
            )
            broker.exchange_info = mock_exchange_info
            broker._load_exchange_info = AsyncMock()
            return broker

    @pytest.fixture
    def live_broker(self, live_config, mock_exchange_info):
        """Live trading broker fixture."""
        with patch('src.trading.broker.binance_broker.Client') as mock_client:
            mock_client.return_value.get_account.return_value = {'accountType': 'SPOT'}
            mock_client.return_value.get_exchange_info.return_value = mock_exchange_info

            broker = BinanceBroker(
                api_key='test_key',
                api_secret='test_secret',
                cash=5000.0,
                config=live_config
            )
            broker.exchange_info = mock_exchange_info
            broker._load_exchange_info = AsyncMock()
            return broker

    def test_broker_initialization_paper_mode(self, paper_broker):
        """Test broker initialization in paper mode."""
        assert paper_broker.trading_mode == TradingMode.PAPER
        assert paper_broker.paper_trading_enabled is True
        assert paper_broker.client.API_URL == 'https://testnet.binance.vision/api'
        assert paper_broker.paper_portfolio is not None
        assert paper_broker.paper_portfolio.initial_balance == 10000.0

    def test_broker_initialization_live_mode(self, live_broker):
        """Test broker initialization in live mode."""
        assert live_broker.trading_mode == TradingMode.LIVE
        assert live_broker.paper_trading_enabled is False
        assert live_broker.client.API_URL == 'https://api.binance.com/api'

    @pytest.mark.asyncio
    async def test_connection_paper_mode(self, paper_broker):
        """Test connection in paper mode."""
        with patch.object(paper_broker, '_start_websocket_connection', new_callable=AsyncMock):
            result = await paper_broker.connect()
            assert result is True
            assert paper_broker.is_connected is True

    @pytest.mark.asyncio
    async def test_connection_live_mode(self, live_broker):
        """Test connection in live mode."""
        result = await live_broker.connect()
        assert result is True
        assert live_broker.is_connected is True

    @pytest.mark.asyncio
    async def test_paper_order_placement_market(self, paper_broker):
        """Test paper trading market order placement."""
        # Mock market price
        paper_broker.update_market_data_cache('BTCUSDT', 50000.0)

        order = Order(
            symbol='BTCUSDT',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.001
        )

        with patch.object(paper_broker, '_fetch_current_price', return_value=50000.0):
            order_id = await paper_broker.place_order(order)

            assert order_id == order.order_id
            assert order.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]

    @pytest.mark.asyncio
    async def test_paper_order_placement_limit(self, paper_broker):
        """Test paper trading limit order placement."""
        # Mock market price
        paper_broker.update_market_data_cache('BTCUSDT', 50000.0)

        order = Order(
            symbol='BTCUSDT',
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=49000.0  # Below market price
        )

        order_id = await paper_broker.place_order(order)

        assert order_id == order.order_id
        assert order.status == OrderStatus.PENDING  # Should remain pending

    @pytest.mark.asyncio
    async def test_live_order_placement(self, live_broker):
        """Test live order placement."""
        order = Order(
            symbol='BTCUSDT',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.001
        )

        mock_response = {
            'orderId': 12345,
            'symbol': 'BTCUSDT',
            'status': 'NEW',
            'clientOrderId': order.client_order_id
        }

        with patch.object(live_broker.client, 'create_order', return_value=mock_response):
            order_id = await live_broker.place_order(order)

            assert order_id == order.order_id
            assert order.status == OrderStatus.PENDING
            assert order.metadata['binance_order_id'] == 12345

    @pytest.mark.asyncio
    async def test_order_validation_binance_rules(self, paper_broker):
        """Test order validation against Binance rules."""
        # Test minimum quantity validation
        order = Order(
            symbol='BTCUSDT',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.000001  # Below minimum
        )

        is_valid, message = await paper_broker._validate_binance_order(order)
        assert is_valid is False
        assert 'minimum' in message.lower()

    @pytest.mark.asyncio
    async def test_order_validation_notional_value(self, paper_broker):
        """Test notional value validation."""
        order = Order(
            symbol='BTCUSDT',
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.0001,
            price=1.0  # Very low price, below min notional
        )

        with patch.object(paper_broker, '_fetch_current_price', return_value=50000.0):
            is_valid, message = await paper_broker._validate_binance_order(order)
            assert is_valid is False
            assert 'notional' in message.lower()

    @pytest.mark.asyncio
    async def test_paper_portfolio_tracking(self, paper_broker):
        """Test paper trading portfolio tracking."""
        # Place a buy order
        paper_broker.update_market_data_cache('BTCUSDT', 50000.0)

        buy_order = Order(
            symbol='BTCUSDT',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.001
        )

        with patch.object(paper_broker, '_fetch_current_price', return_value=50000.0):
            await paper_broker.place_order(buy_order)

        # Check portfolio
        portfolio = await paper_broker.get_paper_portfolio()
        assert portfolio.cash < 10000.0  # Cash should decrease

        positions = await paper_broker.get_paper_positions()
        assert 'BTCUSDT' in positions
        assert positions['BTCUSDT'].quantity == 0.001

    @pytest.mark.asyncio
    async def test_websocket_message_processing(self, paper_broker):
        """Test WebSocket message processing."""
        # Test ticker message
        ticker_message = {
            's': 'BTCUSDT',
            'c': '50000.00',
            'P': '2.5',
            'v': '1000.0'
        }

        paper_broker._process_websocket_message(ticker_message)

        # Check if market data was updated
        assert 'BTCUSDT' in paper_broker.market_data_cache
        assert paper_broker.market_data_cache['BTCUSDT']['price'] == 50000.0

    @pytest.mark.asyncio
    async def test_pending_order_processing(self, paper_broker):
        """Test processing of pending orders against market data."""
        # Place a limit order below market
        order = Order(
            symbol='BTCUSDT',
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=49000.0
        )

        paper_broker.update_market_data_cache('BTCUSDT', 50000.0)
        await paper_broker.place_order(order)

        # Simulate price drop to trigger order
        market_data = {'BTCUSDT': 48000.0}
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
        assert OrderType.OCO in supported_types

    @pytest.mark.asyncio
    async def test_account_info_paper_mode(self, paper_broker):
        """Test account info in paper mode."""
        account_info = await paper_broker.get_account_info()

        assert account_info['account_type'] == 'paper'
        assert account_info['trading_mode'] == 'paper'
        assert 'total_value' in account_info
        assert 'paper_trading_config' in account_info

    @pytest.mark.asyncio
    async def test_account_info_live_mode(self, live_broker):
        """Test account info in live mode."""
        mock_account = {
            'accountType': 'SPOT',
            'canTrade': True,
            'canWithdraw': True,
            'canDeposit': True,
            'balances': [],
            'makerCommission': 10,
            'takerCommission': 10
        }

        with patch.object(live_broker.client, 'get_account', return_value=mock_account):
            account_info = await live_broker.get_account_info()

            assert account_info['account_type'] == 'live'
            assert account_info['trading_mode'] == 'live'
            assert account_info['can_trade'] is True

    @pytest.mark.asyncio
    async def test_binance_specific_info(self, paper_broker):
        """Test Binance-specific broker information."""
        info = await paper_broker.get_binance_specific_info()

        assert info['broker_type'] == 'binance'
        assert info['trading_mode'] == 'paper'
        assert info['paper_trading'] is True
        assert 'supported_order_types' in info
        assert 'api_url' in info


class TestBinanceUtils:
    """Test cases for Binance utility functions."""

    @pytest.fixture
    def mock_exchange_info(self):
        """Mock exchange info for testing."""
        return {
            'symbols': [
                {
                    'symbol': 'BTCUSDT',
                    'status': 'TRADING',
                    'baseAsset': 'BTC',
                    'quoteAsset': 'USDT',
                    'baseAssetPrecision': 8,
                    'quoteAssetPrecision': 8,
                    'orderTypes': ['LIMIT', 'MARKET'],
                    'icebergAllowed': True,
                    'ocoAllowed': True,
                    'filters': [
                        {
                            'filterType': 'PRICE_FILTER',
                            'minPrice': '0.01000000',
                            'maxPrice': '1000000.00000000',
                            'tickSize': '0.01000000'
                        },
                        {
                            'filterType': 'LOT_SIZE',
                            'minQty': '0.00001000',
                            'maxQty': '9000.00000000',
                            'stepSize': '0.00001000'
                        },
                        {
                            'filterType': 'MIN_NOTIONAL',
                            'minNotional': '10.00000000'
                        }
                    ]
                }
            ]
        }

    def test_symbol_validator_initialization(self, mock_exchange_info):
        """Test BinanceSymbolValidator initialization."""
        validator = BinanceSymbolValidator(mock_exchange_info)

        assert validator.is_symbol_valid('BTCUSDT')
        assert not validator.is_symbol_valid('INVALID')

    def test_quantity_validation(self, mock_exchange_info):
        """Test quantity validation."""
        validator = BinanceSymbolValidator(mock_exchange_info)

        # Valid quantity
        is_valid, error, adjusted = validator.validate_quantity('BTCUSDT', 0.001)
        assert is_valid is True
        assert adjusted == 0.001

        # Below minimum
        is_valid, error, adjusted = validator.validate_quantity('BTCUSDT', 0.000001)
        assert is_valid is False
        assert 'minimum' in error.lower()

    def test_price_validation(self, mock_exchange_info):
        """Test price validation."""
        validator = BinanceSymbolValidator(mock_exchange_info)

        # Valid price
        is_valid, error, adjusted = validator.validate_price('BTCUSDT', 50000.0)
        assert is_valid is True
        assert adjusted == 50000.0

        # Below minimum
        is_valid, error, adjusted = validator.validate_price('BTCUSDT', 0.001)
        assert is_valid is False
        assert 'minimum' in error.lower()

    def test_notional_validation(self, mock_exchange_info):
        """Test notional value validation."""
        validator = BinanceSymbolValidator(mock_exchange_info)

        # Valid notional
        is_valid, error = validator.validate_notional('BTCUSDT', 0.001, 50000.0)
        assert is_valid is True

        # Below minimum notional
        is_valid, error = validator.validate_notional('BTCUSDT', 0.0001, 1.0)
        assert is_valid is False
        assert 'notional' in error.lower()

    def test_commission_calculator(self):
        """Test commission calculation."""
        calculator = BinanceCommissionCalculator(maker_rate=0.001, taker_rate=0.001)

        # Test basic commission
        commission = calculator.calculate_commission(0.001, 50000.0, is_maker=False)
        expected = 0.001 * 50000.0 * 0.001  # quantity * price * rate
        assert commission == expected

        # Test BNB discount
        discounted = calculator.calculate_bnb_discount_commission(commission, 0.25)
        expected_discounted = commission * 0.75
        assert discounted == expected_discounted

    def test_config_template_creation(self):
        """Test Binance configuration template creation."""
        paper_config = create_binance_config_template('paper')
        live_config = create_binance_config_template('live')

        assert paper_config['type'] == 'binance'
        assert paper_config['trading_mode'] == 'paper'
        assert 'paper_trading_config' in paper_config
        assert 'binance_config' in paper_config

        assert live_config['trading_mode'] == 'live'
        assert live_config['live_trading_confirmed'] is False
        assert '_WARNING' in live_config


class TestBinanceIntegration:
    """Integration tests for Binance broker."""

    @pytest.mark.asyncio
    async def test_full_paper_trading_workflow(self):
        """Test complete paper trading workflow."""
        config = create_binance_config_template('paper')
        config['notifications']['email_enabled'] = False
        config['notifications']['telegram_enabled'] = False

        with patch('src.trading.broker.binance_broker.Client') as mock_client:
            mock_client.return_value.get_account.return_value = {'accountType': 'SPOT'}
            mock_client.return_value.get_exchange_info.return_value = {
                'symbols': [{
                    'symbol': 'BTCUSDT',
                    'status': 'TRADING',
                    'baseAsset': 'BTC',
                    'quoteAsset': 'USDT',
                    'baseAssetPrecision': 8,
                    'quoteAssetPrecision': 8,
                    'orderTypes': ['LIMIT', 'MARKET'],
                    'icebergAllowed': True,
                    'ocoAllowed': True,
                    'filters': [
                        {'filterType': 'PRICE_FILTER', 'minPrice': '0.01', 'maxPrice': '1000000', 'tickSize': '0.01'},
                        {'filterType': 'LOT_SIZE', 'minQty': '0.00001', 'maxQty': '9000', 'stepSize': '0.00001'},
                        {'filterType': 'MIN_NOTIONAL', 'minNotional': '10.0'}
                    ]
                }]
            }

            broker = BinanceBroker('test_key', 'test_secret', 10000.0, config)

            # Connect
            await broker.connect()
            assert broker.is_connected

            # Update market data
            broker.update_market_data_cache('BTCUSDT', 50000.0)

            # Place buy order
            buy_order = Order(
                symbol='BTCUSDT',
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=0.001
            )

            with patch.object(broker, '_fetch_current_price', return_value=50000.0):
                buy_order_id = await broker.place_order(buy_order)
                assert buy_order_id == buy_order.order_id

            # Check positions
            positions = await broker.get_positions()
            assert 'BTCUSDT' in positions

            # Place sell order
            sell_order = Order(
                symbol='BTCUSDT',
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=0.001
            )

            with patch.object(broker, '_fetch_current_price', return_value=51000.0):
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