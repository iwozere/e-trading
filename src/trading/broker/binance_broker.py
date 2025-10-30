#!/usr/bin/env python3
"""
Enhanced Binance Broker Implementation
-------------------------------------

This module provides a comprehensive Binance broker implementation with seamless
paper-to-live trading mode switching, realistic paper trading simulation, and
integrated notifications.

Features:
- Seamless paper/live mode switching via configuration
- Automatic testnet/mainnet URL selection based on trading mode
- Realistic paper trading simulation with WebSocket market data integration
- Advanced order types support (market, limit, stop-loss, OCO)
- Comprehensive execution quality metrics and analytics
- Integrated position notifications (email/Telegram)
- Binance-specific trading rules validation
- Real-time market data integration via WebSocket

Classes:
- BinanceBroker: Enhanced Binance broker with dual-mode support
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
import logging
import websocket
import threading

from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException

from src.trading.broker.base_broker import (
    BaseBroker, Order, Position, Portfolio, OrderStatus, OrderSide,
    OrderType, TradingMode, PaperTradingMode, ExecutionMetrics
)
from src.trading.broker.paper_trading_mixin import PaperTradingMixin

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class BinanceBroker(BaseBroker, PaperTradingMixin):
    """
    Enhanced Binance broker with seamless paper-to-live trading support.

    Features:
    - Automatic testnet/mainnet selection based on trading_mode
    - Realistic paper trading simulation with WebSocket integration
    - Advanced order types (market, limit, stop-loss, OCO)
    - Comprehensive execution quality metrics
    - Integrated position notifications
    - Binance-specific trading rules validation
    """

    def __init__(self, api_key: str, api_secret: str, cash: float = 10000.0, config: Dict[str, Any] = None):
        # Initialize configuration
        if config is None:
            config = {}

        # Set default Binance configuration
        config.setdefault('name', 'binance_broker')
        config.setdefault('type', 'binance')

        # Initialize parent classes
        super().__init__(config)

        # Binance-specific configuration
        self.api_key = api_key
        self.api_secret = api_secret

        # Set up Binance client based on trading mode
        self._setup_binance_client()

        # WebSocket connection for real-time data
        self.ws_client = None
        self.ws_thread = None
        self.subscribed_symbols = set()

        # Binance-specific order tracking
        self.binance_orders: Dict[str, Dict[str, Any]] = {}

        # Trading rules cache
        self.exchange_info = None
        self.symbol_filters = {}

        _logger.info("Enhanced Binance broker initialized - Mode: %s, URL: %s", self.trading_mode.value, self.client.API_URL)

    def _setup_binance_client(self):
        """Set up Binance client based on trading mode."""
        self.client = Client(self.api_key, self.api_secret)

        if self.trading_mode == TradingMode.PAPER:
            # Use Binance testnet for paper trading
            self.client.API_URL = 'https://testnet.binance.vision/api'
            self.ws_base_url = 'wss://testnet.binance.vision/ws'
            _logger.info("Using Binance testnet for paper trading")
        else:
            # Use Binance mainnet for live trading
            self.client.API_URL = 'https://api.binance.com/api'
            self.ws_base_url = 'wss://stream.binance.com:9443/ws'
            _logger.info("Using Binance mainnet for live trading")

    async def connect(self) -> bool:
        """Connect to Binance API and WebSocket."""
        try:
            # Test API connection
            account_info = self.client.get_account()
            _logger.info("Connected to Binance API - Account status: %s", account_info.get('accountType', 'Unknown'))

            # Load exchange info and trading rules
            await self._load_exchange_info()

            # Start WebSocket connection for real-time data
            if self.paper_trading_enabled:
                await self._start_websocket_connection()

            self.is_connected = True
            return True

        except Exception as e:
            _logger.exception("Failed to connect to Binance:")
            await self.notify_error(f"Binance connection failed: {str(e)}")
            return False

    async def disconnect(self) -> bool:
        """Disconnect from Binance API and WebSocket."""
        try:
            # Stop WebSocket connection
            if self.ws_client:
                self.ws_client.close()

            if self.ws_thread and self.ws_thread.is_alive():
                self.ws_thread.join(timeout=5)

            self.is_connected = False
            _logger.info("Disconnected from Binance")
            return True

        except Exception as e:
            _logger.exception("Error disconnecting from Binance:")
            return False

    async def _load_exchange_info(self):
        """Load Binance exchange information and trading rules."""
        try:
            self.exchange_info = self.client.get_exchange_info()

            # Cache symbol filters for quick access
            for symbol_info in self.exchange_info['symbols']:
                symbol = symbol_info['symbol']
                self.symbol_filters[symbol] = {
                    'status': symbol_info['status'],
                    'baseAsset': symbol_info['baseAsset'],
                    'quoteAsset': symbol_info['quoteAsset'],
                    'filters': {f['filterType']: f for f in symbol_info['filters']}
                }

            _logger.info("Loaded exchange info for %d symbols", len(self.symbol_filters))

        except Exception as e:
            _logger.exception("Failed to load exchange info:")

    async def _start_websocket_connection(self):
        """Start WebSocket connection for real-time market data."""
        if not self.paper_trading_enabled:
            return

        try:
            def on_message(ws, message):
                try:
                    data = json.loads(message)
                    self._process_websocket_message(data)
                except Exception as e:
                    _logger.exception("Error processing WebSocket message:")

            def on_error(ws, error):
                _logger.error("WebSocket error: %s", error)

            def on_close(ws, close_status_code, close_msg):
                _logger.info("WebSocket connection closed")

            def on_open(ws):
                _logger.info("WebSocket connection opened")

            # Start WebSocket in separate thread
            def run_websocket():
                self.ws_client = websocket.WebSocketApp(
                    f"{self.ws_base_url}/!ticker@arr",
                    on_message=on_message,
                    on_error=on_error,
                    on_close=on_close,
                    on_open=on_open
                )
                self.ws_client.run_forever()

            self.ws_thread = threading.Thread(target=run_websocket, daemon=True)
            self.ws_thread.start()

            _logger.info("Started WebSocket connection for market data")

        except Exception as e:
            _logger.exception("Failed to start WebSocket connection:")

    def _process_websocket_message(self, data):
        """Process incoming WebSocket market data."""
        try:
            if isinstance(data, list):
                # Process ticker array
                for ticker in data:
                    symbol = ticker.get('s')
                    price = float(ticker.get('c', 0))

                    if symbol and price > 0:
                        self.update_market_data_cache(symbol, price)

            elif isinstance(data, dict):
                # Process individual ticker
                symbol = data.get('s')
                price = float(data.get('c', 0))

                if symbol and price > 0:
                    self.update_market_data_cache(symbol, price)

        except Exception as e:
            _logger.exception("Error processing WebSocket data:")

    async def place_order(self, order: Order) -> str:
        """Place an order on Binance with mode-specific handling."""
        try:
            # Validate order
            is_valid, validation_message = await self.validate_order(order)
            if not is_valid:
                order.status = OrderStatus.REJECTED
                order.metadata['rejection_reason'] = validation_message
                _logger.warning("Order validation failed: %s", validation_message)
                return order.order_id

            # Apply Binance-specific validations
            binance_validation = await self._validate_binance_order(order)
            if not binance_validation[0]:
                order.status = OrderStatus.REJECTED
                order.metadata['rejection_reason'] = binance_validation[1]
                _logger.warning("Binance validation failed: %s", binance_validation[1])
                return order.order_id

            if self.paper_trading_enabled:
                # Handle paper trading
                current_price = await self._get_current_market_price(order.symbol)
                if current_price is None:
                    # Fetch price from API if not available in cache
                    current_price = await self._fetch_current_price(order.symbol)

                if current_price:
                    return await self.paper_place_order(order, current_price)
                else:
                    order.status = OrderStatus.REJECTED
                    order.metadata['rejection_reason'] = "Unable to get current market price"
                    return order.order_id
            else:
                # Handle live trading
                return await self._place_live_order(order)

        except Exception as e:
            _logger.exception("Error placing order:")
            order.status = OrderStatus.REJECTED
            order.metadata['rejection_reason'] = f"Order placement error: {str(e)}"
            await self.notify_error(f"Order placement failed: {str(e)}", {'order_id': order.order_id})
            return order.order_id

    async def _validate_binance_order(self, order: Order) -> Tuple[bool, str]:
        """Validate order against Binance-specific rules."""
        try:
            symbol = order.symbol

            # Check if symbol exists and is trading
            if symbol not in self.symbol_filters:
                return False, f"Symbol {symbol} not found or not supported"

            symbol_info = self.symbol_filters[symbol]
            if symbol_info['status'] != 'TRADING':
                return False, f"Symbol {symbol} is not currently trading (status: {symbol_info['status']})"

            filters = symbol_info['filters']

            # Validate lot size
            if 'LOT_SIZE' in filters:
                lot_filter = filters['LOT_SIZE']
                min_qty = float(lot_filter['minQty'])
                max_qty = float(lot_filter['maxQty'])
                step_size = float(lot_filter['stepSize'])

                if order.quantity < min_qty:
                    return False, f"Order quantity {order.quantity} below minimum {min_qty}"

                if order.quantity > max_qty:
                    return False, f"Order quantity {order.quantity} above maximum {max_qty}"

                # Check step size
                if step_size > 0:
                    remainder = (order.quantity - min_qty) % step_size
                    if remainder != 0:
                        return False, f"Order quantity {order.quantity} does not match step size {step_size}"

            # Validate price (for limit orders)
            if order.order_type == OrderType.LIMIT and order.price:
                if 'PRICE_FILTER' in filters:
                    price_filter = filters['PRICE_FILTER']
                    min_price = float(price_filter['minPrice'])
                    max_price = float(price_filter['maxPrice'])
                    tick_size = float(price_filter['tickSize'])

                    if order.price < min_price:
                        return False, f"Order price {order.price} below minimum {min_price}"

                    if order.price > max_price:
                        return False, f"Order price {order.price} above maximum {max_price}"

                    # Check tick size
                    if tick_size > 0:
                        remainder = (order.price - min_price) % tick_size
                        if remainder != 0:
                            return False, f"Order price {order.price} does not match tick size {tick_size}"

            # Validate notional value
            if 'MIN_NOTIONAL' in filters:
                notional_filter = filters['MIN_NOTIONAL']
                min_notional = float(notional_filter['minNotional'])

                order_notional = order.quantity * (order.price or await self._fetch_current_price(symbol) or 0)
                if order_notional < min_notional:
                    return False, f"Order notional {order_notional} below minimum {min_notional}"

            return True, "Order validated successfully"

        except Exception as e:
            _logger.exception("Error validating Binance order:")
            return False, f"Validation error: {str(e)}"

    async def _place_live_order(self, order: Order) -> str:
        """Place a live order on Binance."""
        try:
            # Convert our order format to Binance format
            binance_params = {
                'symbol': order.symbol,
                'side': SIDE_BUY if order.side == OrderSide.BUY else SIDE_SELL,
                'quantity': order.quantity,
                'newClientOrderId': order.client_order_id
            }

            # Set order type and additional parameters
            if order.order_type == OrderType.MARKET:
                binance_params['type'] = ORDER_TYPE_MARKET
            elif order.order_type == OrderType.LIMIT:
                binance_params['type'] = ORDER_TYPE_LIMIT
                binance_params['price'] = str(order.price)
                binance_params['timeInForce'] = TIME_IN_FORCE_GTC
            elif order.order_type == OrderType.STOP:
                binance_params['type'] = ORDER_TYPE_STOP_LOSS
                binance_params['stopPrice'] = str(order.stop_price)
            elif order.order_type == OrderType.STOP_LIMIT:
                binance_params['type'] = ORDER_TYPE_STOP_LOSS_LIMIT
                binance_params['price'] = str(order.price)
                binance_params['stopPrice'] = str(order.stop_price)
                binance_params['timeInForce'] = TIME_IN_FORCE_GTC
            else:
                raise ValueError(f"Unsupported order type for live trading: {order.order_type}")

            # Place order on Binance
            response = self.client.create_order(**binance_params)

            # Update order with Binance response
            order.metadata['binance_order_id'] = response['orderId']
            order.metadata['binance_response'] = response
            order.status = OrderStatus.PENDING

            # Store order for tracking
            self.binance_orders[order.order_id] = response

            _logger.info("Live order placed on Binance: %s -> %s", order.order_id, response['orderId'])

            # Send position notification for live trading
            await self.notify_position_event("opened", {
                'symbol': order.symbol,
                'side': order.side.value,
                'price': order.price or 0,
                'size': order.quantity,
                'timestamp': datetime.now(timezone.utc),
                'order_id': order.order_id,
                'binance_order_id': response['orderId']
            })

            return order.order_id

        except BinanceAPIException as e:
            _logger.exception("Binance API error placing live order:")
            order.status = OrderStatus.REJECTED
            order.metadata['rejection_reason'] = f"Binance API error: {e.message}"
            await self.notify_error(f"Live order failed: {e.message}", {'order_id': order.order_id})
            return order.order_id

        except Exception as e:
            _logger.exception("Error placing live order:")
            order.status = OrderStatus.REJECTED
            order.metadata['rejection_reason'] = f"Live order error: {str(e)}"
            await self.notify_error(f"Live order failed: {str(e)}", {'order_id': order.order_id})
            return order.order_id

    async def _fetch_current_price(self, symbol: str) -> Optional[float]:
        """Fetch current price from Binance API."""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            price = float(ticker['price'])

            # Update cache
            self.update_market_data_cache(symbol, price)

            return price

        except Exception as e:
            _logger.exception("Error fetching price for %s:", symbol)
            return None

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        try:
            if self.paper_trading_enabled:
                return await self.paper_cancel_order(order_id)
            else:
                # Cancel live order
                if order_id in self.binance_orders:
                    binance_order = self.binance_orders[order_id]
                    symbol = binance_order['symbol']
                    binance_order_id = binance_order['orderId']

                    response = self.client.cancel_order(
                        symbol=symbol,
                        orderId=binance_order_id
                    )

                    _logger.info("Live order cancelled: %s -> %s", order_id, binance_order_id)
                    return True
                else:
                    _logger.warning("Order %s not found for cancellation", order_id)
                    return False

        except Exception as e:
            _logger.exception("Error cancelling order %s:", order_id)
            return False

    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get order status."""
        try:
            if self.paper_trading_enabled:
                return self.get_paper_order_status(order_id)
            else:
                # Get live order status
                if order_id in self.binance_orders:
                    binance_order = self.binance_orders[order_id]
                    symbol = binance_order['symbol']
                    binance_order_id = binance_order['orderId']

                    response = self.client.get_order(
                        symbol=symbol,
                        orderId=binance_order_id
                    )

                    # Convert Binance status to our format
                    # This would need proper mapping implementation
                    return None  # Placeholder
                else:
                    return None

        except Exception as e:
            _logger.exception("Error getting order status for %s:", order_id)
            return None

    async def get_positions(self) -> Dict[str, Position]:
        """Get current positions."""
        try:
            if self.paper_trading_enabled:
                return await self.get_paper_positions()
            else:
                # Get live positions from Binance account
                account = self.client.get_account()
                positions = {}

                for balance in account['balances']:
                    asset = balance['asset']
                    free = float(balance['free'])
                    locked = float(balance['locked'])
                    total = free + locked

                    if total > 0:
                        # This is a simplified position representation
                        # Real implementation would need more sophisticated position tracking
                        positions[asset] = Position(
                            symbol=asset,
                            quantity=total,
                            average_price=0.0,  # Would need to track this
                            market_value=0.0,   # Would need current price
                            unrealized_pnl=0.0,
                            paper_trading=False
                        )

                return positions

        except Exception as e:
            _logger.exception("Error getting positions:")
            return {}

    async def get_portfolio(self) -> Portfolio:
        """Get portfolio information."""
        try:
            if self.paper_trading_enabled:
                return await self.get_paper_portfolio()
            else:
                # Get live portfolio from Binance account
                account = self.client.get_account()

                # Calculate total value (simplified)
                total_value = 0.0
                positions = await self.get_positions()

                # This would need proper implementation with current prices
                portfolio = Portfolio(
                    total_value=total_value,
                    cash=0.0,  # Binance doesn't have a single cash balance
                    positions=positions,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    paper_trading=False
                )

                return portfolio

        except Exception as e:
            _logger.exception("Error getting portfolio:")
            return Portfolio(
                total_value=0.0,
                cash=0.0,
                positions={},
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                paper_trading=False
            )

    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        try:
            if self.paper_trading_enabled:
                # Return paper trading account info
                portfolio = await self.get_paper_portfolio()
                return {
                    'account_type': 'paper',
                    'trading_mode': self.trading_mode.value,
                    'total_value': portfolio.total_value,
                    'cash': portfolio.cash,
                    'positions_count': len(portfolio.positions),
                    'paper_trading_config': {
                        'mode': self.paper_trading_config.mode.value,
                        'initial_balance': self.paper_trading_config.initial_balance,
                        'commission_rate': self.paper_trading_config.commission_rate
                    }
                }
            else:
                # Return live account info
                account = self.client.get_account()
                return {
                    'account_type': 'live',
                    'trading_mode': self.trading_mode.value,
                    'account_status': account.get('accountType', 'Unknown'),
                    'can_trade': account.get('canTrade', False),
                    'can_withdraw': account.get('canWithdraw', False),
                    'can_deposit': account.get('canDeposit', False),
                    'balances_count': len(account.get('balances', [])),
                    'maker_commission': account.get('makerCommission', 0),
                    'taker_commission': account.get('takerCommission', 0)
                }

        except Exception as e:
            _logger.exception("Error getting account info:")
            return {'error': str(e)}

    def get_supported_order_types(self) -> List[OrderType]:
        """Get list of supported order types for Binance."""
        return [
            OrderType.MARKET,
            OrderType.LIMIT,
            OrderType.STOP,
            OrderType.STOP_LIMIT,
            OrderType.OCO  # One-Cancels-Other (Binance specific)
        ]

    def get_supported_symbols(self) -> List[str]:
        """Get list of supported trading symbols."""
        if self.symbol_filters:
            return [symbol for symbol, info in self.symbol_filters.items()
                   if info['status'] == 'TRADING']
        return []

    async def get_binance_specific_info(self) -> Dict[str, Any]:
        """Get Binance-specific broker information."""
        return {
            'broker_type': 'binance',
            'api_url': self.client.API_URL,
            'websocket_url': getattr(self, 'ws_base_url', None),
            'trading_mode': self.trading_mode.value,
            'paper_trading': self.paper_trading_enabled,
            'supported_order_types': [ot.value for ot in self.get_supported_order_types()],
            'supported_symbols_count': len(self.get_supported_symbols()),
            'websocket_connected': self.ws_client is not None and hasattr(self.ws_client, 'sock') and self.ws_client.sock,
            'exchange_info_loaded': self.exchange_info is not None,
            'market_data_symbols': len(self.market_data_cache) if hasattr(self, 'market_data_cache') else 0
        }

    async def process_market_data_update(self):
        """Process pending orders against current market data (for paper trading)."""
        if self.paper_trading_enabled and hasattr(self, 'market_data_cache'):
            market_data = {symbol: data['price'] for symbol, data in self.market_data_cache.items()}
            await self.process_pending_paper_orders(market_data)