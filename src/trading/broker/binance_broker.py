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
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Dict, List, Optional, Any, Tuple
import websocket
import threading

from binance.client import Client
from binance.enums import (
    ORDER_TYPE_LIMIT,
    ORDER_TYPE_MARKET,
    ORDER_TYPE_STOP_LOSS,
    ORDER_TYPE_STOP_LOSS_LIMIT,
    SIDE_BUY,
    SIDE_SELL,
    TIME_IN_FORCE_GTC,
)
from binance.exceptions import BinanceAPIException

from src.trading.broker.base_broker import (
    BaseBroker, Order, Position, Portfolio, OrderStatus, OrderSide,
    OrderType, TradingMode
)
from src.trading.broker.paper_trading_mixin import PaperTradingMixin

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

# Assets that are themselves denominated in USD — price treated as 1.0 when
# computing market value and cost basis.
_QUOTE_ASSETS: frozenset = frozenset({"USDT", "BUSD", "USDC", "TUSD", "FDUSD", "DAI"})


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
            account_info = await asyncio.to_thread(self.client.get_account)
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

        except Exception:
            _logger.exception("Error disconnecting from Binance:")
            return False

    async def _load_exchange_info(self):
        """Load Binance exchange information and trading rules."""
        try:
            self.exchange_info = await asyncio.to_thread(self.client.get_exchange_info)

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

        except Exception:
            _logger.exception("Failed to load exchange info:")

    async def _start_websocket_connection(self):
        """Start WebSocket connection for real-time market data with auto-reconnect."""
        if not self.paper_trading_enabled:
            return

        try:
            self._ws_last_message_time = datetime.now(timezone.utc)
            self._ws_stop_event = threading.Event()

            def on_message(ws, message):
                try:
                    self._ws_last_message_time = datetime.now(timezone.utc)
                    data = json.loads(message)
                    self._process_websocket_message(data)
                except Exception:
                    _logger.exception("Error processing WebSocket message:")

            def on_error(ws, error):
                _logger.error("WebSocket error: %s", error)

            def on_close(ws, close_status_code, close_msg):
                _logger.warning(
                    "WebSocket connection closed (status=%s, msg=%s); reconnect loop will retry",
                    close_status_code, close_msg
                )

            def on_open(ws):
                _logger.info("WebSocket connection opened")
                self._ws_last_message_time = datetime.now(timezone.utc)

            def run_websocket_with_reconnect():
                backoff = 1
                while not self._ws_stop_event.is_set():
                    try:
                        self.ws_client = websocket.WebSocketApp(
                            f"{self.ws_base_url}/!ticker@arr",
                            on_message=on_message,
                            on_error=on_error,
                            on_close=on_close,
                            on_open=on_open
                        )
                        self.ws_client.run_forever(ping_interval=30, ping_timeout=10)
                        # run_forever returned → connection closed; apply backoff before retry
                        if not self._ws_stop_event.is_set():
                            _logger.warning("WebSocket disconnected; reconnecting in %ds", backoff)
                            self._ws_stop_event.wait(timeout=backoff)
                            backoff = min(backoff * 2, 60)
                    except Exception:
                        _logger.exception("WebSocket reconnect loop error; retrying in %ds:", backoff)
                        self._ws_stop_event.wait(timeout=backoff)
                        backoff = min(backoff * 2, 60)
                _logger.info("WebSocket reconnect loop exited cleanly")

            self.ws_thread = threading.Thread(target=run_websocket_with_reconnect, daemon=True)
            self.ws_thread.start()

            _logger.info("Started WebSocket connection for market data (auto-reconnect enabled)")

        except Exception:
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

        except Exception:
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

            # Validate and round lot size using Decimal to avoid float-modulo artefacts.
            # Binance requires quantities that are exact multiples of stepSize; we round
            # down rather than reject so valid orders are never spuriously refused.
            if 'LOT_SIZE' in filters:
                lot_filter = filters['LOT_SIZE']
                min_qty = Decimal(lot_filter['minQty'])
                max_qty = Decimal(lot_filter['maxQty'])
                step_size = Decimal(lot_filter['stepSize'])
                qty = Decimal(str(order.quantity))

                if step_size > 0:
                    qty = (qty / step_size).to_integral_value(rounding=ROUND_DOWN) * step_size

                if qty < min_qty:
                    return False, f"Order quantity {order.quantity} below minimum {min_qty}"
                if qty > max_qty:
                    return False, f"Order quantity {order.quantity} above maximum {max_qty}"

                order.quantity = float(qty)

            # Validate and round price for limit orders using Decimal (same rationale).
            if order.order_type == OrderType.LIMIT and order.price:
                if 'PRICE_FILTER' in filters:
                    price_filter = filters['PRICE_FILTER']
                    min_price = Decimal(price_filter['minPrice'])
                    max_price = Decimal(price_filter['maxPrice'])
                    tick_size = Decimal(price_filter['tickSize'])
                    price = Decimal(str(order.price))

                    if tick_size > 0:
                        price = (price / tick_size).to_integral_value(rounding=ROUND_DOWN) * tick_size

                    if price < min_price:
                        return False, f"Order price {order.price} below minimum {min_price}"
                    if price > max_price:
                        return False, f"Order price {order.price} above maximum {max_price}"

                    order.price = float(price)

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
            response = await asyncio.to_thread(lambda: self.client.create_order(**binance_params))

            # Update order with Binance response
            order.metadata['binance_order_id'] = response['orderId']
            order.metadata['binance_response'] = response
            order.status = OrderStatus.PENDING

            # Store order for tracking
            self.binance_orders[order.order_id] = response

            _logger.info("Live order placed on Binance: %s -> %s", order.order_id, response['orderId'])

            # NOTE: position notification is intentionally deferred to fill confirmation
            # (get_order_status / order-update webhook) so "Position Opened" is only
            # emitted after the order actually executes, not on submission.

            return order.order_id

        except BinanceAPIException as e:
            _logger.exception("Binance API error placing live order:")
            order.status = OrderStatus.REJECTED
            order.metadata['rejection_reason'] = f"Binance API error: {e.message}"
            await self.notify_error(f"Live order failed: {e.message}", {'order_id': order.order_id})
            return order.order_id

        except Exception as e:
            _logger.exception("Error placing live order:")
            # The exception may have occurred after Binance accepted the order (e.g. network
            # timeout).  Query by clientOrderId before marking REJECTED to avoid a ghost
            # position where the exchange holds a real order the bot thinks was rejected.
            if order.client_order_id:
                try:
                    reconciled = await asyncio.to_thread(
                        lambda: self.client.get_order(
                            symbol=order.symbol,
                            origClientOrderId=order.client_order_id,
                        )
                    )
                    order.metadata['binance_order_id'] = reconciled['orderId']
                    order.metadata['binance_response'] = reconciled
                    order.status = OrderStatus.PENDING
                    self.binance_orders[order.order_id] = reconciled
                    _logger.warning(
                        "Order %s found on Binance after exception — marked PENDING for reconciliation",
                        order.client_order_id,
                    )
                    return order.order_id
                except BinanceAPIException:
                    pass  # Order genuinely not on exchange; safe to mark REJECTED below
            order.status = OrderStatus.REJECTED
            order.metadata['rejection_reason'] = f"Live order error: {str(e)}"
            await self.notify_error(f"Live order failed: {str(e)}", {'order_id': order.order_id})
            return order.order_id

    async def _fetch_current_price(self, symbol: str) -> Optional[float]:
        """Fetch current price from Binance API."""
        try:
            ticker = await asyncio.to_thread(lambda: self.client.get_symbol_ticker(symbol=symbol))
            price = float(ticker['price'])

            # Update cache
            self.update_market_data_cache(symbol, price)

            return price

        except Exception:
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

                    response = await asyncio.to_thread(
                        lambda: self.client.cancel_order(symbol=symbol, orderId=binance_order_id)
                    )

                    _logger.info("Live order cancelled: %s -> %s", order_id, binance_order_id)
                    return True
                else:
                    _logger.warning("Order %s not found for cancellation", order_id)
                    return False

        except Exception:
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

                    # Fetch live order status from Binance API
                    try:
                        binance_order = await asyncio.to_thread(
                            lambda: self.client.get_order(symbol=symbol, orderId=binance_order_id)
                        )
                        
                        # Map Binance status to OrderStatus
                        status_map = {
                            'NEW': OrderStatus.PENDING,
                            'PARTIALLY_FILLED': OrderStatus.PARTIALLY_FILLED,
                            'FILLED': OrderStatus.FILLED,
                            'CANCELED': OrderStatus.CANCELLED,
                            'REJECTED': OrderStatus.REJECTED,
                            'EXPIRED': OrderStatus.REJECTED
                        }
                        
                        # Create Order object
                        order = Order(
                            symbol=symbol,
                            side=OrderSide.BUY if binance_order['side'] == 'BUY' else OrderSide.SELL,
                            quantity=float(binance_order['origQty']),
                            price=float(binance_order['price']) if float(binance_order['price']) > 0 else None,
                            order_type=OrderType.MARKET if binance_order['type'] == 'MARKET' else OrderType.LIMIT,
                            order_id=order_id,
                            status=status_map.get(binance_order['status'], OrderStatus.PENDING),
                            client_order_id=binance_order.get('clientOrderId')
                        )
                        
                        # Add metadata
                        order.metadata['binance_response'] = binance_order
                        order.metadata['filled_quantity'] = float(binance_order['executedQty'])
                        order.metadata['average_fill_price'] = float(binance_order['cummulativeQuoteQty']) / float(binance_order['executedQty']) if float(binance_order['executedQty']) > 0 else 0.0
                        
                        return order
                        
                    except BinanceAPIException as e:
                        _logger.error("Binance API error getting status for %s: %s", order_id, e.message)
                        return None
                else:
                    return None

        except NotImplementedError:
            raise
        except Exception:
            _logger.exception("Error getting order status for %s:", order_id)
            return None

    async def _build_price_map(self) -> Dict[str, float]:
        """
        Return a {symbol: price} map for all tickers using a single API call.
        Quote assets (USDT, USDC …) are mapped to 1.0 directly.
        """
        price_map: Dict[str, float] = {asset: 1.0 for asset in _QUOTE_ASSETS}
        try:
            tickers = await asyncio.to_thread(self.client.get_all_tickers)
            for ticker in tickers:
                price_map[ticker['symbol']] = float(ticker['price'])
        except Exception:
            _logger.exception("Error fetching all tickers for price map:")
        return price_map

    def _asset_usdt_price(self, asset: str, price_map: Dict[str, float]) -> float:
        """Return the USDT price for a single asset, falling back to 0.0."""
        if asset in _QUOTE_ASSETS:
            return 1.0
        return price_map.get(f"{asset}USDT", price_map.get(f"{asset}BUSD", 0.0))

    def _compute_cost_basis(self) -> Dict[str, float]:
        """
        Estimate average purchase price per asset from orders placed this session.

        Returns {base_asset: avg_price}.  Only BUY fills contribute; SELL fills
        reduce the tracked quantity.  The result is best-effort — it covers only
        orders placed since the broker was initialised (no cross-session history).
        """
        totals: Dict[str, List] = {}  # asset -> [cumulative_cost, cumulative_qty]
        for response in self.binance_orders.values():
            symbol = response.get('symbol', '')
            side = response.get('side', '')
            exec_qty = float(response.get('executedQty') or 0)
            quote_qty = float(response.get('cummulativeQuoteQty') or 0)
            if exec_qty <= 0:
                continue
            base = None
            for q in sorted(_QUOTE_ASSETS, key=len, reverse=True):
                if symbol.endswith(q):
                    base = symbol[: -len(q)]
                    break
            if base is None:
                continue
            if base not in totals:
                totals[base] = [0.0, 0.0]
            if side == 'BUY':
                totals[base][0] += quote_qty
                totals[base][1] += exec_qty
            elif side == 'SELL':
                totals[base][0] -= quote_qty
                totals[base][1] -= exec_qty

        result: Dict[str, float] = {}
        for asset, (cost, qty) in totals.items():
            if qty > 0:
                result[asset] = cost / qty
        return result

    async def get_positions(self) -> Dict[str, Position]:
        """Get current positions with live market values and estimated cost basis."""
        try:
            if self.paper_trading_enabled:
                return await self.get_paper_positions()

            account = await asyncio.to_thread(self.client.get_account)
            price_map = await self._build_price_map()
            cost_basis = self._compute_cost_basis()
            positions: Dict[str, Position] = {}

            for balance in account['balances']:
                asset = balance['asset']
                free = float(balance['free'])
                locked = float(balance['locked'])
                total = free + locked
                if total <= 0:
                    continue

                price = self._asset_usdt_price(asset, price_map)
                market_value = total * price
                avg_price = cost_basis.get(asset, price)
                cost = avg_price * total
                unrealized_pnl = market_value - cost if avg_price > 0 else 0.0

                positions[asset] = Position(
                    symbol=asset,
                    quantity=total,
                    average_price=avg_price,
                    market_value=market_value,
                    unrealized_pnl=unrealized_pnl,
                    paper_trading=False,
                )

            return positions

        except Exception:
            _logger.exception("Error getting positions:")
            return {}

    async def get_portfolio(self) -> Portfolio:
        """Get portfolio with real market values derived from live ticker prices."""
        try:
            if self.paper_trading_enabled:
                return await self.get_paper_portfolio()

            account = await asyncio.to_thread(self.client.get_account)
            price_map = await self._build_price_map()
            cost_basis = self._compute_cost_basis()

            cash = 0.0
            total_value = 0.0
            total_unrealized_pnl = 0.0
            positions: Dict[str, Position] = {}

            for balance in account['balances']:
                asset = balance['asset']
                free = float(balance['free'])
                locked = float(balance['locked'])
                total = free + locked
                if total <= 0:
                    continue

                price = self._asset_usdt_price(asset, price_map)
                market_value = total * price
                avg_price = cost_basis.get(asset, price)
                unrealized_pnl = (price - avg_price) * total if avg_price > 0 else 0.0

                total_value += market_value
                total_unrealized_pnl += unrealized_pnl

                if asset in _QUOTE_ASSETS:
                    cash += free  # locked quote is in open orders
                else:
                    positions[asset] = Position(
                        symbol=asset,
                        quantity=total,
                        average_price=avg_price,
                        market_value=market_value,
                        unrealized_pnl=unrealized_pnl,
                        paper_trading=False,
                    )

            return Portfolio(
                total_value=total_value,
                cash=cash,
                positions=positions,
                unrealized_pnl=total_unrealized_pnl,
                realized_pnl=0.0,
                paper_trading=False,
            )

        except Exception:
            _logger.exception("Error getting portfolio:")
            return Portfolio(
                total_value=0.0,
                cash=0.0,
                positions={},
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                paper_trading=False,
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
                account = await asyncio.to_thread(self.client.get_account)
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