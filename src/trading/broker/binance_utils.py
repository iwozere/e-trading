#!/usr/bin/env python3
"""
Binance Utilities Module
-----------------------

This module provides utility functions and configurations specific to Binance
trading operations, including order validation, symbol management, and
WebSocket data processing.

Features:
- Binance trading rules validation
- Symbol filter management
- Order size and price validation
- WebSocket message processing
- Commission calculation
- Market data utilities

Functions:
- validate_binance_symbol: Validate trading symbol
- calculate_binance_commission: Calculate trading commission
- format_binance_quantity: Format quantity according to symbol rules
- format_binance_price: Format price according to symbol rules
- process_binance_ticker: Process ticker data from WebSocket
"""

import math
from typing import Dict, List, Optional, Any, Tuple

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class BinanceSymbolValidator:
    """Validator for Binance symbol trading rules."""

    def __init__(self, exchange_info: Dict[str, Any]):
        self.exchange_info = exchange_info
        self.symbol_filters = self._build_symbol_filters()

    def _build_symbol_filters(self) -> Dict[str, Dict[str, Any]]:
        """Build symbol filters dictionary for quick access."""
        filters = {}

        if not self.exchange_info or 'symbols' not in self.exchange_info:
            return filters

        for symbol_info in self.exchange_info['symbols']:
            symbol = symbol_info['symbol']
            filters[symbol] = {
                'status': symbol_info['status'],
                'baseAsset': symbol_info['baseAsset'],
                'quoteAsset': symbol_info['quoteAsset'],
                'baseAssetPrecision': symbol_info['baseAssetPrecision'],
                'quoteAssetPrecision': symbol_info['quoteAssetPrecision'],
                'orderTypes': symbol_info['orderTypes'],
                'icebergAllowed': symbol_info['icebergAllowed'],
                'ocoAllowed': symbol_info['ocoAllowed'],
                'filters': {f['filterType']: f for f in symbol_info['filters']}
            }

        return filters

    def is_symbol_valid(self, symbol: str) -> bool:
        """Check if symbol is valid and tradeable."""
        return (symbol in self.symbol_filters and
                self.symbol_filters[symbol]['status'] == 'TRADING')

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get symbol information."""
        return self.symbol_filters.get(symbol)

    def validate_quantity(self, symbol: str, quantity: float) -> Tuple[bool, str, float]:
        """
        Validate and adjust quantity according to symbol rules.

        Returns:
            Tuple of (is_valid, error_message, adjusted_quantity)
        """
        if symbol not in self.symbol_filters:
            return False, f"Symbol {symbol} not found", quantity

        symbol_info = self.symbol_filters[symbol]
        filters = symbol_info['filters']

        if 'LOT_SIZE' not in filters:
            return True, "", quantity

        lot_filter = filters['LOT_SIZE']
        min_qty = float(lot_filter['minQty'])
        max_qty = float(lot_filter['maxQty'])
        step_size = float(lot_filter['stepSize'])

        # Check minimum quantity
        if quantity < min_qty:
            return False, f"Quantity {quantity} below minimum {min_qty}", quantity

        # Check maximum quantity
        if quantity > max_qty:
            return False, f"Quantity {quantity} above maximum {max_qty}", quantity

        # Adjust quantity to step size
        if step_size > 0:
            # Calculate precision from step size
            precision = len(str(step_size).split('.')[-1]) if '.' in str(step_size) else 0

            # Adjust quantity to nearest valid step
            adjusted_qty = math.floor(quantity / step_size) * step_size
            adjusted_qty = round(adjusted_qty, precision)

            if adjusted_qty < min_qty:
                return False, f"Adjusted quantity {adjusted_qty} below minimum {min_qty}", adjusted_qty

            return True, "", adjusted_qty

        return True, "", quantity

    def validate_price(self, symbol: str, price: float) -> Tuple[bool, str, float]:
        """
        Validate and adjust price according to symbol rules.

        Returns:
            Tuple of (is_valid, error_message, adjusted_price)
        """
        if symbol not in self.symbol_filters:
            return False, f"Symbol {symbol} not found", price

        symbol_info = self.symbol_filters[symbol]
        filters = symbol_info['filters']

        if 'PRICE_FILTER' not in filters:
            return True, "", price

        price_filter = filters['PRICE_FILTER']
        min_price = float(price_filter['minPrice'])
        max_price = float(price_filter['maxPrice'])
        tick_size = float(price_filter['tickSize'])

        # Check minimum price
        if price < min_price:
            return False, f"Price {price} below minimum {min_price}", price

        # Check maximum price
        if price > max_price:
            return False, f"Price {price} above maximum {max_price}", price

        # Adjust price to tick size
        if tick_size > 0:
            # Calculate precision from tick size
            precision = len(str(tick_size).split('.')[-1]) if '.' in str(tick_size) else 0

            # Adjust price to nearest valid tick
            adjusted_price = math.floor(price / tick_size) * tick_size
            adjusted_price = round(adjusted_price, precision)

            if adjusted_price < min_price:
                return False, f"Adjusted price {adjusted_price} below minimum {min_price}", adjusted_price

            return True, "", adjusted_price

        return True, "", price

    def validate_notional(self, symbol: str, quantity: float, price: float) -> Tuple[bool, str]:
        """
        Validate notional value (quantity * price) according to symbol rules.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if symbol not in self.symbol_filters:
            return False, f"Symbol {symbol} not found"

        symbol_info = self.symbol_filters[symbol]
        filters = symbol_info['filters']

        notional_value = quantity * price

        # Check MIN_NOTIONAL filter
        if 'MIN_NOTIONAL' in filters:
            min_notional_filter = filters['MIN_NOTIONAL']
            min_notional = float(min_notional_filter['minNotional'])

            if notional_value < min_notional:
                return False, f"Notional value {notional_value} below minimum {min_notional}"

        # Check NOTIONAL filter (newer version)
        if 'NOTIONAL' in filters:
            notional_filter = filters['NOTIONAL']
            min_notional = float(notional_filter['minNotional'])
            max_notional = float(notional_filter.get('maxNotional', float('inf')))

            if notional_value < min_notional:
                return False, f"Notional value {notional_value} below minimum {min_notional}"

            if notional_value > max_notional:
                return False, f"Notional value {notional_value} above maximum {max_notional}"

        return True, ""

    def get_supported_order_types(self, symbol: str) -> List[str]:
        """Get supported order types for a symbol."""
        if symbol not in self.symbol_filters:
            return []

        return self.symbol_filters[symbol]['orderTypes']

    def is_oco_allowed(self, symbol: str) -> bool:
        """Check if OCO orders are allowed for symbol."""
        if symbol not in self.symbol_filters:
            return False

        return self.symbol_filters[symbol]['ocoAllowed']


class BinanceCommissionCalculator:
    """Calculator for Binance trading commissions."""

    def __init__(self, maker_rate: float = 0.001, taker_rate: float = 0.001):
        self.maker_rate = maker_rate  # 0.1% default
        self.taker_rate = taker_rate  # 0.1% default

    def calculate_commission(self, quantity: float, price: float, is_maker: bool = False) -> float:
        """
        Calculate trading commission.

        Args:
            quantity: Order quantity
            price: Order price
            is_maker: True if maker order, False if taker order

        Returns:
            Commission amount in quote currency
        """
        notional_value = quantity * price
        rate = self.maker_rate if is_maker else self.taker_rate
        return notional_value * rate

    def calculate_bnb_discount_commission(self, base_commission: float, bnb_discount: float = 0.25) -> float:
        """
        Calculate commission with BNB discount.

        Args:
            base_commission: Base commission amount
            bnb_discount: BNB discount rate (default 25%)

        Returns:
            Discounted commission amount
        """
        return base_commission * (1 - bnb_discount)


class BinanceWebSocketProcessor:
    """Processor for Binance WebSocket messages."""

    @staticmethod
    def process_ticker_message(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process ticker message from WebSocket.

        Args:
            message: Raw WebSocket message

        Returns:
            Processed ticker data or None if invalid
        """
        try:
            if 's' not in message or 'c' not in message:
                return None

            return {
                'symbol': message['s'],
                'price': float(message['c']),
                'price_change': float(message.get('P', 0)),
                'price_change_percent': float(message.get('P', 0)),
                'volume': float(message.get('v', 0)),
                'quote_volume': float(message.get('q', 0)),
                'open_price': float(message.get('o', 0)),
                'high_price': float(message.get('h', 0)),
                'low_price': float(message.get('l', 0)),
                'bid_price': float(message.get('b', 0)),
                'ask_price': float(message.get('a', 0)),
                'timestamp': int(message.get('E', 0))
            }

        except (ValueError, KeyError) as e:
            _logger.warning("Error processing ticker message: %s", e)
            return None

    @staticmethod
    def process_trade_message(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process trade message from WebSocket.

        Args:
            message: Raw WebSocket message

        Returns:
            Processed trade data or None if invalid
        """
        try:
            if 's' not in message or 'p' not in message:
                return None

            return {
                'symbol': message['s'],
                'price': float(message['p']),
                'quantity': float(message['q']),
                'trade_time': int(message.get('T', 0)),
                'is_buyer_maker': message.get('m', False),
                'trade_id': int(message.get('t', 0))
            }

        except (ValueError, KeyError) as e:
            _logger.warning("Error processing trade message: %s", e)
            return None

    @staticmethod
    def process_depth_message(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process order book depth message from WebSocket.

        Args:
            message: Raw WebSocket message

        Returns:
            Processed depth data or None if invalid
        """
        try:
            if 's' not in message or 'b' not in message or 'a' not in message:
                return None

            return {
                'symbol': message['s'],
                'bids': [[float(bid[0]), float(bid[1])] for bid in message['b']],
                'asks': [[float(ask[0]), float(ask[1])] for ask in message['a']],
                'timestamp': int(message.get('E', 0))
            }

        except (ValueError, KeyError) as e:
            _logger.warning("Error processing depth message: %s", e)
            return None


class BinanceOrderFormatter:
    """Formatter for Binance order parameters."""

    def __init__(self, symbol_validator: BinanceSymbolValidator):
        self.validator = symbol_validator

    def format_quantity(self, symbol: str, quantity: float) -> str:
        """Format quantity according to symbol precision rules."""
        is_valid, error, adjusted_qty = self.validator.validate_quantity(symbol, quantity)

        if not is_valid:
            _logger.warning("Quantity validation failed: %s", error)
            return str(quantity)

        # Get symbol info for precision
        symbol_info = self.validator.get_symbol_info(symbol)
        if symbol_info and 'baseAssetPrecision' in symbol_info:
            precision = symbol_info['baseAssetPrecision']
            return f"{adjusted_qty:.{precision}f}".rstrip('0').rstrip('.')

        return str(adjusted_qty)

    def format_price(self, symbol: str, price: float) -> str:
        """Format price according to symbol precision rules."""
        is_valid, error, adjusted_price = self.validator.validate_price(symbol, price)

        if not is_valid:
            _logger.warning("Price validation failed: %s", error)
            return str(price)

        # Get symbol info for precision
        symbol_info = self.validator.get_symbol_info(symbol)
        if symbol_info and 'quoteAssetPrecision' in symbol_info:
            precision = symbol_info['quoteAssetPrecision']
            return f"{adjusted_price:.{precision}f}".rstrip('0').rstrip('.')

        return str(adjusted_price)

    def validate_and_format_order(self, symbol: str, quantity: float, price: Optional[float] = None) -> Dict[str, Any]:
        """
        Validate and format order parameters.

        Returns:
            Dictionary with validation results and formatted parameters
        """
        result = {
            'valid': True,
            'errors': [],
            'formatted_quantity': None,
            'formatted_price': None,
            'adjusted_quantity': quantity,
            'adjusted_price': price
        }

        # Validate and format quantity
        qty_valid, qty_error, adj_qty = self.validator.validate_quantity(symbol, quantity)
        if not qty_valid:
            result['valid'] = False
            result['errors'].append(qty_error)
        else:
            result['formatted_quantity'] = self.format_quantity(symbol, adj_qty)
            result['adjusted_quantity'] = adj_qty

        # Validate and format price (if provided)
        if price is not None:
            price_valid, price_error, adj_price = self.validator.validate_price(symbol, price)
            if not price_valid:
                result['valid'] = False
                result['errors'].append(price_error)
            else:
                result['formatted_price'] = self.format_price(symbol, adj_price)
                result['adjusted_price'] = adj_price

                # Validate notional value
                notional_valid, notional_error = self.validator.validate_notional(
                    symbol, adj_qty, adj_price
                )
                if not notional_valid:
                    result['valid'] = False
                    result['errors'].append(notional_error)

        return result


def create_binance_config_template(trading_mode: str = 'paper') -> Dict[str, Any]:
    """
    Create a Binance-specific configuration template.

    Args:
        trading_mode: 'paper' or 'live'

    Returns:
        Binance configuration template
    """
    base_config = {
        'type': 'binance',
        'trading_mode': trading_mode,
        'name': f'binance_{trading_mode}_broker',
        'cash': 10000.0 if trading_mode == 'paper' else 5000.0,

        # Binance-specific paper trading config
        'paper_trading_config': {
            'mode': 'realistic',
            'initial_balance': 10000.0,
            'commission_rate': 0.001,  # 0.1% (Binance standard)
            'slippage_model': 'linear',
            'base_slippage': 0.0005,   # 0.05%
            'latency_simulation': True,
            'min_latency_ms': 20,      # Binance typical latency
            'max_latency_ms': 150,
            'market_impact_enabled': True,
            'market_impact_factor': 0.0001,
            'realistic_fills': True,
            'partial_fill_probability': 0.15,  # Higher for crypto markets
            'reject_probability': 0.02,        # 2% rejection rate
            'enable_execution_quality': True
        },

        # Binance-specific settings
        'binance_config': {
            'enable_websocket': True,
            'websocket_symbols': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'],  # Default symbols
            'enable_bnb_discount': True,
            'bnb_discount_rate': 0.25,  # 25% discount with BNB
            'enable_margin_trading': False,
            'enable_futures_trading': False,
            'rate_limit_orders_per_second': 10,
            'rate_limit_weight_per_minute': 1200
        },

        'notifications': {
            'position_opened': True,
            'position_closed': True,
            'email_enabled': True,
            'telegram_enabled': True,
            'error_notifications': True,
            'binance_specific_notifications': {
                'api_errors': True,
                'rate_limit_warnings': True,
                'websocket_disconnections': True
            }
        },

        'risk_management': {
            'max_position_size': 1000.0 if trading_mode == 'paper' else 500.0,
            'max_daily_loss': 500.0 if trading_mode == 'paper' else 250.0,
            'max_portfolio_risk': 0.02 if trading_mode == 'paper' else 0.01,
            'position_sizing_method': 'fixed_dollar',
            'stop_loss_enabled': True,
            'stop_loss_percentage': 0.02,
            'take_profit_enabled': True,
            'take_profit_percentage': 0.04,
            'binance_specific_limits': {
                'max_orders_per_symbol': 200,
                'max_open_orders': 1000,
                'min_notional_usdt': 10.0
            }
        }
    }

    # Live trading specific additions
    if trading_mode == 'live':
        base_config['live_trading_confirmed'] = False
        base_config['_WARNING'] = 'LIVE TRADING MODE - REAL CRYPTOCURRENCY WILL BE USED'

        # More conservative settings for live trading
        base_config['binance_config']['rate_limit_orders_per_second'] = 5
        base_config['risk_management']['max_orders_per_symbol'] = 50

    return base_config


def get_popular_binance_symbols() -> List[str]:
    """Get list of popular Binance trading symbols."""
    return [
        # Major cryptocurrencies
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT',
        'SOLUSDT', 'DOTUSDT', 'DOGEUSDT', 'AVAXUSDT', 'SHIBUSDT',

        # DeFi tokens
        'UNIUSDT', 'LINKUSDT', 'LTCUSDT', 'BCHUSDT', 'FILUSDT',

        # Stablecoins pairs
        'BTCBUSD', 'ETHBUSD', 'BNBBUSD',

        # Popular altcoins
        'MATICUSDT', 'ALGOUSDT', 'VETUSDT', 'ICPUSDT', 'THETAUSDT'
    ]


def calculate_binance_fees(quantity: float, price: float, is_maker: bool = False,
                          use_bnb: bool = False, vip_level: int = 0) -> Dict[str, float]:
    """
    Calculate Binance trading fees based on VIP level and BNB usage.

    Args:
        quantity: Order quantity
        price: Order price
        is_maker: True for maker orders, False for taker orders
        use_bnb: True if using BNB for fee discount
        vip_level: VIP level (0-9)

    Returns:
        Dictionary with fee information
    """
    # Binance fee structure (as of 2024)
    fee_structure = {
        0: {'maker': 0.001, 'taker': 0.001},  # VIP 0: 0.1%/0.1%
        1: {'maker': 0.0009, 'taker': 0.001},  # VIP 1: 0.09%/0.1%
        2: {'maker': 0.0008, 'taker': 0.001},  # VIP 2: 0.08%/0.1%
        3: {'maker': 0.0007, 'taker': 0.0009},  # VIP 3: 0.07%/0.09%
        4: {'maker': 0.0007, 'taker': 0.0008},  # VIP 4: 0.07%/0.08%
        5: {'maker': 0.0006, 'taker': 0.0007},  # VIP 5: 0.06%/0.07%
        6: {'maker': 0.0005, 'taker': 0.0006},  # VIP 6: 0.05%/0.06%
        7: {'maker': 0.0004, 'taker': 0.0005},  # VIP 7: 0.04%/0.05%
        8: {'maker': 0.0003, 'taker': 0.0004},  # VIP 8: 0.03%/0.04%
        9: {'maker': 0.0002, 'taker': 0.0003},  # VIP 9: 0.02%/0.03%
    }

    vip_level = max(0, min(9, vip_level))  # Clamp to valid range
    rates = fee_structure[vip_level]

    notional_value = quantity * price
    base_rate = rates['maker'] if is_maker else rates['taker']
    base_fee = notional_value * base_rate

    # Apply BNB discount (25% discount)
    final_fee = base_fee * 0.75 if use_bnb else base_fee

    return {
        'notional_value': notional_value,
        'base_rate': base_rate,
        'base_fee': base_fee,
        'bnb_discount': use_bnb,
        'final_fee': final_fee,
        'savings': base_fee - final_fee if use_bnb else 0.0,
        'vip_level': vip_level,
        'order_type': 'maker' if is_maker else 'taker'
    }