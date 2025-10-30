#!/usr/bin/env python3
"""
IBKR Utilities Module
--------------------

This module provides utility functions and configurations specific to Interactive
Brokers (IBKR) trading operations, including contract management, margin calculations,
and market data processing.

Features:
- IBKR contract creation and validation
- Multi-asset support (stocks, options, futures, forex)
- Margin calculation and requirements
- Commission calculation with tiered structure
- Market data utilities and processing
- Order validation and formatting

Functions:
- create_ibkr_contract: Create IBKR contracts for different asset classes
- calculate_ibkr_commission: Calculate IBKR trading commission
- validate_ibkr_symbol: Validate trading symbol format
- format_ibkr_order: Format order parameters for IBKR API
- calculate_margin_requirements: Calculate margin requirements
"""

import math
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal, ROUND_DOWN
from datetime import datetime, timezone, timedelta
from ib_insync import Stock, Option, Future, Forex, Contract

from src.notification.logger import setup_logger
_logger = setup_logger(__name__)


class IBKRContractManager:
    """Manager for IBKR contract creation and validation."""

    def __init__(self):
        self.contract_cache: Dict[str, Contract] = {}

    def create_stock_contract(self, symbol: str, exchange: str = 'SMART',
                            currency: str = 'USD') -> Stock:
        """
        Create a stock contract.

        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'MSFT')
            exchange: Exchange (default: 'SMART')
            currency: Currency (default: 'USD')

        Returns:
            Stock contract
        """
        return Stock(symbol, exchange, currency)

    def create_option_contract(self, symbol: str, expiry: str, strike: float,
                             right: str, exchange: str = 'SMART',
                             currency: str = 'USD') -> Option:
        """
        Create an option contract.

        Args:
            symbol: Underlying symbol
            expiry: Expiry date (YYYYMMDD format)
            strike: Strike price
            right: 'C' for call, 'P' for put
            exchange: Exchange (default: 'SMART')
            currency: Currency (default: 'USD')

        Returns:
            Option contract
        """
        return Option(symbol, expiry, strike, right, exchange, currency)

    def create_future_contract(self, symbol: str, expiry: str,
                             exchange: str, currency: str = 'USD') -> Future:
        """
        Create a futures contract.

        Args:
            symbol: Future symbol
            expiry: Expiry date (YYYYMM format)
            exchange: Exchange (e.g., 'GLOBEX', 'NYMEX')
            currency: Currency (default: 'USD')

        Returns:
            Future contract
        """
        return Future(symbol, expiry, exchange, currency=currency)

    def create_forex_contract(self, base_currency: str, quote_currency: str = 'USD') -> Forex:
        """
        Create a forex contract.

        Args:
            base_currency: Base currency (e.g., 'EUR', 'GBP')
            quote_currency: Quote currency (default: 'USD')

        Returns:
            Forex contract
        """
        return Forex(base_currency + quote_currency)

    def get_contract(self, symbol: str, asset_class: str = 'STK', **kwargs) -> Optional[Contract]:
        """
        Get or create a contract with caching.

        Args:
            symbol: Trading symbol
            asset_class: Asset class ('STK', 'OPT', 'FUT', 'CASH')
            **kwargs: Additional parameters for contract creation

        Returns:
            Contract object or None if creation fails
        """
        cache_key = f"{symbol}_{asset_class}_{hash(str(sorted(kwargs.items())))}"

        if cache_key in self.contract_cache:
            return self.contract_cache[cache_key]

        try:
            if asset_class == 'STK':
                contract = self.create_stock_contract(
                    symbol,
                    kwargs.get('exchange', 'SMART'),
                    kwargs.get('currency', 'USD')
                )
            elif asset_class == 'OPT':
                contract = self.create_option_contract(
                    symbol,
                    kwargs.get('expiry'),
                    kwargs.get('strike'),
                    kwargs.get('right'),
                    kwargs.get('exchange', 'SMART'),
                    kwargs.get('currency', 'USD')
                )
            elif asset_class == 'FUT':
                contract = self.create_future_contract(
                    symbol,
                    kwargs.get('expiry'),
                    kwargs.get('exchange'),
                    kwargs.get('currency', 'USD')
                )
            elif asset_class == 'CASH':
                base_currency = symbol[:3] if len(symbol) >= 6 else symbol
                quote_currency = symbol[3:] if len(symbol) >= 6 else 'USD'
                contract = self.create_forex_contract(base_currency, quote_currency)
            else:
                _logger.warning("Unsupported asset class: %s", asset_class)
                return None

            self.contract_cache[cache_key] = contract
            return contract

        except Exception as e:
            _logger.exception("Error creating contract for %s:", symbol)
            return None

    def validate_symbol_format(self, symbol: str, asset_class: str) -> Tuple[bool, str]:
        """
        Validate symbol format for different asset classes.

        Args:
            symbol: Trading symbol
            asset_class: Asset class

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not symbol or not isinstance(symbol, str):
            return False, "Symbol must be a non-empty string"

        symbol = symbol.upper().strip()

        if asset_class == 'STK':
            # Stock symbols: 1-5 characters, alphanumeric
            if not (1 <= len(symbol) <= 5) or not symbol.isalnum():
                return False, "Stock symbol must be 1-5 alphanumeric characters"

        elif asset_class == 'CASH':
            # Forex pairs: 6 characters (EURUSD, GBPUSD, etc.)
            if len(symbol) != 6 or not symbol.isalpha():
                return False, "Forex symbol must be 6 alphabetic characters (e.g., EURUSD)"

        elif asset_class in ['OPT', 'FUT']:
            # Options and futures: more flexible validation
            if len(symbol) < 1 or len(symbol) > 10:
                return False, f"{asset_class} symbol must be 1-10 characters"

        return True, ""


class IBKRCommissionCalculator:
    """Calculator for IBKR trading commissions with tiered structure."""

    def __init__(self):
        # IBKR US stock commission structure (as of 2024)
        self.us_stock_tiers = [
            {'min_shares': 0, 'max_shares': 300, 'rate': 0.005, 'min_commission': 1.0},
            {'min_shares': 301, 'max_shares': float('inf'), 'rate': 0.005, 'min_commission': 1.0}
        ]

        # IBKR option commission structure
        self.option_commission = {
            'base_rate': 0.70,  # $0.70 per contract
            'min_commission': 1.0,
            'max_commission_pct': 0.12  # 12% of trade value
        }

        # IBKR forex commission (spread-based)
        self.forex_spreads = {
            'EURUSD': 0.2,  # 0.2 pips
            'GBPUSD': 0.5,  # 0.5 pips
            'USDJPY': 0.2,  # 0.2 pips
            'USDCHF': 0.5,  # 0.5 pips
            'AUDUSD': 0.5,  # 0.5 pips
            'USDCAD': 0.5,  # 0.5 pips
            'default': 1.0  # 1.0 pip for other pairs
        }

    def calculate_stock_commission(self, quantity: int, price: float) -> Dict[str, float]:
        """
        Calculate commission for US stock trades.

        Args:
            quantity: Number of shares
            price: Price per share

        Returns:
            Dictionary with commission details
        """
        # Find applicable tier
        tier = None
        for t in self.us_stock_tiers:
            if t['min_shares'] <= quantity <= t['max_shares']:
                tier = t
                break

        if not tier:
            tier = self.us_stock_tiers[-1]  # Use highest tier

        # Calculate commission
        commission = max(quantity * tier['rate'], tier['min_commission'])

        # IBKR maximum commission is 1% of trade value
        max_commission = (quantity * price) * 0.01
        commission = min(commission, max_commission)

        return {
            'commission': commission,
            'rate_per_share': tier['rate'],
            'min_commission': tier['min_commission'],
            'max_commission': max_commission,
            'tier_used': f"{tier['min_shares']}-{tier['max_shares']} shares"
        }

    def calculate_option_commission(self, contracts: int, premium: float) -> Dict[str, float]:
        """
        Calculate commission for option trades.

        Args:
            contracts: Number of option contracts
            premium: Premium per contract

        Returns:
            Dictionary with commission details
        """
        base_commission = contracts * self.option_commission['base_rate']
        trade_value = contracts * premium * 100  # Options are per 100 shares

        # Apply minimum commission
        commission = max(base_commission, self.option_commission['min_commission'])

        # Apply maximum commission (percentage of trade value)
        max_commission = trade_value * self.option_commission['max_commission_pct']
        commission = min(commission, max_commission)

        return {
            'commission': commission,
            'base_commission': base_commission,
            'min_commission': self.option_commission['min_commission'],
            'max_commission': max_commission,
            'rate_per_contract': self.option_commission['base_rate']
        }

    def calculate_forex_cost(self, symbol: str, quantity: float,
                           is_base_currency: bool = True) -> Dict[str, float]:
        """
        Calculate forex trading cost (spread-based).

        Args:
            symbol: Forex pair (e.g., 'EURUSD')
            quantity: Trade quantity
            is_base_currency: True if quantity is in base currency

        Returns:
            Dictionary with cost details
        """
        spread_pips = self.forex_spreads.get(symbol, self.forex_spreads['default'])

        # Convert pips to decimal (for most pairs, 1 pip = 0.0001)
        pip_value = 0.0001 if 'JPY' not in symbol else 0.01
        spread_decimal = spread_pips * pip_value

        # Calculate cost
        if is_base_currency:
            cost = quantity * spread_decimal
        else:
            cost = (quantity / 100000) * spread_decimal  # Assuming standard lot size

        return {
            'cost': cost,
            'spread_pips': spread_pips,
            'spread_decimal': spread_decimal,
            'pip_value': pip_value
        }


class IBKRMarginCalculator:
    """Calculator for IBKR margin requirements."""

    def __init__(self):
        # Standard margin requirements (can be higher for specific stocks)
        self.stock_margin_rates = {
            'initial': 0.50,  # 50% initial margin
            'maintenance': 0.25,  # 25% maintenance margin
            'day_trading': 0.25  # 25% day trading margin
        }

        # Option margin requirements (simplified)
        self.option_margin_rates = {
            'long_option': 1.0,  # 100% premium for long options
            'short_call_covered': 0.0,  # No margin for covered calls
            'short_call_naked': 0.20,  # 20% + premium for naked calls
            'short_put': 0.20  # 20% + premium for short puts
        }

    def calculate_stock_margin(self, symbol: str, quantity: int, price: float,
                             margin_type: str = 'initial') -> Dict[str, float]:
        """
        Calculate margin requirement for stock position.

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            price: Price per share
            margin_type: 'initial', 'maintenance', or 'day_trading'

        Returns:
            Dictionary with margin details
        """
        if margin_type not in self.stock_margin_rates:
            margin_type = 'initial'

        position_value = abs(quantity) * price
        margin_rate = self.stock_margin_rates[margin_type]

        # Special cases for certain stocks (would need real-time data)
        if symbol in ['GME', 'AMC']:  # Example high-volatility stocks
            margin_rate = min(margin_rate * 2, 1.0)  # Double margin, max 100%

        margin_requirement = position_value * margin_rate

        return {
            'margin_requirement': margin_requirement,
            'position_value': position_value,
            'margin_rate': margin_rate,
            'margin_type': margin_type,
            'buying_power_used': margin_requirement
        }

    def calculate_option_margin(self, option_type: str, contracts: int,
                              premium: float, underlying_price: float,
                              strike: float) -> Dict[str, float]:
        """
        Calculate margin requirement for option position.

        Args:
            option_type: 'long_call', 'long_put', 'short_call', 'short_put'
            contracts: Number of contracts
            premium: Premium per contract
            underlying_price: Current underlying price
            strike: Strike price

        Returns:
            Dictionary with margin details
        """
        contract_value = contracts * premium * 100  # Options are per 100 shares

        if option_type.startswith('long_'):
            # Long options: pay full premium, no margin requirement
            margin_requirement = 0.0
            buying_power_used = contract_value

        elif option_type == 'short_call':
            # Short call: 20% of underlying + premium - out-of-money amount
            underlying_value = contracts * underlying_price * 100
            margin_base = underlying_value * 0.20
            out_of_money = max(0, strike - underlying_price) * contracts * 100
            margin_requirement = max(margin_base + contract_value - out_of_money,
                                   contract_value + (underlying_value * 0.10))
            buying_power_used = margin_requirement

        elif option_type == 'short_put':
            # Short put: 20% of underlying + premium - out-of-money amount
            underlying_value = contracts * underlying_price * 100
            margin_base = underlying_value * 0.20
            out_of_money = max(0, underlying_price - strike) * contracts * 100
            margin_requirement = max(margin_base + contract_value - out_of_money,
                                   contract_value + (strike * contracts * 100 * 0.10))
            buying_power_used = margin_requirement

        else:
            margin_requirement = 0.0
            buying_power_used = contract_value

        return {
            'margin_requirement': margin_requirement,
            'contract_value': contract_value,
            'buying_power_used': buying_power_used,
            'option_type': option_type
        }


class IBKROrderValidator:
    """Validator for IBKR order parameters."""

    def __init__(self):
        self.contract_manager = IBKRContractManager()

    def validate_stock_order(self, symbol: str, quantity: int, price: Optional[float] = None) -> Tuple[bool, str]:
        """
        Validate stock order parameters.

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            price: Price per share (for limit orders)

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Validate symbol format
        is_valid, error = self.contract_manager.validate_symbol_format(symbol, 'STK')
        if not is_valid:
            return False, error

        # Validate quantity
        if quantity <= 0:
            return False, "Quantity must be positive"

        if quantity != int(quantity):
            return False, "Stock quantity must be a whole number"

        # Validate price (if provided)
        if price is not None:
            if price <= 0:
                return False, "Price must be positive"

            if price > 1000000:  # Reasonable upper limit
                return False, "Price exceeds reasonable limit ($1,000,000)"

        return True, ""

    def validate_option_order(self, symbol: str, expiry: str, strike: float,
                            right: str, contracts: int) -> Tuple[bool, str]:
        """
        Validate option order parameters.

        Args:
            symbol: Underlying symbol
            expiry: Expiry date (YYYYMMDD)
            strike: Strike price
            right: 'C' or 'P'
            contracts: Number of contracts

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Validate underlying symbol
        is_valid, error = self.contract_manager.validate_symbol_format(symbol, 'STK')
        if not is_valid:
            return False, f"Invalid underlying symbol: {error}"

        # Validate expiry format
        try:
            expiry_date = datetime.strptime(expiry, '%Y%m%d')
            if expiry_date <= datetime.now():
                return False, "Option expiry must be in the future"
        except ValueError:
            return False, "Invalid expiry format (use YYYYMMDD)"

        # Validate strike price
        if strike <= 0:
            return False, "Strike price must be positive"

        # Validate right
        if right not in ['C', 'P']:
            return False, "Option right must be 'C' (call) or 'P' (put)"

        # Validate contracts
        if contracts <= 0:
            return False, "Number of contracts must be positive"

        if contracts != int(contracts):
            return False, "Number of contracts must be a whole number"

        return True, ""

    def validate_forex_order(self, symbol: str, quantity: float) -> Tuple[bool, str]:
        """
        Validate forex order parameters.

        Args:
            symbol: Forex pair (e.g., 'EURUSD')
            quantity: Trade quantity

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Validate symbol format
        is_valid, error = self.contract_manager.validate_symbol_format(symbol, 'CASH')
        if not is_valid:
            return False, error

        # Validate quantity
        if quantity <= 0:
            return False, "Quantity must be positive"

        # IBKR minimum forex trade size is typically 25,000 units
        if quantity < 25000:
            return False, "Minimum forex trade size is 25,000 units"

        return True, ""


def create_ibkr_config_template(trading_mode: str = 'paper') -> Dict[str, Any]:
    """
    Create an IBKR-specific configuration template.

    Args:
        trading_mode: 'paper' or 'live'

    Returns:
        IBKR configuration template
    """
    base_config = {
        'type': 'ibkr',
        'trading_mode': trading_mode,
        'name': f'ibkr_{trading_mode}_broker',
        'cash': 25000.0 if trading_mode == 'paper' else 50000.0,

        # IBKR connection settings
        'connection': {
            'host': '127.0.0.1',
            'port': 7497 if trading_mode == 'paper' else 4001,
            'client_id': 1,
            'timeout': 30
        },

        # IBKR-specific paper trading config
        'paper_trading_config': {
            'mode': 'advanced',
            'initial_balance': 25000.0,
            'commission_rate': 0.0005,  # 0.05% (IBKR typical)
            'slippage_model': 'sqrt',
            'base_slippage': 0.0003,    # 0.03%
            'latency_simulation': True,
            'min_latency_ms': 5,        # IBKR typical latency
            'max_latency_ms': 50,
            'market_impact_enabled': True,
            'market_impact_factor': 0.00005,
            'realistic_fills': True,
            'partial_fill_probability': 0.1,
            'reject_probability': 0.005,  # Lower rejection rate
            'enable_execution_quality': True
        },

        # IBKR-specific settings
        'ibkr_config': {
            'enable_market_data': True,
            'market_data_symbols': ['AAPL', 'MSFT', 'GOOGL', 'TSLA'],  # Default symbols
            'enable_options_trading': True,
            'enable_futures_trading': False,
            'enable_forex_trading': True,
            'enable_margin_trading': True,
            'auto_subscribe_market_data': True,
            'market_data_update_interval': 1,  # seconds
            'order_management': {
                'enable_bracket_orders': True,
                'enable_trailing_stops': True,
                'default_time_in_force': 'GTC'
            }
        },

        'notifications': {
            'position_opened': True,
            'position_closed': True,
            'email_enabled': True,
            'telegram_enabled': True,
            'error_notifications': True,
            'ibkr_specific_notifications': {
                'connection_status': True,
                'margin_calls': True,
                'order_rejections': True,
                'market_data_issues': True
            }
        },

        'risk_management': {
            'max_position_size': 2500.0 if trading_mode == 'paper' else 5000.0,
            'max_daily_loss': 1000.0 if trading_mode == 'paper' else 2000.0,
            'max_portfolio_risk': 0.015 if trading_mode == 'paper' else 0.008,
            'position_sizing_method': 'volatility_adjusted',
            'stop_loss_enabled': True,
            'stop_loss_percentage': 0.015,
            'take_profit_enabled': True,
            'take_profit_percentage': 0.03,
            'ibkr_specific_limits': {
                'max_margin_usage': 0.5,  # 50% of available margin
                'day_trading_buying_power_limit': 0.25,
                'enable_pattern_day_trader_protection': True
            }
        }
    }

    # Live trading specific additions
    if trading_mode == 'live':
        base_config['live_trading_confirmed'] = False
        base_config['_WARNING'] = 'LIVE TRADING MODE - REAL MONEY WILL BE USED'

        # More conservative settings for live trading
        base_config['ibkr_config']['market_data_update_interval'] = 0.5  # Faster updates
        base_config['risk_management']['max_margin_usage'] = 0.3  # More conservative

    return base_config


def get_popular_ibkr_symbols() -> Dict[str, List[str]]:
    """Get list of popular IBKR trading symbols by asset class."""
    return {
        'stocks': [
            # Large cap US stocks
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK.B',
            'JNJ', 'V', 'WMT', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'ADBE',

            # Popular ETFs
            'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'GLD', 'SLV'
        ],

        'forex': [
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD',
            'NZDUSD', 'EURGBP', 'EURJPY', 'GBPJPY', 'CHFJPY', 'EURCHF'
        ],

        'futures': [
            'ES',   # E-mini S&P 500
            'NQ',   # E-mini NASDAQ
            'YM',   # E-mini Dow
            'RTY',  # E-mini Russell 2000
            'CL',   # Crude Oil
            'GC',   # Gold
            'SI',   # Silver
            'ZN',   # 10-Year Treasury Note
        ],

        'options_underlyings': [
            'SPY', 'QQQ', 'AAPL', 'TSLA', 'MSFT', 'AMZN', 'GOOGL', 'META',
            'NVDA', 'AMD', 'NFLX', 'CRM', 'BABA', 'IWM', 'GLD', 'SLV'
        ]
    }


def calculate_ibkr_fees_comprehensive(asset_class: str, **kwargs) -> Dict[str, float]:
    """
    Calculate comprehensive IBKR fees for different asset classes.

    Args:
        asset_class: 'STK', 'OPT', 'FUT', 'CASH'
        **kwargs: Asset-specific parameters

    Returns:
        Dictionary with comprehensive fee information
    """
    calculator = IBKRCommissionCalculator()

    if asset_class == 'STK':
        quantity = kwargs.get('quantity', 100)
        price = kwargs.get('price', 100.0)
        return calculator.calculate_stock_commission(quantity, price)

    elif asset_class == 'OPT':
        contracts = kwargs.get('contracts', 1)
        premium = kwargs.get('premium', 5.0)
        return calculator.calculate_option_commission(contracts, premium)

    elif asset_class == 'CASH':
        symbol = kwargs.get('symbol', 'EURUSD')
        quantity = kwargs.get('quantity', 100000.0)
        return calculator.calculate_forex_cost(symbol, quantity)

    else:
        return {'error': f'Unsupported asset class: {asset_class}'}


def validate_ibkr_trading_hours(symbol: str, asset_class: str,
                               timestamp: datetime = None) -> Tuple[bool, str]:
    """
    Validate if trading is allowed for a symbol at a given time.

    Args:
        symbol: Trading symbol
        asset_class: Asset class
        timestamp: Time to check (default: now)

    Returns:
        Tuple of (is_trading_allowed, message)
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    # Simplified trading hours validation
    # In a real implementation, this would use IBKR's contract details API

    weekday = timestamp.weekday()  # 0 = Monday, 6 = Sunday
    hour = timestamp.hour

    if asset_class == 'STK':
        # US stock market hours (simplified)
        if weekday >= 5:  # Weekend
            return False, "US stock market is closed on weekends"

        if not (14 <= hour < 21):  # 9:30 AM - 4:00 PM EST in UTC
            return False, "US stock market is closed (trading hours: 9:30 AM - 4:00 PM EST)"

    elif asset_class == 'CASH':
        # Forex market (24/5)
        if weekday == 6 or (weekday == 0 and hour < 22):  # Sunday before 5 PM EST
            return False, "Forex market is closed"

        if weekday == 4 and hour >= 21:  # Friday after 4 PM EST
            return False, "Forex market is closed"

    elif asset_class in ['OPT', 'FUT']:
        # Options and futures have various hours
        if weekday >= 5:
            return False, f"{asset_class} market is closed on weekends"

    return True, "Trading is allowed"