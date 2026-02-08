#!/usr/bin/env python3
"""
Enhanced Broker Factory Module
-----------------------------

Factory module for creating broker instances with seamless paper-to-live trading
mode switching based on configuration.

Features:
- Automatic paper/live mode selection based on trading_mode configuration
- Credential and connection parameter selection based on mode
- Support for Binance (dual account), IBKR (dual port), and Mock brokers
- Comprehensive broker configuration validation
- Broker capability detection and health monitoring

Functions:
- get_broker: Main factory function for broker creation
- validate_broker_config: Configuration validation
- get_broker_capabilities: Broker capability detection
"""

from typing import Any, Dict, List

from src.trading.broker.binance_broker import BinanceBroker
from src.trading.broker.ibkr_broker import IBKRBroker
from src.trading.broker.mock_broker import MockBroker

from config.donotshare.donotshare import (
    BINANCE_KEY, BINANCE_PAPER_KEY, BINANCE_PAPER_SECRET, BINANCE_SECRET,
    IBKR_CLIENT_ID, IBKR_HOST, IBKR_PORT, IBKR_PAPER_PORT
)

from src.notification.logger import setup_logger
_logger = setup_logger(__name__)


class BrokerConfigurationError(Exception):
    """Exception raised for broker configuration errors."""
    pass


class LiveTradingValidator:
    """Validator for live trading configuration to ensure safety."""

    @staticmethod
    def validate_live_trading_config(config: Dict[str, Any]) -> None:
        """
        Validate configuration before enabling live trading.

        Args:
            config: Broker configuration dictionary

        Raises:
            BrokerConfigurationError: If configuration is invalid for live trading
        """
        if config.get('trading_mode') != 'live':
            return

        # Require explicit confirmation for live trading
        if not config.get('live_trading_confirmed', False):
            raise BrokerConfigurationError(
                "Live trading requires explicit confirmation. "
                "Set 'live_trading_confirmed': true in configuration."
            )

        # Validate risk management configuration
        risk_config = config.get('risk_management', {})
        if not risk_config:
            raise BrokerConfigurationError(
                "Live trading requires risk management configuration"
            )

        required_risk_params = ['max_position_size', 'max_daily_loss']
        missing_params = [param for param in required_risk_params if param not in risk_config]
        if missing_params:
            raise BrokerConfigurationError(
                f"Live trading requires risk management parameters: {missing_params}"
            )

        # Validate broker credentials are available
        broker_type = config.get("type", "").lower()
        if broker_type == "binance" and (not BINANCE_KEY or not BINANCE_SECRET):
            raise BrokerConfigurationError("Live Binance trading requires valid API credentials")
        elif broker_type == "ibkr":
            # IBKR doesn't require API keys in this implementation (connected via Gateway/TWS)
            pass

        _logger.warning("LIVE TRADING MODE ENABLED - Real money will be used!")


def validate_broker_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize broker configuration.

    Args:
        config: Raw broker configuration

    Returns:
        Validated and normalized configuration

    Raises:
        BrokerConfigurationError: If configuration is invalid
    """
    # Set defaults
    normalized_config = {
        'type': 'mock',
        'trading_mode': 'paper',
        'cash': 10000.0,
        'paper_trading_config': {
            'mode': 'realistic',
            'initial_balance': 10000.0,
            'commission_rate': 0.001,
            'slippage_model': 'linear',
            'base_slippage': 0.0005,
            'latency_simulation': True,
            'min_latency_ms': 10,
            'max_latency_ms': 100,
            'market_impact_enabled': True,
            'market_impact_factor': 0.0001,
            'realistic_fills': True,
            'partial_fill_probability': 0.1,
            'reject_probability': 0.01,
            'enable_execution_quality': True
        },
        'notifications': {
            'position_opened': True,
            'position_closed': True,
            'email_enabled': True,
            'telegram_enabled': True,
            'error_notifications': True
        }
    }

    # Update with provided config
    normalized_config.update(config)

    # Validate broker type
    supported_types = ['binance', 'ibkr', 'mock', 'file_broker']
    broker_type = normalized_config.get('type', '').lower()
    if broker_type not in supported_types:
        raise BrokerConfigurationError(f"Unsupported broker type: {broker_type}. Supported: {supported_types}")

    # Validate trading mode
    supported_modes = ['paper', 'live']
    trading_mode = normalized_config.get('trading_mode', '').lower()
    if trading_mode not in supported_modes:
        raise BrokerConfigurationError(f"Unsupported trading mode: {trading_mode}. Supported: {supported_modes}")

    # Set paper_trading flag based on mode
    normalized_config['paper_trading'] = (trading_mode == 'paper')

    # Validate live trading configuration
    LiveTradingValidator.validate_live_trading_config(normalized_config)

    return normalized_config


def get_broker_credentials(broker_type: str, trading_mode: str) -> Dict[str, Any]:
    """
    Get appropriate credentials and connection parameters based on broker type and trading mode.

    Args:
        broker_type: Type of broker ('binance', 'ibkr', 'mock')
        trading_mode: Trading mode ('paper', 'live')

    Returns:
        Dictionary with credentials and connection parameters
    """
    credentials = {}

    if broker_type == 'binance':
        if trading_mode == 'paper':
            credentials = {
                'api_key': BINANCE_PAPER_KEY,
                'api_secret': BINANCE_PAPER_SECRET,
                'base_url': 'https://testnet.binance.vision',
                'testnet': True
            }
        else:  # live
            credentials = {
                'api_key': BINANCE_KEY,
                'api_secret': BINANCE_SECRET,
                'base_url': 'https://api.binance.com',
                'testnet': False
            }

    elif broker_type == 'ibkr':
        if trading_mode == 'paper':
            credentials = {
                'host': IBKR_HOST,
                'port': int(IBKR_PAPER_PORT) if IBKR_PAPER_PORT else 4002,
                'client_id': int(IBKR_CLIENT_ID) if IBKR_CLIENT_ID else 1,
                'paper_trading': True
            }
        else:  # live
            credentials = {
                'host': IBKR_HOST,
                'port': int(IBKR_PORT) if IBKR_PORT else 4001,
                'client_id': int(IBKR_CLIENT_ID) if IBKR_CLIENT_ID else 1,
                'paper_trading': False
            }

    return credentials


def get_broker(config: Dict[str, Any]):
    """
    Enhanced factory function to instantiate the correct broker based on configuration.

    Supports seamless paper-to-live trading mode switching via configuration.

    Args:
        config: Broker configuration dictionary with the following structure:
        {
            'type': 'binance' | 'ibkr' | 'mock',
            'trading_mode': 'paper' | 'live',
            'cash': float,  # Initial balance
            'live_trading_confirmed': bool,  # Required for live trading
            'paper_trading_config': {...},  # Paper trading simulation config
            'risk_management': {...},  # Risk management config (required for live)
            'notifications': {...}  # Notification settings
        }

    Returns:
        Configured broker instance

    Raises:
        BrokerConfigurationError: If configuration is invalid
    """
    try:
        # Validate and normalize configuration
        validated_config = validate_broker_config(config)

        broker_type = validated_config['type'].lower()
        trading_mode = validated_config['trading_mode'].lower()

        # Get appropriate credentials
        credentials = get_broker_credentials(broker_type, trading_mode)

        # Merge credentials into config
        broker_config = {**validated_config, **credentials}

        _logger.info("Creating %s broker in %s mode", broker_type, trading_mode)

        # Create broker instance based on type and mode
        if broker_type == 'binance':
            # Enhanced Binance broker handles both paper and live modes
            return BinanceBroker(
                api_key=credentials['api_key'],
                api_secret=credentials['api_secret'],
                cash=broker_config.get('cash', 10000.0),
                config=broker_config
            )

        elif broker_type == 'ibkr':
            # Enhanced IBKR broker handles both paper and live modes internally
            return IBKRBroker(
                host=credentials['host'],
                port=credentials['port'],
                client_id=credentials['client_id'],
                cash=broker_config.get('cash', 25000.0),
                config=broker_config
            )

        elif broker_type == 'mock' or broker_type == 'file_broker':
            return MockBroker(
                cash=broker_config.get('cash', 10000.0),
                config=broker_config
            )

        else:
            raise BrokerConfigurationError(f"Unsupported broker type: {broker_type}")

    except Exception as e:
        _logger.exception("Error creating broker:")
        raise BrokerConfigurationError(f"Failed to create broker: {e}")


def get_broker_capabilities(broker_type: str) -> Dict[str, Any]:
    """
    Get capabilities and features supported by a broker type.

    Args:
        broker_type: Type of broker ('binance', 'ibkr', 'mock')

    Returns:
        Dictionary with broker capabilities
    """
    capabilities = {
        'binance': {
            'paper_trading': True,
            'live_trading': True,
            'order_types': ['market', 'limit', 'stop', 'stop_limit', 'oco'],
            'asset_classes': ['crypto'],
            'markets': ['spot', 'futures'],
            'real_time_data': True,
            'historical_data': True,
            'notifications': True,
            'dual_mode_switching': True
        },
        'ibkr': {
            'paper_trading': True,
            'live_trading': True,
            'order_types': ['market', 'limit', 'stop', 'stop_limit', 'trailing_stop', 'bracket'],
            'asset_classes': ['stocks', 'options', 'futures', 'forex', 'bonds'],
            'markets': ['global'],
            'real_time_data': True,
            'historical_data': True,
            'notifications': True,
            'dual_mode_switching': True,
            'margin_trading': True
        },
        'mock': {
            'paper_trading': True,
            'live_trading': False,
            'order_types': ['market', 'limit'],
            'asset_classes': ['any'],
            'markets': ['simulated'],
            'real_time_data': False,
            'historical_data': False,
            'notifications': True,
            'dual_mode_switching': False
        }
    }

    return capabilities.get(broker_type.lower(), {})


def list_available_brokers() -> List[Dict[str, Any]]:
    """
    List all available brokers with their capabilities.

    Returns:
        List of broker information dictionaries
    """
    brokers = []

    for broker_type in ['binance', 'ibkr', 'mock']:
        capabilities = get_broker_capabilities(broker_type)

        # Check if credentials are available
        credentials_available = True
        if broker_type == 'binance':
            credentials_available = bool(BINANCE_KEY and BINANCE_SECRET and BINANCE_PAPER_KEY and BINANCE_PAPER_SECRET)
        elif broker_type == 'ibkr':
            credentials_available = bool(IBKR_HOST and IBKR_PORT and IBKR_CLIENT_ID)

        brokers.append({
            'type': broker_type,
            'name': broker_type.upper(),
            'capabilities': capabilities,
            'credentials_available': credentials_available,
            'recommended_for': {
                'binance': 'Cryptocurrency trading',
                'ibkr': 'Multi-asset global trading',
                'mock': 'Testing and development'
            }.get(broker_type, 'General use')
        })

    return brokers


def create_sample_config(broker_type: str, trading_mode: str = 'paper') -> Dict[str, Any]:
    """
    Create a sample configuration for a broker type and trading mode.

    Args:
        broker_type: Type of broker ('binance', 'ibkr', 'mock')
        trading_mode: Trading mode ('paper', 'live')

    Returns:
        Sample configuration dictionary
    """
    base_config = {
        'type': broker_type,
        'trading_mode': trading_mode,
        'cash': 10000.0,
        'paper_trading_config': {
            'mode': 'realistic',
            'initial_balance': 10000.0,
            'commission_rate': 0.001,
            'slippage_model': 'linear',
            'base_slippage': 0.0005,
            'latency_simulation': True,
            'min_latency_ms': 10,
            'max_latency_ms': 100,
            'market_impact_enabled': True,
            'realistic_fills': True,
            'partial_fill_probability': 0.1,
            'reject_probability': 0.01
        },
        'notifications': {
            'position_opened': True,
            'position_closed': True,
            'email_enabled': True,
            'telegram_enabled': True,
            'error_notifications': True
        }
    }

    # Add live trading specific configuration
    if trading_mode == 'live':
        base_config.update({
            'live_trading_confirmed': False,  # Must be manually set to True
            'risk_management': {
                'max_position_size': 1000.0,
                'max_daily_loss': 500.0,
                'max_portfolio_risk': 0.02,  # 2%
                'position_sizing_method': 'fixed_dollar'
            }
        })

    return base_config


# Legacy function for backward compatibility
def get_broker_legacy(config: Dict[str, Any]):
    """
    Legacy broker factory function for backward compatibility.

    This function maintains compatibility with the old broker factory interface
    while internally using the new enhanced factory.
    """
    # Convert legacy config format to new format
    broker_type = config.get("type", "mock").lower()

    # Map legacy broker types
    if broker_type == "binance_paper":
        broker_type = "binance"
        config["trading_mode"] = "paper"
    elif broker_type == "binance":
        config["trading_mode"] = config.get("trading_mode", "live")

    # Use new factory
    return get_broker(config)
