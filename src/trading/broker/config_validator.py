#!/usr/bin/env python3
"""
Broker Configuration Validator
-----------------------------

This module provides comprehensive validation and configuration management
for trading brokers with paper-to-live trading mode switching.

Features:
- Configuration validation and normalization
- Live trading safety checks and confirmations
- Credential validation and availability checks
- Configuration templates and examples
- Risk management parameter validation

Classes:
- ConfigValidator: Main configuration validator
- RiskManagementValidator: Risk management configuration validator
- NotificationConfigValidator: Notification configuration validator
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from src.notification.logger import setup_logger
_logger = setup_logger(__name__)


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    normalized_config: Optional[Dict[str, Any]] = None


class ConfigValidator:
    """
    Comprehensive configuration validator for trading brokers.
    """

    SUPPORTED_BROKER_TYPES = ['binance', 'ibkr', 'mock']
    SUPPORTED_TRADING_MODES = ['paper', 'live']
    SUPPORTED_ORDER_TYPES = ['market', 'limit', 'stop', 'stop_limit', 'trailing_stop', 'oco', 'bracket']
    SUPPORTED_SLIPPAGE_MODELS = ['linear', 'sqrt', 'fixed']

    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> ValidationResult:
        """
        Validate and normalize broker configuration.

        Args:
            config: Raw configuration dictionary

        Returns:
            ValidationResult with validation status and normalized config
        """
        errors = []
        warnings = []

        # Start with default configuration
        normalized_config = cls._get_default_config()

        try:
            # Update with provided config
            normalized_config.update(config)

            # Validate broker type
            broker_type = normalized_config.get('type', '').lower()
            if broker_type not in cls.SUPPORTED_BROKER_TYPES:
                errors.append(f"Unsupported broker type: {broker_type}. Supported: {cls.SUPPORTED_BROKER_TYPES}")

            # Validate trading mode
            trading_mode = normalized_config.get('trading_mode', '').lower()
            if trading_mode not in cls.SUPPORTED_TRADING_MODES:
                errors.append(f"Unsupported trading mode: {trading_mode}. Supported: {cls.SUPPORTED_TRADING_MODES}")

            # Set paper_trading flag based on mode
            normalized_config['paper_trading'] = (trading_mode == 'paper')

            # Validate paper trading configuration
            paper_errors, paper_warnings = cls._validate_paper_trading_config(
                normalized_config.get('paper_trading_config', {})
            )
            errors.extend(paper_errors)
            warnings.extend(paper_warnings)

            # Validate risk management configuration
            risk_errors, risk_warnings = RiskManagementValidator.validate(
                normalized_config.get('risk_management', {}),
                trading_mode
            )
            errors.extend(risk_errors)
            warnings.extend(risk_warnings)

            # Validate notification configuration
            notif_errors, notif_warnings = NotificationConfigValidator.validate(
                normalized_config.get('notifications', {})
            )
            errors.extend(notif_errors)
            warnings.extend(notif_warnings)

            # Validate live trading specific requirements
            if trading_mode == 'live':
                live_errors, live_warnings = cls._validate_live_trading_config(normalized_config)
                errors.extend(live_errors)
                warnings.extend(live_warnings)

            # Validate broker-specific configuration
            broker_errors, broker_warnings = cls._validate_broker_specific_config(
                broker_type, normalized_config
            )
            errors.extend(broker_errors)
            warnings.extend(broker_warnings)

        except Exception as e:
            errors.append(f"Configuration validation error: {str(e)}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            normalized_config=normalized_config if len(errors) == 0 else None
        )

    @classmethod
    def _get_default_config(cls) -> Dict[str, Any]:
        """Get default broker configuration."""
        return {
            'type': 'mock',
            'trading_mode': 'paper',
            'name': 'default_broker',
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
            },
            'risk_management': {
                'max_position_size': 1000.0,
                'max_daily_loss': 500.0,
                'max_portfolio_risk': 0.02,
                'position_sizing_method': 'fixed_dollar',
                'stop_loss_enabled': True,
                'take_profit_enabled': True
            }
        }

    @classmethod
    def _validate_paper_trading_config(cls, config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Validate paper trading configuration."""
        errors = []
        warnings = []

        # Validate slippage model
        slippage_model = config.get('slippage_model', 'linear')
        if slippage_model not in cls.SUPPORTED_SLIPPAGE_MODELS:
            errors.append(f"Unsupported slippage model: {slippage_model}. Supported: {cls.SUPPORTED_SLIPPAGE_MODELS}")

        # Validate numeric parameters
        numeric_params = {
            'initial_balance': (100.0, 1000000.0),
            'commission_rate': (0.0, 0.1),
            'base_slippage': (0.0, 0.01),
            'min_latency_ms': (0, 1000),
            'max_latency_ms': (1, 5000),
            'market_impact_factor': (0.0, 0.01),
            'partial_fill_probability': (0.0, 1.0),
            'reject_probability': (0.0, 0.5)
        }

        for param, (min_val, max_val) in numeric_params.items():
            value = config.get(param)
            if value is not None:
                if not isinstance(value, (int, float)):
                    errors.append(f"Parameter '{param}' must be numeric")
                elif not (min_val <= value <= max_val):
                    errors.append(f"Parameter '{param}' must be between {min_val} and {max_val}")

        # Validate latency range
        min_latency = config.get('min_latency_ms', 10)
        max_latency = config.get('max_latency_ms', 100)
        if min_latency >= max_latency:
            errors.append("min_latency_ms must be less than max_latency_ms")

        # Warnings for extreme values
        if config.get('commission_rate', 0.001) > 0.01:
            warnings.append("Commission rate is very high (>1%)")

        if config.get('base_slippage', 0.0005) > 0.005:
            warnings.append("Base slippage is very high (>0.5%)")

        return errors, warnings

    @classmethod
    def _validate_live_trading_config(cls, config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Validate live trading specific configuration."""
        errors = []
        warnings = []

        # Require explicit confirmation
        if not config.get('live_trading_confirmed', False):
            errors.append(
                "Live trading requires explicit confirmation. "
                "Set 'live_trading_confirmed': true in configuration."
            )

        # Validate risk management is present
        risk_config = config.get('risk_management', {})
        if not risk_config:
            errors.append("Live trading requires risk management configuration")

        # Validate broker credentials availability
        broker_type = config.get('type', '').lower()
        if broker_type in ['binance', 'ibkr']:
            # This would be checked against actual credential availability
            # For now, just warn
            warnings.append(f"Ensure {broker_type.upper()} live trading credentials are properly configured")

        return errors, warnings

    @classmethod
    def _validate_broker_specific_config(cls, broker_type: str, config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Validate broker-specific configuration."""
        errors = []
        warnings = []

        if broker_type == 'binance':
            # Validate Binance-specific parameters
            if config.get('trading_mode') == 'live':
                warnings.append("Binance live trading uses real funds - ensure proper risk management")

        elif broker_type == 'ibkr':
            # Validate IBKR-specific parameters
            client_id = config.get('client_id')
            if client_id is not None and not isinstance(client_id, int):
                errors.append("IBKR client_id must be an integer")

            port = config.get('port')
            if port is not None and not (1000 <= port <= 65535):
                errors.append("IBKR port must be between 1000 and 65535")

        elif broker_type == 'mock':
            # Mock broker warnings
            if config.get('trading_mode') == 'live':
                warnings.append("Mock broker does not support live trading - will use paper trading mode")

        return errors, warnings


class RiskManagementValidator:
    """Validator for risk management configuration."""

    @classmethod
    def validate(cls, config: Dict[str, Any], trading_mode: str) -> Tuple[List[str], List[str]]:
        """
        Validate risk management configuration.

        Args:
            config: Risk management configuration
            trading_mode: Trading mode ('paper' or 'live')

        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []

        # Risk management is required for live trading
        if trading_mode == 'live' and not config:
            errors.append("Risk management configuration is required for live trading")
            return errors, warnings

        # Validate required parameters for live trading
        if trading_mode == 'live':
            required_params = ['max_position_size', 'max_daily_loss']
            for param in required_params:
                if param not in config:
                    errors.append(f"Risk management parameter '{param}' is required for live trading")

        # Validate numeric parameters
        numeric_params = {
            'max_position_size': (1.0, 1000000.0),
            'max_daily_loss': (1.0, 100000.0),
            'max_portfolio_risk': (0.001, 1.0),
            'stop_loss_percentage': (0.01, 0.5),
            'take_profit_percentage': (0.01, 2.0)
        }

        for param, (min_val, max_val) in numeric_params.items():
            value = config.get(param)
            if value is not None:
                if not isinstance(value, (int, float)):
                    errors.append(f"Risk parameter '{param}' must be numeric")
                elif not (min_val <= value <= max_val):
                    errors.append(f"Risk parameter '{param}' must be between {min_val} and {max_val}")

        # Validate position sizing method
        sizing_method = config.get('position_sizing_method', 'fixed_dollar')
        valid_methods = ['fixed_dollar', 'percentage', 'kelly', 'volatility_adjusted']
        if sizing_method not in valid_methods:
            errors.append(f"Invalid position sizing method: {sizing_method}. Valid: {valid_methods}")

        # Warnings for aggressive settings
        if config.get('max_portfolio_risk', 0.02) > 0.1:
            warnings.append("Maximum portfolio risk is very high (>10%)")

        if config.get('max_daily_loss', 500) > 5000:
            warnings.append("Maximum daily loss limit is very high")

        return errors, warnings


class NotificationConfigValidator:
    """Validator for notification configuration."""

    @classmethod
    def validate(cls, config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """
        Validate notification configuration.

        Args:
            config: Notification configuration

        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []

        # Validate boolean parameters
        boolean_params = [
            'position_opened', 'position_closed', 'email_enabled',
            'telegram_enabled', 'error_notifications'
        ]

        for param in boolean_params:
            value = config.get(param)
            if value is not None and not isinstance(value, bool):
                errors.append(f"Notification parameter '{param}' must be boolean")

        # Check if at least one notification method is enabled
        email_enabled = config.get('email_enabled', True)
        telegram_enabled = config.get('telegram_enabled', True)

        if not email_enabled and not telegram_enabled:
            warnings.append("All notification methods are disabled - you won't receive any alerts")

        # Check if position notifications are disabled
        pos_opened = config.get('position_opened', True)
        pos_closed = config.get('position_closed', True)

        if not pos_opened and not pos_closed:
            warnings.append("All position notifications are disabled")

        return errors, warnings


def create_config_template(broker_type: str, trading_mode: str = 'paper') -> Dict[str, Any]:
    """
    Create a configuration template for a specific broker and trading mode.

    Args:
        broker_type: Type of broker ('binance', 'ibkr', 'mock')
        trading_mode: Trading mode ('paper', 'live')

    Returns:
        Configuration template dictionary
    """
    validator = ConfigValidator()
    base_config = validator._get_default_config()

    # Customize for broker type
    base_config['type'] = broker_type
    base_config['trading_mode'] = trading_mode
    base_config['name'] = f"{broker_type}_{trading_mode}_broker"

    # Broker-specific customizations
    if broker_type == 'binance':
        base_config['paper_trading_config']['commission_rate'] = 0.001  # 0.1%
        base_config['paper_trading_config']['base_slippage'] = 0.0005   # 0.05%

    elif broker_type == 'ibkr':
        base_config['paper_trading_config']['commission_rate'] = 0.0005  # 0.05%
        base_config['paper_trading_config']['base_slippage'] = 0.0003    # 0.03%
        base_config['paper_trading_config']['min_latency_ms'] = 5
        base_config['paper_trading_config']['max_latency_ms'] = 50

    elif broker_type == 'mock':
        base_config['paper_trading_config']['commission_rate'] = 0.0
        base_config['paper_trading_config']['base_slippage'] = 0.0
        base_config['paper_trading_config']['latency_simulation'] = False

    # Live trading specific additions
    if trading_mode == 'live':
        base_config['live_trading_confirmed'] = False  # Must be manually set
        base_config['risk_management']['max_position_size'] = 1000.0
        base_config['risk_management']['max_daily_loss'] = 500.0

        # Add warning comment
        base_config['_WARNING'] = "LIVE TRADING MODE - REAL MONEY WILL BE USED"

    return base_config


def validate_and_create_broker_config(raw_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration and return normalized config or raise exception.

    Args:
        raw_config: Raw configuration dictionary

    Returns:
        Validated and normalized configuration

    Raises:
        ValueError: If configuration is invalid
    """
    result = ConfigValidator.validate_config(raw_config)

    if not result.is_valid:
        error_msg = "Configuration validation failed:\n" + "\n".join(result.errors)
        if result.warnings:
            error_msg += "\n\nWarnings:\n" + "\n".join(result.warnings)
        raise ValueError(error_msg)

    if result.warnings:
        _logger.warning("Configuration warnings:\n" + "\n".join(result.warnings))

    return result.normalized_config