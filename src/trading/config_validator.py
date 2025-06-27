"""
Configuration Validator Module
-----------------------------

This module provides validation for live trading bot configurations.
It ensures all required parameters are present and valid before starting the bot.

Classes:
- ConfigValidator: Validates trading bot configurations
"""

import json
from typing import Dict, Any, List, Tuple
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class ConfigValidator:
    """
    Validates live trading bot configurations.
    
    This class ensures that all required parameters are present and valid
    before the bot starts, preventing runtime errors.
    """
    
    def __init__(self):
        """Initialize the configuration validator."""
        self.errors = []
        self.warnings = []
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """
        Validate a complete trading bot configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []
        
        # Validate required sections
        self._validate_required_sections(config)
        
        # Validate each section
        if "broker" in config:
            self._validate_broker_config(config["broker"])
        
        if "trading" in config:
            self._validate_trading_config(config["trading"])
        
        if "data" in config:
            self._validate_data_config(config["data"])
        
        if "strategy" in config:
            self._validate_strategy_config(config["strategy"])
        
        if "notifications" in config:
            self._validate_notifications_config(config["notifications"])
        
        if "risk_management" in config:
            self._validate_risk_management_config(config["risk_management"])
        
        if "logging" in config:
            self._validate_logging_config(config["logging"])
        
        return len(self.errors) == 0, self.errors, self.warnings
    
    def _validate_required_sections(self, config: Dict[str, Any]):
        """Validate that all required sections are present."""
        required_sections = ["broker", "trading", "data", "strategy"]
        
        for section in required_sections:
            if section not in config:
                self.errors.append(f"Missing required section: {section}")
            elif not isinstance(config[section], dict):
                self.errors.append(f"Section '{section}' must be a dictionary")
    
    def _validate_broker_config(self, broker_config: Dict[str, Any]):
        """Validate broker configuration."""
        required_fields = ["type"]
        for field in required_fields:
            if field not in broker_config:
                self.errors.append(f"Missing required broker field: {field}")
        
        if "type" in broker_config:
            broker_type = broker_config["type"]
            valid_types = ["binance", "binance_paper", "ibkr", "mock"]
            if broker_type not in valid_types:
                self.errors.append(f"Invalid broker type: {broker_type}. Valid types: {valid_types}")
        
        # Validate numeric fields
        numeric_fields = ["initial_balance", "commission"]
        for field in numeric_fields:
            if field in broker_config:
                if not isinstance(broker_config[field], (int, float)):
                    self.errors.append(f"Broker field '{field}' must be numeric")
                elif field == "commission" and (broker_config[field] < 0 or broker_config[field] > 1):
                    self.warnings.append(f"Commission rate {broker_config[field]} seems unusual")
    
    def _validate_trading_config(self, trading_config: Dict[str, Any]):
        """Validate trading configuration."""
        required_fields = ["symbol"]
        for field in required_fields:
            if field not in trading_config:
                self.errors.append(f"Missing required trading field: {field}")
        
        if "symbol" in trading_config:
            symbol = trading_config["symbol"]
            if not isinstance(symbol, str) or len(symbol) == 0:
                self.errors.append("Trading symbol must be a non-empty string")
        
        # Validate numeric fields
        numeric_fields = ["position_size", "max_positions", "max_drawdown_pct", "max_exposure"]
        for field in numeric_fields:
            if field in trading_config:
                if not isinstance(trading_config[field], (int, float)):
                    self.errors.append(f"Trading field '{field}' must be numeric")
                elif field == "position_size" and (trading_config[field] <= 0 or trading_config[field] > 1):
                    self.errors.append(f"Position size must be between 0 and 1, got: {trading_config[field]}")
                elif field == "max_exposure" and (trading_config[field] <= 0 or trading_config[field] > 1):
                    self.errors.append(f"Max exposure must be between 0 and 1, got: {trading_config[field]}")
    
    def _validate_data_config(self, data_config: Dict[str, Any]):
        """Validate data configuration."""
        required_fields = ["data_source"]
        for field in required_fields:
            if field not in data_config:
                self.errors.append(f"Missing required data field: {field}")
        
        if "data_source" in data_config:
            data_source = data_config["data_source"]
            valid_sources = ["binance", "yahoo", "ibkr"]
            if data_source not in valid_sources:
                self.errors.append(f"Invalid data source: {data_source}. Valid sources: {valid_sources}")
        
        # Validate symbol if present
        if "symbol" in data_config:
            symbol = data_config["symbol"]
            if not isinstance(symbol, str) or len(symbol) == 0:
                self.errors.append("Data symbol must be a non-empty string")
        
        # Validate interval if present
        if "interval" in data_config:
            interval = data_config["interval"]
            valid_intervals = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
            if interval not in valid_intervals:
                self.errors.append(f"Invalid interval: {interval}. Valid intervals: {valid_intervals}")
        
        # Validate numeric fields
        numeric_fields = ["lookback_bars", "retry_interval", "polling_interval"]
        for field in numeric_fields:
            if field in data_config:
                if not isinstance(data_config[field], (int, float)):
                    self.errors.append(f"Data field '{field}' must be numeric")
                elif data_config[field] <= 0:
                    self.errors.append(f"Data field '{field}' must be positive")
    
    def _validate_strategy_config(self, strategy_config: Dict[str, Any]):
        """Validate strategy configuration."""
        required_fields = ["type", "entry_logic", "exit_logic"]
        for field in required_fields:
            if field not in strategy_config:
                self.errors.append(f"Missing required strategy field: {field}")
        
        if "type" in strategy_config:
            strategy_type = strategy_config["type"]
            if strategy_type != "custom":
                self.errors.append(f"Only 'custom' strategy type is supported, got: {strategy_type}")
        
        # Validate entry logic
        if "entry_logic" in strategy_config:
            entry_logic = strategy_config["entry_logic"]
            if not isinstance(entry_logic, dict):
                self.errors.append("Entry logic must be a dictionary")
            else:
                if "name" not in entry_logic:
                    self.errors.append("Entry logic missing 'name' field")
                if "params" not in entry_logic:
                    self.errors.append("Entry logic missing 'params' field")
                elif not isinstance(entry_logic["params"], dict):
                    self.errors.append("Entry logic params must be a dictionary")
        
        # Validate exit logic
        if "exit_logic" in strategy_config:
            exit_logic = strategy_config["exit_logic"]
            if not isinstance(exit_logic, dict):
                self.errors.append("Exit logic must be a dictionary")
            else:
                if "name" not in exit_logic:
                    self.errors.append("Exit logic missing 'name' field")
                if "params" not in exit_logic:
                    self.errors.append("Exit logic missing 'params' field")
                elif not isinstance(exit_logic["params"], dict):
                    self.errors.append("Exit logic params must be a dictionary")
        
        # Validate position size
        if "position_size" in strategy_config:
            position_size = strategy_config["position_size"]
            if not isinstance(position_size, (int, float)):
                self.errors.append("Strategy position_size must be numeric")
            elif position_size <= 0 or position_size > 1:
                self.errors.append(f"Strategy position_size must be between 0 and 1, got: {position_size}")
    
    def _validate_notifications_config(self, notifications_config: Dict[str, Any]):
        """Validate notifications configuration."""
        if not isinstance(notifications_config, dict):
            self.errors.append("Notifications configuration must be a dictionary")
            return
        
        # Validate enabled flag
        if "enabled" in notifications_config:
            if not isinstance(notifications_config["enabled"], bool):
                self.errors.append("Notifications enabled must be a boolean")
        
        # Validate Telegram configuration
        if "telegram" in notifications_config:
            telegram_config = notifications_config["telegram"]
            if not isinstance(telegram_config, dict):
                self.errors.append("Telegram configuration must be a dictionary")
            else:
                if "enabled" in telegram_config and not isinstance(telegram_config["enabled"], bool):
                    self.errors.append("Telegram enabled must be a boolean")
                
                if "notify_on" in telegram_config:
                    notify_on = telegram_config["notify_on"]
                    if not isinstance(notify_on, list):
                        self.errors.append("Telegram notify_on must be a list")
                    else:
                        valid_events = ["trade_entry", "trade_exit", "error", "daily_summary", "status"]
                        for event in notify_on:
                            if event not in valid_events:
                                self.warnings.append(f"Unknown Telegram notification event: {event}")
        
        # Validate Email configuration
        if "email" in notifications_config:
            email_config = notifications_config["email"]
            if not isinstance(email_config, dict):
                self.errors.append("Email configuration must be a dictionary")
            else:
                if "enabled" in email_config and not isinstance(email_config["enabled"], bool):
                    self.errors.append("Email enabled must be a boolean")
                
                if "notify_on" in email_config:
                    notify_on = email_config["notify_on"]
                    if not isinstance(notify_on, list):
                        self.errors.append("Email notify_on must be a list")
                    else:
                        valid_events = ["trade_entry", "trade_exit", "error", "daily_summary"]
                        for event in notify_on:
                            if event not in valid_events:
                                self.warnings.append(f"Unknown Email notification event: {event}")
    
    def _validate_risk_management_config(self, risk_config: Dict[str, Any]):
        """Validate risk management configuration."""
        if not isinstance(risk_config, dict):
            self.errors.append("Risk management configuration must be a dictionary")
            return
        
        # Validate numeric fields
        numeric_fields = ["stop_loss_pct", "take_profit_pct", "max_daily_trades", "max_daily_loss"]
        for field in numeric_fields:
            if field in risk_config:
                if not isinstance(risk_config[field], (int, float)):
                    self.errors.append(f"Risk management field '{field}' must be numeric")
                elif risk_config[field] < 0:
                    self.errors.append(f"Risk management field '{field}' must be non-negative")
        
        # Validate trailing stop configuration
        if "trailing_stop" in risk_config:
            trailing_config = risk_config["trailing_stop"]
            if not isinstance(trailing_config, dict):
                self.errors.append("Trailing stop configuration must be a dictionary")
            else:
                if "enabled" in trailing_config and not isinstance(trailing_config["enabled"], bool):
                    self.errors.append("Trailing stop enabled must be a boolean")
                
                numeric_fields = ["activation_pct", "trailing_pct"]
                for field in numeric_fields:
                    if field in trailing_config:
                        if not isinstance(trailing_config[field], (int, float)):
                            self.errors.append(f"Trailing stop field '{field}' must be numeric")
                        elif trailing_config[field] < 0:
                            self.errors.append(f"Trailing stop field '{field}' must be non-negative")
    
    def _validate_logging_config(self, logging_config: Dict[str, Any]):
        """Validate logging configuration."""
        if not isinstance(logging_config, dict):
            self.errors.append("Logging configuration must be a dictionary")
            return
        
        # Validate log level
        if "level" in logging_config:
            level = logging_config["level"]
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if level not in valid_levels:
                self.errors.append(f"Invalid log level: {level}. Valid levels: {valid_levels}")
        
        # Validate boolean fields
        boolean_fields = ["save_trades", "save_equity_curve"]
        for field in boolean_fields:
            if field in logging_config:
                if not isinstance(logging_config[field], bool):
                    self.errors.append(f"Logging field '{field}' must be a boolean")
        
        # Validate log file path
        if "log_file" in logging_config:
            log_file = logging_config["log_file"]
            if not isinstance(log_file, str) or len(log_file) == 0:
                self.errors.append("Log file path must be a non-empty string")


def validate_config_file(config_file: str) -> Tuple[bool, List[str], List[str]]:
    """
    Validate a configuration file.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        validator = ConfigValidator()
        return validator.validate_config(config)
        
    except FileNotFoundError:
        return False, [f"Configuration file not found: {config_file}"], []
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON in configuration file: {e}"], []
    except Exception as e:
        return False, [f"Error reading configuration file: {e}"], []


def print_validation_results(is_valid: bool, errors: List[str], warnings: List[str]):
    """Print validation results in a formatted way."""
    if is_valid:
        print("✅ Configuration validation passed!")
    else:
        print("❌ Configuration validation failed!")
    
    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"  ❌ {error}")
    
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  ⚠️  {warning}")
    
    if not errors and not warnings:
        print("No issues found.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python config_validator.py <config_file>")
        print("Example: python config_validator.py config/trading/0001.json")
        sys.exit(1)
    
    config_file = sys.argv[1]
    is_valid, errors, warnings = validate_config_file(config_file)
    print_validation_results(is_valid, errors, warnings)
    
    if not is_valid:
        sys.exit(1) 