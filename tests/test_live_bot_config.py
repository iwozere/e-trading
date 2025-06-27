#!/usr/bin/env python3
"""
Test Live Trading Bot Configuration
---------------------------------

This script tests the live trading bot configuration and components.
It validates the configuration file and tests individual components.

Usage:
    python test_live_bot_config.py [config_file]

Examples:
    python test_live_bot_config.py 0001.json
    python test_live_bot_config.py live_data_example.json
"""

import sys
import json
import time
from typing import Dict, Any

from src.trading.config_validator import validate_config_file, print_validation_results
from src.data.data_feed_factory import DataFeedFactory
from src.broker.broker_factory import get_broker
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def test_configuration(config_file: str) -> bool:
    """
    Test configuration file validation.
    
    Args:
        config_file: Configuration file path
        
    Returns:
        True if validation passed, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Testing Configuration: {config_file}")
    print(f"{'='*60}")
    
    is_valid, errors, warnings = validate_config_file(config_file)
    print_validation_results(is_valid, errors, warnings)
    
    return is_valid


def test_data_feed_creation(config_file: str) -> bool:
    """
    Test data feed creation.
    
    Args:
        config_file: Configuration file path
        
    Returns:
        True if data feed creation succeeded, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Testing Data Feed Creation")
    print(f"{'='*60}")
    
    try:
        # Load configuration
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        if "data" not in config:
            print("❌ No data configuration found")
            return False
        
        data_config = config["data"]
        
        # Create data feed
        print(f"Creating data feed with config: {data_config}")
        data_feed = DataFeedFactory.create_data_feed(data_config)
        
        if data_feed is None:
            print("❌ Failed to create data feed")
            return False
        
        # Get status
        status = data_feed.get_status()
        print(f"✅ Data feed created successfully")
        print(f"   Symbol: {status.get('symbol')}")
        print(f"   Interval: {status.get('interval')}")
        print(f"   Connected: {status.get('is_connected')}")
        
        # Stop data feed
        data_feed.stop()
        print("✅ Data feed stopped successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing data feed creation: {e}")
        return False


def test_broker_creation(config_file: str) -> bool:
    """
    Test broker creation.
    
    Args:
        config_file: Configuration file path
        
    Returns:
        True if broker creation succeeded, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Testing Broker Creation")
    print(f"{'='*60}")
    
    try:
        # Load configuration
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        if "broker" not in config:
            print("❌ No broker configuration found")
            return False
        
        broker_config = config["broker"]
        
        # Create broker
        print(f"Creating broker with config: {broker_config}")
        broker = get_broker(broker_config)
        
        if broker is None:
            print("❌ Failed to create broker")
            return False
        
        print(f"✅ Broker created successfully")
        print(f"   Type: {broker_config.get('type')}")
        print(f"   Initial Balance: {broker_config.get('initial_balance')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing broker creation: {e}")
        return False


def test_strategy_configuration(config_file: str) -> bool:
    """
    Test strategy configuration.
    
    Args:
        config_file: Configuration file path
        
    Returns:
        True if strategy configuration is valid, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Testing Strategy Configuration")
    print(f"{'='*60}")
    
    try:
        # Load configuration
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        if "strategy" not in config:
            print("❌ No strategy configuration found")
            return False
        
        strategy_config = config["strategy"]
        
        print(f"Strategy Type: {strategy_config.get('type')}")
        
        # Check entry logic
        if "entry_logic" in strategy_config:
            entry_logic = strategy_config["entry_logic"]
            print(f"✅ Entry Logic: {entry_logic.get('name')}")
            print(f"   Parameters: {len(entry_logic.get('params', {}))} parameters")
        else:
            print("❌ No entry logic configured")
            return False
        
        # Check exit logic
        if "exit_logic" in strategy_config:
            exit_logic = strategy_config["exit_logic"]
            print(f"✅ Exit Logic: {exit_logic.get('name')}")
            print(f"   Parameters: {len(exit_logic.get('params', {}))} parameters")
        else:
            print("❌ No exit logic configured")
            return False
        
        # Check position size
        position_size = strategy_config.get("position_size", 0.1)
        print(f"✅ Position Size: {position_size}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing strategy configuration: {e}")
        return False


def test_notification_configuration(config_file: str) -> bool:
    """
    Test notification configuration.
    
    Args:
        config_file: Configuration file path
        
    Returns:
        True if notification configuration is valid, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Testing Notification Configuration")
    print(f"{'='*60}")
    
    try:
        # Load configuration
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        if "notifications" not in config:
            print("❌ No notification configuration found")
            return False
        
        notifications_config = config["notifications"]
        enabled = notifications_config.get("enabled", False)
        print(f"Notifications Enabled: {enabled}")
        
        if not enabled:
            print("⚠️  Notifications are disabled")
            return True
        
        # Check Telegram
        if "telegram" in notifications_config:
            telegram_config = notifications_config["telegram"]
            telegram_enabled = telegram_config.get("enabled", False)
            print(f"✅ Telegram: {'Enabled' if telegram_enabled else 'Disabled'}")
            
            if telegram_enabled:
                notify_on = telegram_config.get("notify_on", [])
                print(f"   Notify on: {', '.join(notify_on)}")
        
        # Check Email
        if "email" in notifications_config:
            email_config = notifications_config["email"]
            email_enabled = email_config.get("enabled", False)
            print(f"✅ Email: {'Enabled' if email_enabled else 'Disabled'}")
            
            if email_enabled:
                notify_on = email_config.get("notify_on", [])
                print(f"   Notify on: {', '.join(notify_on)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing notification configuration: {e}")
        return False


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python test_live_bot_config.py <config_file>")
        print("Example: python test_live_bot_config.py 0001.json")
        sys.exit(1)
    
    config_file = f"config/trading/{sys.argv[1]}"
    
    print("🧪 Live Trading Bot Configuration Test")
    print("=" * 60)
    
    # Test configuration validation
    config_valid = test_configuration(config_file)
    
    if not config_valid:
        print("\n❌ Configuration validation failed. Please fix the errors above.")
        sys.exit(1)
    
    # Test individual components
    data_feed_ok = test_data_feed_creation(config_file)
    broker_ok = test_broker_creation(config_file)
    strategy_ok = test_strategy_configuration(config_file)
    notifications_ok = test_notification_configuration(config_file)
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Configuration Validation: {'✅ PASS' if config_valid else '❌ FAIL'}")
    print(f"Data Feed Creation: {'✅ PASS' if data_feed_ok else '❌ FAIL'}")
    print(f"Broker Creation: {'✅ PASS' if broker_ok else '❌ FAIL'}")
    print(f"Strategy Configuration: {'✅ PASS' if strategy_ok else '❌ FAIL'}")
    print(f"Notification Configuration: {'✅ PASS' if notifications_ok else '❌ FAIL'}")
    
    all_tests_passed = all([config_valid, data_feed_ok, broker_ok, strategy_ok, notifications_ok])
    
    if all_tests_passed:
        print(f"\n🎉 All tests passed! The configuration is ready for live trading.")
    else:
        print(f"\n⚠️  Some tests failed. Please review the errors above.")
    
    return all_tests_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 