#!/usr/bin/env python3
"""
Configuration Management Example
================================

This example demonstrates how to use the new centralized configuration
management system for the crypto trading platform.

Features demonstrated:
- Creating configurations from templates
- Environment-specific configuration management
- Schema validation
- Configuration registry and discovery
- Hot-reload support
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import ConfigManager, TradingConfig, OptimizerConfig, DataConfig


def setup_logging():
    """Setup logging for the example"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def example_basic_usage():
    """Demonstrate basic configuration management usage"""
    print("\n=== Basic Configuration Management ===")
    
    # Initialize configuration manager
    config_manager = ConfigManager(environment="development")
    
    # Create a trading bot configuration from template
    trading_config = config_manager.create_config(
        "trading_paper",
        bot_id="example_bot_001",
        symbol="BTCUSDT",
        initial_balance=10000.0,
        description="Example trading bot for demonstration"
    )
    
    print(f"Created trading config: {trading_config.bot_id}")
    print(f"Symbol: {trading_config.trading['symbol']}")
    print(f"Initial balance: ${trading_config.broker['initial_balance']}")
    
    # Save the configuration
    config_path = config_manager.save_config(trading_config, "example_bot_001.json")
    print(f"Saved to: {config_path}")
    
    # Load the configuration back
    loaded_config = config_manager.get_config("example_bot_001")
    print(f"Loaded config: {loaded_config.bot_id}")
    
    return config_manager, trading_config


def example_environment_management():
    """Demonstrate environment-specific configuration management"""
    print("\n=== Environment Management ===")
    
    # Create configurations for different environments
    environments = ["development", "staging", "production"]
    
    for env in environments:
        config_manager = ConfigManager(environment=env)
        
        # Create environment-specific config
        config = config_manager.create_config(
            "trading_paper",
            bot_id=f"env_bot_{env}",
            symbol="ETHUSDT",
            initial_balance=5000.0 if env == "development" else 10000.0,
            description=f"Environment-specific bot for {env}"
        )
        
        # Adjust settings based on environment
        if env == "development":
            config.logging.level = "DEBUG"
            config.notifications.enabled = False
        elif env == "staging":
            config.logging.level = "INFO"
            config.notifications.telegram.enabled = True
        elif env == "production":
            config.logging.level = "WARNING"
            config.notifications.telegram.enabled = True
            config.notifications.email.enabled = True
            config.risk_management.stop_loss_pct = 3.0  # Tighter for production
        
        # Save environment-specific config
        config_path = config_manager.save_config(config, f"env_bot_{env}.json")
        print(f"Created {env} config: {config_path}")
        print(f"  Log level: {config.logging.level}")
        print(f"  Notifications: {config.notifications.enabled}")
        print(f"  Stop loss: {config.risk_management.stop_loss_pct}%")


def example_optimizer_configuration():
    """Demonstrate optimizer configuration management"""
    print("\n=== Optimizer Configuration ===")
    
    config_manager = ConfigManager(environment="development")
    
    # Create basic optimizer config
    basic_optimizer = config_manager.create_config(
        "optimizer_basic",
        description="Basic optimization example"
    )
    
    print(f"Basic optimizer: {basic_optimizer.n_trials} trials")
    print(f"Initial capital: ${basic_optimizer.initial_capital}")
    
    # Create advanced optimizer config
    advanced_optimizer = config_manager.create_config(
        "optimizer_advanced",
        n_trials=500,
        n_jobs=-1,  # Use all cores
        description="Advanced optimization example"
    )
    
    print(f"Advanced optimizer: {advanced_optimizer.n_trials} trials")
    print(f"Parallel jobs: {advanced_optimizer.n_jobs}")
    print(f"Entry strategies: {len(advanced_optimizer.entry_strategies)}")
    print(f"Exit strategies: {len(advanced_optimizer.exit_strategies)}")
    
    # Save both configurations
    config_manager.save_config(basic_optimizer, "basic_optimizer.json")
    config_manager.save_config(advanced_optimizer, "advanced_optimizer.json")


def example_data_configuration():
    """Demonstrate data feed configuration management"""
    print("\n=== Data Configuration ===")
    
    config_manager = ConfigManager(environment="development")
    
    # Create different data feed configurations
    data_sources = [
        ("binance", "BTCUSDT", "1h"),
        ("yahoo", "AAPL", "5m"),
        ("ibkr", "SPY", "1m")
    ]
    
    for source, symbol, interval in data_sources:
        if source == "binance":
            config = config_manager.create_config(
                "data_binance",
                symbol=symbol,
                interval=interval,
                testnet=True
            )
        elif source == "yahoo":
            config = config_manager.create_config(
                "data_yahoo",
                symbol=symbol,
                interval=interval
            )
        elif source == "ibkr":
            config = config_manager.create_config(
                "data_ibkr",
                symbol=symbol,
                interval=interval
            )
        
        print(f"{source.upper()} data feed:")
        print(f"  Symbol: {config.symbol}")
        print(f"  Interval: {config.interval}")
        print(f"  Lookback: {config.lookback_bars} bars")
        
        # Save configuration
        config_manager.save_config(config, f"{source}_{symbol.lower()}.json")


def example_configuration_discovery():
    """Demonstrate configuration discovery and search"""
    print("\n=== Configuration Discovery ===")
    
    config_manager = ConfigManager(environment="development")
    
    # List all configurations by type
    configs_by_type = config_manager.list_configs()
    print("Configurations by type:")
    for config_type, config_ids in configs_by_type.items():
        print(f"  {config_type}: {len(config_ids)} configs")
        for config_id in config_ids[:3]:  # Show first 3
            print(f"    - {config_id}")
        if len(config_ids) > 3:
            print(f"    ... and {len(config_ids) - 3} more")
    
    # Search configurations
    search_results = config_manager.search_configs("BTCUSDT")
    print(f"\nSearch results for 'BTCUSDT': {len(search_results)} configs")
    for config_id in search_results:
        print(f"  - {config_id}")
    
    # Get configuration statistics
    stats = config_manager.get_config_summary()
    print(f"\nConfiguration statistics:")
    print(f"  Total configs: {stats['total_configs']}")
    print(f"  Environment: {stats['environment']}")
    print(f"  Hot reload: {stats['hot_reload_enabled']}")


def example_schema_validation():
    """Demonstrate schema validation"""
    print("\n=== Schema Validation ===")
    
    config_manager = ConfigManager(environment="development")
    
    # Test valid configuration
    try:
        valid_config = config_manager.create_config(
            "trading_paper",
            bot_id="valid_bot",
            symbol="BTCUSDT",
            initial_balance=1000.0
        )
        print("✓ Valid configuration created successfully")
    except Exception as e:
        print(f"✗ Valid configuration failed: {e}")
    
    # Test invalid configuration (will fail validation)
    try:
        invalid_config = config_manager.create_config(
            "trading_paper",
            bot_id="",  # Empty bot_id should fail
            symbol="BTCUSDT",
            initial_balance=-1000.0  # Negative balance should fail
        )
        print("✗ Invalid configuration should have failed")
    except Exception as e:
        print(f"✓ Invalid configuration correctly rejected: {e}")
    
    # Test configuration file validation
    config_files = list(Path("config/development").glob("*.json"))
    if config_files:
        test_file = str(config_files[0])
        is_valid, errors, warnings = config_manager.validate_config_file(test_file)
        if is_valid:
            print(f"✓ Configuration file valid: {test_file}")
        else:
            print(f"✗ Configuration file invalid: {test_file}")
            for error in errors:
                print(f"  Error: {error}")


def example_hot_reload():
    """Demonstrate hot-reload functionality"""
    print("\n=== Hot-Reload Demo ===")
    
    config_manager = ConfigManager(environment="development")
    
    # Enable hot-reload
    config_manager.enable_hot_reload(True)
    print("✓ Hot-reload enabled")
    print("  (In a real scenario, modifying config files would trigger automatic reload)")
    
    # Disable hot-reload
    config_manager.enable_hot_reload(False)
    print("✓ Hot-reload disabled")


def example_templates():
    """Demonstrate configuration templates"""
    print("\n=== Configuration Templates ===")
    
    config_manager = ConfigManager(environment="development")
    
    # List available templates
    templates = config_manager.templates.list_templates()
    print("Available templates:")
    for template_name in templates:
        description = config_manager.templates.get_template_description(template_name)
        print(f"  {template_name}: {description}")
    
    # Show template details
    print("\nTemplate details:")
    for template_name in ["trading_paper", "optimizer_basic", "data_binance"]:
        template = config_manager.templates.get_template(template_name)
        print(f"\n{template_name}:")
        print(f"  Environment: {template.get('environment', 'N/A')}")
        print(f"  Description: {template.get('description', 'N/A')}")
        
        if template_name == "trading_paper":
            print(f"  Broker type: {template.get('broker', {}).get('type', 'N/A')}")
            print(f"  Symbol: {template.get('trading', {}).get('symbol', 'N/A')}")
        elif template_name == "optimizer_basic":
            print(f"  Trials: {template.get('n_trials', 'N/A')}")
            print(f"  Optimizer: {template.get('optimizer_type', 'N/A')}")
        elif template_name == "data_binance":
            print(f"  Data source: {template.get('data_source', 'N/A')}")
            print(f"  Symbol: {template.get('symbol', 'N/A')}")


def main():
    """Run all configuration examples"""
    print("Configuration Management System Examples")
    print("=" * 50)
    
    setup_logging()
    
    try:
        # Run all examples
        example_basic_usage()
        example_environment_management()
        example_optimizer_configuration()
        example_data_configuration()
        example_configuration_discovery()
        example_schema_validation()
        example_hot_reload()
        example_templates()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("\nNext steps:")
        print("1. Check the 'config/development' directory for created configurations")
        print("2. Try modifying a configuration file and see hot-reload in action")
        print("3. Explore the configuration registry and templates")
        print("4. Create your own custom configurations")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 