#!/usr/bin/env python3
"""
Broker Management System Demo
----------------------------

This script demonstrates the comprehensive broker management system including:
- Broker factory with seamless paper-to-live trading
- Broker lifecycle management and health monitoring
- Configuration management with templates and versioning
- Environment-specific configurations
- Connection pooling and performance monitoring

Usage:
    python examples/broker_management_demo.py --demo factory
    python examples/broker_management_demo.py --demo manager
    python examples/broker_management_demo.py --demo config
    python examples/broker_management_demo.py --demo full
"""

import asyncio
import argparse
import json
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.trading.broker.broker_factory import (
    get_broker, list_available_brokers, create_sample_config, validate_broker_config
)
from src.trading.broker.broker_manager import BrokerManager
from src.trading.broker.config_manager import ConfigManager, Environment


async def demo_broker_factory():
    """Demonstrate broker factory capabilities."""
    print("🏭 Broker Factory Demo")
    print("=" * 50)

    # List available brokers
    print("📋 Available Brokers:")
    brokers = list_available_brokers()

    for broker in brokers:
        print(f"   {broker['name']}:")
        print(f"     Type: {broker['type']}")
        print(f"     Credentials Available: {broker['credentials_available']}")
        print(f"     Recommended For: {broker['recommended_for']}")

        # Show capabilities
        capabilities = broker['capabilities']
        print("     Capabilities:")
        print(f"       Paper Trading: {capabilities['paper_trading']}")
        print(f"       Live Trading: {capabilities['live_trading']}")
        print(f"       Asset Classes: {', '.join(capabilities['asset_classes'])}")
        print(f"       Order Types: {', '.join(capabilities['order_types'])}")
        print()

    # Demonstrate configuration creation
    print("⚙️  Configuration Creation:")

    # Create sample configurations
    configs = [
        ('binance_paper', 'binance', 'paper'),
        ('ibkr_paper', 'ibkr', 'paper'),
        ('mock_dev', 'mock', 'paper')
    ]

    for name, broker_type, trading_mode in configs:
        config = create_sample_config(broker_type, trading_mode)

        print(f"   {name}:")
        print(f"     Type: {config['type']}")
        print(f"     Trading Mode: {config['trading_mode']}")
        print(f"     Initial Balance: ${config['cash']:,.2f}")
        print(f"     Commission Rate: {config['paper_trading_config']['commission_rate']:.4f}")

        # Validate configuration
        try:
            validated = validate_broker_config(config)
            print("     ✅ Configuration valid")
        except Exception as e:
            print(f"     ❌ Configuration invalid: {e}")
        print()

    # Demonstrate broker creation
    print("🔧 Broker Creation:")

    # Create mock broker (safe for demo)
    mock_config = create_sample_config('mock', 'paper')

    try:
        broker = get_broker(mock_config)
        print(f"   ✅ Created {broker.get_name()}")
        print(f"      Trading Mode: {broker.get_trading_mode().value}")
        print(f"      Paper Trading: {broker.is_paper_trading()}")

        # Get broker status
        status = broker.get_status()
        print(f"      Status: {json.dumps(status, indent=8, default=str)}")

    except Exception as e:
        print(f"   ❌ Failed to create broker: {e}")

    print("✅ Broker factory demo completed!")


async def demo_broker_manager():
    """Demonstrate broker management capabilities."""
    print("🎛️  Broker Manager Demo")
    print("=" * 50)

    # Create broker manager
    manager_config = {
        'health_check_interval': 5,
        'auto_restart_enabled': True,
        'max_connections_per_type': 3
    }

    manager = BrokerManager(manager_config)

    try:
        # Start management system
        manager.start_management()
        print("✅ Started broker management system")

        # Add multiple brokers
        print("\n📝 Adding Brokers:")

        broker_configs = [
            ('mock_broker_1', {
                'type': 'mock',
                'trading_mode': 'paper',
                'cash': 10000.0,
                'name': 'Mock Broker 1'
            }),
            ('mock_broker_2', {
                'type': 'mock',
                'trading_mode': 'paper',
                'cash': 15000.0,
                'name': 'Mock Broker 2'
            }),
            ('mock_broker_3', {
                'type': 'mock',
                'trading_mode': 'paper',
                'cash': 20000.0,
                'name': 'Mock Broker 3'
            })
        ]

        for broker_id, config in broker_configs:
            result = await manager.add_broker(broker_id, config)
            if result:
                print(f"   ✅ Added {broker_id}")
            else:
                print(f"   ❌ Failed to add {broker_id}")

        # List managed brokers
        print("\n📊 Managed Brokers:")
        brokers = manager.list_brokers()

        for broker in brokers:
            print(f"   {broker['broker_id']}:")
            print(f"     Type: {broker['broker_type']}")
            print(f"     Mode: {broker['trading_mode']}")
            print(f"     Status: {broker['status']}")
            print(f"     Errors: {broker['error_count']}")

        # Start all brokers
        print("\n🚀 Starting All Brokers:")
        start_results = await manager.start_all_brokers()

        for broker_id, success in start_results.items():
            if success:
                print(f"   ✅ Started {broker_id}")
            else:
                print(f"   ❌ Failed to start {broker_id}")

        # Wait a moment for health monitoring
        print("\n⏳ Waiting for health monitoring...")
        await asyncio.sleep(3)

        # Get management status
        print("\n📈 Management Status:")
        status = manager.get_management_status()

        print(f"   Manager Active: {status['manager_active']}")
        print(f"   Auto Restart: {status['auto_restart_enabled']}")
        print(f"   Total Brokers: {status['total_managed_brokers']}")

        health_summary = status['health_summary']
        print("   Health Summary:")
        print(f"     Running: {health_summary['running_brokers']}/{health_summary['total_brokers']}")
        print(f"     Health: {health_summary['health_percentage']:.1f}%")
        print(f"     Errors: {health_summary['error_brokers']}")

        # Connection pool stats
        pool_stats = status['connection_pool_stats']
        print("   Connection Pool:")
        print(f"     Active Connections: {pool_stats['total_active_connections']}")
        print(f"     Max Per Type: {pool_stats['max_connections_per_type']}")

        # Demonstrate restart
        print("\n🔄 Restarting a Broker:")
        restart_result = await manager.restart_broker('mock_broker_2')
        if restart_result:
            print("   ✅ Broker restarted successfully")
        else:
            print("   ❌ Broker restart failed")

        # Stop all brokers
        print("\n🛑 Stopping All Brokers:")
        stop_results = await manager.stop_all_brokers()

        for broker_id, success in stop_results.items():
            if success:
                print(f"   ✅ Stopped {broker_id}")
            else:
                print(f"   ❌ Failed to stop {broker_id}")

    finally:
        # Cleanup
        await manager.cleanup()
        print("\n🧹 Cleanup completed")

    print("✅ Broker manager demo completed!")


async def demo_config_manager():
    """Demonstrate configuration management capabilities."""
    print("⚙️  Configuration Manager Demo")
    print("=" * 50)

    # Create config manager with temporary directory
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        config_manager = ConfigManager(temp_dir)

        # Template management
        print("📋 Template Management:")

        template_manager = config_manager.get_template_manager()
        templates = template_manager.list_templates()

        print(f"   Available Templates: {len(templates)}")
        for template in templates[:5]:  # Show first 5
            print(f"     - {template}")

        # Create configuration from template
        print("\n🔧 Creating Configuration from Template:")

        result = config_manager.create_from_template(
            'demo_binance',
            'binance_paper',
            {'cash': 25000.0, 'name': 'Demo Binance Broker'}
        )

        if result:
            print("   ✅ Created configuration from template")
        else:
            print("   ❌ Failed to create configuration")

        # Create custom configuration
        print("\n📝 Creating Custom Configuration:")

        custom_config = {
            'type': 'ibkr',
            'trading_mode': 'paper',
            'cash': 50000.0,
            'name': 'Custom IBKR Broker',
            'paper_trading_config': {
                'mode': 'advanced',
                'commission_rate': 0.0003,
                'base_slippage': 0.0002
            }
        }

        result = config_manager.create_configuration('demo_ibkr', custom_config)

        if result:
            print("   ✅ Created custom configuration")
        else:
            print("   ❌ Failed to create custom configuration")

        # List configurations
        print("\n📊 Configuration List:")
        configs = config_manager.list_configurations()

        for config in configs:
            print(f"   {config['name']}:")
            print(f"     Type: {config['type']}")
            print(f"     Mode: {config['trading_mode']}")
            print(f"     Versions: {config['version_count']}")
            print(f"     Last Updated: {config['last_updated']}")

        # Update configuration (creates new version)
        print("\n🔄 Updating Configuration:")

        updated_config = custom_config.copy()
        updated_config['cash'] = 75000.0

        result = config_manager.update_configuration(
            'demo_ibkr',
            updated_config,
            "Increased initial balance"
        )

        if result:
            print("   ✅ Configuration updated")
        else:
            print("   ❌ Configuration update failed")

        # Show version history
        print("\n📚 Version History:")
        versions = config_manager.get_configuration_versions('demo_ibkr')

        for version in versions:
            print(f"   {version.version}:")
            print(f"     Description: {version.description}")
            print(f"     Timestamp: {version.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"     Hash: {version.config_hash}")

        # Environment management
        print("\n🌍 Environment Management:")

        env_manager = config_manager.get_environment_manager()

        # Create production environment overrides
        prod_overrides = {
            'notifications': {
                'email_enabled': True,
                'telegram_enabled': False
            },
            'risk_management': {
                'max_position_size': 1000.0,
                'max_daily_loss': 500.0
            }
        }

        env_manager.save_environment_config(Environment.PRODUCTION, prod_overrides)
        print("   ✅ Saved production environment overrides")

        # Get configuration with environment overrides
        base_config = config_manager.get_configuration('demo_ibkr')
        prod_config = config_manager.get_configuration('demo_ibkr', Environment.PRODUCTION)

        print(f"   Base Config Notifications: {base_config.get('notifications', {})}")
        print(f"   Prod Config Notifications: {prod_config.get('notifications', {})}")

        # Export/Import
        print("\n📤 Export/Import:")

        # Export configuration
        exported = config_manager.export_configuration('demo_binance')
        print(f"   ✅ Exported configuration ({len(exported)} characters)")

        # Import configuration
        result = config_manager.import_configuration('imported_demo', exported)
        if result:
            print("   ✅ Imported configuration successfully")
        else:
            print("   ❌ Import failed")

        # Final configuration list
        print("\n📋 Final Configuration List:")
        final_configs = config_manager.list_configurations()

        for config in final_configs:
            print(f"   - {config['name']} ({config['type']}, {config['trading_mode']})")

    print("✅ Configuration manager demo completed!")


async def demo_full_system():
    """Demonstrate the complete integrated system."""
    print("🚀 Full System Integration Demo")
    print("=" * 50)

    # Create temporary config directory
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:

        # Initialize systems
        print("🔧 Initializing Systems:")

        config_manager = ConfigManager(temp_dir)
        broker_manager = BrokerManager({
            'health_check_interval': 3,
            'auto_restart_enabled': True
        })

        print("   ✅ Configuration manager initialized")
        print("   ✅ Broker manager initialized")

        try:
            # Start broker management
            broker_manager.start_management()
            print("   ✅ Broker management started")

            # Create configurations using templates
            print("\n📝 Creating Broker Configurations:")

            configs_to_create = [
                ('conservative_trader', 'conservative', {'cash': 5000.0}),
                ('aggressive_trader', 'aggressive', {'cash': 25000.0}),
                ('dev_tester', 'mock_dev', {'cash': 100000.0})
            ]

            for name, template, overrides in configs_to_create:
                result = config_manager.create_from_template(name, template, overrides)
                if result:
                    print(f"   ✅ Created {name} configuration")
                else:
                    print(f"   ❌ Failed to create {name} configuration")

            # Add brokers to management using configurations
            print("\n🎛️  Adding Brokers to Management:")

            for name, _, _ in configs_to_create:
                config = config_manager.get_configuration(name)
                if config:
                    result = await broker_manager.add_broker(f"broker_{name}", config)
                    if result:
                        print(f"   ✅ Added broker_{name} to management")
                    else:
                        print(f"   ❌ Failed to add broker_{name}")

            # Start all brokers
            print("\n🚀 Starting All Brokers:")
            start_results = await broker_manager.start_all_brokers()

            for broker_id, success in start_results.items():
                status = "✅ Started" if success else "❌ Failed to start"
                print(f"   {status} {broker_id}")

            # Monitor system for a few seconds
            print("\n📊 System Monitoring (5 seconds):")

            for i in range(5):
                await asyncio.sleep(1)

                # Get system status
                mgmt_status = broker_manager.get_management_status()
                health = mgmt_status['health_summary']

                print(f"   [{i+1}/5] Running: {health['running_brokers']}/{health['total_brokers']} "
                      f"({health['health_percentage']:.0f}% healthy)")

            # Demonstrate configuration update and broker restart
            print("\n🔄 Dynamic Configuration Update:")

            # Update a configuration
            config = config_manager.get_configuration('conservative_trader')
            config['cash'] = 7500.0

            result = config_manager.update_configuration(
                'conservative_trader',
                config,
                "Increased balance for demo"
            )

            if result:
                print("   ✅ Updated conservative_trader configuration")

                # Restart the corresponding broker to apply changes
                restart_result = await broker_manager.restart_broker('broker_conservative_trader')
                if restart_result:
                    print("   ✅ Restarted broker with new configuration")
                else:
                    print("   ❌ Failed to restart broker")

            # Final system status
            print("\n📈 Final System Status:")

            # Configuration summary
            configs = config_manager.list_configurations()
            print(f"   Configurations: {len(configs)} total")

            # Broker summary
            brokers = broker_manager.list_brokers()
            running_count = sum(1 for b in brokers if b['status'] == 'running')
            print(f"   Brokers: {running_count}/{len(brokers)} running")

            # Management status
            mgmt_status = broker_manager.get_management_status()
            print(f"   System Health: {mgmt_status['health_summary']['health_percentage']:.1f}%")
            print(f"   Uptime: {mgmt_status['uptime']:.1f} seconds")

        finally:
            # Cleanup
            print("\n🧹 System Cleanup:")
            await broker_manager.cleanup()
            print("   ✅ Broker manager cleanup completed")

    print("✅ Full system integration demo completed!")


async def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='Broker Management System Demo')
    parser.add_argument('--demo', choices=['factory', 'manager', 'config', 'full'],
                       default='full', help='Demo to run (default: full)')

    args = parser.parse_args()

    print("🎯 Broker Management System Demo")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    try:
        if args.demo == 'factory':
            await demo_broker_factory()
        elif args.demo == 'manager':
            await demo_broker_manager()
        elif args.demo == 'config':
            await demo_config_manager()
        elif args.demo == 'full':
            await demo_full_system()

        print(f"\n🎉 Demo '{args.demo}' completed successfully!")

    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()