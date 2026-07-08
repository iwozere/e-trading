#!/usr/bin/env python3
"""
Quick Start Script for Enhanced Multi-Strategy Trading
-----------------------------------------------------

This script provides a simple way to get started with the enhanced
multi-strategy trading system.
"""

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def print_banner():
    """Print startup banner."""
    print("🚀 Enhanced Multi-Strategy Trading System")
    print("=" * 50)
    print("Quick Start Menu")
    print()


def show_menu():
    """Show the main menu."""
    print("Choose an option:")
    print("1. 🔧 Setup system (first time)")
    print("2. 🎯 Run simple demo (2 strategies)")
    print("3. 🚀 Run full system (3+ strategies)")
    print("4. 📊 Run broker management demo")
    print("5. 🧪 Test individual broker")
    print("6. ❌ Exit")
    print()


async def run_setup():
    """Run the setup process."""
    print("🔧 Running system setup...")
    print()

    try:
        import setup_enhanced_trading
        setup_enhanced_trading.main()
        print()
        print("✅ Setup complete!")
        print("💡 Next: Edit .env file with your API keys")
        return True
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False


async def run_simple_demo():
    """Run the simple 2-strategy demo."""
    print("🎯 Starting simple demo with 2 strategies...")
    print("Strategies: RSI BTC + Bollinger Bands ETH")
    print()

    try:
        from enhanced_multi_strategy_runner import EnhancedMultiStrategyRunner

        runner = EnhancedMultiStrategyRunner("config/enhanced_trading/simple_multi_strategy.json")
        await runner.run()

    except FileNotFoundError:
        print("❌ Configuration file not found. Please run setup first (option 1)")
    except Exception as e:
        print(f"❌ Demo failed: {e}")


async def run_full_system():
    """Run the full multi-strategy system."""
    print("🚀 Starting full multi-strategy system...")
    print("Strategies: RSI BTC + BB ETH + MACD ADA")
    print()

    try:
        from enhanced_multi_strategy_runner import EnhancedMultiStrategyRunner

        runner = EnhancedMultiStrategyRunner("config/enhanced_trading/multi_strategy_binance.json")
        await runner.run()

    except FileNotFoundError:
        print("❌ Configuration file not found. Please run setup first (option 1)")
    except Exception as e:
        print(f"❌ System failed: {e}")


async def run_broker_demo():
    """Run the broker management demo."""
    print("📊 Starting broker management demo...")
    print()

    try:
        import examples.broker_management_demo
        # The demo will run its own main function
        await examples.broker_management_demo.main()

    except Exception as e:
        print(f"❌ Broker demo failed: {e}")


async def test_individual_broker():
    """Test an individual broker."""
    print("🧪 Testing individual broker...")
    print()
    print("Choose broker to test:")
    print("1. Binance Paper Trading")
    print("2. IBKR Paper Trading")
    print("3. Mock Broker")

    choice = input("Enter choice (1-3): ").strip()

    try:
        if choice == "1":
            import examples.enhanced_binance_broker_demo
            await examples.enhanced_binance_broker_demo.main()
        elif choice == "2":
            import examples.enhanced_ibkr_broker_demo
            await examples.enhanced_ibkr_broker_demo.main()
        elif choice == "3":
            print("Mock broker test - creating simple mock broker...")
            from src.trading.broker.broker_factory import BrokerFactory

            config = {
                "type": "mock",
                "trading_mode": "paper",
                "name": "test_mock_broker",
                "cash": 10000.0
            }

            broker = await BrokerFactory.create_broker("test_mock", config)
            if broker:
                print("✅ Mock broker created successfully")
                portfolio = await broker.get_portfolio()
                print(f"Portfolio: {portfolio}")
            else:
                print("❌ Failed to create mock broker")
        else:
            print("❌ Invalid choice")

    except Exception as e:
        print(f"❌ Broker test failed: {e}")


async def main():
    """Main function."""
    print_banner()

    while True:
        show_menu()
        choice = input("Enter your choice (1-6): ").strip()
        print()

        try:
            if choice == "1":
                await run_setup()
            elif choice == "2":
                await run_simple_demo()
            elif choice == "3":
                await run_full_system()
            elif choice == "4":
                await run_broker_demo()
            elif choice == "5":
                await test_individual_broker()
            elif choice == "6":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-6.")

        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

        if choice in ["2", "3", "4", "5"]:
            print("\n" + "="*50)
            input("Press Enter to return to menu...")
            print()


if __name__ == "__main__":
    asyncio.run(main())