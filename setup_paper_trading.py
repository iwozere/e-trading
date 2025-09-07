#!/usr/bin/env python3
"""
Paper Trading Setup Script
=========================

This script helps you set up paper trading with your selected strategies.
Run this script to validate your configuration and test the setup.
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.trading.config_validator import validate_config_file, print_validation_results
from src.trading.live_trading_bot import LiveTradingBot
from src.notification.logger import setup_logger

logger = setup_logger(__name__)

def check_environment():
    """Check if required environment variables are set"""
    required_vars = ['BINANCE_API_KEY', 'BINANCE_API_SECRET']
    missing_vars = []

    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n📝 Please set these environment variables:")
        print("   export BINANCE_API_KEY='your_testnet_api_key'")
        print("   export BINANCE_API_SECRET='your_testnet_secret_key'")
        return False

    print("✅ Environment variables are set")
    return True

def validate_configs():
    """Validate paper trading configurations"""
    configs = [
        "config/trading/paper_trading_rsi_or_bb.json",
        "config/trading/paper_trading_simple_atr.json"
    ]

    all_valid = True

    for config_path in configs:
        print(f"\n🔍 Validating {config_path}...")
        is_valid, errors, warnings = validate_config_file(config_path)
        print_validation_results(is_valid, errors, warnings)

        if not is_valid:
            all_valid = False
            print(f"❌ {config_path} validation failed")
        else:
            print(f"✅ {config_path} is valid")

    return all_valid

def test_broker_connection():
    """Test connection to Binance testnet"""
    try:
        from src.trading.broker.binance_paper_broker import BinancePaperBroker

        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')

        if not api_key or not api_secret:
            print("❌ Cannot test broker connection - missing API credentials")
            return False

        broker = BinancePaperBroker(api_key, api_secret, cash=10000.0)
        print("✅ Binance testnet connection successful")
        return True

    except Exception as e:
        print(f"❌ Broker connection failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Paper Trading Setup")
    print("=" * 50)

    # Step 1: Check environment
    print("\n1️⃣ Checking environment variables...")
    if not check_environment():
        return False

    # Step 2: Validate configurations
    print("\n2️⃣ Validating configurations...")
    if not validate_configs():
        return False

    # Step 3: Test broker connection
    print("\n3️⃣ Testing broker connection...")
    if not test_broker_connection():
        return False

    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run paper trading with RSI/BB strategy:")
    print("   python src/trading/run_bot.py paper_trading_rsi_or_bb.json")
    print("\n2. Run paper trading with Simple ATR strategy:")
    print("   python src/trading/run_bot.py paper_trading_simple_atr.json")
    print("\n3. Monitor logs in the logs/ directory")
    print("4. Check database for trade records (if enabled)")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
