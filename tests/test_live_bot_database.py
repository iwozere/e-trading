"""
Test LiveTradingBot Database Integration
---------------------------------------

This script tests that LiveTradingBot can be initialized with database integration
and that it properly handles bot_id and trade_type.
"""

import os
import sys
import json
from datetime import datetime

# Add src to path
sys.path.append('src')

from src.trading.live_trading_bot import LiveTradingBot
from src.data.trade_repository import TradeRepository


def create_test_config():
    """Create a test configuration file."""
    config = {
        "broker": {
            "type": "binance_paper",
            "initial_balance": 1000.0,
            "commission": 0.001
        },
        "trading": {
            "symbol": "BTCUSDT",
            "position_size": 0.1,
            "interval": "1h"
        },
        "data": {
            "data_source": "binance",
            "symbol": "BTCUSDT",
            "interval": "1h",
            "limit": 1000
        },
        "strategy": {
            "type": "custom",
            "entry_logic": {
                "name": "RSIEntry",
                "params": {
                    "rsi_period": 14,
                    "oversold": 30,
                    "overbought": 70
                }
            },
            "exit_logic": {
                "name": "ATRExit",
                "params": {
                    "atr_period": 14,
                    "atr_multiplier": 2.0
                }
            }
        }
    }
    
    # Save test config
    os.makedirs("config/trading", exist_ok=True)
    with open("config/trading/test_bot.json", "w") as f:
        json.dump(config, f, indent=2)
    
    return "test_bot.json"


def test_live_bot_initialization():
    """Test LiveTradingBot initialization with database integration."""
    print("🤖 Testing LiveTradingBot initialization...")
    
    try:
        # Create test config
        config_file = create_test_config()
        
        # Initialize LiveTradingBot
        bot = LiveTradingBot(config_file)
        
        # Verify database integration
        assert hasattr(bot, 'bot_id')
        assert hasattr(bot, 'trade_type')
        assert hasattr(bot, 'trade_repository')
        
        print(f"✅ Bot ID: {bot.bot_id}")
        print(f"✅ Trade Type: {bot.trade_type}")
        print(f"✅ Trading Pair: {bot.trading_pair}")
        
        # Verify bot instance was created in database
        repo = TradeRepository()
        bot_instance = repo.get_bot_instance(bot.bot_id)
        assert bot_instance is not None
        print(f"✅ Bot instance created in database: {bot_instance.id}")
        
        # Verify bot type is correct
        assert bot_instance.type == bot.trade_type
        print(f"✅ Bot type matches: {bot_instance.type}")
        
        repo.close()
        
        # Clean up
        bot.stop()
        
        return True
        
    except Exception as e:
        print(f"❌ LiveTradingBot initialization failed: {e}")
        return False


def test_bot_id_naming():
    """Test that bot_id uses config filename correctly."""
    print("\n📝 Testing bot_id naming convention...")
    
    try:
        # Create test config
        config_file = create_test_config()
        
        # Initialize bot
        bot = LiveTradingBot(config_file)
        
        # Verify bot_id is the config filename
        assert bot.bot_id == config_file
        print(f"✅ Bot ID correctly set to config filename: {bot.bot_id}")
        
        # Verify trade_type is paper (since we're using binance_paper)
        assert bot.trade_type == "paper"
        print(f"✅ Trade type correctly set to: {bot.trade_type}")
        
        bot.stop()
        return True
        
    except Exception as e:
        print(f"❌ Bot ID naming test failed: {e}")
        return False


def test_restart_recovery():
    """Test that bot can recover open positions on restart."""
    print("\n🔄 Testing restart recovery...")
    
    try:
        config_file = create_test_config()
        
        # Initialize first bot instance
        bot1 = LiveTradingBot(config_file)
        
        # Simulate an open position (this would normally be created by a trade)
        repo = TradeRepository()
        
        # Create a test open trade
        trade_data = {
            'bot_id': bot1.bot_id,
            'trade_type': bot1.trade_type,
            'strategy_name': 'TestStrategy',
            'entry_logic_name': 'RSIEntry',
            'exit_logic_name': 'ATRExit',
            'symbol': bot1.trading_pair,
            'interval': '1h',
            'entry_time': datetime.utcnow(),
            'buy_order_created': datetime.utcnow(),
            'entry_price': 50000.0,
            'entry_value': 1000.0,
            'size': 0.02,
            'direction': 'long',
            'status': 'open',
            'extra_metadata': {'test': True}
        }
        
        trade = repo.create_trade(trade_data)
        print(f"✅ Created test open trade: {trade.id}")
        
        # Stop first bot
        bot1.stop()
        
        # Initialize second bot instance (simulating restart)
        bot2 = LiveTradingBot(config_file)
        
        # Verify open positions were loaded
        open_trades = repo.get_open_trades(bot_id=bot2.bot_id, symbol=bot2.trading_pair)
        assert len(open_trades) == 1
        print(f"✅ Found {len(open_trades)} open trade on restart")
        
        # Verify trade details
        open_trade = open_trades[0]
        assert open_trade.symbol == bot2.trading_pair
        assert open_trade.status == 'open'
        print(f"✅ Open trade details: {open_trade.symbol} @ {open_trade.entry_price}")
        
        # Clean up
        repo.delete_trade(str(trade.id))
        bot2.stop()
        repo.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Restart recovery test failed: {e}")
        return False


def cleanup():
    """Clean up test files."""
    try:
        # Remove test config file
        if os.path.exists("config/trading/test_bot.json"):
            os.remove("config/trading/test_bot.json")
            print("✅ Cleaned up test config file")
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")


def main():
    """Run all LiveTradingBot database integration tests."""
    print("🚀 Starting LiveTradingBot Database Integration Tests")
    print("=" * 60)
    
    tests = [
        ("LiveTradingBot Initialization", test_live_bot_initialization),
        ("Bot ID Naming Convention", test_bot_id_naming),
        ("Restart Recovery", test_restart_recovery),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! LiveTradingBot database integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    # Clean up
    cleanup()
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 