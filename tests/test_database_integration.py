"""
Test Database Integration
------------------------

This script tests the database integration by:
1. Creating a test trade
2. Updating the trade
3. Loading open positions
4. Testing bot instance management
"""

import os
import sys
import uuid
from datetime import datetime, timedelta

# Add src to path
sys.path.append('src')

from src.data.database import get_database_manager, Trade, BotInstance, PerformanceMetrics
from src.data.db.trade_repository import TradeRepository


def test_database_connection():
    """Test database connection and table creation."""
    print("🔧 Testing database connection...")

    try:
        # Initialize database manager
        db_manager = get_database_manager()
        session = db_manager.get_session()

        # Test basic operations
        print("✅ Database connection successful")
        print(f"✅ Database engine: {db_manager.engine}")

        session.close()
        return True

    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False


def test_trade_operations():
    """Test trade CRUD operations."""
    print("\n📊 Testing trade operations...")

    try:
        repo = TradeRepository()

        # Test data
        bot_id = "test_bot_001"
        symbol = "BTCUSDT"
        entry_time = datetime.utcnow()

        # Create trade
        trade_data = {
            'bot_id': bot_id,
            'trade_type': 'paper',
            'strategy_name': 'TestStrategy',
            'entry_logic_name': 'RSIEntry',
            'exit_logic_name': 'ATRExit',
            'symbol': symbol,
            'interval': '1h',
            'entry_time': entry_time,
            'buy_order_created': entry_time,
            'entry_price': 50000.0,
            'entry_value': 1000.0,
            'size': 0.02,
            'direction': 'long',
            'status': 'open',
            'extra_metadata': {
                'test': True,
                'paper_trading': True
            }
        }

        # Create trade
        trade = repo.create_trade(trade_data)
        print(f"✅ Created trade: {trade.id}")

        # Get trade by ID
        retrieved_trade = repo.get_trade_by_id(str(trade.id))
        assert retrieved_trade is not None
        print(f"✅ Retrieved trade: {retrieved_trade.symbol} @ {retrieved_trade.entry_price}")

        # Update trade
        exit_time = datetime.utcnow() + timedelta(hours=1)
        update_data = {
            'exit_time': exit_time,
            'sell_order_created': exit_time,
            'sell_order_closed': exit_time,
            'exit_price': 51000.0,
            'exit_value': 1020.0,
            'commission': 1.0,
            'gross_pnl': 20.0,
            'net_pnl': 19.0,
            'pnl_percentage': 1.9,
            'exit_reason': 'take_profit',
            'status': 'closed'
        }

        updated_trade = repo.update_trade(str(trade.id), update_data)
        assert updated_trade is not None
        print(f"✅ Updated trade: PnL = {updated_trade.net_pnl}")

        # Test queries
        open_trades = repo.get_open_trades(bot_id=bot_id)
        print(f"✅ Open trades: {len(open_trades)}")

        closed_trades = repo.get_closed_trades(bot_id=bot_id)
        print(f"✅ Closed trades: {len(closed_trades)}")

        # Test summary
        summary = repo.get_trade_summary(bot_id=bot_id)
        print(f"✅ Trade summary: {summary}")

        repo.close()
        return True

    except Exception as e:
        print(f"❌ Trade operations failed: {e}")
        return False


def test_bot_instance_operations():
    """Test bot instance operations."""
    print("\n🤖 Testing bot instance operations...")

    try:
        repo = TradeRepository()

        # Test data
        bot_id = "test_bot_instance_001"

        # Create bot instance
        bot_data = {
            'id': bot_id,
            'type': 'paper',
            'config_file': 'test_config.json',
            'status': 'running',
            'current_balance': 1000.0,
            'total_pnl': 50.0,
            'extra_metadata': {
                'trading_pair': 'BTCUSDT',
                'strategy': 'TestStrategy'
            }
        }

        # Create bot instance
        bot_instance = repo.create_bot_instance(bot_data)
        print(f"✅ Created bot instance: {bot_instance.id}")

        # Get bot instance
        retrieved_bot = repo.get_bot_instance(bot_id)
        assert retrieved_bot is not None
        print(f"✅ Retrieved bot: {retrieved_bot.type} - {retrieved_bot.status}")

        # Update bot instance
        update_data = {
            'status': 'stopped',
            'current_balance': 1050.0,
            'total_pnl': 60.0,
            'last_heartbeat': datetime.utcnow()
        }

        updated_bot = repo.update_bot_instance(bot_id, update_data)
        assert updated_bot is not None
        print(f"✅ Updated bot: balance = {updated_bot.current_balance}")

        # Test queries
        running_bots = repo.get_running_bots()
        print(f"✅ Running bots: {len(running_bots)}")

        paper_bots = repo.get_bot_instances_by_type('paper')
        print(f"✅ Paper bots: {len(paper_bots)}")

        repo.close()
        return True

    except Exception as e:
        print(f"❌ Bot instance operations failed: {e}")
        return False


def test_performance_metrics():
    """Test performance metrics operations."""
    print("\n📈 Testing performance metrics...")

    try:
        repo = TradeRepository()

        # Test data
        bot_id = "test_metrics_001"

        # Create performance metrics
        metrics_data = {
            'bot_id': bot_id,
            'trade_type': 'paper',
            'symbol': 'BTCUSDT',
            'interval': '1h',
            'entry_logic_name': 'RSIEntry',
            'exit_logic_name': 'ATRExit',
            'metrics': {
                'sharpe_ratio': 1.5,
                'win_rate': 65.0,
                'profit_factor': 1.8,
                'max_drawdown': -5.2,
                'total_trades': 100,
                'total_pnl': 150.0
            }
        }

        # Create metrics
        metrics = repo.create_performance_metrics(metrics_data)
        print(f"✅ Created performance metrics: {metrics.id}")

        # Get metrics
        retrieved_metrics = repo.get_performance_metrics(bot_id)
        assert len(retrieved_metrics) > 0
        print(f"✅ Retrieved metrics: {retrieved_metrics[0].metrics['sharpe_ratio']}")

        repo.close()
        return True

    except Exception as e:
        print(f"❌ Performance metrics failed: {e}")
        return False


def test_restart_recovery():
    """Test restart recovery by simulating bot restart."""
    print("\n🔄 Testing restart recovery...")

    try:
        repo = TradeRepository()

        # Create an open trade
        bot_id = "restart_test_bot"
        symbol = "ETHUSDT"

        trade_data = {
            'bot_id': bot_id,
            'trade_type': 'paper',
            'strategy_name': 'RestartTestStrategy',
            'entry_logic_name': 'RSIEntry',
            'exit_logic_name': 'ATRExit',
            'symbol': symbol,
            'interval': '1h',
            'entry_time': datetime.utcnow(),
            'buy_order_created': datetime.utcnow(),
            'entry_price': 3000.0,
            'entry_value': 600.0,
            'size': 0.2,
            'direction': 'long',
            'status': 'open',
            'extra_metadata': {'restart_test': True}
        }

        trade = repo.create_trade(trade_data)
        print(f"✅ Created open trade: {trade.id}")

        # Simulate bot restart - load open positions
        open_trades = repo.get_open_trades(bot_id=bot_id, symbol=symbol)
        assert len(open_trades) == 1
        print(f"✅ Found {len(open_trades)} open trade on restart")

        # Verify trade details
        open_trade = open_trades[0]
        assert open_trade.symbol == symbol
        assert open_trade.status == 'open'
        print(f"✅ Open trade details: {open_trade.symbol} @ {open_trade.entry_price}")

        repo.close()
        return True

    except Exception as e:
        print(f"❌ Restart recovery failed: {e}")
        return False


def cleanup_test_data():
    """Clean up test data."""
    print("\n🧹 Cleaning up test data...")

    try:
        repo = TradeRepository()

        # Clean up test trades
        test_bot_ids = ["test_bot_001", "test_bot_instance_001", "test_metrics_001", "restart_test_bot"]

        for bot_id in test_bot_ids:
            # Delete trades
            trades = repo.get_trades_by_bot(bot_id)
            for trade in trades:
                repo.delete_trade(str(trade.id))

            # Delete bot instance
            bot_instance = repo.get_bot_instance(bot_id)
            if bot_instance:
                repo.session.delete(bot_instance)

        repo.commit()
        print("✅ Test data cleaned up")
        repo.close()

    except Exception as e:
        print(f"❌ Cleanup failed: {e}")


def main():
    """Run all database integration tests."""
    print("🚀 Starting Database Integration Tests")
    print("=" * 50)

    tests = [
        ("Database Connection", test_database_connection),
        ("Trade Operations", test_trade_operations),
        ("Bot Instance Operations", test_bot_instance_operations),
        ("Performance Metrics", test_performance_metrics),
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

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Database integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

    # Clean up test data
    cleanup_test_data()

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)