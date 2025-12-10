"""
Test Database Integration
------------------------

This script tests the database integration by:
1. Creating a test trade
2. Updating the trade
3. Loading open positions
4. Testing bot instance management

IMPORTANT: This test uses isolated test database fixtures, NOT production database.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pytest

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.data.db.repos.repo_trading import TradingBotsRepo, TradingTradesRepo, TradingPositionsRepo, TradingMetricsRepo


class TestDatabaseIntegration:
    """Database integration tests using isolated test database."""

    def test_database_connection(self, db_session):
        """Test database connection and session management."""
        # Test basic session operations
        assert db_session is not None
        result = db_session.execute("SELECT 1")
        assert result is not None
        print("✅ Database connection successful")

    def test_trade_operations(self, db_session):
        """Test trade CRUD operations."""
        trades_repo = TradingTradesRepo(db_session)
        bots_repo = TradingBotsRepo(db_session)

        # Test data
        bot_id = "test_bot_001"
        symbol = "BTCUSDT"
        entry_time = datetime.now(timezone.utc)

        # Create bot instance first
        bot_data = {
            'id': bot_id,
            'type': 'paper',
            'config_file': 'test_config.json',
            'status': 'running',
            'current_balance': 10000.0,
        }
        bot_instance = bots_repo.create(bot_data)
        db_session.flush()

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

        trade = trades_repo.create(trade_data)
        db_session.flush()
        print(f"✅ Created trade: {trade.id}")

        # Get trade by ID
        retrieved_trade = trades_repo.get_by_id(trade.id)
        assert retrieved_trade is not None
        assert retrieved_trade.symbol == symbol
        print(f"✅ Retrieved trade: {retrieved_trade.symbol} @ {retrieved_trade.entry_price}")

        # Update trade
        exit_time = datetime.now(timezone.utc) + timedelta(hours=1)
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

        trades_repo.update(trade.id, update_data)
        db_session.flush()

        updated_trade = trades_repo.get_by_id(trade.id)
        assert updated_trade.net_pnl == 19.0
        print(f"✅ Updated trade: PnL = {updated_trade.net_pnl}")

    def test_bot_instance_operations(self, db_session):
        """Test bot instance operations."""
        bots_repo = TradingBotsRepo(db_session)

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

        bot_instance = bots_repo.create(bot_data)
        db_session.flush()
        print(f"✅ Created bot instance: {bot_instance.id}")

        # Get bot instance
        retrieved_bot = bots_repo.get_by_id(bot_id)
        assert retrieved_bot is not None
        assert retrieved_bot.type == 'paper'
        print(f"✅ Retrieved bot: {retrieved_bot.type} - {retrieved_bot.status}")

        # Update bot instance
        update_data = {
            'status': 'stopped',
            'current_balance': 1050.0,
            'total_pnl': 60.0,
            'last_heartbeat': datetime.now(timezone.utc)
        }

        bots_repo.update(bot_id, update_data)
        db_session.flush()

        updated_bot = bots_repo.get_by_id(bot_id)
        assert updated_bot.current_balance == 1050.0
        print(f"✅ Updated bot: balance = {updated_bot.current_balance}")

    def test_performance_metrics(self, db_session):
        """Test performance metrics operations."""
        metrics_repo = TradingMetricsRepo(db_session)
        bots_repo = TradingBotsRepo(db_session)

        # Create bot first
        bot_id = "test_metrics_001"
        bot_data = {
            'id': bot_id,
            'type': 'paper',
            'config_file': 'test_config.json',
            'status': 'running',
        }
        bots_repo.create(bot_data)
        db_session.flush()

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

        metrics = metrics_repo.create(metrics_data)
        db_session.flush()
        print(f"✅ Created performance metrics: {metrics.id}")

        # Get metrics
        retrieved_metrics = metrics_repo.get_by_id(metrics.id)
        assert retrieved_metrics is not None
        assert retrieved_metrics.metrics['sharpe_ratio'] == 1.5
        print(f"✅ Retrieved metrics: {retrieved_metrics.metrics['sharpe_ratio']}")

    def test_restart_recovery(self, db_session):
        """Test restart recovery by simulating bot restart."""
        trades_repo = TradingTradesRepo(db_session)
        bots_repo = TradingBotsRepo(db_session)

        # Create bot
        bot_id = "restart_test_bot"
        symbol = "ETHUSDT"

        bot_data = {
            'id': bot_id,
            'type': 'paper',
            'config_file': 'test_config.json',
            'status': 'running',
        }
        bots_repo.create(bot_data)
        db_session.flush()

        # Create an open trade
        trade_data = {
            'bot_id': bot_id,
            'trade_type': 'paper',
            'strategy_name': 'RestartTestStrategy',
            'entry_logic_name': 'RSIEntry',
            'exit_logic_name': 'ATRExit',
            'symbol': symbol,
            'interval': '1h',
            'entry_time': datetime.now(timezone.utc),
            'buy_order_created': datetime.now(timezone.utc),
            'entry_price': 3000.0,
            'entry_value': 600.0,
            'size': 0.2,
            'direction': 'long',
            'status': 'open',
            'extra_metadata': {'restart_test': True}
        }

        trade = trades_repo.create(trade_data)
        db_session.flush()
        print(f"✅ Created open trade: {trade.id}")

        # Simulate bot restart - load open positions
        # This would normally use a method like get_open_trades_by_bot
        all_trades = trades_repo.get_all()
        open_trades = [t for t in all_trades if t.bot_id == bot_id and t.status == 'open']

        assert len(open_trades) == 1
        print(f"✅ Found {len(open_trades)} open trade on restart")

        # Verify trade details
        open_trade = open_trades[0]
        assert open_trade.symbol == symbol
        assert open_trade.status == 'open'
        print(f"✅ Open trade details: {open_trade.symbol} @ {open_trade.entry_price}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
