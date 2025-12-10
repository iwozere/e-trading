"""
Test LiveTradingBot Database Integration
---------------------------------------

This script tests that LiveTradingBot database operations work correctly
with the database integration layer.

IMPORTANT: This test uses isolated test database fixtures, NOT production database.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

import pytest

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.data.db.repos.repo_trading import TradingBotsRepo, TradingTradesRepo


class TestLiveBotDatabase:
    """Test LiveTradingBot database integration using isolated test database."""

    def test_bot_initialization_database(self, db_session):
        """Test bot instance creation in database."""
        bots_repo = TradingBotsRepo(db_session)

        # Create a bot instance
        bot_id = "test_bot"
        bot_data = {
            'id': bot_id,
            'type': 'paper',
            'config_file': 'test_bot.json',
            'status': 'running',
            'current_balance': 1000.0,
            'extra_metadata': {
                'trading_pair': 'BTCUSDT',
                'interval': '1h'
            }
        }

        bot_instance = bots_repo.create(bot_data)
        db_session.flush()

        # Verify bot instance was created
        assert bot_instance is not None
        assert bot_instance.id == bot_id
        print(f"✅ Bot ID: {bot_instance.id}")
        print(f"✅ Trade Type: {bot_instance.type}")

        # Verify we can retrieve it
        retrieved_bot = bots_repo.get_by_id(bot_id)
        assert retrieved_bot is not None
        assert retrieved_bot.type == 'paper'
        print(f"✅ Bot instance created in database: {retrieved_bot.id}")

    def test_bot_id_naming(self, db_session):
        """Test that bot_id uses config filename correctly."""
        bots_repo = TradingBotsRepo(db_session)

        config_file = "test_bot.json"
        bot_id = config_file  # Bot ID should match config filename

        bot_data = {
            'id': bot_id,
            'type': 'paper',
            'config_file': config_file,
            'status': 'running',
            'current_balance': 1000.0,
        }

        bot = bots_repo.create(bot_data)
        db_session.flush()

        # Verify bot_id is the config filename
        assert bot.id == config_file
        print(f"✅ Bot ID correctly set to config filename: {bot.id}")

        # Verify trade_type is paper
        assert bot.type == "paper"
        print(f"✅ Trade type correctly set to: {bot.type}")

    def test_restart_recovery(self, db_session):
        """Test that bot can recover open positions on restart."""
        bots_repo = TradingBotsRepo(db_session)
        trades_repo = TradingTradesRepo(db_session)

        config_file = "test_bot.json"
        bot_id = config_file
        trading_pair = "BTCUSDT"

        # Create bot instance
        bot_data = {
            'id': bot_id,
            'type': 'paper',
            'config_file': config_file,
            'status': 'running',
        }
        bot = bots_repo.create(bot_data)
        db_session.flush()

        # Create a test open trade
        trade_data = {
            'bot_id': bot_id,
            'trade_type': 'paper',
            'strategy_name': 'TestStrategy',
            'entry_logic_name': 'RSIEntry',
            'exit_logic_name': 'ATRExit',
            'symbol': trading_pair,
            'interval': '1h',
            'entry_time': datetime.now(timezone.utc),
            'buy_order_created': datetime.now(timezone.utc),
            'entry_price': 50000.0,
            'entry_value': 1000.0,
            'size': 0.02,
            'direction': 'long',
            'status': 'open',
            'extra_metadata': {'test': True}
        }

        trade = trades_repo.create(trade_data)
        db_session.flush()
        print(f"✅ Created test open trade: {trade.id}")

        # Simulate restart - load open positions
        all_trades = trades_repo.get_all()
        open_trades = [
            t for t in all_trades
            if t.bot_id == bot_id and t.symbol == trading_pair and t.status == 'open'
        ]

        assert len(open_trades) == 1
        print(f"✅ Found {len(open_trades)} open trade on restart")

        # Verify trade details
        open_trade = open_trades[0]
        assert open_trade.symbol == trading_pair
        assert open_trade.status == 'open'
        print(f"✅ Open trade details: {open_trade.symbol} @ {open_trade.entry_price}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
