"""
Trading Model Tests

Tests for BotInstance, Trade, Position, and PerformanceMetric models
to validate the corrected PostgreSQL schema alignment.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

import pytest
from datetime import datetime
from decimal import Decimal
from sqlalchemy.exc import IntegrityError

from src.data.db.models.model_users import User
from src.data.db.models.model_trading import BotInstance, Trade, Position, PerformanceMetric


@pytest.fixture
def test_user(dbsess):
    """Create a test user for relationship testing."""
    # Check if user already exists
    existing_user = dbsess.query(User).filter_by(email="trader@example.com").first()
    if existing_user:
        return existing_user

    user = User(
        email="trader@example.com",
        role="trader",
        is_active=True
    )
    dbsess.add(user)
    dbsess.commit()
    return user


@pytest.fixture
def test_bot(dbsess, test_user):
    """Create a test bot instance."""
    bot = BotInstance(
        user_id=test_user.id,
        type="paper",
        status="running",
        config={"strategy": "test", "balance": 1000},
        description="Test bot"
    )
    dbsess.add(bot)
    dbsess.commit()
    return bot


class TestBotInstanceModel:
    """Test BotInstance model with corrected PostgreSQL schema."""

    def test_bot_creation(self, dbsess, test_user):
        """Test creating BotInstance with required fields."""
        bot = BotInstance(
            user_id=test_user.id,
            type="paper",
            status="running",
            config={"strategy": "rsi_bb", "initial_balance": 10000}
        )

        dbsess.add(bot)
        dbsess.commit()

        # Verify the record was created
        assert bot.id is not None
        assert bot.user_id == test_user.id
        assert bot.type == "paper"
        assert bot.status == "running"
        assert bot.config["strategy"] == "rsi_bb"

    def test_bot_with_optional_fields(self, dbsess, test_user):
        """Test BotInstance with all optional fields."""
        bot = BotInstance(
            user_id=test_user.id,
            type="live",
            status="stopped",
            started_at=datetime.now(),
            last_heartbeat=datetime.now(),
            error_count=2,
            current_balance=Decimal("9500.50"),
            total_pnl=Decimal("-499.50"),
            extra_metadata={"exchange": "binance", "api_key_id": "test123"},
            config={"strategy": "macd", "risk_level": "medium"},
            description="Live trading bot for BTCUSDT"
        )

        dbsess.add(bot)
        dbsess.commit()

        assert bot.current_balance == Decimal("9500.50")
        assert bot.total_pnl == Decimal("-499.50")
        assert bot.extra_metadata["exchange"] == "binance"
        assert bot.description == "Live trading bot for BTCUSDT"

    def test_bot_user_relationship(self, dbsess, test_user):
        """Test foreign key relationship to User."""
        bot = BotInstance(
            user_id=test_user.id,
            type="paper",
            status="running",
            config={"test": True}
        )

        dbsess.add(bot)
        dbsess.commit()

        # Test the relationship works
        assert bot.user_id == test_user.id

    def test_bot_config_required(self, dbsess, test_user):
        """Test that config field is required."""
        bot = BotInstance(
            user_id=test_user.id,
            type="paper",
            status="running"
            # Missing required config field
        )

        dbsess.add(bot)

        with pytest.raises(IntegrityError):
            dbsess.commit()


class TestTradeModel:
    """Test Trade model with corrected PostgreSQL schema."""

    def test_trade_creation(self, dbsess, test_bot):
        """Test creating Trade with required fields."""
        trade = Trade(
            bot_id=test_bot.id,
            trade_type="paper",
            entry_logic_name="RSIBBVolumeEntryMixin",
            exit_logic_name="ATRExitMixin",
            symbol="BTCUSDT",
            interval="1h",
            direction="long",
            status="open"
        )

        dbsess.add(trade)
        dbsess.commit()

        assert trade.id is not None
        assert trade.bot_id == test_bot.id
        assert trade.entry_logic_name == "RSIBBVolumeEntryMixin"
        assert trade.exit_logic_name == "ATRExitMixin"
        assert trade.symbol == "BTCUSDT"
        assert trade.direction == "long"

    def test_trade_with_all_fields(self, dbsess, test_bot):
        """Test Trade with all optional fields."""
        now = datetime.now()
        trade = Trade(
            bot_id=test_bot.id,
            trade_type="live",
            strategy_name="Advanced RSI Strategy",
            entry_logic_name="RSIBBVolumeEntryMixin",
            exit_logic_name="ATRExitMixin",
            symbol="ETHUSDT",
            interval="4h",
            entry_time=now,
            exit_time=now,
            buy_order_created=now,
            buy_order_closed=now,
            sell_order_created=now,
            sell_order_closed=now,
            entry_price=Decimal("2500.50"),
            exit_price=Decimal("2600.75"),
            entry_value=Decimal("2500.50"),
            exit_value=Decimal("2600.75"),
            size=Decimal("1.0"),
            direction="long",
            commission=Decimal("2.50"),
            gross_pnl=Decimal("100.25"),
            net_pnl=Decimal("97.75"),
            pnl_percentage=Decimal("3.91"),
            exit_reason="take_profit",
            status="closed",
            extra_metadata={"order_ids": ["123", "456"]}
        )

        dbsess.add(trade)
        dbsess.commit()

        assert trade.entry_price == Decimal("2500.50")
        assert trade.net_pnl == Decimal("97.75")
        assert trade.pnl_percentage == Decimal("3.91")
        assert trade.exit_reason == "take_profit"
        assert trade.extra_metadata["order_ids"] == ["123", "456"]

    def test_trade_bot_relationship(self, dbsess, test_bot):
        """Test foreign key relationship to BotInstance."""
        trade = Trade(
            bot_id=test_bot.id,
            trade_type="paper",
            entry_logic_name="TestEntry",
            exit_logic_name="TestExit",
            symbol="BTCUSDT",
            interval="1h",
            direction="long",
            status="open"
        )

        dbsess.add(trade)
        dbsess.commit()

        # Test the relationship
        assert trade.bot.id == test_bot.id
        assert trade.bot.type == test_bot.type


class TestPositionModel:
    """Test Position model with corrected PostgreSQL schema."""

    def test_position_creation(self, dbsess, test_bot):
        """Test creating Position with required fields."""
        position = Position(
            bot_id=test_bot.id,
            trade_type="paper",
            symbol="BTCUSDT",
            direction="long",
            qty_open=Decimal("0.5"),
            status="open"
        )

        dbsess.add(position)
        dbsess.commit()

        assert position.id is not None
        assert position.bot_id == test_bot.id
        assert position.symbol == "BTCUSDT"
        assert position.direction == "long"
        assert position.qty_open == Decimal("0.5")
        assert position.status == "open"

    def test_position_with_optional_fields(self, dbsess, test_bot):
        """Test Position with all optional fields."""
        now = datetime.now()
        position = Position(
            bot_id=test_bot.id,
            trade_type="live",
            symbol="ETHUSDT",
            direction="short",
            opened_at=now,
            closed_at=now,
            qty_open=Decimal("2.5"),
            avg_price=Decimal("2500.00"),
            realized_pnl=Decimal("150.75"),
            status="closed",
            extra_metadata={"notes": "Profitable short position"}
        )

        dbsess.add(position)
        dbsess.commit()

        assert position.avg_price == Decimal("2500.00")
        assert position.realized_pnl == Decimal("150.75")
        assert position.status == "closed"
        assert position.extra_metadata["notes"] == "Profitable short position"

    def test_position_bot_relationship(self, dbsess, test_bot):
        """Test foreign key relationship to BotInstance."""
        position = Position(
            bot_id=test_bot.id,
            trade_type="paper",
            symbol="BTCUSDT",
            direction="long",
            qty_open=Decimal("1.0"),
            status="open"
        )

        dbsess.add(position)
        dbsess.commit()

        # Test the relationship
        assert position.bot.id == test_bot.id
        assert position.bot.type == test_bot.type


class TestPerformanceMetricModel:
    """Test PerformanceMetric model with corrected PostgreSQL schema."""

    def test_performance_metric_creation(self, dbsess, test_bot):
        """Test creating PerformanceMetric with required fields."""
        metric = PerformanceMetric(
            bot_id=test_bot.id,
            trade_type="paper",
            metrics={
                "sharpe_ratio": 1.5,
                "win_rate": 0.65,
                "total_trades": 100,
                "profit_factor": 2.1
            }
        )

        dbsess.add(metric)
        dbsess.commit()

        assert metric.id is not None
        assert metric.bot_id == test_bot.id
        assert metric.metrics["sharpe_ratio"] == 1.5
        assert metric.metrics["win_rate"] == 0.65

    def test_performance_metric_with_optional_fields(self, dbsess, test_bot):
        """Test PerformanceMetric with all optional fields."""
        now = datetime.now()
        metric = PerformanceMetric(
            bot_id=test_bot.id,
            trade_type="live",
            symbol="BTCUSDT",
            interval="1h",
            entry_logic_name="RSIBBVolumeEntryMixin",
            exit_logic_name="ATRExitMixin",
            metrics={
                "sharpe_ratio": 2.3,
                "win_rate": 0.72,
                "total_trades": 250,
                "profit_factor": 3.2,
                "max_drawdown": 0.15,
                "total_return": 0.45
            },
            calculated_at=now
        )

        dbsess.add(metric)
        dbsess.commit()

        assert metric.symbol == "BTCUSDT"
        assert metric.interval == "1h"
        assert metric.entry_logic_name == "RSIBBVolumeEntryMixin"
        assert metric.metrics["profit_factor"] == 3.2
        assert metric.calculated_at == now

    def test_performance_metric_bot_relationship(self, dbsess, test_bot):
        """Test foreign key relationship to BotInstance."""
        metric = PerformanceMetric(
            bot_id=test_bot.id,
            trade_type="paper",
            metrics={"test_metric": 1.0}
        )

        dbsess.add(metric)
        dbsess.commit()

        # Test the relationship
        assert metric.bot.id == test_bot.id
        assert metric.bot.type == test_bot.type


class TestTradingModelRelationships:
    """Test relationships between trading models."""

    def test_trade_position_relationship(self, dbsess, test_bot):
        """Test Trade to Position foreign key relationship."""
        # Create a position first
        position = Position(
            bot_id=test_bot.id,
            trade_type="paper",
            symbol="BTCUSDT",
            direction="long",
            qty_open=Decimal("1.0"),
            status="open"
        )
        dbsess.add(position)
        dbsess.commit()

        # Create a trade that references the position
        trade = Trade(
            bot_id=test_bot.id,
            trade_type="paper",
            entry_logic_name="TestEntry",
            exit_logic_name="TestExit",
            symbol="BTCUSDT",
            interval="1h",
            direction="long",
            status="open",
            position_id=position.id
        )
        dbsess.add(trade)
        dbsess.commit()

        # Test the relationship
        assert trade.position_id == position.id
        assert trade.position.symbol == "BTCUSDT"

    def test_cascade_delete_behavior(self, dbsess, test_user):
        """Test foreign key cascade behavior."""
        # Create bot
        bot = BotInstance(
            user_id=test_user.id,
            type="paper",
            status="running",
            config={"test": True}
        )
        dbsess.add(bot)
        dbsess.commit()
        bot_id = bot.id

        # Create related records
        trade = Trade(
            bot_id=bot_id,
            trade_type="paper",
            entry_logic_name="TestEntry",
            exit_logic_name="TestExit",
            symbol="BTCUSDT",
            interval="1h",
            direction="long",
            status="open"
        )
        position = Position(
            bot_id=bot_id,
            trade_type="paper",
            symbol="BTCUSDT",
            direction="long",
            qty_open=Decimal("1.0"),
            status="open"
        )
        metric = PerformanceMetric(
            bot_id=bot_id,
            trade_type="paper",
            metrics={"test": 1.0}
        )

        dbsess.add_all([trade, position, metric])
        dbsess.commit()

        # Delete the bot - should cascade to related records
        dbsess.delete(bot)
        dbsess.commit()

        # Verify related records were deleted
        assert dbsess.get(Trade, trade.id) is None
        assert dbsess.get(Position, position.id) is None
        assert dbsess.get(PerformanceMetric, metric.id) is None