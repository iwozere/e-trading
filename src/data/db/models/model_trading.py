"""
Trading Models

SQLAlchemy models for the trading system.
Includes BotInstance, Trade, Position, and PerformanceMetric models.
"""


from __future__ import annotations
from typing import Optional
from sqlalchemy import (
    Integer, String, DateTime, Text, Numeric, ForeignKey, func, Index,
    event
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.exc import IntegrityError
from src.data.db.core.json_types import JsonType

from src.data.db.core.base import Base


class BotInstance(Base):
    """Trading bot instance model."""

    __tablename__ = "trading_bots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("usr_users.id", ondelete="CASCADE"))
    status: Mapped[str] = mapped_column(String(20))  # 'running', 'stopped', etc.
    started_at: Mapped[DateTime | None] = mapped_column(DateTime)
    last_heartbeat: Mapped[DateTime | None] = mapped_column(DateTime)
    error_count: Mapped[int | None] = mapped_column(Integer)
    current_balance: Mapped[Numeric | None] = mapped_column(Numeric(20, 8))
    total_pnl: Mapped[Numeric | None] = mapped_column(Numeric(20, 8))
    extra_metadata: Mapped[dict | None] = mapped_column(JsonType())
    created_at: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True), default=func.now())
    updated_at: Mapped[DateTime | None] = mapped_column(DateTime)
    config: Mapped[dict] = mapped_column(JsonType(), info={"required": True})
    description: Mapped[str | None] = mapped_column(Text)

    # Relationships
    trades = relationship("Trade", back_populates="bot", cascade="all, delete-orphan")
    positions = relationship("Position", back_populates="bot", cascade="all, delete-orphan")
    performance_metrics = relationship("PerformanceMetric", back_populates="bot", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<BotInstance(id={self.id}, type='{self.type}', status='{self.status}')>"


class Trade(Base):
    """Trade execution model."""

    __tablename__ = "trading_trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    bot_id: Mapped[int] = mapped_column(ForeignKey("trading_bots.id", ondelete="CASCADE"))
    trade_type: Mapped[str] = mapped_column(String(10))  # 'paper' or 'live'
    strategy_name: Mapped[str | None] = mapped_column(String(100))
    entry_logic_name: Mapped[str] = mapped_column(String(100))
    exit_logic_name: Mapped[str] = mapped_column(String(100))
    symbol: Mapped[str] = mapped_column(String(20))
    interval: Mapped[str] = mapped_column(String(10))
    entry_time: Mapped[DateTime | None] = mapped_column(DateTime)
    exit_time: Mapped[DateTime | None] = mapped_column(DateTime)
    buy_order_created: Mapped[DateTime | None] = mapped_column(DateTime)
    buy_order_closed: Mapped[DateTime | None] = mapped_column(DateTime)
    sell_order_created: Mapped[DateTime | None] = mapped_column(DateTime)
    sell_order_closed: Mapped[DateTime | None] = mapped_column(DateTime)
    entry_price: Mapped[Numeric | None] = mapped_column(Numeric(20, 8))
    exit_price: Mapped[Numeric | None] = mapped_column(Numeric(20, 8))
    entry_value: Mapped[Numeric | None] = mapped_column(Numeric(20, 8))
    exit_value: Mapped[Numeric | None] = mapped_column(Numeric(20, 8))
    size: Mapped[Numeric | None] = mapped_column(Numeric(20, 8))
    direction: Mapped[str] = mapped_column(String(10))
    commission: Mapped[Numeric | None] = mapped_column(Numeric(20, 8))
    gross_pnl: Mapped[Numeric | None] = mapped_column(Numeric(20, 8))
    net_pnl: Mapped[Numeric | None] = mapped_column(Numeric(20, 8))
    pnl_percentage: Mapped[Numeric | None] = mapped_column(Numeric(10, 4))
    exit_reason: Mapped[str | None] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(20))
    extra_metadata: Mapped[dict | None] = mapped_column(JsonType())
    created_at: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True), default=func.now())
    updated_at: Mapped[DateTime | None] = mapped_column(DateTime)
    position_id: Mapped[int | None] = mapped_column(ForeignKey("trading_positions.id", ondelete="SET NULL"))

    # Relationships
    bot = relationship("BotInstance", foreign_keys=[bot_id], back_populates="trades")
    position = relationship("Position", foreign_keys=[position_id], back_populates="trades")

    __table_args__ = (
        Index("ix_trading_trades_bot_id", "bot_id"),
        Index("ix_trading_trades_entry_time", "entry_time"),
        Index("ix_trading_trades_status", "status"),
        Index("ix_trading_trades_strategy_name", "strategy_name"),
        Index("ix_trading_trades_symbol", "symbol"),
        Index("ix_trading_trades_trade_type", "trade_type"),
    )

    def __repr__(self):
        return f"<Trade(id={self.id}, symbol='{self.symbol}', direction='{self.direction}', size={self.size})>"


class Position(Base):
    """Trading position model."""

    __tablename__ = "trading_positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    bot_id: Mapped[int] = mapped_column(ForeignKey("trading_bots.id", ondelete="CASCADE"))
    trade_type: Mapped[str] = mapped_column(String(10))  # 'paper' or 'live'
    symbol: Mapped[str] = mapped_column(String(20))
    direction: Mapped[str] = mapped_column(String(10))
    opened_at: Mapped[DateTime | None] = mapped_column(DateTime)
    closed_at: Mapped[DateTime | None] = mapped_column(DateTime)
    qty_open: Mapped[Numeric] = mapped_column(Numeric(20, 8), default=0)
    avg_price: Mapped[Numeric | None] = mapped_column(Numeric(20, 8))
    realized_pnl: Mapped[Numeric | None] = mapped_column(Numeric(20, 8), default=0)
    status: Mapped[str] = mapped_column(String(12))
    extra_metadata: Mapped[dict | None] = mapped_column(JsonType())

    # Relationship
    bot = relationship("BotInstance", foreign_keys=[bot_id], back_populates="positions")
    trades = relationship("Trade", back_populates="position", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_trading_positions_bot_id", "bot_id"),
        Index("ix_trading_positions_symbol", "symbol"),
    )

    def __repr__(self):
        return f"<Position(id={self.id}, symbol='{self.symbol}', qty_open={self.qty_open}, status='{self.status}')>"


class PerformanceMetric(Base):
    """Performance metrics model."""

    __tablename__ = "trading_performance_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    bot_id: Mapped[int] = mapped_column(ForeignKey("trading_bots.id", ondelete="CASCADE"))
    trade_type: Mapped[str] = mapped_column(String(10))  # 'paper' or 'live'
    symbol: Mapped[str | None] = mapped_column(String(20))
    interval: Mapped[str | None] = mapped_column(String(10))
    entry_logic_name: Mapped[str | None] = mapped_column(String(100))
    exit_logic_name: Mapped[str | None] = mapped_column(String(100))
    metrics: Mapped[dict] = mapped_column(JsonType())
    calculated_at: Mapped[DateTime | None] = mapped_column(DateTime)
    created_at: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True), default=func.now())

    # Relationship
    bot = relationship("BotInstance", foreign_keys=[bot_id], back_populates="performance_metrics")

    __table_args__ = (
        Index("ix_trading_performance_metrics_bot_id", "bot_id"),
        Index("ix_trading_performance_metrics_calculated_at", "calculated_at"),
        Index("ix_trading_performance_metrics_symbol", "symbol"),
    )

    def __repr__(self):
        return f"<PerformanceMetric(id={self.id}, bot_id={self.bot_id}, symbol='{self.symbol}')>"


@event.listens_for(BotInstance, 'before_insert')
def check_config_required(mapper, connection, instance):
        """Ensure config is provided before insert.

        Note:
        - Do NOT call connection.rollback() inside event listeners. Doing so
            will deassociate the surrounding Session transaction and cause
            teardown warnings in tests (and can surprise callers elsewhere).
            Raise an exception instead and let the Session manage rollback.
        """
        if not hasattr(instance, 'config') or instance.config is None:
                # Let the Session handle rollback when this exception bubbles up.
                raise IntegrityError("Config field is required", None, None)