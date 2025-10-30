"""
Trading Models

SQLAlchemy models for the trading system.
Includes BotInstance, Trade, Position, and PerformanceMetric models.
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    Column, Integer, String, DateTime, Text, Numeric, Boolean, ForeignKey, func, Index
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from src.data.db.core.base import Base


class BotInstance(Base):
    """Trading bot instance model."""

    __tablename__ = "trading_bots"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("usr_users.id", ondelete="CASCADE"), nullable=False)
    type = Column(String(20), nullable=False)  # 'paper' or 'live'
    status = Column(String(20), nullable=False)  # 'running', 'stopped', etc.
    started_at = Column(DateTime, nullable=True)
    last_heartbeat = Column(DateTime, nullable=True)
    error_count = Column(Integer, nullable=True)
    current_balance = Column(Numeric(20, 8), nullable=True)
    total_pnl = Column(Numeric(20, 8), nullable=True)
    extra_metadata = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=True, default=func.now())
    updated_at = Column(DateTime, nullable=True)
    config = Column(JSONB, nullable=False)
    description = Column(Text, nullable=True)

    def __repr__(self):
        return f"<BotInstance(id={self.id}, type='{self.type}', status='{self.status}')>"


class Trade(Base):
    """Trade execution model."""

    __tablename__ = "trading_trades"

    id = Column(Integer, primary_key=True, index=True)
    bot_id = Column(Integer, ForeignKey("trading_bots.id", ondelete="CASCADE"), nullable=False)
    trade_type = Column(String(10), nullable=False)  # 'paper' or 'live'
    strategy_name = Column(String(100), nullable=True)
    entry_logic_name = Column(String(100), nullable=False)
    exit_logic_name = Column(String(100), nullable=False)
    symbol = Column(String(20), nullable=False)
    interval = Column(String(10), nullable=False)
    entry_time = Column(DateTime, nullable=True)
    exit_time = Column(DateTime, nullable=True)
    buy_order_created = Column(DateTime, nullable=True)
    buy_order_closed = Column(DateTime, nullable=True)
    sell_order_created = Column(DateTime, nullable=True)
    sell_order_closed = Column(DateTime, nullable=True)
    entry_price = Column(Numeric(20, 8), nullable=True)
    exit_price = Column(Numeric(20, 8), nullable=True)
    entry_value = Column(Numeric(20, 8), nullable=True)
    exit_value = Column(Numeric(20, 8), nullable=True)
    size = Column(Numeric(20, 8), nullable=True)
    direction = Column(String(10), nullable=False)
    commission = Column(Numeric(20, 8), nullable=True)
    gross_pnl = Column(Numeric(20, 8), nullable=True)
    net_pnl = Column(Numeric(20, 8), nullable=True)
    pnl_percentage = Column(Numeric(10, 4), nullable=True)
    exit_reason = Column(String(100), nullable=True)
    status = Column(String(20), nullable=False)
    extra_metadata = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=True, default=func.now())
    updated_at = Column(DateTime, nullable=True)
    position_id = Column(Integer, ForeignKey("trading_positions.id", ondelete="SET NULL"), nullable=True)

    # Relationships
    bot = relationship("BotInstance", backref="trades")
    position = relationship("Position", backref="trades")

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

    id = Column(Integer, primary_key=True, index=True)
    bot_id = Column(Integer, ForeignKey("trading_bots.id", ondelete="CASCADE"), nullable=False)
    trade_type = Column(String(10), nullable=False)  # 'paper' or 'live'
    symbol = Column(String(20), nullable=False)
    direction = Column(String(10), nullable=False)
    opened_at = Column(DateTime, nullable=True)
    closed_at = Column(DateTime, nullable=True)
    qty_open = Column(Numeric(20, 8), nullable=False, default=0)
    avg_price = Column(Numeric(20, 8), nullable=True)
    realized_pnl = Column(Numeric(20, 8), nullable=True, default=0)
    status = Column(String(12), nullable=False)
    extra_metadata = Column(JSONB, nullable=True)

    # Relationship
    bot = relationship("BotInstance", backref="positions")

    __table_args__ = (
        Index("ix_trading_positions_bot_id", "bot_id"),
        Index("ix_trading_positions_symbol", "symbol"),
    )

    def __repr__(self):
        return f"<Position(id={self.id}, symbol='{self.symbol}', qty_open={self.qty_open}, status='{self.status}')>"


class PerformanceMetric(Base):
    """Performance metrics model."""

    __tablename__ = "trading_performance_metrics"

    id = Column(Integer, primary_key=True, index=True)
    bot_id = Column(Integer, ForeignKey("trading_bots.id", ondelete="CASCADE"), nullable=False)
    trade_type = Column(String(10), nullable=False)  # 'paper' or 'live'
    symbol = Column(String(20), nullable=True)
    interval = Column(String(10), nullable=True)
    entry_logic_name = Column(String(100), nullable=True)
    exit_logic_name = Column(String(100), nullable=True)
    metrics = Column(JSONB, nullable=False)
    calculated_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=True, default=func.now())

    # Relationship
    bot = relationship("BotInstance", backref="performance_metrics")

    __table_args__ = (
        Index("ix_trading_performance_metrics_bot_id", "bot_id"),
        Index("ix_trading_performance_metrics_calculated_at", "calculated_at"),
        Index("ix_trading_performance_metrics_symbol", "symbol"),
    )

    def __repr__(self):
        return f"<PerformanceMetric(id={self.id}, bot_id={self.bot_id}, symbol='{self.symbol}')>"