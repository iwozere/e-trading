"""
Trading Models

SQLAlchemy models for the trading system.
Includes BotInstance, Trade, Position, and PerformanceMetric models.
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    Column, Integer, String, DateTime, Text, Numeric, Boolean, ForeignKey, func
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from src.data.db.core.base import Base


class BotInstance(Base):
    """Trading bot instance model."""

    __tablename__ = "trading_bots"

    id = Column(String(50), primary_key=True)
    type = Column(String(20), nullable=False)  # 'paper' or 'live'
    status = Column(String(20), nullable=False)  # 'running', 'stopped', etc.
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<BotInstance(id='{self.id}', type='{self.type}', status='{self.status}')>"


class Trade(Base):
    """Trade execution model."""

    __tablename__ = "trading_trades"

    id = Column(Integer, primary_key=True, index=True)
    bot_id = Column(String(50), ForeignKey("trading_bots.id"), nullable=False)
    trade_type = Column(String(10), nullable=False)  # 'paper' or 'live'
    symbol = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)  # 'buy' or 'sell'
    quantity = Column(Numeric(20, 8), nullable=False)
    price = Column(Numeric(20, 8), nullable=False)
    executed_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    strategy_name = Column(String(100), nullable=True)
    order_id = Column(String(100), nullable=True)
    commission = Column(Numeric(20, 8), nullable=True, default=0)

    # Relationship
    bot = relationship("BotInstance", backref="trades")

    def __repr__(self):
        return f"<Trade(id={self.id}, symbol='{self.symbol}', side='{self.side}', quantity={self.quantity})>"


class Position(Base):
    """Trading position model."""

    __tablename__ = "trading_positions"

    id = Column(Integer, primary_key=True, index=True)
    bot_id = Column(String(50), ForeignKey("trading_bots.id"), nullable=False)
    trade_type = Column(String(10), nullable=False)  # 'paper' or 'live'
    symbol = Column(String(20), nullable=False)
    quantity = Column(Numeric(20, 8), nullable=False)
    avg_price = Column(Numeric(20, 8), nullable=False)
    current_price = Column(Numeric(20, 8), nullable=True)
    unrealized_pnl = Column(Numeric(20, 8), nullable=True, default=0)
    realized_pnl = Column(Numeric(20, 8), nullable=True, default=0)
    opened_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
    closed_at = Column(DateTime(timezone=True), nullable=True)
    is_open = Column(Boolean, nullable=False, default=True)

    # Relationship
    bot = relationship("BotInstance", backref="positions")

    def __repr__(self):
        return f"<Position(id={self.id}, symbol='{self.symbol}', quantity={self.quantity}, is_open={self.is_open})>"


class PerformanceMetric(Base):
    """Performance metrics model."""

    __tablename__ = "trading_performance_metrics"

    id = Column(Integer, primary_key=True, index=True)
    bot_id = Column(String(50), ForeignKey("trading_bots.id"), nullable=False)
    trade_type = Column(String(10), nullable=False)  # 'paper' or 'live'
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Numeric(20, 8), nullable=False)
    calculated_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    period_start = Column(DateTime(timezone=True), nullable=True)
    period_end = Column(DateTime(timezone=True), nullable=True)

    # Relationship
    bot = relationship("BotInstance", backref="performance_metrics")

    def __repr__(self):
        return f"<PerformanceMetric(id={self.id}, bot_id='{self.bot_id}', metric='{self.metric_name}', value={self.metric_value})>"