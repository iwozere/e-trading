# model_trading.py  (aligned to DB)

from __future__ import annotations
from decimal import Decimal
from typing import Optional
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import (
    String, Integer, DateTime, JSON, ForeignKey, CheckConstraint,
    Index, text, Numeric
)

from src.data.db.core.base import Base

# --- trading_bot_instances
class BotInstance(Base):
    __tablename__ = "trading_bots"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("usr_users.id", ondelete="CASCADE"), index=True)
    type: Mapped[str] = mapped_column(String(20))
    config: Mapped[Optional[str]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(20))
    started_at: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True))
    last_heartbeat: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True))
    error_count: Mapped[Optional[int]] = mapped_column(Integer)
    current_balance: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8))
    total_pnl: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8))
    extra_metadata: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))
    updated_at: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True))

    trades: Mapped[list["Trade"]] = relationship(back_populates="bot", cascade="all, delete-orphan")
    positions: Mapped[list["Position"]] = relationship(back_populates="bot", cascade="all, delete-orphan")
    metrics: Mapped[list["PerformanceMetric"]] = relationship(back_populates="bot", cascade="all, delete-orphan")

    __table_args__ = (
        CheckConstraint("type IN ('live','paper','optimization')", name="ck_trading_bots_valid_bot_type"),
        CheckConstraint("status IN ('running','stopped','error','completed')", name="ck_trading_bots_valid_bot_status"),
        Index("ix_trading_bots_type", "type"),
        Index("ix_trading_bots_status", "status"),
        Index("ix_trading_bots_last_heartbeat", "last_heartbeat"),
    )

# --- trading_performance_metrics
class PerformanceMetric(Base):
    __tablename__ = "trading_performance_metrics"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    bot_id: Mapped[str] = mapped_column(ForeignKey("trading_bot_instances.id", ondelete="CASCADE"))
    trade_type: Mapped[str] = mapped_column(String(10))
    symbol: Mapped[Optional[str]] = mapped_column(String(20))
    interval: Mapped[Optional[str]] = mapped_column(String(10))
    entry_logic_name: Mapped[Optional[str]] = mapped_column(String(100))
    exit_logic_name: Mapped[Optional[str]] = mapped_column(String(100))
    metrics: Mapped[dict] = mapped_column(JSON)
    calculated_at: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))

    bot: Mapped["BotInstance"] = relationship(back_populates="metrics")

    __table_args__ = (
        CheckConstraint("trade_type IN ('paper','live','optimization')", name="ck_trading_performance_metrics_trade_type"),
        Index("ix_trading_performance_metrics_bot_id", "bot_id"),
        Index("ix_trading_performance_metrics_calculated_at", "calculated_at"),
        Index("ix_trading_performance_metrics_symbol", "symbol"),
    )

# --- trading_trades
class Trade(Base):
    __tablename__ = "trading_trades"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    bot_id: Mapped[str] = mapped_column(ForeignKey("trading_bot_instances.id", ondelete="CASCADE"))
    position_id: Mapped[Optional[str]] = mapped_column(ForeignKey("trading_positions.id", ondelete="SET NULL"), nullable=True)
    trade_type: Mapped[str] = mapped_column(String(10))
    strategy_name: Mapped[Optional[str]] = mapped_column(String(100))
    entry_logic_name: Mapped[str] = mapped_column(String(100))
    exit_logic_name: Mapped[str] = mapped_column(String(100))
    symbol: Mapped[str] = mapped_column(String(20))
    interval: Mapped[str] = mapped_column(String(10))
    entry_time: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True))
    exit_time: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True))
    buy_order_created: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True))
    buy_order_closed: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True))
    sell_order_created: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True))
    sell_order_closed: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True))
    entry_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8))
    exit_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8))
    entry_value: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8))
    exit_value: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8))
    size: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8))
    direction: Mapped[str] = mapped_column(String(10))
    commission: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8))
    gross_pnl: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8))
    net_pnl: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8))
    pnl_percentage: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))
    exit_reason: Mapped[Optional[str]] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(20))
    extra_metadata: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))
    updated_at: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True))

    bot: Mapped["BotInstance"] = relationship(back_populates="trades")
    position: Mapped[Optional["Position"]] = relationship(back_populates="trades")

    __table_args__ = (
        CheckConstraint("trade_type IN ('paper','live','optimization')", name="ck_trading_trades_trade_type"),
        CheckConstraint("direction IN ('long','short')", name="ck_trading_trades_direction"),
        CheckConstraint("status IN ('open','closed','cancelled')", name="ck_trading_trades_status"),
        Index("ix_trading_trades_entry_time", "entry_time"),
        Index("ix_trading_trades_bot_id", "bot_id"),
        Index("ix_trading_trades_trade_type", "trade_type"),
        Index("ix_trading_trades_symbol", "symbol"),
        Index("ix_trading_trades_status", "status"),
        Index("ix_trading_trades_strategy_name", "strategy_name"),
    )

# --- trading_positions
class Position(Base):
    __tablename__ = "trading_positions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    bot_id: Mapped[str] = mapped_column(ForeignKey("trading_bot_instances.id", ondelete="CASCADE"))
    trade_type: Mapped[str] = mapped_column(String(10))
    symbol: Mapped[str] = mapped_column(String(20))
    direction: Mapped[str] = mapped_column(String(10))
    opened_at: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True))
    closed_at: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True))
    qty_open: Mapped[Decimal] = mapped_column(Numeric(20, 8), server_default=text("0"), nullable=False)
    avg_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8))
    realized_pnl: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8), server_default=text("0"))
    status: Mapped[str] = mapped_column(String(12))
    extra_metadata: Mapped[Optional[dict]] = mapped_column(JSON)

    bot: Mapped["BotInstance"] = relationship(back_populates="positions")
    trades: Mapped[list["Trade"]] = relationship(back_populates="position")

    __table_args__ = (
        CheckConstraint("direction IN ('long','short')", name="ck_trading_positions_direction"),
        CheckConstraint("status IN ('open','closed')", name="ck_trading_positions_status"),
        Index("ix_trading_positions_bot_id", "bot_id"),
        Index("ix_trading_positions_symbol", "symbol"),
    )
