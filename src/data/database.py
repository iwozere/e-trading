"""
Database Models for Crypto Trading Platform
------------------------------------------

This module contains SQLAlchemy models for:
- Trades: Complete trade lifecycle tracking
- Bot Instances: Bot session management
- Performance Metrics: Strategy performance tracking

Features:
- UUID primary keys for distributed systems
- Proper data types for financial calculations
- Audit trails with created_at/updated_at
- JSONB fields for flexible metadata storage
- Comprehensive indexing for performance
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    CheckConstraint, Column, DateTime, Index, Integer, String, Numeric
)
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

Base = declarative_base()


class Trade(Base):
    """
    Trade table for tracking complete trade lifecycle.
    
    Supports:
    - Live trading (paper and real)
    - Optimization backtesting
    - Order lifecycle tracking
    - Restart recovery
    """
    
    __tablename__ = 'trades'
    
    # Primary identification
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Bot/Config identification
    bot_id = Column(String(255), nullable=False, index=True)
    trade_type = Column(String(10), nullable=False, index=True)  # 'paper', 'live', 'optimization'
    
    # Strategy identification
    strategy_name = Column(String(100), nullable=True)
    entry_logic_name = Column(String(100), nullable=False)
    exit_logic_name = Column(String(100), nullable=False)
    
    # Trade identification
    symbol = Column(String(20), nullable=False, index=True)
    interval = Column(String(10), nullable=False)  # 15m, 1h, 4h, etc.
    
    # Trade timing
    entry_time = Column(DateTime, nullable=True)
    exit_time = Column(DateTime, nullable=True)
    
    # Order tracking (for live/paper trading)
    buy_order_created = Column(DateTime, nullable=True)
    buy_order_closed = Column(DateTime, nullable=True)
    sell_order_created = Column(DateTime, nullable=True)
    sell_order_closed = Column(DateTime, nullable=True)
    
    # Trade details
    entry_price = Column(Numeric(20, 8), nullable=True)
    exit_price = Column(Numeric(20, 8), nullable=True)
    entry_value = Column(Numeric(20, 8), nullable=True)
    exit_value = Column(Numeric(20, 8), nullable=True)
    size = Column(Numeric(20, 8), nullable=True)
    direction = Column(String(10), nullable=False)  # 'long', 'short'
    
    # Financial calculations
    commission = Column(Numeric(20, 8), nullable=True)
    gross_pnl = Column(Numeric(20, 8), nullable=True)
    net_pnl = Column(Numeric(20, 8), nullable=True)
    pnl_percentage = Column(Numeric(10, 4), nullable=True)
    
    # Trade metadata
    exit_reason = Column(String(100), nullable=True)
    status = Column(String(20), nullable=False, index=True)  # 'open', 'closed', 'cancelled'
    
    # Additional metadata
    extra_metadata = Column(JSON, nullable=True)  # Store additional context
    
    # System fields
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Constraints
    __table_args__ = (
        CheckConstraint(
            trade_type.in_(['paper', 'live', 'optimization']),
            name='valid_trade_type'
        ),
        CheckConstraint(
            direction.in_(['long', 'short']),
            name='valid_direction'
        ),
        CheckConstraint(
            status.in_(['open', 'closed', 'cancelled']),
            name='valid_status'
        ),
        Index('idx_symbol_status', 'symbol', 'status'),
        Index('idx_entry_time', 'entry_time'),
        Index('idx_strategy', 'entry_logic_name', 'exit_logic_name'),
        Index('idx_bot_trade_type', 'bot_id', 'trade_type'),
    )
    
    def __repr__(self):
        return f"<Trade(id={self.id}, symbol={self.symbol}, status={self.status}, bot_id={self.bot_id})>"
    
    def to_dict(self):
        """Convert trade to dictionary for JSON serialization."""
        return {
            'id': str(self.id),
            'bot_id': self.bot_id,
            'trade_type': self.trade_type,
            'strategy_name': self.strategy_name,
            'entry_logic_name': self.entry_logic_name,
            'exit_logic_name': self.exit_logic_name,
            'symbol': self.symbol,
            'interval': self.interval,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'buy_order_created': self.buy_order_created.isoformat() if self.buy_order_created else None,
            'buy_order_closed': self.buy_order_closed.isoformat() if self.buy_order_closed else None,
            'sell_order_created': self.sell_order_created.isoformat() if self.sell_order_created else None,
            'sell_order_closed': self.sell_order_closed.isoformat() if self.sell_order_closed else None,
            'entry_price': float(self.entry_price) if self.entry_price else None,
            'exit_price': float(self.exit_price) if self.exit_price else None,
            'entry_value': float(self.entry_value) if self.entry_value else None,
            'exit_value': float(self.exit_value) if self.exit_value else None,
            'size': float(self.size) if self.size else None,
            'direction': self.direction,
            'commission': float(self.commission) if self.commission else None,
            'gross_pnl': float(self.gross_pnl) if self.gross_pnl else None,
            'net_pnl': float(self.net_pnl) if self.net_pnl else None,
            'pnl_percentage': float(self.pnl_percentage) if self.pnl_percentage else None,
            'exit_reason': self.exit_reason,
            'status': self.status,
            'metadata': self.extra_metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class BotInstance(Base):
    """
    Bot instance tracking for managing bot lifecycle.
    
    Tracks:
    - Bot sessions (live/paper trading)
    - Optimization runs
    - Performance metrics
    - Error tracking
    """
    
    __tablename__ = 'bot_instances'
    
    # Primary identification
    id = Column(String(255), primary_key=True)  # Config filename or optimization result filename
    
    # Bot type and configuration
    type = Column(String(20), nullable=False)  # 'live', 'paper', 'optimization'
    config_file = Column(String(255), nullable=True)  # Full path to config file
    
    # Status tracking
    status = Column(String(20), nullable=False, default='stopped')  # 'running', 'stopped', 'error', 'completed'
    started_at = Column(DateTime, default=datetime.utcnow)
    last_heartbeat = Column(DateTime, default=datetime.utcnow)
    error_count = Column(Integer, default=0)
    
    # Performance tracking
    current_balance = Column(Numeric(20, 8), nullable=True)
    total_pnl = Column(Numeric(20, 8), nullable=True)
    
    # Additional metadata
    extra_metadata = Column(JSON, nullable=True)  # Store additional config info
    
    # System fields
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Constraints
    __table_args__ = (
        CheckConstraint(
            type.in_(['live', 'paper', 'optimization']),
            name='valid_bot_type'
        ),
        CheckConstraint(
            status.in_(['running', 'stopped', 'error', 'completed']),
            name='valid_bot_status'
        ),
        Index('idx_bot_type', 'type'),
        Index('idx_bot_status', 'status'),
        Index('idx_last_heartbeat', 'last_heartbeat'),
    )
    
    def __repr__(self):
        return f"<BotInstance(id={self.id}, type={self.type}, status={self.status})>"
    
    def to_dict(self):
        """Convert bot instance to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'type': self.type,
            'config_file': self.config_file,
            'status': self.status,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'last_heartbeat': self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            'error_count': self.error_count,
            'current_balance': float(self.current_balance) if self.current_balance else None,
            'total_pnl': float(self.total_pnl) if self.total_pnl else None,
            'metadata': self.extra_metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class PerformanceMetrics(Base):
    """
    Performance metrics table for storing strategy performance data.
    
    Stores:
    - Sharpe ratio, win rate, profit factor
    - Drawdown analysis
    - Risk metrics
    - Strategy parameters
    """
    
    __tablename__ = 'performance_metrics'
    
    # Primary identification
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Bot identification
    bot_id = Column(String(255), nullable=False, index=True)
    trade_type = Column(String(10), nullable=False)  # 'paper', 'live', 'optimization'
    
    # Strategy identification
    symbol = Column(String(20), nullable=True)
    interval = Column(String(10), nullable=True)
    entry_logic_name = Column(String(100), nullable=True)
    exit_logic_name = Column(String(100), nullable=True)
    
    # Performance metrics
    metrics = Column(JSON, nullable=False)  # Store all performance metrics
    
    # Calculation metadata
    calculated_at = Column(DateTime, default=datetime.utcnow)
    
    # System fields
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Constraints
    __table_args__ = (
        CheckConstraint(
            trade_type.in_(['paper', 'live', 'optimization']),
            name='valid_metrics_trade_type'
        ),
        Index('idx_metrics_bot_id', 'bot_id'),
        Index('idx_metrics_calculated_at', 'calculated_at'),
        Index('idx_metrics_strategy', 'entry_logic_name', 'exit_logic_name'),
    )
    
    def __repr__(self):
        return f"<PerformanceMetrics(id={self.id}, bot_id={self.bot_id}, calculated_at={self.calculated_at})>"
    
    def to_dict(self):
        """Convert performance metrics to dictionary for JSON serialization."""
        return {
            'id': str(self.id),
            'bot_id': self.bot_id,
            'trade_type': self.trade_type,
            'symbol': self.symbol,
            'interval': self.interval,
            'entry_logic_name': self.entry_logic_name,
            'exit_logic_name': self.exit_logic_name,
            'metrics': self.metrics,
            'calculated_at': self.calculated_at.isoformat() if self.calculated_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


# Database connection and session management
class DatabaseManager:
    """Database manager for handling connections and sessions."""
    
    def __init__(self, database_url: str = None):
        """
        Initialize database manager.
        
        Args:
            database_url: SQLAlchemy database URL. If None, uses SQLite for development.
        """
        if database_url is None:
            # Use SQLite for development
            database_url = "sqlite:///db/trading.db"
        
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self):
        """Get database session."""
        return self.SessionLocal()
    
    def close_session(self, session):
        """Close database session."""
        session.close()


# Global database manager instance
_db_manager = None


def get_database_manager(database_url: str = None) -> DatabaseManager:
    """Get or create database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(database_url)
    return _db_manager


def get_session():
    """Get database session."""
    return get_database_manager().get_session()


def close_session(session):
    """Close database session."""
    get_database_manager().close_session(session)
