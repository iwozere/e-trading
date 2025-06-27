"""
Trade Repository for Database Operations
---------------------------------------

This module provides a repository pattern for database operations:
- Trade CRUD operations
- Bot instance management
- Performance metrics storage
- Query and filtering capabilities

Features:
- Transaction management
- Error handling
- Query optimization
- Data validation
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc

from src.data.database import Trade, BotInstance, PerformanceMetrics, get_session, close_session

_logger = logging.getLogger(__name__)


class TradeRepository:
    """Repository for trade-related database operations."""
    
    def __init__(self, session: Session = None):
        """
        Initialize trade repository.
        
        Args:
            session: Database session. If None, creates a new session.
        """
        self.session = session or get_session()
        self._owns_session = session is None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with proper cleanup."""
        self.close()
    
    def close(self):
        """Close the database session."""
        if self._owns_session and self.session:
            close_session(self.session)
            self.session = None
    
    def commit(self):
        """Commit the current transaction."""
        try:
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            _logger.error(f"Error committing transaction: {e}")
            raise
    
    def rollback(self):
        """Rollback the current transaction."""
        self.session.rollback()
    
    # Trade Operations
    def create_trade(self, trade_data: Dict[str, Any]) -> Trade:
        """
        Create a new trade record.
        
        Args:
            trade_data: Dictionary containing trade data
            
        Returns:
            Created Trade object
        """
        try:
            trade = Trade(**trade_data)
            self.session.add(trade)
            self.commit()
            _logger.info(f"Created trade: {trade.id} for {trade.symbol}")
            return trade
        except Exception as e:
            self.rollback()
            _logger.error(f"Error creating trade: {e}")
            raise
    
    def get_trade_by_id(self, trade_id: str) -> Optional[Trade]:
        """
        Get trade by ID.
        
        Args:
            trade_id: Trade UUID
            
        Returns:
            Trade object or None
        """
        try:
            return self.session.query(Trade).filter(Trade.id == trade_id).first()
        except Exception as e:
            _logger.error(f"Error getting trade by ID: {e}")
            return None
    
    def update_trade(self, trade_id: str, update_data: Dict[str, Any]) -> Optional[Trade]:
        """
        Update an existing trade.
        
        Args:
            trade_id: Trade UUID
            update_data: Dictionary containing fields to update
            
        Returns:
            Updated Trade object or None
        """
        try:
            trade = self.get_trade_by_id(trade_id)
            if trade:
                for key, value in update_data.items():
                    if hasattr(trade, key):
                        setattr(trade, key, value)
                trade.updated_at = datetime.utcnow()
                self.commit()
                _logger.info(f"Updated trade: {trade_id}")
                return trade
            return None
        except Exception as e:
            self.rollback()
            _logger.error(f"Error updating trade: {e}")
            raise
    
    def get_open_trades(self, bot_id: str = None, symbol: str = None) -> List[Trade]:
        """
        Get all open trades.
        
        Args:
            bot_id: Optional bot ID filter
            symbol: Optional symbol filter
            
        Returns:
            List of open Trade objects
        """
        try:
            query = self.session.query(Trade).filter(Trade.status == 'open')
            
            if bot_id:
                query = query.filter(Trade.bot_id == bot_id)
            
            if symbol:
                query = query.filter(Trade.symbol == symbol)
            
            return query.all()
        except Exception as e:
            _logger.error(f"Error getting open trades: {e}")
            return []
    
    def get_trades_by_bot(self, bot_id: str, limit: int = 100) -> List[Trade]:
        """
        Get trades for a specific bot.
        
        Args:
            bot_id: Bot ID
            limit: Maximum number of trades to return
            
        Returns:
            List of Trade objects
        """
        try:
            return (self.session.query(Trade)
                    .filter(Trade.bot_id == bot_id)
                    .order_by(desc(Trade.created_at))
                    .limit(limit)
                    .all())
        except Exception as e:
            _logger.error(f"Error getting trades by bot: {e}")
            return []
    
    def get_trades_by_symbol(self, symbol: str, limit: int = 100) -> List[Trade]:
        """
        Get trades for a specific symbol.
        
        Args:
            symbol: Trading symbol
            limit: Maximum number of trades to return
            
        Returns:
            List of Trade objects
        """
        try:
            return (self.session.query(Trade)
                    .filter(Trade.symbol == symbol)
                    .order_by(desc(Trade.created_at))
                    .limit(limit)
                    .all())
        except Exception as e:
            _logger.error(f"Error getting trades by symbol: {e}")
            return []
    
    def get_trades_by_date_range(self, start_date: datetime, end_date: datetime, 
                                bot_id: str = None, symbol: str = None) -> List[Trade]:
        """
        Get trades within a date range.
        
        Args:
            start_date: Start date
            end_date: End date
            bot_id: Optional bot ID filter
            symbol: Optional symbol filter
            
        Returns:
            List of Trade objects
        """
        try:
            query = self.session.query(Trade).filter(
                and_(
                    Trade.entry_time >= start_date,
                    Trade.entry_time <= end_date
                )
            )
            
            if bot_id:
                query = query.filter(Trade.bot_id == bot_id)
            
            if symbol:
                query = query.filter(Trade.symbol == symbol)
            
            return query.order_by(desc(Trade.entry_time)).all()
        except Exception as e:
            _logger.error(f"Error getting trades by date range: {e}")
            return []
    
    def get_closed_trades(self, bot_id: str = None, symbol: str = None, 
                         limit: int = 100) -> List[Trade]:
        """
        Get closed trades.
        
        Args:
            bot_id: Optional bot ID filter
            symbol: Optional symbol filter
            limit: Maximum number of trades to return
            
        Returns:
            List of closed Trade objects
        """
        try:
            query = self.session.query(Trade).filter(Trade.status == 'closed')
            
            if bot_id:
                query = query.filter(Trade.bot_id == bot_id)
            
            if symbol:
                query = query.filter(Trade.symbol == symbol)
            
            return query.order_by(desc(Trade.exit_time)).limit(limit).all()
        except Exception as e:
            _logger.error(f"Error getting closed trades: {e}")
            return []
    
    def delete_trade(self, trade_id: str) -> bool:
        """
        Delete a trade.
        
        Args:
            trade_id: Trade UUID
            
        Returns:
            True if deleted, False otherwise
        """
        try:
            trade = self.get_trade_by_id(trade_id)
            if trade:
                self.session.delete(trade)
                self.commit()
                _logger.info(f"Deleted trade: {trade_id}")
                return True
            return False
        except Exception as e:
            self.rollback()
            _logger.error(f"Error deleting trade: {e}")
            return False
    
    # Bot Instance Operations
    def create_bot_instance(self, bot_data: Dict[str, Any]) -> BotInstance:
        """
        Create a new bot instance.
        
        Args:
            bot_data: Dictionary containing bot instance data
            
        Returns:
            Created BotInstance object
        """
        try:
            bot_instance = BotInstance(**bot_data)
            self.session.add(bot_instance)
            self.commit()
            _logger.info(f"Created bot instance: {bot_instance.id}")
            return bot_instance
        except Exception as e:
            self.rollback()
            _logger.error(f"Error creating bot instance: {e}")
            raise
    
    def get_bot_instance(self, bot_id: str) -> Optional[BotInstance]:
        """
        Get bot instance by ID.
        
        Args:
            bot_id: Bot instance ID
            
        Returns:
            BotInstance object or None
        """
        try:
            return self.session.query(BotInstance).filter(BotInstance.id == bot_id).first()
        except Exception as e:
            _logger.error(f"Error getting bot instance: {e}")
            return None
    
    def update_bot_instance(self, bot_id: str, update_data: Dict[str, Any]) -> Optional[BotInstance]:
        """
        Update a bot instance.
        
        Args:
            bot_id: Bot instance ID
            update_data: Dictionary containing fields to update
            
        Returns:
            Updated BotInstance object or None
        """
        try:
            bot_instance = self.get_bot_instance(bot_id)
            if bot_instance:
                for key, value in update_data.items():
                    if hasattr(bot_instance, key):
                        setattr(bot_instance, key, value)
                bot_instance.updated_at = datetime.utcnow()
                self.commit()
                _logger.info(f"Updated bot instance: {bot_id}")
                return bot_instance
            return None
        except Exception as e:
            self.rollback()
            _logger.error(f"Error updating bot instance: {e}")
            raise
    
    def get_running_bots(self) -> List[BotInstance]:
        """
        Get all running bot instances.
        
        Returns:
            List of running BotInstance objects
        """
        try:
            return self.session.query(BotInstance).filter(BotInstance.status == 'running').all()
        except Exception as e:
            _logger.error(f"Error getting running bots: {e}")
            return []
    
    def get_bot_instances_by_type(self, bot_type: str) -> List[BotInstance]:
        """
        Get bot instances by type.
        
        Args:
            bot_type: Bot type ('live', 'paper', 'optimization')
            
        Returns:
            List of BotInstance objects
        """
        try:
            return self.session.query(BotInstance).filter(BotInstance.type == bot_type).all()
        except Exception as e:
            _logger.error(f"Error getting bot instances by type: {e}")
            return []
    
    # Performance Metrics Operations
    def create_performance_metrics(self, metrics_data: Dict[str, Any]) -> PerformanceMetrics:
        """
        Create performance metrics record.
        
        Args:
            metrics_data: Dictionary containing metrics data
            
        Returns:
            Created PerformanceMetrics object
        """
        try:
            metrics = PerformanceMetrics(**metrics_data)
            self.session.add(metrics)
            self.commit()
            _logger.info(f"Created performance metrics: {metrics.id}")
            return metrics
        except Exception as e:
            self.rollback()
            _logger.error(f"Error creating performance metrics: {e}")
            raise
    
    def get_performance_metrics(self, bot_id: str, limit: int = 10) -> List[PerformanceMetrics]:
        """
        Get performance metrics for a bot.
        
        Args:
            bot_id: Bot ID
            limit: Maximum number of records to return
            
        Returns:
            List of PerformanceMetrics objects
        """
        try:
            return (self.session.query(PerformanceMetrics)
                    .filter(PerformanceMetrics.bot_id == bot_id)
                    .order_by(desc(PerformanceMetrics.calculated_at))
                    .limit(limit)
                    .all())
        except Exception as e:
            _logger.error(f"Error getting performance metrics: {e}")
            return []
    
    # Utility Methods
    def get_trade_summary(self, bot_id: str = None, symbol: str = None) -> Dict[str, Any]:
        """
        Get trade summary statistics.
        
        Args:
            bot_id: Optional bot ID filter
            symbol: Optional symbol filter
            
        Returns:
            Dictionary with summary statistics
        """
        try:
            query = self.session.query(Trade)
            
            if bot_id:
                query = query.filter(Trade.bot_id == bot_id)
            
            if symbol:
                query = query.filter(Trade.symbol == symbol)
            
            total_trades = query.count()
            closed_trades = query.filter(Trade.status == 'closed').count()
            open_trades = query.filter(Trade.status == 'open').count()
            
            # Calculate total PnL
            closed_trades_list = query.filter(Trade.status == 'closed').all()
            total_pnl = sum(float(trade.net_pnl or 0) for trade in closed_trades_list)
            
            return {
                'total_trades': total_trades,
                'closed_trades': closed_trades,
                'open_trades': open_trades,
                'total_pnl': total_pnl,
                'win_rate': (len([t for t in closed_trades_list if t.net_pnl and float(t.net_pnl) > 0]) / closed_trades * 100) if closed_trades > 0 else 0
            }
        except Exception as e:
            _logger.error(f"Error getting trade summary: {e}")
            return {
                'total_trades': 0,
                'closed_trades': 0,
                'open_trades': 0,
                'total_pnl': 0,
                'win_rate': 0
            }
    
    def cleanup_old_data(self, days_to_keep: int = 90) -> int:
        """
        Clean up old trade data.
        
        Args:
            days_to_keep: Number of days of data to keep
            
        Returns:
            Number of records deleted
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            deleted_count = self.session.query(Trade).filter(
                Trade.created_at < cutoff_date
            ).delete()
            self.commit()
            _logger.info(f"Cleaned up {deleted_count} old trade records")
            return deleted_count
        except Exception as e:
            self.rollback()
            _logger.error(f"Error cleaning up old data: {e}")
            return 0 