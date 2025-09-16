"""
Unified Database Service
-----------------------

Central service for managing all database operations across the application.
Provides consistent access patterns and connection management.
"""

from typing import Optional
from contextlib import contextmanager

from src.data.database import DatabaseManager
from src.data.db.trade_repository import TradeRepository
from src.data.db.telegram_repository import TelegramRepository
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class DatabaseService:
    """Unified database service for the entire application."""

    def __init__(self, trading_db_url: str = None, telegram_db_url: str = None):
        """
        Initialize database service.

        Args:
            trading_db_url: URL for trading database
            telegram_db_url: URL for telegram database (defaults to same as trading)
        """
        # For now, use single database until migration is complete
        self.trading_db_url = trading_db_url or "sqlite:///db/trading.db"
        self.telegram_db_url = telegram_db_url or self.trading_db_url

        self._trading_manager = None
        self._telegram_manager = None

    @property
    def trading_manager(self) -> DatabaseManager:
        """Get trading database manager."""
        if self._trading_manager is None:
            self._trading_manager = DatabaseManager(self.trading_db_url)
        return self._trading_manager

    @property
    def telegram_manager(self) -> DatabaseManager:
        """Get telegram database manager."""
        if self._telegram_manager is None:
            # Use same database for now
            self._telegram_manager = self.trading_manager
        return self._telegram_manager

    @contextmanager
    def get_trading_repo(self):
        """Get trading repository with automatic session management."""
        session = self.trading_manager.get_session()
        try:
            yield TradeRepository(session)
        finally:
            session.close()

    @contextmanager
    def get_telegram_repo(self):
        """Get telegram repository with automatic session management."""
        session = self.telegram_manager.get_session()
        try:
            yield TelegramRepository(session)
        finally:
            session.close()


# Global database service instance
_db_service = None


def get_database_service() -> DatabaseService:
    """Get or create global database service instance."""
    global _db_service
    if _db_service is None:
        _db_service = DatabaseService()
    return _db_service


def init_databases():
    """Initialize all databases and create tables."""
    service = get_database_service()

    # Initialize trading database
    trading_session = service.trading_manager.get_session()
    trading_session.close()

    # Initialize telegram database (same for now)
    telegram_session = service.telegram_manager.get_session()
    telegram_session.close()

    _logger.info("Databases initialized successfully")