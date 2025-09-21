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

    def __init__(self, database_url: str = None):
        """
        Initialize database service with single consolidated database.

        Args:
            database_url: URL for the consolidated database
        """
        # Use single consolidated database
        self.database_url = database_url or "sqlite:///db/trading.db"
        self._database_manager = None

    @property
    def database_manager(self) -> DatabaseManager:
        """Get consolidated database manager."""
        if self._database_manager is None:
            self._database_manager = DatabaseManager(self.database_url)
        return self._database_manager

    @property
    def trading_manager(self) -> DatabaseManager:
        """Get trading database manager (same as consolidated database)."""
        return self.database_manager

    @property
    def telegram_manager(self) -> DatabaseManager:
        """Get telegram database manager (same as consolidated database)."""
        return self.database_manager

    @property
    def webui_manager(self) -> DatabaseManager:
        """Get web UI database manager (same as consolidated database)."""
        return self.database_manager

    @contextmanager
    def get_trading_repo(self):
        """Get trading repository with automatic session management."""
        session = self.database_manager.get_session()
        try:
            yield TradeRepository(session)
        finally:
            session.close()

    @contextmanager
    def get_telegram_repo(self):
        """Get telegram repository with automatic session management."""
        session = self.database_manager.get_session()
        try:
            yield TelegramRepository(session)
        finally:
            session.close()

    @contextmanager
    def get_webui_session(self):
        """Get web UI database session with automatic session management."""
        session = self.database_manager.get_session()
        try:
            yield session
        finally:
            session.close()

    @contextmanager
    def get_session(self):
        """Get database session with automatic session management."""
        session = self.database_manager.get_session()
        try:
            yield session
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
    """Initialize consolidated database and create tables."""
    service = get_database_service()

    # Initialize consolidated database
    session = service.database_manager.get_session()
    session.close()

    _logger.info("Consolidated database initialized successfully")