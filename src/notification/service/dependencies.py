"""
Notification Service Dependencies

FastAPI dependency injection for database sessions, repositories, and services.
Provides clean separation of concerns and proper resource management.
"""

from typing import Generator
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from fastapi import Depends

from src.data.db.repos.repo_notification import NotificationRepository

from src.notification.logger import setup_logger
_logger = setup_logger(__name__)


# Database engine and session factory
engine = None
SessionLocal = None


def init_database(database_url: str = None, echo: bool = False, pool_size: int = 10, max_overflow: int = 20):
    """Initialize database engine and session factory."""
    global engine, SessionLocal

    if engine is not None:
        return

    # Use default database URL if not provided
    if database_url is None:
        # Import the main database configuration
        from config.donotshare.donotshare import DB_URL
        database_url = DB_URL

    # Create engine with appropriate configuration
    engine_kwargs = {
        "echo": echo,
        "pool_pre_ping": True,
    }

    # Handle SQLite-specific configuration
    if database_url.startswith("sqlite"):
        engine_kwargs.update({
            "poolclass": StaticPool,
            "connect_args": {"check_same_thread": False}
        })
    else:
        engine_kwargs.update({
            "pool_size": pool_size,
            "max_overflow": max_overflow
        })

    engine = create_engine(database_url, **engine_kwargs)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    _logger.info("Database initialized with URL: %s", database_url)


def get_db_session() -> Generator[Session, None, None]:
    """
    FastAPI dependency for database sessions.

    Yields:
        Database session
    """
    if SessionLocal is None:
        init_database()

    session = SessionLocal()
    try:
        yield session
    except Exception as e:
        session.rollback()
        _logger.error("Database session error: %s", e)
        raise
    finally:
        session.close()


def get_notification_repository(session: Session = None) -> NotificationRepository:
    """
    Get notification repository instance.

    Args:
        session: Database session (optional, will create new if not provided)

    Returns:
        NotificationRepository instance
    """
    if session is None:
        if SessionLocal is None:
            init_database()
        session = SessionLocal()

    return NotificationRepository(session)


@contextmanager
def get_repository_context():
    """
    Context manager for repository operations.

    Yields:
        NotificationRepository instance with automatic session management
    """
    if SessionLocal is None:
        init_database()

    session = SessionLocal()
    try:
        repo = NotificationRepository(session)
        yield repo
        session.commit()
    except Exception as e:
        session.rollback()
        _logger.error("Repository context error: %s", e)
        raise
    finally:
        session.close()


# FastAPI dependency functions
def get_notification_repo(session: Session = Depends(get_db_session)) -> NotificationRepository:
    """
    FastAPI dependency for notification repository.

    Args:
        session: Database session from dependency injection

    Returns:
        NotificationRepository instance
    """
    return NotificationRepository(session)


def get_config():
    """
    FastAPI dependency for configuration.

    Returns:
        Configuration instance
    """
    return config