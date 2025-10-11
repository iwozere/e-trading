"""
Database Configuration and Session Management
-------------------------------------------

SQLAlchemy database configuration, session management,
and initialization utilities.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from pathlib import Path
import sys
from typing import Generator

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.data.db.services.database_service import get_database_service
from src.data.db.models.model_users import User
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

# Database configuration - Updated to use consolidated database
DATABASE_URL = f"sqlite:///{PROJECT_ROOT}/db/trading.db"

# Create engine with connection pooling for SQLite
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
    echo=False  # Set to True for SQL debugging
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    Dependency function to get database session.

    Uses the consolidated database service for consistency.

    Yields:
        Session: SQLAlchemy database session
    """
    db_service = get_database_service()
    with db_service.uow() as session:
        yield session.session


def init_database():
    """
    Initialize database tables and create default users.
    """
    try:
        # Create database directory if it doesn't exist
        db_dir = Path(DATABASE_URL.replace("sqlite:///", "")).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        # Create all tables
        Base.metadata.create_all(bind=engine)
        _logger.info("Database tables created successfully")

        # Create default users if they don't exist
        create_default_users()

    except Exception as e:
        _logger.error("Failed to initialize database: %s", e)
        raise


def create_default_users():
    """
    Create default admin and trader users if they don't exist.
    """
    db = SessionLocal()
    try:
        # Check if any web users exist
        existing_users = db.query(User).filter(User.username.isnot(None)).count()
        if existing_users > 0:
            _logger.info("Web users already exist, skipping default user creation")
            return

        # Create default users
        default_users = [
            User(
                username="admin",
                email="admin@trading-system.local",
                hashed_password=User.hash_password("admin"),
                role="admin",
                is_active=True
            ),
            User(
                username="trader",
                email="trader@trading-system.local",
                hashed_password=User.hash_password("trader"),
                role="trader",
                is_active=True
            ),
            User(
                username="viewer",
                email="viewer@trading-system.local",
                hashed_password=User.hash_password("viewer"),
                role="viewer",
                is_active=True
            )
        ]

        for user in default_users:
            db.add(user)

        db.commit()
        _logger.info("Created default users: admin, trader, viewer")

    except Exception as e:
        _logger.error("Failed to create default users: %s", e)
        db.rollback()
        raise
    finally:
        db.close()


def reset_database():
    """
    Drop all tables and recreate them.
    WARNING: This will delete all data!
    """
    try:
        Base.metadata.drop_all(bind=engine)
        _logger.warning("All database tables dropped")

        init_database()
        _logger.info("Database reset completed")

    except Exception as e:
        _logger.error("Failed to reset database: %s", e)
        raise


if __name__ == "__main__":
    # Initialize database when run directly
    init_database()
    print("Database initialized successfully")