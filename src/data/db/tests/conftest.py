# src/data/db/tests/conftest.py
from __future__ import annotations
import pytest
import os
from sqlalchemy import create_engine, event, MetaData, text
from sqlalchemy.orm import sessionmaker

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.data.db.core.base import Base
# Import models to register them with the Base metadata
import src.data.db.models.model_users
import src.data.db.models.model_telegram
import src.data.db.models.model_trading
import src.data.db.models.model_webui
import src.data.db.models.model_jobs

def _ddl_create_all(engine):
    """Create ALL tables once using the shared Base metadata."""
    Base.metadata.create_all(engine)

def _make_users_visible_in_other_metadatas():
    """
    Since we're using a shared Base, all tables are already in the same metadata.
    This function is no longer needed but kept for compatibility.
    """
    pass

@pytest.fixture(scope="session")
def engine():
    """Create test database engine."""
    # Import the actual database configuration
    try:
        from config.donotshare.donotshare import (
            POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER,
            POSTGRES_PASSWORD, POSTGRES_DATABASE
        )
        # Use actual database configuration but with a test database name
        test_db_name = f"{POSTGRES_DATABASE}_test"
        test_db_url = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{test_db_name}"
    except ImportError:
        # Fallback to environment variable or default
        test_db_url = os.getenv(
            "TEST_DATABASE_URL",
            "postgresql://postgres:password@localhost:5432/etrading_test"
        )

    eng = create_engine(test_db_url, future=True)

    # Create only the tables we need for testing (safer approach)
    from src.data.db.models.model_users import User
    from src.data.db.models.model_trading import BotInstance, Trade, Position, PerformanceMetric

    # Create tables in dependency order to avoid foreign key errors
    tables_to_create = [
        User.__table__,
        BotInstance.__table__,
        Position.__table__,  # Create Position before Trade since Trade references Position
        PerformanceMetric.__table__,
        Trade.__table__
    ]

    for table in tables_to_create:
        try:
            table.create(eng, checkfirst=True)
        except Exception as e:
            if "already exists" in str(e):
                pass  # Table already exists, continue
            else:
                print(f"Error creating table {table.name}: {e}")
                # Continue with other tables

    yield eng

    # No cleanup - leave test database intact for reuse
    eng.dispose()

@pytest.fixture()
def dbsess(engine):
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    s = SessionLocal()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
