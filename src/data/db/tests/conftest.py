# src/data/db/tests/conftest.py
from __future__ import annotations
import pytest
import os
from sqlalchemy import create_engine, event, MetaData
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
    """Create PostgreSQL test database engine."""
    # Use test database URL or default to a test database
    test_db_url = os.getenv(
        "TEST_DATABASE_URL",
        "postgresql://postgres:password@localhost:5432/etrading_test"
    )

    eng = create_engine(test_db_url, future=True)

    # Drop all tables and recreate for clean test environment
    Base.metadata.drop_all(eng)
    _ddl_create_all(eng)

    yield eng

    # Cleanup after tests
    Base.metadata.drop_all(eng)
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
