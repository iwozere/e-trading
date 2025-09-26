# src/data/db/tests/conftest.py
from __future__ import annotations
import pytest
from sqlalchemy import create_engine, event, MetaData
from sqlalchemy.orm import sessionmaker

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.data.db.models.model_users import Base as UsersBase
from src.data.db.models.model_telegram import Base as TelegramBase
from src.data.db.models.model_trading import Base as TradingBase
from src.data.db.models.model_webui import Base as WebUIBase

def _ddl_create_all(engine):
    """Create ALL tables once using a merged MetaData (for correct DDL order)."""
    merged = MetaData()
    for md in (UsersBase.metadata, TelegramBase.metadata, TradingBase.metadata, WebUIBase.metadata):
        for t in md.tables.values():
            # SA 2.x spelling:
            t.to_metadata(merged)
    merged.create_all(engine)

def _make_users_visible_in_other_metadatas():
    """
    ORM resolution of ForeignKey('users.id') looks up within the *parent table's* metadata.
    Copy users Table into WebUI/Telegram metadatas so ORM can resolve dependencies.
    """
    users_tbl = UsersBase.metadata.tables["users"]
    for md in (WebUIBase.metadata, TelegramBase.metadata):
        if "users" not in md.tables:
            users_tbl.to_metadata(md)

@pytest.fixture(scope="session")
def engine():
    eng = create_engine("sqlite+pysqlite:///:memory:", future=True)
    @event.listens_for(eng, "connect")
    def _fk_on(dbapi_con, _):
        dbapi_con.execute("PRAGMA foreign_keys=ON")
    _ddl_create_all(eng)
    # IMPORTANT: do this after imports and before any ORM flush
    _make_users_visible_in_other_metadatas()
    return eng

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
