from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterable

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy import MetaData
from src.data.db.core.base import Base

from config.donotshare.donotshare import DB_PATH, SQL_ECHO, DB_URL as CONFIG_DB_URL

# ---- ONE shared metadata + Base for the whole app ----
_convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}
_shared_metadata = MetaData(naming_convention=_convention)



# --- Config ------------------------------------------------------------------

# Use PostgreSQL by default, fallback to SQLite if needed
# Environment variable DB_URL can override this
DB_URL = os.getenv("DB_URL", CONFIG_DB_URL)

# Flip this on to see SQL in logs (or set SQL_ECHO=1 in env)
SQL_ECHO = bool(int(os.getenv("SQL_ECHO", SQL_ECHO)))


# --- Engine / Session --------------------------------------------------------

def _is_sqlite(url: str) -> bool:
    return url.startswith("sqlite:")

def make_engine(url: str | None = None, *, echo: bool | None = None) -> Engine:
    """
    Create an SQLAlchemy Engine suitable for your bots (SQLite or Postgres).
    - SQLite: check_same_thread=False for threaded access; WAL + FK enforced.
    - Postgres/MySQL/etc.: defaults are fine.
    """
    url = url or DB_URL
    echo = SQL_ECHO if echo is None else echo

    connect_args = {}
    if _is_sqlite(url):
        # Needed if you use multiple threads (Telegram bot + trading loop).
        connect_args = {"check_same_thread": False}

    engine = create_engine(
        url,
        future=True,
        pool_pre_ping=True,
        echo=echo,
        connect_args=connect_args,
    )

    if _is_sqlite(url):
        _attach_sqlite_pragmas(engine)

    return engine


def _attach_sqlite_pragmas(engine: Engine) -> None:
    """Apply SQLite PRAGMAs on every new connection."""
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        # Concurrency + durability balance
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        # Enforce referential integrity
        cur.execute("PRAGMA foreign_keys=ON;")
        # Helpful extras (tune as needed)
        cur.execute("PRAGMA busy_timeout=5000;")   # 5s wait on locks
        cur.execute("PRAGMA temp_store=MEMORY;")
        cur.close()


# Create a module-level engine + sessionmaker for app usage
engine: Engine = make_engine(DB_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


@contextmanager
def session_scope() -> Session:
    """
    Short-lived session pattern:
        with session_scope() as s:
            s.add(obj)
            ...
    Commits on success, rolls back on error, always closes.
    """
    s: Session = SessionLocal()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


# --- Utilities ---------------------------------------------------------------

def create_all_tables(*bases: Iterable) -> None:
    """
    If you keep multiple Declarative 'Base' objects (users/telegram/trading),
    call this once at startup to ensure all tables exist:
        from src.data.db.models.model_users import Base as UsersBase
        from src.data.db.models.model_telegram import Base as TgBase
        from src.data.db.models.model_trading import Base as TradingBase
        create_all_tables(UsersBase, TgBase, TradingBase)
    """
    for base in bases:
        base.metadata.create_all(engine)


def drop_all_tables(*bases: Iterable) -> None:
    """Handy in tests."""
    for base in bases:
        base.metadata.drop_all(engine)
