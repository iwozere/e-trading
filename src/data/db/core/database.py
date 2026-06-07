from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Generator, Optional

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session

from src.data.db.core.base import Base
from config.donotshare.donotshare import SQL_ECHO, DB_URL as CONFIG_DB_URL



# --- Config ------------------------------------------------------------------

# Use PostgreSQL by default, fallback to SQLite if needed
# Environment variable DB_URL can override this
DB_URL = os.getenv("DB_URL", CONFIG_DB_URL)

# SQL query logging — disabled by default; enable only in development via SQL_ECHO=1 env var.
# Never enable in production: exposes bound values (PII) in logs.
SQL_ECHO = bool(int(os.getenv("SQL_ECHO", "0")))


def get_database_url() -> str:
    """Get the current database URL."""
    return DB_URL


# --- Engine / Session --------------------------------------------------------

def _is_sqlite(url: str) -> bool:
    return url.startswith("sqlite:")

def make_engine(url: str | None = None, *, echo: bool | None = None) -> Engine:
    """
    Create an SQLAlchemy Engine suitable for your bots (SQLite or Postgres).

    - **SQLite**: ``check_same_thread=False`` for threaded access; WAL + FK enforced.
    - **PostgreSQL / other**: configurable connection pool via environment variables:

      =========================================  =======  ========================
      Env var                                    Default  Meaning
      =========================================  =======  ========================
      ``DB_POOL_SIZE``                           ``10``   Number of persistent connections
      ``DB_MAX_OVERFLOW``                        ``20``   Extra connections above pool_size
      ``DB_POOL_TIMEOUT``                        ``30``   Seconds to wait for a connection
      ``DB_POOL_RECYCLE``                        ``1800`` Recycle connections after N seconds
      =========================================  =======  ========================
    """
    url = url or DB_URL
    echo = SQL_ECHO if echo is None else echo

    connect_args: dict = {}
    engine_kwargs: dict = {
        "future": True,
        "pool_pre_ping": True,
        "echo": echo,
        "connect_args": connect_args,
    }

    if _is_sqlite(url):
        # Needed if you use multiple threads (Telegram bot + trading loop).
        connect_args["check_same_thread"] = False
    else:
        # PostgreSQL / MySQL: tune pool for concurrent bot + pipeline workloads.
        engine_kwargs.update(
            pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20")),
            pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "30")),
            pool_recycle=int(os.getenv("DB_POOL_RECYCLE", "1800")),
        )

    engine = create_engine(url, **engine_kwargs)

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


# --- Lazy engine / session ---------------------------------------------------
# The engine is created on first use so that a bad DB_URL or unavailable
# database doesn't crash the entire application on import.

import threading

_engine_lock = threading.RLock()
_engine: Optional[Engine] = None
_session_factory: Optional[sessionmaker] = None


def get_engine() -> Engine:
    """Return the module-level engine, creating it on first call (thread-safe)."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:          # double-checked locking
                _engine = make_engine(DB_URL)
    return _engine


def get_session_factory() -> sessionmaker:
    """Return the module-level sessionmaker, creating it on first call."""
    global _session_factory
    if _session_factory is None:
        with _engine_lock:
            if _session_factory is None:
                _session_factory = sessionmaker(
                    bind=get_engine(), autoflush=False, autocommit=False, future=True,
                    expire_on_commit=False
                )
    return _session_factory


# Backwards-compatible module-level names (resolved lazily on first access).
# Code that does `from src.data.db.core.database import SessionLocal` continues
# to work because calling SessionLocal() delegates to get_session_factory()().
class _LazySessionLocal:
    """Proxy that behaves like a sessionmaker but initialises lazily."""
    def __call__(self, *args, **kwargs):
        return get_session_factory()(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(get_session_factory(), item)


SessionLocal = _LazySessionLocal()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """
    Short-lived session pattern:
        with session_scope() as s:
            s.add(obj)
            ...
    Commits on success, rolls back on error, always closes.
    """
    s: Session = get_session_factory()()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


# --- Utilities ---------------------------------------------------------------

def create_all_tables(*bases: Any) -> None:
    """
    If you keep multiple Declarative 'Base' objects (users/telegram/trading),
    call this once at startup to ensure all tables exist:
        from src.data.db.models.model_users import Base as UsersBase
        from src.data.db.models.model_telegram import Base as TgBase
        from src.data.db.models.model_trading import Base as TradingBase
        create_all_tables(UsersBase, TgBase, TradingBase)

    Note: Since we now use a single Base, you can simply call:
        Base.metadata.create_all(engine)
    """
    if not bases:
        Base.metadata.create_all(get_engine())
    else:
        for base in bases:
            base.metadata.create_all(get_engine())


def drop_all_tables(*bases: Any) -> None:
    """
    Drop all database tables. Handy in tests.

    ⚠️  SAFETY: This function requires the environment variable
    ``ALLOW_DROP_TABLES=1`` to be set explicitly.  This prevents accidental
    data loss when the variable is absent in production deployments.
    """
    if os.getenv("ALLOW_DROP_TABLES") != "1":
        raise RuntimeError(
            "CRITICAL SAFETY CHECK FAILED: Attempted to drop tables without explicit permission.\n"
            "Set the environment variable ALLOW_DROP_TABLES=1 to allow this operation.\n"
            "⚠️  This will permanently delete all data in the database."
        )

    if not bases:
        Base.metadata.drop_all(get_engine())
    else:
        for base in bases:
            base.metadata.drop_all(get_engine())

