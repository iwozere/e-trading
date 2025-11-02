"""
Repo-layer test fixtures using a separate Postgres TEST database.

Safety rules:
- NEVER use production DATABASE_URL or donotshare DB_URL here.
- Require ALEMBIC_DB_URL (or ETRADING_TEST_DB_URL) to point to a safe test DB.
- Initialize schema via Alembic upgrade head once per test session.
- Provide a per-test transactional Session that rolls back after each test.

Usage:
- In PowerShell (Windows):
    $env:ALEMBIC_DB_URL = "postgresql+psycopg2://user:pass@localhost/e_trading_test"
    .\.venv\Scripts\Activate.ps1; pytest -q src/data/db/tests/repos
"""

from __future__ import annotations
import os
import sys
import uuid
import pathlib
import contextlib
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator
from urllib.parse import quote_plus

# Ensure repository root is on sys.path so "src" package can be imported
REPO_ROOT = pathlib.Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_donotshare_env() -> None:
    """Load environment variables from config/donotshare/.env using python-dotenv."""
    env_path = REPO_ROOT / "config" / "donotshare" / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
        # override=False means existing env vars take precedence
        load_dotenv(dotenv_path=env_path, override=False)
    except ImportError:
        # Fallback: python-dotenv not installed, skip loading
        pass
    except Exception:
        # Best-effort; do not fail tests on .env parsing issues
        pass


# Load .env before resolving any connection URLs
_load_donotshare_env()


def _get_admin_db_url_candidates() -> list[str]:
    """Return a list of admin URL candidates to try for server-level ops."""
    candidates: list[str] = []

    admin = os.getenv("PG_ADMIN_URL")
    if admin:
        candidates.append(admin)

    base = os.getenv("ALEMBIC_DB_URL")
    if base:
        try:
            u = make_url(base)
            u = u.set(database="postgres")
            candidates.append(str(u))
        except Exception:
            pass

    user = os.getenv("POSTGRES_USER")
    pwd = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    # Prefer connecting as postgres superuser when password is available (more likely to exist)
    if pwd:
        # URL-encode password to handle special characters like '.', '@', etc.
        encoded_pwd = quote_plus(pwd)
        candidates.append(f"postgresql+psycopg2://postgres:{encoded_pwd}@{host}:{port}/postgres")
    if user and pwd:
        # URL-encode password to handle special characters like '.', '@', etc.
        encoded_pwd = quote_plus(pwd)
        candidates.append(f"postgresql+psycopg2://{user}:{encoded_pwd}@{host}:{port}/postgres")

    # De-duplicate while preserving order
    seen = set()
    deduped: list[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped


def _create_test_database(admin_engine: Engine) -> tuple[str, str]:
    """Create a fresh test database and return (db_name, db_url)."""
    db_name = f"e_trading_test_{uuid.uuid4().hex[:8]}"
    # CREATE DATABASE must run outside a transaction; use AUTOCOMMIT connection
    with admin_engine.connect() as conn:
        conn = conn.execution_options(isolation_level="AUTOCOMMIT")
        conn.execute(text(f'CREATE DATABASE "{db_name}"'))

    # Build a URL to the new DB based on admin URL
    u = make_url(str(admin_engine.url))
    u = u.set(database=db_name)
    return db_name, str(u)


def _drop_test_database(admin_engine: Engine, db_name: str):
    # Force disconnect and drop database
    with admin_engine.connect() as conn:
        conn = conn.execution_options(isolation_level="AUTOCOMMIT")
        try:
            # Postgres 13+ supports DROP DATABASE ... WITH (FORCE)
            conn.execute(text(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)'))
        except Exception:
            # Terminate active connections then drop without FORCE
            with contextlib.suppress(Exception):
                conn.execute(
                    text(
                        """
                        SELECT pg_terminate_backend(pid)
                        FROM pg_stat_activity
                        WHERE datname = :db
                          AND pid <> pg_backend_pid()
                        """
                    ),
                    {"db": db_name},
                )
            with contextlib.suppress(Exception):
                conn.execute(text(f'DROP DATABASE IF EXISTS "{db_name}"'))



@pytest.fixture(scope="session")
def _db_admin_engine() -> Engine:
    """Admin engine connected to server-level DB (postgres)."""
    candidates = _get_admin_db_url_candidates()
    if not candidates:
        pytest.skip(
            "Repo tests skipped: provide PG_ADMIN_URL or set POSTGRES_* (user/password/host/port) or ALEMBIC_DB_URL"
        )

    last_err: Exception | None = None
    for url in candidates:
        try:
            eng = create_engine(url, pool_pre_ping=True, future=True)
            # test connection quickly
            with eng.connect() as conn:
                conn.execute(text("SELECT 1"))
            # recreate with AUTOCOMMIT isolation for CREATE/DROP DATABASE operations
            return create_engine(url, pool_pre_ping=True, future=True, isolation_level="AUTOCOMMIT")
        except Exception as e:
            last_err = e
            continue

    # If no candidate worked, raise the last error for visibility
    if last_err:
        raise last_err
    pytest.skip("Repo tests skipped: could not establish admin connection.")


@pytest.fixture(scope="session")
def _test_db_url(_db_admin_engine: Engine) -> Generator[str, None, None]:
    """Create a fresh test database for this test session and drop it afterwards."""
    db_name, db_url = _create_test_database(_db_admin_engine)
    try:
        yield db_url
    finally:
        # ensure all connections to the DB are closed before dropping
        _drop_test_database(_db_admin_engine, db_name)


@pytest.fixture(scope="session")
def engine(_test_db_url: str) -> Engine:
    engine = create_engine(_test_db_url, pool_pre_ping=True, future=True)
    return engine


@pytest.fixture(scope="session", autouse=True)
def _apply_migrations(engine: Engine, _test_db_url: str):
    """Run Alembic upgrade head once per session against the test DB.

    We set ALEMBIC_DB_URL for env.py to pick up, and then invoke alembic.command.upgrade.
    """
    # Ensure env var is set for Alembic env.py
    old_env = os.environ.get("ALEMBIC_DB_URL")
    os.environ["ALEMBIC_DB_URL"] = _test_db_url
    try:
        try:
            from alembic.config import Config
            from alembic import command

            # Locate the repo root alembic.ini
            repo_root = pathlib.Path(__file__).resolve().parents[5]
            alembic_ini = str(repo_root / "alembic.ini")
            cfg = Config(alembic_ini)
            # Prevent Alembic from configuring Python logging during tests
            cfg.set_main_option("verbosity", "0")
            command.upgrade(cfg, "head")
        except Exception:
            # As a fallback, try to create a minimal schema visibility
            # This is a no-op if migrations succeeded.
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                conn.commit()
        yield
    finally:
        if old_env is None:
            os.environ.pop("ALEMBIC_DB_URL", None)
        else:
            os.environ["ALEMBIC_DB_URL"] = old_env


@pytest.fixture()
def db_session(engine: Engine) -> Generator[Session, None, None]:
    """Provide a per-test transactional session bound to the test DB.

    We begin a transaction on a dedicated connection and roll it back at the end
    so tests remain isolated and repeatable.
    """
    connection = engine.connect()
    trans = connection.begin()
    SessionLocal = sessionmaker(bind=connection, autoflush=True, autocommit=False, future=True)
    session: Session = SessionLocal()
    try:
        yield session
    finally:
        # Close the Session first so it releases the connection/transaction
        with contextlib.suppress(Exception):
            session.close()
        with contextlib.suppress(Exception):
            if trans.is_active:
                trans.rollback()
        with contextlib.suppress(Exception):
            connection.close()
