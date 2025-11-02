import os
import sys
import uuid
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

import pytest

# Ensure repository root is on sys.path so `config` package can be imported
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Prefer safe import of donotshare; fall back to env vars for CI/dev where the private module isn't present
try:
    from config.donotshare.donotshare import DB_URL  # type: ignore
except Exception:
    DB_URL = os.getenv("DATABASE_URL") or os.getenv("ALEMBIC_DB_URL")


# if psycopg2 is not available, skip DB tests (keeps test suite friendly when DB driver missing)
try:
    import psycopg2  # noqa: F401
    _HAS_PG = True
except Exception:
    _HAS_PG = False


@pytest.fixture(scope="session")
def engine():
    # Use DB_URL from config; tests require a running Postgres instance
    if not _HAS_PG:
        pytest.skip("psycopg2 (Postgres DB driver) not installed; skipping DB model tests")
    if not DB_URL:
        pytest.skip("DB_URL not available (donotshare missing and no DATABASE_URL/ALEMBIC_DB_URL set); skipping DB model tests")
    engine = create_engine(DB_URL, future=True)
    yield engine
    engine.dispose()


@pytest.fixture(scope="session")
def tmp_schema(engine):
    """
    Create a temporary Postgres schema for tests and create tables for ORM metadata.

    Important: This does NOT create or drop the database. It only creates/drops a schema
    inside the existing Postgres database pointed to by DB_URL. This follows the user's
    constraint to avoid creating/dropping databases.
    """
    schema = f"test_models_{uuid.uuid4().hex[:8]}"

    with engine.connect() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
        # set search_path on this connection and create tables using metadata on the same connection
        conn.execute(text(f"SET search_path TO {schema}"))

        # import models to register them on Base.metadata
        # we import lazily here so tests can be discovered without immediately requiring DB
        from src.data.db.core.base import Base  # noqa: E402

        # Import all model modules so they are registered on Base.metadata
        import src.data.db.models.model_users as _mu  # noqa: F401,E402
        import src.data.db.models.model_trading as _mt  # noqa: F401,E402
        import src.data.db.models.model_jobs as _mj  # noqa: F401,E402
        import src.data.db.models.model_webui as _mw  # noqa: F401,E402
        import src.data.db.models.model_notification as _mn  # noqa: F401,E402
        import src.data.db.models.model_system_health as _ms  # noqa: F401,E402
        import src.data.db.models.model_telegram as _mtg  # noqa: F401,E402
        import src.data.db.models.model_short_squeeze as _mss  # noqa: F401,E402

        # Create the tables in order based on dependencies
        tables_to_create = [
            # Core user tables
            "usr_users",
            "usr_auth_identities",
            "usr_verification_codes",
            # Trading tables
            "trading_bots",
            "trading_positions",
            "trading_trades",
            "trading_performance_metrics",
            # Job tables
            "job_schedules",
            "job_schedule_runs",
            # Message tables
            "msg_messages",
            "msg_delivery_status",
            "msg_rate_limits",
            "msg_channel_configs",
            "msg_system_health",
            # Other tables in order
            "webui_audit_logs",
            "webui_strategy_templates",
            "webui_performance_snapshots",
            "webui_system_config",
            "telegram_settings",
            "telegram_feedbacks",
            "telegram_command_audits",
            "telegram_broadcast_logs"
        ]

        tables_dict = {t.name: t for t in Base.metadata.tables.values()}
        created_tables = set()

        # Create tables in order; use explicit commits to finalize DDL
        try:
            for table_name in tables_to_create:
                if table_name not in tables_dict:
                    continue

                table = tables_dict[table_name]
                if table_name in created_tables:
                    continue

                # Remove any postgresql-specific indexes with 'where' clauses
                for idx in list(table.indexes):
                    pg_opts = idx.dialect_options.get("postgresql", {})
                    if pg_opts and (pg_opts.get("where") is not None or pg_opts.get("postgresql_where") is not None):
                        try:
                            table.indexes.remove(idx)
                        except KeyError:
                            pass

                print(f"Creating table {table_name}...")
                # Create the table
                table.create(bind=conn)
                created_tables.add(table_name)

                # Verify table exists by attempting to select from it
                try:
                    conn.execute(text(f"SELECT 1 FROM {table_name} LIMIT 0"))
                except Exception as e:
                    print(f"Failed to verify table {table_name}: {e}")
                    raise

            # Commit the DDL so subsequent sessions can see the tables
            conn.commit()
            print("All tables created successfully and committed")
        except Exception as e:
            print(f"Error creating tables: {e}")
            raise

        yield schema

        # cleanup: drop schema cascade
        # use a separate connection for cleanup
    with engine.connect() as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))


@pytest.fixture()
def db_session(engine, tmp_schema):
    """Provide a transactional session bound to a single connection.

    Each test gets a rollback at the end to keep the schema clean.

        Important:
        - Some model-level event listeners may raise to enforce invariants.
            Those listeners must NOT call connection.rollback() themselves,
            as it deassociates the transaction and produces warnings during
            teardown. We therefore guard the rollback here so the fixture
            remains quiet even if a listener prematurely ended the transaction.
    """
    # Use a single connection and transaction so everything can be rolled back per test
    connection = engine.connect()
    transaction = connection.begin()

    # ensure operations target our schema
    connection.execute(text(f"SET search_path TO {tmp_schema}"))

    Session = sessionmaker(bind=connection)
    session = Session()

    try:
        yield session
    finally:
        # Close ORM session first
        session.close()
        # Roll back only if the transaction is still active; some tests or
        # model-level listeners may have already ended it.
        try:
            if transaction.is_active:  # SQLAlchemy Transaction has is_active
                transaction.rollback()
            else:
                # Ensure connection is clean for safety; ignore if already clean
                try:
                    connection.rollback()
                except Exception:
                    pass
        finally:
            connection.close()
