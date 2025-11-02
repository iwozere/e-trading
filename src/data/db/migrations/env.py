import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

# Ensure repository root on path
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Import Base and ensure models are imported so tables are registered on metadata
from src.data.db.core.base import Base  # noqa: E402

# Import model modules to populate Base.metadata
import src.data.db.models.model_users  # noqa: F401,E402
import src.data.db.models.model_trading  # noqa: F401,E402
import src.data.db.models.model_jobs  # noqa: F401,E402
import src.data.db.models.model_webui  # noqa: F401,E402
import src.data.db.models.model_notification  # noqa: F401,E402
import src.data.db.models.model_system_health  # noqa: F401,E402
import src.data.db.models.model_telegram  # noqa: F401,E402
import src.data.db.models.model_short_squeeze  # noqa: F401,E402

# this is the Alembic Config object, which provides access to the values within
# the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
# target_metadata = None
target_metadata = Base.metadata


def _get_db_url() -> str:
    # Prefer explicit env var, then ini value, then code config
    env_url = os.getenv("ALEMBIC_DB_URL") or os.getenv("DATABASE_URL")
    if env_url:
        return env_url.replace("postgres://", "postgresql://")

    ini_url = config.get_main_option("sqlalchemy.url")
    if ini_url and "driver://" not in ini_url:
        return ini_url.replace("postgres://", "postgresql://")

    # Fallback to code config if available
    try:
        from config.donotshare.donotshare import DB_URL  # type: ignore

        return DB_URL.replace("postgres://", "postgresql://")
    except Exception:
        # Last resort, return ini placeholder which will fail loudly if used
        return ini_url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = _get_db_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""

    configuration = config.get_section(config.config_ini_section)
    if configuration is None:
        configuration = {}
    configuration["sqlalchemy.url"] = _get_db_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            # when using multiple schemas or custom include/exclude, add here
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
