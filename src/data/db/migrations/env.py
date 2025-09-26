"""
alembic revision --autogenerate -m "init schema"
alembic upgrade head
# Or: alembic stamp head   (to baseline an existing DB)
"""

from __future__ import annotations
import sys
from pathlib import Path
from logging.config import fileConfig
from typing import Any

from alembic import context
from sqlalchemy import MetaData, Table
from sqlalchemy.engine import Connection
from sqlalchemy.orm import declarative_base

# -----------------------------------------------------------------------------
# Make absolute imports work (project root contains 'src')
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[5]  # <repo>/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# -----------------------------------------------------------------------------
# Import your app's DB URL/engine factory and model Bases
# -----------------------------------------------------------------------------
from src.data.db.core.database import make_engine, DB_URL
from src.data.db.models.model_users import Base as UsersBase
from src.data.db.models.model_telegram import Base as TelegramBase
from src.data.db.models.model_trading import Base as TradingBase
try:
    from src.data.db.models.model_webui import Base as WebUIBase
    _HAS_WEBUI = True
except Exception:
    _HAS_WEBUI = False

# Alembic Config object
config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# -----------------------------------------------------------------------------
# Global naming convention (important for autogenerate & consistent names)
# -----------------------------------------------------------------------------
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_name)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}
metadata = MetaData(naming_convention=NAMING_CONVENTION)
Base = declarative_base(metadata=metadata)

# -----------------------------------------------------------------------------
# Merge all Base.metadatas into one MetaData so autogenerate sees everything
# -----------------------------------------------------------------------------
all_metadata = MetaData(naming_convention=NAMING_CONVENTION)

def _safe_merge(src_md: MetaData, dst_md: MetaData) -> None:
    """Copy tables from src_md into dst_md, skipping duplicates safely."""
    for t in src_md.tables.values():
        name = t.name
        if name in dst_md.tables:
            # Already merged (or duplicate import). Skip to avoid redefinition.
            continue
        # Table.tometadata is valid API for copying tables into another MetaData
        t.to_metadata(dst_md)

bases = (UsersBase.metadata, TelegramBase.metadata, TradingBase.metadata)
if _HAS_WEBUI:
    bases = bases + (WebUIBase.metadata,)

for md in bases:
    _safe_merge(md, all_metadata)

target_metadata = all_metadata

# Make sure Alembic uses the same DB URL as your app
config.set_main_option("sqlalchemy.url", DB_URL)

# -----------------------------------------------------------------------------
# Optional: keep 'users' table untouched by autogenerate
#   Set EXCLUDE_USERS = True if you want to *never* autogenerate changes to it.
#   (Leave False if your Users model matches the live table and you want diffs.)
# -----------------------------------------------------------------------------
EXCLUDE_USERS = False

def include_object(
    object: Any, name: str, type_: str, reflected: bool, compare_to: Any
):
    # Skip SQLite internal objects
    if name.startswith("sqlite_"):
        return False
    # Optionally protect 'users' table
    if EXCLUDE_USERS and type_ == "table" and name == "users":
        return False
    return True

# -----------------------------------------------------------------------------
# Offline/online runners
# -----------------------------------------------------------------------------
def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
        compare_server_default=True,
        include_object=include_object,
        render_as_batch=url.startswith("sqlite"),  # important for SQLite
    )
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    engine = make_engine(DB_URL, echo=False)
    is_sqlite = engine.dialect.name == "sqlite"
    with engine.connect() as connection:  # type: Connection
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            include_object=include_object,
            render_as_batch=is_sqlite,  # safe ALTER TABLE via batch mode
            include_schemas=False,
        )
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
