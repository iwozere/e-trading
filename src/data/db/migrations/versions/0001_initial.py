"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2025-11-02

"""
from alembic import op
import sqlalchemy as sa

# Import Base and all models so metadata is populated
from src.data.db.core.base import Base
import src.data.db.models.model_users  # noqa: F401
import src.data.db.models.model_trading  # noqa: F401
import src.data.db.models.model_jobs  # noqa: F401
import src.data.db.models.model_webui  # noqa: F401
import src.data.db.models.model_notification  # noqa: F401
import src.data.db.models.model_system_health  # noqa: F401
import src.data.db.models.model_telegram  # noqa: F401
import src.data.db.models.model_short_squeeze  # noqa: F401

# revision identifiers, used by Alembic.
revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create all tables from SQLAlchemy metadata.

    Note: For initial migration only. Future migrations should be explicit.
    """
    bind = op.get_bind()
    Base.metadata.create_all(bind=bind)


def downgrade() -> None:
    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind)
