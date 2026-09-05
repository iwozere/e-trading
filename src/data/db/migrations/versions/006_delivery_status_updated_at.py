"""delivery_status_updated_at

Revision ID: 006_delivery_status_updated_at
Revises: 005_p22_review_item_created_at
Create Date: 2026-09-05

Adds `updated_at` to `msg_delivery_status`, used to detect a per-channel
delivery row stuck in SENT (a consumer claimed it then crashed before writing
back DELIVERED/FAILED) so it can be reclaimed. Needed for the per-channel-aware
claim query that replaces the whole-message overlap claim for notification-bot
and telegram-bot's queue consumers -- see monitoring.txt, 2026-09-02
"Channel instance not available: telegram" investigation. Does not modify any
other table.
"""

import sqlalchemy as sa
from alembic import op

revision = "006_delivery_status_updated_at"
down_revision = "005_p22_review_item_created_at"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "msg_delivery_status",
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )


def downgrade() -> None:
    op.drop_column("msg_delivery_status", "updated_at")
