"""p22_review_item_created_at

Revision ID: 005_p22_review_item_created_at
Revises: 004_p22_price_archive
Create Date: 2026-08-30

Adds `created_at` to `p22_review_item`. Not in the spec §3.4 SQL sketch, but
"queue depth and median age by item_type are reported in every run" (§3.4) is
unanswerable without an insertion timestamp — same rationale as adding
`p22_review_item`/`p22_fetch_failure` themselves beyond the original §3.2
table list. Does NOT modify any other table.
"""

import sqlalchemy as sa
from alembic import op

revision = "005_p22_review_item_created_at"
down_revision = "004_p22_price_archive"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "p22_review_item",
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("p22_review_item", "created_at")
