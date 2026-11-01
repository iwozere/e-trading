"""p22_trial_asset_link

Revision ID: 013_p22_trial_asset_link
Revises: 012_p22_block_g_known_from
Create Date: 2026-11-01

Replaces `p22_trial.asset_id` (one asset per NCT ID) with the many-to-many `p22_trial_asset`.
The single column made a trial landed under several companies flip to whichever company
`run_trial_normalization.py` wrote last, and a later unlinked write erased a good link (1,452 of
17,432 landed trials sit under more than one company; 72 targets affected, 2026-10-30 partition).

The old links are NOT copied: they were made without a lead-sponsor check, so many point at the
wrong company (CROs, licensors). The next P22 Trial Normalization run rebuilds every company's
links from the latest CT.gov partition; until then Block B reads no trials. Downgrade restores the
column empty, for the same reason.
"""

import sqlalchemy as sa
from alembic import op

revision = "013_p22_trial_asset_link"
down_revision = "012_p22_block_g_known_from"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "p22_trial_asset",
        sa.Column("nct_id", sa.Text(), sa.ForeignKey("p22_trial.nct_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("asset_id", sa.BigInteger(), sa.ForeignKey("p22_asset.asset_id", ondelete="CASCADE"), primary_key=True),
    )
    op.create_index("idx_p22_trial_asset_asset", "p22_trial_asset", ["asset_id"])
    # Dropping the column also drops its FK and idx_p22_trial_asset.
    op.drop_column("p22_trial", "asset_id")


def downgrade() -> None:
    op.add_column("p22_trial", sa.Column("asset_id", sa.BigInteger(), sa.ForeignKey("p22_asset.asset_id"), nullable=True))
    op.create_index("idx_p22_trial_asset", "p22_trial", ["asset_id"])
    op.drop_index("idx_p22_trial_asset_asset", table_name="p22_trial_asset")
    op.drop_table("p22_trial_asset")
