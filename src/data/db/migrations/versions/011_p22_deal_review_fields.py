"""p22_deal_review_fields

Revision ID: 011_p22_deal_review_fields
Revises: 010_signal_ledger
Create Date: 2026-10-29

Adds the hand-verification fields `cli/deal_entry_cli.py` writes to `p22_deal` (spec §2.5):
`acquirer_name` (private and PE acquirers have no `p22_company` row for `acquirer_id`),
`accession_no` + `review_item_id` (which filing/candidate the row was entered from),
`stock_consideration_pct` + `target_holders_pct` (inputs to the reverse-merger heuristics),
`review_flags` (heuristics that fired when the row was entered), and `entered_by`/`entered_at`/`note`.
Also adds a unique `(target_id, announcement_date)` so the same deal cannot be entered twice from two
of its filings (SC 14D9 + S-4 + amendments). `p22_deal` has never been written, so the constraint is
safe to add. Does NOT modify any other table.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "011_p22_deal_review_fields"
down_revision = "010_signal_ledger"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("p22_deal", sa.Column("acquirer_name", sa.Text(), nullable=True))
    op.add_column("p22_deal", sa.Column("accession_no", sa.Text(), nullable=True))
    op.add_column(
        "p22_deal",
        sa.Column("review_item_id", sa.BigInteger(), sa.ForeignKey("p22_review_item.item_id"), nullable=True),
    )
    op.add_column("p22_deal", sa.Column("stock_consideration_pct", sa.Numeric(), nullable=True))
    op.add_column("p22_deal", sa.Column("target_holders_pct", sa.Numeric(), nullable=True))
    op.add_column("p22_deal", sa.Column("review_flags", postgresql.JSONB(), nullable=True))
    op.add_column("p22_deal", sa.Column("entered_by", sa.Text(), nullable=True))
    op.add_column(
        "p22_deal",
        sa.Column("entered_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
    )
    op.add_column("p22_deal", sa.Column("note", sa.Text(), nullable=True))
    op.create_unique_constraint("uq_p22_deal_target_announcement", "p22_deal", ["target_id", "announcement_date"])


def downgrade() -> None:
    op.drop_constraint("uq_p22_deal_target_announcement", "p22_deal", type_="unique")
    for column in (
        "note", "entered_at", "entered_by", "review_flags", "target_holders_pct",
        "stock_consideration_pct", "review_item_id", "accession_no", "acquirer_name",
    ):
        op.drop_column("p22_deal", column)
