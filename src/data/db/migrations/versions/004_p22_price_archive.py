"""p22_price_archive

Revision ID: 004_p22_price_archive
Revises: 003_p22_biotech_ma
Create Date: 2026-08-30

Adds `p22_price_daily` and `p22_corporate_action` (spec §2.0.7, added v0.6:
"price archive design"). Raw-price storage with read-time adjustment —
never store adjusted prices, adjust at read time via `P22Repo.get_adjusted_close`.
Does NOT modify any existing tables.
"""

import sqlalchemy as sa
from alembic import op

revision = "004_p22_price_archive"
down_revision = "003_p22_biotech_ma"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "p22_price_daily",
        sa.Column("company_id", sa.BigInteger(), nullable=False),
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column("vendor", sa.Text(), nullable=False),
        sa.Column("open_raw", sa.Numeric(), nullable=True),
        sa.Column("high_raw", sa.Numeric(), nullable=True),
        sa.Column("low_raw", sa.Numeric(), nullable=True),
        sa.Column("close_raw", sa.Numeric(), nullable=True),
        sa.Column("volume_raw", sa.BigInteger(), nullable=True),
        sa.Column("known_from", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["company_id"], ["p22_company.company_id"], name="fk_p22_price_daily_company_id_p22_company"),
        sa.PrimaryKeyConstraint("company_id", "trade_date", "vendor"),
    )
    op.create_index("idx_p22_price_daily_company_date", "p22_price_daily", ["company_id", "trade_date"])

    op.create_table(
        "p22_corporate_action",
        sa.Column("company_id", sa.BigInteger(), nullable=False),
        sa.Column("ex_date", sa.Date(), nullable=False),
        sa.Column("action_type", sa.Text(), nullable=False),
        sa.Column("ratio", sa.Numeric(), nullable=True),
        sa.Column("cash_amount", sa.Numeric(), nullable=True),
        sa.Column("new_ticker", sa.Text(), nullable=True),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("is_verified", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("known_from", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["company_id"], ["p22_company.company_id"], name="fk_p22_corporate_action_company_id_p22_company"),
        sa.CheckConstraint(
            "action_type IN ('split','reverse_split','dividend','spinoff','ticker_change')",
            name="ck_p22_corporate_action_type",
        ),
        sa.PrimaryKeyConstraint("company_id", "ex_date", "action_type"),
    )
    op.create_index("idx_p22_corporate_action_company", "p22_corporate_action", ["company_id"])


def downgrade() -> None:
    op.drop_table("p22_corporate_action")
    op.drop_table("p22_price_daily")
