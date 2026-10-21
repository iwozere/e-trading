"""signal_ledger

Revision ID: 010_signal_ledger
Revises: 009_k20_sentiment_wiki_source
Create Date: 2026-10-21

Adds `signal_ledger` (every pipeline buy/sell signal plus its forward returns vs VT/SPY/IWM)
and `signal_book_nav` (daily NAV of each stream's simulated paper book), written by
`src/ml/pipeline/signal_ledger/jobs/run_signal_ledger.py`.
Does NOT modify any existing tables.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "010_signal_ledger"
down_revision = "009_k20_sentiment_wiki_source"
branch_labels = None
depends_on = None

_RETURN_COLUMNS = [f"{prefix}_{h}d" for prefix in ("ret", "vt", "spy", "iwm") for h in (1, 5, 10, 20)]


def upgrade() -> None:
    op.create_table(
        "signal_ledger",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("pipeline", sa.Text(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("signal_date", sa.Date(), nullable=False),
        sa.Column("ticker", sa.Text(), nullable=False),
        sa.Column("side", sa.Text(), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=True),
        sa.Column("score", sa.Numeric(), nullable=True),
        sa.Column("horizon_days", sa.SmallInteger(), nullable=False),
        sa.Column("meta", postgresql.JSONB(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("entry_date", sa.Date(), nullable=True),
        sa.Column("entry_price", sa.Numeric(), nullable=True),
        sa.Column("data_flag", sa.Text(), nullable=True),
        *(sa.Column(name, sa.Numeric(), nullable=True) for name in _RETURN_COLUMNS),
        sa.Column("harvested_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source", "signal_date", "ticker", "side", name="uq_signal_ledger_key"),
        sa.CheckConstraint("side IN ('long','short')", name="side"),
        sa.CheckConstraint("status IN ('pending','complete','no_data')", name="status"),
    )
    op.create_index("idx_signal_ledger_source_date", "signal_ledger", ["source", "signal_date"])
    op.create_index("idx_signal_ledger_status", "signal_ledger", ["status"])

    op.create_table(
        "signal_book_nav",
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("nav_date", sa.Date(), nullable=False),
        sa.Column("nav", sa.Numeric(), nullable=False),
        sa.Column("benchmark_nav", sa.Numeric(), nullable=False),
        sa.Column("invested", sa.Numeric(), nullable=False),
        sa.Column("positions", sa.Integer(), nullable=False),
        sa.Column("opened", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("closed", sa.Integer(), nullable=False, server_default="0"),
        sa.PrimaryKeyConstraint("source", "nav_date"),
    )


def downgrade() -> None:
    op.drop_table("signal_book_nav")
    op.drop_index("idx_signal_ledger_status", table_name="signal_ledger")
    op.drop_index("idx_signal_ledger_source_date", table_name="signal_ledger")
    op.drop_table("signal_ledger")
