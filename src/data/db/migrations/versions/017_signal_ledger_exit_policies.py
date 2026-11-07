"""signal_ledger_exit_policies

Revision ID: 017_signal_ledger_exit_policies
Revises: 016_drop_ss_sentiment_history
Create Date: 2026-11-07

Signal ledger step 2 (exit policies):
- `signal_ledger`: 60/120/250-session returns (ticker + VT/SPY/IWM) and the 20-session MAE/MFE.
  Signals that were `complete` at 20 sessions go back to `pending` so the nightly job fills
  the new columns (their 1-20 session returns are recomputed from the same prices).
- `signal_book_nav`: a `policy` column (existing rows = `horizon`, the original rule) in the key.
- `signal_book_trades`: every paper trade of every stream/policy book (rebuilt nightly).
- `signal_book_stats`: nightly per stream/policy summary (capped book + uncapped trade stats).
Only the signal-ledger tables are touched.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "017_signal_ledger_exit_policies"
down_revision = "016_drop_ss_sentiment_history"
branch_labels = None
depends_on = None

_NEW_COLUMNS = [f"{prefix}_{h}d" for prefix in ("ret", "vt", "spy", "iwm") for h in (60, 120, 250)] + ["mae_20d", "mfe_20d"]


def upgrade() -> None:
    """Add the long-horizon/excursion columns, the NAV policy key and the trades table."""
    for name in _NEW_COLUMNS:
        op.add_column("signal_ledger", sa.Column(name, sa.Numeric(), nullable=True))
    op.execute("UPDATE signal_ledger SET status = 'pending' WHERE status = 'complete'")

    op.add_column("signal_book_nav", sa.Column("policy", sa.Text(), nullable=False, server_default="horizon"))
    # The key was created as pk_signal_book_nav (naming convention) or signal_book_nav_pkey (no convention).
    op.execute("ALTER TABLE signal_book_nav DROP CONSTRAINT IF EXISTS pk_signal_book_nav")
    op.execute("ALTER TABLE signal_book_nav DROP CONSTRAINT IF EXISTS signal_book_nav_pkey")
    op.create_primary_key("pk_signal_book_nav", "signal_book_nav", ["source", "policy", "nav_date"])

    op.create_table(
        "signal_book_trades",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("policy", sa.Text(), nullable=False),
        sa.Column("signal_id", sa.BigInteger(), nullable=True),
        sa.Column("ticker", sa.Text(), nullable=False),
        sa.Column("signal_date", sa.Date(), nullable=False),
        sa.Column("entry_date", sa.Date(), nullable=False),
        sa.Column("entry_price", sa.Numeric(), nullable=False),
        sa.Column("exit_date", sa.Date(), nullable=True),
        sa.Column("exit_price", sa.Numeric(), nullable=True),
        sa.Column("exit_reason", sa.Text(), nullable=False),
        sa.Column("sessions_held", sa.Integer(), nullable=False),
        sa.Column("net_return", sa.Numeric(), nullable=True),
        sa.Column("last_price", sa.Numeric(), nullable=True),
        sa.Column("mae", sa.Numeric(), nullable=True),
        sa.Column("mfe", sa.Numeric(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_signal_book_trades"),
        sa.ForeignKeyConstraint(["signal_id"], ["signal_ledger.id"], ondelete="CASCADE",
                                name="fk_signal_book_trades_signal_id_signal_ledger"),
    )
    op.create_index("idx_signal_book_trades_source_policy", "signal_book_trades", ["source", "policy"])

    op.create_table(
        "signal_book_stats",
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("policy", sa.Text(), nullable=False),
        sa.Column("stats", postgresql.JSONB(), nullable=False),
        sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.PrimaryKeyConstraint("source", "policy", name="pk_signal_book_stats"),
    )


def downgrade() -> None:
    """Drop the stats and trades tables, keep only the default-policy NAV rows, drop the new columns."""
    op.drop_table("signal_book_stats")
    op.drop_index("idx_signal_book_trades_source_policy", table_name="signal_book_trades")
    op.drop_table("signal_book_trades")
    op.execute("DELETE FROM signal_book_nav WHERE policy <> 'horizon'")
    op.execute("ALTER TABLE signal_book_nav DROP CONSTRAINT IF EXISTS pk_signal_book_nav")
    op.create_primary_key("pk_signal_book_nav", "signal_book_nav", ["source", "nav_date"])
    op.drop_column("signal_book_nav", "policy")
    for name in reversed(_NEW_COLUMNS):
        op.drop_column("signal_ledger", name)
