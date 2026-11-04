"""trade_ledger

Revision ID: 014_trade_ledger
Revises: 013_p22_trial_asset_link
Create Date: 2026-11-04

Adds the five raw-event tables of the monthly trade ledger report
(`src/portfolio/trade_ledger`): `trade_ledger_executions`,
`trade_ledger_corporate_actions`, `trade_ledger_transfers`,
`trade_ledger_cash_transactions` and `trade_ledger_open_positions`, filled from IBKR's "My Trades" Flex Query and
keyed on IBKR's own ids so re-imports are idempotent.
Does NOT modify any existing tables.
"""

import sqlalchemy as sa
from alembic import op

revision = "014_trade_ledger"
down_revision = "013_p22_trial_asset_link"
branch_labels = None
depends_on = None


def _common_tail() -> list[sa.Column]:
    """Columns every trade-ledger table ends with."""
    return [
        sa.Column("source_file", sa.Text(), nullable=False),
        sa.Column("imported_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
    ]


def upgrade() -> None:
    op.create_table(
        "trade_ledger_executions",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("account_id", sa.Text(), nullable=False),
        sa.Column("trade_id", sa.Text(), nullable=False),
        sa.Column("transaction_id", sa.Text(), nullable=True),
        sa.Column("ib_order_id", sa.Text(), nullable=True),
        sa.Column("ib_exec_id", sa.Text(), nullable=True),
        sa.Column("conid", sa.Text(), nullable=False),
        sa.Column("isin", sa.Text(), nullable=True),
        sa.Column("symbol", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("listing_exchange", sa.Text(), nullable=True),
        sa.Column("sub_category", sa.Text(), nullable=True),
        sa.Column("currency", sa.Text(), nullable=False),
        sa.Column("fx_rate_to_base", sa.Numeric(), nullable=True),
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column("trade_datetime", sa.DateTime(), nullable=False),
        sa.Column("buy_sell", sa.Text(), nullable=False),
        sa.Column("quantity", sa.Numeric(), nullable=False),
        sa.Column("trade_price", sa.Numeric(), nullable=False),
        sa.Column("proceeds", sa.Numeric(), nullable=False),
        sa.Column("ib_commission", sa.Numeric(), nullable=False),
        sa.Column("ib_commission_currency", sa.Text(), nullable=False),
        sa.Column("taxes", sa.Numeric(), nullable=False),
        sa.Column("net_cash", sa.Numeric(), nullable=False),
        sa.Column("cost", sa.Numeric(), nullable=True),
        sa.Column("fifo_pnl_realized", sa.Numeric(), nullable=True),
        sa.Column("open_close", sa.Text(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        *_common_tail(),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("account_id", "trade_id", name="uq_trade_ledger_executions_key"),
    )
    op.create_index("idx_trade_ledger_executions_datetime", "trade_ledger_executions", ["trade_datetime"])

    op.create_table(
        "trade_ledger_corporate_actions",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("account_id", sa.Text(), nullable=False),
        sa.Column("transaction_id", sa.Text(), nullable=False),
        sa.Column("action_id", sa.Text(), nullable=True),
        sa.Column("action_type", sa.Text(), nullable=False),
        sa.Column("conid", sa.Text(), nullable=False),
        sa.Column("isin", sa.Text(), nullable=True),
        sa.Column("symbol", sa.Text(), nullable=False),
        sa.Column("action_description", sa.Text(), nullable=True),
        sa.Column("report_date", sa.Date(), nullable=True),
        sa.Column("action_datetime", sa.DateTime(), nullable=False),
        sa.Column("quantity", sa.Numeric(), nullable=False),
        sa.Column("proceeds", sa.Numeric(), nullable=False),
        sa.Column("value", sa.Numeric(), nullable=False),
        sa.Column("cost_basis", sa.Numeric(), nullable=True),
        sa.Column("currency", sa.Text(), nullable=False),
        *_common_tail(),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("account_id", "transaction_id", name="uq_trade_ledger_corporate_actions_key"),
    )

    op.create_table(
        "trade_ledger_transfers",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("account_id", sa.Text(), nullable=False),
        sa.Column("transaction_id", sa.Text(), nullable=False),
        sa.Column("transfer_type", sa.Text(), nullable=False),
        sa.Column("direction", sa.Text(), nullable=False),
        sa.Column("counterparty_account", sa.Text(), nullable=True),
        sa.Column("conid", sa.Text(), nullable=False),
        sa.Column("isin", sa.Text(), nullable=True),
        sa.Column("symbol", sa.Text(), nullable=False),
        sa.Column("transfer_datetime", sa.DateTime(), nullable=False),
        sa.Column("quantity", sa.Numeric(), nullable=False),
        sa.Column("transfer_price", sa.Numeric(), nullable=True),
        sa.Column("position_amount", sa.Numeric(), nullable=True),
        sa.Column("currency", sa.Text(), nullable=False),
        *_common_tail(),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("account_id", "transaction_id", name="uq_trade_ledger_transfers_key"),
    )

    op.create_table(
        "trade_ledger_cash_transactions",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("account_id", sa.Text(), nullable=False),
        sa.Column("transaction_id", sa.Text(), nullable=False),
        sa.Column("cash_type", sa.Text(), nullable=False),
        sa.Column("trade_id", sa.Text(), nullable=True),
        sa.Column("action_id", sa.Text(), nullable=True),
        sa.Column("conid", sa.Text(), nullable=False),
        sa.Column("isin", sa.Text(), nullable=True),
        sa.Column("symbol", sa.Text(), nullable=False),
        sa.Column("currency", sa.Text(), nullable=False),
        sa.Column("cash_datetime", sa.DateTime(), nullable=False),
        sa.Column("amount", sa.Numeric(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        *_common_tail(),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("account_id", "transaction_id", name="uq_trade_ledger_cash_transactions_key"),
    )

    op.create_table(
        "trade_ledger_open_positions",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("account_id", sa.Text(), nullable=False),
        sa.Column("report_date", sa.Date(), nullable=False),
        sa.Column("conid", sa.Text(), nullable=False),
        sa.Column("isin", sa.Text(), nullable=True),
        sa.Column("symbol", sa.Text(), nullable=False),
        sa.Column("position", sa.Numeric(), nullable=False),
        sa.Column("cost_basis_price", sa.Numeric(), nullable=True),
        sa.Column("cost_basis_money", sa.Numeric(), nullable=True),
        sa.Column("mark_price", sa.Numeric(), nullable=True),
        sa.Column("currency", sa.Text(), nullable=False),
        *_common_tail(),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("account_id", "report_date", "conid", name="uq_trade_ledger_open_positions_key"),
    )


def downgrade() -> None:
    op.drop_table("trade_ledger_open_positions")
    op.drop_table("trade_ledger_cash_transactions")
    op.drop_table("trade_ledger_transfers")
    op.drop_table("trade_ledger_corporate_actions")
    op.drop_index("idx_trade_ledger_executions_datetime", table_name="trade_ledger_executions")
    op.drop_table("trade_ledger_executions")
