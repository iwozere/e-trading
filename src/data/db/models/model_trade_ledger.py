"""
SQLAlchemy ORM models for the trade ledger (src/portfolio/trade_ledger).

Raw IBKR Flex Query events — executions, corporate-action legs, transfers,
per-security cash transactions and period-end open positions — stored idempotently on IBKR's own ids. The ledger
engine recomputes everything from these rows; no derived state is stored.
Column names equal the field names of the dataclasses in
``src.portfolio.trade_ledger.models``. No existing tables are modified.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import BigInteger, Date, DateTime, Index, Numeric, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from src.data.db.core.base import Base


class TradeLedgerExecution(Base):
    """One STK trade execution."""

    __tablename__ = "trade_ledger_executions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    account_id: Mapped[str] = mapped_column(Text, nullable=False)
    trade_id: Mapped[str] = mapped_column(Text, nullable=False)
    transaction_id: Mapped[str | None] = mapped_column(Text)
    ib_order_id: Mapped[str | None] = mapped_column(Text)
    ib_exec_id: Mapped[str | None] = mapped_column(Text)
    conid: Mapped[str] = mapped_column(Text, nullable=False)
    isin: Mapped[str | None] = mapped_column(Text)
    symbol: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    listing_exchange: Mapped[str | None] = mapped_column(Text)
    sub_category: Mapped[str | None] = mapped_column(Text)
    currency: Mapped[str] = mapped_column(Text, nullable=False)
    fx_rate_to_base: Mapped[Decimal | None] = mapped_column(Numeric)
    trade_date: Mapped[date] = mapped_column(Date, nullable=False)
    trade_datetime: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    buy_sell: Mapped[str] = mapped_column(Text, nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric, nullable=False)
    trade_price: Mapped[Decimal] = mapped_column(Numeric, nullable=False)
    proceeds: Mapped[Decimal] = mapped_column(Numeric, nullable=False)
    ib_commission: Mapped[Decimal] = mapped_column(Numeric, nullable=False)
    ib_commission_currency: Mapped[str] = mapped_column(Text, nullable=False)
    taxes: Mapped[Decimal] = mapped_column(Numeric, nullable=False)
    net_cash: Mapped[Decimal] = mapped_column(Numeric, nullable=False)
    cost: Mapped[Decimal | None] = mapped_column(Numeric)
    fifo_pnl_realized: Mapped[Decimal | None] = mapped_column(Numeric)
    open_close: Mapped[str | None] = mapped_column(Text)
    notes: Mapped[str | None] = mapped_column(Text)
    source_file: Mapped[str] = mapped_column(Text, nullable=False)
    imported_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("account_id", "trade_id", name="uq_trade_ledger_executions_key"),
        Index("idx_trade_ledger_executions_datetime", "trade_datetime"),
    )


class TradeLedgerCorporateAction(Base):
    """One corporate-action leg (splits / ticker changes come as two legs sharing ``action_id``)."""

    __tablename__ = "trade_ledger_corporate_actions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    account_id: Mapped[str] = mapped_column(Text, nullable=False)
    transaction_id: Mapped[str] = mapped_column(Text, nullable=False)
    action_id: Mapped[str | None] = mapped_column(Text)
    action_type: Mapped[str] = mapped_column(Text, nullable=False)
    conid: Mapped[str] = mapped_column(Text, nullable=False)
    isin: Mapped[str | None] = mapped_column(Text)
    symbol: Mapped[str] = mapped_column(Text, nullable=False)
    action_description: Mapped[str | None] = mapped_column(Text)
    report_date: Mapped[date | None] = mapped_column(Date)
    action_datetime: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric, nullable=False)
    proceeds: Mapped[Decimal] = mapped_column(Numeric, nullable=False)
    value: Mapped[Decimal] = mapped_column(Numeric, nullable=False)
    cost_basis: Mapped[Decimal | None] = mapped_column(Numeric)
    currency: Mapped[str] = mapped_column(Text, nullable=False)
    source_file: Mapped[str] = mapped_column(Text, nullable=False)
    imported_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (UniqueConstraint("account_id", "transaction_id", name="uq_trade_ledger_corporate_actions_key"),)


class TradeLedgerTransfer(Base):
    """One STK position transfer (internal between own accounts, or external)."""

    __tablename__ = "trade_ledger_transfers"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    account_id: Mapped[str] = mapped_column(Text, nullable=False)
    transaction_id: Mapped[str] = mapped_column(Text, nullable=False)
    transfer_type: Mapped[str] = mapped_column(Text, nullable=False)
    direction: Mapped[str] = mapped_column(Text, nullable=False)
    counterparty_account: Mapped[str | None] = mapped_column(Text)
    conid: Mapped[str] = mapped_column(Text, nullable=False)
    isin: Mapped[str | None] = mapped_column(Text)
    symbol: Mapped[str] = mapped_column(Text, nullable=False)
    transfer_datetime: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric, nullable=False)
    transfer_price: Mapped[Decimal | None] = mapped_column(Numeric)
    position_amount: Mapped[Decimal | None] = mapped_column(Numeric)
    currency: Mapped[str] = mapped_column(Text, nullable=False)
    source_file: Mapped[str] = mapped_column(Text, nullable=False)
    imported_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (UniqueConstraint("account_id", "transaction_id", name="uq_trade_ledger_transfers_key"),)


class TradeLedgerCashTransaction(Base):
    """A per-security cash transaction: commission adjustment, dividend, payment in lieu or withholding tax."""

    __tablename__ = "trade_ledger_cash_transactions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    account_id: Mapped[str] = mapped_column(Text, nullable=False)
    transaction_id: Mapped[str] = mapped_column(Text, nullable=False)
    cash_type: Mapped[str] = mapped_column(Text, nullable=False)
    trade_id: Mapped[str | None] = mapped_column(Text)
    action_id: Mapped[str | None] = mapped_column(Text)
    conid: Mapped[str] = mapped_column(Text, nullable=False)
    isin: Mapped[str | None] = mapped_column(Text)
    symbol: Mapped[str] = mapped_column(Text, nullable=False)
    currency: Mapped[str] = mapped_column(Text, nullable=False)
    cash_datetime: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    amount: Mapped[Decimal] = mapped_column(Numeric, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    source_file: Mapped[str] = mapped_column(Text, nullable=False)
    imported_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (UniqueConstraint("account_id", "transaction_id", name="uq_trade_ledger_cash_transactions_key"),)


class TradeLedgerOpenPosition(Base):
    """IBKR's position at the end of a statement period (reconciliation checkpoint)."""

    __tablename__ = "trade_ledger_open_positions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    account_id: Mapped[str] = mapped_column(Text, nullable=False)
    report_date: Mapped[date] = mapped_column(Date, nullable=False)
    conid: Mapped[str] = mapped_column(Text, nullable=False)
    isin: Mapped[str | None] = mapped_column(Text)
    symbol: Mapped[str] = mapped_column(Text, nullable=False)
    position: Mapped[Decimal] = mapped_column(Numeric, nullable=False)
    cost_basis_price: Mapped[Decimal | None] = mapped_column(Numeric)
    cost_basis_money: Mapped[Decimal | None] = mapped_column(Numeric)
    mark_price: Mapped[Decimal | None] = mapped_column(Numeric)
    currency: Mapped[str] = mapped_column(Text, nullable=False)
    source_file: Mapped[str] = mapped_column(Text, nullable=False)
    imported_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (UniqueConstraint("account_id", "report_date", "conid", name="uq_trade_ledger_open_positions_key"),)
