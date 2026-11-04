"""Builders for synthetic ledger events shared by the trade-ledger tests (no real account data)."""

from datetime import datetime
from decimal import Decimal
from typing import Optional

from src.portfolio.trade_ledger.models import (
    CASH_COMMISSION_ADJUSTMENT,
    CASH_DEPOSITS_WITHDRAWALS,
    CASH_DIVIDEND,
    CashTransaction,
    CorporateActionLeg,
    Execution,
    TransferEvent,
)

D = Decimal
ACC = "U0000001"
ACC2 = "U0000002"
_counter = iter(range(1, 1_000_000))


def trade(
    when: str, qty: str, price: str, commission: str = "-1", conid: str = "100", symbol: str = "ABC", account: str = ACC
) -> Execution:
    """Build an execution; `when` is 'YYYY-MM-DD HH:MM'."""
    dt = datetime.fromisoformat(when)
    quantity = D(qty)
    proceeds = -quantity * D(price)
    return Execution(
        account_id=account, trade_id=f"T{next(_counter)}", transaction_id=None, ib_order_id=None, ib_exec_id=None,
        conid=conid, isin=None, symbol=symbol, description=None, listing_exchange=None, sub_category=None,
        currency="USD", fx_rate_to_base=None, trade_date=dt.date(), trade_datetime=dt,
        buy_sell="BUY" if quantity > 0 else "SELL", quantity=quantity, trade_price=D(price), proceeds=proceeds,
        ib_commission=D(commission), ib_commission_currency="USD", taxes=D(0), net_cash=proceeds + D(commission),
        cost=None, fifo_pnl_realized=None, open_close=None, notes=None, source_file="t.xml",
    )


def leg(
    when: str, conid: str, qty: str, action_id: str = "A1", action_type: str = "RS", symbol: str = "ABC", proceeds: str = "0"
) -> CorporateActionLeg:
    """Build one corporate-action leg."""
    return CorporateActionLeg(
        account_id=ACC, transaction_id=f"C{next(_counter)}", action_id=action_id, action_type=action_type, conid=conid,
        isin=None, symbol=symbol, action_description=f"{symbol} {action_type}", report_date=None,
        action_datetime=datetime.fromisoformat(when), quantity=D(qty), proceeds=D(proceeds), value=D(0), cost_basis=None,
        currency="USD", source_file="t.xml",
    )


def transfer(when: str, qty: str, account: str, counterparty: str, kind: str = "INTERNAL", amount: Optional[str] = None) -> TransferEvent:
    """Build one position transfer."""
    return TransferEvent(
        account_id=account, transaction_id=f"X{next(_counter)}", transfer_type=kind, direction="IN" if D(qty) > 0 else "OUT",
        counterparty_account=counterparty, conid="100", isin=None, symbol="ABC", transfer_datetime=datetime.fromisoformat(when),
        quantity=D(qty), transfer_price=None, position_amount=D(amount) if amount else None, currency="USD", source_file="t.xml",
    )


def refund(when: str, execution: Execution, amount: str) -> CashTransaction:
    """Build a commission refund for `execution`."""
    return CashTransaction(
        account_id=execution.account_id, transaction_id=f"R{next(_counter)}", cash_type=CASH_COMMISSION_ADJUSTMENT, trade_id=execution.trade_id,
        action_id=None, conid=execution.conid, isin=None, symbol=execution.symbol, currency="USD", cash_datetime=datetime.fromisoformat(when),
        amount=D(amount), description=None, source_file="t.xml",
    )


def deposit(when: str, amount: str, currency: str = "CHF", description: str = "CASH RECEIPTS / ELECTRONIC FUND TRANSFERS", account: str = ACC) -> CashTransaction:
    """Build an account-level deposit (positive) or withdrawal (negative)."""
    return CashTransaction(
        account_id=account, transaction_id=f"W{next(_counter)}", cash_type=CASH_DEPOSITS_WITHDRAWALS, trade_id=None, action_id=None,
        conid="", isin=None, symbol="", currency=currency, cash_datetime=datetime.fromisoformat(when), amount=D(amount),
        description=description, source_file="t.xml",
    )


def cash(when: str, amount: str, cash_type: str = CASH_DIVIDEND, conid: str = "100", symbol: str = "ABC") -> CashTransaction:
    """Build a dividend-type cash transaction (dividend, withholding tax, payment in lieu)."""
    return CashTransaction(
        account_id=ACC, transaction_id=f"D{next(_counter)}", cash_type=cash_type, trade_id=None, action_id=f"DA{next(_counter)}", conid=conid,
        isin=None, symbol=symbol, currency="USD", cash_datetime=datetime.fromisoformat(when), amount=D(amount), description=None,
        source_file="t.xml",
    )
