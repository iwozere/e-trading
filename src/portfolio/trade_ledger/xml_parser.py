"""
Parser for IBKR Flex Query XML ("My Trades" Activity query).

Reads every `FlexStatement` in a file (one per account) and returns STK
executions, corporate-action legs, transfers, per-security cash transactions
(commission adjustments, dividends, withholding tax) and period-end open
positions as `Decimal`-based dataclasses. Non-STK rows (e.g. CASH = FX
conversions) and malformed rows are skipped and counted, never raised.

Expected query settings: Date Format `yyyyMMdd`, Time Format `HHmmss`,
Date/Time Separator `;` (see docs/flex-query-setup.md).
"""

import xml.etree.ElementTree as ET
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Callable, List, Optional, TypeVar

from src.notification.logger import setup_logger
from src.portfolio.trade_ledger.models import (
    CASH_DEPOSITS_WITHDRAWALS,
    CASH_TYPES,
    CashTransaction,
    CorporateActionLeg,
    Execution,
    IbkrOpenPosition,
    ParsedStatement,
    ParseResult,
    TransferEvent,
)

_logger = setup_logger(__name__)

_STK = "STK"
_EXECUTION_LEVEL = "EXECUTION"
_SUMMARY_LEVEL = "SUMMARY"

T = TypeVar("T")


class FlexParseError(ValueError):
    """Raised when a required attribute is missing or unparseable."""


def mask_account(account_id: str) -> str:
    """
    Mask an IBKR account id for logs (``U1234567`` -> ``U1***67``).

    Args:
        account_id: Raw account id.

    Returns:
        Masked account id.
    """
    if len(account_id) <= 4:
        return "***"
    return f"{account_id[:2]}***{account_id[-2:]}"


def _req(el: ET.Element, name: str) -> str:
    """Return a required, non-empty attribute or raise `FlexParseError`."""
    value = (el.get(name) or "").strip()
    if not value:
        raise FlexParseError(f"missing attribute {name!r}")
    return value


def _opt(el: ET.Element, name: str) -> Optional[str]:
    """Return an optional attribute; empty strings and IBKR's '--' become None."""
    value = (el.get(name) or "").strip()
    if not value or value == "--":
        return None
    return value


def _to_decimal(raw: str, name: str) -> Decimal:
    """Convert a string to Decimal or raise `FlexParseError`."""
    try:
        return Decimal(raw)
    except InvalidOperation as exc:
        raise FlexParseError(f"attribute {name!r} is not a number: {raw!r}") from exc


def _dec(el: ET.Element, name: str) -> Decimal:
    """Required Decimal attribute."""
    return _to_decimal(_req(el, name), name)


def _dec_or_zero(el: ET.Element, name: str) -> Decimal:
    """Decimal attribute that defaults to 0 when empty."""
    raw = _opt(el, name)
    return _to_decimal(raw, name) if raw is not None else Decimal(0)


def _dec_opt(el: ET.Element, name: str) -> Optional[Decimal]:
    """Optional Decimal attribute."""
    raw = _opt(el, name)
    return _to_decimal(raw, name) if raw is not None else None


def parse_flex_date(raw: str) -> date:
    """
    Parse a Flex date (``yyyyMMdd``).

    Args:
        raw: Date string, e.g. ``20251231``.

    Returns:
        Parsed date.

    Raises:
        FlexParseError: If the string is not in ``yyyyMMdd`` format.
    """
    try:
        return datetime.strptime(raw, "%Y%m%d").date()
    except ValueError as exc:
        raise FlexParseError(f"bad date {raw!r} (expected yyyyMMdd)") from exc


def parse_flex_datetime(raw: str) -> datetime:
    """
    Parse a Flex date/time (``yyyyMMdd;HHmmss``) or a bare date (midnight).

    Args:
        raw: Date/time string, e.g. ``20250521;150756``.

    Returns:
        Naive datetime.

    Raises:
        FlexParseError: If the string matches neither format.
    """
    try:
        if ";" in raw:
            return datetime.strptime(raw, "%Y%m%d;%H%M%S")
        return datetime.strptime(raw, "%Y%m%d")
    except ValueError as exc:
        raise FlexParseError(f"bad datetime {raw!r} (expected yyyyMMdd;HHmmss)") from exc


def _date_opt(el: ET.Element, name: str) -> Optional[date]:
    """Optional date attribute."""
    raw = _opt(el, name)
    return parse_flex_date(raw) if raw is not None else None


def _parse_execution(el: ET.Element, source_file: str) -> Execution:
    """Build an `Execution` from a `Trade` element."""
    return Execution(
        account_id=_req(el, "accountId"),
        trade_id=_req(el, "tradeID"),
        transaction_id=_opt(el, "transactionID"),
        ib_order_id=_opt(el, "ibOrderID"),
        ib_exec_id=_opt(el, "ibExecID"),
        conid=_req(el, "conid"),
        isin=_opt(el, "isin"),
        symbol=_req(el, "symbol"),
        description=_opt(el, "description"),
        listing_exchange=_opt(el, "listingExchange"),
        sub_category=_opt(el, "subCategory"),
        currency=_req(el, "currency"),
        fx_rate_to_base=_dec_opt(el, "fxRateToBase"),
        trade_date=parse_flex_date(_req(el, "tradeDate")),
        trade_datetime=parse_flex_datetime(_req(el, "dateTime")),
        buy_sell=_req(el, "buySell"),
        quantity=_dec(el, "quantity"),
        trade_price=_dec(el, "tradePrice"),
        proceeds=_dec_or_zero(el, "proceeds"),
        ib_commission=_dec_or_zero(el, "ibCommission"),
        ib_commission_currency=_opt(el, "ibCommissionCurrency") or _req(el, "currency"),
        taxes=_dec_or_zero(el, "taxes"),
        net_cash=_dec_or_zero(el, "netCash"),
        cost=_dec_opt(el, "cost"),
        fifo_pnl_realized=_dec_opt(el, "fifoPnlRealized"),
        open_close=_opt(el, "openCloseIndicator"),
        notes=_opt(el, "notes"),
        source_file=source_file,
    )


def _parse_corporate_action(el: ET.Element, source_file: str) -> CorporateActionLeg:
    """Build a `CorporateActionLeg` from a `CorporateAction` element."""
    return CorporateActionLeg(
        account_id=_req(el, "accountId"),
        transaction_id=_req(el, "transactionID"),
        action_id=_opt(el, "actionID"),
        action_type=_req(el, "type"),
        conid=_req(el, "conid"),
        isin=_opt(el, "isin"),
        symbol=_req(el, "symbol"),
        action_description=_opt(el, "actionDescription"),
        report_date=_date_opt(el, "reportDate"),
        action_datetime=parse_flex_datetime(_req(el, "dateTime")),
        quantity=_dec(el, "quantity"),
        proceeds=_dec_or_zero(el, "proceeds"),
        value=_dec_or_zero(el, "value"),
        cost_basis=_dec_opt(el, "costBasis"),
        currency=_req(el, "currency"),
        source_file=source_file,
    )


def _parse_transfer(el: ET.Element, source_file: str) -> TransferEvent:
    """Build a `TransferEvent` from a `Transfer` element."""
    raw_dt = _opt(el, "dateTime") or _req(el, "date")
    return TransferEvent(
        account_id=_req(el, "accountId"),
        transaction_id=_req(el, "transactionID"),
        transfer_type=_req(el, "type"),
        direction=_req(el, "direction"),
        counterparty_account=_opt(el, "account"),
        conid=_req(el, "conid"),
        isin=_opt(el, "isin"),
        symbol=_req(el, "symbol"),
        transfer_datetime=parse_flex_datetime(raw_dt),
        quantity=_dec(el, "quantity"),
        transfer_price=_dec_opt(el, "transferPrice"),
        position_amount=_dec_opt(el, "positionAmount"),
        currency=_req(el, "currency"),
        source_file=source_file,
    )


def _parse_cash_transaction(el: ET.Element, source_file: str) -> CashTransaction:
    """Build a `CashTransaction` from a `CashTransaction` element whose type is in `CASH_TYPES`."""
    return CashTransaction(
        account_id=_req(el, "accountId"),
        transaction_id=_req(el, "transactionID"),
        cash_type=_req(el, "type"),
        trade_id=_opt(el, "tradeID"),
        action_id=_opt(el, "actionID"),
        conid=_req(el, "conid"),
        isin=_opt(el, "isin"),
        symbol=_req(el, "symbol"),
        currency=_req(el, "currency"),
        cash_datetime=parse_flex_datetime(_req(el, "dateTime")),
        amount=_dec(el, "amount"),
        description=_opt(el, "description"),
        source_file=source_file,
    )


def _parse_deposit(el: ET.Element, source_file: str) -> CashTransaction:
    """Build an account-level `CashTransaction` (Deposits/Withdrawals; no security, so conid / symbol are "")."""
    return CashTransaction(
        account_id=_req(el, "accountId"),
        transaction_id=_req(el, "transactionID"),
        cash_type=_req(el, "type"),
        trade_id=None,
        action_id=None,
        conid=_opt(el, "conid") or "",
        isin=None,
        symbol="",
        currency=_req(el, "currency"),
        cash_datetime=parse_flex_datetime(_req(el, "dateTime")),
        amount=_dec(el, "amount"),
        description=_opt(el, "description"),
        source_file=source_file,
    )


def _parse_open_position(el: ET.Element, source_file: str, fallback_date: date) -> IbkrOpenPosition:
    """Build an `IbkrOpenPosition` from an `OpenPosition` element."""
    return IbkrOpenPosition(
        account_id=_req(el, "accountId"),
        report_date=_date_opt(el, "reportDate") or fallback_date,
        conid=_req(el, "conid"),
        isin=_opt(el, "isin"),
        symbol=_req(el, "symbol"),
        position=_dec(el, "position"),
        cost_basis_price=_dec_opt(el, "costBasisPrice"),
        cost_basis_money=_dec_opt(el, "costBasisMoney"),
        mark_price=_dec_opt(el, "markPrice"),
        currency=_req(el, "currency"),
        source_file=source_file,
    )


def _collect(
    elements: List[ET.Element],
    build: Callable[[ET.Element], T],
    kind: str,
    account_id: str,
    result: ParseResult,
    stk_only: bool = True,
) -> List[T]:
    """
    Build rows from STK elements, counting and logging skipped ones.

    Args:
        elements: Raw XML elements of one kind.
        build: Element -> dataclass factory.
        kind: Row kind for skip-reason keys and logs ("trade", ...).
        account_id: Statement account id (for masked logging).
        result: ParseResult whose skip counters are updated.
        stk_only: Skip rows whose assetCategory is not STK (False for account-level cash).

    Returns:
        Successfully built rows.
    """
    rows: List[T] = []
    for el in elements:
        if stk_only and el.get("assetCategory") != _STK:
            result.count_skip(f"{kind}_non_stk")
            continue
        try:
            rows.append(build(el))
        except FlexParseError as exc:
            result.count_skip(f"{kind}_malformed")
            _logger.warning(
                "Skipping malformed %s row (account %s, symbol %s) in %s: %s",
                kind,
                mask_account(account_id),
                el.get("symbol"),
                result.source_file,
                exc,
            )
    return rows


def _parse_statement(st: ET.Element, result: ParseResult) -> ParsedStatement:
    """Parse one `FlexStatement` element."""
    account_id = _req(st, "accountId")
    source = result.source_file
    to_date = parse_flex_date(_req(st, "toDate"))
    statement = ParsedStatement(
        account_id=account_id,
        from_date=parse_flex_date(_req(st, "fromDate")),
        to_date=to_date,
    )

    trades = []
    for el in st.iter("Trade"):
        level = el.get("levelOfDetail")
        if level and level != _EXECUTION_LEVEL:
            result.count_skip("trade_not_execution")
            continue
        trades.append(el)
    statement.executions = _collect(trades, lambda e: _parse_execution(e, source), "trade", account_id, result)

    statement.corporate_actions = _collect(
        list(st.iter("CorporateAction")), lambda e: _parse_corporate_action(e, source), "corporate_action", account_id, result
    )
    statement.transfers = _collect(list(st.iter("Transfer")), lambda e: _parse_transfer(e, source), "transfer", account_id, result)

    # Per-security cash (commission adjustments, dividends) plus account-level deposits / withdrawals.
    # Interest, fees etc. are ignored.
    cash_elements = list(st.iter("CashTransaction"))
    cash = [el for el in cash_elements if el.get("type") in CASH_TYPES]
    statement.cash_transactions = _collect(cash, lambda e: _parse_cash_transaction(e, source), "cash_transaction", account_id, result)
    deposits = [el for el in cash_elements if el.get("type") == CASH_DEPOSITS_WITHDRAWALS]
    statement.cash_transactions += _collect(deposits, lambda e: _parse_deposit(e, source), "deposit", account_id, result, stk_only=False)

    positions = []
    for el in st.iter("OpenPosition"):
        level = el.get("levelOfDetail")
        if level and level != _SUMMARY_LEVEL:
            result.count_skip("open_position_not_summary")
            continue
        positions.append(el)
    statement.open_positions = _collect(
        positions, lambda e: _parse_open_position(e, source, to_date), "open_position", account_id, result
    )
    return statement


def parse_flex_bytes(xml_bytes: bytes, source_file: str) -> ParseResult:
    """
    Parse a Flex Query XML document held in memory.

    Args:
        xml_bytes: Raw `FlexQueryResponse` XML.
        source_file: Name recorded on every row (file name, not a full path).

    Returns:
        ParseResult with one ParsedStatement per account.

    Raises:
        FlexParseError: If the document is not a FlexQueryResponse or a
            statement header is unusable.
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise FlexParseError(f"{source_file}: not valid XML: {exc}") from exc
    if root.tag != "FlexQueryResponse":
        raise FlexParseError(f"{source_file}: root element is {root.tag!r}, expected 'FlexQueryResponse'")

    result = ParseResult(source_file=source_file)
    for st in root.iter("FlexStatement"):
        result.statements.append(_parse_statement(st, result))

    _logger.info(
        "Parsed %s: %d statement(s), %d executions, %d corporate-action legs, %d transfers, %d cash transactions, %d open positions, skipped %s",
        source_file,
        len(result.statements),
        sum(len(s.executions) for s in result.statements),
        sum(len(s.corporate_actions) for s in result.statements),
        sum(len(s.transfers) for s in result.statements),
        sum(len(s.cash_transactions) for s in result.statements),
        sum(len(s.open_positions) for s in result.statements),
        result.skipped,
    )
    return result


def parse_flex_file(path: Path) -> ParseResult:
    """
    Parse a Flex Query XML file.

    Args:
        path: Path to the XML file.

    Returns:
        ParseResult with one ParsedStatement per account.

    Raises:
        FlexParseError: If the file is not a usable FlexQueryResponse.
        OSError: If the file cannot be read.
    """
    return parse_flex_bytes(path.read_bytes(), path.name)
