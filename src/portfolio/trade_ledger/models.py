"""
Data models for the trade ledger.

Input events parsed from IBKR Flex Query XML ("My Trades" query), and the
outputs of the ledger engine. All money and quantity values are `Decimal` —
IBKR reports fractional shares (e.g. 0.0002 after chained reverse splits) and
sub-cent commissions, so floats are never used.

Datetimes are naive and in the timezone IBKR uses for the account's
statements (exchange-local for US listings); they are only compared with each
other, never with wall-clock time.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Execution:
    """One STK trade execution (Flex `Trade` row, levelOfDetail=EXECUTION)."""

    account_id: str
    trade_id: str
    transaction_id: Optional[str]
    ib_order_id: Optional[str]
    ib_exec_id: Optional[str]
    conid: str
    isin: Optional[str]
    symbol: str
    description: Optional[str]
    listing_exchange: Optional[str]
    sub_category: Optional[str]
    currency: str
    fx_rate_to_base: Optional[Decimal]
    trade_date: date
    trade_datetime: datetime
    buy_sell: str
    quantity: Decimal
    trade_price: Decimal
    proceeds: Decimal
    ib_commission: Decimal
    ib_commission_currency: str
    taxes: Decimal
    net_cash: Decimal
    cost: Optional[Decimal]
    fifo_pnl_realized: Optional[Decimal]
    open_close: Optional[str]
    notes: Optional[str]
    source_file: str


@dataclass(frozen=True)
class CorporateActionLeg:
    """
    One leg of a corporate action (Flex `CorporateAction` row).

    A split / ticker change appears as two legs sharing `action_id`: the old
    line removed (negative quantity) and the new line added (positive
    quantity, usually a new conid).
    """

    account_id: str
    transaction_id: str
    action_id: Optional[str]
    action_type: str
    conid: str
    isin: Optional[str]
    symbol: str
    action_description: Optional[str]
    report_date: Optional[date]
    action_datetime: datetime
    quantity: Decimal
    proceeds: Decimal
    value: Decimal
    cost_basis: Optional[Decimal]
    currency: str
    source_file: str


@dataclass(frozen=True)
class TransferEvent:
    """One STK position transfer (Flex `Transfer` row)."""

    account_id: str
    transaction_id: str
    transfer_type: str
    direction: str
    counterparty_account: Optional[str]
    conid: str
    isin: Optional[str]
    symbol: str
    transfer_datetime: datetime
    quantity: Decimal
    transfer_price: Optional[Decimal]
    position_amount: Optional[Decimal]
    currency: str
    source_file: str


CASH_COMMISSION_ADJUSTMENT = "Commission Adjustments"
CASH_DIVIDEND = "Dividends"
CASH_WITHHOLDING_TAX = "Withholding Tax"
CASH_PAYMENT_IN_LIEU = "Payment In Lieu Of Dividends"
CASH_TYPES = frozenset({CASH_COMMISSION_ADJUSTMENT, CASH_DIVIDEND, CASH_WITHHOLDING_TAX, CASH_PAYMENT_IN_LIEU})
DIVIDEND_CASH_TYPES = frozenset({CASH_DIVIDEND, CASH_WITHHOLDING_TAX, CASH_PAYMENT_IN_LIEU})


@dataclass(frozen=True)
class CashTransaction:
    """
    A per-security cash transaction (Flex `CashTransaction`) the ledger uses.

    `cash_type` is one of `CASH_TYPES`:
    - "Commission Adjustments": a commission refund / charge booked after the
      trade; positive `amount` is a refund; `trade_id` points at the execution.
    - "Dividends", "Payment In Lieu Of Dividends" (positive) and
      "Withholding Tax" (negative, or positive when IBKR reverses it): summed
      per ticker into net dividends; they never change the average cost.
    """

    account_id: str
    transaction_id: str
    cash_type: str
    trade_id: Optional[str]
    action_id: Optional[str]
    conid: str
    isin: Optional[str]
    symbol: str
    currency: str
    cash_datetime: datetime
    amount: Decimal
    description: Optional[str]
    source_file: str


@dataclass(frozen=True)
class IbkrOpenPosition:
    """IBKR's position at the end of a statement period (Flex `OpenPosition`, SUMMARY level)."""

    account_id: str
    report_date: date
    conid: str
    isin: Optional[str]
    symbol: str
    position: Decimal
    cost_basis_price: Optional[Decimal]
    cost_basis_money: Optional[Decimal]
    mark_price: Optional[Decimal]
    currency: str
    source_file: str


@dataclass
class ParsedStatement:
    """Everything parsed from one `FlexStatement` (one account, one period)."""

    account_id: str
    from_date: date
    to_date: date
    executions: List[Execution] = field(default_factory=list)
    corporate_actions: List[CorporateActionLeg] = field(default_factory=list)
    transfers: List[TransferEvent] = field(default_factory=list)
    cash_transactions: List[CashTransaction] = field(default_factory=list)
    open_positions: List[IbkrOpenPosition] = field(default_factory=list)


@dataclass
class LedgerEvents:
    """Every stored event, as dataclasses, oldest first."""

    executions: List[Execution] = field(default_factory=list)
    corporate_actions: List[CorporateActionLeg] = field(default_factory=list)
    transfers: List[TransferEvent] = field(default_factory=list)
    cash_transactions: List[CashTransaction] = field(default_factory=list)
    open_positions: List[IbkrOpenPosition] = field(default_factory=list)


@dataclass
class ParseResult:
    """Result of parsing one Flex XML file."""

    source_file: str
    statements: List[ParsedStatement] = field(default_factory=list)
    skipped: Dict[str, int] = field(default_factory=dict)

    def count_skip(self, reason: str) -> None:
        """
        Increment the skip counter for `reason`.

        Args:
            reason: Short reason key, e.g. "trade_non_stk".
        """
        self.skipped[reason] = self.skipped.get(reason, 0) + 1


# --------------------------------------------------------------------------
# Ledger engine outputs
# --------------------------------------------------------------------------


@dataclass
class PositionState:
    """
    Running state of one position (all accounts combined).

    `cost_excl` is the purchase cost of the shares still held at trade price;
    `cost_incl` additionally includes buy commissions and taxes. Both are
    reduced pro rata on sells, so the average prices do not change on a sell.
    """

    position_key: str
    symbol: str
    currency: str
    isin: Optional[str] = None
    shares: Decimal = Decimal(0)
    cost_excl: Decimal = Decimal(0)
    cost_incl: Decimal = Decimal(0)
    cycle_no: int = 0
    cycle_open_date: Optional[date] = None
    cycle_shares_bought: Decimal = Decimal(0)
    cycle_commissions: Decimal = Decimal(0)
    cycle_realized_excl: Decimal = Decimal(0)
    cycle_realized_incl: Decimal = Decimal(0)
    is_short: bool = False

    @property
    def wac_excl(self) -> Optional[Decimal]:
        """Average price per held share, excluding commissions (None when flat or short)."""
        if self.shares <= 0 or self.is_short:
            return None
        return self.cost_excl / self.shares

    @property
    def wac_incl(self) -> Optional[Decimal]:
        """Average price per held share, including buy commissions (None when flat or short)."""
        if self.shares <= 0 or self.is_short:
            return None
        return self.cost_incl / self.shares


@dataclass(frozen=True)
class ExecutionRow:
    """An execution annotated with the position state right after it."""

    execution: Execution
    position_key: str
    cycle_no: int
    shares_after: Decimal
    wac_excl_after: Optional[Decimal]
    wac_incl_after: Optional[Decimal]
    realized_excl: Optional[Decimal]
    realized_incl: Optional[Decimal]


@dataclass(frozen=True)
class CorporateActionRow:
    """One applied corporate action (all legs of one action), as seen by the ledger."""

    action_datetime: datetime
    position_key: str
    symbol: str
    action_type: str
    description: Optional[str]
    shares_before: Decimal
    shares_after: Decimal
    wac_incl_before: Optional[Decimal]
    wac_incl_after: Optional[Decimal]
    proceeds: Decimal
    realized_incl: Optional[Decimal]


@dataclass(frozen=True)
class ClosedCycle:
    """A position cycle from first buy to shares reaching zero."""

    position_key: str
    symbol: str
    currency: str
    cycle_no: int
    open_date: Optional[date]
    close_date: date
    shares_bought: Decimal
    commissions: Decimal
    realized_excl: Decimal
    realized_incl: Decimal


@dataclass(frozen=True)
class LedgerWarning:
    """Something the engine could not handle with certainty; shown in the report."""

    when: datetime
    symbol: str
    kind: str
    message: str


@dataclass
class PeriodTotals:
    """Activity of one position within a period (report month, year to date or lifetime).

    `dividends` is net: dividends + payments in lieu + withholding tax (negative).
    """

    shares_bought: Decimal = Decimal(0)
    shares_sold: Decimal = Decimal(0)
    commissions: Decimal = Decimal(0)
    realized_excl: Decimal = Decimal(0)
    realized_incl: Decimal = Decimal(0)
    dividends: Decimal = Decimal(0)
    executions: int = 0


@dataclass(frozen=True)
class PositionSummary:
    """Per-position line of the monthly report."""

    position_key: str
    symbol: str
    currency: str
    isin: Optional[str]
    shares: Decimal
    wac_excl: Optional[Decimal]
    wac_incl: Optional[Decimal]
    cycle_no: int
    cycle_open_date: Optional[date]
    month: PeriodTotals
    ytd: PeriodTotals


@dataclass
class LedgerResult:
    """Everything the ledger engine computed up to `period_end`."""

    period_start: date
    period_end: date
    execution_rows: List[ExecutionRow] = field(default_factory=list)
    corporate_action_rows: List[CorporateActionRow] = field(default_factory=list)
    positions: Dict[str, PositionState] = field(default_factory=dict)
    summaries: List[PositionSummary] = field(default_factory=list)
    closed_cycles: List[ClosedCycle] = field(default_factory=list)
    warnings: List[LedgerWarning] = field(default_factory=list)
    conid_to_key: Dict[str, str] = field(default_factory=dict)
    month_totals: Dict[str, PeriodTotals] = field(default_factory=dict)
    ytd_totals: Dict[str, PeriodTotals] = field(default_factory=dict)
    lifetime_totals: Dict[str, PeriodTotals] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Reconciliation outputs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReconciliationLine:
    """Ledger vs IBKR for one position at the snapshot date (all accounts combined)."""

    position_key: str
    symbol: str
    ledger_shares: Decimal
    ibkr_shares: Decimal
    wac_incl: Optional[Decimal]
    ibkr_cost_basis_price: Optional[Decimal]
    ibkr_mark_price: Optional[Decimal] = None

    @property
    def shares_match(self) -> bool:
        """True when share counts are exactly equal."""
        return self.ledger_shares == self.ibkr_shares


@dataclass
class ReconciliationReport:
    """
    Result of comparing the ledger with one IBKR open-positions snapshot.

    `status` is "ok", "mismatch", or "no_snapshot" (no IBKR snapshot in the
    period — e.g. historical months between yearly backfill files).
    """

    as_of: date
    snapshot_date: Optional[date]
    status: str
    lines: List[ReconciliationLine] = field(default_factory=list)

    @property
    def mismatches(self) -> List[ReconciliationLine]:
        """Lines whose share counts differ."""
        return [line for line in self.lines if not line.shares_match]

    def ibkr_cost_basis_by_key(self) -> Dict[str, Decimal]:
        """IBKR's combined cost-basis price per position key (for showing next to our WAC)."""
        return {line.position_key: line.ibkr_cost_basis_price for line in self.lines if line.ibkr_cost_basis_price is not None}

    def mark_price_by_key(self) -> Dict[str, Decimal]:
        """IBKR's closing (mark) price per position key on the snapshot date."""
        return {line.position_key: line.ibkr_mark_price for line in self.lines if line.ibkr_mark_price is not None}
