"""
Ledger engine: replays the full trade history into weighted average costs.

Pure — no I/O. Given every stored event, `build_ledger` replays them in
chronological order (all accounts combined) and returns per-position state,
annotated executions, closed cycles, month / YTD totals and warnings.

Rules (see docs/Design.md, "Ledger Engine"):
- BUY adds shares and cost; SELL removes shares at the current average, so the
  average does not change on a sell; realized P&L = proceeds − fees − cost.
- Two averages: excluding and including buy commissions/taxes.
- A position that reaches exactly 0 shares closes its cycle; the next buy
  starts a fresh one.
- Corporate-action legs sharing an `actionID` are applied together; shares
  move to the new conid, total cost is carried unchanged (splits, ticker /
  ISIN changes).
- Internal transfers between the user's own accounts are no-ops.
- Commission adjustments (refunds booked weeks later) lower the cost of a
  still-held buy, otherwise they count as realized P&L.
- Dividends, payments in lieu and withholding tax are summed per ticker
  (net dividends); they never change the average cost.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Dict, Iterable, List, Optional, Tuple, Union

from src.notification.logger import setup_logger
from src.portfolio.trade_ledger.models import (
    CASH_COMMISSION_ADJUSTMENT,
    CASH_DEPOSITS_WITHDRAWALS,
    DIVIDEND_CASH_TYPES,
    ClosedCycle,
    CashTransaction,
    CorporateActionLeg,
    CorporateActionRow,
    DepositTotals,
    Execution,
    ExecutionRow,
    LedgerEvents,
    LedgerResult,
    LedgerWarning,
    PeriodTotals,
    PositionState,
    PositionSummary,
    TransferEvent,
)

_logger = setup_logger(__name__)

_ZERO = Decimal(0)

# Corporate-action types that only re-label / re-scale a holding (cost carried unchanged):
# forward split, reverse split, ticker change, CUSIP/ISIN change.
SPLIT_LIKE_TYPES = frozenset({"FS", "RS", "TC", "IC"})

_RANK_TRADE = 0
_RANK_CORPORATE_ACTION = 1
_RANK_TRANSFER = 2
_RANK_CASH = 3

_Payload = Union[Execution, List[CorporateActionLeg], TransferEvent, CashTransaction]


@dataclass(frozen=True)
class _Event:
    """One replayable event; sorted by (when, rank, tiebreak)."""

    when: datetime
    rank: int
    tiebreak: str
    payload: _Payload


def _group_corporate_actions(legs: Iterable[CorporateActionLeg]) -> List[List[CorporateActionLeg]]:
    """Group legs by `action_id` (legs without one are their own group)."""
    groups: Dict[str, List[CorporateActionLeg]] = defaultdict(list)
    for leg in legs:
        groups[leg.action_id or f"tx:{leg.account_id}:{leg.transaction_id}"].append(leg)
    return list(groups.values())


def _build_events(events: LedgerEvents, period_end: date) -> List[_Event]:
    """Merge executions, corporate-action groups and transfers up to `period_end`, oldest first."""
    out: List[_Event] = []
    for ex in events.executions:
        if ex.trade_datetime.date() <= period_end:
            out.append(_Event(ex.trade_datetime, _RANK_TRADE, ex.trade_id, ex))
    for group in _group_corporate_actions(events.corporate_actions):
        when = min(leg.action_datetime for leg in group)
        if when.date() <= period_end:
            out.append(_Event(when, _RANK_CORPORATE_ACTION, group[0].action_id or group[0].transaction_id, group))
    for tr in events.transfers:
        if tr.transfer_datetime.date() <= period_end:
            out.append(_Event(tr.transfer_datetime, _RANK_TRANSFER, tr.transaction_id, tr))
    for adj in events.cash_transactions:
        if adj.cash_datetime.date() <= period_end:
            out.append(_Event(adj.cash_datetime, _RANK_CASH, adj.transaction_id, adj))
    out.sort(key=lambda e: (e.when, e.rank, e.tiebreak))
    return out


def _own_accounts(events: LedgerEvents) -> set[str]:
    """Every account id that appears in the data (the user's own accounts)."""
    accounts = {e.account_id for e in events.executions}
    accounts.update(c.account_id for c in events.corporate_actions)
    accounts.update(t.account_id for t in events.transfers)
    accounts.update(p.account_id for p in events.open_positions)
    return accounts


class _Engine:
    """Mutable replay state; use `build_ledger` instead of this class directly."""

    def __init__(self, events: LedgerEvents, period_start: date, period_end: date) -> None:
        self.events = events
        self.period_start = period_start
        self.period_end = period_end
        self.ytd_start = date(period_end.year, 1, 1)
        self.own_accounts = _own_accounts(events)
        self.trade_side = {(e.account_id, e.trade_id): e.quantity > 0 for e in events.executions}
        self.alias: Dict[str, str] = {}
        self.positions: Dict[str, PositionState] = {}
        self.month: Dict[str, PeriodTotals] = defaultdict(PeriodTotals)
        self.ytd: Dict[str, PeriodTotals] = defaultdict(PeriodTotals)
        self.lifetime: Dict[str, PeriodTotals] = defaultdict(PeriodTotals)
        self.result = LedgerResult(period_start=period_start, period_end=period_end)

    # ---------------------------------------------------------------- helpers

    def _resolve(self, conid: str) -> str:
        """Follow corporate-action aliases to the position key of a conid."""
        seen = set()
        while conid in self.alias and conid not in seen:
            seen.add(conid)
            conid = self.alias[conid]
        return conid

    def _position(self, conid: str, symbol: str, currency: str, isin: Optional[str]) -> PositionState:
        """Return the position for a conid, creating it on first sight."""
        key = self._resolve(conid)
        pos = self.positions.get(key)
        if pos is None:
            pos = PositionState(position_key=key, symbol=symbol, currency=currency, isin=isin)
            self.positions[key] = pos
        return pos

    def _warn(self, when: datetime, symbol: str, kind: str, message: str) -> None:
        """Record a warning for the report."""
        self.result.warnings.append(LedgerWarning(when=when, symbol=symbol, kind=kind, message=message))
        _logger.warning("Ledger warning [%s] %s %s: %s", kind, when.date(), symbol, message)

    def _totals(self, key: str, day: date) -> List[PeriodTotals]:
        """Period buckets (lifetime, YTD, month) that an event on `day` counts towards."""
        buckets: List[PeriodTotals] = [self.lifetime[key]] if day <= self.period_end else []
        if self.ytd_start <= day <= self.period_end:
            buckets.append(self.ytd[key])
        if self.period_start <= day <= self.period_end:
            buckets.append(self.month[key])
        return buckets

    def _open_cycle(self, pos: PositionState, day: date) -> None:
        """Start a new position cycle."""
        pos.cycle_no += 1
        pos.cycle_open_date = day
        pos.cycle_shares_bought = _ZERO
        pos.cycle_commissions = _ZERO
        pos.cycle_realized_excl = _ZERO
        pos.cycle_realized_incl = _ZERO

    def _close_cycle(self, pos: PositionState, day: date) -> None:
        """Record a finished cycle and reset cost to exactly zero."""
        self.result.closed_cycles.append(
            ClosedCycle(
                position_key=pos.position_key,
                symbol=pos.symbol,
                currency=pos.currency,
                cycle_no=pos.cycle_no,
                open_date=pos.cycle_open_date,
                close_date=day,
                shares_bought=pos.cycle_shares_bought,
                commissions=pos.cycle_commissions,
                realized_excl=pos.cycle_realized_excl,
                realized_incl=pos.cycle_realized_incl,
            )
        )
        pos.cost_excl = _ZERO
        pos.cost_incl = _ZERO

    def _add_shares(self, pos: PositionState, qty: Decimal, cost_excl: Decimal, cost_incl: Decimal, day: date) -> None:
        """Add shares and their cost; opens a cycle when the position was flat; covers shorts."""
        if pos.is_short:
            pos.shares += qty
            if pos.shares < 0:
                return
            pos.is_short = False
            if pos.shares == 0:
                return
            fraction = pos.shares / qty  # part of this buy that is a new long position
            pos.cost_excl = cost_excl * fraction
            pos.cost_incl = cost_incl * fraction
            self._open_cycle(pos, day)
            pos.cycle_shares_bought = pos.shares
            return
        if pos.shares == 0:
            self._open_cycle(pos, day)
        pos.shares += qty
        pos.cost_excl += cost_excl
        pos.cost_incl += cost_incl
        pos.cycle_shares_bought += qty

    def _remove_shares(
        self, pos: PositionState, qty: Decimal, gross: Decimal, fees: Decimal, when: datetime
    ) -> Optional[Tuple[Decimal, Decimal]]:
        """
        Remove shares at the current average cost.

        Args:
            pos: Position to reduce.
            qty: Shares removed (positive).
            gross: Gross proceeds for all `qty` shares.
            fees: Commission + taxes for all `qty` shares (positive).
            when: Event time.

        Returns:
            (realized_excl, realized_incl) for the shares that were held, or
            None when nothing was held (short sale).
        """
        if pos.is_short or pos.shares <= 0:
            self._warn(when, pos.symbol, "short", f"sold {qty} shares with none held; average not computed")
            pos.shares -= qty
            pos.is_short = True
            return None

        held = min(qty, pos.shares)
        fraction_of_position = held / pos.shares
        fraction_of_event = held / qty
        cost_excl_sold = pos.cost_excl * fraction_of_position
        cost_incl_sold = pos.cost_incl * fraction_of_position
        gross_held = gross * fraction_of_event
        fees_held = fees * fraction_of_event
        realized_excl = gross_held - cost_excl_sold
        realized_incl = gross_held - fees_held - cost_incl_sold

        pos.cost_excl -= cost_excl_sold
        pos.cost_incl -= cost_incl_sold
        pos.shares -= held
        pos.cycle_realized_excl += realized_excl
        pos.cycle_realized_incl += realized_incl
        pos.cycle_commissions += fees_held

        if pos.shares == 0:
            self._close_cycle(pos, when.date())
        if qty > held:
            self._warn(when, pos.symbol, "short", f"sold {qty} shares but only {held} held; remainder treated as short")
            pos.shares = -(qty - held)
            pos.is_short = True
        return realized_excl, realized_incl

    # ----------------------------------------------------------------- events

    def apply_trade(self, ex: Execution) -> None:
        """Apply one execution and record its annotated row."""
        pos = self._position(ex.conid, ex.symbol, ex.currency, ex.isin)
        pos.symbol = ex.symbol
        pos.isin = ex.isin or pos.isin
        when = ex.trade_datetime
        if ex.currency != pos.currency:
            self._warn(when, ex.symbol, "currency", f"trade in {ex.currency} on a {pos.currency} position")
        if ex.ib_commission_currency != ex.currency:
            self._warn(when, ex.symbol, "commission_currency", f"commission in {ex.ib_commission_currency}, trade in {ex.currency}; not converted")

        fees = abs(ex.ib_commission) + abs(ex.taxes)
        gross = abs(ex.proceeds) if ex.proceeds else abs(ex.quantity * ex.trade_price)
        realized: Optional[Tuple[Decimal, Decimal]] = None
        if ex.quantity > 0:
            self._add_shares(pos, ex.quantity, gross, gross + fees, when.date())
            pos.cycle_commissions += fees
        elif ex.quantity < 0:
            realized = self._remove_shares(pos, -ex.quantity, gross, fees, when)
        else:
            self._warn(when, ex.symbol, "zero_quantity", f"trade {ex.trade_id} has quantity 0; ignored")

        for bucket in self._totals(pos.position_key, ex.trade_date):
            bucket.executions += 1
            bucket.commissions += fees
            if ex.quantity > 0:
                bucket.shares_bought += ex.quantity
            else:
                bucket.shares_sold += -ex.quantity
            if realized is not None:
                bucket.realized_excl += realized[0]
                bucket.realized_incl += realized[1]

        self.result.execution_rows.append(
            ExecutionRow(
                execution=ex,
                position_key=pos.position_key,
                cycle_no=pos.cycle_no,
                shares_after=pos.shares,
                wac_excl_after=pos.wac_excl,
                wac_incl_after=pos.wac_incl,
                realized_excl=realized[0] if realized else None,
                realized_incl=realized[1] if realized else None,
            )
        )

    def _merge_into(self, primary: str, other: str, when: datetime) -> None:
        """Move all shares and cost of position `other` into `primary` and alias it."""
        if other == primary or other not in self.positions:
            self.alias[other] = primary
            return
        src = self.positions.pop(other)
        dst = self.positions[primary]
        if src.shares != 0 and dst.shares != 0:
            self._warn(when, dst.symbol, "merge", f"merged holding {src.symbol} into {dst.symbol}; averages combined")
        dst.shares += src.shares
        dst.cost_excl += src.cost_excl
        dst.cost_incl += src.cost_incl
        self.alias[other] = primary

    def apply_corporate_action(self, legs: List[CorporateActionLeg]) -> None:
        """Apply all legs of one corporate action together."""
        first = legs[0]
        when = min(leg.action_datetime for leg in legs)
        removed = [leg for leg in legs if leg.quantity < 0]
        added = [leg for leg in legs if leg.quantity > 0]
        proceeds = sum((leg.proceeds for leg in legs), _ZERO)

        involved: List[str] = []
        for leg in removed + added:
            key = self._resolve(leg.conid)
            if key in self.positions and key not in involved:
                involved.append(key)
        if involved:
            primary = involved[0]
        else:
            source = added[0] if added else first
            primary = self._position(source.conid, source.symbol, source.currency, source.isin).position_key
        for key in involved[1:]:
            self._merge_into(primary, key, when)
        for leg in legs:
            if self._resolve(leg.conid) != primary:
                self.alias[leg.conid] = primary

        pos = self.positions[primary]
        if added:
            pos.symbol = added[-1].symbol
            pos.isin = added[-1].isin or pos.isin
        shares_before = pos.shares
        wac_before = pos.wac_incl
        removed_qty = -sum((leg.quantity for leg in removed), _ZERO)
        added_qty = sum((leg.quantity for leg in added), _ZERO)
        realized: Optional[Tuple[Decimal, Decimal]] = None

        if first.action_type not in SPLIT_LIKE_TYPES:
            self._warn(when, pos.symbol, "corporate_action_type", f"type {first.action_type!r} applied generically, please verify: {first.action_description}")

        if not added:
            # Shares removed with no replacement: cash merger / tender / worthless — a sale at `proceeds`.
            if proceeds == 0:
                self._warn(when, pos.symbol, "removed_without_proceeds", f"{removed_qty} shares removed with no proceeds (worthless/delisted?)")
            realized = self._remove_shares(pos, removed_qty, abs(proceeds), _ZERO, when)
        elif not removed:
            # Shares received without giving any up: stock dividend / spin-off — zero cost basis.
            self._warn(when, pos.symbol, "zero_cost_shares", f"{added_qty} shares received at zero cost basis (spin-off/stock dividend?)")
            self._add_shares(pos, added_qty, _ZERO, _ZERO, when.date())
        else:
            # Conversion (split, ticker/ISIN change): quantity changes, total cost carried unchanged.
            pos.shares += added_qty - removed_qty
            if pos.shares == 0 and shares_before != 0:
                self._close_cycle(pos, when.date())
            if proceeds != 0:
                self._warn(when, pos.symbol, "cash_in_lieu", f"cash {proceeds} received with the action; booked as realized gain at zero cost")
                realized = (proceeds, proceeds)
                pos.cycle_realized_excl += proceeds
                pos.cycle_realized_incl += proceeds

        if realized is not None:
            for bucket in self._totals(primary, when.date()):
                bucket.realized_excl += realized[0]
                bucket.realized_incl += realized[1]

        self.result.corporate_action_rows.append(
            CorporateActionRow(
                action_datetime=when,
                position_key=primary,
                symbol=pos.symbol,
                action_type=first.action_type,
                description=first.action_description,
                shares_before=shares_before,
                shares_after=pos.shares,
                wac_incl_before=wac_before,
                wac_incl_after=pos.wac_incl,
                proceeds=proceeds,
                realized_incl=realized[1] if realized else None,
            )
        )

    def apply_transfer(self, tr: TransferEvent) -> None:
        """Apply a position transfer; internal transfers between own accounts are no-ops."""
        if tr.transfer_type.upper() == "INTERNAL" and tr.counterparty_account in self.own_accounts:
            return
        when = tr.transfer_datetime
        pos = self._position(tr.conid, tr.symbol, tr.currency, tr.isin)
        if tr.quantity > 0:
            cost = abs(tr.position_amount) if tr.position_amount else abs(tr.quantity * (tr.transfer_price or _ZERO))
            self._warn(when, tr.symbol, "external_transfer", f"{tr.quantity} shares transferred in at cost {cost}; verify cost basis")
            self._add_shares(pos, tr.quantity, cost, cost, when.date())
        elif tr.quantity < 0:
            self._warn(when, tr.symbol, "external_transfer", f"{-tr.quantity} shares transferred out; removed at average cost, no P&L")
            if not pos.is_short and pos.shares > 0:
                qty = min(-tr.quantity, pos.shares)
                fraction = qty / pos.shares
                pos.cost_excl -= pos.cost_excl * fraction
                pos.cost_incl -= pos.cost_incl * fraction
                pos.shares -= qty
                if pos.shares == 0:
                    self._close_cycle(pos, when.date())

    def apply_cash_transaction(self, cash: CashTransaction) -> None:
        """Dispatch a cash transaction by type."""
        if cash.cash_type == CASH_DEPOSITS_WITHDRAWALS:
            self.apply_deposit(cash)
        elif cash.cash_type == CASH_COMMISSION_ADJUSTMENT:
            self.apply_commission_adjustment(cash)
        elif cash.cash_type in DIVIDEND_CASH_TYPES:
            self.apply_dividend(cash)

    def apply_deposit(self, cash: CashTransaction) -> None:
        """
        Add a deposit (positive) / withdrawal (negative) to the per-currency totals.

        Never touches positions. Booking corrections IBKR files under the same
        type are ignored. Cash moved between the user's own accounts appears as
        a withdrawal plus a deposit and nets to zero.
        """
        if not cash.is_deposit:
            _logger.debug("Ignoring non-deposit %s row %s: %s", cash.cash_type, cash.transaction_id, cash.description)
            return
        day = cash.cash_datetime.date()
        totals = self.result.deposits.setdefault(cash.currency, DepositTotals(currency=cash.currency))
        totals.lifetime += cash.amount
        if totals.first_date is None or day < totals.first_date:
            totals.first_date = day
        if self.ytd_start <= day:
            totals.ytd += cash.amount
        if self.period_start <= day:
            totals.month += cash.amount
            totals.count_month += 1
        self.result.deposit_rows.append(cash)

    def apply_dividend(self, cash: CashTransaction) -> None:
        """Add a dividend / payment in lieu / withholding tax to the ticker's net dividends."""
        pos = self._position(cash.conid, cash.symbol, cash.currency, cash.isin)
        if cash.currency != pos.currency:
            self._warn(cash.cash_datetime, cash.symbol, "currency", f"{cash.cash_type} in {cash.currency} on a {pos.currency} position")
        for bucket in self._totals(pos.position_key, cash.cash_datetime.date()):
            bucket.dividends += cash.amount

    def apply_commission_adjustment(self, adj: CashTransaction) -> None:
        """
        Apply a commission refund (positive amount) or extra charge (negative).

        A refund on a buy whose shares are still held lowers `cost_incl` (so the
        incl.-commission average drops); otherwise it is realized P&L.
        """
        pos = self._position(adj.conid, adj.symbol, adj.currency, adj.isin)
        is_buy = self.trade_side.get((adj.account_id, adj.trade_id or ""))
        if is_buy is None:
            self._warn(adj.cash_datetime, adj.symbol, "commission_adjustment", f"adjustment {adj.amount} for unknown trade {adj.trade_id}; booked as realized")
        held = pos.shares > 0 and not pos.is_short
        buckets = self._totals(pos.position_key, adj.cash_datetime.date())
        for bucket in buckets:
            bucket.commissions -= adj.amount
        if held:
            pos.cycle_commissions -= adj.amount
        if is_buy and held:
            pos.cost_incl -= adj.amount
            return
        # Refund on a sell, or the position was already closed: the money is realized now.
        # A closed cycle's record is not reopened; the amount counts in the period totals.
        if held:
            pos.cycle_realized_incl += adj.amount
        for bucket in buckets:
            bucket.realized_incl += adj.amount

    # ---------------------------------------------------------------- results

    def run(self) -> LedgerResult:
        """Replay every event and assemble the result."""
        for event in _build_events(self.events, self.period_end):
            payload = event.payload
            if isinstance(payload, Execution):
                self.apply_trade(payload)
            elif isinstance(payload, TransferEvent):
                self.apply_transfer(payload)
            elif isinstance(payload, CashTransaction):
                self.apply_cash_transaction(payload)
            else:
                self.apply_corporate_action(payload)

        result = self.result
        result.positions = self.positions
        conids = {e.conid for e in self.events.executions} | {c.conid for c in self.events.corporate_actions}
        conids |= {t.conid for t in self.events.transfers} | {p.conid for p in self.events.open_positions}
        result.conid_to_key = {conid: self._resolve(conid) for conid in conids}
        result.summaries = self._summaries()
        result.month_totals = dict(self.month)
        result.ytd_totals = dict(self.ytd)
        result.lifetime_totals = dict(self.lifetime)
        return result

    def _summaries(self) -> List[PositionSummary]:
        """Report lines: positions held at period end or active in the period, by symbol."""
        lines: List[PositionSummary] = []
        for key, pos in self.positions.items():
            month = self.month.get(key, PeriodTotals())
            active = month.executions > 0 or month.realized_incl != 0
            if pos.shares == 0 and not active:
                continue
            lines.append(
                PositionSummary(
                    position_key=key,
                    symbol=pos.symbol,
                    currency=pos.currency,
                    isin=pos.isin,
                    shares=pos.shares,
                    wac_excl=pos.wac_excl,
                    wac_incl=pos.wac_incl,
                    cycle_no=pos.cycle_no,
                    cycle_open_date=pos.cycle_open_date if pos.shares != 0 else None,
                    month=month,
                    ytd=self.ytd.get(key, PeriodTotals()),
                )
            )
        lines.sort(key=lambda line: line.symbol)
        return lines


def build_ledger(events: LedgerEvents, period_start: date, period_end: date) -> LedgerResult:
    """
    Replay all events up to `period_end` and compute the ledger.

    Args:
        events: Every stored event (any order; sorted internally).
        period_start: First day of the report period (month totals start here).
        period_end: Last day of the report period, inclusive; later events are ignored.
            Year-to-date totals run from 1 January of this year.

    Returns:
        LedgerResult with positions at `period_end`, annotated executions,
        corporate actions, closed cycles, per-position summaries and warnings.

    Raises:
        ValueError: If `period_start` is after `period_end`.
    """
    if period_start > period_end:
        raise ValueError(f"period_start {period_start} is after period_end {period_end}")
    return _Engine(events, period_start, period_end).run()
