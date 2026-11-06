"""
P24 Falling Knife — paper positions (spec §9.4).

One position per live signal, fixed notional, no portfolio cap. A position is never advanced by hand: every run
re-finds its signal on the latest bars (the detector is causal, so the signal is still there) and replays
``simulate_trade`` from it. Status, fills and today's events therefore depend only on the data, which makes a re-run
on the same date idempotent.

Statuses: ``pending_entry`` (no bar after the signal yet), ``open``, ``exit_pending`` (an exit rule fired on the last
bar; fills at the next open), ``closed``, ``lost`` (the signal no longer reproduces, e.g. revised data).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.ml.pipeline.p24_falling_knife.config import CostModel, TimeframeParams
from src.ml.pipeline.p24_falling_knife.exits import END_OF_DATA, simulate_trade
from src.ml.pipeline.p24_falling_knife.scan import SymbolScan
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

PENDING_ENTRY = "pending_entry"
OPEN = "open"
EXIT_PENDING = "exit_pending"
CLOSED = "closed"
LOST = "lost"
ACTIVE = (PENDING_ENTRY, OPEN, EXIT_PENDING)


@dataclass
class PaperPosition:
    """One paper trade (all dates ISO strings in the JSON state)."""

    timeframe: str
    ticker: str
    signal_date: str
    signal_close: float
    notional: float
    status: str = PENDING_ENTRY
    entry_date: Optional[str] = None
    entry_price: Optional[float] = None
    exit_signal_date: Optional[str] = None
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    last_date: Optional[str] = None
    last_close: Optional[float] = None
    last_cci: Optional[float] = None
    bars_held: int = 0
    net_return: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        """Unique id: timeframe, ticker and signal date."""
        return f"{self.timeframe}:{self.ticker}:{self.signal_date}"


def load_state(path: Path) -> List[PaperPosition]:
    """
    Read the paper state (empty when the file does not exist).

    Args:
        path: ``_state/paper_positions.json``.

    Returns:
        Positions in file order.
    """
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [PaperPosition(**row) for row in payload.get("positions", [])]


def save_state(path: Path, positions: List[PaperPosition]) -> None:
    """
    Write the paper state atomically (temp file + replace).

    Args:
        path: ``_state/paper_positions.json``.
        positions: Every position (closed ones are kept as history).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"positions": [asdict(p) for p in positions]}, indent=2), encoding="utf-8")
    tmp.replace(path)


def add_signals(positions: List[PaperPosition], rows: List[Dict[str, Any]], notional: float) -> List[PaperPosition]:
    """
    Open a pending position for every new signal row (existing keys are left alone).

    Args:
        positions: Current state.
        rows: ``scan.signal_row`` dicts (suspect-gap signals are skipped).
        notional: Paper notional per position.

    Returns:
        The newly added positions.
    """
    known = {p.key for p in positions}
    added: List[PaperPosition] = []
    for row in rows:
        if row.get("suspect_gap"):
            continue
        pos = PaperPosition(timeframe=str(row["timeframe"]), ticker=str(row["ticker"]), signal_date=str(row["signal_date"]),
                            signal_close=float(row["close"]), notional=notional, meta={"score": row.get("score")})
        if pos.key not in known:
            positions.append(pos)
            known.add(pos.key)
            added.append(pos)
    return added


def update_position(pos: PaperPosition, scan: SymbolScan, params: TimeframeParams, costs: CostModel) -> PaperPosition:
    """
    Replay one active position on the latest bars.

    Args:
        pos: Position (``status`` in ``ACTIVE``).
        scan: Fresh ``SymbolScan`` of the position's security and timeframe.
        params: Live parameters of the timeframe.
        costs: Cost model.

    Returns:
        The same object, updated.
    """
    ctx = scan.ctx
    signal = next((s for s in scan.signals if ctx.index[s.t].date().isoformat() == pos.signal_date), None)
    if signal is None:
        _logger.warning("P24 paper: signal %s no longer reproduces; marking lost", pos.key)
        pos.status = LOST
        return pos
    last = ctx.n - 1
    pos.last_date = ctx.index[last].date().isoformat()
    pos.last_close = float(ctx.close[last])
    pos.last_cci = float(ctx.cci[last]) if np.isfinite(ctx.cci[last]) else None
    trade = simulate_trade(ctx, signal, params, costs, delisted=False)
    if trade is None:
        pos.status = PENDING_ENTRY
        return pos
    pos.entry_date = ctx.index[trade.entry_bar].date().isoformat()
    pos.entry_price = trade.entry_price
    pos.bars_held = trade.bars_held
    pos.net_return = trade.net_return
    if trade.exit_reason == END_OF_DATA:
        pos.status = OPEN
        pos.exit_signal_date = pos.exit_date = pos.exit_price = pos.exit_reason = None
        return pos
    pos.exit_reason = trade.exit_reason
    if trade.exit_at_open:
        pos.status = CLOSED
        pos.exit_signal_date = ctx.index[trade.exit_bar - 1].date().isoformat()
        pos.exit_date = ctx.index[trade.exit_bar].date().isoformat()
        pos.exit_price = trade.exit_price
    else:
        pos.status = EXIT_PENDING
        pos.exit_signal_date = ctx.index[trade.exit_bar].date().isoformat()
        pos.exit_date = pos.exit_price = None
    return pos


def events_on(positions: List[PaperPosition], timeframe: str, session: date) -> Dict[str, List[PaperPosition]]:
    """
    Today's paper events of a timeframe, derived from dates (so a re-run reports the same events).

    Args:
        positions: State after the update.
        timeframe: ``"daily"`` or ``"weekly"``.
        session: Session scanned.

    Returns:
        ``{"entered": [...], "exit_pending": [...], "exited": [...]}``.
    """
    day = session.isoformat()
    mine = [p for p in positions if p.timeframe == timeframe]
    return {
        "entered": [p for p in mine if p.entry_date == day],
        "exit_pending": [p for p in mine if p.status == EXIT_PENDING and p.exit_signal_date == day],
        "exited": [p for p in mine if p.status == CLOSED and p.exit_date == day],
    }


def position_rows(positions: List[PaperPosition]) -> List[Dict[str, Any]]:
    """Flat rows for ``paper_positions_<tf>.csv`` / ``paper_exits_<tf>.csv`` (meta dropped, P&L in dollars added)."""
    rows = []
    for p in positions:
        row = {k: v for k, v in asdict(p).items() if k != "meta"}
        row["pnl_usd"] = round(p.notional * p.net_return, 2) if p.net_return is not None else None
        rows.append(row)
    return rows
