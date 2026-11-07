"""
Signal Ledger — paper book simulation per signal stream and exit policy.

Rules (user decision 2026-10-21):
- start with ``START_CAPITAL_USD`` fully in VT;
- every signal buys ``POSITION_USD`` at the next session's open, funded by selling VT
  at that open, best rank first, while fewer than ``MAX_POSITIONS`` are open;
- a ticker already held in the stream is not bought twice (a repeat signal only
  refreshes the "last signalled" session the decay exit watches);
- the exit policy (``exits.py``) decides when a position is sold; proceeds go back into VT
  at the price the sale happened (open or close);
- half the round-trip cost is charged on each side;
- long-only: ``short`` signals are skipped (no borrow model);
- signals with a ``data_flag`` (suspected bad prices) are skipped.

Order inside a session: exits at the open (native/decay exits, stops gapped through),
then entries, then intraday stop/target hits (a session that touches both counts as
the stop), then time exits at the close. A held ticker without a price on a day is
marked at its last close and, if it never trades again, exits at that price.
High/low default to the open/close range where missing.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.ml.pipeline.signal_ledger.config import (
    ATR_PERIOD,
    DECAY_SESSIONS,
    DEFAULT_HOLD_SESSIONS,
    JUMP_RATIO,
    MAX_POSITIONS,
    POSITION_USD,
    START_CAPITAL_USD,
    TRAIL_FALLBACK_PCT,
    cost_bps,
)
from src.ml.pipeline.signal_ledger.evaluator import aligned_array
from src.ml.pipeline.signal_ledger.exits import (
    DEFAULT_POLICY,
    EXIT_DATA,
    EXIT_DECAY,
    EXIT_NATIVE,
    EXIT_OPEN,
    EXIT_STOP,
    EXIT_TARGET,
    EXIT_TIME,
    EXIT_TRAIL,
    ExitPolicy,
)

BOOK_BENCHMARK = "VT"


@dataclass
class BookResult:
    """Daily NAV rows and per-trade rows of one simulated book."""

    nav: List[Dict[str, Any]] = field(default_factory=list)
    trades: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class _Position:
    signal: Mapping[str, Any]
    col: int
    shares: float
    entry_pos: int
    entry_price: float
    last_exit_pos: int
    stop: Optional[float]
    target: Optional[float]
    trail_width: Optional[float]
    native_exit_pos: Optional[int]
    last_seen: int
    hwm: float
    low: float
    high: float

    def stop_level(self) -> Tuple[Optional[float], str]:
        """Active protective level (the higher of fixed stop and trailing stop) and its exit reason."""
        trail = self.hwm - self.trail_width if self.trail_width is not None else None
        if trail is not None and (self.stop is None or trail >= self.stop):
            return trail, EXIT_TRAIL
        return self.stop, EXIT_STOP


class _Panels:
    """Numpy views of the price panels, aligned to the close panel's columns."""

    def __init__(self, opens: pd.DataFrame, closes: pd.DataFrame, highs: Optional[pd.DataFrame], lows: Optional[pd.DataFrame]) -> None:
        self.columns = {ticker: i for i, ticker in enumerate(closes.columns)}
        self.open = aligned_array(opens, closes)
        self.close = aligned_array(closes, closes)
        self.marks = pd.DataFrame(self.close).ffill().to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            body_high, body_low = np.fmax(self.open, self.close), np.fmin(self.open, self.close)
            high, low = aligned_array(highs, closes), aligned_array(lows, closes)
            self.high = np.where(high > 0, high, body_high)
            self.low = np.where(low > 0, low, body_low)

    def atr(self, col: int, position: int) -> Optional[float]:
        """Average true range of the ``ATR_PERIOD`` sessions before ``position`` (None without data)."""
        start = max(position - ATR_PERIOD, 1)
        if start >= position:
            return None
        high, low = self.high[start:position, col], self.low[start:position, col]
        previous = self.marks[start - 1 : position - 1, col]
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN sessions (no prices yet) are expected
            ranges = np.nanmax(np.stack([high - low, np.abs(high - previous), np.abs(low - previous)]), axis=0)
        ranges = ranges[np.isfinite(ranges)]
        return float(ranges.mean()) if ranges.size else None



def _finite(value: float) -> bool:
    return bool(np.isfinite(value)) and value > 0


def simulate_book(
    source: str,
    signals: Sequence[Mapping[str, Any]],
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    highs: Optional[pd.DataFrame] = None,
    lows: Optional[pd.DataFrame] = None,
    policy: ExitPolicy = DEFAULT_POLICY,
    start_capital: float = START_CAPITAL_USD,
    position_usd: float = POSITION_USD,
    max_positions: int = MAX_POSITIONS,
) -> BookResult:
    """
    Simulate the paper book of one stream under one exit policy.

    Args:
        source: Stream source (cost estimate and decay window).
        signals: Ledger rows with ``signal_date``, ``ticker``, ``side``, ``rank``, ``horizon_days``,
            optional ``id``, ``data_flag`` and ``meta`` (``exit_signal_date`` for native exits).
        opens: Session-indexed opens incl. VT.
        closes: Session-indexed closes incl. VT (no unfinished bar), same index as ``opens``.
        highs: Session-indexed highs (optional).
        lows: Session-indexed lows (optional).
        policy: Exit policy.
        start_capital: Initial capital (all in VT).
        position_usd: Dollars per new position.
        max_positions: Concurrent position cap.

    Returns:
        ``BookResult``: daily NAV rows (``date``, ``nav``, ``benchmark_nav``, ``invested``, ``positions``,
        ``opened``, ``closed``) and one trade row per position (closed or still open).
    """
    if BOOK_BENCHMARK not in closes.columns or not signals:
        return BookResult()
    sessions = closes.index
    entries = _entries_by_position(signals, sessions)
    if not entries:
        return BookResult()
    sim = _Simulation(source, sessions, _Panels(opens, closes, highs, lows), policy, position_usd, max_positions)
    return sim.run(entries, start_capital)


def simulate_uncapped(
    source: str,
    signals: Sequence[Mapping[str, Any]],
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    highs: Optional[pd.DataFrame] = None,
    lows: Optional[pd.DataFrame] = None,
    policy: ExitPolicy = DEFAULT_POLICY,
) -> BookResult:
    """
    Trade every signal of a stream under one exit policy, with no position cap and no cash limit.

    The capped book measures a realistic portfolio, but with long holds its slots fill within days
    and later signals are never taken, so exit policies are compared on this uncapped run (a ticker
    already held is still not bought twice). Its NAV is not meaningful; use the trades.

    Args:
        source: Stream source.
        signals: Ledger rows (see ``simulate_book``).
        opens: Session-indexed opens incl. VT.
        closes: Session-indexed closes incl. VT.
        highs: Session-indexed highs (optional).
        lows: Session-indexed lows (optional).
        policy: Exit policy.

    Returns:
        ``BookResult`` whose ``trades`` cover every tradable signal.
    """
    return simulate_book(source, signals, opens, closes, highs, lows, policy=policy,
                         start_capital=POSITION_USD * (len(signals) + 1), max_positions=len(signals) + 1)


class _Simulation:
    """State of one book replay."""

    def __init__(self, source: str, sessions: pd.Index, panels: _Panels, policy: ExitPolicy, position_usd: float, max_positions: int) -> None:
        self.source = source
        self.sessions = sessions
        self.px = panels
        self.policy = policy
        self.position_usd = position_usd
        self.max_positions = max_positions
        self.half_cost = cost_bps(source) / 20_000.0
        self.decay = DECAY_SESSIONS.get(source) if policy.native else None
        vt = panels.columns[BOOK_BENCHMARK]
        self.vt_open, self.vt_close = panels.open[:, vt], panels.close[:, vt]
        self.vt_units = 0.0
        self.held: Dict[str, _Position] = {}
        self.trades: List[Dict[str, Any]] = []

    def run(self, entries: Dict[int, List[Mapping[str, Any]]], start_capital: float) -> BookResult:
        """Replay every session from the first entry to the end."""
        first = min(entries)
        self.vt_units = start_capital / float(self.vt_open[first])
        benchmark_units = self.vt_units
        nav_rows: List[Dict[str, Any]] = []
        for position in range(first, len(self.sessions)):
            # A repeat signal is known before the open, so it keeps a held ticker from decaying today.
            for signal in entries.get(position, []):
                if signal["ticker"] in self.held:
                    self.held[signal["ticker"]].last_seen = position
            closed = self._exits_at_open(position)
            opened = self._enter(position, entries.get(position, []))
            closed += self._exits_in_session(position)
            invested = sum(p.shares * self._mark(p.col, position) for p in self.held.values())
            vt_close = float(self.vt_close[position])
            nav_rows.append({
                "date": self.sessions[position].date(),
                "nav": self.vt_units * vt_close + invested,
                "benchmark_nav": benchmark_units * vt_close,
                "invested": invested,
                "positions": len(self.held),
                "opened": opened,
                "closed": closed,
            })
        last = len(self.sessions) - 1
        for ticker in sorted(self.held):
            self.trades.append(self._trade(self.held[ticker], None, self._mark(self.held[ticker].col, last), EXIT_OPEN, last))
        return BookResult(nav=nav_rows, trades=self.trades)

    def _mark(self, col: int, position: int) -> float:
        value = self.px.marks[position, col]
        return float(value) if np.isfinite(value) else 0.0

    def _exits_at_open(self, position: int) -> int:
        """Native/decay exits and stops/targets gapped through at the open (proceeds into VT at the open)."""
        closed = 0
        for ticker in list(self.held):
            pos = self.held[ticker]
            price = self.px.open[position, pos.col]
            if position <= pos.entry_pos or not _finite(price):
                continue
            reason = None
            level, level_reason = pos.stop_level()
            if pos.native_exit_pos is not None and position >= pos.native_exit_pos:
                reason = EXIT_NATIVE
            elif self.decay is not None and position >= pos.last_seen + self.decay:
                reason = EXIT_DECAY
            elif level is not None and price <= level:
                reason = level_reason
            elif pos.target is not None and price >= pos.target:
                reason = EXIT_TARGET
            if reason:
                self._close(ticker, float(price), reason, position, float(self.vt_open[position]))
                closed += 1
        return closed

    def _enter(self, position: int, signals: List[Mapping[str, Any]]) -> int:
        """Buy the day's signals at the open, best rank first."""
        opened = 0
        for signal in signals:
            ticker = signal["ticker"]
            if ticker in self.held:
                self.held[ticker].last_seen = position
                continue
            col = self.px.columns.get(ticker)
            if col is None or len(self.held) >= self.max_positions:
                continue
            price = self.px.open[position, col]
            vt_open = float(self.vt_open[position])
            if not _finite(price) or self.vt_units * vt_open < self.position_usd:
                continue
            self.vt_units -= self.position_usd / vt_open
            self.held[ticker] = self._open_position(signal, col, position, float(price))
            opened += 1
        return opened

    def _open_position(self, signal: Mapping[str, Any], col: int, position: int, price: float) -> _Position:
        policy = self.policy
        hold = policy.max_hold or int(signal.get("horizon_days") or DEFAULT_HOLD_SESSIONS)
        trail_width = None
        if policy.trail_atr is not None:
            atr = self.px.atr(col, position)
            trail_width = policy.trail_atr * atr if atr else TRAIL_FALLBACK_PCT * price
        return _Position(
            signal=signal, col=col, shares=self.position_usd * (1 - self.half_cost) / price, entry_pos=position,
            entry_price=price, last_exit_pos=position + hold - 1,
            stop=price * (1 - policy.stop_pct) if policy.stop_pct is not None else None,
            target=price * (1 + policy.target_pct) if policy.target_pct is not None else None,
            trail_width=trail_width, native_exit_pos=self._native_exit_pos(signal, position),
            last_seen=position, hwm=price, low=price, high=price,
        )

    def _native_exit_pos(self, signal: Mapping[str, Any], entry_pos: int) -> Optional[int]:
        """Session whose open executes the pipeline's own exit signal (after the entry), if any."""
        if not self.policy.native:
            return None
        raw = (signal.get("meta") or {}).get("exit_signal_date")
        if not raw:
            return None
        exit_signal = pd.Timestamp(raw)
        return max(int(self.sessions.searchsorted(exit_signal, side="right")), entry_pos + 1)

    def _exits_in_session(self, position: int) -> int:
        """Data breaks, intraday stop/target hits and time exits at the close (proceeds into VT at the close)."""
        closed = 0
        vt_close = float(self.vt_close[position])
        for ticker in list(self.held):
            pos = self.held[ticker]
            o, h, low, c = (float(a[position, pos.col]) for a in (self.px.open, self.px.high, self.px.low, self.px.close))
            previous = self._mark(pos.col, position - 1) if position > pos.entry_pos else pos.entry_price
            if position > pos.entry_pos and _jumped(previous, (o, c)):
                self._close(ticker, previous, EXIT_DATA, position, vt_close)
                closed += 1
                continue
            if _finite(low):
                pos.low = min(pos.low, low)
            if _finite(h):
                pos.high = max(pos.high, h)
            level, level_reason = pos.stop_level()
            if level is not None and _finite(low) and low <= level:
                self._close(ticker, level, level_reason, position, vt_close)
            elif pos.target is not None and _finite(h) and h >= pos.target:
                self._close(ticker, pos.target, EXIT_TARGET, position, vt_close)
            elif position >= pos.last_exit_pos:
                self._close(ticker, self._mark(pos.col, position), EXIT_TIME, position, vt_close)
            else:
                if _finite(c):
                    pos.hwm = max(pos.hwm, c)
                continue
            closed += 1
        return closed

    def _close(self, ticker: str, price: float, reason: str, position: int, vt_price: float) -> None:
        pos = self.held.pop(ticker)
        self.vt_units += pos.shares * price * (1 - self.half_cost) / vt_price
        self.trades.append(self._trade(pos, self.sessions[position].date(), price, reason, position))

    def _trade(self, pos: _Position, exit_date: Optional[date], price: float, reason: str, position: int) -> Dict[str, Any]:
        signal = pos.signal
        net = pos.shares * price * (1 - self.half_cost) / self.position_usd - 1.0
        return {
            "signal_id": signal.get("id"),
            "ticker": signal["ticker"],
            "signal_date": signal["signal_date"],
            "entry_date": self.sessions[pos.entry_pos].date(),
            "entry_price": pos.entry_price,
            "exit_date": exit_date,
            "exit_price": price if exit_date is not None else None,
            "exit_reason": reason,
            "sessions_held": position - pos.entry_pos + 1,
            "net_return": net if math.isfinite(net) else None,
            "last_price": price,
            "mae": pos.low / pos.entry_price - 1.0,
            "mfe": pos.high / pos.entry_price - 1.0,
        }


def _jumped(previous: float, prices: Tuple[float, ...]) -> bool:
    """True if any price moved ``JUMP_RATIO`` or more versus the previous close."""
    if not _finite(previous):
        return False
    return any(_finite(p) and (p / previous >= JUMP_RATIO or p / previous <= 1.0 / JUMP_RATIO) for p in prices)


def _entries_by_position(signals: Sequence[Mapping[str, Any]], sessions: pd.Index) -> Dict[int, List[Mapping[str, Any]]]:
    """Map entry-session position -> long signals entering there, best rank first."""
    entries: Dict[int, List[Mapping[str, Any]]] = {}
    for signal in signals:
        if signal.get("side") == "short" or signal.get("data_flag"):
            continue
        signal_date: date = signal["signal_date"]
        position = int(sessions.searchsorted(pd.Timestamp(signal_date), side="right"))
        if position < len(sessions):
            entries.setdefault(position, []).append(signal)
    for day_signals in entries.values():
        day_signals.sort(key=lambda s: (s.get("rank") is None, s.get("rank") or 0))
    return entries


def summarize_book(nav_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    Headline numbers for a simulated book.

    Args:
        nav_rows: Daily NAV rows of one book (time-ordered).

    Returns:
        ``total_return``, ``benchmark_return``, ``excess_return``, ``max_drawdown``,
        ``avg_invested_pct``, ``days`` (empty dict when there are no rows).
    """
    if not nav_rows:
        return {}
    nav = np.array([float(r["nav"]) for r in nav_rows])
    bench = np.array([float(r["benchmark_nav"]) for r in nav_rows])
    invested_pct = np.array([float(r["invested"]) / nav_value if nav_value else 0.0 for r, nav_value in zip(nav_rows, nav)])
    total, bench_total = float(nav[-1] / nav[0] - 1), float(bench[-1] / bench[0] - 1)
    return {
        "start_date": nav_rows[0]["date"],
        "end_date": nav_rows[-1]["date"],
        "days": len(nav_rows),
        "nav": float(nav[-1]),
        "total_return": total,
        "benchmark_return": bench_total,
        "excess_return": total - bench_total,
        "max_drawdown": float((nav / np.maximum.accumulate(nav) - 1).min()),
        "avg_invested_pct": float(invested_pct.mean()),
        "positions": int(nav_rows[-1]["positions"]),
    }


def summarize_trades(trades: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    Trade statistics of one book: closed-trade returns, holding time, MAE/MFE and exit reasons.

    Args:
        trades: Trade rows of one book (``BookResult.trades``).

    Returns:
        ``trades``, ``closed``, ``open``, ``hit_rate``, ``mean_return``, ``median_return``, ``avg_sessions``,
        ``median_mae``, ``median_mfe``, ``exit_reasons`` (stats over closed trades; None without any).
    """
    closed = [t for t in trades if t["exit_reason"] != EXIT_OPEN and t.get("net_return") is not None]
    returns = np.array([float(t["net_return"]) for t in closed])
    reasons: Dict[str, int] = {}
    for trade in closed:
        reasons[trade["exit_reason"]] = reasons.get(trade["exit_reason"], 0) + 1

    def _stat(values: np.ndarray, fn: Any) -> Optional[float]:
        return float(fn(values)) if values.size else None

    return {
        "trades": len(trades),
        "closed": len(closed),
        "open": len(trades) - len(closed),
        "hit_rate": _stat(returns > 0, np.mean),
        "mean_return": _stat(returns, np.mean),
        "median_return": _stat(returns, np.median),
        "avg_sessions": _stat(np.array([float(t["sessions_held"]) for t in closed]), np.mean),
        "median_mae": _stat(np.array([float(t["mae"]) for t in closed]), np.median),
        "median_mfe": _stat(np.array([float(t["mfe"]) for t in closed]), np.median),
        "exit_reasons": reasons,
    }
