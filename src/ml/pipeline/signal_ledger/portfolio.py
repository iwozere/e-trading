"""
Signal Ledger — paper book simulation per signal stream.

Rules (user decision 2026-10-21):
- start with ``START_CAPITAL_USD`` fully in VT;
- every signal buys ``POSITION_USD`` at the next session's open, funded by selling VT
  at that open, best rank first, while fewer than ``MAX_POSITIONS`` are open;
- a ticker already held in the stream is not bought twice;
- a position is sold at the close ``horizon_days`` sessions after entry and the
  proceeds go back into VT at that close;
- half the round-trip cost is charged on each side;
- long-only: ``short`` signals are skipped (no borrow model);
- signals with a ``data_flag`` (suspected bad prices) are skipped.

A held ticker without a price on a day is marked at its last close, and exits
at that price if it never trades again.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.ml.pipeline.signal_ledger.config import (
    DEFAULT_HOLD_SESSIONS,
    MAX_POSITIONS,
    POSITION_USD,
    START_CAPITAL_USD,
    cost_bps,
)

BOOK_BENCHMARK = "VT"


@dataclass
class _Position:
    ticker: str
    shares: float
    exit_position: int


def simulate_book(
    source: str,
    signals: Sequence[Mapping[str, Any]],
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    start_capital: float = START_CAPITAL_USD,
    position_usd: float = POSITION_USD,
    max_positions: int = MAX_POSITIONS,
) -> List[Dict[str, Any]]:
    """
    Simulate the paper book of one stream and return its daily NAV.

    Args:
        source: Stream source (for the cost estimate).
        signals: Ledger rows with ``signal_date``, ``ticker``, ``side``, ``rank``, ``horizon_days``.
        opens: Session-indexed opens incl. VT.
        closes: Session-indexed closes incl. VT (no unfinished bar), same index as ``opens``.
        start_capital: Initial capital (all in VT).
        position_usd: Dollars per new position.
        max_positions: Concurrent position cap.

    Returns:
        Daily rows: ``date``, ``nav``, ``benchmark_nav``, ``invested``, ``positions``, ``opened``, ``closed``.
    """
    if BOOK_BENCHMARK not in closes.columns or not signals:
        return []
    sessions = closes.index
    half_cost = cost_bps(source) / 20_000.0
    marks = closes.ffill()
    entries = _entries_by_position(signals, sessions)
    if not entries:
        return []

    first = min(entries)
    vt_open, vt_close = opens[BOOK_BENCHMARK], closes[BOOK_BENCHMARK]
    vt_units = start_capital / float(vt_open.iloc[first])
    benchmark_units = vt_units
    held: Dict[str, _Position] = {}
    nav_rows: List[Dict[str, Any]] = []

    for position in range(first, len(sessions)):
        opened = closed = 0
        for signal in entries.get(position, []):
            ticker = signal["ticker"]
            if len(held) >= max_positions or ticker in held or ticker not in opens.columns:
                continue
            price = _price(opens[ticker], position)
            if price is None or vt_units * float(vt_open.iloc[position]) < position_usd:
                continue
            vt_units -= position_usd / float(vt_open.iloc[position])
            hold = int(signal.get("horizon_days") or DEFAULT_HOLD_SESSIONS)
            held[ticker] = _Position(ticker, position_usd * (1 - half_cost) / price, position + hold - 1)
            opened += 1

        for ticker in [t for t, p in held.items() if p.exit_position <= position]:
            proceeds = held.pop(ticker).shares * _mark(marks, ticker, position) * (1 - half_cost)
            vt_units += proceeds / float(vt_close.iloc[position])
            closed += 1

        invested = sum(p.shares * _mark(marks, p.ticker, position) for p in held.values())
        nav_rows.append({
            "date": sessions[position].date(),
            "nav": vt_units * float(vt_close.iloc[position]) + invested,
            "benchmark_nav": benchmark_units * float(vt_close.iloc[position]),
            "invested": invested,
            "positions": len(held),
            "opened": opened,
            "closed": closed,
        })
    return nav_rows


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


def _price(series: pd.Series, position: int) -> Optional[float]:
    """Positive price at a session, else None."""
    value = series.iloc[position]
    return float(value) if pd.notna(value) and value > 0 else None


def _mark(marks: pd.DataFrame, ticker: str, position: int) -> float:
    """Last known close of a held ticker (0 if it never had one)."""
    value = marks[ticker].iloc[position]
    return float(value) if pd.notna(value) else 0.0


def summarize_book(nav_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    Headline numbers for a simulated book.

    Args:
        nav_rows: Output of ``simulate_book`` (time-ordered).

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
