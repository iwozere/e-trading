"""
Signal Ledger — forward returns of each signal.

Entry = open of the first session strictly after ``signal_date``; exit = close
``h`` sessions after entry (entry session counts as session 1). Benchmarks are
measured over the identical open-to-close window. Returns are the raw price
return of the ticker; the side (long/short) is applied by the scorecard.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from src.ml.pipeline.signal_ledger.config import BENCHMARKS, HORIZONS, JUMP_RATIO, NO_DATA_AFTER_DAYS

STATUS_PENDING = "pending"
STATUS_COMPLETE = "complete"
STATUS_NO_DATA = "no_data"
FLAG_PRICE_JUMP = "price_jump"


def return_column(prefix: str, horizon: int) -> str:
    """Ledger column name for a return, e.g. ``ret_20d`` or ``vt_5d``."""
    return f"{prefix}_{horizon}d"


OUTCOME_COLUMNS: List[str] = [
    return_column(prefix, h) for prefix in ("ret", *(b.lower() for b in BENCHMARKS)) for h in HORIZONS
]


def _finite(value: Any) -> Optional[float]:
    """Return value as a positive finite float, else None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def evaluate_signal(
    signal: Mapping[str, Any],
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    today: date,
) -> Dict[str, Any]:
    """
    Compute entry and forward returns for one signal.

    Args:
        signal: Row with ``signal_date`` and ``ticker``.
        opens: Session-indexed open prices (one column per ticker, benchmarks included).
        closes: Session-indexed close prices, same index as ``opens``; must exclude today's unfinished bar.
        today: Run date, used to decide when missing data becomes ``no_data``.

    Returns:
        Outcome fields: ``status``, ``entry_date``, ``entry_price``, ``data_flag`` and every
        ``OUTCOME_COLUMNS`` entry (None if not yet known).
    """
    outcome: Dict[str, Any] = {"status": STATUS_PENDING, "entry_date": None, "entry_price": None, "data_flag": None}
    outcome.update({column: None for column in OUTCOME_COLUMNS})

    signal_date: date = signal["signal_date"]
    ticker: str = signal["ticker"]
    sessions = closes.index
    position = int(sessions.searchsorted(pd.Timestamp(signal_date), side="right"))
    if position >= len(sessions):
        return outcome  # entry session has not happened yet

    entry_session = sessions[position]
    entry = _finite(opens[ticker].iloc[position]) if ticker in opens.columns else None
    if entry is None:
        if today - entry_session.date() > timedelta(days=NO_DATA_AFTER_DAYS):
            outcome["status"] = STATUS_NO_DATA
        return outcome

    outcome["entry_date"] = entry_session.date()
    outcome["entry_price"] = entry
    complete = True
    for horizon in HORIZONS:
        exit_position = position + horizon - 1
        if exit_position >= len(sessions):
            complete = False
            continue
        exit_price = _last_close(closes[ticker], position, exit_position)
        if exit_price is not None:
            outcome[return_column("ret", horizon)] = exit_price / entry - 1.0
        for benchmark in BENCHMARKS:
            if benchmark not in closes.columns:
                continue
            bench_open = _finite(opens[benchmark].iloc[position])
            bench_close = _finite(closes[benchmark].iloc[exit_position])
            if bench_open and bench_close:
                outcome[return_column(benchmark.lower(), horizon)] = bench_close / bench_open - 1.0
    if complete:
        outcome["status"] = STATUS_COMPLETE
    last = min(position + max(HORIZONS) - 1, len(sessions) - 1)
    if _has_price_jump(opens[ticker], closes[ticker], position, last):
        outcome["data_flag"] = FLAG_PRICE_JUMP
    return outcome


def _has_price_jump(opens: pd.Series, closes: pd.Series, entry_position: int, last_position: int) -> bool:
    """
    True if any session from entry to last moves by ``JUMP_RATIO`` or more versus the previous close.

    Both the overnight gap (open / previous close) and the session move (close / previous close) count.
    """
    start = max(entry_position - 1, 0)
    window_closes = pd.to_numeric(closes.iloc[start : last_position + 1], errors="coerce").ffill()
    window_opens = pd.to_numeric(opens.iloc[start : last_position + 1], errors="coerce")
    previous = window_closes.shift(1)
    ratios = pd.concat([window_closes / previous, window_opens / previous]).dropna()
    ratios = ratios[ratios > 0]
    return bool(((ratios >= JUMP_RATIO) | (ratios <= 1.0 / JUMP_RATIO)).any())


def _last_close(series: pd.Series, entry_position: int, exit_position: int) -> Optional[float]:
    """
    Close at the exit session, or the last close before it inside the holding window.

    A ticker that stops trading (halt, delisting, merger) keeps its last traded
    price instead of dropping out — dropping losers would bias the scorecard up.
    """
    window = pd.to_numeric(series.iloc[entry_position : exit_position + 1], errors="coerce").dropna()
    window = window[window > 0]
    return float(window.iloc[-1]) if not window.empty else None


def evaluate_signals(
    signals: Sequence[Mapping[str, Any]],
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    today: date,
) -> List[Dict[str, Any]]:
    """
    Evaluate many signals; each result carries the signal's ``id``.

    Args:
        signals: Ledger rows with ``id``, ``signal_date`` and ``ticker``.
        opens: Session-indexed opens.
        closes: Session-indexed closes (no unfinished bar).
        today: Run date.

    Returns:
        One outcome dict per signal, including ``id``.
    """
    return [{"id": signal["id"], **evaluate_signal(signal, opens, closes, today)} for signal in signals]
