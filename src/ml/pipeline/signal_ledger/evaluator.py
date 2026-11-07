"""
Signal Ledger — forward returns of each signal.

Entry = open of the first session strictly after ``signal_date``; exit = close
``h`` sessions after entry (entry session counts as session 1). Benchmarks are
measured over the identical open-to-close window. Returns are the raw price
return of the ticker; the side (long/short) is applied by the scorecard.
MAE/MFE are the lowest low and highest high over the first ``CORE_HORIZON``
sessions relative to the entry price.

The panels are converted to numpy arrays once per batch (``PricePanels``): with
signals pending for up to 250 sessions, the nightly job evaluates every signal
of the last year.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.ml.pipeline.signal_ledger.config import BENCHMARKS, CORE_HORIZON, HORIZONS, JUMP_RATIO, NO_DATA_AFTER_DAYS

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
# Maximum adverse / favourable excursion over the core window.
MAE_COLUMN = return_column("mae", CORE_HORIZON)
MFE_COLUMN = return_column("mfe", CORE_HORIZON)
EXCURSION_COLUMNS: List[str] = [MAE_COLUMN, MFE_COLUMN]


def aligned_array(frame: Optional[pd.DataFrame], like: pd.DataFrame) -> np.ndarray:
    """
    Return a frame aligned to the index and columns of ``like`` as a float array.

    Args:
        frame: Price frame, or None.
        like: Frame whose index/columns define the shape.

    Returns:
        2-D float array (NaN where the frame has no value or is None).
    """
    if frame is None:
        return np.full(like.shape, np.nan)
    return frame.reindex(index=like.index, columns=like.columns).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


class PricePanels:
    """Open/high/low/close as arrays aligned to the close panel (sessions x tickers)."""

    def __init__(self, opens: pd.DataFrame, closes: pd.DataFrame, highs: Optional[pd.DataFrame] = None,
                 lows: Optional[pd.DataFrame] = None) -> None:
        self.sessions = closes.index
        self.columns = {ticker: i for i, ticker in enumerate(closes.columns)}
        self.open = aligned_array(opens, closes)
        self.close = aligned_array(closes, closes)
        self.high = aligned_array(highs, closes)
        self.low = aligned_array(lows, closes)


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
    highs: Optional[pd.DataFrame] = None,
    lows: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Compute entry, forward returns and the core-window MAE/MFE of one signal.

    A suspected bad price (a ``JUMP_RATIO`` move) inside the first ``CORE_HORIZON`` sessions flags the
    whole signal (``data_flag``); one later only voids the longer horizons whose window contains it.

    Args:
        signal: Row with ``signal_date`` and ``ticker``.
        opens: Session-indexed open prices (one column per ticker, benchmarks included).
        closes: Session-indexed close prices, same index as ``opens``; must exclude today's unfinished bar.
        today: Run date, used to decide when missing data becomes ``no_data``.
        highs: Session-indexed highs (optional; the open/close range is used where missing).
        lows: Session-indexed lows (optional; the open/close range is used where missing).

    Returns:
        Outcome fields: ``status``, ``entry_date``, ``entry_price``, ``data_flag``, every
        ``OUTCOME_COLUMNS`` and ``EXCURSION_COLUMNS`` entry (None if not yet known).
    """
    return _evaluate(signal, PricePanels(opens, closes, highs, lows), today)


def _evaluate(signal: Mapping[str, Any], px: PricePanels, today: date) -> Dict[str, Any]:
    """Evaluate one signal on prepared arrays (see ``evaluate_signal``)."""
    outcome: Dict[str, Any] = {"status": STATUS_PENDING, "entry_date": None, "entry_price": None, "data_flag": None}
    outcome.update({column: None for column in (*OUTCOME_COLUMNS, *EXCURSION_COLUMNS)})

    sessions = px.sessions
    position = int(sessions.searchsorted(pd.Timestamp(signal["signal_date"]), side="right"))
    if position >= len(sessions):
        return outcome  # entry session has not happened yet

    entry_session = sessions[position]
    col = px.columns.get(signal["ticker"])
    entry = _finite(px.open[position, col]) if col is not None else None
    if col is None or entry is None:
        if today - entry_session.date() > timedelta(days=NO_DATA_AFTER_DAYS):
            outcome["status"] = STATUS_NO_DATA
        return outcome

    outcome["entry_date"] = entry_session.date()
    outcome["entry_price"] = entry
    opens, closes = px.open[:, col], px.close[:, col]
    jump = _first_jump(opens, closes, position, position + max(HORIZONS) - 1)
    complete = True
    for horizon in HORIZONS:
        exit_position = position + horizon - 1
        if exit_position >= len(sessions):
            complete = False
            continue
        exit_price = _last_close(closes, position, exit_position)
        if exit_price is not None and (jump is None or jump > exit_position):
            outcome[return_column("ret", horizon)] = exit_price / entry - 1.0
        for benchmark in BENCHMARKS:
            bench_col = px.columns.get(benchmark)
            if bench_col is None:
                continue
            bench_open = _finite(px.open[position, bench_col])
            bench_close = _finite(px.close[exit_position, bench_col])
            if bench_open and bench_close:
                outcome[return_column(benchmark.lower(), horizon)] = bench_close / bench_open - 1.0
    if complete:
        outcome["status"] = STATUS_COMPLETE
    core_end = position + CORE_HORIZON - 1
    if jump is not None and jump <= core_end:
        outcome["data_flag"] = FLAG_PRICE_JUMP
    elif core_end < len(sessions):
        low, high = _excursion(px, col, position, core_end)
        if low is not None and high is not None:
            outcome[MAE_COLUMN] = low / entry - 1.0
            outcome[MFE_COLUMN] = high / entry - 1.0
    return outcome


def _first_jump(opens: np.ndarray, closes: np.ndarray, entry_position: int, last_position: int) -> Optional[int]:
    """
    Position of the first session from entry to last that moves by ``JUMP_RATIO`` or more vs the previous close.

    Both the overnight gap (open / previous close) and the session move (close / previous close) count;
    missing closes carry the last known close forward.
    """
    last_position = min(last_position, len(closes) - 1)
    start = max(entry_position - 1, 0)
    window_closes = _ffill(closes[start : last_position + 1])
    window_opens = opens[start : last_position + 1]
    previous = np.concatenate(([np.nan], window_closes[:-1]))
    with np.errstate(invalid="ignore", divide="ignore"):
        ratios = np.stack([window_closes / previous, window_opens / previous])
        hit = ((ratios > 0) & ((ratios >= JUMP_RATIO) | (ratios <= 1.0 / JUMP_RATIO))).any(axis=0)
    for offset in np.flatnonzero(hit):
        if start + offset >= entry_position:
            return int(start + offset)
    return None


def _ffill(values: np.ndarray) -> np.ndarray:
    """Carry the last finite value forward (leading NaNs stay NaN)."""
    valid = np.isfinite(values)
    index = np.where(valid, np.arange(len(values)), 0)
    np.maximum.accumulate(index, out=index)
    filled = values[index]
    filled[~valid & (np.cumsum(valid) == 0)] = np.nan
    return filled


def _excursion(px: PricePanels, col: int, start: int, end: int) -> Tuple[Optional[float], Optional[float]]:
    """Lowest low and highest high from session ``start`` to ``end`` (open/close range where high/low is missing)."""
    window = slice(start, end + 1)
    o, c = px.open[window, col], px.close[window, col]
    lo, hi = px.low[window, col], px.high[window, col]
    with np.errstate(invalid="ignore"):
        low = np.where(lo > 0, lo, np.fmin(o, c))
        high = np.where(hi > 0, hi, np.fmax(o, c))
        low, high = low[low > 0], high[high > 0]
    if not low.size or not high.size:
        return None, None
    return float(low.min()), float(high.max())


def _last_close(closes: np.ndarray, entry_position: int, exit_position: int) -> Optional[float]:
    """
    Close at the exit session, or the last close before it inside the holding window.

    A ticker that stops trading (halt, delisting, merger) keeps its last traded
    price instead of dropping out — dropping losers would bias the scorecard up.
    """
    window = closes[entry_position : exit_position + 1]
    with np.errstate(invalid="ignore"):
        window = window[np.isfinite(window) & (window > 0)]
    return float(window[-1]) if window.size else None


def evaluate_signals(
    signals: Sequence[Mapping[str, Any]],
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    today: date,
    highs: Optional[pd.DataFrame] = None,
    lows: Optional[pd.DataFrame] = None,
) -> List[Dict[str, Any]]:
    """
    Evaluate many signals; each result carries the signal's ``id``.

    Args:
        signals: Ledger rows with ``id``, ``signal_date`` and ``ticker``.
        opens: Session-indexed opens.
        closes: Session-indexed closes (no unfinished bar).
        today: Run date.
        highs: Session-indexed highs (optional).
        lows: Session-indexed lows (optional).

    Returns:
        One outcome dict per signal, including ``id``.
    """
    px = PricePanels(opens, closes, highs, lows)
    return [{"id": signal["id"], **_evaluate(signal, px, today)} for signal in signals]
