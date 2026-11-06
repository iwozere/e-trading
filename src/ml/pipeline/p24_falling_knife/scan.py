"""
P24 Falling Knife — live scan of one security (spec §9).

Runs the same detector as the backtest on the latest bars. A **signal** is a detector signal on the last completed
bar of the session being scanned; a **forming** setup is a last-bar evaluation that passed the drop rule but misses at
most ``forming_max_missing`` conditions (spec §4.8 watchlist). Entry is the next session's open.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.ml.pipeline.p24_falling_knife.backtest import price_bucket
from src.ml.pipeline.p24_falling_knife.config import TimeframeParams, UniverseParams
from src.ml.pipeline.p24_falling_knife.data import series_for_timeframe
from src.ml.pipeline.p24_falling_knife.detector import BarEval, SeriesContext, build_context, detect
from src.ml.pipeline.p24_falling_knife.indicators import compute_indicators


@dataclass
class SymbolScan:
    """Scan result of one security on one timeframe."""

    symbol: str
    timeframe: str
    ctx: SeriesContext
    bars: pd.DataFrame
    signals: List[BarEval]
    evals: List[BarEval]
    signal: Optional[BarEval] = None
    forming: Optional[BarEval] = None


def build_series(symbol: str, daily: pd.DataFrame, timeframe: str, params: TimeframeParams, universe: UniverseParams,
                 session: date) -> Optional[SymbolScan]:
    """
    Build the context and every causal signal of one security up to ``session``.

    Args:
        symbol: Ticker.
        daily: P24 daily bars (``live.prepare_live_daily``).
        timeframe: ``"daily"`` or ``"weekly"``.
        params: Live parameters of the timeframe.
        universe: Universe filters.
        session: Last session the data is complete up to.

    Returns:
        ``SymbolScan`` without the last-bar classification, or None if the history is too short.
    """
    bars = series_for_timeframe(daily, timeframe, universe, session)
    if len(bars) < params.min_history_bars + 2:
        return None
    frame = compute_indicators(bars, params)
    ctx = build_context(frame, params)
    signals, evals = detect(ctx, params, collect_evals=True)
    return SymbolScan(symbol=symbol, timeframe=timeframe, ctx=ctx, bars=frame, signals=signals, evals=evals)


def scan_symbol(symbol: str, daily: pd.DataFrame, timeframe: str, params: TimeframeParams, universe: UniverseParams,
                session: date, forming_max_missing: int) -> Optional[SymbolScan]:
    """
    Scan one security: signal or forming setup on the bar of ``session``.

    Args:
        symbol: Ticker.
        daily: P24 daily bars.
        timeframe: ``"daily"`` or ``"weekly"``.
        params: Live parameters of the timeframe.
        universe: Universe filters.
        session: Session being scanned (daily: that day; weekly: the week's last session).
        forming_max_missing: Forming watchlist threshold.

    Returns:
        ``SymbolScan`` (``signal`` / ``forming`` set when found), or None if there is no bar for ``session``.
    """
    scan = build_series(symbol, daily, timeframe, params, universe, session)
    if scan is None or scan.ctx.index[-1].date() != session:
        return None
    last = scan.ctx.n - 1
    scan.signal = next((s for s in scan.signals if s.t == last), None)
    if scan.signal is None:
        ev = next((e for e in scan.evals if e.t == last), None)
        if ev is not None and ev.missing and len(ev.missing) <= forming_max_missing:
            scan.forming = ev
    return scan


def _day(ctx: SeriesContext, bar: Optional[int]) -> Optional[date]:
    return None if bar is None else ctx.index[bar].date()


def _num(value: float, digits: int = 4) -> Optional[float]:
    return round(float(value), digits) if np.isfinite(value) else None


def signal_row(scan: SymbolScan, ev: BarEval) -> Dict[str, Any]:
    """
    One ``signals_<tf>.csv`` / ``forming_<tf>.csv`` row (spec §9.2).

    Args:
        scan: Scan of the security.
        ev: The signal (or forming evaluation).

    Returns:
        Row dict.
    """
    ctx = scan.ctx
    close = float(ctx.close[ev.t])
    dollar = scan.bars["dollar_volume_20"].iloc[ev.t] if "dollar_volume_20" in scan.bars.columns else np.nan
    return {
        "timeframe": scan.timeframe,
        "ticker": scan.symbol,
        "signal_date": _day(ctx, ev.t),
        "close": round(close, 4),
        "peak_date": _day(ctx, ev.p1),
        "peak_close": round(float(ctx.close[ev.p1]), 4),
        "drop_pct": _num(ev.drop),
        "l2_date": _day(ctx, ev.l2),
        "s3_date": _day(ctx, ev.s3),
        "adx_peak": _num(ev.adx_peak, 2),
        "ppo_trough": _num(ev.ppo_trough),
        "w_min": _num(ev.w_min),
        "cci_2": _num(ev.cci_2, 1),
        "cci_3": _num(ev.cci_3, 1),
        "score": _num(ev.score),
        "knife_seq": ev.knife_seq,
        "suspect_gap": ev.suspect_gap,
        "suggested_entry": "next open",
        "price_bucket": price_bucket(close * float(ctx.raw_scale[ev.t])),
        "dollar_volume_20": _num(float(dollar), 0) if pd.notna(dollar) else None,
        "missing": ",".join(ev.missing),
    }
