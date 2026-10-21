"""
P21 Momentum — live Tracks B-E (docs/pipeline-specification.md §9), replayed from inception.

Track A is the real paper sleeve kept in ``_state/current_positions.json``. Tracks
B-E are pure functions of the archived monthly ``targets.json`` files, the regime
history and daily prices, so instead of keeping four more state files they are
replayed from the first execution every time they are needed (daily mark, monthly
report). Replaying is deterministic and backfills history automatically.

Mechanics match ``backtest/p21_momentum/runner.py``:
- B: the same 20 names sized with ``regime_scalar = 1`` (``targets_b``), same fills,
  same catastrophic stop (flag at the close, sell at the next open, like live Track A);
- C: MTUM at ``SLEEVE_TARGET_PCT`` of track NAV, rebalanced on each execution date;
- D: as C, scaled by that month's applied regime scalar;
- C and D lose the TER adjustment daily (``TER_ADJUSTMENT_ANNUAL / 252`` of their shares);
- E: 100 % SPY bought at the first execution's open.
Every track starts at ``NAV_TOTAL_USD`` on the first execution date.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import pandas as pd

from src.ml.pipeline.p21_momentum.calendar import first_trading_day_of_month
from src.ml.pipeline.p21_momentum.config import (
    CATASTROPHIC_STOP_PCT,
    MIN_TRADE_USD,
    NAV_TOTAL_USD,
    PROXY_TICKER,
    SLEEVE_TARGET_PCT,
    TER_ADJUSTMENT_ANNUAL,
)
from src.ml.pipeline.p21_momentum.execution.fills import TradeIntent, apply_chatter_threshold, execute_trades, simulate_fill

MARKET_TICKER = "SPY"
TRACKS = ("B", "C", "D", "E")
_TER_DAILY = TER_ADJUSTMENT_ANNUAL / 252


@dataclass(frozen=True)
class TrackExecution:
    """One monthly execution as seen by Tracks B-E."""

    execution_date: date
    targets_b: Dict[str, float]  # ticker -> target USD with regime_scalar = 1
    scalar: float  # applied regime scalar of the signal month (Track D)


@dataclass
class _StockPosition:
    shares: float
    avg_cost: float


@dataclass
class _StockTrack:
    cash: float
    holdings: Dict[str, _StockPosition] = field(default_factory=dict)
    pending_stops: List[str] = field(default_factory=list)


def _next_month_first_trading_day(as_of: date) -> date:
    """First NYSE trading day of the month after ``as_of`` (the execution date of a signal)."""
    year, month = (as_of.year + 1, 1) if as_of.month == 12 else (as_of.year, as_of.month + 1)
    return first_trading_day_of_month(year, month)


def load_executions(results_dir: Path, regime_history: List[Dict]) -> List[TrackExecution]:
    """
    Build the execution list from every dated ``targets.json``.

    ``targets_b`` is read when present (written by monthly_rebalance since 2026-10); older
    files only hold Track A's sizing, which is converted back by dividing by the scalar
    (exact when the scalar was 1, which holds for every pre-2026-10 month).

    Args:
        results_dir: ``results/p21_momentum``.
        regime_history: Entries of ``_state/regime_history.json``.

    Returns:
        Executions ordered by date.
    """
    scalar_by_date = {h["date"]: float(h.get("scalar_applied", 1.0)) for h in regime_history}
    executions: List[TrackExecution] = []
    for path in sorted(results_dir.glob("????-??-??/targets.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        as_of = date.fromisoformat(payload.get("as_of", path.parent.name))
        scalar = scalar_by_date.get(as_of.isoformat(), 1.0)
        targets_b = payload.get("targets_b")
        if targets_b is None:
            divisor = scalar if scalar > 0 else 1.0
            targets_b = {t["ticker"]: float(t["target_usd"]) / divisor for t in payload.get("targets", [])}
        executions.append(TrackExecution(_next_month_first_trading_day(as_of), {k: float(v) for k, v in targets_b.items()}, scalar))
    return sorted(executions, key=lambda e: e.execution_date)


def _price(frame: pd.DataFrame, ticker: str, ts: pd.Timestamp) -> Optional[float]:
    """Positive price of ticker at ts, else None."""
    if ticker not in frame.columns:
        return None
    value = pd.to_numeric(frame.at[ts, ticker], errors="coerce")
    number = float(value) if pd.notna(value) else 0.0
    return number if number > 0 else None


def _rebalance_stocks(track: _StockTrack, targets: Dict[str, float], opens: pd.DataFrame, ts: pd.Timestamp) -> None:
    """Rebalance Track B to its targets at the open (same intent/fill logic as live Track A)."""
    open_prices = {t: p for t in set(targets) | set(track.holdings) if (p := _price(opens, t, ts)) is not None}
    intents: List[TradeIntent] = []
    for ticker, pos in track.holdings.items():
        if ticker not in targets:
            value = pos.shares * open_prices.get(ticker, pos.avg_cost)
            intents.append(TradeIntent(ticker, "SELL", pos.shares, value, 0.0))
    for ticker, target_usd in targets.items():
        price = open_prices.get(ticker)
        if price is None:
            continue
        current = track.holdings[ticker].shares if ticker in track.holdings else 0.0
        delta = target_usd / price - current
        if abs(delta) > 1e-9:
            side: Literal["BUY", "SELL"] = "BUY" if delta > 0 else "SELL"
            intents.append(TradeIntent(ticker, side, abs(delta), current * price, target_usd))
    outcome = execute_trades(apply_chatter_threshold(intents), open_prices, available_cash=track.cash)
    for trade in outcome.trades:
        held = track.holdings.get(trade.ticker)
        if trade.side == "BUY":
            track.cash -= trade.net_usd
            if held is None:
                track.holdings[trade.ticker] = _StockPosition(trade.shares, trade.fill_price)
            else:
                total = held.shares + trade.shares
                held.avg_cost = (held.avg_cost * held.shares + trade.fill_price * trade.shares) / total
                held.shares = total
        else:
            track.cash += trade.net_usd
            if held is not None:
                held.shares -= trade.shares
                if held.shares <= 1e-9:
                    del track.holdings[trade.ticker]


def _execute_stops(track: _StockTrack, opens: pd.DataFrame, ts: pd.Timestamp) -> None:
    """Sell Track B positions flagged at the previous close."""
    remaining = []
    for ticker in track.pending_stops:
        pos = track.holdings.get(ticker)
        price = _price(opens, ticker, ts)
        if pos is None:
            continue
        if price is None:
            remaining.append(ticker)
            continue
        fill, comm = simulate_fill("SELL", pos.shares, price)
        track.cash += fill * pos.shares - comm
        del track.holdings[ticker]
    track.pending_stops = remaining


def _flag_stops(track: _StockTrack, closes: pd.DataFrame, ts: pd.Timestamp) -> None:
    """Flag positions whose close fell below the catastrophic stop level."""
    for ticker, pos in track.holdings.items():
        price = _price(closes, ticker, ts)
        if price is not None and price < pos.avg_cost * (1 + CATASTROPHIC_STOP_PCT) and ticker not in track.pending_stops:
            track.pending_stops.append(ticker)


def _rebalance_single(shares: float, cash: float, target_usd: float, price: float) -> Tuple[float, float]:
    """Rebalance a one-instrument sleeve (MTUM) to target_usd at price."""
    delta_usd = target_usd - shares * price
    if abs(delta_usd) < MIN_TRADE_USD:
        return shares, cash
    side: Literal["BUY", "SELL"] = "BUY" if delta_usd > 0 else "SELL"
    delta_shares = abs(delta_usd) / price
    fill, comm = simulate_fill(side, delta_shares, price)
    if side == "BUY":
        return shares + delta_shares, cash - fill * delta_shares - comm
    return shares - delta_shares, cash + fill * delta_shares - comm


def replay_tracks(
    executions: List[TrackExecution],
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    initial_nav: float = NAV_TOTAL_USD,
) -> pd.DataFrame:
    """
    Replay Tracks B-E from the first execution date.

    Args:
        executions: Monthly executions, ordered by date.
        opens: Session-indexed (normalized dates) adjusted opens incl. MTUM and SPY.
        closes: Same index, adjusted closes.
        initial_nav: Starting NAV of every track.

    Returns:
        Frame indexed by session date with columns ``nav_b`` .. ``nav_e`` (empty before the first execution).
    """
    columns = [f"nav_{t.lower()}" for t in TRACKS]
    if not executions:
        return pd.DataFrame(columns=columns)
    by_date = {pd.Timestamp(e.execution_date): e for e in executions}
    first = pd.Timestamp(executions[0].execution_date)
    track_b = _StockTrack(cash=initial_nav)
    mtum = {"C": [0.0, initial_nav], "D": [0.0, initial_nav]}  # [shares, cash]
    spy_shares = 0.0
    rows = []
    marks = closes.ffill()

    for ts in closes.index[closes.index >= first]:
        _execute_stops(track_b, opens, ts)
        execution = by_date.get(ts)
        if execution is not None:
            _rebalance_stocks(track_b, execution.targets_b, opens, ts)
            mtum_open = _price(opens, PROXY_TICKER, ts)
            if mtum_open is not None:
                for name, scalar in (("C", 1.0), ("D", execution.scalar)):
                    shares, cash = mtum[name]
                    target = (cash + shares * mtum_open) * SLEEVE_TARGET_PCT * scalar
                    mtum[name] = list(_rebalance_single(shares, cash, target, mtum_open))
            spy_open = _price(opens, MARKET_TICKER, ts)
            if spy_shares == 0.0 and spy_open is not None:
                spy_shares = initial_nav / spy_open
        _flag_stops(track_b, closes, ts)
        for name in ("C", "D"):
            mtum[name][0] *= 1 - _TER_DAILY

        mtum_close = _price(marks, PROXY_TICKER, ts) or 0.0
        spy_close = _price(marks, MARKET_TICKER, ts)
        stock_value = sum(pos.shares * (_price(marks, t, ts) or pos.avg_cost) for t, pos in track_b.holdings.items())
        rows.append({
            "date": ts,
            "nav_b": track_b.cash + stock_value,
            "nav_c": mtum["C"][1] + mtum["C"][0] * mtum_close,
            "nav_d": mtum["D"][1] + mtum["D"][0] * mtum_close,
            "nav_e": spy_shares * spy_close if spy_shares and spy_close else initial_nav,
        })
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).set_index("date")[columns]


def track_tickers(executions: List[TrackExecution]) -> List[str]:
    """Every ticker Tracks B-E can hold, plus MTUM and SPY."""
    tickers = {t for e in executions for t in e.targets_b} | {PROXY_TICKER, MARKET_TICKER}
    return sorted(tickers)


def wide_frames(panel: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Turn a ``fetch_price_panel`` result into session-indexed open/close frames.

    Sessions are SPY's trading days; rows where only some other symbol traded are dropped
    (the same gap that broke the 200-day average in the regime check).

    Args:
        panel: ``{ticker: frame with timestamp/open/close}``.

    Returns:
        ``(opens, closes)`` indexed by normalized session date.
    """
    opens: Dict[str, pd.Series] = {}
    closes: Dict[str, pd.Series] = {}
    for ticker, df in panel.items():
        if df is None or df.empty:
            continue
        frame = df.copy()
        stamps = pd.to_datetime(frame["timestamp"])
        if stamps.dt.tz is not None:
            stamps = stamps.dt.tz_localize(None)
        frame.index = pd.DatetimeIndex(stamps).normalize()
        frame = frame[~frame.index.duplicated(keep="last")]
        opens[ticker] = pd.to_numeric(frame["open"], errors="coerce")
        closes[ticker] = pd.to_numeric(frame["close"], errors="coerce")
    open_frame, close_frame = pd.DataFrame(opens), pd.DataFrame(closes)
    if MARKET_TICKER in close_frame.columns:
        sessions = close_frame.index[close_frame[MARKET_TICKER].notna()]
        open_frame, close_frame = open_frame.reindex(sessions), close_frame.reindex(sessions)
    return open_frame.sort_index(), close_frame.sort_index()


def summarize_navs(navs: pd.DataFrame, as_of: date) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Month / YTD / since-inception returns and max drawdown per track for the monthly report.

    Args:
        navs: Frame indexed by date with columns nav_a..nav_e (e.g. _state/nav_daily.csv).
        as_of: Report date; rows after it are ignored.

    Returns:
        ``(cum_returns, max_drawdown)``: ``{track: {"month", "ytd", "since_inception"}}`` and ``{track: drawdown}``.
    """
    frame = navs[navs.index <= pd.Timestamp(as_of)].sort_index()
    cum: Dict[str, Dict[str, float]] = {}
    drawdown: Dict[str, float] = {}
    if frame.empty:
        return cum, drawdown
    month_start = pd.Timestamp(as_of.replace(day=1))
    year_start = pd.Timestamp(as_of.replace(month=1, day=1))
    for track in ("A", "B", "C", "D", "E"):
        series = pd.to_numeric(frame[f"nav_{track.lower()}"], errors="coerce").dropna()
        if series.empty:
            continue
        now = float(series.iloc[-1])
        cum[track] = {
            "month": now / _base_before(series, month_start) - 1.0,
            "ytd": now / _base_before(series, year_start) - 1.0,
            "since_inception": now / float(series.iloc[0]) - 1.0,
        }
        drawdown[track] = float((series / series.cummax() - 1.0).min())
    return cum, drawdown


def _base_before(series: pd.Series, before: pd.Timestamp) -> float:
    """Last value before a date (the period's starting NAV), else the first value."""
    prior = series[series.index < before]
    return float(prior.iloc[-1]) if not prior.empty else float(series.iloc[0])
