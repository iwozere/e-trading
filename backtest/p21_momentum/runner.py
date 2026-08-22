"""
P21 Momentum backtest — core engine (docs/pipeline-specification.md §14).

Iterates NYSE trading days over a historical range, calling the exact same
``src.ml.pipeline.p21_momentum.strategy``/``execution`` modules the live
pipeline uses — this is not a separate reimplementation of the strategy
logic (docs/implementation-plan.md §1). Only the outer loop differs: here it
walks a frozen in-memory panel instead of calling ``YahooDataDownloader``.

**Universe: Option A** (docs/pipeline-specification.md §14.3, confirmed by
the user over the ~$50-70/mo Option B point-in-time alternative). The
universe passed in is the *current* S&P 500 membership applied to every
historical month — free, but survivorship-biased +1-3% p.a. on return
figures. Every report this module's output feeds (``phase0_report.py``)
carries the exact §14.3 banner verbatim. Because the universe is fixed for
the whole backtest under Option A, there is no month-to-month
index-membership churn to model — the "dropped from index" forced-exit case
from the live pipeline's ``jobs/run_monthly_rebalance.py`` never fires here;
only filter failures and delistings do.

**Single daily pass.** One loop over every NYSE trading day drives both the
monthly signal/select/execute cycle (on the two days each month it applies)
and the daily NAV mark (every day) against one shared, live piece of state
— not two passes where a "monthly" loop hands off to a "daily reconstruction"
loop. That two-pass shape was tried and dropped during development: tracks
C/D/E never carried a per-trade ledger entry, so a from-scratch replay of
"trades so far" could never recover their share counts. A single pass
mutating live state directly has no such reconstruction gap.

**Documented simplifications, not silent shortcuts:**
- **F4 (quality) and F6 (earnings blackout) are not evaluated.** F4 needs
  point-in-time fundamentals (the frozen panel only has OHLCV); applying
  *today's* fundamentals to 2005 would be a fresh look-ahead bias the spec's
  own bias table (§14.2) explicitly warns against. F6 needs an earnings
  calendar with 20+ years of history, which no free source here provides.
  Both are treated as always-pass, which is already F4's real-world
  "pass on missing data" behavior (§5) — this does not change the mechanical
  metrics (§14.8) the backtest exists to produce; it would only matter for
  return figures, which §14.1 says must not be trusted anyway.
- **Cash/sleeve interpretation for tracks A-E** (spec §9 does not fully
  specify this): tracks A/B allocate ``SLEEVE_TARGET_PCT`` of each track's
  *own* current NAV to the 20-stock portfolio (rebalanced monthly), with the
  remainder held as zero-yield cash — mirroring the live pipeline's own
  sizing model (§7). Tracks C/D apply the identical sleeve treatment to a
  single MTUM position. Track E (SPY) is 100% invested at all times — it is
  a "plain beta" anchor, not a diluted sleeve position (spec §9: "the entire
  momentum complex trails plain beta"). This interpretation is called out
  explicitly here so it can be corrected if wrong; it does not affect the
  §14.9 B1-B4 mechanical acceptance criteria, only the B5-B8 track-NAV ones.
- **Catastrophic stop executes same-day at close, not next-day open.** The
  live pipeline flags on close and executes at the next open (spec §11).
  This backtest exits at the triggering day's close plus slippage instead,
  to avoid a next-day execution queue in an already-long daily loop. This
  slightly understates the real exit price gap.
- **Tracks C/D/E do not get itemized LedgerEntry rows.** Only the stock
  sleeve (tracks A/B) writes to the trade ledger returned in
  ``BacktestResult.trades`` — C/D/E rebalancing is fully reflected in
  ``nav_daily`` (it drives the same live share/cash state everything else
  reads), just not itemized as trades, since no §14.8 required metric needs
  a C/D/E trade-level audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Literal, Optional, Set, Tuple

import pandas as pd

from backtest.p21_momentum.missing_data import (
    MAX_FORWARD_FILL_DAYS,
    detect_and_liquidate_delisting,
    forward_fill_gaps,
)
from src.ml.pipeline.p21_momentum.calendar import (
    first_trading_day_of_month,
    last_trading_day_of_month,
    trading_days,
)
from src.ml.pipeline.p21_momentum.config import (
    ANOMALY_DAILY_CHANGE_PCT,
    CATASTROPHIC_STOP_PCT,
    ENTRY_RANK,
    FALLBACK_POOL_RANK,
    HOLD_RANK,
    MAX_PER_SECTOR,
    MAX_POSITION_PCT,
    MIN_TRADE_USD,
    NAV_TOTAL_USD,
    SLEEVE_TARGET_PCT,
    TARGET_COUNT,
    TER_ADJUSTMENT_ANNUAL,
    VIX_THRESHOLD,
)
from src.ml.pipeline.p21_momentum.execution.fills import simulate_fill
from src.ml.pipeline.p21_momentum.schemas import LedgerEntry
from src.ml.pipeline.p21_momentum.strategy.filters import f1_history, f2_liquidity, f3_gap
from src.ml.pipeline.p21_momentum.strategy.regime import PriorRegimeState, compute_regime
from src.ml.pipeline.p21_momentum.strategy.selection import (
    CurrentHolding,
    RankedCandidate,
    rank_candidates,
    select_portfolio,
)
from src.ml.pipeline.p21_momentum.strategy.signal import compute_signal
from src.ml.pipeline.p21_momentum.strategy.sizing import size_positions

PROXY_TICKER = "MTUM"
MARKET_TICKER = "SPY"
REGIME_INDEX_TICKER = "^GSPC"
REGIME_VOL_TICKER = "^VIX"

_TER_DAILY = TER_ADJUSTMENT_ANNUAL / 252


@dataclass(slots=True)
class BacktestParams:
    """Overridable strategy parameters — spec §14.5's robustness grid dimensions, plus slippage."""

    lookback_start: int = 252
    skip_recent: int = 21
    entry_rank: int = ENTRY_RANK
    hold_rank: int = HOLD_RANK
    max_per_sector: int = MAX_PER_SECTOR
    vix_threshold: float = VIX_THRESHOLD
    slippage_bps: float = 3.0
    target_count: int = TARGET_COUNT
    fallback_pool_rank: int = FALLBACK_POOL_RANK
    initial_nav: float = NAV_TOTAL_USD
    sleeve_pct: float = SLEEVE_TARGET_PCT
    max_position_pct: float = MAX_POSITION_PCT


@dataclass(slots=True)
class _Position:
    shares: float
    avg_cost: float
    entry_date: date
    entry_rank: int


@dataclass(slots=True)
class MonthlyMetrics:
    """One month's mechanical metrics — spec §14.8's primary, bias-resistant output."""

    month: str  # YYYY-MM, keyed by signal date
    turnover_two_way_usd: float
    position_count: int
    sector_herfindahl: float
    f1_removed: int
    f2_removed: int
    f3_removed: int
    regime_scalar: float
    warn_underfilled: bool
    manual_review_count: int = 0
    exit_delisted_count: int = 0
    max_sector_count: int = 0  # largest single-sector count among `selected` — §14.9 B4 check


@dataclass(slots=True)
class BacktestResult:
    """Full output of one run_backtest() call."""

    nav_daily: pd.DataFrame  # index=date, columns=[nav_a, nav_b, nav_c, nav_d, nav_e]
    trades: List[LedgerEntry] = field(default_factory=list)
    monthly_metrics: List[MonthlyMetrics] = field(default_factory=list)
    regime_history: List[dict] = field(default_factory=list)


@dataclass(slots=True)
class _PendingTarget:
    """A rebalance decision made at a signal date, awaiting execution the next trading day."""

    execution_date: date
    month_key: str
    selected: List[RankedCandidate]
    ranked: List[RankedCandidate]
    alloc_a: Dict[str, float]
    alloc_b: Dict[str, float]
    exit_delisted_prices: Dict[str, float]
    scalar_applied: float


# ---------------------------------------------------------------------------
# Panel preparation
# ---------------------------------------------------------------------------


def _prepare_field_frame(
    panel: Dict[str, pd.DataFrame], tickers: List[str], calendar_index: pd.DatetimeIndex, field_name: str
) -> pd.DataFrame:
    cols = {}
    for t in tickers:
        df = panel.get(t)
        if df is None or df.empty or field_name not in df.columns:
            cols[t] = pd.Series(index=calendar_index, dtype=float)
            continue
        s = df.set_index("timestamp")[field_name]
        s.index = pd.to_datetime(s.index)
        s = s.reindex(calendar_index)
        cols[t] = forward_fill_gaps(s, max_days=MAX_FORWARD_FILL_DAYS).series
    return pd.DataFrame(cols, index=calendar_index)


def _month_range(start: date, end: date) -> List[Tuple[int, int]]:
    months = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def _signal_execution_pairs(start: date, end: date) -> List[Tuple[date, date, str]]:
    """One (signal_date, execution_date, month_key) triple per calendar month in range."""
    pairs = []
    for (y, m) in _month_range(start, end):
        try:
            signal_date = last_trading_day_of_month(y, m)
        except ValueError:
            continue
        if signal_date < start or signal_date > end:
            continue
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        try:
            execution_date = first_trading_day_of_month(ny, nm)
        except ValueError:
            continue
        if execution_date > end:
            continue
        pairs.append((signal_date, execution_date, f"{y:04d}-{m:02d}"))
    return pairs


def _herfindahl(sector_counts: Dict[Optional[str], int], total: int) -> float:
    if total == 0:
        return 0.0
    return sum((c / total) ** 2 for c in sector_counts.values())


def _price_at(frame: pd.DataFrame, ticker: str, ts: pd.Timestamp) -> Optional[float]:
    if ticker not in frame.columns or ts not in frame.index:
        return None
    val = frame.at[ts, ticker]
    return None if pd.isna(val) else float(val)  # type: ignore[arg-type]


def _sleeve_value(holdings: Dict[str, _Position], close_frame: pd.DataFrame, ts: pd.Timestamp) -> float:
    total = 0.0
    for ticker, pos in holdings.items():
        price = _price_at(close_frame, ticker, ts)
        total += pos.shares * (price if price is not None else pos.avg_cost)
    return total


def _ledger_entry(
    ts: date,
    track: str,
    ticker: str,
    side: str,
    shares: float,
    ref_open: float,
    fill_price: float,
    slippage_bps: float,
    commission_usd: float,
    gross_usd: float,
    net_usd: float,
    reason: str,
) -> LedgerEntry:
    return LedgerEntry(
        ts=ts.isoformat(),
        track=track,
        ticker=ticker,
        side=side,  # type: ignore[arg-type]
        shares=shares,
        ref_open=ref_open,
        fill_price=fill_price,
        slippage_bps=slippage_bps,
        commission_usd=commission_usd,
        gross_usd=gross_usd,
        net_usd=net_usd,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Stock-sleeve rebalancing (shared by tracks A and B)
# ---------------------------------------------------------------------------


def _rebalance_stock_sleeve(
    track: str,
    holdings: Dict[str, _Position],
    cash: float,
    selected: List[RankedCandidate],
    ranked: List[RankedCandidate],
    alloc: Dict[str, float],
    open_frame: pd.DataFrame,
    ts: pd.Timestamp,
    execution_date: date,
    slippage_bps: float,
    exit_delisted_prices: Dict[str, float],
    trades: List[LedgerEntry],
) -> Tuple[float, float]:
    """Rebalance one stock-sleeve track (A or B) to the target allocation, mutating holdings in place."""
    turnover = 0.0
    ranked_tickers = {c.ticker for c in ranked}
    target_tickers = {c.ticker for c in selected}

    def _log(
        ticker: str, side: str, shares: float, ref_open: float, fill: float,
        comm: float, gross: float, net: float, reason: str,
    ) -> None:
        entry = _ledger_entry(
            execution_date, track, ticker, side, shares, ref_open, fill, slippage_bps, comm, gross, net, reason
        )
        trades.append(entry)

    for ticker, liquidation_price in exit_delisted_prices.items():
        pos = holdings.pop(ticker, None)
        if pos is None:
            continue
        proceeds = pos.shares * liquidation_price
        cash += proceeds
        turnover += proceeds
        _log(ticker, "SELL", pos.shares, 0.0, liquidation_price, 0.0, proceeds, proceeds, "EXIT_DELISTED")

    for ticker in list(holdings.keys()):
        if ticker in target_tickers:
            continue
        pos = holdings.pop(ticker)
        price_f = _price_at(open_frame, ticker, ts)
        if price_f is None or price_f <= 0:
            continue
        fill, comm = simulate_fill("SELL", pos.shares, price_f, slippage_bps=slippage_bps)
        gross = fill * pos.shares
        net = gross - comm
        cash += net
        turnover += gross
        reason = "EXIT_RANK_DROP" if ticker in ranked_tickers else "EXIT_FILTER_FAIL"
        _log(ticker, "SELL", pos.shares, price_f, fill, comm, gross, net, reason)

    for c in selected:
        price_f = _price_at(open_frame, c.ticker, ts)
        if price_f is None or price_f <= 0:
            continue
        target_usd = alloc.get(c.ticker, 0.0)
        target_shares = target_usd / price_f
        existing = holdings.get(c.ticker)
        delta = target_shares - (existing.shares if existing else 0.0)
        if abs(delta * price_f) < MIN_TRADE_USD:
            continue
        side: Literal["BUY", "SELL"] = "BUY" if delta > 0 else "SELL"
        fill, comm = simulate_fill(side, abs(delta), price_f, slippage_bps=slippage_bps)
        gross = fill * abs(delta)
        if side == "BUY":
            net = gross + comm
            cash -= net
        else:
            net = gross - comm
            cash += net
        turnover += gross
        reason = f"ENTRY_RANK_{c.rank}" if existing is None else ("REBAL_ADD" if side == "BUY" else "REBAL_TRIM")
        _log(c.ticker, side, abs(delta), price_f, fill, comm, gross, net, reason)

        new_shares = (existing.shares if existing else 0.0) + delta
        if new_shares > 1e-9:
            holdings[c.ticker] = _Position(
                shares=new_shares,
                avg_cost=price_f if existing is None else existing.avg_cost,
                entry_date=existing.entry_date if existing else execution_date,
                entry_rank=existing.entry_rank if existing else c.rank,
            )
        else:
            holdings.pop(c.ticker, None)

    return cash, turnover


def _rebalance_single_asset(
    shares: float, cash: float, target_usd: float, price: float, slippage_bps: float
) -> Tuple[float, float]:
    """Rebalance a single-instrument sleeve (MTUM, tracks C/D) to target_usd."""
    current_usd = shares * price
    delta_usd = target_usd - current_usd
    if abs(delta_usd) < MIN_TRADE_USD:
        return shares, cash
    side: Literal["BUY", "SELL"] = "BUY" if delta_usd > 0 else "SELL"
    delta_shares = abs(delta_usd) / price
    fill, comm = simulate_fill(side, delta_shares, price, slippage_bps=slippage_bps)
    gross = fill * delta_shares
    if side == "BUY":
        cash -= gross + comm
        shares += delta_shares
    else:
        cash += gross - comm
        shares -= delta_shares
    return shares, cash


def _catastrophic_stop_check(
    holdings_a: Dict[str, _Position],
    holdings_b: Dict[str, _Position],
    close_frame: pd.DataFrame,
    ts: pd.Timestamp,
    cash_a: float,
    cash_b: float,
    slippage_bps: float,
    trades: List[LedgerEntry],
) -> Tuple[float, float]:
    """spec §11 step 4 — see module docstring for the same-day-at-close simplification."""
    day = ts.date()
    for ticker in list(holdings_a.keys()):
        price = _price_at(close_frame, ticker, ts)
        pos = holdings_a[ticker]
        if price is None or pos.avg_cost <= 0 or price >= pos.avg_cost * (1 + CATASTROPHIC_STOP_PCT):
            continue
        fill, comm = simulate_fill("SELL", pos.shares, price, slippage_bps=slippage_bps)
        gross = fill * pos.shares
        net = gross - comm
        cash_a += net
        trades.append(
            _ledger_entry(
                day, "A", ticker, "SELL", pos.shares, price, fill,
                slippage_bps, comm, gross, net, "EXIT_CATASTROPHIC_STOP",
            )
        )
        del holdings_a[ticker]

        b_pos = holdings_b.pop(ticker, None)
        if b_pos is not None:
            b_fill, b_comm = simulate_fill("SELL", b_pos.shares, price, slippage_bps=slippage_bps)
            b_gross = b_fill * b_pos.shares
            b_net = b_gross - b_comm
            cash_b += b_net
            trades.append(
                _ledger_entry(
                    day, "B", ticker, "SELL", b_pos.shares, price, b_fill,
                    slippage_bps, b_comm, b_gross, b_net, "EXIT_CATASTROPHIC_STOP",
                )
            )
    return cash_a, cash_b


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------


def run_backtest(
    panel: Dict[str, pd.DataFrame],
    sector_by_ticker: Dict[str, str],
    start: date,
    end: date,
    params: Optional[BacktestParams] = None,
) -> BacktestResult:
    """
    Run the full mechanical backtest from start to end against a frozen panel.

    Args:
        panel: {ticker: OHLCV DataFrame} — the frozen price panel, same
            shape as YahooDataDownloader.get_ohlcv_batch()'s output, plus
            MTUM/SPY/^GSPC/^VIX. Must be pre-fetched
            (backtest/p21_momentum/fetch_frozen_panel.py) and never
            re-downloaded mid-study (spec §14.2's "price adjustment drift"
            mitigation).
        sector_by_ticker: {ticker: GICS sector} for every ticker in the
            universe (from data/universe.py's Option A fetch).
        start, end: Backtest date range (spec §14.4: 2005-01-01 to
            2026-06-30, with warmup before the first tradeable signal).
        params: Strategy parameters; defaults to the live pipeline's own
            spec §16 values.

    Returns:
        BacktestResult with daily NAV for all five tracks, the stock-sleeve
        trade ledger (tracks A/B only — see module docstring), one
        MonthlyMetrics row per rebalance month, and the regime history.
    """
    p = params or BacktestParams()
    universe = sorted(sector_by_ticker.keys())
    all_tickers = sorted(set(universe) | {PROXY_TICKER, MARKET_TICKER, REGIME_INDEX_TICKER, REGIME_VOL_TICKER})

    cal_days = trading_days(start, end)
    if not cal_days:
        raise ValueError(f"No trading days between {start} and {end}")
    calendar_index = pd.DatetimeIndex([pd.Timestamp(d) for d in cal_days])

    close_frame = _prepare_field_frame(panel, all_tickers, calendar_index, "close")
    open_frame = _prepare_field_frame(panel, all_tickers, calendar_index, "open")
    volume_frame = _prepare_field_frame(panel, universe, calendar_index, "volume")

    pairs = _signal_execution_pairs(start, end)
    signal_dates: Dict[date, Tuple[date, str]] = {sd: (ed, key) for sd, ed, key in pairs}
    execution_dates: Set[date] = {ed for _, ed, _ in pairs}

    holdings_a: Dict[str, _Position] = {}
    holdings_b: Dict[str, _Position] = {}
    cash_a = cash_b = cash_c = cash_d = p.initial_nav
    mtum_shares_c = mtum_shares_d = 0.0
    spy_shares_e = 0.0

    trades: List[LedgerEntry] = []
    monthly_metrics: List[MonthlyMetrics] = []
    month_index: Dict[str, int] = {}
    regime_history_raw: List[dict] = []
    scalar_applied = 1.0
    pending: Optional[_PendingTarget] = None
    prev_close: Dict[str, float] = {}
    nav_rows: List[dict] = []

    for ts in calendar_index:
        day = ts.date()

        # --- 1. Signal date: compute regime, filters, selection, sizing; queue for execution ---
        if day in signal_dates:
            execution_date, month_key = signal_dates[day]

            gspc_hist = close_frame[REGIME_INDEX_TICKER].loc[:ts].dropna()
            vix_hist = close_frame[REGIME_VOL_TICKER].loc[:ts].dropna()
            if len(gspc_hist) >= 253 and len(vix_hist) >= 20:
                recent_raw = [(h["bear"], h["high_vol"]) for h in reversed(regime_history_raw[-1:])]
                prior_state = PriorRegimeState(
                    scalar_applied=scalar_applied,
                    months_at_state=regime_history_raw[-1]["months_at_state"] if regime_history_raw else 0,
                    recent_raw_states=recent_raw,
                )
                regime_result = compute_regime(
                    gspc_hist, vix_hist, prior_state, as_of=day, vix_threshold=p.vix_threshold
                )
                scalar_applied = regime_result.scalar_applied
                regime_history_raw.append(
                    {
                        "date": day.isoformat(),
                        "bear": regime_result.bear,
                        "high_vol": regime_result.high_vol,
                        "scalar_raw": regime_result.scalar_raw,
                        "scalar_applied": regime_result.scalar_applied,
                        "months_at_state": regime_result.months_at_state,
                    }
                )
            # else: insufficient regime history yet -> scalar_applied stays at its prior value (default 1.0)

            survivors: List[RankedCandidate] = []
            vol_by_ticker: Dict[str, float] = {}
            f1_removed = f2_removed = f3_removed = 0
            exit_delisted_prices: Dict[str, float] = {}

            for ticker in universe:
                series = close_frame[ticker].loc[:ts].dropna()
                if ticker in holdings_a:
                    delisting = detect_and_liquidate_delisting(ticker, series, day)
                    if delisting is not None:
                        exit_delisted_prices[ticker] = delisting.liquidation_price
                        continue

                sig = compute_signal(series)
                if sig is None:
                    continue

                f1 = f1_history(series)
                if not f1.passed:
                    f1_removed += 1
                    continue
                vol = volume_frame[ticker].loc[:ts].dropna()
                f2 = f2_liquidity(series, vol)
                if not f2.passed:
                    f2_removed += 1
                    continue
                window = series.iloc[-p.lookback_start : -p.skip_recent] if len(series) >= p.lookback_start else series
                f3 = f3_gap(window)
                if not f3.passed:
                    f3_removed += 1
                    continue

                survivors.append(RankedCandidate(ticker=ticker, signal=sig.signal, sector=sector_by_ticker.get(ticker)))
                vol_by_ticker[ticker] = sig.vol

            ranked = rank_candidates(survivors)
            holding_inputs = [
                CurrentHolding(ticker=t, sector=sector_by_ticker.get(t))
                for t in holdings_a
                if t not in exit_delisted_prices
            ]
            selection = select_portfolio(
                ranked,
                holding_inputs,
                forced_exits=set(exit_delisted_prices.keys()),
                entry_rank=p.entry_rank,
                hold_rank=p.hold_rank,
                max_per_sector=p.max_per_sector,
                target_count=p.target_count,
                fallback_pool_rank=p.fallback_pool_rank,
            )

            sector_counts: Dict[Optional[str], int] = {}
            for c in selection.selected:
                sector_counts[c.sector] = sector_counts.get(c.sector, 0) + 1

            vols = {c.ticker: vol_by_ticker.get(c.ticker, 0.20) for c in selection.selected}
            nav_a_now = cash_a + _sleeve_value(holdings_a, close_frame, ts)
            nav_b_now = cash_b + _sleeve_value(holdings_b, close_frame, ts)
            alloc_a = size_positions(
                vols, nav_total=nav_a_now, sleeve_pct=p.sleeve_pct,
                max_pos_pct=p.max_position_pct, regime_scalar=scalar_applied,
            )
            alloc_b = size_positions(
                vols, nav_total=nav_b_now, sleeve_pct=p.sleeve_pct, max_pos_pct=p.max_position_pct, regime_scalar=1.0
            )

            pending = _PendingTarget(
                execution_date=execution_date,
                month_key=month_key,
                selected=selection.selected,
                ranked=ranked,
                alloc_a=alloc_a,
                alloc_b=alloc_b,
                exit_delisted_prices=exit_delisted_prices,
                scalar_applied=scalar_applied,
            )
            monthly_metrics.append(
                MonthlyMetrics(
                    month=month_key,
                    turnover_two_way_usd=0.0,  # filled in at execution below
                    position_count=len(selection.selected),
                    sector_herfindahl=_herfindahl(sector_counts, len(selection.selected)),
                    f1_removed=f1_removed,
                    f2_removed=f2_removed,
                    f3_removed=f3_removed,
                    regime_scalar=scalar_applied,
                    warn_underfilled=selection.underfilled,
                    exit_delisted_count=len(exit_delisted_prices),
                    max_sector_count=max(sector_counts.values(), default=0),
                )
            )
            month_index[month_key] = len(monthly_metrics) - 1

        # --- 2. Execution date: execute the pending target ---
        if pending is not None and day == pending.execution_date and day in execution_dates and ts in open_frame.index:
            cash_a, turnover_a = _rebalance_stock_sleeve(
                "A", holdings_a, cash_a, pending.selected, pending.ranked, pending.alloc_a,
                open_frame, ts, day, p.slippage_bps, pending.exit_delisted_prices, trades,
            )
            # Track B's own turnover is not part of any §14.8 required metric — only A's is.
            cash_b, _ = _rebalance_stock_sleeve(
                "B", holdings_b, cash_b, pending.selected, pending.ranked, pending.alloc_b,
                open_frame, ts, day, p.slippage_bps, pending.exit_delisted_prices, trades,
            )
            monthly_metrics[month_index[pending.month_key]].turnover_two_way_usd = turnover_a

            mtum_price_f = _price_at(open_frame, PROXY_TICKER, ts)
            if mtum_price_f is not None and mtum_price_f > 0:
                nav_c = cash_c + mtum_shares_c * mtum_price_f
                mtum_shares_c, cash_c = _rebalance_single_asset(
                    mtum_shares_c, cash_c, nav_c * p.sleeve_pct, mtum_price_f, p.slippage_bps
                )
                nav_d = cash_d + mtum_shares_d * mtum_price_f
                mtum_shares_d, cash_d = _rebalance_single_asset(
                    mtum_shares_d, cash_d, nav_d * p.sleeve_pct * pending.scalar_applied, mtum_price_f, p.slippage_bps
                )

            spy_price_f = _price_at(open_frame, MARKET_TICKER, ts)
            if spy_shares_e == 0.0 and spy_price_f is not None and spy_price_f > 0:
                spy_shares_e = p.initial_nav / spy_price_f

            pending = None

        # --- 3. Catastrophic stop check (§11 step 4) ---
        cash_a, cash_b = _catastrophic_stop_check(
            holdings_a, holdings_b, close_frame, ts, cash_a, cash_b, p.slippage_bps, trades
        )

        # --- 4. Anomaly check (§11 step 5) — held names only, do not auto-trade ---
        month_key_today = f"{day.year:04d}-{day.month:02d}"
        for ticker in holdings_a:
            price = _price_at(close_frame, ticker, ts)
            prior = prev_close.get(ticker)
            if price is not None and prior is not None and prior > 0:
                if abs(price / prior - 1.0) > ANOMALY_DAILY_CHANGE_PCT and month_key_today in month_index:
                    monthly_metrics[month_index[month_key_today]].manual_review_count += 1
        for ticker in all_tickers:
            price = _price_at(close_frame, ticker, ts)
            if price is not None:
                prev_close[ticker] = price

        # --- 5. TER drag (§9), applied daily to tracks C/D's MTUM position ---
        mtum_shares_c *= 1 - _TER_DAILY
        mtum_shares_d *= 1 - _TER_DAILY

        # --- 6. Daily NAV mark (§11 step 2) ---
        mtum_price_today = _price_at(close_frame, PROXY_TICKER, ts)
        spy_price_today = _price_at(close_frame, MARKET_TICKER, ts)
        nav_a = cash_a + _sleeve_value(holdings_a, close_frame, ts)
        nav_b = cash_b + _sleeve_value(holdings_b, close_frame, ts)
        nav_c = cash_c + (mtum_shares_c * mtum_price_today if mtum_price_today is not None else 0.0)
        nav_d = cash_d + (mtum_shares_d * mtum_price_today if mtum_price_today is not None else 0.0)
        nav_e = spy_shares_e * spy_price_today if spy_price_today is not None and spy_shares_e > 0 else p.initial_nav
        # Keyed by `ts` (a pd.Timestamp), not `day` (a datetime.date) — BacktestResult.nav_daily
        # must carry a genuine DatetimeIndex so downstream .resample()/.rolling() calls in
        # metrics.py work directly against it, and so date comparisons in stress_windows.py
        # don't hit Python's date-vs-datetime TypeError.
        nav_rows.append({"date": ts, "nav_a": nav_a, "nav_b": nav_b, "nav_c": nav_c, "nav_d": nav_d, "nav_e": nav_e})

    nav_daily = (
        pd.DataFrame(nav_rows).set_index("date")
        if nav_rows
        else pd.DataFrame(columns=["nav_a", "nav_b", "nav_c", "nav_d", "nav_e"], index=pd.DatetimeIndex([]))
    )
    nav_daily.index = pd.DatetimeIndex(nav_daily.index)
    return BacktestResult(
        nav_daily=nav_daily, trades=trades, monthly_metrics=monthly_metrics, regime_history=regime_history_raw
    )
