"""
P21 Momentum — scheduler entry point: monthly rebalance (docs/pipeline-specification.md §1, §4-§8).

fetch -> validate -> signal -> filter -> rank -> select -> size -> regime -> write TARGET

Runs daily (self-guarding, Open Decision #2 of docs/implementation-plan.md
§10): no-ops unless today is the last NYSE trading day of the month.

**Documented simplification vs. the letter of spec §5:** F6 (earnings
blackout) is evaluated only against the tickers select_portfolio() actually
picks as *new* entries, after selection, rather than against every top-40
survivor before ranking. This means fewer earnings-calendar lookups (only
actual new entries, not the whole candidate pool) and does not backfill a
replacement when a new entry fails F6 — the position count simply comes in
lower, which is exactly the WARN_UNDERFILLED path §6 already defines for
"fewer than target_count after filtering." No backfill loop is added on top
of that existing path.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.ml.pipeline.p21_momentum.calendar import first_trading_day_of_month, last_trading_day_of_month
from src.ml.pipeline.p21_momentum.config import (
    CURRENT_POSITIONS_PATH,
    EARNINGS_BLACKOUT_DAYS,
    EXCLUSIONS_PATH,
    MAX_POSITION_PCT,
    NAV_TOTAL_USD,
    PROXY_TICKER,
    REGIME_HISTORY_PATH,
    REGIME_INDEX_TICKER,
    REGIME_VOL_TICKER,
    RESULTS_DIR,
    SLEEVE_TARGET_PCT,
    UPGRADE_CONFIRMATION_MONTHS,
)
from src.ml.pipeline.p21_momentum.data.earnings import next_earnings_date
from src.ml.pipeline.p21_momentum.data.exclusions import load_exclusions
from src.ml.pipeline.p21_momentum.data.prices import fetch_fundamentals_cached, fetch_price_panel
from src.ml.pipeline.p21_momentum.data.universe import fetch_universe, universe_to_json
from src.ml.pipeline.p21_momentum.execution.ledger import read_current_positions
from src.ml.pipeline.p21_momentum.jobs.run_common import send_abort_alert, setup_run_logging
from src.ml.pipeline.p21_momentum.quality.gates import (
    GateOutcome,
    PipelineAbort,
    check_constituent_count,
    check_no_weight_exceeds_cap,
    check_position_count,
    check_regime_inputs_available,
    check_signal_date_is_trading_day,
    check_target_weight_sum,
    run_gates,
)
from src.ml.pipeline.p21_momentum.results.run_io import (
    already_processed,
    append_regime_history,
    write_targets,
    write_universe,
)
from src.ml.pipeline.p21_momentum.schemas import SignalRow, TargetPosition
from src.ml.pipeline.p21_momentum.strategy.filters import f6_earnings
from src.ml.pipeline.p21_momentum.strategy.filters import run_all as run_filters
from src.ml.pipeline.p21_momentum.strategy.regime import PriorRegimeState, RegimeResult, compute_regime
from src.ml.pipeline.p21_momentum.strategy.selection import (
    CurrentHolding,
    RankedCandidate,
    rank_candidates,
    select_portfolio,
)
from src.ml.pipeline.p21_momentum.strategy.signal import compute_signal
from src.ml.pipeline.p21_momentum.strategy.sizing import size_positions
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_PRICE_HISTORY_BUFFER_DAYS = 430  # calendar days of buffer for ~260 trading-day lookback + holidays


def _series_from_panel(df: Optional[pd.DataFrame], column: str) -> pd.Series:
    if df is None or df.empty or column not in df.columns:
        return pd.Series(dtype=float)
    s = df.set_index("timestamp")[column]
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def _build_prior_regime_state(history: List[Dict]) -> PriorRegimeState:
    if not history:
        return PriorRegimeState(scalar_applied=1.0, months_at_state=0, recent_raw_states=[])
    last = history[-1]
    recent = [(h["bear"], h["high_vol"]) for h in reversed(history[:-1][-(UPGRADE_CONFIRMATION_MONTHS - 1) :])]
    return PriorRegimeState(
        scalar_applied=last["scalar_applied"], months_at_state=last["months_at_state"], recent_raw_states=recent
    )


def run(
    run_date: Optional[date] = None,
    force: bool = False,
    results_dir: Path = RESULTS_DIR,
    regime_history_path: Path = REGIME_HISTORY_PATH,
    current_positions_path: Path = CURRENT_POSITIONS_PATH,
    exclusions_path: Path = EXCLUSIONS_PATH,
) -> Dict:
    """
    Execute one monthly_rebalance cycle.

    Args:
        run_date: Override "today" (for tests / manual backfill). Defaults
            to date.today().
        force: Bypass the idempotency check (spec §3).
        results_dir, regime_history_path, current_positions_path,
            exclusions_path: Overridable for tests (real determinism/
            round-trip checks against a tmp dir instead of the real
            results/p21_momentum/) — production callers never pass these.

    Returns:
        Summary dict for __SCHEDULER_RESULT__.
    """
    today = run_date or date.today()

    if today != _last_trading_day_safe(today):
        _logger.info("SKIP: %s is not the last NYSE trading day of its month", today)
        return {"skipped": True, "reason": "not_last_trading_day", "date": today.isoformat()}

    signal_date = today
    if already_processed(signal_date, "targets.json", results_dir=results_dir) and not force:
        _logger.info("SKIP: already processed for %s", signal_date)
        return {"skipped": True, "reason": "already_processed", "date": signal_date.isoformat()}

    try:
        return _run_rebalance(signal_date, results_dir, regime_history_path, current_positions_path, exclusions_path)
    except PipelineAbort as exc:
        _logger.error("ABORT: %s", exc)
        send_abort_alert("monthly_rebalance", exc)
        return {"aborted": True, "check": exc.check, "context": exc.context}


def _last_trading_day_safe(today: date) -> date:
    return last_trading_day_of_month(today.year, today.month)


def _run_rebalance(
    signal_date: date,
    results_dir: Path = RESULTS_DIR,
    regime_history_path: Path = REGIME_HISTORY_PATH,
    current_positions_path: Path = CURRENT_POSITIONS_PATH,
    exclusions_path: Path = EXCLUSIONS_PATH,
) -> Dict:
    gate_results = []

    def _check(result):
        gate_results.append(result)
        run_gates([result])  # raises PipelineAbort immediately on ABORT
        return result

    # By construction run() only reaches here when signal_date == the last trading
    # day of its month (checked above), so this is always True — kept as an
    # explicit §13 gate row for symmetry with the other checks and so a future
    # caller of _run_rebalance() directly (e.g. a backtest reusing this function)
    # doesn't silently skip the check.
    _check(check_signal_date_is_trading_day(signal_date == _last_trading_day_safe(signal_date)))

    constituents = fetch_universe()
    _check(check_constituent_count(len(constituents)))
    write_universe(signal_date, universe_to_json(constituents, as_of=signal_date.isoformat()), results_dir=results_dir)

    sector_by_ticker: Dict[str, str] = {c.ticker: c.sector for c in constituents}
    constituent_tickers = set(sector_by_ticker.keys())

    current_positions = read_current_positions(path=current_positions_path)
    held_tickers = {p.ticker for p in current_positions}

    benchmark_tickers = {PROXY_TICKER, "SPY", REGIME_INDEX_TICKER, REGIME_VOL_TICKER}
    all_tickers = sorted(constituent_tickers | held_tickers | benchmark_tickers)

    start_date = datetime.combine(signal_date, datetime.min.time()) - timedelta(days=_PRICE_HISTORY_BUFFER_DAYS)
    end_date = datetime.combine(signal_date, datetime.min.time()) + timedelta(days=1)
    panel = fetch_price_panel(all_tickers, start_date, end_date)  # raises PipelineAbort on low coverage

    excluded = load_exclusions(path=exclusions_path, as_of=signal_date)

    # --- Regime (§8) ---
    gspc_df, vix_df = panel.get(REGIME_INDEX_TICKER), panel.get(REGIME_VOL_TICKER)
    gspc_available = gspc_df is not None and not gspc_df.empty
    vix_available = vix_df is not None and not vix_df.empty
    regime_available = gspc_available and vix_available
    _check(check_regime_inputs_available(gspc_available, vix_available))

    history = json.loads(regime_history_path.read_text(encoding="utf-8")) if regime_history_path.exists() else []
    prior_state = _build_prior_regime_state(history)
    if regime_available:
        spx = _series_from_panel(gspc_df, "close")
        vix = _series_from_panel(vix_df, "close")
        regime_result = compute_regime(spx, vix, prior_state, as_of=signal_date)
    else:
        # HOLD: retain prior scalar (spec §13)
        regime_result = RegimeResult(
            date=signal_date.isoformat(),
            spx_12m_return=0.0,
            spx_vs_200dma=1.0,
            vix_20d_avg=0.0,
            bear=history[-1]["bear"] if history else False,
            high_vol=history[-1]["high_vol"] if history else False,
            scalar_raw=prior_state.scalar_applied,
            scalar_applied=prior_state.scalar_applied,
            months_at_state=prior_state.months_at_state + 1,
        )
    append_regime_history(asdict(regime_result), regime_history_path)

    # --- Signal + filters (§4, §5) ---
    fundamentals = fetch_fundamentals_cached(sorted(constituent_tickers | held_tickers))
    signal_rows: List[SignalRow] = []
    survivors: List[RankedCandidate] = []
    survivor_vol: Dict[str, float] = {}
    forced_exits = set()

    candidate_universe = constituent_tickers | held_tickers
    for ticker in sorted(candidate_universe):
        df = panel.get(ticker)
        adj_close = _series_from_panel(df, "close") if df is not None else pd.Series(dtype=float)
        volume = _series_from_panel(df, "volume") if df is not None else pd.Series(dtype=float)

        sig = compute_signal(adj_close)
        sector = sector_by_ticker.get(ticker)
        if ticker not in constituent_tickers and ticker in held_tickers:
            existing = next((p for p in current_positions if p.ticker == ticker), None)
            sector = existing.sector if existing else sector

        if sig is None:
            if ticker in held_tickers:
                forced_exits.add(ticker)  # delisted / no data
            continue

        fund = fundamentals.get(ticker, {})
        signal_window = adj_close.iloc[-252:-21] if len(adj_close) >= 273 else adj_close
        outcome = run_filters(
            ticker=ticker,
            adj_close=adj_close,
            volume=volume,
            signal_window=signal_window,
            fcf_ttm=fund.get("fcf_ttm"),
            net_income_ttm=fund.get("net_income_ttm"),
            excluded=excluded,
            next_earnings=None,
            execution_date=signal_date,
            is_new_entry=False,  # F6 deferred, see module docstring
        )
        signal_rows.append(
            SignalRow(
                ticker=ticker,
                raw_return=sig.raw_return,
                vol=sig.vol,
                signal=sig.signal,
                rank=0,
                sector=sector,
                filters_passed={k: {"passed": v.passed, "flag": v.flag} for k, v in outcome.filters.items()},
            )
        )

        if ticker not in constituent_tickers and ticker in held_tickers:
            forced_exits.add(ticker)  # dropped from index

        if outcome.passed:
            survivors.append(RankedCandidate(ticker=ticker, signal=sig.signal, sector=sector))
            survivor_vol[ticker] = sig.vol
        elif ticker in held_tickers:
            forced_exits.add(ticker)  # failed F1/F2/F3/F5

    ranked = rank_candidates(survivors)
    for row in signal_rows:
        match = next((c for c in ranked if c.ticker == row.ticker), None)
        if match:
            row.rank = match.rank

    holding_inputs = [CurrentHolding(ticker=p.ticker, sector=p.sector) for p in current_positions]
    selection = select_portfolio(ranked, holding_inputs, forced_exits)

    if selection.underfilled:
        _logger.warning("WARN_UNDERFILLED: only %d of target positions selected", len(selection.selected))

    # --- F6 on new entries only (see module docstring) ---
    next_execution_date = first_trading_day_of_month(
        signal_date.year + (1 if signal_date.month == 12 else 0),
        1 if signal_date.month == 12 else signal_date.month + 1,
    )
    final_selected = []
    for c in selection.selected:
        is_new = c.ticker not in held_tickers
        if is_new:
            earnings = next_earnings_date(c.ticker)
            f6 = f6_earnings(earnings, next_execution_date, is_new_entry=True, blackout_days=EARNINGS_BLACKOUT_DAYS)
            if not f6.passed:
                _logger.info("Excluding new entry %s: F6 earnings blackout", c.ticker)
                continue
        final_selected.append(c)

    _check(check_position_count(len(final_selected)))

    # --- Sizing (§7) ---
    vol_by_ticker = {c.ticker: survivor_vol.get(c.ticker, 0.20) for c in final_selected}
    allocation = size_positions(
        vol_by_ticker,
        nav_total=NAV_TOTAL_USD,
        sleeve_pct=SLEEVE_TARGET_PCT,
        max_pos_pct=MAX_POSITION_PCT,
        regime_scalar=regime_result.scalar_applied,
    )
    sleeve_usd = NAV_TOTAL_USD * SLEEVE_TARGET_PCT * regime_result.scalar_applied
    if allocation:
        _check(check_target_weight_sum(sum(allocation.values()), sleeve_usd))
        _check(
            check_no_weight_exceeds_cap(
                max(allocation.values()), nav_total=NAV_TOTAL_USD, max_position_pct=MAX_POSITION_PCT
            )
        )

    targets = [
        TargetPosition(
            ticker=c.ticker,
            target_weight_pct=allocation.get(c.ticker, 0.0) / NAV_TOTAL_USD,
            target_usd=allocation.get(c.ticker, 0.0),
            rank=c.rank,
            sector=c.sector,
        )
        for c in final_selected
    ]
    write_targets(signal_date, targets, results_dir=results_dir)

    warn_count = sum(1 for r in gate_results if r.outcome == GateOutcome.WARN)
    _logger.info("monthly_rebalance complete for %s: %d targets, %d warnings", signal_date, len(targets), warn_count)
    return {
        "signal_date": signal_date.isoformat(),
        "targets_count": len(targets),
        "warn_underfilled": selection.underfilled,
        "regime_scalar": regime_result.scalar_applied,
        "gate_warnings": warn_count,
    }


def main() -> None:
    """Run monthly_rebalance and print scheduler result."""
    setup_run_logging()
    result = run()
    _logger.info("monthly_rebalance result: %s", result)
    print(f"__SCHEDULER_RESULT__:{json.dumps(result, default=str)}")


if __name__ == "__main__":
    main()
