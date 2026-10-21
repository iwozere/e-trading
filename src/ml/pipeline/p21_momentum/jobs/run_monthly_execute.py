"""
P21 Momentum — scheduler entry point: monthly execute (docs/pipeline-specification.md §1, §9-§12).

read TARGET -> fetch opens -> simulate fills -> apply costs -> write POSITIONS + LEDGER
-> update all 4 tracks -> generate REPORT

Runs daily (self-guarding, same pattern as run_monthly_rebalance.py):
no-ops unless today is the first NYSE trading day of the month.

Tracks B-E are replayed from inception (strategy/live_tracks.py) for the
report's NAV table, the §9 decomposition, the month/YTD/since-inception
returns and max drawdowns. **Still a known gap:** the §12.5/§12.6 realized
metrics (annualized turnover, costs in bps, decision criteria) are literal
0.0 placeholders in the render_report() call below, not fabricated numbers.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p21_momentum.calendar import first_trading_day_of_month, last_trading_day_of_month
from src.ml.pipeline.p21_momentum.config import MAX_PER_SECTOR, NAV_TOTAL_USD, PROXY_TICKER, RESULTS_DIR, STATE_DIR
from src.ml.pipeline.p21_momentum.data.prices import fetch_price_panel
from src.ml.pipeline.p21_momentum.execution.fills import TradeIntent, apply_chatter_threshold, execute_trades
from src.ml.pipeline.p21_momentum.execution.ledger import (
    append_ledger_entries,
    read_current_positions,
    write_current_positions,
)
from src.ml.pipeline.p21_momentum.jobs.run_common import replay_live_tracks, send_abort_alert, setup_run_logging
from src.ml.pipeline.p21_momentum.quality.gates import PipelineAbort, check_cash_after_execution, run_gates
from src.ml.pipeline.p21_momentum.reporting.monthly_report import DecisionMetrics, render_report
from src.ml.pipeline.p21_momentum.results.run_io import (
    already_processed,
    read_regime_history,
    read_targets,
    write_positions,
    write_report,
)
from src.ml.pipeline.p21_momentum.schemas import LedgerEntry, Position
from src.ml.pipeline.p21_momentum.strategy.live_tracks import summarize_navs
from src.ml.pipeline.p21_momentum.strategy.tracks import compute_attribution
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def run(
    run_date: Optional[date] = None,
    force: bool = False,
    results_dir: Path = RESULTS_DIR,
    state_dir: Path = STATE_DIR,
    current_positions_path: Optional[Path] = None,
) -> Dict:
    """
    Execute one monthly_execute cycle.

    Args:
        run_date: Override "today" (for tests / manual backfill).
        force: Bypass the idempotency check.
        results_dir, state_dir: Overridable for tests (real determinism/
            round-trip checks against a tmp dir) — production callers never
            pass these.
        current_positions_path: Overridable independently of state_dir for
            tests that need a specific starting-positions fixture; defaults
            to state_dir / "current_positions.json".

    Returns:
        Summary dict for __SCHEDULER_RESULT__.
    """
    today = run_date or date.today()
    expected = first_trading_day_of_month(today.year, today.month)
    if today != expected:
        _logger.info("SKIP: %s is not the first NYSE trading day of its month", today)
        return {"skipped": True, "reason": "not_first_trading_day", "date": today.isoformat()}

    execution_date = today
    if already_processed(execution_date, "positions.json", results_dir=results_dir) and not force:
        _logger.info("SKIP: already processed for %s", execution_date)
        return {"skipped": True, "reason": "already_processed", "date": execution_date.isoformat()}

    # TARGET is read from the prior trading day (the signal date, last
    # trading day of the PRIOR month) — targets.json lives in that run's
    # dated folder, not today's.
    signal_month = today.month - 1 or 12
    signal_year = today.year if today.month > 1 else today.year - 1
    signal_date = last_trading_day_of_month(signal_year, signal_month)
    targets = read_targets(signal_date, results_dir=results_dir)
    if not targets:
        _logger.warning("No targets.json found for signal date %s — nothing to execute", signal_date)
        return {"skipped": True, "reason": "no_targets", "signal_date": signal_date.isoformat()}

    positions_path = current_positions_path or (state_dir / "current_positions.json")
    try:
        return _run_execute(execution_date, signal_date, targets, results_dir, state_dir, positions_path)
    except PipelineAbort as exc:
        _logger.error("ABORT: %s", exc)
        send_abort_alert("monthly_execute", exc)
        return {"aborted": True, "check": exc.check, "context": exc.context}


def _run_execute(
    execution_date: date,
    signal_date: date,
    targets: List,
    results_dir: Path,
    state_dir: Path,
    positions_path: Path,
) -> Dict:
    current_positions = read_current_positions(path=positions_path)
    current_by_ticker = {p.ticker: p for p in current_positions}
    target_tickers = {t.ticker for t in targets}

    tickers_needed = sorted(target_tickers | set(current_by_ticker.keys()) | {PROXY_TICKER, "SPY"})
    start_date = datetime.combine(execution_date, datetime.min.time()) - timedelta(days=10)
    end_date = datetime.combine(execution_date, datetime.min.time()) + timedelta(days=1)
    panel = fetch_price_panel(tickers_needed, start_date, end_date)

    open_prices: Dict[str, float] = {}
    for ticker, df in panel.items():
        if df is None or df.empty:
            continue
        row = df[df["timestamp"].dt.date == execution_date]
        if not row.empty:
            open_prices[ticker] = float(row.iloc[0]["open"])

    # Build trade intents: sells for dropped holdings, buys/rebalances for targets.
    intents: List[TradeIntent] = []

    for ticker, pos in current_by_ticker.items():
        current_value = pos.shares * open_prices.get(ticker, pos.avg_cost)
        if ticker not in target_tickers:
            intents.append(
                TradeIntent(
                    ticker=ticker, side="SELL", shares=pos.shares, current_value_usd=current_value, target_value_usd=0.0
                )
            )

    for t in targets:
        existing = current_by_ticker.get(t.ticker)
        current_shares = existing.shares if existing else 0.0
        price = open_prices.get(t.ticker)
        if price is None or price <= 0:
            continue
        target_shares = t.target_usd / price
        delta_shares = target_shares - current_shares
        if abs(delta_shares) < 1e-9:
            continue
        side: Literal["BUY", "SELL"] = "BUY" if delta_shares > 0 else "SELL"
        current_value = current_shares * price
        intents.append(
            TradeIntent(
                ticker=t.ticker,
                side=side,
                shares=abs(delta_shares),
                current_value_usd=current_value,
                target_value_usd=t.target_usd,
            )
        )

    intents = apply_chatter_threshold(intents)

    prior_cash = _read_prior_cash(positions_path)
    # No prior _state/current_positions.json (first-ever run): the sleeve has
    # not yet drawn from NAV, so cash on hand is NAV minus the reserved
    # sleeve fraction that this run's targets are about to allocate.
    cash_before = prior_cash if prior_cash is not None else NAV_TOTAL_USD * (1 - 0.20)
    outcome = execute_trades(intents, open_prices, available_cash=cash_before)

    ledger_entries: List[LedgerEntry] = []
    for trade in outcome.trades:
        target = next((t for t in targets if t.ticker == trade.ticker), None)
        is_new_entry = target is not None and trade.ticker not in current_by_ticker
        if is_new_entry and target is not None:
            reason = f"ENTRY_RANK_{target.rank}"
        elif trade.side == "BUY":
            reason = "REBAL_ADD"
        elif trade.ticker not in target_tickers:
            reason = "EXIT_RANK_DROP"
        else:
            reason = "REBAL_TRIM"
        ledger_entries.append(
            LedgerEntry(
                ts=datetime.combine(execution_date, datetime.min.time()).isoformat(),
                track="A",
                ticker=trade.ticker,
                side=trade.side,
                shares=trade.shares,
                ref_open=trade.ref_open,
                fill_price=trade.fill_price,
                slippage_bps=trade.slippage_bps,
                commission_usd=trade.commission_usd,
                gross_usd=trade.gross_usd,
                net_usd=trade.net_usd,
                reason=reason,
            )
        )
    append_ledger_entries(ledger_entries, path=state_dir / "ledger.jsonl")

    # Recompute new positions from fills.
    new_positions: List[Position] = []
    cash = cash_before
    shares_by_ticker: Dict[str, float] = {t: p.shares for t, p in current_by_ticker.items()}
    for trade in outcome.trades:
        delta = trade.shares if trade.side == "BUY" else -trade.shares
        shares_by_ticker[trade.ticker] = shares_by_ticker.get(trade.ticker, 0.0) + delta
        cash += -trade.net_usd if trade.side == "BUY" else trade.net_usd

    for t in targets:
        shares = shares_by_ticker.get(t.ticker, 0.0)
        if shares <= 0:
            continue
        prior = current_by_ticker.get(t.ticker)
        price = open_prices.get(t.ticker, prior.avg_cost if prior else 0.0)
        new_positions.append(
            Position(
                ticker=t.ticker,
                shares=shares,
                avg_cost=prior.avg_cost if prior else price,
                entry_date=prior.entry_date if prior else execution_date.isoformat(),
                entry_rank=prior.entry_rank if prior else t.rank,
                current_rank=t.rank,
                sector=t.sector,
                target_weight_pct=t.target_weight_pct,
                high_water_price=max(prior.high_water_price, price) if prior else price,
            )
        )

    sleeve_market_value = sum(p.shares * open_prices.get(p.ticker, p.avg_cost) for p in new_positions)
    check = check_cash_after_execution(cash)
    run_gates([check])

    regime = _regime_for(signal_date, state_dir / "regime_history.json")
    write_current_positions(
        new_positions,
        as_of=execution_date,
        track="A",
        nav_total=NAV_TOTAL_USD,
        cash=cash,
        sleeve_market_value=sleeve_market_value,
        regime_scalar=float(regime["scalar_applied"]),
        path=positions_path,
    )
    write_positions(execution_date, new_positions, results_dir=results_dir)

    # --- Report (§12) ---
    # Track A's real NAV is cash + sleeve market value (spec §9). Tracks
    # B-E need the full NAV-history walk flagged in the module docstring —
    # NAV_TOTAL_USD is a literal placeholder for those, not an estimate.
    nav_a = cash + sleeve_market_value
    nav_by_track, cum_returns, max_drawdowns = _track_report_inputs(execution_date, nav_a, results_dir, state_dir)
    attribution = compute_attribution(
        nav_by_track["A"], nav_by_track["B"], nav_by_track["C"], nav_by_track["D"], nav_by_track["E"],
        as_of=execution_date.isoformat(), initial_nav=NAV_TOTAL_USD,
    )
    report_md = render_report(
        signal_date=signal_date,
        execution_date=execution_date,
        regime=regime,
        nav_by_track=nav_by_track,
        monthly_a_minus_d_diffs=[],
        trades_this_month=ledger_entries,
        rank_before={p.ticker: p.entry_rank for p in new_positions},
        rank_after={p.ticker: p.current_rank for p in new_positions if p.current_rank is not None},
        current_positions=new_positions,
        max_per_sector=MAX_PER_SECTOR,
        cum_returns=cum_returns,
        differences={
            "stock_selection_effect": attribution.stock_selection_effect,
            "overlay_effect_on_stocks": attribution.overlay_effect_on_stocks,
            "overlay_effect_on_etf": attribution.overlay_effect_on_etf,
            "total_diy_benefit": attribution.total_diy_benefit,
        },
        turnover_annualized_pct=0.0,
        costs_bps=0.0,
        max_drawdown_by_track=max_drawdowns,
        months_elapsed=0,
        decision_metrics=DecisionMetrics(),
    )
    write_report(execution_date, report_md, results_dir=results_dir)

    _logger.info(
        "monthly_execute complete for %s: %d trades, %d positions",
        execution_date, len(ledger_entries), len(new_positions),
    )
    return {
        "execution_date": execution_date.isoformat(),
        "signal_date": signal_date.isoformat(),
        "trades_count": len(ledger_entries),
        "positions_count": len(new_positions),
        "warn_insufficient_cash": outcome.warn_insufficient_cash,
        "cash": cash,
    }


def _regime_for(signal_date: date, regime_history_path: Path) -> Dict:
    """Regime entry monthly_rebalance recorded for the signal date (neutral default if missing)."""
    for entry in reversed(read_regime_history(regime_history_path)):
        if entry.get("date") == signal_date.isoformat():
            return entry
    return {"scalar_applied": 1.0, "bear": False, "high_vol": False, "months_at_state": 1}


def _track_report_inputs(
    execution_date: date, nav_a: float, results_dir: Path, state_dir: Path
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    NAV per track at execution, plus returns and drawdowns from the NAV history.

    Tracks B-E come from the replay (falling back to the last nav_daily.csv row, else the
    starting NAV); Track A is today's real cash + sleeve value.
    """
    nav_csv = state_dir / "nav_daily.csv"
    history = pd.read_csv(nav_csv, parse_dates=["date"]).set_index("date") if nav_csv.exists() else pd.DataFrame()
    track_navs = replay_live_tracks(execution_date, results_dir=results_dir, regime_history_path=state_dir / "regime_history.json")
    navs = {t: float(NAV_TOTAL_USD) for t in ("B", "C", "D", "E")}
    if not history.empty:
        navs = {t: float(history[f"nav_{t.lower()}"].iloc[-1]) for t in navs}
    if track_navs is not None and not track_navs.empty:
        navs = {t: float(track_navs[f"nav_{t.lower()}"].iloc[-1]) for t in navs}
        if not history.empty:
            for column in track_navs.columns:
                history[column] = track_navs[column].reindex(history.index).fillna(history[column])
    nav_by_track = {"A": nav_a, **navs}
    today_row = pd.DataFrame([{f"nav_{t.lower()}": v for t, v in nav_by_track.items()}], index=[pd.Timestamp(execution_date)])
    frame = pd.concat([history[~history.index.isin(today_row.index)], today_row]) if not history.empty else today_row
    cum_returns, max_drawdowns = summarize_navs(frame, execution_date)
    return nav_by_track, cum_returns, max_drawdowns


def _read_prior_cash(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("cash")


def main() -> None:
    """Run monthly_execute and print scheduler result."""
    setup_run_logging()
    result = run()
    _logger.info("monthly_execute result: %s", result)
    print(f"__SCHEDULER_RESULT__:{json.dumps(result, default=str)}")


if __name__ == "__main__":
    main()
