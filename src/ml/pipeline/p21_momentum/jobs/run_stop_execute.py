"""
P21 Momentum — scheduler entry point: catastrophic-stop execution (docs/pipeline-specification.md §11).

Runs every NYSE trading day, at the open, a few minutes ahead of
monthly_execute's own open-time slot (see src/data/pipeline/specs/p21_specs.py
for the cron offset and why it matters on the one day both jobs run).

daily_mark flags a catastrophic stop (adj close < avg_cost * 0.65) into
_state/pending_stops.json but does not itself trade — spec §11 says "flag
EXIT_CATASTROPHIC_STOP, execute at next open", a different time of day than
daily_mark's own close-time run. This job is that next-open execution:

1. Read _state/pending_stops.json. Empty -> no-op (the common case, every day).
2. For each queued ticker still actually held, sell the full position at
   today's open via execution/fills.simulate_fill() directly — this is
   always a full exit, so neither the chatter threshold nor
   execute_trades()'s cash-scaling (both meant for ordinary rebalance
   trades) apply.
3. Append EXIT_CATASTROPHIC_STOP ledger entries, update current_positions.json.
4. Clear executed (and any no-longer-held, e.g. already exited via a forced
   rebalance exit) entries from the queue. A ticker with no open price today
   is left queued and retried the next trading day.

**Unconditional exit, by design** (spec §11: "the only stop in the system
... catches genuine accidents only"): the fill happens regardless of what
the price does between the close that triggered the flag and this open —
there is no re-check of the -35% threshold here. No re-entry cooldown either:
a stopped-out ticker is eligible to be selected again at any future
rebalance like any other candidate, purely on its own merits.

**Defense in depth:** if this job is ever down long enough to miss a queued
stop, run_monthly_rebalance.py also unions any still-pending entries into
its own forced_exits set, so the position is still guaranteed out at the
next rebalance rather than silently persisting forever.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p21_momentum.calendar import is_trading_day
from src.ml.pipeline.p21_momentum.config import NAV_TOTAL_USD, RESULTS_DIR, SLIPPAGE_BPS, STATE_DIR
from src.ml.pipeline.p21_momentum.data.prices import fetch_price_panel
from src.ml.pipeline.p21_momentum.execution.fills import simulate_fill
from src.ml.pipeline.p21_momentum.execution.ledger import (
    append_ledger_entries,
    read_current_positions,
    read_pending_stops,
    write_current_positions,
    write_pending_stops,
)
from src.ml.pipeline.p21_momentum.jobs.run_common import send_abort_alert, setup_run_logging
from src.ml.pipeline.p21_momentum.quality.gates import PipelineAbort
from src.ml.pipeline.p21_momentum.results.run_io import already_processed, write_stop_exits
from src.ml.pipeline.p21_momentum.schemas import LedgerEntry
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def run(
    run_date: Optional[date] = None,
    force: bool = False,
    results_dir: Path = RESULTS_DIR,
    state_dir: Path = STATE_DIR,
    current_positions_path: Optional[Path] = None,
    pending_stops_path: Optional[Path] = None,
) -> Dict:
    """
    Execute one stop_execute cycle.

    Args:
        run_date: Override "today" (for tests / manual backfill).
        force: Bypass the idempotency check.
        results_dir, state_dir: Overridable for tests (real determinism/
            round-trip checks against a tmp dir) — production callers never
            pass these.
        current_positions_path, pending_stops_path: Overridable
            independently of state_dir; default to
            state_dir / "current_positions.json" and
            state_dir / "pending_stops.json".

    Returns:
        Summary dict for __SCHEDULER_RESULT__.
    """
    today = run_date or date.today()
    if not is_trading_day(today):
        _logger.info("SKIP: %s is not an NYSE trading day", today)
        return {"skipped": True, "reason": "not_trading_day", "date": today.isoformat()}

    if already_processed(today, "stop_exits.json", results_dir=results_dir) and not force:
        _logger.info("SKIP: already processed for %s", today)
        return {"skipped": True, "reason": "already_processed", "date": today.isoformat()}

    positions_path = current_positions_path or (state_dir / "current_positions.json")
    stops_path = pending_stops_path or (state_dir / "pending_stops.json")
    try:
        return _run_stop_execute(today, results_dir, state_dir, positions_path, stops_path)
    except PipelineAbort as exc:
        _logger.error("ABORT: %s", exc)
        send_abort_alert("stop_execute", exc)
        return {"aborted": True, "check": exc.check, "context": exc.context}


def _run_stop_execute(
    today: date, results_dir: Path, state_dir: Path, positions_path: Path, pending_stops_path: Path
) -> Dict:
    pending = read_pending_stops(path=pending_stops_path)
    if not pending:
        write_stop_exits(today, [], results_dir=results_dir)
        return {"date": today.isoformat(), "exits_count": 0}

    positions = read_current_positions(path=positions_path)
    positions_by_ticker = {p.ticker: p for p in positions}

    still_held = [s for s in pending if s.ticker in positions_by_ticker]
    no_longer_held = [s for s in pending if s.ticker not in positions_by_ticker]
    for s in no_longer_held:
        _logger.info("Dropping pending stop for %s — no longer held (already exited elsewhere)", s.ticker)

    if not still_held:
        write_pending_stops([], path=pending_stops_path)
        write_stop_exits(today, [], results_dir=results_dir)
        return {"date": today.isoformat(), "exits_count": 0, "dropped_not_held": len(no_longer_held)}

    tickers_to_exit = [s.ticker for s in still_held]
    start_date = datetime.combine(today, datetime.min.time()) - timedelta(days=5)
    end_date = datetime.combine(today, datetime.min.time()) + timedelta(days=1)
    # min_coverage_pct=0.0: a missing open price here is a soft-fail (retry
    # next trading day, see the loop below), never an ABORT — this is a
    # narrow, few-ticker batch, not the full-universe fetch §13's coverage
    # gate is meant to guard.
    panel = fetch_price_panel(tickers_to_exit, start_date, end_date, min_coverage_pct=0.0)

    open_prices: Dict[str, float] = {}
    for ticker, df in panel.items():
        if df is None or df.empty:
            continue
        row = df[df["timestamp"].dt.date == today]
        if not row.empty:
            open_prices[ticker] = float(row.iloc[0]["open"])

    ledger_entries: List[LedgerEntry] = []
    exits: List[Dict] = []
    executed_tickers: set[str] = set()
    cash_delta = 0.0

    for s in still_held:
        price = open_prices.get(s.ticker)
        pos = positions_by_ticker[s.ticker]
        if price is None or price <= 0:
            _logger.warning("No open price for pending stop %s today — leaving queued for next trading day", s.ticker)
            continue

        fill, comm = simulate_fill("SELL", pos.shares, price)
        gross = fill * pos.shares
        net = gross - comm
        cash_delta += net
        executed_tickers.add(s.ticker)

        ledger_entries.append(
            LedgerEntry(
                ts=datetime.combine(today, datetime.min.time()).isoformat(),
                track="A",
                ticker=s.ticker,
                side="SELL",
                shares=pos.shares,
                ref_open=price,
                fill_price=fill,
                slippage_bps=SLIPPAGE_BPS,
                commission_usd=comm,
                gross_usd=gross,
                net_usd=net,
                reason="EXIT_CATASTROPHIC_STOP",
            )
        )
        exits.append({"ticker": s.ticker, "shares": pos.shares, "fill_price": fill, "net_usd": net})
        _logger.warning(
            "EXIT_CATASTROPHIC_STOP executed for %s: %.4f shares at %.2f (flagged %s at %.2f, avg_cost %.2f)",
            s.ticker,
            pos.shares,
            fill,
            s.flagged_date,
            s.price_at_flag,
            s.avg_cost,
        )

    if ledger_entries:
        append_ledger_entries(ledger_entries, path=state_dir / "ledger.jsonl")

    remaining_positions = [p for p in positions if p.ticker not in executed_tickers]
    sleeve_market_value = sum(p.shares * open_prices.get(p.ticker, p.avg_cost) for p in remaining_positions)
    prior_cash, prior_regime_scalar = _read_prior_state(positions_path)
    cash = (prior_cash if prior_cash is not None else 0.0) + cash_delta
    nav_total = _read_nav_total(positions_path)

    write_current_positions(
        remaining_positions,
        as_of=today,
        track="A",
        nav_total=nav_total,
        cash=cash,
        sleeve_market_value=sleeve_market_value,
        regime_scalar=prior_regime_scalar,
        path=positions_path,
    )

    # Re-queue anything that didn't get a price today; drop executed and
    # no-longer-held entries.
    still_pending = [s for s in still_held if s.ticker not in executed_tickers]
    write_pending_stops(still_pending, path=pending_stops_path)

    write_stop_exits(today, exits, results_dir=results_dir)

    _logger.info(
        "stop_execute complete for %s: %d exits, %d still queued", today, len(exits), len(still_pending)
    )
    return {
        "date": today.isoformat(),
        "exits_count": len(exits),
        "still_queued": len(still_pending),
        "dropped_not_held": len(no_longer_held),
    }


def _read_prior_state(path: Path) -> tuple[Optional[float], float]:
    """Return (cash, regime_scalar) from the existing current_positions.json, if any."""
    if not path.exists():
        return None, 1.0
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("cash"), payload.get("regime_scalar", 1.0)


def _read_nav_total(path: Path) -> float:
    """Return nav_total from the existing current_positions.json, falling back to config's constant."""
    if not path.exists():
        return NAV_TOTAL_USD
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("nav_total", NAV_TOTAL_USD)


def main() -> None:
    """Run stop_execute and print scheduler result."""
    setup_run_logging()
    result = run()
    _logger.info("stop_execute result: %s", result)
    print(f"__SCHEDULER_RESULT__:{json.dumps(result, default=str)}")


if __name__ == "__main__":
    main()
