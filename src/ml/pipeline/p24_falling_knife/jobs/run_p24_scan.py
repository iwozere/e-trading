"""
P24 Falling Knife — live scan job (spec §9).

Scans the live universe on the last completed session (daily) or week (weekly), books paper positions, writes
``results/p24_falling_knife/YYYY-MM-DD/`` (folder date = signal date) and notifies when there are new signals or paper
exits. Prints ``__SCHEDULER_RESULT__:{json}`` for the scheduler.

Examples::

    .venv/Scripts/python.exe src/ml/pipeline/p24_falling_knife/jobs/run_p24_scan.py --timeframe daily --no-notify
    .venv/Scripts/python.exe src/ml/pipeline/p24_falling_knife/jobs/run_p24_scan.py --timeframe weekly --symbols AAPL MPWR
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time as time_mod
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.ml.pipeline.p24_falling_knife.chart import DAILY_WINDOW, WEEKLY_WINDOW, plot_signal  # noqa: E402
from src.ml.pipeline.p24_falling_knife.config import (  # noqa: E402
    DAILY,
    LIVE_DIR,
    PAPER_STATE_PATH,
    WEEKLY,
    CostModel,
    LiveParams,
    UniverseParams,
    live_params,
)
from src.ml.pipeline.p24_falling_knife.data import to_weekly  # noqa: E402
from src.ml.pipeline.p24_falling_knife.exits import simulate_trade  # noqa: E402
from src.ml.pipeline.p24_falling_knife.live import fetch_daily, load_universe  # noqa: E402
from src.ml.pipeline.p24_falling_knife.notify import compose, send  # noqa: E402
from src.ml.pipeline.p24_falling_knife.paper import (  # noqa: E402
    ACTIVE,
    PaperPosition,
    add_signals,
    events_on,
    load_state,
    position_rows,
    save_state,
    update_position,
)
from src.ml.pipeline.p24_falling_knife.scan import SymbolScan, build_series, scan_symbol, signal_row  # noqa: E402
from src.notification.logger import setup_logger  # noqa: E402

_logger = setup_logger(__name__)

CALENDAR_SYMBOL = "SPY"
# Before this UTC time today's US session may still be trading (close 20:00 UTC in EDT, 21:00 UTC in EST).
SESSION_FINAL_UTC = time(21, 15)
RESULT_PREFIX = "__SCHEDULER_RESULT__:"
# A run date can be at most 4 calendar days after the last session: Monday after a Friday holiday (weekly runs on
# Saturday: 1–2 days). More means the bars did not update.
MAX_SESSION_LAG_DAYS = 4


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="P24 Falling Knife live scan")
    parser.add_argument("--timeframe", choices=[DAILY, WEEKLY], default=DAILY)
    parser.add_argument("--date", type=date.fromisoformat, default=None, help="Scan up to this date (default: today)")
    parser.add_argument("--symbols", nargs="+", default=None, help="Only these symbols (debugging)")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N universe symbols (smoke run)")
    parser.add_argument("--no-notify", action="store_true", help="Write results but send nothing")
    parser.add_argument("--state", type=Path, default=PAPER_STATE_PATH, help="Paper state JSON")
    parser.add_argument("--user-id", type=str, default=None, help="User ID injected by the scheduler")
    return parser.parse_args(argv)


def default_as_of(now: datetime) -> date:
    """
    Last date whose US session is surely closed at ``now``.

    Args:
        now: Current time (timezone-aware).

    Returns:
        Today after ``SESSION_FINAL_UTC``, else yesterday.
    """
    utc = now.astimezone(timezone.utc)
    return utc.date() if utc.time() >= SESSION_FINAL_UTC else utc.date() - timedelta(days=1)


def is_stale(last_bar: date, as_of: date) -> bool:
    """
    True when the calendar symbol's last bar is older than any market closure explains.

    Args:
        last_bar: Date of the calendar symbol's last daily bar.
        as_of: Run date.

    Returns:
        True if more than ``MAX_SESSION_LAG_DAYS`` calendar days lie between them.
    """
    return (as_of - last_bar).days > MAX_SESSION_LAG_DAYS


def session_of(calendar_daily: pd.DataFrame, timeframe: str, as_of: date) -> date:
    """
    The session a run scans: the last calendar-symbol bar (daily) or the last bar of the last complete week (weekly).

    Args:
        calendar_daily: Daily bars of the calendar symbol.
        timeframe: ``"daily"`` or ``"weekly"``.
        as_of: Run date.

    Returns:
        Session date.
    """
    if timeframe == WEEKLY:
        return to_weekly(calendar_daily, as_of).index[-1].date()
    return calendar_daily.index[-1].date()


def attach_file_log(run_dir: Path, timeframe: str) -> None:
    """Also log into ``<run_dir>/pipeline_<tf>.log``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(run_dir / f"pipeline_{timeframe}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)-8s - %(name)s - %(message)s"))
    logging.getLogger().addHandler(handler)


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def chart_signal(scan: SymbolScan, run_dir: Path, suffix: str = "", position: Optional[PaperPosition] = None,
                 params: Any = None, costs: Optional[CostModel] = None) -> Optional[Path]:
    """Chart of a signal (or of a paper position's trade) into ``<run_dir>/charts/``."""
    ctx = scan.ctx
    if position is None:
        signal = scan.signal
        title = f"{scan.symbol} {scan.timeframe} signal {ctx.index[-1].date()}"
        trade = None
    else:
        signal = next((s for s in scan.signals if ctx.index[s.t].date().isoformat() == position.signal_date), None)
        trade = simulate_trade(ctx, signal, params, costs or CostModel()) if signal is not None else None
        title = f"{scan.symbol} {scan.timeframe} exit ({position.exit_reason}) of {position.signal_date} signal"
    if signal is None:
        return None
    window = WEEKLY_WINDOW if scan.timeframe == WEEKLY else DAILY_WINDOW
    try:
        return plot_signal(ctx, signal, run_dir / "charts" / f"{scan.symbol}_{scan.timeframe}{suffix}.png", title, trade, window)
    except (ValueError, IndexError):
        _logger.exception("P24 scan: chart failed for %s", scan.symbol)
        return None


def run(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run one scan.

    Args:
        args: Parsed arguments.

    Returns:
        Scheduler result dict (``ok`` False on a fatal data problem).
    """
    started = time_mod.monotonic()
    timeframe = args.timeframe
    as_of = args.date or default_as_of(datetime.now(timezone.utc))
    params = live_params(timeframe)
    live, universe_params, costs = LiveParams(), UniverseParams(), CostModel()
    positions = load_state(args.state)
    active = [p for p in positions if p.timeframe == timeframe and p.status in ACTIVE]

    if args.symbols:
        universe = sorted({s.upper() for s in args.symbols})
    else:
        universe = load_universe(LIVE_DIR / "_universe" / as_of.isoformat())
        universe = universe[: args.limit] if args.limit else universe
    if not universe:
        return {"ok": False, "error": "empty universe", "timeframe": timeframe}
    wanted = sorted(set(universe) | {p.ticker for p in active} | {CALENDAR_SYMBOL})
    data = fetch_daily(wanted, as_of, live.lookback_days)
    if CALENDAR_SYMBOL not in data:
        return {"ok": False, "error": f"no {CALENDAR_SYMBOL} bars", "timeframe": timeframe}
    calendar_last = data[CALENDAR_SYMBOL].index[-1].date()
    if is_stale(calendar_last, as_of):
        _logger.error("P24 scan: %s bars end %s, run date %s — data is stale, nothing scanned", CALENDAR_SYMBOL,
                      calendar_last, as_of)
        return {"ok": False, "error": f"stale data: {CALENDAR_SYMBOL} ends {calendar_last}, run date {as_of}",
                "timeframe": timeframe}
    session = session_of(data[CALENDAR_SYMBOL], timeframe, as_of)
    run_dir = LIVE_DIR / session.isoformat()
    attach_file_log(run_dir, timeframe)
    _logger.info("P24 scan %s: session %s, %d universe symbols, %d with data, %d active paper positions", timeframe,
                 session, len(universe), len(data), len(active))

    scans: Dict[str, SymbolScan] = {}
    signal_rows: List[Dict[str, Any]] = []
    forming_rows: List[Dict[str, Any]] = []
    charts: List[Path] = []
    failures = 0
    for symbol in universe:
        daily = data.get(symbol)
        if daily is None:
            continue
        try:
            scan = scan_symbol(symbol, daily, timeframe, params, universe_params, session, live.forming_max_missing)
        except (ValueError, IndexError, KeyError):
            _logger.exception("P24 scan failed for %s", symbol)
            failures += 1
            continue
        if scan is None:
            continue
        scans[symbol] = scan
        if scan.signal is not None:
            signal_rows.append(signal_row(scan, scan.signal))
            path = chart_signal(scan, run_dir)
            if path is not None:
                charts.append(path)
        elif scan.forming is not None:
            forming_rows.append(signal_row(scan, scan.forming))
    signal_rows.sort(key=lambda r: -(r["score"] or 0.0))

    add_signals(positions, signal_rows, live.position_usd)
    for pos in [p for p in positions if p.timeframe == timeframe and p.status in ACTIVE]:
        scan_or_none = scans.get(pos.ticker)
        if scan_or_none is None and pos.ticker in data:
            scan_or_none = build_series(pos.ticker, data[pos.ticker], timeframe, params, universe_params, session)
        if scan_or_none is None:
            _logger.warning("P24 paper: no bars for %s, position left as is", pos.key)
            continue
        update_position(pos, scan_or_none, params, costs)
    events = events_on(positions, timeframe, session)
    for pos in events["exit_pending"] + events["exited"]:
        scan_or_none = scans.get(pos.ticker)
        if scan_or_none is not None:
            path = chart_signal(scan_or_none, run_dir, suffix="_exit", position=pos, params=params, costs=costs)
            if path is not None:
                charts.append(path)

    _write_csv(signal_rows, run_dir / f"signals_{timeframe}.csv")
    _write_csv(forming_rows, run_dir / f"forming_{timeframe}.csv")
    _write_csv(position_rows([p for p in positions if p.timeframe == timeframe and p.status in ACTIVE]),
               run_dir / f"paper_positions_{timeframe}.csv")
    _write_csv(position_rows(events["exit_pending"] + events["exited"]), run_dir / f"paper_exits_{timeframe}.csv")
    save_state(args.state, positions)

    notified = 0
    message = compose(timeframe, session, signal_rows, events["exit_pending"], events["exited"], len(forming_rows))
    if message is not None and not args.no_notify:
        notified = send(message, charts, args.user_id, live.max_charts)
    summary = {
        "ok": True,
        "timeframe": timeframe,
        "session": session.isoformat(),
        "universe": len(universe),
        "with_data": len(data),
        "scanned": len(scans),
        "scan_failures": failures,
        "signals_count": len(signal_rows),
        "forming_count": len(forming_rows),
        "entries_count": len(events["entered"]),
        "exits_count": len(events["exited"]) + len(events["exit_pending"]),
        "open_positions": sum(1 for p in positions if p.timeframe == timeframe and p.status in ACTIVE),
        "notifications_queued": notified,
        "duration_s": round(time_mod.monotonic() - started, 1),
    }
    (run_dir / f"run_summary_{timeframe}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _logger.info("P24 scan %s done: %s", timeframe, summary)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the scan, print the scheduler result line; return the exit code."""
    args = parse_args(argv)
    try:
        result = run(args)
    except Exception as exc:  # the scheduler only sees the exit code; log the cause first
        _logger.exception("P24 scan crashed")
        result = {"ok": False, "error": repr(exc), "timeframe": args.timeframe}
    sys.stdout.write(RESULT_PREFIX + json.dumps(result, default=str) + "\n")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
