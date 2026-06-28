"""
P19 Intraday Penny-Stock Monitor — CLI entry point (Phase 0 scaffold).

Subcommands map to the run modes in spec §5. Implementations land in later phases;
this scaffold wires the CLI, config, and scheduler contract (`--user-id` injection)
so the loop and state file can be built incrementally.

Usage:
    python run_p19.py build-watchlist
    python run_p19.py run-once --mode shadow      # Phase 1: log only
    python run_p19.py run-once --mode live        # Phase 2+: alert
    python run_p19.py eod-backfill
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.ml.pipeline.p19_penny_intraday.config import P19Config

_logger = setup_logger(__name__)


def _today() -> str:
    """Trading date for the watchlist (UTC today; pre-market build)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _not_implemented(phase: str, what: str) -> int:
    _logger.warning("P19 %s not implemented yet (%s) — scaffold only", what, phase)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="P19 intraday penny-stock monitor")
    # The scheduler appends `--user-id <id>` *after* the subcommand, so it must be
    # available on every subparser — share it via a parent parser.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--user-id", default=None, help="Scheduler-injected alert recipient")
    common.add_argument("--date", default=None, help="Trading date YYYY-MM-DD (default: today UTC)")

    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-watchlist", parents=[common], help="Build the daily watchlist (pre-market)")
    rp = sub.add_parser("run-once", parents=[common], help="One intraday poll (stateless)")
    rp.add_argument("--mode", choices=["shadow", "live"], default="shadow")
    sub.add_parser("eod-backfill", parents=[common], help="Backfill EOD OHLC into shadow rows")

    args = parser.parse_args()
    config = P19Config.create_default()
    if args.user_id:
        config.user_id = args.user_id

    if args.cmd == "build-watchlist":
        from src.ml.pipeline.p19_penny_intraday.watchlist_builder import WatchlistBuilder
        target = args.date or _today()
        summary = WatchlistBuilder(config, target).run()
        print(f"P19 watchlist {target}: {summary['count']} names {summary['sources']}")
        print(f"__SCHEDULER_RESULT__:{json.dumps(summary)}")
        return 0
    if args.cmd == "run-once":
        config.shadow_mode = args.mode == "shadow"
        if args.mode == "live":
            return _not_implemented("Phase 2", "run-once (live alerting)")
        from src.ml.pipeline.p19_penny_intraday.shadow_loop import ShadowLoop
        target = args.date or _today()
        summary = ShadowLoop(config, target).run_once()
        print(f"P19 shadow poll {target}: logged={summary.get('logged')} "
              f"of polled={summary.get('polled')}")
        print(f"__SCHEDULER_RESULT__:{json.dumps(summary)}")
        return 0
    if args.cmd == "eod-backfill":
        from src.ml.pipeline.p19_penny_intraday.shadow_loop import ShadowLoop
        target = args.date or _today()
        summary = ShadowLoop(config, target).eod_backfill()
        print(f"P19 eod-backfill {target}: {summary.get('rows_updated')} rows")
        print(f"__SCHEDULER_RESULT__:{json.dumps(summary)}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
