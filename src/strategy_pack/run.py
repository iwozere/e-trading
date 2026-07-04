"""CLI runner for trading-strategy-pack batch jobs.

Invocation modes:

* Directly as a script (used by ``SchedulerService._execute_data_processing_job``
  which runs ``python <script_path> <args>``)::

      python src/strategy_pack/run.py run -s 2 -v A --user-id 2

* From the shell using the same file path, from any working directory::

      python /abs/path/to/src/strategy_pack/run.py run -s 2

Both paths rely on the ``sys.path`` bootstrap at the top of this module so
imports of ``src.*`` resolve regardless of the current working directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Bootstrap sys.path so this file is runnable both as ``python -m ...`` style
# (already on the path) and as a plain script (``python src/strategy_pack/run.py``),
# which is how SchedulerService invokes data_processing jobs.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import asyncio
import json
from datetime import UTC, datetime
from typing import Any, Dict, List, Set

from src.data.data_manager import DataManager
from src.notification.logger import setup_logger
from src.notification.service.client import NotificationServiceClient
from src.strategy_pack.io import DedupStore, append_signals
from src.strategy_pack.models import PackSignal
from src.strategy_pack.notify import send_pack_notifications
from src.strategy_pack.strategies import RUNNERS, RunContext

_logger = setup_logger(__name__)


def _load_config(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_strategies(arg: str) -> List[int]:
    if arg.strip().lower() == "all":
        return [1, 2, 3, 4, 5, 6]
    out: Set[int] = set()
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    return sorted(out)


def _jsonl_path(results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    day = datetime.now(UTC).strftime("%Y%m%d")
    return results_dir / f"pack_signals_{day}.jsonl"


def _emit_scheduler_result(payload: Dict[str, Any]) -> None:
    """Print the ``__SCHEDULER_RESULT__:`` line that SchedulerService parses."""
    print(f"__SCHEDULER_RESULT__:{json.dumps(payload)}", flush=True)


async def _run_async(args: argparse.Namespace) -> int:
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    cfg = _load_config(cfg_path)

    results_dir = Path(cfg.get("results_dir", "results/signals"))
    if not results_dir.is_absolute():
        results_dir = PROJECT_ROOT / results_dir

    strategies = _parse_strategies(args.strategy)
    variant = args.variant or "A"
    user_id: int | None = args.user_id

    end = datetime.now(UTC).replace(tzinfo=None)
    dm = DataManager()

    all_signals: List[PackSignal] = []
    for n in strategies:
        runner = RUNNERS.get(n)
        if not runner:
            _logger.error("Unknown strategy %s", n)
            _emit_scheduler_result(
                {
                    "success": False,
                    "error": f"unknown strategy {n}",
                    "strategies_requested": strategies,
                    "user_id": user_id,
                }
            )
            return 2
        v = variant
        if n == 5 and args.variant:
            v = args.variant.upper()
        ctx = RunContext(dm=dm, end=end, config=cfg, variant=v)
        sigs = runner(ctx)
        _logger.info("Strategy %s produced %d signal rows", n, len(sigs))
        all_signals.extend(sigs)

    if not args.no_jsonl and not args.dry_run and all_signals:
        append_signals(_jsonl_path(results_dir), all_signals)

    notifications_sent = 0
    if not args.no_notify and not args.dry_run and all_signals:
        dedup_path = results_dir / "dedup_cache.json"
        dedup = DedupStore(dedup_path)
        client = NotificationServiceClient()
        try:
            recipient_id = str(user_id) if user_id is not None else None
            notifications_sent = await send_pack_notifications(client, all_signals, dedup, recipient_id=recipient_id)
            _logger.info("Sent %d notifications", notifications_sent)
        finally:
            await client.close()
    elif args.dry_run:
        for s in all_signals:
            _logger.info("DRY RUN %s", json.dumps(s.to_jsonl_dict()))

    notifiable = sum(1 for s in all_signals if s.notify_recommended)
    _emit_scheduler_result(
        {
            "success": True,
            "user_id": user_id,
            "strategies_requested": strategies,
            "variant": variant,
            "signal_rows": len(all_signals),
            "notifiable_signals": notifiable,
            "notifications_sent": notifications_sent,
            "dry_run": bool(args.dry_run),
        }
    )
    return 0


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run trading-strategy-pack signal jobs.")
    p.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=["run"],
        help="Only 'run' is supported for now.",
    )
    p.add_argument(
        "--strategy",
        "-s",
        default="2",
        help="Strategy number 1-6, comma-separated, or 'all' (default: 2).",
    )
    p.add_argument("--variant", "-v", default="A", help="Variant label (e.g. A/B/C for Strategy 5).")
    p.add_argument(
        "--config",
        "-c",
        default="config/strategy_pack/default.json",
        help="Path to JSON config (project-relative or absolute).",
    )
    p.add_argument("--dry-run", action="store_true", help="Log signals only; no JSONL or notifications.")
    p.add_argument("--no-notify", action="store_true", help="Write JSONL but skip notifications.")
    p.add_argument("--no-jsonl", action="store_true", help="Skip JSONL append.")
    p.add_argument(
        "--user-id",
        type=int,
        default=None,
        help=(
            "Owner user_id for this run. Auto-injected by SchedulerService from "
            "job_schedules.user_id; used as recipient_id when sending notifications."
        ),
    )
    args = p.parse_args(argv)

    if args.command != "run":
        p.print_help()
        return 2

    try:
        return asyncio.run(_run_async(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        _logger.exception("strategy_pack run failed")
        _emit_scheduler_result(
            {
                "success": False,
                "error": str(exc),
                "user_id": getattr(args, "user_id", None),
            }
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
