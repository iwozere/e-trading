"""CLI runner for trading-strategy-pack batch jobs."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

from src.data.data_manager import DataManager
from src.notification.logger import setup_logger
from src.notification.service.client import NotificationServiceClient
from src.strategy_pack.io import DedupStore, append_signals
from src.strategy_pack.models import PackSignal
from src.strategy_pack.notify import send_pack_notifications
from src.strategy_pack.strategies import RUNNERS, RunContext

_logger = setup_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    return results_dir / f"pack_signals_{day}.jsonl"


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

    end = datetime.now(timezone.utc).replace(tzinfo=None)
    dm = DataManager()

    all_signals: List[PackSignal] = []
    for n in strategies:
        runner = RUNNERS.get(n)
        if not runner:
            _logger.error("Unknown strategy %s", n)
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

    if not args.no_notify and not args.dry_run and all_signals:
        dedup_path = results_dir / "dedup_cache.json"
        dedup = DedupStore(dedup_path)
        client = NotificationServiceClient()
        try:
            sent = await send_pack_notifications(client, all_signals, dedup)
            _logger.info("Sent %d notifications", sent)
        finally:
            await client.close()
    elif args.dry_run:
        for s in all_signals:
            _logger.info("DRY RUN %s", json.dumps(s.to_jsonl_dict()))

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
        help="Strategy number 1–6, comma-separated, or 'all' (default: 2).",
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
    args = p.parse_args(argv)

    if args.command != "run":
        p.print_help()
        return 2

    try:
        return asyncio.run(_run_async(args))
    except KeyboardInterrupt:
        return 130
    except Exception:
        _logger.exception("strategy_pack run failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
