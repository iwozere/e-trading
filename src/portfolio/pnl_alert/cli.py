"""
Command-line interface for the portfolio PnL alert pipeline.

Usage::

    python -m src.portfolio.pnl_alert [--config PATH] [--dry-run] [--threshold 0.15]
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger  # noqa: E402
from src.portfolio.pnl_alert.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from src.portfolio.pnl_alert.runner import run_once, summary_to_dict  # noqa: E402

_logger = setup_logger(__name__)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m src.portfolio.pnl_alert",
        description="Daily portfolio PnL alert: IBKR + watchlist digest above a threshold.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to pipeline YAML (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Format the notification and log it but do not send",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override threshold_pct from config (e.g. 0.15 for +15%%)",
    )
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    """Load config and run once; return an exit code."""
    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        _logger.error("Config error: %s", exc)
        return 2

    summary = await run_once(
        cfg,
        dry_run=args.dry_run,
        threshold_override=args.threshold,
    )

    print(json.dumps(summary_to_dict(summary), indent=2, sort_keys=True))

    if summary.errors:
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
