"""
P22 — review-queue CLI (spec §3.4: "a minimal review UI (or, acceptably for
v1, a CLI plus a spreadsheet export/import round-trip)").

Interactive, human-run tool — not a `jobs/run_*.py` scheduler script (no
`__SCHEDULER_RESULT__` output, takes argparse subcommands). Business logic
(what a confirm/reject actually does) lives in `ingest/review_queue.py` so
it's unit-testable without a terminal; this module is thin argparse + DB glue.

Usage:
    python -m src.ml.pipeline.p22_biotech_ma.cli.review_queue_cli status
    python -m src.ml.pipeline.p22_biotech_ma.cli.review_queue_cli list [--item-type entity_match]
    python -m src.ml.pipeline.p22_biotech_ma.cli.review_queue_cli show ITEM_ID
    python -m src.ml.pipeline.p22_biotech_ma.cli.review_queue_cli confirm ITEM_ID --reviewer NAME [--note TEXT]
    python -m src.ml.pipeline.p22_biotech_ma.cli.review_queue_cli reject ITEM_ID --reviewer NAME [--note TEXT]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db.services.database_service import DatabaseService
from src.ml.pipeline.p22_biotech_ma.ingest.review_queue import (
    UnknownReviewItemReasonError,
    confirm_item,
    queue_depth_report,
    reject_item,
)
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def _cmd_status(_args: argparse.Namespace) -> int:
    with DatabaseService().uow() as uow:
        pending = uow.p22.get_pending_review_items()
    report = queue_depth_report(pending, now=datetime.now(timezone.utc))
    if not report:
        print("Review queue is empty.")
        return 0
    for item_type, stats in sorted(report.items()):
        median = stats["median_age_hours"]
        oldest = stats["oldest_age_hours"]
        median_str = f"{median:.1f}h" if median is not None else "n/a"
        oldest_str = f"{oldest:.1f}h" if oldest is not None else "n/a"
        print(f"{item_type}: {stats['count']} pending, median age {median_str}, oldest {oldest_str}")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    with DatabaseService().uow() as uow:
        items = uow.p22.get_pending_review_items(item_type=args.item_type)
    if not items:
        print("No pending items.")
        return 0
    for item in items:
        reason = item["payload"].get("reason", "?")
        print(f"[{item['item_id']}] {item['item_type']} priority={item['priority']} reason={reason}")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    with DatabaseService().uow() as uow:
        item = uow.p22.get_review_item(args.item_id)
    if item is None:
        print(f"No review item with id={args.item_id}")
        return 1
    print(json.dumps(item, indent=2, default=str))
    return 0


def _cmd_confirm(args: argparse.Namespace) -> int:
    with DatabaseService().uow() as uow:
        item = uow.p22.get_review_item(args.item_id)
        if item is None:
            print(f"No review item with id={args.item_id}")
            return 1
        try:
            outcome = confirm_item(item, uow.p22, reviewed_by=args.reviewer, note=args.note)
        except UnknownReviewItemReasonError as exc:
            _logger.error("Cannot confirm item %s: %s", args.item_id, exc)
            print(f"Error: {exc}")
            return 1
    print(f"Confirmed item {args.item_id}: {outcome}")
    return 0


def _cmd_reject(args: argparse.Namespace) -> int:
    with DatabaseService().uow() as uow:
        item = uow.p22.get_review_item(args.item_id)
        if item is None:
            print(f"No review item with id={args.item_id}")
            return 1
        reject_item(args.item_id, uow.p22, reviewed_by=args.reviewer, note=args.note)
    print(f"Rejected item {args.item_id}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P22 review-queue CLI (spec §3.4)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status", help="Queue depth and median age by item_type").set_defaults(func=_cmd_status)

    list_parser = subparsers.add_parser("list", help="List pending review items")
    list_parser.add_argument("--item-type", default=None)
    list_parser.set_defaults(func=_cmd_list)

    show_parser = subparsers.add_parser("show", help="Show one review item's full payload")
    show_parser.add_argument("item_id", type=int)
    show_parser.set_defaults(func=_cmd_show)

    confirm_parser = subparsers.add_parser("confirm", help="Confirm a review item and apply its write")
    confirm_parser.add_argument("item_id", type=int)
    confirm_parser.add_argument("--reviewer", required=True)
    confirm_parser.add_argument("--note", default=None)
    confirm_parser.set_defaults(func=_cmd_confirm)

    reject_parser = subparsers.add_parser("reject", help="Reject a review item (no downstream write)")
    reject_parser.add_argument("item_id", type=int)
    reject_parser.add_argument("--reviewer", required=True)
    reject_parser.add_argument("--note", default=None)
    reject_parser.set_defaults(func=_cmd_reject)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
