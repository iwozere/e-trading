"""
P22 — deal-label entry CLI (spec §2.5, M6): turns reviewed `deal_candidate` items into `p22_deal` rows.

Interactive, human-run tool — not a scheduler job. Logic lives in `ingest/deal_entry.py`; this module
is argparse + DB glue. Two ways to work through the queue:

One at a time:
    python -m src.ml.pipeline.p22_biotech_ma.cli.deal_entry_cli list
    python -m src.ml.pipeline.p22_biotech_ma.cli.deal_entry_cli show ITEM_ID
    python -m src.ml.pipeline.p22_biotech_ma.cli.deal_entry_cli enter ITEM_ID --reviewer NAME \
        --announcement-date 2024-01-08 --deal-type strategic_acquisition --acquirer-name "Merck & Co Inc" \
        --acquirer-cik 310158 --upfront-per-share 115 --has-cvr no --status completed --completion-date 2024-03-01
    python -m src.ml.pipeline.p22_biotech_ma.cli.deal_entry_cli link ITEM_ID --reviewer NAME [--deal-id N]
    python -m src.ml.pipeline.p22_biotech_ma.cli.deal_entry_cli reject ITEM_ID --reviewer NAME --note "acquirer side"

Spreadsheet round trip (spec §3.4 allows "a CLI plus a spreadsheet export/import round-trip"):
    ... deal_entry_cli export deals.csv          # pending candidates, grouped by company, blank entry columns
    ... deal_entry_cli import deals.csv --reviewer NAME --dry-run
    ... deal_entry_cli import deals.csv --reviewer NAME

Each CSV row's `action` is `deal`, `link`, `reject` or blank (skip). `deal` rows are written first, so a
`link` row with a blank `link_deal_id` can attach to a deal entered earlier in the same file. Import is
all-or-nothing: any invalid row aborts the whole file before anything is committed.

Historical candidates (the daily job only scans the last week):
    ... deal_entry_cli scan --start 2010-01-01 --end 2026-10-29
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db.services.database_service import DatabaseService
from src.ml.pipeline.p22_biotech_ma.ingest.deal_candidates import run as run_deal_candidates
from src.ml.pipeline.p22_biotech_ma.ingest.deal_entry import (
    DEAL_STATUSES,
    DEAL_TYPES,
    DealContext,
    DealEntryError,
    entry_from_mapping,
    gather_context,
    link_candidate,
    parse_bool,
    record_deal,
)
from src.ml.pipeline.p22_biotech_ma.ingest.review_queue import reject_item
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

ENTRY_COLUMNS = [
    "action", "link_deal_id", "announcement_date", "deal_type", "acquirer_name", "acquirer_cik", "status",
    "completion_date", "upfront_per_share", "has_cvr", "cvr_max_per_share", "stock_consideration_pct",
    "target_holders_pct", "acquirer_is_private", "acquirer_has_approved_product", "ack_flags", "note",
]
CANDIDATE_COLUMNS = [
    "item_id", "company_id", "ticker", "company_name", "form_type", "filed_date", "accession_no",
    "names_in_filing", "evidence_url",
]
_ACTION_ORDER = {"deal": 0, "link": 1, "reject": 2}


class _DryRunRollback(Exception):
    """Raised inside the unit of work to roll back a dry-run import."""


def _format_context(context: DealContext) -> str:
    def pct(value: Optional[float]) -> str:
        return f"{value:+.1%}" if value is not None else "n/a"

    lines = [
        f"  premium_1d={pct(context.premium_1d)}  premium_30d={pct(context.premium_30d)}",
        f"  prior close: {context.prior_close}",
        f"  close 30d before: {context.close_30d}",
        f"  market cap day before: {context.market_cap if context.market_cap is not None else context.market_cap_reason}",
        f"  reverse splits within 90d: {context.reverse_splits or 'none'}",
        f"  flags fired: {context.flags_fired or 'none'}",
        f"  flags unchecked: {context.flags_unchecked}",
    ]
    return "\n".join(lines)


def _company_labels(uow: Any) -> Dict[int, Dict[str, Any]]:
    return {c["company_id"]: c for c in uow.p22.list_companies_full()}


def _load_candidate(uow: Any, item_id: int) -> Dict[str, Any]:
    item = uow.p22.get_review_item(item_id)
    if item is None:
        raise DealEntryError(f"No review item with id={item_id}")
    return item


def _cmd_list(_args: argparse.Namespace) -> int:
    with DatabaseService().uow() as uow:
        items = uow.p22.get_pending_review_items(item_type="deal_candidate")
        companies = _company_labels(uow)
    if not items:
        _logger.info("No pending deal candidates.")
        return 0
    for item in sorted(items, key=lambda i: (i["payload"]["company_id"], i["payload"].get("filed_date") or "")):
        payload = item["payload"]
        company = companies.get(payload["company_id"], {})
        _logger.info(
            "[%s] %s %s %s filed %s %s", item["item_id"], company.get("ticker") or "-", company.get("name", "?"),
            payload.get("form_type"), payload.get("filed_date"), payload.get("accession_no"),
        )
    _logger.info("%d pending", len(items))
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    with DatabaseService().uow() as uow:
        item = _load_candidate(uow, args.item_id)
        company = _company_labels(uow).get(item["payload"].get("company_id"), {})
        deals = uow.p22.list_deals(target_id=item["payload"]["company_id"]) if item["item_type"] == "deal_candidate" else []
    _logger.info("%s", json.dumps(item, indent=2, default=str))
    _logger.info("Company: %s %s (active=%s)", company.get("ticker"), company.get("name"), company.get("is_active"))
    _logger.info("Deals already entered for this company: %s", deals or "none")
    return 0


def _entry_fields_from_args(args: argparse.Namespace) -> Dict[str, Optional[str]]:
    return {column: getattr(args, column, None) for column in ENTRY_COLUMNS}


def _cmd_preview(args: argparse.Namespace) -> int:
    entry = entry_from_mapping(_entry_fields_from_args(args))
    with DatabaseService().uow() as uow:
        item = _load_candidate(uow, args.item_id)
        context = gather_context(uow.p22, int(item["payload"]["company_id"]), entry)
    _logger.info("%s", _format_context(context))
    return 0


def _cmd_enter(args: argparse.Namespace) -> int:
    entry = entry_from_mapping(_entry_fields_from_args(args))
    with DatabaseService().uow() as uow:
        item = _load_candidate(uow, args.item_id)
        result = record_deal(uow.p22, item, entry, entered_by=args.reviewer, acknowledge_flags=args.ack_flags)
    _logger.info("Recorded deal_id=%s from item %s", result["deal_id"], args.item_id)
    _logger.info("%s", _format_context(result["context"]))
    return 0


def _cmd_link(args: argparse.Namespace) -> int:
    with DatabaseService().uow() as uow:
        item = _load_candidate(uow, args.item_id)
        deal_id = link_candidate(uow.p22, item, args.deal_id, reviewed_by=args.reviewer, note=args.note)
    _logger.info("Linked item %s to deal_id=%s", args.item_id, deal_id)
    return 0


def _cmd_reject(args: argparse.Namespace) -> int:
    with DatabaseService().uow() as uow:
        item = _load_candidate(uow, args.item_id)
        if item["item_type"] != "deal_candidate" or item["status"] != "pending":
            raise DealEntryError(f"Item {args.item_id} is not a pending deal_candidate")
        reject_item(args.item_id, uow.p22, reviewed_by=args.reviewer, note=args.note)
    _logger.info("Rejected item %s", args.item_id)
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    with DatabaseService().uow() as uow:
        items = uow.p22.get_pending_review_items(item_type="deal_candidate")
        companies = _company_labels(uow)
    items.sort(key=lambda i: (i["payload"]["company_id"], i["payload"].get("filed_date") or "", i["item_id"]))
    with open(args.path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CANDIDATE_COLUMNS + ENTRY_COLUMNS)
        writer.writeheader()
        for item in items:
            payload = item["payload"]
            company = companies.get(payload["company_id"], {})
            writer.writerow({
                "item_id": item["item_id"],
                "company_id": payload["company_id"],
                "ticker": company.get("ticker") or "",
                "company_name": company.get("name") or "",
                "form_type": payload.get("form_type"),
                "filed_date": payload.get("filed_date"),
                "accession_no": payload.get("accession_no"),
                "names_in_filing": " | ".join(payload.get("all_names_in_filing") or []),
                "evidence_url": item.get("evidence_url") or "",
            })
    _logger.info("Exported %d pending candidates to %s", len(items), args.path)
    _logger.info("deal_type: %s; status: %s; fractions as 0.45 or 45%%", ", ".join(DEAL_TYPES), ", ".join(DEAL_STATUSES))
    return 0


def _read_import_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = [row for row in csv.DictReader(fh) if (row.get("action") or "").strip()]
    for row in rows:
        row["action"] = row["action"].strip().lower()
        if row["action"] not in _ACTION_ORDER:
            raise DealEntryError(f"Row for item {row.get('item_id')}: unknown action {row['action']!r}")
    rows.sort(key=lambda r: _ACTION_ORDER[r["action"]])
    return rows


def _apply_import_row(uow: Any, row: Dict[str, str], reviewer: str) -> str:
    item_id = int(row["item_id"])
    item = _load_candidate(uow, item_id)
    action = row["action"]
    if action == "deal":
        entry = entry_from_mapping(row)
        result = record_deal(
            uow.p22, item, entry, entered_by=reviewer, acknowledge_flags=bool(parse_bool(row.get("ack_flags")))
        )
        context = result["context"]
        return f"item {item_id}: deal_id={result['deal_id']} premium_1d={context.premium_1d} flags={context.flags_fired}"
    if action == "link":
        link_text = (row.get("link_deal_id") or "").strip()
        deal_id = link_candidate(
            uow.p22, item, int(link_text) if link_text else None, reviewed_by=reviewer, note=row.get("note") or None
        )
        return f"item {item_id}: linked to deal_id={deal_id}"
    if item["item_type"] != "deal_candidate" or item["status"] != "pending":
        raise DealEntryError(f"Item {item_id} is not a pending deal_candidate")
    reject_item(item_id, uow.p22, reviewed_by=reviewer, note=row.get("note") or None)
    return f"item {item_id}: rejected"


def _cmd_import(args: argparse.Namespace) -> int:
    rows = _read_import_rows(args.path)
    if not rows:
        _logger.info("No rows with an action — nothing to import.")
        return 0
    # Parse every `deal` row up front so a typo deep in the file fails before any DB work.
    problems = []
    for row in rows:
        if row["action"] == "deal":
            try:
                entry_from_mapping(row)
            except DealEntryError as exc:
                problems.append(f"item {row.get('item_id')}: {exc}")
    if problems:
        _logger.error("Import aborted, nothing written:\n  %s", "\n  ".join(problems))
        return 1

    messages: List[str] = []
    try:
        with DatabaseService().uow() as uow:
            for row in rows:
                messages.append(_apply_import_row(uow, row, args.reviewer))
            if args.dry_run:
                raise _DryRunRollback()
    except _DryRunRollback:
        _logger.info("Dry run — rolled back:")
    except DealEntryError as exc:
        _logger.info("%s", "\n".join(messages))
        _logger.error("Import aborted at the next row, nothing written: %s", exc)
        return 1
    _logger.info("%s", "\n".join(messages))
    return 0


def _cmd_list_deals(_args: argparse.Namespace) -> int:
    with DatabaseService().uow() as uow:
        deals = uow.p22.list_deals()
        companies = _company_labels(uow)
    for deal in deals:
        company = companies.get(deal["target_id"], {})
        flags = (deal.get("review_flags") or {}).get("fired") or []
        _logger.info(
            "[%s] %s %s %s <- %s %s %s upfront=%s premium_1d=%s flags=%s",
            deal["deal_id"], deal["announcement_date"], company.get("ticker") or "-", company.get("name"),
            deal.get("acquirer_name"), deal["deal_type"], deal.get("status"), deal.get("upfront_per_share"),
            deal.get("premium_1d"), flags,
        )
    positives = sum(1 for d in deals if d["deal_type"] == "strategic_acquisition")
    _logger.info("%d deals, %d strategic_acquisition (positive labels)", len(deals), positives)
    return 0


def _cmd_scan(args: argparse.Namespace) -> int:
    from src.data.downloader.edgar_downloader import EdgarDownloader

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else date.today()
    edgar = EdgarDownloader()
    totals = {"filings_matched": 0, "candidates_queued": 0, "already_queued": 0}
    chunk_start = start
    # One calendar year per pass keeps each EFTS query well under its result-window cap.
    while chunk_start <= end:
        chunk_end = min(date(chunk_start.year, 12, 31), end)
        with DatabaseService().uow() as uow:
            summary = run_deal_candidates(uow.p22, edgar, start_dt=str(chunk_start), end_dt=str(chunk_end))
        _logger.info("%s..%s: %s", chunk_start, chunk_end, summary)
        for key in totals:
            totals[key] += summary[key]
        chunk_start = chunk_end + timedelta(days=1)
    _logger.info("Total: %s", totals)
    return 0


def _add_entry_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("item_id", type=int)
    parser.add_argument("--announcement-date", dest="announcement_date", required=True)
    parser.add_argument("--deal-type", dest="deal_type", required=True, choices=DEAL_TYPES)
    parser.add_argument("--acquirer-name", dest="acquirer_name", required=True)
    parser.add_argument("--acquirer-cik", dest="acquirer_cik")
    parser.add_argument("--status", choices=DEAL_STATUSES)
    parser.add_argument("--completion-date", dest="completion_date")
    parser.add_argument("--upfront-per-share", dest="upfront_per_share")
    parser.add_argument("--has-cvr", dest="has_cvr")
    parser.add_argument("--cvr-max-per-share", dest="cvr_max_per_share")
    parser.add_argument("--stock-consideration-pct", dest="stock_consideration_pct", help="0-1 or with %%")
    parser.add_argument("--target-holders-pct", dest="target_holders_pct", help="pro-forma ownership kept, 0-1 or %%")
    parser.add_argument("--acquirer-is-private", dest="acquirer_is_private")
    parser.add_argument("--acquirer-has-approved-product", dest="acquirer_has_approved_product")
    parser.add_argument("--note")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P22 deal-label entry CLI (spec §2.5)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="Pending deal candidates, grouped by company").set_defaults(func=_cmd_list)

    show = sub.add_parser("show", help="One candidate's payload and the company's existing deals")
    show.add_argument("item_id", type=int)
    show.set_defaults(func=_cmd_show)

    preview = sub.add_parser("preview", help="Premiums and review flags for an entry, without writing")
    _add_entry_arguments(preview)
    preview.set_defaults(func=_cmd_preview)

    enter = sub.add_parser("enter", help="Write a p22_deal row from a candidate and confirm it")
    _add_entry_arguments(enter)
    enter.add_argument("--reviewer", required=True)
    enter.add_argument("--ack-flags", dest="ack_flags", action="store_true",
                       help="Enter a strategic_acquisition even though reverse-merger heuristics fired")
    enter.set_defaults(func=_cmd_enter)

    link = sub.add_parser("link", help="Confirm a candidate as another filing of an existing deal")
    link.add_argument("item_id", type=int)
    link.add_argument("--deal-id", dest="deal_id", type=int, default=None)
    link.add_argument("--reviewer", required=True)
    link.add_argument("--note")
    link.set_defaults(func=_cmd_link)

    reject = sub.add_parser("reject", help="Reject a candidate (not a deal, or the company is the acquirer)")
    reject.add_argument("item_id", type=int)
    reject.add_argument("--reviewer", required=True)
    reject.add_argument("--note")
    reject.set_defaults(func=_cmd_reject)

    export = sub.add_parser("export", help="Write pending candidates to a CSV for spreadsheet review")
    export.add_argument("path")
    export.set_defaults(func=_cmd_export)

    imp = sub.add_parser("import", help="Apply a reviewed CSV (all-or-nothing)")
    imp.add_argument("path")
    imp.add_argument("--reviewer", required=True)
    imp.add_argument("--dry-run", dest="dry_run", action="store_true")
    imp.set_defaults(func=_cmd_import)

    sub.add_parser("list-deals", help="Every entered deal").set_defaults(func=_cmd_list_deals)

    scan = sub.add_parser("scan", help="Queue historical candidates (the daily job only scans the last week)")
    scan.add_argument("--start", required=True)
    scan.add_argument("--end", default=None)
    scan.set_defaults(func=_cmd_scan)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except DealEntryError as exc:
        _logger.error("Deal entry failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
