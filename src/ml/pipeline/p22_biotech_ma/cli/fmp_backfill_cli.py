"""
P22 — FMP historical bulk backfill CLI (spec §2.0.6/§2.4, M3, 2026-08-31).

Human-run, ONE-TIME tool for the window a Premium-tier FMP subscription is
active — not a `jobs/run_*.py` scheduler script (no `__SCHEDULER_RESULT__`
output, takes argparse flags a human tunes interactively, same reasoning as
`review_queue_cli.py`). Business logic lives in `ingest/fmp_backfill.py`;
this is thin CLI glue.

**Recommended order of operations for the Premium month (the user's plan:
buy ~2026-09-15 for one month) — several pieces here are genuinely
unverified against a real Premium key (see `ingest/fmp_client.py`'s
docstring):**

    # 1. Smoke-test the name-search endpoint first — cheap, fast, and tells
    #    you immediately if that endpoint needs updating before it wastes
    #    quota across hundreds of unresolved companies:
    python -m src.ml.pipeline.p22_biotech_ma.cli.fmp_backfill_cli test-search "Moderna"

    # 2. Smoke-test ratios/analyst-estimates against the SPECIFIC tickers
    #    live-verified 402ing on the current (non-Premium) tier — confirms
    #    the upgrade actually resolved that entitlement gap before spending
    #    a full run on it (docs/Tasks.md items 9-10):
    python -m src.ml.pipeline.p22_biotech_ma.cli.fmp_backfill_cli test-fundamentals AMGN GILD SRPT MRK

    # 3. Dry run: see the target universe size without calling FMP for prices:
    python -m src.ml.pipeline.p22_biotech_ma.cli.fmp_backfill_cli backfill --dry-run

    # 4. Small real batch — also confirms whether `close` looks raw or
    #    already-adjusted for a name you can sanity-check by eye (e.g. a
    #    company with a known recent split):
    python -m src.ml.pipeline.p22_biotech_ma.cli.fmp_backfill_cli backfill --limit 5

    # 5. Full price-history run. Safe to Ctrl-C and re-run later — already-
    #    landed tickers are skipped (ingest/fmp_backfill.py's
    #    `skip_already_landed`). At 5 req/s (config.FMP_RATE_LIMIT_RPS) a
    #    universe of a few thousand tickers is on the order of an hour, not
    #    a month — re-run this every few days over the month rather than
    #    treating it as one make-or-break session (newly name-search-
    #    resolved tickers and transient failures both self-heal on a re-run):
    python -m src.ml.pipeline.p22_biotech_ma.cli.fmp_backfill_cli backfill

    # 6. Fundamentals (ratios + analyst-estimates) run, same resumability:
    python -m src.ml.pipeline.p22_biotech_ma.cli.fmp_backfill_cli backfill-fundamentals
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db.services.database_service import DatabaseService
from src.ml.pipeline.p22_biotech_ma.ingest.fmp_backfill import (
    build_backfill_targets,
    land_historical_prices,
    land_ratios_and_estimates,
)
from src.ml.pipeline.p22_biotech_ma.ingest.fmp_client import FMPClient


def _cmd_test_search(args: argparse.Namespace) -> int:
    with FMPClient() as client:
        results = client.search_company_by_name(args.query)
    if not results:
        print(
            f"No results for {args.query!r} — could be a genuinely obscure name, or the endpoint "
            "shape has changed (see ingest/fmp_client.py's docstring). Try a well-known name "
            '(e.g. "Moderna") to distinguish the two before trusting this endpoint at all.'
        )
        return 1
    print(f"{len(results)} result(s) for {args.query!r}:")
    for r in results[:10]:
        print(f"  {r}")
    return 0


def _cmd_test_fundamentals(args: argparse.Namespace) -> int:
    """Smoke-test /stable/ratios + /stable/analyst-estimates against specific tickers — meant for
    re-checking the exact symbols live-verified 402ing on the pre-Premium tier (AMGN, GILD, SRPT,
    MRK; docs/Tasks.md items 9-10) once the account is upgraded, before committing to a full run."""
    any_failed = False
    with FMPClient() as client:
        for ticker in args.tickers:
            ratios = client.fetch_ratios(ticker)
            estimates = client.fetch_analyst_estimates(ticker, period="annual")
            ratios_ok = "OK" if ratios else "402/404/empty"
            estimates_ok = "OK" if estimates else "402/404/empty"
            print(f"  {ticker}: ratios={ratios_ok}  analyst-estimates={estimates_ok}")
            if not ratios and not estimates:
                any_failed = True
    if any_failed:
        print("At least one ticker got nothing from either endpoint — check the account's plan/entitlement.")
    return 1 if any_failed else 0


def _cmd_backfill(args: argparse.Namespace) -> int:
    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date) if args.end_date else date.today()

    with DatabaseService().uow() as uow:
        assembly = build_backfill_targets(uow.p22, include_unresolved=not args.known_only)

    targets = assembly["targets"]
    if args.limit:
        targets = targets[: args.limit]

    print(
        f"Universe: {len(targets)} target(s) to fetch "
        f"({assembly['resolved_via_search']} resolved via name search, "
        f"{len(assembly['still_unresolved'])} still unresolved)"
    )
    unresolved = assembly["still_unresolved"]
    for company in unresolved[:20]:
        print(f"  UNRESOLVED (needs manual ticker lookup): cik={company.cik} name={company.name!r}")
    if len(unresolved) > 20:
        print(f"  ... and {len(unresolved) - 20} more (see log)")

    if args.dry_run:
        print("(--dry-run: not calling FMP for price data)")
        return 0

    result = land_historical_prices(
        targets,
        start_date=start_date,
        end_date=end_date,
        skip_already_landed=not args.force,
    )
    print(
        f"Done: landed={result['landed']} skipped_already_landed={result['skipped_already_landed']} "
        f"failed={len(result['failed'])}"
    )
    if result["failed"]:
        shown = result["failed"][:30]
        suffix = "..." if len(result["failed"]) > 30 else ""
        print(f"  Failed tickers: {shown}{suffix}")
    return 0


def _cmd_backfill_fundamentals(args: argparse.Namespace) -> int:
    with DatabaseService().uow() as uow:
        assembly = build_backfill_targets(uow.p22, include_unresolved=not args.known_only)

    targets = assembly["targets"]
    if args.limit:
        targets = targets[: args.limit]

    print(f"Universe: {len(targets)} target(s) to fetch ratios + analyst-estimates for")

    if args.dry_run:
        print("(--dry-run: not calling FMP)")
        return 0

    result = land_ratios_and_estimates(targets, skip_already_landed=not args.force)
    print(
        f"Done: landed_ratios={result['landed_ratios']} landed_estimates={result['landed_estimates']} "
        f"skipped_already_landed={result['skipped_already_landed']} failed={len(result['failed'])}"
    )
    if result["failed"]:
        shown = result["failed"][:30]
        suffix = "..." if len(result["failed"]) > 30 else ""
        print(f"  Failed tickers (both endpoints empty — likely still not entitled): {shown}{suffix}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="P22 FMP historical bulk backfill (run during a Premium month)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_test = subparsers.add_parser("test-search", help="Smoke-test the (unverified) FMP name-search endpoint")
    p_test.add_argument("query", help="Company name to search for, e.g. 'Moderna'")
    p_test.set_defaults(func=_cmd_test_search)

    p_backfill = subparsers.add_parser("backfill", help="Land full historical price JSON for the target universe")
    p_backfill.add_argument("--limit", type=int, default=None, help="Cap the number of tickers (for a test run)")
    p_backfill.add_argument(
        "--known-only", action="store_true",
        help="Skip the unresolved-CIK name-search step; only fetch companies that already have a ticker on file",
    )
    p_backfill.add_argument("--start-date", default="2000-01-01", help="YYYY-MM-DD, default 2000-01-01 (ask wide)")
    p_backfill.add_argument("--end-date", default=None, help="YYYY-MM-DD, default today")
    p_backfill.add_argument("--dry-run", action="store_true", help="Build the target list; don't call FMP for prices")
    p_backfill.add_argument(
        "--force", action="store_true",
        help="Re-fetch even tickers already landed in a prior run (default: skip them)",
    )
    p_backfill.set_defaults(func=_cmd_backfill)

    p_test_fund = subparsers.add_parser(
        "test-fundamentals", help="Smoke-test /stable/ratios + /stable/analyst-estimates against specific tickers"
    )
    p_test_fund.add_argument("tickers", nargs="+", help="Tickers to check, e.g. AMGN GILD SRPT MRK")
    p_test_fund.set_defaults(func=_cmd_test_fundamentals)

    p_backfill_fund = subparsers.add_parser(
        "backfill-fundamentals", help="Land ratios + analyst-estimates JSON for the target universe (Block A inputs)"
    )
    p_backfill_fund.add_argument("--limit", type=int, default=None, help="Cap the number of tickers (for a test run)")
    p_backfill_fund.add_argument(
        "--known-only", action="store_true",
        help="Skip the unresolved-CIK name-search step; only fetch companies that already have a ticker on file",
    )
    p_backfill_fund.add_argument("--dry-run", action="store_true", help="Build the target list; don't call FMP")
    p_backfill_fund.add_argument(
        "--force", action="store_true",
        help="Re-fetch even tickers already landed in a prior run (default: skip them)",
    )
    p_backfill_fund.set_defaults(func=_cmd_backfill_fundamentals)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
