"""
P22 — load historical raw prices and corporate actions from the FMP vintage (M3, 2026-10-27).

Human-run, like `fmp_backfill_cli.py`, which it supersedes for prices: the vintage changes only when the
owner builds a new one, so there is nothing to schedule. Business logic lives in `ingest/fmp_vintage.py`.
Writes are idempotent (existing price keys and corporate actions are left alone) and committed per company,
so the run is safe to Ctrl-C and re-run; companies that already have FMP price rows are skipped unless
`--reload` is given.

    # Match report only — how many p22_company rows map to vintage listings, no DB writes:
    python -m src.ml.pipeline.p22_biotech_ma.cli.fmp_vintage_load_cli --dry-run

    # Small real batch first, then the full run:
    python -m src.ml.pipeline.p22_biotech_ma.cli.fmp_vintage_load_cli --limit 5
    python -m src.ml.pipeline.p22_biotech_ma.cli.fmp_vintage_load_cli
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db.services.database_service import DatabaseService
from src.data.fmp_pipeline.datamanager_adapter.factory import default_data_root
from src.data.fmp_pipeline.datamanager_adapter.fmp_store import FmpStore
from src.ml.pipeline.p22_biotech_ma.ingest.fmp_vintage import VENDOR, load_company, load_vintage_securities, match_companies
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_PROGRESS_EVERY = 50


def run(data_root: Path, *, dry_run: bool, limit: Optional[int], reload: bool, company_ids: List[int]) -> int:
    """
    Match every `p22_company` to the active vintage and load the matched companies' history.

    Returns:
        Process exit code (0 on success, 1 if any company failed to load).
    """
    store = FmpStore(data_root)
    securities = load_vintage_securities(store.dir)
    db_service = DatabaseService()
    with db_service.uow() as uow:
        companies = uow.p22.list_companies_full()
        already_loaded = set() if reload else uow.p22.get_company_ids_with_prices(VENDOR)
    if company_ids:
        companies = [c for c in companies if c["company_id"] in set(company_ids)]

    matches, unmatched = match_companies(companies, securities)
    by_method = {m: sum(1 for x in matches.values() if x.method == m) for m in ("cik", "name")}
    _logger.info(
        "FMP vintage %s: %d companies, %d matched by CIK, %d by name, %d unmatched, %d already loaded",
        store.vintage, len(companies), by_method["cik"], by_method["name"], len(unmatched),
        len(already_loaded & set(matches)),
    )
    if dry_run:
        return 0

    todo = [m for cid, m in sorted(matches.items()) if cid not in already_loaded]
    if limit is not None:
        todo = todo[:limit]
    totals = {"prices_written": 0, "bars_skipped": 0, "actions_written": 0}
    failed: List[int] = []
    for i, match in enumerate(todo, 1):
        try:
            # One unit of work per company: a failure rolls back only that company and the run goes on.
            with db_service.uow() as uow:
                result = load_company(store, uow.p22, match)
        except Exception:
            _logger.exception("FMP vintage load failed for company_id %d (securities %s)", match.company_id,
                              match.security_ids)
            failed.append(match.company_id)
            continue
        for key in totals:
            totals[key] += result[key]
        if i % _PROGRESS_EVERY == 0 or i == len(todo):
            _logger.info("FMP vintage load progress: %d/%d companies, %s, %d failed", i, len(todo), totals,
                         len(failed))

    _logger.info("FMP vintage load complete: %d companies, %s, failed=%s", len(todo), totals, failed)
    return 1 if failed else 0


def main(argv: Optional[List[str]] = None) -> int:
    """Parse arguments and run the load."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Report matches only; write nothing")
    parser.add_argument("--limit", type=int, default=None, help="Load at most N companies")
    parser.add_argument("--reload", action="store_true", help="Also process companies that already have FMP rows")
    parser.add_argument("--company-id", type=int, action="append", default=[], help="Only this company (repeatable)")
    parser.add_argument("--data-root", type=Path, default=None, help="FMP data root (default DATA_CACHE_DIR/fmp-data)")
    args = parser.parse_args(argv)
    return run(args.data_root or default_data_root(), dry_run=args.dry_run, limit=args.limit, reload=args.reload,
               company_ids=args.company_id)


if __name__ == "__main__":
    sys.exit(main())
