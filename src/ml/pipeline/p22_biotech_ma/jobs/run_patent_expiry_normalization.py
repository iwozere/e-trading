"""
P22 job — normalize landed Orange Book `products.txt`/`patent.txt` into
`p22_patent_expiry` (spec §2.3, §4.1 Block A input).

Reads the most recently landed `orange_book` raw-zone partition
(`run_orange_book_ingest.py` must have run first) and the acquirer roster
(`run_acquirer_load.py` must have run first — an applicant name has nothing
to resolve against otherwise). See `ingest.patent_expiry_normalization`'s
docstring for exactly what's normalized this pass and what's deliberately
scoped out (`exclusivity.txt`, `therapeutic_area`, `ttm_revenue_usd`, fuzzy
applicant matching).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db.services.database_service import DatabaseService
from src.ml.pipeline.p22_biotech_ma.ingest import raw_zone
from src.ml.pipeline.p22_biotech_ma.ingest.patent_expiry_normalization import (
    extract_patent_expiry_records,
    write_patent_expiry_records,
)
from src.ml.pipeline.p22_biotech_ma.jobs.run_common import setup_run_logging
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def run() -> dict:
    setup_run_logging()

    payloads_by_entity = raw_zone.read_latest_partition_with_manifest("orange_book")
    products_rows = next((p for p, m in payloads_by_entity if m.get("entity") == "products.txt"), None)
    patent_rows = next((p for p, m in payloads_by_entity if m.get("entity") == "patent.txt"), None)

    if not products_rows or not patent_rows:
        _logger.warning("No Orange Book products.txt/patent.txt landed yet — run run_orange_book_ingest.py first")
        return {"records_extracted": 0, "records_written": 0, "records_unresolved": 0}

    records = extract_patent_expiry_records(products_rows, patent_rows)

    db_service = DatabaseService()
    with db_service.uow() as uow:
        acquirer_companies = uow.p22.list_acquirer_companies()
        if not acquirer_companies:
            _logger.warning("No acquirer-roster companies on file — run run_acquirer_load.py first")
            return {"records_extracted": len(records), "records_written": 0, "records_unresolved": len(records)}

        counts = write_patent_expiry_records(records, acquirer_companies, uow.p22)

    summary = {
        "records_extracted": len(records),
        "records_written": counts["written"],
        "records_unresolved": counts["unresolved"],
    }
    _logger.info("Patent-expiry normalization complete: %s", summary)
    return summary


if __name__ == "__main__":
    result = run()
    print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")
