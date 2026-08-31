"""
P22 job — normalize landed SEC XBRL company-facts payloads into
`p22_financial_fact` (spec §2.1, §3.1, M3 Block C input).

Reads the most recently landed `sec_company_facts` raw-zone partition
(`run_sec_ingest.py` must have run first), resolves each payload's CIK
(the raw-zone manifest's `entity`) to a `company_id` via `p22_company`
(`run_entity_resolution.py` must have run first — a CIK with no resolved
company is skipped, not guessed at), and writes the metrics in
`ingest/financial_facts.FACT_TAG_MAP`. See that module's docstring for
exactly which metrics this covers and why the list is short.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db.services.database_service import DatabaseService
from src.ml.pipeline.p22_biotech_ma.ingest import raw_zone
from src.ml.pipeline.p22_biotech_ma.ingest.financial_facts import (
    DURATION_DELTA_TAG_MAP,
    FACT_TAG_MAP,
    extract_fact_series,
    extract_quarterly_delta_series,
    write_financial_facts,
)
from src.ml.pipeline.p22_biotech_ma.jobs.run_common import setup_run_logging
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def run() -> dict:
    setup_run_logging()

    companyfacts_by_cik = raw_zone.read_latest_partition_with_manifest("sec_company_facts")
    if not companyfacts_by_cik:
        _logger.warning("No SEC company-facts payloads landed yet — run run_sec_ingest.py first")
        return {"ciks_attempted": 0, "ciks_matched": 0, "facts_written": 0}

    ciks_matched = 0
    facts_written = 0

    db_service = DatabaseService()
    with db_service.uow() as uow:
        for companyfacts, manifest in companyfacts_by_cik:
            cik = manifest.get("entity")
            if not cik or not isinstance(companyfacts, dict):
                continue

            company = uow.p22.get_company_by_cik(cik)
            if company is None:
                _logger.warning("No resolved p22_company for CIK %s — skipping (run run_entity_resolution.py?)", cik)
                continue
            ciks_matched += 1

            for metric in FACT_TAG_MAP:
                facts = extract_fact_series(companyfacts, cik, metric)
                if facts:
                    facts_written += write_financial_facts(company["company_id"], facts, uow.p22)

            for metric in DURATION_DELTA_TAG_MAP:
                delta_facts = extract_quarterly_delta_series(companyfacts, cik, metric)
                if delta_facts:
                    facts_written += write_financial_facts(company["company_id"], delta_facts, uow.p22)

    summary = {"ciks_attempted": len(companyfacts_by_cik), "ciks_matched": ciks_matched, "facts_written": facts_written}
    _logger.info("Financial-facts normalization complete: %s", summary)
    return summary


if __name__ == "__main__":
    result = run()
    print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")
