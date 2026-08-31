"""
P22 — SEC EDGAR raw-zone landing (spec §2.1).

Thin wrapper over the existing `EdgarDownloader`: fetches submissions and
XBRL company facts for a set of CIKs and lands each in the P22 raw zone,
`known_from`-stamped. `EdgarDownloader` already owns rate limiting, retries,
and the SEC User-Agent requirement (spec §2.1) — this module does not
duplicate any of that, it only adds the raw-zone landing step.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.downloader.edgar_downloader import EdgarDownloader
from src.ml.pipeline.p22_biotech_ma.config import EDGAR_USER_AGENT
from src.ml.pipeline.p22_biotech_ma.ingest import raw_zone
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def land_submissions_and_facts(ciks: List[str], downloader: EdgarDownloader | None = None) -> Dict[str, Dict[str, bool]]:
    """
    Fetch submissions history and XBRL company facts for each CIK and land
    both in the raw zone.

    Args:
        ciks: CIKs to fetch, zero-padded or not (EdgarDownloader normalizes).
        downloader: Optional shared EdgarDownloader instance (tests inject a mock).

    Returns:
        Per-CIK dict of {"submissions": bool, "company_facts": bool} indicating
        whether each landed successfully (a failed fetch after EdgarDownloader's
        own retries logs there and is skipped here, not raised).
    """
    dl = downloader or EdgarDownloader(user_agent=EDGAR_USER_AGENT)
    today = date.today()
    outcomes: Dict[str, Dict[str, bool]] = {}

    for cik in ciks:
        outcome = {"submissions": False, "company_facts": False}

        try:
            submissions = dl.load_submissions(cik)
            if submissions:
                raw_zone.write(source="sec_submissions", entity=cik, as_of_date=today, payload=submissions)
                outcome["submissions"] = True
        except Exception:
            _logger.exception("Failed to land SEC submissions for CIK %s", cik)

        try:
            facts = dl.load_company_facts(cik)
            if facts:
                raw_zone.write(source="sec_company_facts", entity=cik, as_of_date=today, payload=facts)
                outcome["company_facts"] = True
        except Exception:
            _logger.exception("Failed to land SEC company facts for CIK %s", cik)

        outcomes[cik] = outcome

    landed = sum(1 for o in outcomes.values() if o["submissions"] or o["company_facts"])
    _logger.info("Landed SEC data for %d/%d CIKs", landed, len(ciks))
    return outcomes
