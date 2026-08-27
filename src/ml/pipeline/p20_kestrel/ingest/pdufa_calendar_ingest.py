"""
P20 Kestrel — PDUFA/AdCom/clinical-readout calendar ingest (Sleeve B1, gap 10.2).

`ingest/calendar_sync.py` only ever implemented the Finnhub earnings half of
what `docs/implementation-plan.md`'s Phase 6 scoped for it — the "PDUFA:
scrape pdufa.bio ... circuit breaker; on failure, log warning, keep existing"
step was never built. Without it, `screening/sleeve_b.py`'s `screen_b1()`
filters `k20_catalysts` for `event_type in {pdufa, adcom, fda_readout,
clinical_readout}`, none of which were ever written -- confirmed via
production logs: B1=0 every single day from at least 2026-08-10 through
2026-08-26.

pdufa.bio serves its whole calendar (PDUFA dates, FDA advisory committee
meetings, and clinical trial readouts) as one static JSON document at
`/search-index.json` -- no per-ticker requests, no pagination, no
JavaScript rendering required. One fetch covers all three of sleeve_b.py's
FDA event types.
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import requests

from src.data.db.services.kestrel_service import KestrelService as _KestrelService

_kestrel = _KestrelService()
finish_job_run = _kestrel.finish_job_run
get_active_tickers = _kestrel.get_active_tickers
start_job_run = _kestrel.start_job_run
upsert_catalyst = _kestrel.upsert_catalyst
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_JOB_NAME = "pdufa_calendar_ingest"

_SEARCH_INDEX_URL = "https://www.pdufa.bio/search-index.json"
_USER_AGENT = "e-trading-research akossyrev@gmail.com"
_REQUEST_TIMEOUT_S = 20

# pdufa.bio's `y` (category) field -> sleeve_b.py's k20_catalysts event_type.
# Non-event rows ("Ticker", "Conference", "Condition" metadata entries) are
# not in this map and are skipped.
_CATEGORY_TO_EVENT_TYPE = {
    "PDUFA": "pdufa",
    "AdComm": "adcom",
    "Readout": "clinical_readout",
}

# pdufa.bio's `p` (date precision) field -> our confidence label.
# "day" is an exact date; "month"/"quarter" are placeholder mid-period dates
# (e.g. the 15th of the month) -- still directionally useful for B1's 10-90
# day entry window, but should read as an estimate, not a confirmed date.
_PRECISION_TO_CONFIDENCE = {
    "day": "confirmed",
    "month": "estimated",
    "quarter": "estimated",
}


def _fetch_pdufa_bio_index() -> List[Dict[str, Any]] | None:
    """
    Fetch pdufa.bio's full calendar JSON.

    Circuit breaker per spec (implementation-plan.md:556): on any failure, log
    a warning and return None so the caller leaves existing k20_catalysts rows
    untouched rather than wiping them.

    Returns:
        Parsed JSON list of calendar entries, or None on failure.
    """
    try:
        resp = requests.get(
            _SEARCH_INDEX_URL,
            headers={"User-Agent": _USER_AGENT},
            timeout=_REQUEST_TIMEOUT_S,
        )
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            _logger.warning("pdufa.bio search-index.json returned unexpected shape: %s", type(data))
            return None
        return data
    except Exception:
        _logger.warning("pdufa.bio fetch failed; keeping existing catalyst data", exc_info=True)
        return None


def run(as_of_date: date | None = None) -> Dict[str, Any]:
    """
    Ingest the PDUFA/AdCom/clinical-readout calendar for the tracked universe.

    Args:
        as_of_date: Date label for this run (defaults to today).

    Returns:
        Summary dict.
    """
    target_date = as_of_date or date.today()
    _logger.info("PDUFA calendar ingest for %s", target_date)
    start_job_run(_JOB_NAME, target_date)

    try:
        entries = _fetch_pdufa_bio_index()
        if entries is None:
            finish_job_run(_JOB_NAME, target_date, status="skipped", error="pdufa.bio fetch failed")
            return {"catalysts_upserted": 0, "entries_seen": 0, "status": "skipped"}

        universe = set(get_active_tickers())
        catalysts_upserted = 0
        entries_matched = 0

        for entry in entries:
            category = entry.get("y") or ""
            event_type = _CATEGORY_TO_EVENT_TYPE.get(category)
            if event_type is None:
                continue  # "Ticker"/"Conference"/"Condition" metadata rows, not events

            ticker = str(entry.get("t") or "").strip().upper()
            if not ticker or ticker not in universe:
                continue  # scope to the tracked universe, same as every other P20 ingest

            date_str = entry.get("d")
            if not date_str:
                continue
            try:
                event_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                continue
            if event_date < target_date:
                continue  # past event, not useful to B1's forward-looking window

            entries_matched += 1
            precision = entry.get("p") or ""
            confidence = _PRECISION_TO_CONFIDENCE.get(precision, "estimated")
            name = entry.get("n") or ""

            upsert_catalyst(
                {
                    "ticker": ticker,
                    "event_type": event_type,
                    "event_date": event_date,
                    "confidence": confidence,
                    "source": "pdufa.bio",
                    "notes": name[:500],
                }
            )
            catalysts_upserted += 1

        summary = {
            "entries_seen": len(entries),
            "entries_matched": entries_matched,
            "catalysts_upserted": catalysts_upserted,
            "status": "ok",
        }
        finish_job_run(_JOB_NAME, target_date, status="ok", rows_out=catalysts_upserted)
        _logger.info(
            "PDUFA calendar ingest complete: %d/%d entries matched tracked universe, %d catalysts upserted",
            entries_matched,
            len(entries),
            catalysts_upserted,
        )
        return summary

    except Exception as exc:
        _logger.exception("PDUFA calendar ingest failed")
        finish_job_run(_JOB_NAME, target_date, status="failed", error=str(exc))
        raise
