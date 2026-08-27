"""
P20 Kestrel — Spin-off registration monitor (Sleeve B2, gap 10.2).

Detects newly-filed Form 10 / 10-12B / 10-12G registration statements (the
filing a company makes to register the stock being distributed in a
spin-off) via EDGAR's quarterly form index, and upserts a `k20_catalysts`
row per resolvable ticker so `screening/sleeve_b.py`'s `screen_b2()` -- which
reads `get_past_spinoffs()`, filtering for `event_type='spinoff'` -- has
something to find. Confirmed via production logs: B2=0 every single day
from at least 2026-08-10 through 2026-08-26, because nothing ever wrote
that event type.

Known simplification (documented, not silently dropped): `event_date` is
the Form 10 *filing* date, not the actual spin-off *distribution* date --
those are frequently weeks apart and the filing itself often doesn't state
a firm distribution date yet. `screen_b2()`'s 20-60 day entry window is
anchored on the real distribution date, so dates here are a `confidence:
"estimated"` proxy, refined when an amendment (`/A`) filing revises it.
Closing that gap for real needs the spec's "mandatory LLM Form-10 dossier"
(§8.1) to read and confirm a distribution date from filing text -- not
built here; see docs/Tasks.md.

Also a known limitation: a spin-off's ticker frequently does not exist yet
in EDGAR's company_tickers.json at initial Form 10 filing time (the entity
is still pre-listing). This job only upserts filings whose CIK already
resolves to a ticker; an unresolved filing is skipped and not retried --
given B2's 20-60 day post-spin window, most real spin-offs do have a
resolvable ticker well before that window opens, but a spin-off whose
ticker is assigned unusually late could be missed. Acceptable trade-off
for a first version; a stateful backlog retry is a possible follow-up.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db.services.kestrel_service import KestrelService as _KestrelService
from src.data.downloader.edgar_downloader import EdgarDownloader

_kestrel = _KestrelService()
finish_job_run = _kestrel.finish_job_run
start_job_run = _kestrel.start_job_run
upsert_catalyst = _kestrel.upsert_catalyst
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_JOB_NAME = "spinoff_ingest"
_AMENDMENT_SUFFIX = "/A"


class _TickerFileSource(Protocol):
    """Just the one EdgarDownloader method _build_cik_to_ticker needs — lets tests pass a lightweight fake."""

    def download_company_tickers(self) -> Path: ...


def _build_cik_to_ticker(edgar: _TickerFileSource) -> Dict[str, str]:
    """
    Build a CIK -> ticker map from EdgarDownloader's company_tickers.json cache.

    Args:
        edgar: EdgarDownloader instance (or test double) to source the cache from.

    Returns:
        Dict mapping zero-stripped CIK strings to uppercase ticker symbols.
        Empty dict if the cache file is absent or malformed.
    """
    import json

    try:
        tickers_file = edgar.download_company_tickers()
        with open(tickers_file, encoding="utf-8") as f:
            raw = json.load(f)
        return {
            str(int(v.get("cik_str", 0))): v.get("ticker", "").upper()
            for v in raw.values()
            if v.get("ticker") and v.get("cik_str")
        }
    except Exception:
        _logger.warning("Could not load company_tickers.json for CIK→ticker mapping")
        return {}


def run(as_of_date: date | None = None) -> Dict[str, Any]:
    """
    Scan yesterday's EDGAR quarterly index for new Form 10 spin-off registrations.

    Args:
        as_of_date: Date label for this run (defaults to today). The EDGAR
            scan itself always looks at (as_of_date - 1 day), matching the
            cadence filings_ingest.py already uses for 8-K/13D-G.

    Returns:
        Summary dict.
    """
    target_date = as_of_date or date.today()
    filing_date = target_date - timedelta(days=1)
    _logger.info("Spin-off ingest for %s (scanning EDGAR filings from %s)", target_date, filing_date)
    start_job_run(_JOB_NAME, target_date)

    try:
        edgar = EdgarDownloader()
        filings = edgar.download_form10_filings(as_of_date=filing_date)

        if filings.empty:
            finish_job_run(_JOB_NAME, target_date, status="ok", rows_out=0)
            return {"filings_seen": 0, "tickers_resolved": 0, "catalysts_upserted": 0}

        cik_to_ticker = _build_cik_to_ticker(edgar)
        catalysts_upserted = 0
        tickers_resolved = 0

        for _, row in filings.iterrows():
            cik = str(row["cik"]).strip()
            ticker = cik_to_ticker.get(cik)
            if not ticker:
                _logger.debug(
                    "Form 10 filing for CIK %s (%s) has no ticker in company_tickers.json yet; skipping",
                    cik,
                    row.get("entity_name"),
                )
                continue
            tickers_resolved += 1

            form_type = str(row["form_type"])
            is_amendment = form_type.endswith(_AMENDMENT_SUFFIX)

            upsert_catalyst(
                {
                    "ticker": ticker,
                    "event_type": "spinoff",
                    "event_date": row["filed_date"],
                    "confidence": "estimated",
                    "source": "edgar_form10",
                    "notes": (
                        f"{form_type} filed {row['filed_date']} ({row['entity_name']}) — "
                        "filing date used as distribution-date proxy, not confirmed"
                        + (" [amendment]" if is_amendment else "")
                    )[:500],
                }
            )
            catalysts_upserted += 1

        summary = {
            "filings_seen": len(filings),
            "tickers_resolved": tickers_resolved,
            "catalysts_upserted": catalysts_upserted,
        }
        finish_job_run(_JOB_NAME, target_date, status="ok", rows_out=catalysts_upserted)
        _logger.info(
            "Spin-off ingest complete: %d/%d filings resolved to a ticker, %d catalysts upserted",
            tickers_resolved,
            len(filings),
            catalysts_upserted,
        )
        return summary

    except Exception as exc:
        _logger.exception("Spin-off ingest failed")
        finish_job_run(_JOB_NAME, target_date, status="failed", error=str(exc))
        raise
