"""
P20 Kestrel — 8-K / PR classifier.

Reads the LLM queue written by filings_ingest, classifies each filing,
fires invalidation push alerts where appropriate.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p20_kestrel.config import HAIKU_MODEL, RESULTS_DIR
from src.ml.pipeline.p20_kestrel.db.repos import (
    finish_job_run,
    log_alert,
    start_job_run,
)
from src.ml.pipeline.p20_kestrel.llm.client import KestrelLLMClient
from src.ml.pipeline.p20_kestrel.llm.prompts import CLASSIFY_8K, SYSTEM_ANALYST
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_JOB_NAME = "llm_classify_filings"
_LLM_QUEUE_FILE = RESULTS_DIR / "llm_queue.json"
_INVALIDATION_TRIGGER = ("invalidates", "high")  # (thesis_impact, materiality)


def _trim_filing_text(raw_text: str, max_chars: int = 8000) -> str:
    """Trim filing text to max_chars, preferring the beginning."""
    if len(raw_text) <= max_chars:
        return raw_text
    return raw_text[:max_chars] + "\n[... truncated ...]"


def _read_queue() -> List[Dict[str, Any]]:
    """Read the 8-K queue file written by filings_ingest."""
    if not _LLM_QUEUE_FILE.exists():
        _logger.warning("LLM queue file not found at %s", _LLM_QUEUE_FILE)
        return []
    try:
        with open(_LLM_QUEUE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        _logger.exception("Failed to read LLM queue")
        return []


def _fetch_filing_body(filing: Dict[str, Any]) -> Optional[str]:
    """
    Fetch the filing body text from P15 cache or EDGAR.

    Returns:
        Filing text or None if unavailable.
    """
    # Try P15 local cache first
    accession = str(filing.get("accession", "")).replace("-", "")
    ticker = str(filing.get("ticker", "")).upper()

    cache_path = Path(f"R:/data-cache/edgar/8k/{ticker}_{accession}.txt")
    if cache_path.exists():
        try:
            return cache_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass

    # Try EDGAR directly (minimal body fetch)
    try:
        import requests
        filename = str(filing.get("filename", ""))
        if not filename:
            return None
        url = f"https://www.sec.gov/Archives/edgar/{filename}"
        resp = requests.get(url, timeout=15, headers={"User-Agent": "KestrelBot akossyrev@gmail.com"})
        if resp.ok:
            return resp.text[:50_000]
    except Exception:
        pass

    return None


def run(as_of_date: Optional[date] = None) -> Dict[str, Any]:
    """
    Classify all 8-K filings in the current LLM queue.

    Args:
        as_of_date: Date for job tracking (defaults to today).

    Returns:
        Summary dict.
    """
    target_date = as_of_date or date.today()
    _logger.info("LLM 8-K classifier for %s", target_date)
    start_job_run(_JOB_NAME, target_date)

    try:
        queue = _read_queue()
        _logger.info("Processing %d filings from LLM queue", len(queue))

        client = KestrelLLMClient()
        classified = 0
        invalidations = 0
        errors = 0

        for filing in queue:
            ticker = str(filing.get("ticker", "")).upper()
            accession = str(filing.get("accession", ""))
            filed_date = str(filing.get("filed_date", ""))
            form_type = str(filing.get("form_type", "8-K"))

            try:
                body = _fetch_filing_body(filing)
                if not body:
                    _logger.debug("No body for %s / %s", ticker, accession)
                    continue

                text = _trim_filing_text(body)
                user_prompt = CLASSIFY_8K.format(
                    ticker=ticker,
                    filed_date=filed_date,
                    form_type=form_type,
                    text=text,
                )
                result = client.call(
                    task_type="classify_8k",
                    input_ref=accession or f"{ticker}_{filed_date}",
                    system_prompt=SYSTEM_ANALYST,
                    user_prompt=user_prompt,
                    model=HAIKU_MODEL,
                    ticker=ticker,
                    max_tokens=512,
                )

                if result:
                    classified += 1
                    impact = result.get("thesis_impact", "")
                    materiality = result.get("materiality", "")

                    if (impact, materiality) == _INVALIDATION_TRIGGER:
                        one_liner = result.get("one_liner", "")
                        _logger.warning("INVALIDATION: %s — %s", ticker, one_liner)
                        log_alert(
                            ticker=ticker,
                            trigger="thesis_invalidation",
                            payload=result,
                            channel="push",
                        )
                        invalidations += 1

            except RuntimeError as exc:
                if "budget" in str(exc).lower():
                    _logger.warning("LLM budget stop; halting 8-K queue: %s", exc)
                    break
                errors += 1
            except Exception:
                _logger.exception("Error classifying %s / %s", ticker, accession)
                errors += 1

        finish_job_run(_JOB_NAME, target_date, status="ok", rows_out=classified)
        return {
            "queue_size": len(queue),
            "classified": classified,
            "invalidations": invalidations,
            "errors": errors,
        }

    except Exception as exc:
        _logger.exception("8-K classifier job failed")
        finish_job_run(_JOB_NAME, target_date, status="failed", error=str(exc))
        raise
