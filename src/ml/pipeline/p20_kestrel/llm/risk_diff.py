"""
P20 Kestrel — 10-K/Q risk factor diff.

Fetches risk factor sections from consecutive annual/quarterly filings
for watchlist tickers and generates a diff via LLM.
"""

from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db.services.kestrel_service import KestrelService as _KestrelService
from src.ml.pipeline.p20_kestrel.config import SONNET_MODEL

_kestrel = _KestrelService()
finish_job_run = _kestrel.finish_job_run
get_watchlist_tickers = _kestrel.get_watchlist_tickers
start_job_run = _kestrel.start_job_run
from src.ml.pipeline.p20_kestrel.llm.client import KestrelLLMClient
from src.ml.pipeline.p20_kestrel.llm.prompts import RISK_DIFF, SYSTEM_ANALYST
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_JOB_NAME = "llm_risk_diff"
_MAX_RISK_CHARS = 12_000
_RISK_ITEM_PATTERN = re.compile(
    r"(?i)(item\s+1a\.?\s*risk\s+factors)(.*?)(?=item\s+[12][abc]?\.|$)",
    re.DOTALL,
)


def _extract_risk_section(text: str) -> str:
    """
    Extract the risk factors section from a 10-K or 10-Q filing text.

    Args:
        text: Full filing text.

    Returns:
        Extracted risk section or first _MAX_RISK_CHARS chars of text.
    """
    m = _RISK_ITEM_PATTERN.search(text)
    if m:
        return m.group(2)[:_MAX_RISK_CHARS]
    return text[:_MAX_RISK_CHARS]


def _fetch_recent_10k_texts(ticker: str) -> List[Dict[str, Any]]:
    """
    Fetch the two most recent 10-K or 10-Q text bodies for a ticker via EDGAR submissions.

    Returns:
        List of dicts with keys: form_type, filed_date, text.
    """
    import requests

    from src.data.downloader.edgar_downloader import EdgarDownloader

    downloader = EdgarDownloader()
    filings: List[Dict[str, Any]] = []

    try:
        # Map ticker → CIK via the company tickers JSON
        tickers_path = downloader.download_company_tickers()
        import json

        with open(tickers_path, encoding="utf-8") as f:
            cik_map = json.load(f)

        cik: str | None = None
        ticker_upper = ticker.upper()
        for entry in cik_map.values():
            if str(entry.get("ticker", "")).upper() == ticker_upper:
                cik = str(entry.get("cik_str", "")).zfill(10)
                break

        if not cik:
            _logger.debug("CIK not found for %s", ticker)
            return []

        # Get recent filings from submissions
        subs_path = downloader.download_submissions(cik)
        if not subs_path or not subs_path.exists():
            return []

        with open(subs_path, encoding="utf-8") as f:
            subs = json.load(f)

        recent_filings = subs.get("filings", {}).get("recent", {})
        forms_list = recent_filings.get("form", [])
        dates_list = recent_filings.get("filingDate", [])
        accessions_list = recent_filings.get("accessionNumber", [])
        docs_list = recent_filings.get("primaryDocument", [])

        collected = 0
        for i, form_type in enumerate(forms_list):
            if form_type not in ("10-K", "10-Q"):
                continue
            if collected >= 2:
                break
            filed_date = dates_list[i] if i < len(dates_list) else ""
            accession = str(accessions_list[i]).replace("-", "") if i < len(accessions_list) else ""
            primary_doc = docs_list[i] if i < len(docs_list) else ""

            if not primary_doc:
                continue

            try:
                url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{primary_doc}"
                resp = requests.get(url, timeout=10, headers={"User-Agent": "KestrelBot akossyrev@gmail.com"})
                filings.append(
                    {
                        "form_type": form_type,
                        "filed_date": filed_date,
                        "text": resp.text[:100_000] if resp.ok else "",
                    }
                )
                collected += 1
            except Exception:
                filings.append({"form_type": form_type, "filed_date": filed_date, "text": ""})
                collected += 1

    except Exception:
        _logger.exception("Could not fetch 10-K/Q filings for %s", ticker)

    return filings


def run(as_of_date: date | None = None) -> Dict[str, Any]:
    """
    Generate risk diffs for all watchlist tickers with recent 10-K/Q filings.

    Args:
        as_of_date: Date for job tracking.

    Returns:
        Summary dict.
    """
    target_date = as_of_date or date.today()
    _logger.info("Risk diff job for %s", target_date)
    start_job_run(_JOB_NAME, target_date)

    try:
        tickers = get_watchlist_tickers()
        client = KestrelLLMClient()
        diffs_ok = 0
        red_flags_found = 0
        errors = 0

        for ticker in tickers:
            try:
                filings = _fetch_recent_10k_texts(ticker)
                if len(filings) < 2:
                    _logger.debug("Not enough filings for risk diff on %s", ticker)
                    continue

                current = filings[0]
                prev = filings[1]
                current_risks = _extract_risk_section(current["text"])
                prev_risks = _extract_risk_section(prev["text"])

                if not current_risks or not prev_risks:
                    continue

                input_ref = f"{ticker}_{current['filed_date']}_vs_{prev['filed_date']}"
                user_prompt = RISK_DIFF.format(
                    ticker=ticker,
                    current_form=current["form_type"],
                    current_date=current["filed_date"],
                    prev_form=prev["form_type"],
                    prev_date=prev["filed_date"],
                    current_risks=current_risks,
                    prev_risks=prev_risks,
                )

                result = client.call(
                    task_type="risk_diff",
                    input_ref=input_ref,
                    system_prompt=SYSTEM_ANALYST,
                    user_prompt=user_prompt,
                    model=SONNET_MODEL,
                    ticker=ticker,
                    max_tokens=1024,
                )

                if result:
                    diffs_ok += 1
                    red_flags = result.get("red_flags", [])
                    red_flags_found += len(red_flags)
                    if red_flags:
                        _logger.info("Risk diff %s: %d red flags", ticker, len(red_flags))

            except RuntimeError as exc:
                if "budget" in str(exc).lower():
                    _logger.warning("Budget stop; halting risk diffs: %s", exc)
                    break
                errors += 1
            except Exception:
                _logger.exception("Risk diff error for %s", ticker)
                errors += 1

        finish_job_run(_JOB_NAME, target_date, status="ok", rows_out=diffs_ok)
        return {
            "tickers_processed": len(tickers),
            "diffs_ok": diffs_ok,
            "red_flags_found": red_flags_found,
            "errors": errors,
        }

    except Exception as exc:
        _logger.exception("Risk diff job failed")
        finish_job_run(_JOB_NAME, target_date, status="failed", error=str(exc))
        raise
