"""
P20 Kestrel — Candidate dossier generator.

Generates JSON dossiers for Sleeve A/B candidates with score ≥ threshold.
Stores results in k20_llm_runs and updates k20_watchlist.llm_verdict.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db.services.kestrel_service import KestrelService as _KestrelService
from src.ml.pipeline.p20_kestrel.config import (
    REVISIONS_FEED_AVAILABLE,
    SLEEVE_A_DOSSIER_THRESHOLD,
    SONNET_MODEL,
)

_kestrel = _KestrelService()
finish_job_run = _kestrel.finish_job_run
get_latest_signal = _kestrel.get_latest_signal
get_universe_row = _kestrel.get_universe_row
get_watchlist = _kestrel.get_watchlist
start_job_run = _kestrel.start_job_run
upsert_watchlist = _kestrel.upsert_watchlist
from src.ml.pipeline.p20_kestrel.llm.client import KestrelLLMClient
from src.ml.pipeline.p20_kestrel.llm.prompts import DOSSIER, SYSTEM_ANALYST
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_JOB_NAME = "llm_dossiers"


def _build_dossier_context(
    universe_row: Dict[str, Any],
) -> Dict[str, str]:
    """
    Build the context strings for a dossier prompt.

    Args:
        universe_row: Row from k20_universe.

    Returns:
        Dict with keys: filings_summary, financials_summary, sentiment_summary.
    """
    revenue_growth = universe_row.get("revenue_yoy_growth")
    gross_margin = universe_row.get("gross_margin")
    net_debt_ebitda = universe_row.get("net_debt_ebitda")

    financials = []
    if revenue_growth is not None:
        financials.append(f"Revenue YoY: {revenue_growth:.1%}")
    if gross_margin is not None:
        financials.append(f"Gross Margin: {gross_margin:.1%}")
    if net_debt_ebitda is not None:
        financials.append(f"Net Debt/EBITDA: {net_debt_ebitda:.1f}×")

    return {
        "filings_summary": "Recent 8-K and Form 4 data from EDGAR (see signals).",
        "financials_summary": "; ".join(financials) if financials else "N/A",
        "sentiment_summary": "GDELT and social sentiment per k20_sentiment_daily.",
    }


def run(as_of_date: date | None = None) -> Dict[str, Any]:
    """
    Generate dossiers for all watchlist candidates above the score threshold.

    Args:
        as_of_date: Date for job tracking (defaults to today).

    Returns:
        Summary dict.
    """
    target_date = as_of_date or date.today()
    _logger.info("LLM dossier generator for %s", target_date)
    start_job_run(_JOB_NAME, target_date)

    try:
        watchlist_rows = get_watchlist()
        candidates = [
            r
            for r in watchlist_rows
            if (r.get("score") or 0) >= SLEEVE_A_DOSSIER_THRESHOLD and r.get("state") in ("screening", "candidate")
        ]
        _logger.info("Generating dossiers for %d candidates", len(candidates))

        client = KestrelLLMClient()
        dossiers_ok = 0
        advances = 0
        rejects = 0
        errors = 0

        for candidate in candidates:
            ticker = str(candidate.get("ticker", "")).upper()
            score = int(candidate.get("score") or 0)
            sleeve = str(candidate.get("sleeve") or "A")
            input_ref = f"{ticker}_{target_date.isoformat()}"

            try:
                universe_row = get_universe_row(ticker) or {}
                ctx = _build_dossier_context(universe_row)

                mcap_b = (universe_row.get("mcap") or 0) / 1e9
                drawdown_db = get_latest_signal(ticker, "drawdown_from_2y_high")
                drawdown_val = drawdown_db if drawdown_db is not None else -0.5

                user_prompt = DOSSIER.format(
                    ticker=ticker,
                    score=score,
                    interim_mode=not REVISIONS_FEED_AVAILABLE,
                    drawdown=drawdown_val,
                    mcap_b=mcap_b,
                    sector=universe_row.get("sector") or "Unknown",
                    filings_summary=ctx["filings_summary"],
                    financials_summary=ctx["financials_summary"],
                    sentiment_summary=ctx["sentiment_summary"],
                )

                result = client.call(
                    task_type="dossier",
                    input_ref=input_ref,
                    system_prompt=SYSTEM_ANALYST,
                    user_prompt=user_prompt,
                    model=SONNET_MODEL,
                    ticker=ticker,
                    max_tokens=2048,
                )

                if result:
                    dossiers_ok += 1
                    verdict = result.get("verdict", "watch")
                    if verdict == "advance":
                        advances += 1
                    elif verdict == "reject":
                        rejects += 1

                    upsert_watchlist(
                        {
                            "ticker": ticker,
                            "sleeve": sleeve,
                            "score": score,
                            "llm_verdict": verdict,
                            "thesis_short": result.get("thesis", "")[:255],
                            "state": "candidate" if verdict in ("advance", "watch") else "rejected",
                        }
                    )

            except RuntimeError as exc:
                if "budget" in str(exc).lower():
                    _logger.warning("Budget stop; halting dossier generation: %s", exc)
                    break
                errors += 1
            except Exception:
                _logger.exception("Dossier error for %s", ticker)
                errors += 1

        finish_job_run(_JOB_NAME, target_date, status="ok", rows_out=dossiers_ok)
        return {
            "candidates": len(candidates),
            "dossiers_ok": dossiers_ok,
            "advances": advances,
            "rejects": rejects,
            "errors": errors,
        }

    except Exception as exc:
        _logger.exception("Dossier job failed")
        finish_job_run(_JOB_NAME, target_date, status="failed", error=str(exc))
        raise
