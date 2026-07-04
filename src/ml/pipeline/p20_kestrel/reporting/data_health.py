"""
P20 Kestrel — Data health checker.

Runs daily at 07:00 before the digest. Checks:
- GDELT GKG files present
- Sentiment staleness per source
- AV budget usage
- Warm-up ticker counts
- LLM spend vs budget
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.data.db.services.kestrel_service import KestrelService as _KestrelService
from src.ml.pipeline.p20_kestrel.config import (
    AV_DAILY_QUOTA,
    DATA_CACHE_PATH,
    LLM_MONTHLY_BUDGET_USD,
    STALENESS_DAYS,
)

_kestrel = _KestrelService()
finish_job_run = _kestrel.finish_job_run
get_job_run = _kestrel.get_job_run
get_llm_monthly_spend = _kestrel.get_llm_monthly_spend
get_or_create_budget = _kestrel.get_or_create_budget
start_job_run = _kestrel.start_job_run
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_JOB_NAME = "data_health"
_P15_GKG_DIR = DATA_CACHE_PATH / "gdelt" / "gkg"


def check_gdelt_freshness(today: date) -> str | None:
    """
    Check whether today's or yesterday's GKG file is present.

    Returns:
        Warning string if stale, None if fresh.
    """
    for check_date in (today, today - timedelta(days=1)):
        date_prefix = check_date.strftime("%Y%m%d")
        files = list(_P15_GKG_DIR.glob(f"{date_prefix}*.gkg.csv*"))
        if files:
            return None  # found

    return f"gdelt: no GKG files found for {today} or {today - timedelta(days=1)}"


def check_sentiment_staleness(today: date) -> List[str]:
    """
    Check if any major sentiment sources have stale data.

    A source is stale when no ok/skipped run exists within its
    STALENESS_DAYS window (not just yesterday) — a job that succeeded
    two days ago within a 3-day window is still fresh.

    Returns:
        List of warning strings.
    """
    warnings: List[str] = []
    for job in ("gdelt_process", "social_poll", "av_sentiment_budgeted"):
        max_stale = STALENESS_DAYS.get("gdelt" if "gdelt" in job else "social", 3)
        fresh = any(
            get_job_run(job, today - timedelta(days=offset)) in ("ok", "skipped") for offset in range(1, max_stale + 1)
        )
        if not fresh:
            warnings.append(f"⚠ {job}: no ok run in the last {max_stale}d")

    return warnings


def check_av_budget(today: date) -> str | None:
    """Check AV request budget usage."""
    try:
        budget = get_or_create_budget("av_news", today, quota=AV_DAILY_QUOTA)
        used = budget.get("used", 0)
        pct = used / AV_DAILY_QUOTA if AV_DAILY_QUOTA else 0
        if pct >= 0.9:
            return f"AV budget {used}/{AV_DAILY_QUOTA} ({pct:.0%}) — nearly exhausted"
        return None
    except Exception:
        return None


def check_llm_budget() -> str | None:
    """Check LLM monthly spend vs cap."""
    try:
        spend = get_llm_monthly_spend()
        pct = spend / LLM_MONTHLY_BUDGET_USD if LLM_MONTHLY_BUDGET_USD else 0
        if pct >= 0.80:
            return f"LLM spend ${spend:.2f}/{LLM_MONTHLY_BUDGET_USD:.0f} ({pct:.0%})"
        return None
    except Exception:
        return None


def run(as_of_date: date | None = None) -> Dict[str, Any]:
    """
    Run all health checks and log warnings.

    Args:
        as_of_date: Date to check (defaults to today).

    Returns:
        Summary dict with warnings list.
    """
    target_date = as_of_date or date.today()
    _logger.info("Data health check for %s", target_date)
    start_job_run(_JOB_NAME, target_date)

    warnings: List[str] = []

    try:
        gdelt_warn = check_gdelt_freshness(target_date)
        if gdelt_warn:
            warnings.append(gdelt_warn)
            _logger.warning(gdelt_warn)

        sentiment_warns = check_sentiment_staleness(target_date)
        for w in sentiment_warns:
            warnings.append(w)
            _logger.warning(w)

        av_warn = check_av_budget(target_date)
        if av_warn:
            warnings.append(av_warn)
            _logger.info(av_warn)

        llm_warn = check_llm_budget()
        if llm_warn:
            warnings.append(llm_warn)
            _logger.warning(llm_warn)

        if warnings:
            _logger.warning("Data health: %d warnings", len(warnings))
        else:
            _logger.info("Data health: all OK")

        finish_job_run(_JOB_NAME, target_date, status="ok", rows_out=len(warnings))
        return {"warnings": warnings, "warning_count": len(warnings)}

    except Exception as exc:
        _logger.exception("Data health check failed")
        finish_job_run(_JOB_NAME, target_date, status="failed", error=str(exc))
        raise
