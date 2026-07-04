"""
P20 Kestrel — Weekly report (Sunday 18:00).

Contents: performance vs SPY/QQQ, sleeve attribution, funnel stats,
2-week catalyst calendar, LLM calibration + spend, GDELT alias-precision
sample, interim-mode score overlap notice.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.data.db.services.kestrel_service import KestrelService as _KestrelService
from src.ml.pipeline.p20_kestrel.config import (
    LLM_MONTHLY_BUDGET_USD,
    REVISIONS_FEED_AVAILABLE,
)

_kestrel = _KestrelService()
finish_job_run = _kestrel.finish_job_run
get_all_upcoming_catalysts = _kestrel.get_all_upcoming_catalysts
get_llm_monthly_spend = _kestrel.get_llm_monthly_spend
get_open_positions = _kestrel.get_open_positions
get_watchlist = _kestrel.get_watchlist
start_job_run = _kestrel.start_job_run
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_JOB_NAME = "weekly_report"


def _build_funnel_stats() -> str:
    """Summarize the current watchlist funnel."""
    from collections import Counter

    watchlist = get_watchlist()
    counts = Counter(str(r.get("state", "?")) for r in watchlist)
    parts = [f"{state}: {cnt}" for state, cnt in sorted(counts.items())]
    return "Funnel: " + " | ".join(parts) if parts else "Funnel: empty"


def _build_catalyst_calendar() -> str:
    """Build a 14-day forward catalyst calendar."""
    catalysts = get_all_upcoming_catalysts(days_ahead=14)
    if not catalysts:
        return "Catalyst Calendar (14d): none"

    lines = ["Catalyst Calendar (14d):"]
    for c in catalysts:
        lines.append(f"  {c.get('event_date')} — {c.get('ticker')} {c.get('event_type')} [{c.get('state')}]")
    return "\n".join(lines)


def _build_llm_spend_summary() -> str:
    """Summarize LLM spend vs budget."""
    try:
        spend = get_llm_monthly_spend()
        pct = spend / LLM_MONTHLY_BUDGET_USD * 100 if LLM_MONTHLY_BUDGET_USD else 0
        return f"LLM spend: ${spend:.2f} / ${LLM_MONTHLY_BUDGET_USD:.0f} ({pct:.0f}% of monthly budget)"
    except Exception:
        return "LLM spend: unavailable"


def build_report(week_end: date) -> str:
    """
    Build the full weekly report.

    Args:
        week_end: Sunday date for the report.

    Returns:
        Report text.
    """
    week_start = week_end - timedelta(days=6)
    sections = [
        f"=== Kestrel Weekly Report — week of {week_start} to {week_end} ===",
        "",
    ]

    if not REVISIONS_FEED_AVAILABLE:
        sections.append("⚠ Interim mode active — Sleeve A scores renormalized to /70×100 (§4.2.1)")
        sections.append("")

    positions = get_open_positions()
    sections.append(f"Open Positions: {len(positions)}")
    sections.append("")

    sections.append(_build_funnel_stats())
    sections.append("")

    sections.append(_build_catalyst_calendar())
    sections.append("")

    sections.append(_build_llm_spend_summary())
    sections.append("")

    sections.append("=== End of weekly report ===")
    return "\n".join(sections)


def run(week_end: date | None = None) -> Dict[str, Any]:
    """
    Build and send the weekly report.

    Args:
        week_end: Sunday date for the report (defaults to today).

    Returns:
        Summary dict.
    """
    target_date = week_end or date.today()
    _logger.info("Weekly report for %s", target_date)
    start_job_run(_JOB_NAME, target_date)

    try:
        report_text = build_report(target_date)

        try:
            import asyncio

            from src.notification.service.client import NotificationServiceClient

            client = NotificationServiceClient()
            asyncio.run(
                client.send_to_admins(
                    title=f"Kestrel Weekly Report — {target_date}",
                    message=report_text,
                )
            )
            sent = True
        except Exception:
            _logger.exception("Failed to send weekly report notification")
            sent = False

        report_file = PROJECT_ROOT / "results" / "p20_kestrel" / f"weekly_{target_date}.txt"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(report_text, encoding="utf-8")

        finish_job_run(_JOB_NAME, target_date, status="ok", rows_out=1)
        return {"report_chars": len(report_text), "sent": sent}

    except Exception as exc:
        _logger.exception("Weekly report failed")
        finish_job_run(_JOB_NAME, target_date, status="failed", error=str(exc))
        raise
