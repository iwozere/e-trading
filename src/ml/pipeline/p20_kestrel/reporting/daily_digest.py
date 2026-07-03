"""
P20 Kestrel — Daily digest builder and sender.

Sends at 07:30 Europe/Zurich hard deadline.
Sections: regime · open positions · catalysts next 5d ·
new candidates (max 3) · sentiment anomalies · data-health warnings.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p20_kestrel.config import REVISIONS_FEED_AVAILABLE
from src.ml.pipeline.p20_kestrel.db.repos import (
    finish_job_run,
    get_catalysts_in_window,
    get_latest_signal,
    get_open_positions,
    get_watchlist,
    start_job_run,
)
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_JOB_NAME = "digest_send"
_MAX_CANDIDATES = 3


def _build_regime_line() -> str:
    """Build the regime summary line from existing signals."""
    spy_200 = get_latest_signal("SPY", "price_vs_200dma")
    vix = get_latest_signal("VIX", "close")

    spy_above = spy_200 and float(spy_200.get("value", 0)) > 0.5
    vix_val = float(vix["value"]) if vix else None

    regime = "RISK-ON" if spy_above else "RISK-OFF"
    vix_str = f"VIX {vix_val:.1f}" if vix_val is not None else "VIX n/a"
    return f"Regime: {regime} | SPY/200DMA: {'above' if spy_above else 'below'} | {vix_str}"


def _build_positions_section() -> str:
    """Build the open-positions section."""
    positions = get_open_positions()
    if not positions:
        return "Open Positions: none tracked (use /pos add to record entries)"

    lines = ["Open Positions:"]
    for p in positions:
        ticker = str(p.get("ticker", ""))
        entry_px = p.get("entry_px")
        stop_px = p.get("stop_px")
        t1_px = p.get("t1_px")

        close_sig = get_latest_signal(ticker, "close")
        close_px = float(close_sig["value"]) if close_sig else None

        if close_px is not None and entry_px:
            pnl = (close_px - float(entry_px)) / float(entry_px)
            pnl_str = f"P&L {pnl:+.1%}"
        else:
            pnl_str = "P&L n/a"

        dist_to_stop = ""
        if close_px is not None and stop_px:
            dist = (float(stop_px) - close_px) / close_px
            dist_to_stop = f" | stop {dist:+.1%} away"

        lines.append(
            f"  {ticker} ({p.get('sleeve','?')}): {pnl_str}{dist_to_stop}"
            f" | t1={t1_px or 'n/a'}"
        )

    return "\n".join(lines)


def _build_catalysts_section() -> str:
    """Build next-5-days catalyst section."""
    catalysts = get_catalysts_in_window(days_ahead=5)
    if not catalysts:
        return "Catalysts next 5d: none"

    lines = ["Catalysts next 5d:"]
    for c in catalysts:
        lines.append(
            f"  {c.get('event_date')} — {c.get('ticker')} {c.get('event_type')} "
            f"[{c.get('state','?')}]"
        )
    return "\n".join(lines)


def _build_candidates_section() -> str:
    """Build new/top candidates section (max 3)."""
    watchlist = get_watchlist()
    candidates = sorted(
        [r for r in watchlist if r.get("state") == "candidate"],
        key=lambda r: float(r.get("score") or 0),
        reverse=True,
    )[:_MAX_CANDIDATES]

    if not candidates:
        return "New Candidates: none"

    interim_tag = " ⚠ revisions:n/a" if not REVISIONS_FEED_AVAILABLE else ""
    lines = ["New Candidates:"]
    for c in candidates:
        verdict = c.get("llm_verdict") or "pending"
        thesis = c.get("thesis_short") or ""
        lines.append(
            f"  {c['ticker']} (Sleeve {c.get('sleeve','?')}): "
            f"score {c.get('score') or 0:.0f}{interim_tag} | {verdict} | {thesis[:80]}"
        )
    return "\n".join(lines)


def _build_data_health_section() -> str:
    """Build data-health warnings section."""
    warnings: List[str] = []

    if not REVISIONS_FEED_AVAILABLE:
        warnings.append("⚠ revisions feed unavailable — Sleeve A in interim mode (§4.2.1)")

    return "Data Health: " + ("; ".join(warnings) if warnings else "OK")


def build_digest(as_of_date: date) -> str:
    """
    Build the full daily digest text.

    Args:
        as_of_date: Date for the digest.

    Returns:
        Digest text as a formatted string.
    """
    sections = [
        f"=== Kestrel Daily Digest — {as_of_date} ===",
        "",
        _build_regime_line(),
        "",
        _build_positions_section(),
        "",
        _build_catalysts_section(),
        "",
        _build_candidates_section(),
        "",
        _build_data_health_section(),
        "",
        "=== End of digest ===",
    ]
    return "\n".join(sections)


def run(as_of_date: Optional[date] = None) -> Dict[str, Any]:
    """
    Build and send the daily digest.

    Args:
        as_of_date: Date for the digest (defaults to today).

    Returns:
        Summary dict.
    """
    target_date = as_of_date or date.today()
    _logger.info("Building digest for %s", target_date)
    start_job_run(_JOB_NAME, target_date)

    try:
        digest_text = build_digest(target_date)
        _logger.info("Digest built (%d chars)", len(digest_text))

        # Send via notification service
        try:
            from src.notification.service.client import NotificationServiceClient
            import asyncio
            client = NotificationServiceClient()
            asyncio.run(client.send_to_admins(
                title=f"Kestrel Daily Digest — {target_date}",
                message=digest_text,
            ))
            sent = True
        except Exception:
            _logger.exception("Failed to send digest notification")
            sent = False

        # Also write to disk for audit
        digest_file = PROJECT_ROOT / "results" / "p20_kestrel" / f"digest_{target_date}.txt"
        digest_file.parent.mkdir(parents=True, exist_ok=True)
        digest_file.write_text(digest_text, encoding="utf-8")

        finish_job_run(_JOB_NAME, target_date, status="ok", rows_out=1)
        return {"digest_chars": len(digest_text), "sent": sent}

    except Exception as exc:
        _logger.exception("Digest failed")
        finish_job_run(_JOB_NAME, target_date, status="failed", error=str(exc))
        raise
