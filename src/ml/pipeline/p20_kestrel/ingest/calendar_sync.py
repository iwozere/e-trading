"""
P20 Kestrel — Catalyst calendar sync.

Pulls earnings dates from Finnhub free tier (no premium), upserts into
k20_catalysts, fires idempotent T-10/T-3 countdown alerts per §5.1.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db.services.kestrel_service import KestrelService as _KestrelService

_kestrel = _KestrelService()
finish_job_run = _kestrel.finish_job_run
get_all_upcoming_catalysts = _kestrel.get_all_upcoming_catalysts
get_watchlist_tickers = _kestrel.get_watchlist_tickers
log_alert = _kestrel.log_alert
stamp_catalyst_alert = _kestrel.stamp_catalyst_alert
start_job_run = _kestrel.start_job_run
upsert_catalyst = _kestrel.upsert_catalyst
from src.ml.pipeline.p20_kestrel.notify import send_push
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_JOB_NAME = "catalyst_sync"


def _fetch_finnhub_earnings(ticker: str, api_key: str) -> List[Dict[str, Any]]:
    """
    Fetch earnings calendar from Finnhub for the given ticker.

    Args:
        ticker: Ticker symbol.
        api_key: Finnhub API key.

    Returns:
        List of earnings event dicts from Finnhub.
    """
    import requests

    try:
        today = date.today()
        url = "https://finnhub.io/api/v1/calendar/earnings"
        params = {
            "symbol": ticker,
            "from": today.isoformat(),
            "to": (today + timedelta(days=90)).isoformat(),
            "token": api_key,
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("earningsCalendar", [])
    except Exception:
        _logger.debug("Finnhub earnings fetch failed for %s", ticker)
        return []


# Shown as a legend at the bottom of the combined countdown digest so the
# event_type abbreviations (which come straight from k20_catalysts / pdufa.bio
# category names) don't need to be looked up elsewhere.
_EVENT_TYPE_LEGEND = {
    "pdufa": (
        "PDUFA date -- the FDA's statutory target date to decide (approve or issue a "
        "Complete Response Letter) on a drug application."
    ),
    "adcom": ("FDA Advisory Committee meeting -- outside experts vote/recommend ahead of a PDUFA decision."),
    "clinical_readout": ("Clinical trial data readout -- topline or interim trial results become public."),
    "earnings": "Scheduled quarterly/annual earnings release.",
}


def _record_countdown_alert(catalyst: Dict[str, Any], days_out: int) -> Dict[str, Any]:
    """Log a T-10 or T-3 catalyst countdown alert and return it for digest delivery."""
    trigger = f"catalyst_t{days_out}"
    payload = {
        "ticker": catalyst["ticker"],
        "event_type": catalyst["event_type"],
        "event_date": str(catalyst.get("event_date", "")),
        "days_out": days_out,
    }
    _logger.info(
        "ALERT: %s %s T-%d: %s on %s",
        trigger,
        catalyst["ticker"],
        days_out,
        catalyst["event_type"],
        catalyst.get("event_date"),
    )
    log_alert(
        ticker=catalyst["ticker"],
        trigger=trigger,
        payload=payload,
        channel="push",
    )
    return payload


def _send_countdown_digest(alerts: List[Dict[str, Any]]) -> None:
    """Send every fired countdown alert as a single combined push/email, fail-soft."""
    if not alerts:
        return

    lines = [
        f"T-{a['days_out']} countdown: {a['ticker']} {a['event_type']} on {a['event_date']}" for a in alerts
    ]
    event_types = sorted({a["event_type"] for a in alerts if a["event_type"] in _EVENT_TYPE_LEGEND})
    legend_lines = [f"  {event_type}: {_EVENT_TYPE_LEGEND[event_type]}" for event_type in event_types]

    body = "\n".join(lines)
    if legend_lines:
        body += "\n\nLegend:\n" + "\n".join(legend_lines)

    send_push(
        title=f"Kestrel: {len(alerts)} catalyst countdown alert{'s' if len(alerts) != 1 else ''}",
        message=body,
    )


def _check_idempotent_countdown(catalyst: Dict[str, Any], today: date, alerts: List[Dict[str, Any]]) -> None:
    """
    Check if T-10 or T-3 countdown alert should fire for this catalyst.

    Only fires when the column is NULL (never fired or reset by date change).
    Stamps the alert column after firing and appends it to `alerts` for the
    combined digest send -- individual catalysts no longer get their own
    push/email.
    """
    event_date = catalyst.get("event_date")
    if not event_date:
        return

    if isinstance(event_date, str):
        event_date = date.fromisoformat(event_date)

    days_out = (event_date - today).days

    if days_out <= 10 and catalyst.get("t10_alerted_at") is None:
        alerts.append(_record_countdown_alert(catalyst, 10))
        stamp_catalyst_alert(catalyst["id"], "t10_alerted_at")

    if days_out <= 3 and catalyst.get("t3_alerted_at") is None:
        alerts.append(_record_countdown_alert(catalyst, 3))
        stamp_catalyst_alert(catalyst["id"], "t3_alerted_at")


def run(as_of_date: date | None = None) -> Dict[str, Any]:
    """
    Sync the catalyst calendar and fire countdown alerts.

    Args:
        as_of_date: Date to run for (defaults to today).

    Returns:
        Summary dict.
    """
    import os

    target_date = as_of_date or date.today()
    _logger.info("Catalyst sync for %s", target_date)
    start_job_run(_JOB_NAME, target_date)

    try:
        finnhub_key = os.environ.get("FINNHUB_API_KEY", "")
        if not finnhub_key:
            _logger.warning("FINNHUB_API_KEY not set; skipping earnings calendar fetch")

        tickers = get_watchlist_tickers()
        catalysts_upserted = 0

        for ticker in tickers:
            if finnhub_key:
                earnings_list = _fetch_finnhub_earnings(ticker, finnhub_key)
                for e in earnings_list:
                    event_date_raw = e.get("date", "")
                    if not event_date_raw:
                        continue
                    try:
                        event_date = date.fromisoformat(event_date_raw)
                    except ValueError:
                        continue

                    upsert_catalyst(
                        {
                            "ticker": ticker,
                            "event_type": "earnings",
                            "event_date": event_date,
                            "confidence": "confirmed",
                            "source": "finnhub",
                            "notes": f"EPS est: {e.get('epsEstimate')}",
                        }
                    )
                    catalysts_upserted += 1

        # Fire countdown alerts for all upcoming catalysts, combined into one digest
        alerts_fired: List[Dict[str, Any]] = []
        upcoming = get_all_upcoming_catalysts(days_ahead=15)
        for c in upcoming:
            if c.get("state") == "upcoming":
                _check_idempotent_countdown(c, target_date, alerts_fired)
        _send_countdown_digest(alerts_fired)

        summary = {
            "tickers_synced": len(tickers),
            "catalysts_upserted": catalysts_upserted,
            "countdown_checks": len(alerts_fired),
        }
        finish_job_run(_JOB_NAME, target_date, status="ok", rows_out=catalysts_upserted)
        return summary

    except Exception as exc:
        _logger.exception("Catalyst sync failed")
        finish_job_run(_JOB_NAME, target_date, status="failed", error=str(exc))
        raise
