"""
P20 Kestrel — Catalyst calendar sync.

Pulls earnings dates from Finnhub free tier (no premium), upserts into
k20_catalysts, fires idempotent T-10/T-3 countdown alerts per §5.1.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p20_kestrel.db.repos import (
    finish_job_run,
    get_all_upcoming_catalysts,
    get_watchlist_tickers,
    log_alert,
    stamp_catalyst_alert,
    start_job_run,
    upsert_catalyst,
)
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


def _fire_countdown_alert(catalyst: Dict[str, Any], days_out: int) -> None:
    """Fire a push alert for T-10 or T-3 catalyst countdown."""
    trigger = f"catalyst_t{days_out}"
    payload = {
        "ticker": catalyst["ticker"],
        "event_type": catalyst["event_type"],
        "event_date": str(catalyst.get("event_date", "")),
        "days_out": days_out,
    }
    _logger.info(
        "ALERT: %s %s T-%d: %s on %s",
        trigger, catalyst["ticker"], days_out, catalyst["event_type"], catalyst.get("event_date"),
    )
    log_alert(
        ticker=catalyst["ticker"],
        trigger=trigger,
        payload=payload,
        channel="push",
    )


def _check_idempotent_countdown(
    catalyst: Dict[str, Any], today: date
) -> None:
    """
    Check if T-10 or T-3 countdown alert should fire for this catalyst.

    Only fires when the column is NULL (never fired or reset by date change).
    Stamps the alert column after firing.
    """
    event_date = catalyst.get("event_date")
    if not event_date:
        return

    if isinstance(event_date, str):
        event_date = date.fromisoformat(event_date)

    days_out = (event_date - today).days

    if days_out <= 10 and catalyst.get("t10_alerted_at") is None:
        _fire_countdown_alert(catalyst, 10)
        stamp_catalyst_alert(catalyst["id"], "t10_alerted_at")

    if days_out <= 3 and catalyst.get("t3_alerted_at") is None:
        _fire_countdown_alert(catalyst, 3)
        stamp_catalyst_alert(catalyst["id"], "t3_alerted_at")


def run(as_of_date: Optional[date] = None) -> Dict[str, Any]:
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

                    upsert_catalyst({
                        "ticker": ticker,
                        "event_type": "earnings",
                        "event_date": event_date,
                        "confidence": "confirmed",
                        "source": "finnhub",
                        "notes": f"EPS est: {e.get('epsEstimate')}",
                    })
                    catalysts_upserted += 1

        # Fire countdown alerts for all upcoming catalysts
        alerts_fired = 0
        upcoming = get_all_upcoming_catalysts(days_ahead=15)
        for c in upcoming:
            if c.get("state") == "upcoming":
                _check_idempotent_countdown(c, target_date)
                alerts_fired += 1

        summary = {
            "tickers_synced": len(tickers),
            "catalysts_upserted": catalysts_upserted,
            "countdown_checks": alerts_fired,
        }
        finish_job_run(_JOB_NAME, target_date, status="ok", rows_out=catalysts_upserted)
        return summary

    except Exception as exc:
        _logger.exception("Catalyst sync failed")
        finish_job_run(_JOB_NAME, target_date, status="failed", error=str(exc))
        raise
