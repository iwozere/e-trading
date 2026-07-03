"""
P20 Kestrel — Risk checker.

Checks open positions against stop/target prices and LLM invalidation signals.
Fires push alerts for stops/targets touched and writes position updates.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.data.db.services.kestrel_service import KestrelService as _KestrelService

_kestrel = _KestrelService()
finish_job_run = _kestrel.finish_job_run
get_latest_signal = _kestrel.get_latest_signal
get_open_positions = _kestrel.get_open_positions
get_today_alerts = _kestrel.get_today_alerts
log_alert = _kestrel.log_alert
start_job_run = _kestrel.start_job_run
from src.ml.pipeline.p20_kestrel.notify import send_push
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_JOB_NAME = "risk_check"
_INTRADAY_LOSS_ALERT_PCT = -0.12  # -12% intraday fires push


def _get_current_price(ticker: str) -> Optional[float]:
    """
    Get the freshest available price for a ticker.

    Tries a delayed intraday quote via yfinance first (positions count is
    small, so one request per ticker per 30-min run is fine); falls back to
    the EOD close from k20_signals when the quote is unavailable.

    Returns:
        Latest price, or None if neither source has data.
    """
    try:
        import yfinance as yf
        px = yf.Ticker(ticker).fast_info.last_price
        if px:
            return float(px)
    except Exception:
        _logger.debug("Intraday quote unavailable for %s; using EOD close", ticker)

    return get_latest_signal(ticker, "close")


def _check_position(
    position: Dict[str, Any],
    close_price: Optional[float],
) -> Optional[Dict[str, Any]]:
    """
    Check one position against stop/target prices.

    Args:
        position: Position dict from k20_positions.
        close_price: Latest close price (may be None if unavailable).

    Returns:
        Alert dict if an alert should be fired, else None.
    """
    if close_price is None:
        return None

    ticker = str(position.get("ticker", ""))
    stop_px = position.get("stop_px")
    t1_px = position.get("t1_px")
    t2_px = position.get("t2_px")
    entry_px = position.get("entry_px")
    realized_thirds = position.get("realized_thirds", 0)

    alert: Optional[Dict[str, Any]] = None

    # Stop check
    if stop_px is not None and close_price <= float(stop_px):
        alert = {
            "trigger": "stop_hit",
            "ticker": ticker,
            "close": close_price,
            "stop_px": stop_px,
            "action": "CLOSE position — stop hit",
        }
        _logger.warning("STOP HIT: %s @ %.2f (stop: %.2f)", ticker, close_price, stop_px)

    # T1 target check
    elif t1_px is not None and realized_thirds == 0 and close_price >= float(t1_px):
        alert = {
            "trigger": "t1_target",
            "ticker": ticker,
            "close": close_price,
            "t1_px": t1_px,
            "action": "Scale out 1/3 @ T1",
        }

    # T2 target check
    elif t2_px is not None and realized_thirds == 1 and close_price >= float(t2_px):
        alert = {
            "trigger": "t2_target",
            "ticker": ticker,
            "close": close_price,
            "t2_px": t2_px,
            "action": "Scale out 1/3 @ T2",
        }

    # Intraday loss guard (vs entry)
    elif entry_px is not None and close_price < float(entry_px) * (1 + _INTRADAY_LOSS_ALERT_PCT):
        pct_loss = (close_price - float(entry_px)) / float(entry_px)
        alert = {
            "trigger": "intraday_loss",
            "ticker": ticker,
            "close": close_price,
            "entry_px": entry_px,
            "pct_loss": round(pct_loss, 4),
            "action": f"Position down {pct_loss:.1%} — review immediately",
        }

    return alert


def run(as_of_date: Optional[date] = None) -> Dict[str, Any]:
    """
    Check all open positions for risk events and fire alerts.

    Args:
        as_of_date: Date to run risk check for (defaults to today).

    Returns:
        Summary dict.
    """
    target_date = as_of_date or date.today()
    _logger.info("Risk checker for %s", target_date)
    start_job_run(_JOB_NAME, target_date)

    try:
        positions = get_open_positions()
        _logger.info("Checking %d open positions", len(positions))

        # Dedup: the job runs every 30 min — fire each (ticker, trigger)
        # at most once per day. A stop that stays breached all day should
        # page the human once, not every half hour.
        already_fired = {
            (str(a.get("ticker", "")).upper(), str(a.get("trigger", "")))
            for a in get_today_alerts()
        }

        alerts_fired = 0
        alerts_deduped = 0
        positions_checked = 0

        for pos in positions:
            ticker = str(pos.get("ticker", "")).upper()
            close_price = _get_current_price(ticker)

            alert = _check_position(pos, close_price)
            if alert:
                if (ticker, alert["trigger"]) in already_fired:
                    alerts_deduped += 1
                else:
                    log_alert(
                        ticker=ticker,
                        trigger=alert["trigger"],
                        payload=alert,
                        channel="push",
                    )
                    send_push(
                        title=f"Kestrel risk: {ticker} {alert['trigger']}",
                        message=f"{ticker} @ {alert['close']:.2f} — {alert['action']}",
                    )
                    already_fired.add((ticker, alert["trigger"]))
                    alerts_fired += 1
                    _logger.info("Alert fired: %s — %s", ticker, alert["trigger"])

            positions_checked += 1

        finish_job_run(_JOB_NAME, target_date, status="ok", rows_out=alerts_fired)
        return {
            "positions_checked": positions_checked,
            "alerts_fired": alerts_fired,
            "alerts_deduped": alerts_deduped,
        }

    except Exception as exc:
        _logger.exception("Risk check failed")
        finish_job_run(_JOB_NAME, target_date, status="failed", error=str(exc))
        raise
