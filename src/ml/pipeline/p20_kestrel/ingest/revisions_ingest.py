"""
P20 Kestrel — Sleeve A revisions feed ingest (gap 10.1, §4.2.1).

Populates `revisions_score` (plus its raw component signals) in `k20_signals`
for watchlist ∪ positions tickers, blending three analyst signals that are all
usable from the first run:

- Finnhub recommendation-trend momentum (Finnhub returns several months of
  history in one call).
- Yahoo analyst rating-change net count, trailing 60d (yfinance
  `upgrades_downgrades` returns dated historical actions in one call).
- Yahoo forward-EPS-consensus 60-day delta (yfinance `eps_trend` carries the
  consensus as it stood 60 days ago, so no snapshot warm-up is needed — the
  literal "forward-EPS revisions up 60d" signal named in §4.2).

Yahoo replaced FMP on 2026-11-08: the FMP free plan restricts both
`analyst-estimates` and `grades` to ~87 sample symbols, so most of the
watchlist never got FMP data.

`REVISIONS_FEED_AVAILABLE` in config.py stays False (shadow mode: this job
writes signals, but sleeve_a.py's scoring ignores them) until the values have
been reviewed for a few weeks. See docs/Design.md for the rollout plan.
"""

from __future__ import annotations

import math
import os
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

from src.data.db.services.kestrel_service import KestrelService as _KestrelService
from src.data.downloader.yahoo_data_downloader import get_shared_session
from src.ml.pipeline.p20_kestrel.ingest.finnhub_client import FinnhubThrottle, finnhub_get
from src.ml.pipeline.p20_kestrel.config import (
    REVISIONS_EPS_DELTA_FULL_CREDIT_PCT,
    REVISIONS_EPS_DELTA_MAX_PTS,
    REVISIONS_FINNHUB_FULL_CREDIT_MOMENTUM,
    REVISIONS_FINNHUB_MAX_PTS,
    REVISIONS_GRADES_FULL_CREDIT_NET,
    REVISIONS_GRADES_MAX_PTS,
    REVISIONS_GRADES_WINDOW_DAYS,
)

_kestrel = _KestrelService()
finish_job_run = _kestrel.finish_job_run
get_open_positions = _kestrel.get_open_positions
get_watchlist_tickers = _kestrel.get_watchlist_tickers
start_job_run = _kestrel.start_job_run
upsert_signals = _kestrel.upsert_signals
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_JOB_NAME = "revisions_ingest"
_SLEEVE = "A"

_FINNHUB_RECOMMENDATION_URL = "https://finnhub.io/api/v1/stock/recommendation"

_SIGNAL_EPS_NEXT_FY = "yahoo_eps_next_fy"
_SIGNAL_EPS_DELTA_60D = "yahoo_eps_delta_60d_pct"
_SIGNAL_GRADES_NET_60D = "yahoo_grade_net_60d"
_SIGNAL_FINNHUB_MOMENTUM = "finnhub_rec_momentum"
_SIGNAL_REVISIONS_SCORE = "revisions_score"

_UPSERT_CHUNK_TICKERS = 50  # flush signals every N tickers so a timeout keeps the work already done
# Finnhub pacing lives in finnhub_client: without it the loop hit the 60/min limit (2026-11-05 kept 77/375).
# The two Yahoo calls per ticker run inside that 1.1s Finnhub cadence, so Yahoo sees < 2 calls/s.

# eps_trend rows: next fiscal year first (the forward consensus), current fiscal year as fallback.
_EPS_TREND_PERIODS = ("+1y", "0y")
_YAHOO_MAX_CONSECUTIVE_ERRORS = 20


class _YahooGate:
    """
    Run-scoped circuit breaker for Yahoo.

    Trips on a rate-limit error, or after ``_YAHOO_MAX_CONSECUTIVE_ERRORS``
    failed calls in a row (a blocked session fails every call), so a Yahoo
    outage costs a few seconds instead of ~750 failing calls.
    An empty answer (ticker without analyst coverage) is not an error.
    """

    def __init__(self) -> None:
        self.closed = False
        self._consecutive_errors = 0

    def note_ok(self) -> None:
        """Reset the consecutive-error counter after a successful call."""
        self._consecutive_errors = 0

    def note_error(self, what: str, rate_limited: bool = False) -> None:
        """Count a failed call; trip the breaker on a rate limit or too many failures in a row."""
        self._consecutive_errors += 1
        if self.closed:
            return
        if rate_limited or self._consecutive_errors >= _YAHOO_MAX_CONSECUTIVE_ERRORS:
            self.closed = True
            _logger.warning(
                "Yahoo %s failing (%s); skipping Yahoo for the rest of this run",
                what,
                "rate limited" if rate_limited else f"{self._consecutive_errors} errors in a row",
            )


def _get_target_tickers() -> List[str]:
    """Union of watchlist + positions tickers (same scope as filings_ingest)."""
    watchlist = set(get_watchlist_tickers())
    positions = {p["ticker"] for p in get_open_positions()}
    return sorted(watchlist | positions)


def _yahoo_frame(ticker: str, attribute: str, gate: _YahooGate | None) -> pd.DataFrame | None:
    """
    Read one yfinance analyst DataFrame property through the shared session.

    Args:
        ticker: Ticker symbol.
        attribute: ``yf.Ticker`` property name (``eps_trend`` or ``upgrades_downgrades``).
        gate: Optional Yahoo circuit breaker; when tripped the call is skipped.

    Returns:
        The DataFrame (possibly empty when Yahoo has no coverage), or None on failure / tripped gate.
    """
    if gate is not None and gate.closed:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # yfinance 1.7 emits pandas deprecation warnings on every call
            frame = getattr(yf.Ticker(ticker, session=get_shared_session()), attribute)
    except YFRateLimitError:
        if gate is not None:
            gate.note_error(attribute, rate_limited=True)
        return None
    except Exception:
        _logger.debug("Yahoo %s fetch failed for %s", attribute, ticker, exc_info=True)
        if gate is not None:
            gate.note_error(attribute)
        return None
    if gate is not None:
        gate.note_ok()
    return frame if isinstance(frame, pd.DataFrame) else None


def _finite(value: Any) -> float | None:
    """Return ``value`` as a float, or None when missing / NaN."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _parse_eps_trend(frame: pd.DataFrame | None) -> tuple[float | None, float | None]:
    """
    Extract forward consensus EPS and its 60-day relative change from a yfinance ``eps_trend`` frame.

    Args:
        frame: ``eps_trend`` DataFrame indexed by period (``0q``, ``+1q``, ``0y``, ``+1y``) with
            ``current`` and ``60daysAgo`` columns.

    Returns:
        ``(current_eps, delta_pct)`` for the first period in ``_EPS_TREND_PERIODS`` that has a current
        value; ``delta_pct`` is None when the 60-day-old value is missing or zero.
    """
    if frame is None or frame.empty or "current" not in frame.columns:
        return None, None
    for period in _EPS_TREND_PERIODS:
        if period not in frame.index:
            continue
        current = _finite(frame.at[period, "current"])
        if current is None:
            continue
        past = _finite(frame.at[period, "60daysAgo"]) if "60daysAgo" in frame.columns else None
        delta = (current - past) / abs(past) if past else None
        return current, delta
    return None, None


def _parse_grades_net(frame: pd.DataFrame | None, today: date) -> int | None:
    """
    Count net analyst rating changes (upgrades - downgrades) in the trailing window.

    Args:
        frame: yfinance ``upgrades_downgrades`` DataFrame indexed by grade timestamp, with an
            ``Action`` column (``up``, ``down``, ``main``, ``init``, ``reit``).
        today: Today's date; defines the window via ``REVISIONS_GRADES_WINDOW_DAYS``.

    Returns:
        Net upgrade count (0 when Yahoo lists no actions), or None when the frame is unusable.
    """
    if frame is None:
        return None
    if frame.empty:
        return 0
    if "Action" not in frame.columns:
        return None
    cutoff = today - timedelta(days=REVISIONS_GRADES_WINDOW_DAYS)
    net = 0
    for stamp, action in zip(frame.index, frame["Action"]):
        try:
            action_date = pd.Timestamp(stamp).date()
        except (TypeError, ValueError):
            continue
        if action_date < cutoff or action_date > today:
            continue
        action_norm = str(action).strip().lower()
        if action_norm == "up":
            net += 1
        elif action_norm == "down":
            net -= 1
    return net


def _net_bullish_score(row: Dict[str, Any]) -> float | None:
    """Weighted analyst-sentiment score in [-2, 2] for one Finnhub recommendation-trend row."""
    total = sum(row.get(k, 0) or 0 for k in ("strongBuy", "buy", "hold", "sell", "strongSell"))
    if total <= 0:
        return None
    return (
        2 * (row.get("strongBuy", 0) or 0)
        + (row.get("buy", 0) or 0)
        - (row.get("sell", 0) or 0)
        - 2 * (row.get("strongSell", 0) or 0)
    ) / total


def _fetch_finnhub_momentum(ticker: str, api_key: str, throttle: FinnhubThrottle | None = None) -> float | None:
    """
    Compute analyst-recommendation momentum from Finnhub's recommendation trend.

    Momentum = net-bullish score of the latest month minus the net-bullish
    score ~3 months ago. Positive means analysts have gotten more bullish.

    Args:
        ticker: Ticker symbol.
        api_key: Finnhub API key.
        throttle: Optional pacer keeping calls under Finnhub's per-minute limit;
            a 429 is retried once after a back-off.

    Returns:
        Momentum, roughly in [-4, 4], or None on failure / insufficient history.
    """
    try:
        resp = finnhub_get(_FINNHUB_RECOMMENDATION_URL, {"symbol": ticker, "token": api_key}, throttle, ticker)
        if resp.status_code != 200:
            _logger.debug("Finnhub recommendation non-200 for %s: %s", ticker, resp.status_code)
            return None
        rows = resp.json()
        if not isinstance(rows, list) or len(rows) < 4:
            return None

        rows_sorted = sorted(rows, key=lambda r: r.get("period", ""), reverse=True)
        latest = _net_bullish_score(rows_sorted[0])
        past = _net_bullish_score(rows_sorted[3])  # ~3 months back; Finnhub periods are monthly

        if latest is None or past is None:
            return None
        return latest - past
    except Exception:
        _logger.debug("Finnhub recommendation fetch failed for %s", ticker)
        return None


def _eps_delta_points(delta_pct: float | None) -> float:
    """
    Score the trailing-60d relative change in forward consensus EPS.

    Args:
        delta_pct: ``(current - 60d ago) / |60d ago|``, or None when unknown.

    Returns:
        Points in ``[0, REVISIONS_EPS_DELTA_MAX_PTS]``; negative revisions earn 0 (not a penalty).
    """
    if delta_pct is None or delta_pct <= 0:
        return 0.0
    return min(REVISIONS_EPS_DELTA_MAX_PTS, (delta_pct / REVISIONS_EPS_DELTA_FULL_CREDIT_PCT) * REVISIONS_EPS_DELTA_MAX_PTS)


def _compute_ticker_signals(
    ticker: str,
    finnhub_api_key: str,
    today: date,
    yahoo_gate: _YahooGate | None = None,
    finnhub_throttle: FinnhubThrottle | None = None,
) -> Dict[str, float]:
    """
    Fetch the raw signals for one ticker and blend them into `revisions_score`.

    Args:
        ticker: Ticker symbol.
        finnhub_api_key: Finnhub API key.
        today: Today's date.
        yahoo_gate: Optional Yahoo circuit breaker shared across the run.
        finnhub_throttle: Optional Finnhub pacer shared across the run.

    Returns:
        Dict of `signal_type -> value` for every signal successfully fetched,
        plus the blended `revisions_score`. Empty dict if nothing could be
        fetched at all (e.g. ticker unrecognized by both providers).
    """
    out: Dict[str, float] = {}

    eps_now, eps_delta = _parse_eps_trend(_yahoo_frame(ticker, "eps_trend", yahoo_gate))
    if eps_now is not None:
        out[_SIGNAL_EPS_NEXT_FY] = eps_now
    if eps_delta is not None:
        out[_SIGNAL_EPS_DELTA_60D] = round(eps_delta, 6)

    grades_net = _parse_grades_net(_yahoo_frame(ticker, "upgrades_downgrades", yahoo_gate), today)
    if grades_net is not None:
        out[_SIGNAL_GRADES_NET_60D] = float(grades_net)

    finnhub_momentum = _fetch_finnhub_momentum(ticker, finnhub_api_key, finnhub_throttle)
    if finnhub_momentum is not None:
        out[_SIGNAL_FINNHUB_MOMENTUM] = finnhub_momentum

    if not out:
        return out

    finnhub_pts = 0.0
    if finnhub_momentum is not None and finnhub_momentum > 0:
        finnhub_pts = min(
            REVISIONS_FINNHUB_MAX_PTS,
            (finnhub_momentum / REVISIONS_FINNHUB_FULL_CREDIT_MOMENTUM) * REVISIONS_FINNHUB_MAX_PTS,
        )

    grades_pts = 0.0
    if grades_net is not None and grades_net > 0:
        grades_pts = min(REVISIONS_GRADES_MAX_PTS, (grades_net / REVISIONS_GRADES_FULL_CREDIT_NET) * REVISIONS_GRADES_MAX_PTS)

    out[_SIGNAL_REVISIONS_SCORE] = round(finnhub_pts + grades_pts + _eps_delta_points(eps_delta), 1)
    return out


def run(as_of_date: date | None = None) -> Dict[str, Any]:
    """
    Ingest Sleeve A revisions signals for watchlist ∪ positions tickers.

    Writes the raw component signals plus the blended `revisions_score`
    into `k20_signals`. Shadow mode: `REVISIONS_FEED_AVAILABLE` stays False
    in config.py until reviewed, so this job has zero effect on live Sleeve A
    scoring until that flag is flipped.

    Args:
        as_of_date: Date label for the signal rows (defaults to today).

    Returns:
        Summary dict.
    """
    target_date = as_of_date or date.today()
    _logger.info("Revisions ingest for %s", target_date)
    start_job_run(_JOB_NAME, target_date)

    try:
        finnhub_api_key = os.environ.get("FINNHUB_API_KEY", "")
        if not finnhub_api_key:
            _logger.warning("FINNHUB_API_KEY not set; skipping revisions ingest")
            finish_job_run(_JOB_NAME, target_date, status="skipped")
            return {"tickers_processed": 0, "tickers_total": 0, "rows_upserted": 0}

        tickers = _get_target_tickers()
        _logger.info("Revisions ingest: %d tickers (watchlist ∪ positions)", len(tickers))

        yahoo_gate = _YahooGate()
        finnhub_throttle = FinnhubThrottle()
        pending: List[Dict[str, Any]] = []
        processed = 0
        rows_upserted = 0
        yahoo_eps_tickers = 0
        for idx, ticker in enumerate(tickers, start=1):
            signals = _compute_ticker_signals(ticker, finnhub_api_key, target_date, yahoo_gate, finnhub_throttle=finnhub_throttle)
            if signals:
                processed += 1
                if _SIGNAL_EPS_DELTA_60D in signals:
                    yahoo_eps_tickers += 1
                for signal_type, value in signals.items():
                    pending.append(
                        {
                            "ticker": ticker,
                            "date": target_date,
                            "signal_type": signal_type,
                            "value": value,
                            "sleeve": _SLEEVE,
                        }
                    )
            if pending and idx % _UPSERT_CHUNK_TICKERS == 0:
                rows_upserted += upsert_signals(pending)
                pending = []

        if pending:
            rows_upserted += upsert_signals(pending)
        finish_job_run(_JOB_NAME, target_date, status="ok", rows_out=rows_upserted)
        _logger.info(
            "Revisions ingest complete: %d/%d tickers, %d signal rows, Yahoo EPS delta for %d",
            processed,
            len(tickers),
            rows_upserted,
            yahoo_eps_tickers,
        )
        return {
            "tickers_processed": processed,
            "tickers_total": len(tickers),
            "rows_upserted": rows_upserted,
            "yahoo_available": not yahoo_gate.closed,
            "yahoo_eps_tickers": yahoo_eps_tickers,
        }

    except Exception as exc:
        _logger.exception("Revisions ingest failed")
        finish_job_run(_JOB_NAME, target_date, status="failed", error=str(exc))
        raise
