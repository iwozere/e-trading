"""
P20 Kestrel — Sleeve A revisions feed ingest (gap 10.1, §4.2.1).

Populates `revisions_score` (plus its three raw component signals) in
`k20_signals` for watchlist ∪ positions tickers, blending three analyst
signals so the score is useful immediately rather than needing a cold start:

- Finnhub recommendation-trend momentum (immediate — Finnhub already returns
  several months of history in one call).
- FMP analyst rating-change net count, trailing 60d (immediate — FMP `grades`
  already returns dated historical actions in one call).
- FMP forward-EPS-consensus 60-day delta (needs ~60-90 days of our own daily
  snapshots before a first comparator exists — the literal "forward-EPS
  revisions up 60d" signal named in §4.2).

`REVISIONS_FEED_AVAILABLE` in config.py stays False (shadow mode: this job
writes signals, but sleeve_a.py's scoring ignores them) until the values have
been reviewed for a few weeks. See docs/Design.md for the rollout plan.
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import requests

from src.data.db.services.kestrel_service import KestrelService as _KestrelService
from src.ml.pipeline.p20_kestrel.config import (
    REVISIONS_EPS_DELTA_FULL_CREDIT_PCT,
    REVISIONS_EPS_DELTA_MAX_PTS,
    REVISIONS_EPS_DELTA_TOLERANCE_DAYS,
    REVISIONS_EPS_DELTA_WINDOW_DAYS,
    REVISIONS_FINNHUB_FULL_CREDIT_MOMENTUM,
    REVISIONS_FINNHUB_MAX_PTS,
    REVISIONS_GRADES_FULL_CREDIT_NET,
    REVISIONS_GRADES_MAX_PTS,
    REVISIONS_GRADES_WINDOW_DAYS,
)

_kestrel = _KestrelService()
finish_job_run = _kestrel.finish_job_run
get_open_positions = _kestrel.get_open_positions
get_signals = _kestrel.get_signals
get_watchlist_tickers = _kestrel.get_watchlist_tickers
start_job_run = _kestrel.start_job_run
upsert_signals = _kestrel.upsert_signals
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_JOB_NAME = "revisions_ingest"
_SLEEVE = "A"

_FMP_STABLE_URL = "https://financialmodelingprep.com/stable"
_FINNHUB_RECOMMENDATION_URL = "https://finnhub.io/api/v1/stock/recommendation"

_SIGNAL_EPS_AVG = "fmp_eps_avg_next_fy"
_SIGNAL_GRADES_NET_60D = "fmp_grade_net_60d"
_SIGNAL_FINNHUB_MOMENTUM = "finnhub_rec_momentum"
_SIGNAL_REVISIONS_SCORE = "revisions_score"


def _get_target_tickers() -> List[str]:
    """Union of watchlist + positions tickers (same scope as filings_ingest)."""
    watchlist = set(get_watchlist_tickers())
    positions = {p["ticker"] for p in get_open_positions()}
    return sorted(watchlist | positions)


def _fetch_fmp_eps_avg_next_fy(ticker: str, api_key: str, today: date) -> float | None:
    """
    Fetch the consensus EPS estimate for the nearest upcoming fiscal year.

    Uses FMP's `/stable/analyst-estimates` with `period=annual` — the only
    period tier available on the account's current plan (`period=quarter`
    returns 402 Payment Required).

    Args:
        ticker: Ticker symbol.
        api_key: FMP API key.
        today: Today's date, used to pick the nearest *future* fiscal year row.

    Returns:
        Consensus EPS average for the nearest upcoming fiscal year, or None
        on failure / no future row found.
    """
    try:
        resp = requests.get(
            f"{_FMP_STABLE_URL}/analyst-estimates",
            params={"symbol": ticker, "period": "annual", "apikey": api_key},
            timeout=15,
        )
        if resp.status_code != 200:
            _logger.debug("FMP analyst-estimates non-200 for %s: %s", ticker, resp.status_code)
            return None
        rows = resp.json()
        if not isinstance(rows, list) or not rows:
            return None

        future_rows: List[tuple[date, float]] = []
        for row in rows:
            try:
                fy_date = datetime.strptime(row["date"], "%Y-%m-%d").date()
                eps_avg = row.get("epsAvg")
            except (KeyError, ValueError, TypeError):
                continue
            if eps_avg is not None and fy_date >= today:
                future_rows.append((fy_date, float(eps_avg)))

        if not future_rows:
            return None

        future_rows.sort(key=lambda r: r[0])
        return future_rows[0][1]
    except Exception:
        _logger.debug("FMP analyst-estimates fetch failed for %s", ticker)
        return None


def _fetch_fmp_grades_net_60d(ticker: str, api_key: str, today: date) -> int | None:
    """
    Count net analyst rating changes (upgrades - downgrades) in the trailing window.

    Args:
        ticker: Ticker symbol.
        api_key: FMP API key.
        today: Today's date; defines the trailing window via
            `REVISIONS_GRADES_WINDOW_DAYS`.

    Returns:
        Net upgrade count, or None on failure.
    """
    try:
        resp = requests.get(
            f"{_FMP_STABLE_URL}/grades",
            params={"symbol": ticker, "apikey": api_key},
            timeout=15,
        )
        if resp.status_code != 200:
            _logger.debug("FMP grades non-200 for %s: %s", ticker, resp.status_code)
            return None
        rows = resp.json()
        if not isinstance(rows, list):
            return None

        cutoff = today - timedelta(days=REVISIONS_GRADES_WINDOW_DAYS)
        net = 0
        for row in rows:
            try:
                action_date = datetime.strptime(row["date"], "%Y-%m-%d").date()
            except (KeyError, ValueError, TypeError):
                continue
            if action_date < cutoff:
                continue
            action = str(row.get("action", "")).strip().lower()
            if action == "upgrade":
                net += 1
            elif action == "downgrade":
                net -= 1
        return net
    except Exception:
        _logger.debug("FMP grades fetch failed for %s", ticker)
        return None


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


def _fetch_finnhub_momentum(ticker: str, api_key: str) -> float | None:
    """
    Compute analyst-recommendation momentum from Finnhub's recommendation trend.

    Momentum = net-bullish score of the latest month minus the net-bullish
    score ~3 months ago. Positive means analysts have gotten more bullish.

    Args:
        ticker: Ticker symbol.
        api_key: Finnhub API key.

    Returns:
        Momentum, roughly in [-4, 4], or None on failure / insufficient history.
    """
    try:
        resp = requests.get(
            _FINNHUB_RECOMMENDATION_URL,
            params={"symbol": ticker, "token": api_key},
            timeout=15,
        )
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


def _eps_delta_points(ticker: str, current_eps_avg: float | None, today: date) -> float:
    """
    Score the trailing-60d change in consensus forward EPS, once warmed up.

    Looks for a previously-stored `fmp_eps_avg_next_fy` snapshot from
    ~`REVISIONS_EPS_DELTA_WINDOW_DAYS` ago (within a tolerance window, since
    the job only runs on trading days). Returns 0.0 — not a penalty, just "no
    signal yet" — until that snapshot exists.

    Args:
        ticker: Ticker symbol.
        current_eps_avg: Today's consensus EPS for the nearest future FY, or None.
        today: Today's date.

    Returns:
        Points in `[0, REVISIONS_EPS_DELTA_MAX_PTS]`.
    """
    if current_eps_avg is None:
        return 0.0

    window_start = today - timedelta(days=REVISIONS_EPS_DELTA_WINDOW_DAYS + REVISIONS_EPS_DELTA_TOLERANCE_DAYS)
    window_end = today - timedelta(days=REVISIONS_EPS_DELTA_WINDOW_DAYS - REVISIONS_EPS_DELTA_TOLERANCE_DAYS)
    past_rows = get_signals(ticker, _SIGNAL_EPS_AVG, start_date=window_start, end_date=window_end)
    if not past_rows:
        return 0.0  # not warmed up yet — no ~60d-old snapshot to compare against

    target = today - timedelta(days=REVISIONS_EPS_DELTA_WINDOW_DAYS)
    past_row = min(past_rows, key=lambda r: abs((r["date"] - target).days))
    past_eps_avg = float(past_row["value"])
    if past_eps_avg == 0:
        return 0.0

    pct_change = (current_eps_avg - past_eps_avg) / abs(past_eps_avg)
    if pct_change <= 0:
        return 0.0
    return min(REVISIONS_EPS_DELTA_MAX_PTS, (pct_change / REVISIONS_EPS_DELTA_FULL_CREDIT_PCT) * REVISIONS_EPS_DELTA_MAX_PTS)


def _compute_ticker_signals(
    ticker: str,
    fmp_api_key: str,
    finnhub_api_key: str,
    today: date,
) -> Dict[str, float]:
    """
    Fetch all three raw signals for one ticker and blend them into `revisions_score`.

    Args:
        ticker: Ticker symbol.
        fmp_api_key: FMP API key.
        finnhub_api_key: Finnhub API key.
        today: Today's date.

    Returns:
        Dict of `signal_type -> value` for every signal successfully fetched,
        plus the blended `revisions_score`. Empty dict if nothing could be
        fetched at all (e.g. ticker unrecognized by both providers).
    """
    out: Dict[str, float] = {}

    eps_avg = _fetch_fmp_eps_avg_next_fy(ticker, fmp_api_key, today)
    if eps_avg is not None:
        out[_SIGNAL_EPS_AVG] = eps_avg

    grades_net = _fetch_fmp_grades_net_60d(ticker, fmp_api_key, today)
    if grades_net is not None:
        out[_SIGNAL_GRADES_NET_60D] = float(grades_net)

    finnhub_momentum = _fetch_finnhub_momentum(ticker, finnhub_api_key)
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

    eps_pts = _eps_delta_points(ticker, eps_avg, today)

    out[_SIGNAL_REVISIONS_SCORE] = round(finnhub_pts + grades_pts + eps_pts, 1)
    return out


def run(as_of_date: date | None = None) -> Dict[str, Any]:
    """
    Ingest Sleeve A revisions signals for watchlist ∪ positions tickers.

    Writes the three raw component signals plus the blended `revisions_score`
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
        fmp_api_key = os.environ.get("FMP_API_KEY", "")
        finnhub_api_key = os.environ.get("FINNHUB_API_KEY", "")
        if not fmp_api_key or not finnhub_api_key:
            _logger.warning("FMP_API_KEY/FINNHUB_API_KEY not set; skipping revisions ingest")
            finish_job_run(_JOB_NAME, target_date, status="skipped")
            return {"tickers_processed": 0, "tickers_total": 0, "rows_upserted": 0}

        tickers = _get_target_tickers()
        _logger.info("Revisions ingest: %d tickers (watchlist ∪ positions)", len(tickers))

        all_rows: List[Dict[str, Any]] = []
        processed = 0
        for ticker in tickers:
            signals = _compute_ticker_signals(ticker, fmp_api_key, finnhub_api_key, target_date)
            if not signals:
                continue
            processed += 1
            for signal_type, value in signals.items():
                all_rows.append(
                    {
                        "ticker": ticker,
                        "date": target_date,
                        "signal_type": signal_type,
                        "value": value,
                        "sleeve": _SLEEVE,
                    }
                )

        rows_upserted = upsert_signals(all_rows) if all_rows else 0
        finish_job_run(_JOB_NAME, target_date, status="ok", rows_out=rows_upserted)
        _logger.info(
            "Revisions ingest complete: %d/%d tickers, %d signal rows",
            processed,
            len(tickers),
            rows_upserted,
        )
        return {"tickers_processed": processed, "tickers_total": len(tickers), "rows_upserted": rows_upserted}

    except Exception as exc:
        _logger.exception("Revisions ingest failed")
        finish_job_run(_JOB_NAME, target_date, status="failed", error=str(exc))
        raise
