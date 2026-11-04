"""
P20 Kestrel — Wikipedia pageviews poll (daily, watchlist + open positions).

Retail-attention signal that replaced the Google Trends/pytrends poll (§7.3):
pytrends was never installed in production, is archived upstream, and Google
429s it by default. Wikimedia's pageviews REST API is official, free, keyless
and returns daily counts, so each run re-fetches the last ``_LOOKBACK_DAYS``
per ticker in one request. That backfill makes a new ticker's z-score usable on
its first run (aggregator needs ``_MIN_PERIODS`` days) and self-heals missed runs.

Rows land in ``k20_sentiment_daily`` as ``source='wikipedia'`` with raw daily
views in ``mentions``; the aggregator derives the z-score.
"""

from __future__ import annotations

import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import quote

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import requests

from src.data.db.services.kestrel_service import KestrelService as _KestrelService
from src.ml.pipeline.p20_kestrel.sentiment.wiki_article_resolver import USER_AGENT, resolve_articles
from src.notification.logger import setup_logger

_kestrel = _KestrelService()
finish_job_run = _kestrel.finish_job_run
get_all_aliases = _kestrel.get_all_aliases
get_open_positions = _kestrel.get_open_positions
get_watchlist_tickers = _kestrel.get_watchlist_tickers
start_job_run = _kestrel.start_job_run
upsert_sentiment = _kestrel.upsert_sentiment

_logger = setup_logger(__name__)

_JOB_NAME = "wiki_pageviews"
SOURCE = "wikipedia"
_PAGEVIEWS_URL = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia.org/all-access/user/{title}/daily/{start}/{end}"
)
_LOOKBACK_DAYS = 35  # > aggregator's 30-day z-score window
_REQUEST_DELAY_S = 0.2
_TIMEOUT_S = 20
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_MAX_RETRIES = 3  # per article; Wikimedia started 429-ing bursts in 2026-11, which used to abort the run
_RETRY_BACKOFF_S = 5.0  # doubled per attempt unless the response carries Retry-After
_MAX_RETRY_WAIT_S = 60.0
_FAILURE_ABORT_THRESHOLD = 10  # consecutive non-404 failures → API is down, stop hammering it


class _PageviewsUnavailable(Exception):
    """Transient API failure (5xx, 429, network) — distinct from a 404 'no data for article'."""


def _legal_names() -> Dict[str, str]:
    """Ticker → legal name from the GDELT alias table (search fallback input for the resolver)."""
    names: Dict[str, str] = {}
    for row in get_all_aliases():
        if row.get("alias_type") == "legal_name":
            names.setdefault(str(row["ticker"]).upper(), str(row["alias"]))
    return names


def _retry_wait_s(resp: requests.Response, attempt: int) -> float:
    """Seconds to wait before retry ``attempt`` (0-based): Retry-After if numeric, else exponential backoff."""
    retry_after = (getattr(resp, "headers", None) or {}).get("Retry-After")
    try:
        wait = float(retry_after) if retry_after is not None else _RETRY_BACKOFF_S * (2 ** attempt)
    except ValueError:
        wait = _RETRY_BACKOFF_S * (2 ** attempt)
    return min(max(wait, 0.0), _MAX_RETRY_WAIT_S)


def _get_with_retry(session: requests.Session, url: str) -> requests.Response:
    """
    GET ``url``, retrying 429/5xx responses up to ``_MAX_RETRIES`` times with backoff.

    Returns:
        The final response (may still be a retryable status once retries are exhausted).

    Raises:
        _PageviewsUnavailable: On a network error.
    """
    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=_TIMEOUT_S)
        except requests.RequestException as exc:
            raise _PageviewsUnavailable(str(exc)) from exc
        if resp.status_code not in _RETRY_STATUSES or attempt == _MAX_RETRIES:
            return resp
        wait = _retry_wait_s(resp, attempt)
        _logger.debug("Pageviews HTTP %s; retry %d/%d in %.1fs", resp.status_code, attempt + 1, _MAX_RETRIES, wait)
        time.sleep(wait)
    raise AssertionError("unreachable")


def _fetch_daily_views(session: requests.Session, title: str, start: date, end: date) -> Dict[date, int]:
    """
    Fetch daily user pageviews for one article.

    Wikimedia omits days with zero views; days between the first and the last
    returned date are filled with 0 so the z-score history isn't biased upward.
    Days after the last returned date are left absent (not yet processed upstream).

    Raises:
        _PageviewsUnavailable: On a transient failure (anything but 200/404).
    """
    url = _PAGEVIEWS_URL.format(
        title=quote(title.replace(" ", "_"), safe=""),
        start=start.strftime("%Y%m%d"),
        end=end.strftime("%Y%m%d"),
    )
    resp = _get_with_retry(session, url)
    if resp.status_code == 404:
        return {}
    if resp.status_code != 200:
        raise _PageviewsUnavailable(f"HTTP {resp.status_code}")

    views = {
        datetime.strptime(item["timestamp"][:8], "%Y%m%d").date(): int(item["views"])
        for item in resp.json().get("items", [])
    }
    if not views:
        return {}
    day, last = min(views), max(views)
    while day <= last:
        views.setdefault(day, 0)
        day += timedelta(days=1)
    return views


def run(as_of_date: date | None = None) -> Dict[str, Any]:
    """
    Fetch Wikipedia pageviews for every watchlist/position ticker and upsert them.

    Args:
        as_of_date: Run date (defaults to today); the newest fetched day is the day before.

    Returns:
        Summary dict.
    """
    target_date = as_of_date or date.today()
    _logger.info("Wikipedia pageviews poll for %s", target_date)
    start_job_run(_JOB_NAME, target_date)

    try:
        tickers = sorted(
            {t.upper() for t in get_watchlist_tickers()} | {str(p["ticker"]).upper() for p in get_open_positions()}
        )
        if not tickers:
            finish_job_run(_JOB_NAME, target_date, status="skipped", rows_out=0)
            return {"tickers": 0, "resolved": 0, "rows_upserted": 0}

        articles = resolve_articles(tickers, _legal_names())
        resolved = {t: title for t, title in articles.items() if title}
        unresolved = sorted(set(tickers) - set(resolved))
        if unresolved:
            _logger.info("No Wikipedia article for %d tickers: %s", len(unresolved), ", ".join(unresolved))

        end = target_date - timedelta(days=1)
        start = end - timedelta(days=_LOOKBACK_DAYS - 1)
        rows: List[Dict[str, Any]] = []
        consecutive_failures = 0
        aborted = False

        with requests.Session() as session:
            session.headers.update({"User-Agent": USER_AGENT})
            for ticker, title in sorted(resolved.items()):
                try:
                    views = _fetch_daily_views(session, title, start, end)
                    consecutive_failures = 0
                except _PageviewsUnavailable as exc:
                    consecutive_failures += 1
                    _logger.warning("Pageviews fetch failed for %s (%s): %s", ticker, title, exc)
                    if consecutive_failures >= _FAILURE_ABORT_THRESHOLD:
                        _logger.error("Pageviews: %d consecutive failures; aborting run", consecutive_failures)
                        aborted = True
                        break
                    continue
                rows.extend(
                    {"ticker": ticker, "date": day, "source": SOURCE, "mentions": float(count)}
                    for day, count in views.items()
                )
                time.sleep(_REQUEST_DELAY_S)

        rows_upserted = upsert_sentiment(rows)
        status = "failed" if aborted and not rows else "ok"
        finish_job_run(_JOB_NAME, target_date, status=status, rows_out=rows_upserted,
                       error="pageviews API unavailable" if status == "failed" else None)
        return {
            "tickers": len(tickers),
            "resolved": len(resolved),
            "rows_upserted": rows_upserted,
            "aborted": aborted,
        }

    except Exception as exc:
        _logger.exception("Wikipedia pageviews poll failed")
        finish_job_run(_JOB_NAME, target_date, status="failed", error=str(exc))
        raise
