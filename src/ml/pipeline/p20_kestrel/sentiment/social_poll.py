"""
P20 Kestrel — Social poll.

Polls StockTwits, Reddit (r/wallstreetbets, r/stocks), and ApeWisdom
for the current watchlist tickers, merges results, and upserts into
k20_sentiment_daily.

Rows store raw daily ``mentions``; the aggregator derives z-scores from their
history, so each source must report a comparable *daily* volume every day:

- **StockTwits** returns the newest 30 messages, not a count. Mentions are the
  messages of the last 24h, extrapolated from the page's time span when all 30
  fall inside it (AAPL fills 30 messages in about an hour).
- **ApeWisdom** ranks ~1,000 tickers over ~10 pages; all pages are read, and
  watchlist tickers absent from a complete table get an explicit 0 so quiet days
  enter the history.

StockTwits sits behind Cloudflare, which fingerprints the TLS handshake: plain
``requests`` got a 403 challenge page on the Pi (Python 3.14 / its OpenSSL) even with
a full browser User-Agent, while the same code passed on Windows. Requests therefore
go through a curl_cffi session impersonating Chrome (as the Yahoo downloader does).
A block or rate limit stops StockTwits for the rest of the run instead of failing
every ticker.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import requests
from curl_cffi import requests as curl_requests

from src.data.db.services.kestrel_service import KestrelService as _KestrelService
from src.notification.logger import setup_logger

_kestrel = _KestrelService()
finish_job_run = _kestrel.finish_job_run
get_watchlist_tickers = _kestrel.get_watchlist_tickers
start_job_run = _kestrel.start_job_run
upsert_sentiment = _kestrel.upsert_sentiment

_logger = setup_logger(__name__)

_JOB_NAME = "social_poll"
_STOCKTWITS_URL = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
_APEWISDOM_URL = "https://apewisdom.io/api/v1.0/filter/all-stocks/page/{page}"
_REDDIT_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
_REDDIT_SEARCH_URL = "https://oauth.reddit.com/r/{sub}/search.json"
# Impersonation sets a TLS fingerprint and matching headers; don't override the User-Agent.
_STOCKTWITS_IMPERSONATE = "chrome"
_STOCKTWITS_DELAY = 2.1  # seconds between calls
_STOCKTWITS_BLOCK_THRESHOLD = 5  # consecutive 403/5xx/network failures → blocked, stop for this run
# Stop StockTwits before the scheduler's 900s timeout; the rotation below spreads the cut across days.
_STOCKTWITS_BUDGET_S = 600
_APEWISDOM_MAX_PAGES = 30
_APEWISDOM_DELAY = 0.5
# .env.example ships "xxx"; a copied template must read as "not configured", not as a failed login.
_PLACEHOLDER_CREDENTIALS = {"xxx", "changeme", "your_key_here"}


class _StockTwitsUnavailable(Exception):
    """StockTwits refused the client (Cloudflare block or rate limit); polling it further is pointless."""


def _parse_ts(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _daily_mentions(messages: List[Dict[str, Any]], now: datetime) -> float:
    """
    Estimate messages per 24h from one stream page (newest first, at most 30 messages).

    If the page reaches back past 24h, the in-window count is exact. If every message
    is inside the window, the page saturated: extrapolate from its time span.
    """
    times = [_parse_ts(m["created_at"]) for m in messages if m.get("created_at")]
    if not times:
        return 0.0
    window_start = now - timedelta(days=1)
    in_window = sum(1 for t in times if t >= window_start)
    if min(times) < window_start:
        return float(in_window)
    span_hours = max((now - min(times)).total_seconds() / 3600, 1.0)
    return round(len(times) * 24 / span_hours, 1)


def _fetch_stocktwits(session: curl_requests.Session, ticker: str) -> Optional[Dict[str, Any]]:
    """
    Fetch the StockTwits public stream for a ticker.

    Args:
        session: curl_cffi session impersonating a browser.
        ticker: Ticker symbol.

    Returns:
        Dict with mentions (estimated per 24h) and bullish_ratio, or None if this ticker failed
        (unknown symbol, malformed response).

    Raises:
        _StockTwitsUnavailable: On 429, 403 (Cloudflare challenge), 5xx or a network error.
    """
    try:
        resp = session.get(_STOCKTWITS_URL.format(ticker=ticker), timeout=10)
    except curl_requests.exceptions.RequestException as exc:
        raise _StockTwitsUnavailable(f"network error: {exc}") from exc
    if resp.status_code == 429 or resp.status_code == 403 or resp.status_code >= 500:
        raise _StockTwitsUnavailable(f"HTTP {resp.status_code}")
    if resp.status_code != 200:
        _logger.debug("StockTwits HTTP %s for %s", resp.status_code, ticker)
        return None
    now = datetime.now(timezone.utc)
    try:
        messages = resp.json().get("messages", [])
        mentions = _daily_mentions(messages, now)
        # Same 24h window as the volume: a quiet ticker's page reaches back months.
        recent = [m for m in messages if m.get("created_at") and _parse_ts(m["created_at"]) >= now - timedelta(days=1)]
    except (ValueError, KeyError, TypeError):
        _logger.debug("StockTwits returned an unexpected payload for %s", ticker)
        return None

    sentiments = [m.get("entities", {}).get("sentiment") or {} for m in recent]
    bull = sum(1 for s in sentiments if s.get("basic") == "Bullish")
    bear = sum(1 for s in sentiments if s.get("basic") == "Bearish")
    ratio = bull / (bull + bear) if (bull + bear) > 0 else None
    return {"mentions": mentions, "bullish_ratio": ratio}


def _fetch_apewisdom() -> Optional[Dict[str, int]]:
    """
    Fetch the full ApeWisdom daily mention table (all pages).

    Returns:
        Dict mapping ticker → mention_count, or None if any page failed: a partial table
        can't tell "not mentioned" apart from "on a page we didn't get".
    """
    table: Dict[str, int] = {}
    page, pages = 1, 1
    try:
        while page <= min(pages, _APEWISDOM_MAX_PAGES):
            resp = requests.get(_APEWISDOM_URL.format(page=page), timeout=15)
            resp.raise_for_status()
            data = resp.json()
            pages = int(data.get("pages", 1))
            for r in data.get("results", []):
                if r.get("ticker"):
                    table[str(r["ticker"]).upper()] = int(r.get("mentions") or 0)
            page += 1
            time.sleep(_APEWISDOM_DELAY)
    except (requests.RequestException, ValueError, TypeError) as exc:
        _logger.warning("ApeWisdom fetch failed on page %d: %s", page, exc)
        return None
    return table


def _fetch_reddit(ticker: str, reddit_headers: Dict[str, str]) -> int:
    """
    Search Reddit for ticker mentions in r/wallstreetbets and r/stocks.

    Args:
        ticker: Ticker symbol (without $).
        reddit_headers: OAuth bearer headers.

    Returns:
        Mention count across both subreddits.
    """
    total = 0
    for sub in ("wallstreetbets", "stocks"):
        try:
            params: Dict[str, Any] = {
                "q": f"${ticker}",
                "sort": "new",
                "restrict_sr": "true",
                "limit": 25,
                "t": "day",
            }
            resp = requests.get(
                _REDDIT_SEARCH_URL.format(sub=sub),
                params=params,
                headers=reddit_headers,
                timeout=10,
            )
            if resp.status_code in (403, 404):
                continue
            resp.raise_for_status()
            posts = resp.json().get("data", {}).get("children", [])
            total += len(posts)
        except (requests.RequestException, ValueError):
            _logger.debug("Reddit fetch failed for %s in r/%s", ticker, sub)
    return total


def _get_reddit_headers() -> Dict[str, str] | None:
    """
    Get Reddit OAuth bearer headers via app-only (client_credentials) flow.

    Uses the project-standard donotshare variables (REDDIT_API_KEY,
    REDDIT_API_SECRET, REDDIT_USER_AGENT). App-only auth is sufficient for
    read-only public subreddit searches — no username/password needed.

    Returns:
        Bearer headers dict, or None if credentials missing or auth fails.
    """
    client_id = os.environ.get("REDDIT_API_KEY", "")
    client_secret = os.environ.get("REDDIT_API_SECRET", "")
    user_agent = os.environ.get("REDDIT_USER_AGENT", "")
    if user_agent.lower() in _PLACEHOLDER_CREDENTIALS:
        user_agent = ""
    user_agent = user_agent or "KestrelBot/1.0"
    if not client_id or not client_secret or {client_id.lower(), client_secret.lower()} & _PLACEHOLDER_CREDENTIALS:
        _logger.info("Reddit not configured (REDDIT_API_KEY/REDDIT_API_SECRET unset or placeholder); "
                     "Reddit mentions come via ApeWisdom only")
        return None
    try:
        resp = requests.post(
            _REDDIT_TOKEN_URL,
            auth=(client_id, client_secret),
            data={"grant_type": "client_credentials"},
            headers={"User-Agent": user_agent},
            timeout=10,
        )
        resp.raise_for_status()
        token = resp.json().get("access_token", "")
        if not token:
            return None
        return {"Authorization": f"bearer {token}", "User-Agent": user_agent}
    except (requests.RequestException, ValueError) as e:
        _logger.warning("Reddit app-only auth failed; skipping Reddit poll: %s", e)
        return None


def _rotated(tickers: List[str], target_date: date) -> List[str]:
    """Start the StockTwits pass at a date-dependent offset so a budget/rate-limit cut hits different tickers daily."""
    if not tickers:
        return tickers
    offset = target_date.toordinal() % len(tickers)
    return tickers[offset:] + tickers[:offset]


def run(as_of_date: date | None = None) -> Dict[str, Any]:
    """
    Poll social sources for all watchlist tickers and upsert sentiment.

    Args:
        as_of_date: Date label for the sentiment rows (defaults to today).

    Returns:
        Summary dict.
    """
    target_date = as_of_date or date.today()
    _logger.info("Social poll for %s", target_date)
    start_job_run(_JOB_NAME, target_date)

    try:
        tickers = sorted({t.upper() for t in get_watchlist_tickers()})
        _logger.info("Polling %d watchlist tickers", len(tickers))

        apewisdom = _fetch_apewisdom()
        reddit_headers = _get_reddit_headers()

        all_rows: List[Dict[str, Any]] = []
        stocktwits_ok = 0
        reddit_ok = 0
        apewisdom_ok = 0
        stocktwits_stop: Optional[str] = None
        consecutive_failures = 0
        started = time.monotonic()

        with curl_requests.Session(impersonate=_STOCKTWITS_IMPERSONATE) as session:
            for ticker in _rotated(tickers, target_date):
                if stocktwits_stop is None and time.monotonic() - started > _STOCKTWITS_BUDGET_S:
                    stocktwits_stop = f"time budget {_STOCKTWITS_BUDGET_S}s reached"
                if stocktwits_stop is None:
                    time.sleep(_STOCKTWITS_DELAY)
                    try:
                        st = _fetch_stocktwits(session, ticker)
                        consecutive_failures = 0
                    except _StockTwitsUnavailable as exc:
                        st = None
                        consecutive_failures += 1
                        if "429" in str(exc) or consecutive_failures >= _STOCKTWITS_BLOCK_THRESHOLD:
                            stocktwits_stop = str(exc)
                    if st is not None:
                        all_rows.append({
                            "ticker": ticker,
                            "date": target_date,
                            "source": "stocktwits",
                            "mentions": st["mentions"],
                            "bullish_ratio": st["bullish_ratio"],
                        })
                        stocktwits_ok += 1

                if reddit_headers:
                    reddit_mentions = _fetch_reddit(ticker, reddit_headers)
                    if reddit_mentions > 0:
                        all_rows.append({"ticker": ticker, "date": target_date, "source": "reddit",
                                         "mentions": reddit_mentions})
                        reddit_ok += 1

                # A complete table makes absence meaningful: unlisted means 0 mentions today.
                if apewisdom is not None:
                    all_rows.append({"ticker": ticker, "date": target_date, "source": "apewisdom",
                                     "mentions": apewisdom.get(ticker, 0)})
                    apewisdom_ok += ticker in apewisdom

        if stocktwits_stop:
            _logger.warning("StockTwits stopped after %d/%d tickers: %s", stocktwits_ok, len(tickers), stocktwits_stop)

        rows_upserted = upsert_sentiment(all_rows)
        summary = {
            "tickers_polled": len(tickers),
            "stocktwits_ok": stocktwits_ok,
            "reddit_ok": reddit_ok,
            "apewisdom_mentioned": apewisdom_ok,
            "apewisdom_available": apewisdom is not None,
            "rows_upserted": rows_upserted,
        }
        finish_job_run(_JOB_NAME, target_date, status="ok", rows_out=rows_upserted)
        return summary

    except Exception as exc:
        _logger.exception("Social poll failed")
        finish_job_run(_JOB_NAME, target_date, status="failed", error=str(exc))
        raise
