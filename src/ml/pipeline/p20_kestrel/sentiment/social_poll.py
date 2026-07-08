"""
P20 Kestrel — Social poll.

Polls StockTwits, Reddit (r/wallstreetbets, r/stocks), and ApeWisdom
for the current watchlist tickers, merges results, and upserts into
k20_sentiment_daily.
"""

from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import requests

from src.data.db.services.kestrel_service import KestrelService as _KestrelService

_kestrel = _KestrelService()
finish_job_run = _kestrel.finish_job_run
get_watchlist_tickers = _kestrel.get_watchlist_tickers
start_job_run = _kestrel.start_job_run
upsert_sentiment = _kestrel.upsert_sentiment
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_JOB_NAME = "social_poll"
_STOCKTWITS_URL = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
_APEWISDOM_URL = "https://apewisdom.io/api/v1.0/filter/all-stocks/page/1"
_REDDIT_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
_REDDIT_SEARCH_URL = "https://oauth.reddit.com/r/{sub}/search.json"
_STOCKTWITS_DELAY = 2.1  # seconds between calls


def _fetch_stocktwits(ticker: str) -> Dict[str, Any] | None:
    """
    Fetch StockTwits public stream for a ticker.

    Args:
        ticker: Ticker symbol.

    Returns:
        Dict with mentions, bullish_ratio, or None on failure.
    """
    try:
        resp = requests.get(
            _STOCKTWITS_URL.format(ticker=ticker),
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if resp.status_code == 429:
            _logger.warning("StockTwits rate-limit on %s; circuit-breaking", ticker)
            return None
        resp.raise_for_status()
        data = resp.json()
        messages = data.get("messages", [])
        if not messages:
            return {"mentions": 0, "bullish_ratio": None}

        bull = sum(1 for m in messages if m.get("entities", {}).get("sentiment", {}).get("basic") == "Bullish")
        bear = sum(1 for m in messages if m.get("entities", {}).get("sentiment", {}).get("basic") == "Bearish")
        ratio = bull / (bull + bear) if (bull + bear) > 0 else None

        return {"mentions": len(messages), "bullish_ratio": ratio}
    except Exception:
        _logger.debug("StockTwits fetch failed for %s", ticker)
        return None


def _fetch_apewisdom() -> Dict[str, int]:
    """
    Fetch ApeWisdom daily mention table.

    Returns:
        Dict mapping ticker → mention_count.
    """
    try:
        resp = requests.get(_APEWISDOM_URL, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        return {str(r.get("ticker", "")).upper(): int(r.get("mentions", 0)) for r in results if r.get("ticker")}
    except Exception:
        _logger.debug("ApeWisdom fetch failed")
        return {}


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
            params = {
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
        except Exception:
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
    import os

    client_id = os.environ.get("REDDIT_API_KEY", "")
    client_secret = os.environ.get("REDDIT_API_SECRET", "")
    user_agent = os.environ.get("REDDIT_USER_AGENT", "") or "KestrelBot/1.0"
    if not client_id or not client_secret:
        _logger.warning("REDDIT_API_KEY/REDDIT_API_SECRET not set; skipping Reddit poll")
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
    except Exception:
        _logger.warning("Reddit app-only auth failed; skipping Reddit poll")
        return None


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
        tickers = get_watchlist_tickers()
        _logger.info("Polling %d watchlist tickers", len(tickers))

        apewisdom = _fetch_apewisdom()
        reddit_headers = _get_reddit_headers()

        all_rows: List[Dict[str, Any]] = []
        stocktwits_ok = 0
        reddit_ok = 0

        for ticker in tickers:
            # StockTwits
            time.sleep(_STOCKTWITS_DELAY)
            st = _fetch_stocktwits(ticker)
            if st is not None:
                row: Dict[str, Any] = {
                    "ticker": ticker,
                    "date": target_date,
                    "source": "stocktwits",
                    "mentions": st.get("mentions", 0),
                    "bullish_ratio": st.get("bullish_ratio"),
                }
                all_rows.append(row)
                stocktwits_ok += 1

            # Reddit
            if reddit_headers:
                reddit_mentions = _fetch_reddit(ticker, reddit_headers)
                if reddit_mentions > 0:
                    all_rows.append(
                        {
                            "ticker": ticker,
                            "date": target_date,
                            "source": "reddit",
                            "mentions": reddit_mentions,
                        }
                    )
                    reddit_ok += 1

            # ApeWisdom cross-check
            ape_mentions = apewisdom.get(ticker, 0)
            if ape_mentions > 0:
                all_rows.append(
                    {
                        "ticker": ticker,
                        "date": target_date,
                        "source": "apewisdom",
                        "mentions": ape_mentions,
                    }
                )

        rows_upserted = upsert_sentiment(all_rows)
        summary = {
            "tickers_polled": len(tickers),
            "stocktwits_ok": stocktwits_ok,
            "reddit_ok": reddit_ok,
            "rows_upserted": rows_upserted,
        }
        finish_job_run(_JOB_NAME, target_date, status="ok", rows_out=rows_upserted)
        return summary

    except Exception as exc:
        _logger.exception("Social poll failed")
        finish_job_run(_JOB_NAME, target_date, status="failed", error=str(exc))
        raise
