"""
P20 Kestrel — Alpha Vantage budgeted sentiment.

Enforces the 20 calls/day quota via k20_request_budget.
Priority order: positions → top candidates by score → rotating watchlist.
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import requests

from src.data.db.services.kestrel_service import KestrelService as _KestrelService
from src.ml.pipeline.p20_kestrel.config import AV_DAILY_QUOTA

_kestrel = _KestrelService()
finish_job_run = _kestrel.finish_job_run
get_open_positions = _kestrel.get_open_positions
get_or_create_budget = _kestrel.get_or_create_budget
get_watchlist = _kestrel.get_watchlist
increment_budget_used = _kestrel.increment_budget_used
start_job_run = _kestrel.start_job_run
upsert_sentiment = _kestrel.upsert_sentiment
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_JOB_NAME = "av_sentiment_budgeted"
_AV_URL = "https://www.alphavantage.co/query"
_RETRY_RESERVE = 5  # calls reserved for retries


def _fetch_av_news_sentiment(ticker: str, api_key: str) -> Dict[str, Any] | None:
    """
    Fetch Alpha Vantage NEWS_SENTIMENT for a single ticker.

    Args:
        ticker: Ticker symbol.
        api_key: AV API key.

    Returns:
        Dict with mentions, avg_tone, pos_score, neg_score or None on failure.
    """
    try:
        av_params: Dict[str, Any] = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "limit": 50,
            "apikey": api_key,
        }
        resp = requests.get(_AV_URL, params=av_params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        feed = data.get("feed", [])
        if not feed:
            return None

        mentions = len(feed)
        ticker_sentiments = []
        for article in feed:
            for ts in article.get("ticker_sentiment", []):
                if ts.get("ticker") == ticker:
                    try:
                        ticker_sentiments.append(float(ts.get("ticker_sentiment_score", 0)))
                    except (ValueError, TypeError):
                        pass

        if not ticker_sentiments:
            return None

        avg_tone = sum(ticker_sentiments) / len(ticker_sentiments)
        pos_score = sum(1 for s in ticker_sentiments if s > 0.15) / len(ticker_sentiments)
        neg_score = sum(1 for s in ticker_sentiments if s < -0.15) / len(ticker_sentiments)

        return {
            "mentions": mentions,
            "avg_tone": round(avg_tone, 4),
            "pos_score": round(pos_score, 4),
            "neg_score": round(neg_score, 4),
        }
    except Exception:
        _logger.debug("AV sentiment fetch failed for %s", ticker)
        return None


def _build_priority_queue(today: date) -> List[str]:
    """
    Build the ordered ticker list: positions first, then top candidates by score.

    Args:
        today: Today's date (unused but for clarity of intent).

    Returns:
        Ordered list of tickers (deduplicated).
    """
    seen: set[str] = set()
    queue: List[str] = []

    positions = get_open_positions()
    for p in positions:
        t = p.get("ticker", "")
        if t and t not in seen:
            queue.append(t)
            seen.add(t)

    watchlist_rows = get_watchlist()
    candidates_sorted = sorted(
        watchlist_rows,
        key=lambda r: float(r.get("score") or 0),
        reverse=True,
    )
    for row in candidates_sorted:
        t = row.get("ticker", "")
        if t and t not in seen:
            queue.append(t)
            seen.add(t)

    return queue


def run(as_of_date: date | None = None) -> Dict[str, Any]:
    """
    Fetch AV news sentiment for up to (quota - reserve) tickers.

    Args:
        as_of_date: Date label for budget and sentiment rows (defaults to today).

    Returns:
        Summary dict.
    """
    target_date = as_of_date or date.today()
    _logger.info("AV sentiment budgeted for %s", target_date)
    start_job_run(_JOB_NAME, target_date)

    try:
        api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
        if not api_key:
            _logger.warning("ALPHA_VANTAGE_API_KEY not set; skipping AV sentiment")
            finish_job_run(_JOB_NAME, target_date, status="skipped")
            return {"tickers_fetched": 0, "rows_upserted": 0}

        budget = get_or_create_budget("av_news", target_date, quota=AV_DAILY_QUOTA)
        remaining = budget["quota"] - budget["used"] - _RETRY_RESERVE
        if remaining <= 0:
            _logger.info("AV budget exhausted for %s", target_date)
            finish_job_run(_JOB_NAME, target_date, status="skipped")
            return {"tickers_fetched": 0, "rows_upserted": 0}

        tickers = _build_priority_queue(target_date)[:remaining]
        _logger.info("AV sentiment: fetching %d tickers (budget remaining: %d)", len(tickers), remaining)

        all_rows: List[Dict[str, Any]] = []
        fetched_ok = 0

        for ticker in tickers:
            result = _fetch_av_news_sentiment(ticker, api_key)
            increment_budget_used("av_news", target_date)
            if result is not None:
                all_rows.append(
                    {
                        "ticker": ticker,
                        "date": target_date,
                        "source": "av_news",
                        **result,
                    }
                )
                fetched_ok += 1

        rows_upserted = upsert_sentiment(all_rows)
        finish_job_run(_JOB_NAME, target_date, status="ok", rows_out=rows_upserted)
        return {"tickers_fetched": fetched_ok, "rows_upserted": rows_upserted}

    except Exception as exc:
        _logger.exception("AV sentiment budgeted failed")
        finish_job_run(_JOB_NAME, target_date, status="failed", error=str(exc))
        raise
