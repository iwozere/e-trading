"""
P20 Kestrel — Google Trends poll (weekly, watchlist-scale).

Uses pytrends AsyncTrendsAdapter with anchor-term normalization per §7.3.
Rate-limited with jittered sleeps; degrades gracefully on 429.
"""

from __future__ import annotations

import random
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p20_kestrel.config import TRENDS_ANCHOR_TERM
from src.ml.pipeline.p20_kestrel.db.repos import (
    finish_job_run,
    get_watchlist_tickers,
    start_job_run,
    upsert_sentiment,
)
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_JOB_NAME = "trends_watchlist"
_BATCH_SIZE = 4  # pytrends max terms per request (5 total incl. anchor)
_SLEEP_MIN = 30
_SLEEP_MAX = 60
_429_ABORT_THRESHOLD = 3


def _build_query_term(ticker: str) -> str:
    """Format a Trends query term for a ticker."""
    return f"{ticker} stock"


def _fetch_trends_batch(
    terms: List[str], anchor: str, timeframe: str = "today 12-m"
) -> Optional[Dict[str, float]]:
    """
    Fetch Google Trends for a batch of terms plus the anchor.

    Args:
        terms: Ticker query terms (max 4).
        anchor: Anchor term for normalization.
        timeframe: Pytrends timeframe string.

    Returns:
        Dict mapping term → anchor-normalized interest (0–100), or None on failure.
    """
    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl="en-US", tz=0, timeout=(10, 25))
        kw_list = [anchor] + terms[:_BATCH_SIZE]
        pytrends.build_payload(kw_list, timeframe=timeframe, geo="US")
        df = pytrends.interest_over_time()
        if df.empty:
            return None

        anchor_ref = float(df[anchor].mean())
        if anchor_ref <= 0:
            return None

        result: Dict[str, float] = {}
        for term in terms[:_BATCH_SIZE]:
            if term in df.columns:
                raw_val = float(df[term].iloc[-4:].mean())  # last 4 weeks
                result[term] = round(raw_val * (100.0 / anchor_ref), 2)

        return result
    except Exception as exc:
        if "429" in str(exc) or "Too Many Requests" in str(exc):
            _logger.warning("Trends 429 received; will abort remaining batches")
            raise RuntimeError("429") from exc
        _logger.debug("Trends fetch failed: %s", exc)
        return None


def run(as_of_date: Optional[date] = None) -> Dict[str, Any]:
    """
    Weekly Trends poll for all watchlist tickers.

    Args:
        as_of_date: Date label (defaults to today).

    Returns:
        Summary dict.
    """
    target_date = as_of_date or date.today()
    _logger.info("Trends poll for %s", target_date)
    start_job_run(_JOB_NAME, target_date)

    try:
        tickers = get_watchlist_tickers()
        if not tickers:
            finish_job_run(_JOB_NAME, target_date, status="skipped", rows_out=0)
            return {"tickers": 0, "rows_upserted": 0}

        terms = [_build_query_term(t) for t in tickers]
        ticker_by_term = {_build_query_term(t): t for t in tickers}

        all_rows: List[Dict[str, Any]] = []
        consecutive_429 = 0

        for i in range(0, len(terms), _BATCH_SIZE):
            batch = terms[i: i + _BATCH_SIZE]
            try:
                result = _fetch_trends_batch(batch, anchor=TRENDS_ANCHOR_TERM)
                if result:
                    for term, norm_val in result.items():
                        ticker = ticker_by_term.get(term)
                        if ticker:
                            all_rows.append({
                                "ticker": ticker,
                                "date": target_date,
                                "source": "trends",
                                "mentions": norm_val,
                            })
                consecutive_429 = 0
            except RuntimeError as exc:
                if "429" in str(exc):
                    consecutive_429 += 1
                    if consecutive_429 >= _429_ABORT_THRESHOLD:
                        _logger.warning("Trends: %d consecutive 429s; aborting run", consecutive_429)
                        break
                else:
                    raise

            sleep_s = random.uniform(_SLEEP_MIN, _SLEEP_MAX)
            _logger.debug("Trends batch done; sleeping %.0fs", sleep_s)
            time.sleep(sleep_s)

        rows_upserted = upsert_sentiment(all_rows)
        finish_job_run(_JOB_NAME, target_date, status="ok", rows_out=rows_upserted)
        return {"tickers": len(tickers), "rows_upserted": rows_upserted}

    except Exception as exc:
        _logger.exception("Trends poll failed")
        finish_job_run(_JOB_NAME, target_date, status="failed", error=str(exc))
        raise
