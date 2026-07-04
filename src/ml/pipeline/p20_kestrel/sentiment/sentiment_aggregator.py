"""
P20 Kestrel — Sentiment aggregator.

Implements §7.6 crowding formula with explicit staleness contract.
Runs after gdelt_process + social_poll + av_sentiment complete (DAG).
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.data.db.services.kestrel_service import KestrelService as _KestrelService
from src.ml.pipeline.p20_kestrel.config import STALENESS_DAYS

_kestrel = _KestrelService()
finish_job_run = _kestrel.finish_job_run
get_latest_sentiment = _kestrel.get_latest_sentiment
get_open_positions = _kestrel.get_open_positions
get_sentiment_history = _kestrel.get_sentiment_history
get_watchlist_tickers = _kestrel.get_watchlist_tickers
start_job_run = _kestrel.start_job_run
upsert_signals = _kestrel.upsert_signals
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_JOB_NAME = "sentiment_aggregate"
_MIN_PERIODS = 15  # same warm-up rule as gdelt_processor


def _get_staleness_days(source: str) -> int:
    return STALENESS_DAYS.get(source, 3)


def _z_from_history(
    ticker: str,
    source: str,
    current_value: float,
    as_of_date: date,
) -> float | None:
    """
    Compute a rolling z-score of `mentions` from 30 days of history.

    Pollers (social_poll, trends_poll) store raw mentions only; the z-score
    is derived here at aggregation time. gdelt rows carry a precomputed
    mention_z20 and never reach this path.

    Returns:
        z-score, or None during warm-up (< _MIN_PERIODS days) or zero variance.
    """
    hist = get_sentiment_history(
        ticker,
        source,
        start=as_of_date - timedelta(days=30),
        end=as_of_date,
    )
    vals = [float(r["mentions"]) for r in hist if r.get("mentions") is not None and r.get("date") != as_of_date]
    if len(vals) < _MIN_PERIODS:
        return None
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    std = var**0.5
    if std <= 0:
        return None
    return (current_value - mean) / std


def _usable_mention_z(
    ticker: str,
    source: str,
    staleness_group: str,
    as_of_date: date,
) -> float | None:
    """
    Return the freshness-checked mention z-score for a (ticker, source) pair.

    Uses the stored mention_z20 when present (gdelt); otherwise derives it
    from history (social/trends sources store raw mentions only).
    """
    row = get_latest_sentiment(ticker, source)
    if not row:
        return None

    age = (as_of_date - row["date"]).days if row.get("date") else 999
    if age > _get_staleness_days(staleness_group):
        return None

    z = row.get("mention_z20")
    if z is None and row.get("mentions") is not None:
        z = _z_from_history(ticker, source, float(row["mentions"]), row["date"])

    try:
        return float(z) if z is not None else None
    except (TypeError, ValueError):
        return None


def _compute_crowding_for_ticker(
    ticker: str,
    as_of_date: date,
) -> Dict[str, Any]:
    """
    Compute crowding score per §7.6 for a single ticker.

    Args:
        ticker: Ticker symbol.
        as_of_date: Date being aggregated.

    Returns:
        Dict with crowding, components_used, and per-source z-scores.
    """
    components: Dict[str, float | None] = {
        "social": None,
        "gdelt": None,
        "trends": None,
    }

    # social = max(stocktwits_mention_z, reddit_mention_z, apewisdom_mention_z)
    social_z_values: List[float] = []
    for source in ("stocktwits", "reddit", "apewisdom"):
        z = _usable_mention_z(ticker, source, "social", as_of_date)
        if z is not None:
            social_z_values.append(z)

    if social_z_values:
        components["social"] = max(social_z_values)

    components["gdelt"] = _usable_mention_z(ticker, "gdelt", "gdelt", as_of_date)
    components["trends"] = _usable_mention_z(ticker, "trends", "trends", as_of_date)

    usable = {k: v for k, v in components.items() if v is not None}
    crowding: float | None = None
    if len(usable) >= 2:
        crowding = round(sum(usable.values()) / len(usable), 4)

    return {
        "crowding": crowding,
        "components_used": sorted(usable.keys()),
        "z_social": components["social"],
        "z_gdelt": components["gdelt"],
        "z_trends": components["trends"],
    }


def run(as_of_date: date | None = None) -> Dict[str, Any]:
    """
    Aggregate sentiment into crowding scores for all active tickers.

    Args:
        as_of_date: Date to aggregate for (defaults to today).

    Returns:
        Summary dict.
    """
    target_date = as_of_date or date.today()
    _logger.info("Sentiment aggregator for %s", target_date)
    start_job_run(_JOB_NAME, target_date)

    try:
        # Only watchlist + positions tickers have sentiment data — the pollers
        # are scoped to them. Aggregating all ~7k active tickers would issue
        # tens of thousands of no-op DB round-trips.
        tickers = sorted(set(get_watchlist_tickers()) | {str(p["ticker"]).upper() for p in get_open_positions()})
        _logger.info("Aggregating sentiment for %d tickers", len(tickers))

        signal_rows: List[Dict[str, Any]] = []
        nulls = 0

        for ticker in tickers:
            agg = _compute_crowding_for_ticker(ticker, target_date)
            crowding = agg["crowding"]

            if crowding is not None:
                signal_rows.append(
                    {
                        "ticker": ticker,
                        "date": target_date,
                        "signal_type": "crowding_score",
                        "value": crowding,
                    }
                )
            else:
                nulls += 1

            # Store z-components as individual signals for audit
            for z_key, z_col in [("z_social", "z_social"), ("z_gdelt", "z_gdelt"), ("z_trends", "z_trends")]:
                z_val = agg.get(z_key)
                if z_val is not None:
                    signal_rows.append(
                        {
                            "ticker": ticker,
                            "date": target_date,
                            "signal_type": z_col,
                            "value": z_val,
                        }
                    )

        upserted = upsert_signals(signal_rows)
        _logger.info(
            "Aggregator: %d tickers, %d with crowding, %d null, %d signals upserted",
            len(tickers),
            len(tickers) - nulls,
            nulls,
            upserted,
        )
        finish_job_run(_JOB_NAME, target_date, status="ok", rows_out=upserted)
        return {
            "tickers_processed": len(tickers),
            "with_crowding": len(tickers) - nulls,
            "null_crowding": nulls,
            "signals_upserted": upserted,
        }

    except Exception as exc:
        _logger.exception("Sentiment aggregator failed")
        finish_job_run(_JOB_NAME, target_date, status="failed", error=str(exc))
        raise
