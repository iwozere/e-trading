"""
P20 Kestrel — Sentiment aggregator.

Implements §7.6 crowding formula with explicit staleness contract.
Runs after gdelt_process + social_poll + av_sentiment complete (DAG).
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p20_kestrel.config import STALENESS_DAYS
from src.data.db.services.kestrel_service import KestrelService as _KestrelService

_kestrel = _KestrelService()
finish_job_run = _kestrel.finish_job_run
get_active_tickers = _kestrel.get_active_tickers
get_latest_sentiment = _kestrel.get_latest_sentiment
start_job_run = _kestrel.start_job_run
upsert_signals = _kestrel.upsert_signals
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_JOB_NAME = "sentiment_aggregate"


def _get_staleness_days(source: str) -> int:
    return STALENESS_DAYS.get(source, 3)


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
    components: Dict[str, Optional[float]] = {
        "social": None,
        "gdelt": None,
        "trends": None,
    }

    # social = max(stocktwits_mention_z, reddit_mention_z, apewisdom_mention_z)
    social_z_values: List[float] = []
    for source in ("stocktwits", "reddit", "apewisdom"):
        row = get_latest_sentiment(ticker, source)
        if row and row.get("mention_z20") is not None:
            age = (as_of_date - row["date"]).days if row.get("date") else 999
            if age <= _get_staleness_days("social"):
                try:
                    social_z_values.append(float(row["mention_z20"]))
                except (TypeError, ValueError):
                    pass

    if social_z_values:
        components["social"] = max(social_z_values)

    # gdelt
    gdelt_row = get_latest_sentiment(ticker, "gdelt")
    if gdelt_row and gdelt_row.get("mention_z20") is not None:
        age = (as_of_date - gdelt_row["date"]).days if gdelt_row.get("date") else 999
        if age <= _get_staleness_days("gdelt"):
            try:
                components["gdelt"] = float(gdelt_row["mention_z20"])
            except (TypeError, ValueError):
                pass

    # trends
    trends_row = get_latest_sentiment(ticker, "trends")
    if trends_row and trends_row.get("mention_z20") is not None:
        age = (as_of_date - trends_row["date"]).days if trends_row.get("date") else 999
        if age <= _get_staleness_days("trends"):
            try:
                components["trends"] = float(trends_row["mention_z20"])
            except (TypeError, ValueError):
                pass

    usable = {k: v for k, v in components.items() if v is not None}
    crowding: Optional[float] = None
    if len(usable) >= 2:
        crowding = round(sum(usable.values()) / len(usable), 4)

    return {
        "crowding": crowding,
        "components_used": sorted(usable.keys()),
        "z_social": components["social"],
        "z_gdelt": components["gdelt"],
        "z_trends": components["trends"],
    }


def run(as_of_date: Optional[date] = None) -> Dict[str, Any]:
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
        # All active tickers get crowding computed, but only watchlist has social data
        tickers = get_active_tickers()
        _logger.info("Aggregating sentiment for %d tickers", len(tickers))

        signal_rows: List[Dict[str, Any]] = []
        nulls = 0

        for ticker in tickers:
            agg = _compute_crowding_for_ticker(ticker, target_date)
            crowding = agg["crowding"]

            if crowding is not None:
                signal_rows.append({
                    "ticker": ticker,
                    "date": target_date,
                    "signal_type": "crowding_score",
                    "value": crowding,
                })
            else:
                nulls += 1

            # Store z-components as individual signals for audit
            for z_key, z_col in [("z_social", "z_social"), ("z_gdelt", "z_gdelt"), ("z_trends", "z_trends")]:
                z_val = agg.get(z_key)
                if z_val is not None:
                    signal_rows.append({
                        "ticker": ticker,
                        "date": target_date,
                        "signal_type": z_col,
                        "value": z_val,
                    })

        upserted = upsert_signals(signal_rows)
        _logger.info(
            "Aggregator: %d tickers, %d with crowding, %d null, %d signals upserted",
            len(tickers), len(tickers) - nulls, nulls, upserted,
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
