"""
Short Squeeze Detection Pipeline Database Operations

This module provides database operations for the short squeeze detection pipeline.
It uses the centralized database infrastructure and acts as a facade for pipeline-specific operations.
"""
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))
from src.data.db.services.short_squeeze_service import ShortSqueezeService
from src.ml.pipeline.p04_short_squeeze.core.models import StructuralMetrics, TransientMetrics, Candidate, ScoredCandidate, Alert, AdHocCandidate, AlertLevel, CandidateSource
from src.notification.logger import setup_logger
_logger = setup_logger(__name__)


class ShortSqueezePipelineDB:
    """
    Database facade for the short squeeze detection pipeline.

    This class provides a simplified interface for pipeline operations
    while using the centralized database infrastructure.
    """

    def __init__(self):
        """Initialize the pipeline database interface."""
        pass

    def save_screener_results(self, results: List[Dict[str, Any]], run_date:
        date) ->int:
        """Save weekly screener results to database."""
        # Service manages sessions internally via UoW pattern
        service = ShortSqueezeService()
        return service.save_screener_results(results, run_date)

    def save_deep_scan_results(self, results: List[Dict[str, Any]],
        scan_date: date) ->int:
        """Save daily deep scan results to database."""
        # Service manages sessions internally via UoW pattern
        service = ShortSqueezeService()
        return service.save_deep_scan_results(results, scan_date)

    def get_candidates_for_deep_scan(self) ->List[str]:
        """Get all tickers that should be analyzed in deep scan."""
        # Service manages sessions internally via UoW pattern
        service = ShortSqueezeService()
        return service.get_candidates_for_deep_scan_tickers()

    def add_adhoc_candidate(self, ticker: str, reason: str, ttl_days: int=7
        ) ->bool:
        """Add an ad-hoc candidate for monitoring."""
        # Service manages sessions internally via UoW pattern
        service = ShortSqueezeService()
        return service.add_adhoc_candidate(ticker, reason, ttl_days)

    def remove_adhoc_candidate(self, ticker: str) ->bool:
        """Remove an ad-hoc candidate."""
        # Service manages sessions internally via UoW pattern
        service = ShortSqueezeService()
        return service.remove_adhoc_candidate(ticker)

    def expire_adhoc_candidates(self) ->List[str]:
        """Expire ad-hoc candidates past their TTL."""
        # Service manages sessions internally via UoW pattern
        service = ShortSqueezeService()
        return service.expire_adhoc_candidates()

    def create_alert(self, ticker: str, alert_level: AlertLevel, reason:
        str, squeeze_score: float, cooldown_days: int=7) ->Optional[int]:
        """Create a new squeeze alert with cooldown."""
        # Service manages sessions internally via UoW pattern
        service = ShortSqueezeService()
        return service.create_alert(ticker, alert_level, reason,
                squeeze_score, cooldown_days)

    def mark_alert_sent(self, alert_id: int, notification_id: str) ->bool:
        """Mark an alert as successfully sent."""
        # Service manages sessions internally via UoW pattern
        service = ShortSqueezeService()
        return service.mark_alert_sent(alert_id, notification_id)

    def get_top_candidates_by_screener_score(self, limit: int=20) ->List[Dict
        [str, Any]]:
        """Get top candidates from latest screener run."""
        # Service manages sessions internally via UoW pattern
        service = ShortSqueezeService()
        return service.get_top_candidates_by_screener_score(limit)

    def get_top_squeeze_scores(self, scan_date: Optional[date]=None, limit:
        int=20) ->List[Dict[str, Any]]:
        """Get top squeeze scores from deep scan."""
        # Service manages sessions internally via UoW pattern
        service = ShortSqueezeService()
        return service.get_top_squeeze_scores(scan_date, limit)

    def get_ticker_analysis(self, ticker: str, days: int=30) ->Dict[str, Any]:
        """Get comprehensive analysis for a ticker."""
        # Service manages sessions internally via UoW pattern
        service = ShortSqueezeService()
        return service.get_ticker_analysis(ticker, days)

    def cleanup_old_data(self, days_to_keep: int=90) ->Dict[str, int]:
        """Clean up old data beyond retention period."""
        # Service manages sessions internally via UoW pattern
        service = ShortSqueezeService()
        return service.cleanup_old_data(days_to_keep)

    def get_pipeline_statistics(self) ->Dict[str, Any]:
        """Get pipeline statistics and health metrics."""
        # Service manages sessions internally via UoW pattern
        service = ShortSqueezeService()
        return service.get_pipeline_statistics()


def save_screener_results(results: List[Dict[str, Any]], run_date: date) ->int:
    """Save weekly screener results to database."""
    db = ShortSqueezePipelineDB()
    return db.save_screener_results(results, run_date)


def save_deep_scan_results(results: List[Dict[str, Any]], scan_date: date
    ) ->int:
    """Save daily deep scan results to database."""
    db = ShortSqueezePipelineDB()
    return db.save_deep_scan_results(results, scan_date)


def get_candidates_for_deep_scan() ->List[str]:
    """Get all tickers that should be analyzed in deep scan."""
    db = ShortSqueezePipelineDB()
    return db.get_candidates_for_deep_scan()


def add_adhoc_candidate(ticker: str, reason: str, ttl_days: int=7) ->bool:
    """Add an ad-hoc candidate for monitoring."""
    db = ShortSqueezePipelineDB()
    return db.add_adhoc_candidate(ticker, reason, ttl_days)


def create_alert(ticker: str, alert_level: AlertLevel, reason: str,
    squeeze_score: float, cooldown_days: int=7) ->Optional[int]:
    """Create a new squeeze alert with cooldown."""
    db = ShortSqueezePipelineDB()
    return db.create_alert(ticker, alert_level, reason, squeeze_score,
        cooldown_days)
