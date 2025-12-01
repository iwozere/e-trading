"""
Short Squeeze Detection Pipeline Service

Service layer for short squeeze detection pipeline business logic.
Provides high-level operations and coordinates between repositories.
"""

from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field


from src.data.db.services.base_service import BaseDBService, with_uow, handle_db_error
from src.data.db.models.model_short_squeeze import AlertLevel, CandidateSource
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


# Business logic dataclasses
@dataclass
class StructuralMetrics:
    """Structural metrics for short squeeze analysis."""
    short_interest_pct: float
    days_to_cover: float
    float_shares: int
    avg_volume_14d: int
    market_cap: int

    def __post_init__(self):
        """Validate structural metrics after initialization."""
        if self.short_interest_pct < 0 or self.short_interest_pct > 1:
            raise ValueError("Short interest percentage must be between 0 and 1")
        if self.days_to_cover < 0:
            raise ValueError("Days to cover must be non-negative")
        if self.float_shares <= 0:
            raise ValueError("Float shares must be positive")
        if self.avg_volume_14d <= 0:
            raise ValueError("Average volume must be positive")
        if self.market_cap <= 0:
            raise ValueError("Market cap must be positive")


@dataclass
class VolumeMetrics:
    """Volume-based metrics for short squeeze analysis."""
    volume_spike_ratio: float
    avg_volume_20d: int
    current_volume: int
    rsi: float
    price_momentum: float

    def __post_init__(self):
        """Validate volume metrics after initialization."""
        if self.volume_spike_ratio < 0:
            raise ValueError("Volume spike ratio must be non-negative")
        if self.avg_volume_20d <= 0:
            raise ValueError("Average volume must be positive")
        if self.current_volume < 0:
            raise ValueError("Current volume must be non-negative")
        if self.rsi < 0 or self.rsi > 100:
            raise ValueError("RSI must be between 0 and 100")


@dataclass
class FINRAMetrics:
    """FINRA short interest metrics."""
    short_interest_shares: int
    short_interest_pct: Optional[float]
    settlement_date: date
    days_to_cover: Optional[float]
    data_age_days: int

    def __post_init__(self):
        """Validate FINRA metrics after initialization."""
        if self.short_interest_shares < 0:
            raise ValueError("Short interest shares must be non-negative")
        if self.short_interest_pct is not None and (self.short_interest_pct < 0 or self.short_interest_pct > 100):
            raise ValueError("Short interest percentage must be between 0 and 100")
        if self.days_to_cover is not None and self.days_to_cover < 0:
            raise ValueError("Days to cover must be non-negative")
        if self.data_age_days < 0:
            raise ValueError("Data age days must be non-negative")


@dataclass
class HybridMetrics:
    """Combined volume and FINRA metrics."""
    volume_metrics: VolumeMetrics
    finra_metrics: Optional[FINRAMetrics]
    transient_metrics: 'TransientMetrics'
    combined_score: float

    def __post_init__(self):
        """Validate hybrid metrics after initialization."""
        if self.combined_score < 0 or self.combined_score > 1:
            raise ValueError("Combined score must be between 0 and 1")


@dataclass
class TransientMetrics:
    """Transient metrics for short squeeze analysis."""
    volume_spike: float
    call_put_ratio: Optional[float]
    sentiment_24h: float
    borrow_fee_pct: Optional[float]

    def __post_init__(self):
        """Validate transient metrics after initialization."""
        if self.volume_spike < 0:
            raise ValueError("Volume spike must be non-negative")
        if self.call_put_ratio is not None and self.call_put_ratio < 0:
            raise ValueError("Call/put ratio must be non-negative")
        if self.sentiment_24h < -1 or self.sentiment_24h > 1:
            raise ValueError("Sentiment must be between -1 and 1")
        if self.borrow_fee_pct is not None and self.borrow_fee_pct < 0:
            raise ValueError("Borrow fee percentage must be non-negative")


@dataclass
class Candidate:
    """Short squeeze candidate."""
    ticker: str
    screener_score: float
    structural_metrics: StructuralMetrics
    last_updated: datetime
    source: CandidateSource = CandidateSource.SCREENER

    def __post_init__(self):
        """Validate candidate after initialization."""
        if not self.ticker or len(self.ticker.strip()) == 0:
            raise ValueError("Ticker cannot be empty")
        if self.screener_score < 0 or self.screener_score > 1:
            raise ValueError("Screener score must be between 0 and 1")
        self.ticker = self.ticker.upper().strip()


@dataclass
class ScoredCandidate:
    """Candidate with transient metrics and final squeeze score."""
    candidate: Candidate
    transient_metrics: TransientMetrics
    squeeze_score: float
    alert_level: Optional[AlertLevel] = None

    def __post_init__(self):
        """Validate scored candidate after initialization."""
        if self.squeeze_score < 0 or self.squeeze_score > 1:
            raise ValueError("Squeeze score must be between 0 and 1")


@dataclass
class Alert:
    """Short squeeze alert."""
    ticker: str
    alert_level: AlertLevel
    reason: str
    squeeze_score: float
    timestamp: datetime = field(default_factory=datetime.now)
    cooldown_expires: Optional[datetime] = None
    sent: bool = False
    notification_id: Optional[str] = None

    def __post_init__(self):
        """Validate alert after initialization."""
        if not self.ticker or len(self.ticker.strip()) == 0:
            raise ValueError("Ticker cannot be empty")
        if self.squeeze_score < 0 or self.squeeze_score > 1:
            raise ValueError("Squeeze score must be between 0 and 1")
        self.ticker = self.ticker.upper().strip()


@dataclass
class AdHocCandidate:
    """Ad-hoc candidate for manual monitoring."""
    ticker: str
    reason: str
    first_seen: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    active: bool = True
    promoted_by_screener: bool = False

    def __post_init__(self):
        """Validate ad-hoc candidate after initialization."""
        if not self.ticker or len(self.ticker.strip()) == 0:
            raise ValueError("Ticker cannot be empty")
        self.ticker = self.ticker.upper().strip()


class ShortSqueezeService(BaseDBService):
    """Service for short squeeze detection pipeline operations."""

    def __init__(self, db_service=None):
        """Initialize the service."""
        super().__init__(db_service)

    @with_uow
    @handle_db_error
    def save_screener_results(self, results: List[Dict[str, Any]], run_date: date) -> int:
        """Save weekly screener results."""
        self._logger.info("Saving %d screener results for %s", len(results), run_date)

        # Check existing data before clearing
        existing_count = self.repos.short_squeeze.screener_snapshots.get_snapshot_count_by_date(run_date)
        self._logger.info("Found %d existing snapshots for %s", existing_count, run_date)

        # Clear existing snapshots for this run date to avoid duplicates
        deleted_count = self.repos.short_squeeze.screener_snapshots.clear_snapshots_for_date(run_date)
        if deleted_count > 0:
            self._logger.info("Cleared %d existing snapshots for %s before inserting new data", deleted_count, run_date)

        # Add run_date to each result
        for result in results:
            result['run_date'] = run_date

        snapshots = self.repos.short_squeeze.screener_snapshots.bulk_create_snapshots(results)

        # Verify the data was saved
        final_count = self.repos.short_squeeze.screener_snapshots.get_snapshot_count_by_date(run_date)
        self._logger.info("Successfully saved %d screener snapshots (replaced %d existing). Final count: %d",
                    len(snapshots), deleted_count, final_count)
        return len(snapshots)

    @with_uow
    @handle_db_error
    def save_deep_scan_results(self, results: List[Dict[str, Any]], scan_date: date) -> int:
        """Save daily deep scan results."""
        self._logger.info("Saving %d deep scan results for %s", len(results), scan_date)

        # Add scan_date to each result
        for result in results:
            result['date'] = scan_date

        metrics = self.repos.short_squeeze.deep_scan_metrics.bulk_upsert_metrics(results)
        self._logger.info("Successfully saved %d deep scan metrics", len(metrics))
        return len(metrics)

    @with_uow
    def get_candidates_for_deep_scan_tickers(self) -> List[str]:
        """Get all tickers that should be analyzed in deep scan."""
        return self.repos.short_squeeze.get_active_candidates_for_deep_scan()

    @with_uow
    @handle_db_error
    def add_adhoc_candidate(self, ticker: str, reason: str, ttl_days: int = 7) -> bool:
        """Add an ad-hoc candidate for monitoring."""
        # Validate ticker
        if not ticker or not ticker.strip():
            self._logger.error("Cannot add ad-hoc candidate: ticker cannot be empty")
            return False

        expires_at = datetime.now() + timedelta(days=ttl_days)
        candidate = self.repos.short_squeeze.adhoc_candidates.add_candidate(ticker, reason, expires_at)
        self._logger.info("Added ad-hoc candidate: %s (reason: %s, expires: %s)",
                    ticker, reason, expires_at)
        return True

    @with_uow
    @handle_db_error
    def remove_adhoc_candidate(self, ticker: str) -> bool:
        """Remove an ad-hoc candidate."""
        success = self.repos.short_squeeze.adhoc_candidates.deactivate_candidate(ticker)
        if success:
            self._logger.info("Deactivated ad-hoc candidate: %s", ticker)
        else:
            self._logger.warning("Ad-hoc candidate not found: %s", ticker)
        return success

    @with_uow
    @handle_db_error
    def expire_adhoc_candidates(self) -> List[str]:
        """Expire ad-hoc candidates past their TTL."""
        expired_tickers = self.repos.short_squeeze.adhoc_candidates.expire_candidates()
        if expired_tickers:
            self._logger.info("Expired %d ad-hoc candidates: %s",
                       len(expired_tickers), expired_tickers)
        return expired_tickers

    @with_uow
    @handle_db_error
    def create_alert(self, ticker: str, alert_level: AlertLevel, reason: str,
                    squeeze_score: float, cooldown_days: int = 7) -> Optional[int]:
        """Create a new squeeze alert with cooldown."""
        # Check if ticker is in cooldown
        if self.repos.short_squeeze.alerts.check_cooldown(ticker, alert_level):
            self._logger.debug("Ticker %s is in cooldown for %s alerts", ticker, alert_level.value)
            return None

        cooldown_expires = datetime.now() + timedelta(days=cooldown_days)

        alert_data = {
            'ticker': ticker,
            'alert_level': alert_level.value,
            'reason': reason,
            'squeeze_score': squeeze_score,
            'cooldown_expires': cooldown_expires
        }

        alert = self.repos.short_squeeze.alerts.create_alert(alert_data)
        self._logger.info("Created %s alert for %s (score: %.3f, cooldown until: %s)",
                    alert_level.value, ticker, squeeze_score, cooldown_expires)
        return alert.id

    @with_uow
    @handle_db_error
    def mark_alert_sent(self, alert_id: int, notification_id: str) -> bool:
        """Mark an alert as successfully sent."""
        success = self.repos.short_squeeze.alerts.mark_alert_sent(alert_id, notification_id)
        if success:
            self._logger.info("Marked alert %d as sent (notification: %s)", alert_id, notification_id)
        return success

    @with_uow
    def get_top_candidates_by_screener_score(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get top candidates from latest screener run."""
        latest_run_date = self.repos.short_squeeze.screener_snapshots.get_latest_run_date()
        if not latest_run_date:
            return []

        snapshots = self.repos.short_squeeze.screener_snapshots.get_top_candidates(latest_run_date, limit)

        results = []
        for snapshot in snapshots:
            results.append({
                'ticker': snapshot.ticker,
                'screener_score': float(snapshot.screener_score) if snapshot.screener_score else 0.0,
                'short_interest_pct': float(snapshot.short_interest_pct) if snapshot.short_interest_pct else None,
                'days_to_cover': float(snapshot.days_to_cover) if snapshot.days_to_cover else None,
                'market_cap': snapshot.market_cap,
                'run_date': snapshot.run_date,
                'data_quality': float(snapshot.data_quality) if snapshot.data_quality else None
            })

        return results

    @with_uow
    def get_top_squeeze_scores(self, scan_date: Optional[date] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """Get top squeeze scores from deep scan."""
        if scan_date is None:
            scan_date = date.today()

        metrics = self.repos.short_squeeze.deep_scan_metrics.get_top_scores_by_date(scan_date, limit)

        results = []
        for metric in metrics:
            results.append({
                'ticker': metric.ticker,
                'squeeze_score': float(metric.squeeze_score) if metric.squeeze_score else 0.0,
                'volume_spike': float(metric.volume_spike) if metric.volume_spike else None,
                'sentiment_24h': float(metric.sentiment_24h) if metric.sentiment_24h else None,
                'call_put_ratio': float(metric.call_put_ratio) if metric.call_put_ratio else None,
                'borrow_fee_pct': float(metric.borrow_fee_pct) if metric.borrow_fee_pct else None,
                'alert_level': metric.alert_level,
                'date': metric.date
            })

        return results

    @with_uow
    def get_ticker_analysis(self, ticker: str, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive analysis for a ticker."""
        ticker = ticker.upper()

        # Get screener history
        screener_history = self.repos.short_squeeze.screener_snapshots.get_ticker_history(ticker, days)

        # Get deep scan history
        metrics_history = self.repos.short_squeeze.deep_scan_metrics.get_ticker_metrics_history(ticker, days)

        # Get alert history
        alert_history = self.repos.short_squeeze.alerts.get_ticker_alert_history(ticker, days)

        # Get ad-hoc candidate info
        adhoc_candidate = self.repos.short_squeeze.adhoc_candidates.get_candidate(ticker)

        return {
            'ticker': ticker,
            'screener_history': [
                {
                    'run_date': s.run_date,
                    'screener_score': float(s.screener_score) if s.screener_score else None,
                    'short_interest_pct': float(s.short_interest_pct) if s.short_interest_pct else None,
                    'days_to_cover': float(s.days_to_cover) if s.days_to_cover else None
                }
                for s in screener_history
            ],
            'metrics_history': [
                {
                    'date': m.date,
                    'squeeze_score': float(m.squeeze_score) if m.squeeze_score else None,
                    'volume_spike': float(m.volume_spike) if m.volume_spike else None,
                    'sentiment_24h': float(m.sentiment_24h) if m.sentiment_24h else None,
                    'alert_level': m.alert_level
                }
                for m in metrics_history
            ],
            'alert_history': [
                {
                    'timestamp': a.timestamp,
                    'alert_level': a.alert_level,
                    'reason': a.reason,
                    'squeeze_score': float(a.squeeze_score) if a.squeeze_score else None,
                    'sent': a.sent
                }
                for a in alert_history
            ],
            'adhoc_candidate': {
                'active': adhoc_candidate.active if adhoc_candidate else False,
                'reason': adhoc_candidate.reason if adhoc_candidate else None,
                'first_seen': adhoc_candidate.first_seen if adhoc_candidate else None,
                'expires_at': adhoc_candidate.expires_at if adhoc_candidate else None
            } if adhoc_candidate else None
        }

    @with_uow
    @handle_db_error
    def cleanup_old_data(self, days_to_keep: int = 90) -> Dict[str, int]:
        """Clean up old data beyond retention period."""
        self._logger.info("Starting cleanup of data older than %d days", days_to_keep)
        cleanup_stats = self.repos.short_squeeze.cleanup_old_data(days_to_keep)
        self._logger.info("Cleanup completed: %s", cleanup_stats)
        return cleanup_stats

    @with_uow
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics and health metrics."""
        try:
            # Get latest run date
            latest_run_date = self.repos.short_squeeze.screener_snapshots.get_latest_run_date()

            # Count active ad-hoc candidates
            active_adhoc = len(self.repos.short_squeeze.adhoc_candidates.get_active_candidates())

            # Count recent alerts
            recent_alerts = len(self.repos.short_squeeze.alerts.get_recent_alerts(7))

            # Get today's deep scan count
            today_metrics = len(self.repos.short_squeeze.deep_scan_metrics.get_metrics_by_date(date.today()))

            # Get FINRA data freshness
            finra_report = self.get_finra_data_freshness_report()

            return {
                'latest_screener_run': latest_run_date,
                'active_adhoc_candidates': active_adhoc,
                'recent_alerts_7d': recent_alerts,
                'todays_deep_scan_count': today_metrics,
                'finra_data_age_days': finra_report.get('data_age_days'),
                'finra_unique_tickers': finra_report.get('unique_tickers', 0),
                'status': 'healthy'
            }
        except Exception as e:
            self._logger.exception("Failed to get pipeline statistics:")
            return {
                'status': 'error',
                'error': str(e)
            }

    @with_uow
    def get_active_adhoc_candidates(self) -> List[Dict[str, Any]]:
        """Get active ad-hoc candidates."""
        try:
            candidates = self.repos.short_squeeze.adhoc_candidates.get_active_candidates()
            result = []
            for candidate in candidates:
                result.append({
                    'ticker': candidate.ticker,
                    'reason': candidate.reason,
                    'first_seen': candidate.first_seen,
                    'expires_at': candidate.expires_at,
                    'active': candidate.active,
                    'promoted_by_screener': candidate.promoted_by_screener
                })
            return result
        except Exception:
            self._logger.exception("Error getting active adhoc candidates:")
            return []

    @with_uow
    def get_candidates_for_deep_scan(self) -> List[Dict[str, Any]]:
        """Get candidates for deep scan from latest screener run."""
        try:
            latest_run_date = self.repos.short_squeeze.screener_snapshots.get_latest_run_date()
            if not latest_run_date:
                return []

            snapshots = self.repos.short_squeeze.screener_snapshots.get_snapshots_by_run_date(latest_run_date)
            result = []
            for snapshot in snapshots:
                result.append({
                    'ticker': snapshot.ticker,
                    'screener_score': float(snapshot.screener_score) if snapshot.screener_score else 0.0,
                    'run_date': snapshot.run_date,
                    'short_interest_pct': float(snapshot.short_interest_pct) if snapshot.short_interest_pct else None,
                    'days_to_cover': float(snapshot.days_to_cover) if snapshot.days_to_cover else None,
                    'market_cap': snapshot.market_cap,
                    'data_quality': float(snapshot.data_quality) if snapshot.data_quality else None
                })
            return result
        except Exception:
            self._logger.exception("Error getting candidates for deep scan:")
            return []

    def get_combined_squeeze_candidates(self, volume_candidates: List[str],
                                      finra_candidates: List[str]) -> List[str]:
        """Combine volume-based and FINRA-based candidates, prioritizing overlap."""
        # Find overlapping candidates (highest priority)
        overlap = set(volume_candidates) & set(finra_candidates)

        # Add remaining volume candidates
        volume_only = set(volume_candidates) - overlap

        # Add remaining FINRA candidates
        finra_only = set(finra_candidates) - overlap

        # Combine in priority order: overlap first, then volume, then FINRA
        combined = list(overlap) + list(volume_only) + list(finra_only)

        self._logger.info("Combined candidates: %d overlap, %d volume-only, %d FINRA-only, %d total",
                    len(overlap), len(volume_only), len(finra_only), len(combined))

        return combined

    @with_uow
    def get_latest_finra_short_interest(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get latest FINRA short interest data for a ticker."""
        try:
            finra_data = self.repos.short_squeeze.finra_short_interest.get_latest_by_ticker(ticker)
            if not finra_data:
                return None

            return {
                'ticker': finra_data.ticker,
                'short_interest_shares': finra_data.short_interest_shares,
                'short_interest_pct': finra_data.short_interest_pct,
                'settlement_date': finra_data.settlement_date,
                'days_to_cover': finra_data.days_to_cover,
                'data_age_days': (date.today() - finra_data.settlement_date).days
            }
        except Exception:
            self._logger.exception("Error getting FINRA data for %s:", ticker)
            return None

    @with_uow
    def get_finra_data_freshness_report(self) -> Dict[str, Any]:
        """Get FINRA data freshness statistics."""
        try:
            latest_date = self.repos.short_squeeze.finra_short_interest.get_latest_settlement_date()
            if not latest_date:
                return {'data_age_days': None, 'unique_tickers': 0}

            data_age = (date.today() - latest_date).days
            unique_tickers = self.repos.short_squeeze.finra_short_interest.count_unique_tickers()

            return {
                'data_age_days': data_age,
                'latest_settlement_date': latest_date,
                'unique_tickers': unique_tickers
            }
        except Exception:
            self._logger.exception("Error getting FINRA freshness report:")
            return {'data_age_days': None, 'unique_tickers': 0}

    @with_uow
    @handle_db_error
    def store_finra_data(self, finra_data_list: List[Dict[str, Any]]) -> int:
        """Store FINRA short interest data."""
        try:
            count = self.repos.short_squeeze.finra_short_interest.bulk_upsert(finra_data_list)
            self._logger.info("Stored %d FINRA records", count)
            return count
        except Exception:
            self._logger.exception("Error storing FINRA data:")
            return 0

    @with_uow
    def get_bulk_finra_short_interest(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get FINRA short interest data for multiple tickers."""
        try:
            result = {}
            for ticker in tickers:
                finra_data = self.get_latest_finra_short_interest(ticker)
                if finra_data:
                    result[ticker.upper()] = finra_data
            return result
        except Exception:
            self._logger.exception("Error getting bulk FINRA data:")
            return {}