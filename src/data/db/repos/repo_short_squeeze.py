"""
Short Squeeze Detection Pipeline Repository

Repository layer for short squeeze detection pipeline database operations.
Provides CRUD operations for all short squeeze related tables.
"""

from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any, Sequence
from decimal import Decimal

from sqlalchemy import and_, or_, desc, func, select, update, delete
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from src.data.db.models.model_short_squeeze import (
    ScreenerSnapshot, DeepScanMetrics, SqueezeAlert, AdHocCandidateModel,
    AlertLevel, CandidateSource
)
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class ScreenerSnapshotRepo:
    """Repository for screener snapshot operations."""

    def __init__(self, session: Session):
        self.session = session

    def create_snapshot(self, snapshot_data: Dict[str, Any]) -> ScreenerSnapshot:
        """Create a new screener snapshot."""
        snapshot = ScreenerSnapshot(**snapshot_data)
        self.session.add(snapshot)
        self.session.flush()
        return snapshot

    def bulk_create_snapshots(self, snapshots_data: List[Dict[str, Any]]) -> List[ScreenerSnapshot]:
        """Create multiple screener snapshots in bulk."""
        snapshots = [ScreenerSnapshot(**data) for data in snapshots_data]
        self.session.add_all(snapshots)
        self.session.flush()
        return snapshots

    def clear_snapshots_for_date(self, run_date: date) -> int:
        """Clear all snapshots for a specific run date."""
        result = self.session.execute(
            delete(ScreenerSnapshot)
            .where(ScreenerSnapshot.run_date == run_date)
        )
        deleted_count = result.rowcount
        _logger.info("Cleared %d existing snapshots for run date %s", deleted_count, run_date)
        return deleted_count

    def get_snapshot_count_by_date(self, run_date: date) -> int:
        """Get count of snapshots for a specific run date."""
        result = self.session.execute(
            select(func.count(ScreenerSnapshot.id))
            .where(ScreenerSnapshot.run_date == run_date)
        ).scalar()
        return result or 0

    def get_latest_run_date(self) -> Optional[date]:
        """Get the most recent run date."""
        result = self.session.execute(
            select(func.max(ScreenerSnapshot.run_date))
        ).scalar()
        return result

    def get_snapshots_by_run_date(self, run_date: date) -> Sequence[ScreenerSnapshot]:
        """Get all snapshots for a specific run date."""
        return list(self.session.execute(
            select(ScreenerSnapshot)
            .where(ScreenerSnapshot.run_date == run_date)
            .order_by(desc(ScreenerSnapshot.screener_score))
        ).scalars())

    def get_top_candidates(self, run_date: date, limit: int = 50) -> Sequence[ScreenerSnapshot]:
        """Get top candidates by screener score for a run date."""
        return list(self.session.execute(
            select(ScreenerSnapshot)
            .where(
                and_(
                    ScreenerSnapshot.run_date == run_date,
                    ScreenerSnapshot.screener_score.is_not(None)
                )
            )
            .order_by(desc(ScreenerSnapshot.screener_score))
            .limit(limit)
        ).scalars())

    def get_ticker_history(self, ticker: str, days: int = 30) -> Sequence[ScreenerSnapshot]:
        """Get historical snapshots for a ticker."""
        cutoff_date = date.today() - timedelta(days=days)
        return list(self.session.execute(
            select(ScreenerSnapshot)
            .where(
                and_(
                    ScreenerSnapshot.ticker == ticker.upper(),
                    ScreenerSnapshot.run_date >= cutoff_date
                )
            )
            .order_by(desc(ScreenerSnapshot.run_date))
        ).scalars())


class DeepScanMetricsRepo:
    """Repository for deep scan metrics operations."""

    def __init__(self, session: Session):
        self.session = session

    def upsert_metrics(self, metrics_data: Dict[str, Any]) -> DeepScanMetrics:
        """Create or update deep scan metrics for a ticker and date."""
        ticker = metrics_data['ticker'].upper()
        scan_date = metrics_data['date']

        # Try to find existing record
        existing = self.session.execute(
            select(DeepScanMetrics)
            .where(
                and_(
                    DeepScanMetrics.ticker == ticker,
                    DeepScanMetrics.date == scan_date
                )
            )
        ).scalar_one_or_none()

        if existing:
            # Update existing record
            for key, value in metrics_data.items():
                if key not in ['ticker', 'date']:  # Don't update key fields
                    setattr(existing, key, value)
            self.session.flush()
            return existing
        else:
            # Create new record
            metrics_data['ticker'] = ticker
            metrics = DeepScanMetrics(**metrics_data)
            self.session.add(metrics)
            self.session.flush()
            return metrics

    def bulk_upsert_metrics(self, metrics_list: List[Dict[str, Any]]) -> List[DeepScanMetrics]:
        """Bulk upsert deep scan metrics."""
        results = []
        for metrics_data in metrics_list:
            result = self.upsert_metrics(metrics_data)
            results.append(result)
        return results

    def get_latest_metrics(self, ticker: str) -> Optional[DeepScanMetrics]:
        """Get the most recent metrics for a ticker."""
        return self.session.execute(
            select(DeepScanMetrics)
            .where(DeepScanMetrics.ticker == ticker.upper())
            .order_by(desc(DeepScanMetrics.date))
            .limit(1)
        ).scalar_one_or_none()

    def get_metrics_by_date(self, scan_date: date) -> Sequence[DeepScanMetrics]:
        """Get all metrics for a specific date."""
        return list(self.session.execute(
            select(DeepScanMetrics)
            .where(DeepScanMetrics.date == scan_date)
            .order_by(desc(DeepScanMetrics.squeeze_score))
        ).scalars())

    def get_top_scores_by_date(self, scan_date: date, limit: int = 20) -> Sequence[DeepScanMetrics]:
        """Get top squeeze scores for a date."""
        return list(self.session.execute(
            select(DeepScanMetrics)
            .where(
                and_(
                    DeepScanMetrics.date == scan_date,
                    DeepScanMetrics.squeeze_score.is_not(None)
                )
            )
            .order_by(desc(DeepScanMetrics.squeeze_score))
            .limit(limit)
        ).scalars())

    def get_ticker_metrics_history(self, ticker: str, days: int = 30) -> Sequence[DeepScanMetrics]:
        """Get historical metrics for a ticker."""
        cutoff_date = date.today() - timedelta(days=days)
        return list(self.session.execute(
            select(DeepScanMetrics)
            .where(
                and_(
                    DeepScanMetrics.ticker == ticker.upper(),
                    DeepScanMetrics.date >= cutoff_date
                )
            )
            .order_by(desc(DeepScanMetrics.date))
        ).scalars())


class SqueezeAlertRepo:
    """Repository for squeeze alert operations."""

    def __init__(self, session: Session):
        self.session = session

    def create_alert(self, alert_data: Dict[str, Any]) -> SqueezeAlert:
        """Create a new squeeze alert."""
        alert_data['ticker'] = alert_data['ticker'].upper()
        alert = SqueezeAlert(**alert_data)
        self.session.add(alert)
        self.session.flush()
        return alert

    def mark_alert_sent(self, alert_id: int, notification_id: str) -> bool:
        """Mark an alert as sent with notification ID."""
        result = self.session.execute(
            update(SqueezeAlert)
            .where(SqueezeAlert.id == alert_id)
            .values(sent=True, notification_id=notification_id)
        )
        return result.rowcount > 0

    def check_cooldown(self, ticker: str, alert_level: AlertLevel) -> bool:
        """Check if ticker is in cooldown period for alert level."""
        now = datetime.now()
        active_cooldown = self.session.execute(
            select(SqueezeAlert)
            .where(
                and_(
                    SqueezeAlert.ticker == ticker.upper(),
                    SqueezeAlert.alert_level == alert_level.value,
                    SqueezeAlert.cooldown_expires > now,
                    SqueezeAlert.sent == True
                )
            )
        ).scalar_one_or_none()

        return active_cooldown is not None

    def get_recent_alerts(self, days: int = 7) -> Sequence[SqueezeAlert]:
        """Get recent alerts within specified days."""
        cutoff_date = datetime.now() - timedelta(days=days)
        return list(self.session.execute(
            select(SqueezeAlert)
            .where(SqueezeAlert.timestamp >= cutoff_date)
            .order_by(desc(SqueezeAlert.timestamp))
        ).scalars())

    def get_ticker_alert_history(self, ticker: str, days: int = 30) -> Sequence[SqueezeAlert]:
        """Get alert history for a ticker."""
        cutoff_date = datetime.now() - timedelta(days=days)
        return list(self.session.execute(
            select(SqueezeAlert)
            .where(
                and_(
                    SqueezeAlert.ticker == ticker.upper(),
                    SqueezeAlert.timestamp >= cutoff_date
                )
            )
            .order_by(desc(SqueezeAlert.timestamp))
        ).scalars())

    def cleanup_expired_cooldowns(self) -> int:
        """Remove expired cooldown records."""
        now = datetime.now()
        result = self.session.execute(
            delete(SqueezeAlert)
            .where(
                and_(
                    SqueezeAlert.cooldown_expires < now,
                    SqueezeAlert.sent == True
                )
            )
        )
        return result.rowcount


class AdHocCandidateRepo:
    """Repository for ad-hoc candidate operations."""

    def __init__(self, session: Session):
        self.session = session

    def add_candidate(self, ticker: str, reason: str, expires_at: Optional[datetime] = None) -> AdHocCandidateModel:
        """Add a new ad-hoc candidate."""
        ticker = ticker.upper()

        # Check if candidate already exists
        existing = self.session.execute(
            select(AdHocCandidateModel)
            .where(AdHocCandidateModel.ticker == ticker)
        ).scalar_one_or_none()

        if existing:
            # Reactivate if inactive
            if not existing.active:
                existing.active = True
                existing.reason = reason
                existing.expires_at = expires_at
                self.session.flush()
            return existing

        # Create new candidate
        candidate = AdHocCandidateModel(
            ticker=ticker,
            reason=reason,
            expires_at=expires_at
        )
        self.session.add(candidate)
        self.session.flush()
        return candidate

    def deactivate_candidate(self, ticker: str) -> bool:
        """Deactivate an ad-hoc candidate."""
        result = self.session.execute(
            update(AdHocCandidateModel)
            .where(AdHocCandidateModel.ticker == ticker.upper())
            .values(active=False)
        )
        return result.rowcount > 0

    def get_active_candidates(self) -> Sequence[AdHocCandidateModel]:
        """Get all active ad-hoc candidates."""
        return list(self.session.execute(
            select(AdHocCandidateModel)
            .where(AdHocCandidateModel.active == True)
            .order_by(AdHocCandidateModel.first_seen)
        ).scalars())

    def get_candidate(self, ticker: str) -> Optional[AdHocCandidateModel]:
        """Get a specific ad-hoc candidate."""
        return self.session.execute(
            select(AdHocCandidateModel)
            .where(AdHocCandidateModel.ticker == ticker.upper())
        ).scalar_one_or_none()

    def expire_candidates(self) -> List[str]:
        """Expire candidates past their expiration date."""
        now = datetime.now()
        expired_candidates = list(self.session.execute(
            select(AdHocCandidateModel.ticker)
            .where(
                and_(
                    AdHocCandidateModel.active == True,
                    AdHocCandidateModel.expires_at < now
                )
            )
        ).scalars())

        if expired_candidates:
            self.session.execute(
                update(AdHocCandidateModel)
                .where(
                    and_(
                        AdHocCandidateModel.active == True,
                        AdHocCandidateModel.expires_at < now
                    )
                )
                .values(active=False)
            )

        return expired_candidates

    def promote_by_screener(self, ticker: str) -> bool:
        """Mark candidate as promoted by screener."""
        result = self.session.execute(
            update(AdHocCandidateModel)
            .where(AdHocCandidateModel.ticker == ticker.upper())
            .values(promoted_by_screener=True)
        )
        return result.rowcount > 0


class ShortSqueezeRepo:
    """Unified repository for all short squeeze operations."""

    def __init__(self, session: Session):
        self.session = session
        self.screener_snapshots = ScreenerSnapshotRepo(session)
        self.deep_scan_metrics = DeepScanMetricsRepo(session)
        self.alerts = SqueezeAlertRepo(session)
        self.adhoc_candidates = AdHocCandidateRepo(session)

    def get_active_candidates_for_deep_scan(self) -> List[str]:
        """Get all tickers that should be included in deep scan."""
        # Get latest screener candidates
        latest_run_date = self.screener_snapshots.get_latest_run_date()
        screener_tickers = []

        if latest_run_date:
            top_candidates = self.screener_snapshots.get_top_candidates(latest_run_date)
            screener_tickers = [candidate.ticker for candidate in top_candidates]

        # Get active ad-hoc candidates
        adhoc_candidates = self.adhoc_candidates.get_active_candidates()
        adhoc_tickers = [candidate.ticker for candidate in adhoc_candidates]

        # Combine and deduplicate
        all_tickers = list(set(screener_tickers + adhoc_tickers))

        _logger.info("Found %d candidates for deep scan: %d from screener, %d ad-hoc",
                    len(all_tickers), len(screener_tickers), len(adhoc_tickers))

        return all_tickers

    def cleanup_old_data(self, days_to_keep: int = 90) -> Dict[str, int]:
        """Clean up old data beyond retention period."""
        cutoff_date = date.today() - timedelta(days=days_to_keep)
        cutoff_datetime = datetime.now() - timedelta(days=days_to_keep)

        # Clean up old snapshots
        snapshot_result = self.session.execute(
            delete(ScreenerSnapshot)
            .where(ScreenerSnapshot.run_date < cutoff_date)
        )

        # Clean up old deep scan metrics
        metrics_result = self.session.execute(
            delete(DeepScanMetrics)
            .where(DeepScanMetrics.date < cutoff_date)
        )

        # Clean up old alerts
        alerts_result = self.session.execute(
            delete(SqueezeAlert)
            .where(SqueezeAlert.timestamp < cutoff_datetime)
        )

        # Clean up old FINRA data (keep longer retention for historical analysis)
        finra_cutoff_date = date.today() - timedelta(days=days_to_keep * 2)  # Keep FINRA data twice as long
        finra_result = self.session.execute(
            delete(FINRAShortInterest)
            .where(FINRAShortInterest.settlement_date < finra_cutoff_date)
        )

        cleanup_stats = {
            'snapshots_deleted': snapshot_result.rowcount,
            'metrics_deleted': metrics_result.rowcount,
            'alerts_deleted': alerts_result.rowcount,
            'finra_records_deleted': finra_result.rowcount
        }

        _logger.info("Cleaned up old data: %s", cleanup_stats)
        return cleanup_stats