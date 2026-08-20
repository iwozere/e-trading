"""
Short Squeeze Detection Pipeline Repository

Repository layer for short squeeze detection pipeline database operations.
Provides CRUD operations for all short squeeze related tables.
"""

from datetime import UTC, date, datetime, timedelta
from typing import Any, Dict, List, Sequence

from sqlalchemy import and_, delete, desc, func, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from src.data.db.models.model_short_squeeze import (
    AdHocCandidateModel,
    AlertLevel,
    DeepScanMetrics,
    HnCorpusItem,
    ScreenerSnapshot,
    SentimentCalibration,
    SqueezeAlert,
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
        result = self.session.execute(delete(ScreenerSnapshot).where(ScreenerSnapshot.run_date == run_date))
        deleted_count = result.rowcount  # type: ignore[attr-defined]
        _logger.info("Cleared %d existing snapshots for run date %s", deleted_count, run_date)
        return deleted_count

    def get_snapshot_count_by_date(self, run_date: date) -> int:
        """Get count of snapshots for a specific run date."""
        result = self.session.execute(
            select(func.count(ScreenerSnapshot.id)).where(ScreenerSnapshot.run_date == run_date)
        ).scalar()
        return result or 0

    def get_latest_run_date(self) -> date | None:
        """Get the most recent run date."""
        result = self.session.execute(select(func.max(ScreenerSnapshot.run_date))).scalar()
        return result

    def get_snapshots_by_run_date(self, run_date: date) -> Sequence[ScreenerSnapshot]:
        """Get all snapshots for a specific run date."""
        return list(
            self.session.execute(
                select(ScreenerSnapshot)
                .where(ScreenerSnapshot.run_date == run_date)
                .order_by(desc(ScreenerSnapshot.screener_score))
            ).scalars()
        )

    def get_top_candidates(self, run_date: date, limit: int = 50) -> Sequence[ScreenerSnapshot]:
        """Get top candidates by screener score for a run date."""
        return list(
            self.session.execute(
                select(ScreenerSnapshot)
                .where(and_(ScreenerSnapshot.run_date == run_date, ScreenerSnapshot.screener_score.is_not(None)))
                .order_by(desc(ScreenerSnapshot.screener_score))
                .limit(limit)
            ).scalars()
        )

    def get_ticker_history(self, ticker: str, days: int = 30) -> Sequence[ScreenerSnapshot]:
        """Get historical snapshots for a ticker."""
        cutoff_date = date.today() - timedelta(days=days)
        return list(
            self.session.execute(
                select(ScreenerSnapshot)
                .where(and_(ScreenerSnapshot.ticker == ticker.upper(), ScreenerSnapshot.run_date >= cutoff_date))
                .order_by(desc(ScreenerSnapshot.run_date))
            ).scalars()
        )


class DeepScanMetricsRepo:
    """Repository for deep scan metrics operations."""

    def __init__(self, session: Session):
        self.session = session

    def upsert_metrics(self, metrics_data: Dict[str, Any]) -> DeepScanMetrics:
        """Create or update deep scan metrics for a ticker and date."""
        ticker = metrics_data["ticker"].upper()
        scan_date = metrics_data["date"]

        # Try to find existing record
        existing = self.session.execute(
            select(DeepScanMetrics).where(and_(DeepScanMetrics.ticker == ticker, DeepScanMetrics.date == scan_date))
        ).scalar_one_or_none()

        if existing:
            # Update existing record
            for key, value in metrics_data.items():
                if key not in ["ticker", "date"]:  # Don't update key fields
                    setattr(existing, key, value)
            self.session.flush()
            return existing
        else:
            # Create new record
            metrics_data["ticker"] = ticker
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

    def get_latest_metrics(self, ticker: str) -> DeepScanMetrics | None:
        """Get the most recent metrics for a ticker."""
        return self.session.execute(
            select(DeepScanMetrics)
            .where(DeepScanMetrics.ticker == ticker.upper())
            .order_by(desc(DeepScanMetrics.date))
            .limit(1)
        ).scalar_one_or_none()

    def get_metrics_by_date(self, scan_date: date) -> Sequence[DeepScanMetrics]:
        """Get all metrics for a specific date."""
        return list(
            self.session.execute(
                select(DeepScanMetrics)
                .where(DeepScanMetrics.date == scan_date)
                .order_by(desc(DeepScanMetrics.squeeze_score))
            ).scalars()
        )

    def get_top_scores_by_date(self, scan_date: date, limit: int = 20) -> Sequence[DeepScanMetrics]:
        """Get top squeeze scores for a date."""
        return list(
            self.session.execute(
                select(DeepScanMetrics)
                .where(and_(DeepScanMetrics.date == scan_date, DeepScanMetrics.squeeze_score.is_not(None)))
                .order_by(desc(DeepScanMetrics.squeeze_score))
                .limit(limit)
            ).scalars()
        )

    def count_raw_payload_older_than(self, retention_days: int) -> int:
        """Count rows with a non-null raw_payload older than retention_days (dry-run helper)."""
        cutoff = date.today() - timedelta(days=retention_days)
        return (
            self.session.execute(
                select(func.count())
                .select_from(DeepScanMetrics)
                .where(and_(DeepScanMetrics.date < cutoff, DeepScanMetrics.raw_payload.is_not(None)))
            ).scalar_one()
            or 0
        )

    def purge_raw_payload_older_than(self, retention_days: int) -> int:
        """
        Null out raw_payload for rows older than retention_days via the DB-side
        ``purge_old_sentiment_raw_payload`` function (migration 005). Returns rows purged.
        """
        result = self.session.execute(text("SELECT purge_old_sentiment_raw_payload(:retention_days)"), {"retention_days": retention_days})
        return int(result.scalar_one())

    def get_metrics_since(self, since_date: date) -> Sequence[DeepScanMetrics]:
        """Get all metrics (all tickers) with ``date >= since_date`` -- backs coverage-report."""
        return list(
            self.session.execute(
                select(DeepScanMetrics).where(DeepScanMetrics.date >= since_date).order_by(desc(DeepScanMetrics.date))
            ).scalars()
        )

    def get_ticker_metrics_history(self, ticker: str, days: int = 30) -> Sequence[DeepScanMetrics]:
        """Get historical metrics for a ticker."""
        cutoff_date = date.today() - timedelta(days=days)
        return list(
            self.session.execute(
                select(DeepScanMetrics)
                .where(and_(DeepScanMetrics.ticker == ticker.upper(), DeepScanMetrics.date >= cutoff_date))
                .order_by(desc(DeepScanMetrics.date))
            ).scalars()
        )


class HnCorpusRepo:
    """
    Repository for the shared Hacker News corpus cache (``ss_hn_corpus``).

    Backs the shared-corpus fetch strategy in ``adapters/async_hackernews.py`` (spec §2.4):
    items already cached within the configured TTL are never re-fetched from the Firebase API,
    keeping the fetch cost O(corpus size) regardless of how many tickers are being scanned.
    """

    def __init__(self, session: Session):
        self.session = session

    def get_cached_item_ids(self, item_ids: List[int], ttl_seconds: int) -> set[int]:
        """Return the subset of ``item_ids`` already cached and still within ``ttl_seconds``."""
        if not item_ids:
            return set()
        cutoff = datetime.now(UTC) - timedelta(seconds=ttl_seconds)
        rows = self.session.execute(
            select(HnCorpusItem.item_id).where(
                and_(HnCorpusItem.item_id.in_(item_ids), HnCorpusItem.fetched_at >= cutoff)
            )
        ).scalars()
        return set(rows)

    def upsert_items(self, items: List[Dict[str, Any]]) -> int:
        """Bulk upsert corpus items (PK: item_id). Returns row count."""
        if not items:
            return 0
        stmt = pg_insert(HnCorpusItem).values(items)
        stmt = stmt.on_conflict_do_update(
            index_elements=["item_id"],
            set_={k: stmt.excluded[k] for k in items[0] if k != "item_id"},
        )
        self.session.execute(stmt)
        return len(items)

    def get_items_since(
        self, since: datetime, fetched_after: datetime | None = None
    ) -> List[Dict[str, Any]]:
        """
        Return cached corpus items created at or after ``since``, as plain dicts.

        Args:
            since: Only return items whose ``created_utc`` is at or after this time.
            fetched_after: When given, additionally require ``fetched_at >= fetched_after`` --
                used to detect a "the whole corpus is still fresh" cache hit (see
                ``async_hackernews.py``'s TTL-based whole-corpus reuse).
        """
        conditions = [HnCorpusItem.created_utc >= since]
        if fetched_after is not None:
            conditions.append(HnCorpusItem.fetched_at >= fetched_after)
        rows = self.session.execute(select(HnCorpusItem).where(and_(*conditions))).scalars()
        return [{c.key: getattr(row, c.key) for c in HnCorpusItem.__table__.columns} for row in rows]


class SentimentCalibrationRepo:
    """
    Repository for per-source daily sentiment calibration rows (``ss_sentiment_calibration``).

    Backs the z-score calibration step in ``processing/calibration.py`` (spec §2.5.6): one row
    per (provider, day) is upserted after each batch run, and a trailing window is read back and
    pooled before the next run's raw scores are calibrated.
    """

    def __init__(self, session: Session):
        self.session = session

    def upsert_daily_stats(self, provider: str, day: date, mean_score: float, std_score: float, n_obs: int) -> None:
        """Upsert one day's (mean, std, n) for one provider (PK: provider, day)."""
        stmt = pg_insert(SentimentCalibration).values(
            provider=provider, day=day, mean_score=mean_score, std_score=std_score, n_obs=n_obs
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["provider", "day"],
            set_={"mean_score": stmt.excluded.mean_score, "std_score": stmt.excluded.std_score, "n_obs": stmt.excluded.n_obs},
        )
        self.session.execute(stmt)

    def get_trailing_observations(self, provider: str, window_days: int) -> List[Dict[str, Any]]:
        """Return the trailing ``window_days`` of daily rows for one provider, as plain dicts."""
        cutoff = date.today() - timedelta(days=window_days)
        rows = self.session.execute(
            select(SentimentCalibration).where(
                and_(SentimentCalibration.provider == provider, SentimentCalibration.day >= cutoff)
            )
        ).scalars()
        return [
            {
                "provider": row.provider,
                "day": row.day.isoformat(),
                "mean_score": float(row.mean_score),
                "std_score": float(row.std_score),
                "n_obs": row.n_obs,
            }
            for row in rows
        ]


class SqueezeAlertRepo:
    """Repository for squeeze alert operations."""

    def __init__(self, session: Session):
        self.session = session

    def create_alert(self, alert_data: Dict[str, Any]) -> SqueezeAlert:
        """Create a new squeeze alert."""
        alert_data["ticker"] = alert_data["ticker"].upper()
        alert = SqueezeAlert(**alert_data)
        self.session.add(alert)
        self.session.flush()
        return alert

    def mark_alert_sent(self, alert_id: int, notification_id: str) -> bool:
        """Mark an alert as sent with notification ID."""
        result = self.session.execute(
            update(SqueezeAlert).where(SqueezeAlert.id == alert_id).values(sent=True, notification_id=notification_id)
        )
        return getattr(result, "rowcount", 0) > 0

    def check_cooldown(self, ticker: str, alert_level: AlertLevel) -> bool:
        """Check if ticker is in cooldown period for alert level."""
        now = datetime.now(UTC)
        active_cooldown = self.session.execute(
            select(SqueezeAlert).where(
                and_(
                    SqueezeAlert.ticker == ticker.upper(),
                    SqueezeAlert.alert_level == alert_level.value,
                    SqueezeAlert.cooldown_expires > now,
                    SqueezeAlert.sent,
                )
            )
        ).scalar_one_or_none()

        return active_cooldown is not None

    def get_recent_alerts(self, days: int = 7) -> Sequence[SqueezeAlert]:
        """Get recent alerts within specified days."""
        cutoff_date = datetime.now(UTC) - timedelta(days=days)
        return list(
            self.session.execute(
                select(SqueezeAlert).where(SqueezeAlert.timestamp >= cutoff_date).order_by(desc(SqueezeAlert.timestamp))
            ).scalars()
        )

    def get_ticker_alert_history(self, ticker: str, days: int = 30) -> Sequence[SqueezeAlert]:
        """Get alert history for a ticker."""
        cutoff_date = datetime.now(UTC) - timedelta(days=days)
        return list(
            self.session.execute(
                select(SqueezeAlert)
                .where(and_(SqueezeAlert.ticker == ticker.upper(), SqueezeAlert.timestamp >= cutoff_date))
                .order_by(desc(SqueezeAlert.timestamp))
            ).scalars()
        )

    def cleanup_expired_cooldowns(self) -> int:
        """Remove expired cooldown records."""
        now = datetime.now(UTC)
        result = self.session.execute(
            delete(SqueezeAlert).where(and_(SqueezeAlert.cooldown_expires < now, SqueezeAlert.sent))
        )
        return result.rowcount  # type: ignore[attr-defined]


class AdHocCandidateRepo:
    """Repository for ad-hoc candidate operations."""

    def __init__(self, session: Session):
        self.session = session

    def add_candidate(self, ticker: str, reason: str, expires_at: datetime | None = None) -> AdHocCandidateModel:
        """Add a new ad-hoc candidate."""
        ticker = ticker.upper()

        # Check if candidate already exists
        existing = self.session.execute(
            select(AdHocCandidateModel).where(AdHocCandidateModel.ticker == ticker)
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
        candidate = AdHocCandidateModel(ticker=ticker, reason=reason, expires_at=expires_at)
        self.session.add(candidate)
        self.session.flush()
        return candidate

    def deactivate_candidate(self, ticker: str) -> bool:
        """Deactivate an ad-hoc candidate."""
        result = self.session.execute(
            update(AdHocCandidateModel).where(AdHocCandidateModel.ticker == ticker.upper()).values(active=False)
        )
        return getattr(result, "rowcount", 0) > 0

    def get_active_candidates(self) -> Sequence[AdHocCandidateModel]:
        """Get all active ad-hoc candidates."""
        return list(
            self.session.execute(
                select(AdHocCandidateModel)
                .where(AdHocCandidateModel.active)
                .order_by(AdHocCandidateModel.first_seen)
            ).scalars()
        )

    def get_candidate(self, ticker: str) -> AdHocCandidateModel | None:
        """Get a specific ad-hoc candidate."""
        return self.session.execute(
            select(AdHocCandidateModel).where(AdHocCandidateModel.ticker == ticker.upper())
        ).scalar_one_or_none()

    def expire_candidates(self) -> List[str]:
        """Expire candidates past their expiration date."""
        now = datetime.now(UTC)
        expired_candidates = list(
            self.session.execute(
                select(AdHocCandidateModel.ticker).where(
                    and_(AdHocCandidateModel.active, AdHocCandidateModel.expires_at < now)
                )
            ).scalars()
        )

        if expired_candidates:
            self.session.execute(
                update(AdHocCandidateModel)
                .where(and_(AdHocCandidateModel.active, AdHocCandidateModel.expires_at < now))
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
        return getattr(result, "rowcount", 0) > 0


class ShortSqueezeRepo:
    """Unified repository for all short squeeze operations."""

    def __init__(self, session: Session):
        self.session = session
        self.screener_snapshots = ScreenerSnapshotRepo(session)
        self.deep_scan_metrics = DeepScanMetricsRepo(session)
        self.alerts = SqueezeAlertRepo(session)
        self.adhoc_candidates = AdHocCandidateRepo(session)
        self.hn_corpus = HnCorpusRepo(session)
        self.sentiment_calibration = SentimentCalibrationRepo(session)

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

        _logger.info(
            "Found %d candidates for deep scan: %d from screener, %d ad-hoc",
            len(all_tickers),
            len(screener_tickers),
            len(adhoc_tickers),
        )

        return all_tickers

    def cleanup_old_data(self, days_to_keep: int = 90) -> Dict[str, int]:
        """Clean up old data beyond retention period."""
        cutoff_date = date.today() - timedelta(days=days_to_keep)
        cutoff_datetime = datetime.now(UTC) - timedelta(days=days_to_keep)

        # Clean up old snapshots
        snapshot_result = self.session.execute(delete(ScreenerSnapshot).where(ScreenerSnapshot.run_date < cutoff_date))

        # Clean up old deep scan metrics
        metrics_result = self.session.execute(delete(DeepScanMetrics).where(DeepScanMetrics.date < cutoff_date))

        # Clean up old alerts
        alerts_result = self.session.execute(delete(SqueezeAlert).where(SqueezeAlert.timestamp < cutoff_datetime))

        # FINRAShortInterest model is missing, removing cleanup


        cleanup_stats = {
            "snapshots_deleted": snapshot_result.rowcount,  # type: ignore[attr-defined]
            "metrics_deleted": metrics_result.rowcount,  # type: ignore[attr-defined]
            "alerts_deleted": alerts_result.rowcount,  # type: ignore[attr-defined]
        }

        _logger.info("Cleaned up old data: %s", cleanup_stats)
        return cleanup_stats
