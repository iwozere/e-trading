"""
Short Squeeze Detection Pipeline Database Integration Tests

This script tests the database integration for the short squeeze detection pipeline by:
1. Testing CRUD operations on all tables using repository and service layers
2. Testing direct usage of centralized service from pipeline modules
3. Testing data model conversions and validations
4. Testing business logic constraints and error handling

IMPORTANT: This test uses isolated test database fixtures, NOT production database.
"""

from pathlib import Path
import sys

# Add src to path using pathlib
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

import pytest
from datetime import datetime, date
from decimal import Decimal

from src.data.db.services.short_squeeze_service import ShortSqueezeService
from src.data.db.models.model_short_squeeze import (
    ScreenerSnapshot, DeepScanMetrics, SqueezeAlert, AdHocCandidateModel,
    AlertLevel
)
from src.ml.pipeline.p04_short_squeeze.core.models import (
    StructuralMetrics, TransientMetrics, Candidate, Alert, AdHocCandidate
)


class TestShortSqueezeDatabase:
    """Test class for short squeeze database integration using isolated test database."""

    def test_database_connection(self, db_session):
        """Test database connection and session management."""
        service = ShortSqueezeService(db_session)
        stats = service.get_pipeline_statistics()

        assert 'status' in stats
        assert stats['status'] in ['healthy', 'error']
        print("✅ Database connection test passed")

    def test_screener_snapshot_operations(self, db_session):
        """Test screener snapshot CRUD operations."""
        service = ShortSqueezeService(db_session)

        # Test data
        run_date = date.today()
        test_results = [
            {
                'ticker': 'TSLA',
                'short_interest_pct': Decimal('0.25'),
                'days_to_cover': Decimal('5.5'),
                'float_shares': 1000000000,
                'avg_volume_14d': 50000000,
                'market_cap': 800000000000,
                'screener_score': Decimal('0.85'),
                'raw_payload': {'test': True},
                'data_quality': Decimal('0.95')
            },
            {
                'ticker': 'GME',
                'short_interest_pct': Decimal('0.30'),
                'days_to_cover': Decimal('8.2'),
                'float_shares': 76000000,
                'avg_volume_14d': 15000000,
                'market_cap': 12000000000,
                'screener_score': Decimal('0.92'),
                'raw_payload': {'test': True},
                'data_quality': Decimal('0.98')
            }
        ]

        # Save screener results
        saved_count = service.save_screener_results(test_results, run_date)
        assert saved_count == 2

        # Get top candidates
        top_candidates = service.get_top_candidates_by_screener_score(limit=10)
        assert len(top_candidates) >= 2

        # Verify data
        gme_candidate = next((c for c in top_candidates if c['ticker'] == 'GME'), None)
        assert gme_candidate is not None
        assert gme_candidate['screener_score'] == 0.92
        assert gme_candidate['short_interest_pct'] == 0.30
        print("✅ Screener snapshot operations test passed")

    def test_deep_scan_metrics_operations(self, db_session):
        """Test deep scan metrics CRUD operations."""
        service = ShortSqueezeService(db_session)

        # Test data
        scan_date = date.today()
        test_metrics = [
            {
                'ticker': 'TSLA',
                'volume_spike': Decimal('3.5'),
                'call_put_ratio': Decimal('1.2'),
                'sentiment_24h': Decimal('0.65'),
                'borrow_fee_pct': Decimal('0.08'),
                'squeeze_score': Decimal('0.78'),
                'alert_level': 'MEDIUM',
                'raw_payload': {'test': True}
            },
            {
                'ticker': 'GME',
                'volume_spike': Decimal('5.8'),
                'call_put_ratio': Decimal('2.1'),
                'sentiment_24h': Decimal('0.82'),
                'borrow_fee_pct': Decimal('0.15'),
                'squeeze_score': Decimal('0.91'),
                'alert_level': 'HIGH',
                'raw_payload': {'test': True}
            }
        ]

        # Save deep scan results
        saved_count = service.save_deep_scan_results(test_metrics, scan_date)
        assert saved_count == 2

        # Get top squeeze scores
        top_scores = service.get_top_squeeze_scores(scan_date, limit=10)
        assert len(top_scores) >= 2

        # Verify data
        gme_score = next((s for s in top_scores if s['ticker'] == 'GME'), None)
        assert gme_score is not None
        assert gme_score['squeeze_score'] == 0.91
        assert gme_score['alert_level'] == 'HIGH'
        print("✅ Deep scan metrics operations test passed")

    def test_alert_operations(self, db_session):
        """Test alert CRUD operations and cooldown logic."""
        service = ShortSqueezeService(db_session)

        # Create alerts
        alert_id_1 = service.create_alert(
            ticker='AAPL',
            alert_level=AlertLevel.HIGH,
            reason='High squeeze score with volume spike',
            squeeze_score=0.89,
            cooldown_days=7
        )
        assert alert_id_1 is not None

        # Mark first alert as sent to activate cooldown
        success = service.mark_alert_sent(alert_id_1, 'notification_123')
        assert success is True

        # Test cooldown - should not create duplicate alert
        alert_id_2 = service.create_alert(
            ticker='AAPL',
            alert_level=AlertLevel.HIGH,
            reason='Another high squeeze score',
            squeeze_score=0.91,
            cooldown_days=7
        )
        assert alert_id_2 is None  # Should be blocked by cooldown

        # Test different ticker - should work
        alert_id_3 = service.create_alert(
            ticker='MSFT',
            alert_level=AlertLevel.HIGH,
            reason='High squeeze score',
            squeeze_score=0.87,
            cooldown_days=7
        )
        assert alert_id_3 is not None
        print("✅ Alert operations test passed")

    def test_adhoc_candidate_operations(self, db_session):
        """Test ad-hoc candidate CRUD operations."""
        service = ShortSqueezeService(db_session)

        # Add ad-hoc candidates
        success_1 = service.add_adhoc_candidate('NVDA', 'Unusual options activity', ttl_days=7)
        assert success_1 is True

        success_2 = service.add_adhoc_candidate('AMD', 'High short interest reported', ttl_days=10)
        assert success_2 is True

        # Get candidates for deep scan
        candidates = service.get_candidates_for_deep_scan()
        assert 'NVDA' in candidates
        assert 'AMD' in candidates

        # Remove candidate
        removed = service.remove_adhoc_candidate('AMD')
        assert removed is True

        # Verify removal
        updated_candidates = service.get_candidates_for_deep_scan()
        assert 'AMD' not in updated_candidates or len([c for c in updated_candidates if c == 'AMD']) == 0
        print("✅ Ad-hoc candidate operations test passed")

    def test_data_model_conversions(self, db_session):
        """Test conversions between database models and business logic models."""
        # Create a screener snapshot with unique ticker
        test_ticker = f'TEST{datetime.now().microsecond}'
        snapshot = ScreenerSnapshot(
            ticker=test_ticker,
            run_date=date.today(),
            short_interest_pct=Decimal('0.25'),
            days_to_cover=Decimal('5.5'),
            float_shares=1000000,
            avg_volume_14d=500000,
            market_cap=1000000000,
            screener_score=Decimal('0.75')
        )
        db_session.add(snapshot)
        db_session.flush()

        # Test conversion to StructuralMetrics
        structural_metrics = snapshot.to_structural_metrics()
        assert structural_metrics is not None
        assert isinstance(structural_metrics, StructuralMetrics)
        assert structural_metrics.short_interest_pct == 0.25
        assert structural_metrics.days_to_cover == 5.5
        print("✅ Data model conversions test passed")

    def test_business_logic_validations(self):
        """Test business logic validations in dataclasses."""
        # Test StructuralMetrics validation
        with pytest.raises(ValueError, match="Short interest percentage must be between 0 and 1"):
            StructuralMetrics(
                short_interest_pct=1.5,  # Invalid
                days_to_cover=5.0,
                float_shares=1000000,
                avg_volume_14d=500000,
                market_cap=1000000000
            )

        with pytest.raises(ValueError, match="Days to cover must be non-negative"):
            StructuralMetrics(
                short_interest_pct=0.25,
                days_to_cover=-1.0,  # Invalid
                float_shares=1000000,
                avg_volume_14d=500000,
                market_cap=1000000000
            )
        print("✅ Business logic validations test passed")

    def test_pipeline_statistics_and_cleanup(self, db_session):
        """Test pipeline statistics and data cleanup functionality."""
        service = ShortSqueezeService(db_session)

        # Get pipeline statistics
        stats = service.get_pipeline_statistics()
        assert 'status' in stats
        assert 'latest_screener_run' in stats
        assert 'active_adhoc_candidates' in stats

        # Test cleanup (with 0 days to clean everything for testing)
        cleanup_stats = service.cleanup_old_data(days_to_keep=0)
        assert 'snapshots_deleted' in cleanup_stats
        assert 'metrics_deleted' in cleanup_stats
        assert 'alerts_deleted' in cleanup_stats

        # Verify cleanup stats are valid integers
        assert isinstance(cleanup_stats['snapshots_deleted'], int)
        assert isinstance(cleanup_stats['metrics_deleted'], int)
        assert isinstance(cleanup_stats['alerts_deleted'], int)
        print("✅ Pipeline statistics and cleanup test passed")

    def test_error_handling(self, db_session):
        """Test error handling in service operations."""
        service = ShortSqueezeService(db_session)

        # Test invalid ticker for ad-hoc candidate
        success = service.add_adhoc_candidate('', 'Empty ticker test')
        assert success is False

        # Test removing non-existent ad-hoc candidate
        removed = service.remove_adhoc_candidate('NONEXISTENT')
        assert removed is False

        # Test marking non-existent alert as sent
        marked = service.mark_alert_sent(99999, 'fake_notification')
        assert marked is False
        print("✅ Error handling test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
