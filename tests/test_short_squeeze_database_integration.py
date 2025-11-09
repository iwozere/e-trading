"""
Short Squeeze Detection Pipeline Database Integration Tests

This script tests the database integration for the short squeeze detection pipeline by:
1. Testing CRUD operations on all tables using repository and service layers
2. Testing direct usage of centralized service from pipeline modules
3. Testing data model conversions and validations
4. Testing business logic constraints and error handling
"""

from pathlib import Path
import sys

# Add src to path using pathlib
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pytest
from datetime import datetime, date
from decimal import Decimal

from src.data.db.core.database import session_scope, create_all_tables
from src.data.db.services.short_squeeze_service import ShortSqueezeService
from src.data.db.models.model_short_squeeze import (
    ScreenerSnapshot, DeepScanMetrics, SqueezeAlert, AdHocCandidateModel,
    AlertLevel
)
from src.ml.pipeline.p04_short_squeeze.core.models import (
    StructuralMetrics, TransientMetrics, Candidate, Alert, AdHocCandidate
)
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class TestShortSqueezeDatabase:
    """Test class for short squeeze database integration."""

    @classmethod
    def setup_class(cls):
        """Set up test database tables."""
        _logger.info("Setting up test database tables")
        create_all_tables()

    def test_database_connection(self):
        """Test database connection and session management."""
        _logger.info("Testing database connection")

        with session_scope() as session:
            service = ShortSqueezeService(session)
            stats = service.get_pipeline_statistics()

            assert 'status' in stats
            assert stats['status'] in ['healthy', 'error']
            _logger.info("Database connection test passed")

    def test_screener_snapshot_operations(self):
        """Test screener snapshot CRUD operations."""
        _logger.info("Testing screener snapshot operations")

        with session_scope() as session:
            service = ShortSqueezeService(session)

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

            _logger.info("Screener snapshot operations test passed")

    def test_deep_scan_metrics_operations(self):
        """Test deep scan metrics CRUD operations."""
        _logger.info("Testing deep scan metrics operations")

        with session_scope() as session:
            service = ShortSqueezeService(session)

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
            assert gme_score['volume_spike'] == 5.8

            # Test upsert functionality - update existing record
            updated_metrics = [{
                'ticker': 'TSLA',
                'volume_spike': Decimal('4.2'),
                'sentiment_24h': Decimal('0.75'),
                'squeeze_score': Decimal('0.82'),
                'alert_level': 'HIGH'
            }]

            updated_count = service.save_deep_scan_results(updated_metrics, scan_date)
            assert updated_count == 1

            # Verify update
            updated_scores = service.get_top_squeeze_scores(scan_date, limit=10)
            tsla_score = next((s for s in updated_scores if s['ticker'] == 'TSLA'), None)
            assert tsla_score is not None
            assert tsla_score['squeeze_score'] == 0.82
            assert tsla_score['alert_level'] == 'HIGH'

            _logger.info("Deep scan metrics operations test passed")

    def test_alert_operations(self):
        """Test alert CRUD operations and cooldown logic."""
        _logger.info("Testing alert operations")

        with session_scope() as session:
            service = ShortSqueezeService(session)

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

            # Test different alert level - should work
            alert_id_3 = service.create_alert(
                ticker='AAPL',
                alert_level=AlertLevel.MEDIUM,
                reason='Medium squeeze score',
                squeeze_score=0.65,
                cooldown_days=5
            )
            assert alert_id_3 is not None

            # This line is now moved above

            # Test different ticker - should work
            alert_id_4 = service.create_alert(
                ticker='MSFT',
                alert_level=AlertLevel.HIGH,
                reason='High squeeze score',
                squeeze_score=0.87,
                cooldown_days=7
            )
            assert alert_id_4 is not None

            _logger.info("Alert operations test passed")

    def test_adhoc_candidate_operations(self):
        """Test ad-hoc candidate CRUD operations."""
        _logger.info("Testing ad-hoc candidate operations")

        with session_scope() as session:
            service = ShortSqueezeService(session)

            # Add ad-hoc candidates
            success_1 = service.add_adhoc_candidate('NVDA', 'Unusual options activity', ttl_days=7)
            assert success_1 is True

            success_2 = service.add_adhoc_candidate('AMD', 'High short interest reported', ttl_days=10)
            assert success_2 is True

            # Test duplicate - should reactivate
            success_3 = service.add_adhoc_candidate('NVDA', 'Updated reason', ttl_days=5)
            assert success_3 is True

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

            _logger.info("Ad-hoc candidate operations test passed")

    def test_data_model_conversions(self):
        """Test conversions between database models and business logic models."""
        _logger.info("Testing data model conversions")

        with session_scope() as session:
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
            session.add(snapshot)
            session.flush()

            # Test conversion to StructuralMetrics
            structural_metrics = snapshot.to_structural_metrics()
            assert structural_metrics is not None
            assert isinstance(structural_metrics, StructuralMetrics)
            assert structural_metrics.short_interest_pct == 0.25
            assert structural_metrics.days_to_cover == 5.5
            assert structural_metrics.float_shares == 1000000

            # Create deep scan metrics
            deep_metrics = DeepScanMetrics(
                ticker=test_ticker,
                date=date.today(),
                volume_spike=Decimal('3.2'),
                call_put_ratio=Decimal('1.5'),
                sentiment_24h=Decimal('0.6'),
                borrow_fee_pct=Decimal('0.1'),
                squeeze_score=Decimal('0.8')
            )
            session.add(deep_metrics)
            session.flush()

            # Test conversion to TransientMetrics
            transient_metrics = deep_metrics.to_transient_metrics()
            assert transient_metrics is not None
            assert isinstance(transient_metrics, TransientMetrics)
            assert transient_metrics.volume_spike == 3.2
            assert transient_metrics.sentiment_24h == 0.6

            # Create alert
            alert = SqueezeAlert(
                ticker=test_ticker,
                alert_level='HIGH',
                reason='Test alert',
                squeeze_score=Decimal('0.85'),
                sent=True,
                notification_id='test_123'
            )
            session.add(alert)
            session.flush()

            # Test conversion to Alert
            alert_model = alert.to_alert()
            assert alert_model is not None
            assert isinstance(alert_model, Alert)
            assert alert_model.ticker == test_ticker
            assert alert_model.alert_level == AlertLevel.HIGH
            assert alert_model.sent is True

            # Create ad-hoc candidate
            adhoc = AdHocCandidateModel(
                ticker=test_ticker,
                reason='Test candidate',
                active=True,
                promoted_by_screener=False
            )
            session.add(adhoc)
            session.flush()

            # Test conversion to AdHocCandidate
            adhoc_model = adhoc.to_adhoc_candidate()
            assert adhoc_model is not None
            assert isinstance(adhoc_model, AdHocCandidate)
            assert adhoc_model.ticker == test_ticker
            assert adhoc_model.active is True

            _logger.info("Data model conversions test passed")

    def test_business_logic_validations(self):
        """Test business logic validations in dataclasses."""
        _logger.info("Testing business logic validations")

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

        # Test TransientMetrics validation
        with pytest.raises(ValueError, match="Volume spike must be non-negative"):
            TransientMetrics(
                volume_spike=-1.0,  # Invalid
                call_put_ratio=1.5,
                sentiment_24h=0.6,
                borrow_fee_pct=0.1
            )

        with pytest.raises(ValueError, match="Sentiment must be between -1 and 1"):
            TransientMetrics(
                volume_spike=3.0,
                call_put_ratio=1.5,
                sentiment_24h=1.5,  # Invalid
                borrow_fee_pct=0.1
            )

        # Test Candidate validation
        with pytest.raises(ValueError, match="Ticker cannot be empty"):
            Candidate(
                ticker="",  # Invalid
                screener_score=0.8,
                structural_metrics=StructuralMetrics(0.25, 5.0, 1000000, 500000, 1000000000),
                last_updated=datetime.now()
            )

        with pytest.raises(ValueError, match="Screener score must be between 0 and 1"):
            Candidate(
                ticker="TEST",
                screener_score=1.5,  # Invalid
                structural_metrics=StructuralMetrics(0.25, 5.0, 1000000, 500000, 1000000000),
                last_updated=datetime.now()
            )

        _logger.info("Business logic validations test passed")

    def test_comprehensive_ticker_analysis(self):
        """Test comprehensive ticker analysis functionality."""
        _logger.info("Testing comprehensive ticker analysis")

        with session_scope() as session:
            service = ShortSqueezeService(session)

            # Set up test data for a ticker
            ticker = 'COMP'

            # Add screener data
            screener_results = [{
                'ticker': ticker,
                'short_interest_pct': Decimal('0.28'),
                'days_to_cover': Decimal('6.5'),
                'float_shares': 500000000,
                'avg_volume_14d': 25000000,
                'market_cap': 50000000000,
                'screener_score': Decimal('0.82'),
                'data_quality': Decimal('0.96')
            }]
            service.save_screener_results(screener_results, date.today())

            # Add deep scan data
            deep_scan_results = [{
                'ticker': ticker,
                'volume_spike': Decimal('4.2'),
                'call_put_ratio': Decimal('1.8'),
                'sentiment_24h': Decimal('0.72'),
                'borrow_fee_pct': Decimal('0.12'),
                'squeeze_score': Decimal('0.86'),
                'alert_level': 'HIGH'
            }]
            service.save_deep_scan_results(deep_scan_results, date.today())

            # Add alert
            service.create_alert(ticker, AlertLevel.HIGH, 'High squeeze probability', 0.86)

            # Add as ad-hoc candidate
            service.add_adhoc_candidate(ticker, 'Comprehensive test candidate')

            # Get comprehensive analysis
            analysis = service.get_ticker_analysis(ticker, days=30)

            # Verify analysis structure
            assert analysis['ticker'] == ticker
            assert 'screener_history' in analysis
            assert 'metrics_history' in analysis
            assert 'alert_history' in analysis
            assert 'adhoc_candidate' in analysis

            # Verify screener history
            assert len(analysis['screener_history']) >= 1
            screener_entry = analysis['screener_history'][0]
            assert screener_entry['screener_score'] == 0.82
            assert screener_entry['short_interest_pct'] == 0.28

            # Verify metrics history
            assert len(analysis['metrics_history']) >= 1
            metrics_entry = analysis['metrics_history'][0]
            assert metrics_entry['squeeze_score'] == 0.86
            assert metrics_entry['alert_level'] == 'HIGH'

            # Verify alert history
            assert len(analysis['alert_history']) >= 1
            alert_entry = analysis['alert_history'][0]
            assert alert_entry['alert_level'] == 'HIGH'
            assert alert_entry['squeeze_score'] == 0.86

            # Verify ad-hoc candidate
            assert analysis['adhoc_candidate'] is not None
            assert analysis['adhoc_candidate']['active'] is True
            assert analysis['adhoc_candidate']['reason'] == 'Comprehensive test candidate'

            _logger.info("Comprehensive ticker analysis test passed")

    def test_pipeline_statistics_and_cleanup(self):
        """Test pipeline statistics and data cleanup functionality."""
        _logger.info("Testing pipeline statistics and cleanup")

        with session_scope() as session:
            service = ShortSqueezeService(session)

            # Get pipeline statistics
            stats = service.get_pipeline_statistics()
            assert 'status' in stats
            assert 'latest_screener_run' in stats
            assert 'active_adhoc_candidates' in stats
            assert 'recent_alerts_7d' in stats
            assert 'todays_deep_scan_count' in stats

            # Test cleanup (with 0 days to clean everything for testing)
            cleanup_stats = service.cleanup_old_data(days_to_keep=0)
            assert 'snapshots_deleted' in cleanup_stats
            assert 'metrics_deleted' in cleanup_stats
            assert 'alerts_deleted' in cleanup_stats

            # Verify cleanup worked - check that cleanup stats are valid integers
            assert isinstance(cleanup_stats['snapshots_deleted'], int)
            assert isinstance(cleanup_stats['metrics_deleted'], int)
            assert isinstance(cleanup_stats['alerts_deleted'], int)

            _logger.info("Pipeline statistics and cleanup test passed")

    def test_error_handling(self):
        """Test error handling in service operations."""
        _logger.info("Testing error handling")

        with session_scope() as session:
            service = ShortSqueezeService(session)

            # Test invalid ticker for ad-hoc candidate
            success = service.add_adhoc_candidate('', 'Empty ticker test')
            assert success is False

            # Test removing non-existent ad-hoc candidate
            removed = service.remove_adhoc_candidate('NONEXISTENT')
            assert removed is False

            # Test marking non-existent alert as sent
            marked = service.mark_alert_sent(99999, 'fake_notification')
            assert marked is False

            _logger.info("Error handling test passed")

    def cleanup_test_data(self):
        """Clean up test data after tests."""
        _logger.info("Cleaning up test data")

        with session_scope() as session:
            service = ShortSqueezeService(session)

            # Clean up all test data
            cleanup_stats = service.cleanup_old_data(days_to_keep=0)
            _logger.info("Cleanup completed: %s", cleanup_stats)


def run_tests():
    """Run all short squeeze database integration tests."""
    _logger.info("Starting Short Squeeze Database Integration Tests")
    print("=" * 60)

    test_instance = TestShortSqueezeDatabase()
    test_instance.setup_class()

    tests = [
        ("Database Connection", test_instance.test_database_connection),
        ("Screener Snapshot Operations", test_instance.test_screener_snapshot_operations),
        ("Deep Scan Metrics Operations", test_instance.test_deep_scan_metrics_operations),
        ("Alert Operations", test_instance.test_alert_operations),
        ("Ad-hoc Candidate Operations", test_instance.test_adhoc_candidate_operations),
        ("Data Model Conversions", test_instance.test_data_model_conversions),
        ("Business Logic Validations", test_instance.test_business_logic_validations),
        ("Comprehensive Ticker Analysis", test_instance.test_comprehensive_ticker_analysis),
        ("Pipeline Statistics and Cleanup", test_instance.test_pipeline_statistics_and_cleanup),
        ("Error Handling", test_instance.test_error_handling),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Running: {test_name}")
            test_func()
            print(f"✅ {test_name} - PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {test_name} - FAILED: {e}")
            _logger.exception("Test %s failed:", test_name)

    # Cleanup
    try:
        test_instance.cleanup_test_data()
    except Exception as e:
        _logger.warning("Cleanup failed: %s", e)

    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Short squeeze database integration is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)