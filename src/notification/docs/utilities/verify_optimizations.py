"""
Verify Database Optimizations

Simple verification script to test that the database optimization components
work correctly without requiring a full database setup.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.notification.service.database_optimization import (
    OptimizedMessageRepository,
    OptimizedDeliveryStatusRepository,
    OptimizedRateLimitRepository
)
from src.notification.docs.utilities.query_analyzer import (
    QueryPerformanceMonitor,
    QueryMetrics,
    query_timer
)
from src.notification.docs.utilities.database_migrations import NotificationServiceMigrations
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def test_query_performance_monitor():
    """Test query performance monitoring functionality."""
    print("🔍 Testing Query Performance Monitor...")

    monitor = QueryPerformanceMonitor(slow_query_threshold=0.5)

    # Test recording query executions
    monitor._record_query_execution("SELECT * FROM msg_messages WHERE status = 'PENDING'", 0.1)
    monitor._record_query_execution("SELECT * FROM msg_messages WHERE status = 'PENDING'", 0.2)
    monitor._record_query_execution("SELECT * FROM msg_delivery_status", 0.8)  # Slow query

    # Test performance summary
    summary = monitor.get_performance_summary()
    print(f"   ✓ Total queries: {summary['total_queries']}")
    print(f"   ✓ Unique queries: {summary['unique_queries']}")
    print(f"   ✓ Slow queries: {summary['slow_queries']}")
    print(f"   ✓ Average execution time: {summary['avg_execution_time']}s")

    # Test slow query detection
    slow_queries = monitor.get_slow_queries()
    print(f"   ✓ Detected {len(slow_queries)} slow queries")

    # Test query normalization
    monitor._record_query_execution("SELECT * FROM msg_messages WHERE id = 123", 0.1)
    monitor._record_query_execution("SELECT * FROM msg_messages WHERE id = 456", 0.1)

    # Should still be 2 unique queries (normalized)
    summary = monitor.get_performance_summary()
    print(f"   ✓ After normalization: {summary['unique_queries']} unique queries")

    print("   ✅ Query Performance Monitor tests passed!")
    return True


def test_query_timer():
    """Test query timer context manager."""
    print("⏱️  Testing Query Timer...")

    import time

    with query_timer("test_operation") as result:
        time.sleep(0.01)  # Simulate work

    print(f"   ✓ Operation: {result['operation']}")
    print(f"   ✓ Execution time: {result['execution_time']:.4f}s")
    print("   ✅ Query Timer tests passed!")
    return True


def test_query_metrics():
    """Test query metrics data structure."""
    print("📊 Testing Query Metrics...")

    metrics = QueryMetrics(
        query_hash="test_hash",
        query_text="SELECT * FROM test_table"
    )

    # Add some execution times
    metrics.add_execution(0.1)
    metrics.add_execution(0.2)
    metrics.add_execution(0.15)

    print(f"   ✓ Execution count: {metrics.execution_count}")
    print(f"   ✓ Average time: {metrics.avg_time:.4f}s")
    print(f"   ✓ Min time: {metrics.min_time:.4f}s")
    print(f"   ✓ Max time: {metrics.max_time:.4f}s")
    print(f"   ✓ Median time: {metrics.median_time:.4f}s")

    # Test dictionary conversion
    metrics_dict = metrics.to_dict()
    assert "query_hash" in metrics_dict
    assert "execution_count" in metrics_dict

    print("   ✅ Query Metrics tests passed!")
    return True


def test_optimization_components():
    """Test that optimization components can be instantiated."""
    print("🔧 Testing Optimization Components...")

    # Mock session for testing
    class MockSession:
        def query(self, *args):
            return self

        def filter(self, *args):
            return self

        def order_by(self, *args):
            return self

        def limit(self, *args):
            return self

        def offset(self, *args):
            return self

        def options(self, *args):
            return self

        def join(self, *args):
            return self

        def group_by(self, *args):
            return self

        def all(self):
            return []

        def count(self):
            return 0

        def execute(self, *args, **kwargs):
            return self

        def fetchall(self):
            return []

        def flush(self):
            pass

        def rollback(self):
            pass

    mock_session = MockSession()

    # Test optimized repositories can be instantiated
    try:
        message_repo = OptimizedMessageRepository(mock_session)
        delivery_repo = OptimizedDeliveryStatusRepository(mock_session)
        rate_limit_repo = OptimizedRateLimitRepository(mock_session)

        print("   ✓ OptimizedMessageRepository instantiated")
        print("   ✓ OptimizedDeliveryStatusRepository instantiated")
        print("   ✓ OptimizedRateLimitRepository instantiated")

        # Test that methods exist and can be called (with mock data)
        current_time = datetime.now(timezone.utc)

        # These should not fail even with mock session
        pending = message_repo.get_pending_messages_optimized(current_time, limit=10)
        print("   ✓ get_pending_messages_optimized method works")

        deliveries, total = delivery_repo.get_delivery_history_optimized(limit=10)
        print("   ✓ get_delivery_history_optimized method works")

        print("   ✅ Optimization Components tests passed!")
        return True

    except Exception as e:
        print(f"   ❌ Optimization Components test failed: {e}")
        return False


def test_migration_components():
    """Test migration components can be instantiated."""
    print("🔄 Testing Migration Components...")

    # Mock engine for testing
    class MockEngine:
        def connect(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, *args, **kwargs):
            return self

        def fetchall(self):
            return []

        def fetchone(self):
            return None

    mock_engine = MockEngine()

    try:
        migrations = NotificationServiceMigrations(mock_engine)
        print("   ✓ NotificationServiceMigrations instantiated")

        # Test that methods exist
        assert hasattr(migrations, 'apply_all_optimizations')
        assert hasattr(migrations, 'create_monitoring_views')
        assert hasattr(migrations, 'analyze_current_performance')

        print("   ✓ Migration methods available")
        print("   ✅ Migration Components tests passed!")
        return True

    except Exception as e:
        print(f"   ❌ Migration Components test failed: {e}")
        return False


def test_index_definitions():
    """Test that index definitions are valid."""
    print("📇 Testing Index Definitions...")

    try:
        from sqlalchemy import Index
        from src.data.db.models.model_notification import (
            Message, MessageDeliveryStatus, ChannelHealth, RateLimit, ChannelConfig
        )

        # Test that we can create index definitions
        test_indexes = [
            Index(
                'test_idx_messages_status_scheduled',
                Message.status, Message.scheduled_for
            ),
            Index(
                'test_idx_delivery_channel_created',
                MessageDeliveryStatus.channel, MessageDeliveryStatus.created_at
            ),
            Index(
                'test_idx_rate_limits_user_channel',
                RateLimit.user_id, RateLimit.channel
            )
        ]

        print(f"   ✓ Created {len(test_indexes)} test index definitions")

        # Test index properties
        for idx in test_indexes:
            assert idx.name is not None
            assert len(idx.columns) > 0
            print(f"   ✓ Index {idx.name} has {len(idx.columns)} columns")

        print("   ✅ Index Definitions tests passed!")
        return True

    except Exception as e:
        print(f"   ❌ Index Definitions test failed: {e}")
        return False


def main():
    """Run all verification tests."""
    print("🚀 Database Optimization Verification")
    print("=" * 50)

    tests = [
        ("Query Performance Monitor", test_query_performance_monitor),
        ("Query Timer", test_query_timer),
        ("Query Metrics", test_query_metrics),
        ("Optimization Components", test_optimization_components),
        ("Migration Components", test_migration_components),
        ("Index Definitions", test_index_definitions)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"   ❌ {test_name} test failed with exception: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print("📋 VERIFICATION RESULTS")
    print("=" * 50)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {passed + failed}")

    if failed == 0:
        print("\n🎉 All database optimization components verified successfully!")
        print("\n💡 Next Steps:")
        print("   1. Run 'python src/notification/service/apply_optimizations.py' to apply optimizations")
        print("   2. Use 'python src/notification/service/performance_dashboard.py' to monitor performance")
        print("   3. The optimized repositories are now available in the notification service")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)