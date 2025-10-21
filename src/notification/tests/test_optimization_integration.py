"""
Integration Test for Database Optimizations

Simple integration test to verify that the database optimization components
work correctly with real database operations.
"""

import pytest
from datetime import datetime, timedelta, timezone
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.data.db.core.base import Base
from src.data.db.models.model_notification import (
    Message, MessageDeliveryStatus, MessagePriority, MessageStatus, DeliveryStatus
)
from src.notification.service.database_optimization import (
    OptimizedMessageRepository,
    OptimizedDeliveryStatusRepository
)
from src.notification.docs.utilities.query_analyzer import QueryPerformanceMonitor


@pytest.fixture(scope="session")
def test_engine():
    """Create a test database engine."""
    # Use in-memory SQLite for testing
    engine = create_engine("sqlite:///:memory:", echo=False)

    # Create all tables
    Base.metadata.create_all(engine)

    return engine


@pytest.fixture
def test_session(test_engine):
    """Create a test database session."""
    Session = sessionmaker(bind=test_engine)
    session = Session()

    yield session

    session.close()


@pytest.fixture
def sample_messages(test_session):
    """Create sample messages for testing."""
    messages = []

    for i in range(5):
        message = Message(
            message_type="test_alert",
            priority=MessagePriority.NORMAL.value,
            channels=["telegram", "email"],
            recipient_id=f"user_{i}",
            content={"title": f"Test Message {i}", "body": f"This is test message {i}"},
            status=MessageStatus.PENDING.value,
            created_at=datetime.now(timezone.utc) - timedelta(minutes=i),
            scheduled_for=datetime.now(timezone.utc) - timedelta(minutes=i)
        )
        test_session.add(message)
        messages.append(message)

    test_session.commit()
    return messages


@pytest.fixture
def sample_delivery_statuses(test_session, sample_messages):
    """Create sample delivery statuses for testing."""
    statuses = []

    for i, message in enumerate(sample_messages):
        for channel in ["telegram", "email"]:
            status = MessageDeliveryStatus(
                message_id=message.id,
                channel=channel,
                status=DeliveryStatus.DELIVERED.value if i % 2 == 0 else DeliveryStatus.FAILED.value,
                delivered_at=datetime.now(timezone.utc) - timedelta(minutes=i),
                response_time_ms=100 + (i * 50),
                created_at=datetime.now(timezone.utc) - timedelta(minutes=i)
            )
            test_session.add(status)
            statuses.append(status)

    test_session.commit()
    return statuses


class TestOptimizedRepositoryIntegration:
    """Test optimized repositories with real database operations."""

    def test_optimized_message_repository_pending_messages(self, test_session, sample_messages):
        """Test optimized pending messages query."""
        repo = OptimizedMessageRepository(test_session)

        current_time = datetime.now(timezone.utc)
        pending_messages = repo.get_pending_messages_optimized(current_time, limit=10)

        # Should return all pending messages
        assert len(pending_messages) == 5

        # Should be ordered by priority and scheduled_for
        for message in pending_messages:
            assert message.status == MessageStatus.PENDING.value

    def test_optimized_message_repository_bulk_update(self, test_session, sample_messages):
        """Test bulk message status update."""
        repo = OptimizedMessageRepository(test_session)

        message_ids = [msg.id for msg in sample_messages[:3]]
        processed_at = datetime.now(timezone.utc)

        updated_count = repo.bulk_update_message_status(
            message_ids,
            MessageStatus.PROCESSING,
            processed_at
        )

        assert updated_count == 3

        # Verify messages were updated
        for message_id in message_ids:
            message = test_session.query(Message).filter(Message.id == message_id).first()
            assert message.status == MessageStatus.PROCESSING.value
            assert message.processed_at == processed_at

    def test_optimized_message_repository_statistics(self, test_session, sample_messages):
        """Test optimized message statistics query."""
        repo = OptimizedMessageRepository(test_session)

        start_date = datetime.now(timezone.utc) - timedelta(days=1)
        end_date = datetime.now(timezone.utc) + timedelta(hours=1)

        stats = repo.get_message_statistics_optimized(start_date, end_date)

        # Should return statistics for our test messages
        assert len(stats) > 0

        # Verify statistics structure
        for stat in stats:
            assert "time_period" in stat
            assert "status" in stat
            assert "priority" in stat
            assert "message_count" in stat

    def test_optimized_delivery_status_repository_history(self, test_session, sample_delivery_statuses):
        """Test optimized delivery history query."""
        repo = OptimizedDeliveryStatusRepository(test_session)

        deliveries, total_count = repo.get_delivery_history_optimized(
            channel="telegram",
            limit=5,
            offset=0
        )

        # Should return telegram deliveries
        assert len(deliveries) > 0
        assert total_count > 0

        for delivery in deliveries:
            assert delivery.channel == "telegram"

    def test_optimized_delivery_status_repository_performance_metrics(self, test_session, sample_delivery_statuses):
        """Test channel performance metrics query."""
        repo = OptimizedDeliveryStatusRepository(test_session)

        start_date = datetime.now(timezone.utc) - timedelta(days=1)
        end_date = datetime.now(timezone.utc) + timedelta(hours=1)

        metrics = repo.get_channel_performance_metrics(start_date, end_date)

        # Should return metrics for both channels
        assert "telegram" in metrics
        assert "email" in metrics

        # Verify metrics structure
        for channel, channel_metrics in metrics.items():
            assert "total_attempts" in channel_metrics
            assert "successful_deliveries" in channel_metrics
            assert "success_rate" in channel_metrics


class TestQueryPerformanceMonitorIntegration:
    """Test query performance monitoring with real queries."""

    def test_query_monitoring_basic_functionality(self, test_engine):
        """Test basic query monitoring functionality."""
        monitor = QueryPerformanceMonitor(slow_query_threshold=0.1)

        # Test recording query execution
        monitor._record_query_execution("SELECT 1", 0.05)
        monitor._record_query_execution("SELECT 2", 0.15)  # Slow query

        # Verify metrics were recorded
        assert len(monitor.query_metrics) == 2

        # Test performance summary
        summary = monitor.get_performance_summary()
        assert summary["total_queries"] == 2
        assert summary["unique_queries"] == 2
        assert summary["slow_queries"] == 1

    def test_query_normalization(self):
        """Test query normalization for grouping."""
        monitor = QueryPerformanceMonitor()

        # These should be normalized to the same query
        queries = [
            "SELECT * FROM msg_messages WHERE id = 123",
            "SELECT * FROM msg_messages WHERE id = 456",
            "select * from msg_messages where id = 789"
        ]

        for query in queries:
            monitor._record_query_execution(query, 0.1)

        # Should be grouped into single normalized query
        assert len(monitor.query_metrics) == 1

        metrics = list(monitor.query_metrics.values())[0]
        assert metrics.execution_count == 3

    def test_slow_query_detection(self):
        """Test slow query detection."""
        monitor = QueryPerformanceMonitor(slow_query_threshold=0.5)

        # Record normal and slow queries
        monitor._record_query_execution("SELECT 1", 0.1)  # Normal
        monitor._record_query_execution("SELECT 2", 0.8)  # Slow

        slow_queries = monitor.get_slow_queries()
        assert len(slow_queries) == 1
        assert slow_queries[0]["avg_time"] == 0.8


class TestDatabaseOptimizationWorkflow:
    """Test the complete database optimization workflow."""

    def test_repository_integration_with_optimizations(self, test_session, sample_messages, sample_delivery_statuses):
        """Test that optimized repositories work correctly together."""
        message_repo = OptimizedMessageRepository(test_session)
        delivery_repo = OptimizedDeliveryStatusRepository(test_session)

        # Test message operations
        current_time = datetime.now(timezone.utc)
        pending_messages = message_repo.get_pending_messages_optimized(current_time)
        assert len(pending_messages) > 0

        # Test delivery operations
        deliveries, total = delivery_repo.get_delivery_history_optimized(limit=10)
        assert len(deliveries) > 0
        assert total > 0

        # Test performance metrics
        start_date = datetime.now(timezone.utc) - timedelta(days=1)
        end_date = datetime.now(timezone.utc) + timedelta(hours=1)

        metrics = delivery_repo.get_channel_performance_metrics(start_date, end_date)
        assert len(metrics) > 0

    def test_query_performance_with_real_operations(self, test_session, sample_messages):
        """Test query performance monitoring with real database operations."""
        monitor = QueryPerformanceMonitor(slow_query_threshold=0.1)
        repo = OptimizedMessageRepository(test_session)

        # Perform some database operations
        current_time = datetime.now(timezone.utc)

        # This should generate some query metrics if monitoring was enabled
        pending_messages = repo.get_pending_messages_optimized(current_time)
        assert len(pending_messages) > 0

        # Test that we can get performance summary
        summary = monitor.get_performance_summary()
        assert isinstance(summary, dict)
        assert "total_queries" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])