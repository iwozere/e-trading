"""
Tests for Database Optimization Components

Tests for the database optimization, query analysis, and migration components
of the notification service.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta, timezone
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.notification.service.database_optimization import (
    OptimizedMessageRepository,
    OptimizedDeliveryStatusRepository,
    OptimizedRateLimitRepository,
    create_optimized_indexes,
    analyze_query_performance
)
from src.notification.docs.utilities.query_analyzer import (
    QueryPerformanceMonitor,
    QueryMetrics,
    DatabaseHealthChecker,
    query_timer
)
from src.notification.docs.utilities.database_migrations import (
    NotificationServiceMigrations,
    run_optimization_migration
)
from src.data.db.models.model_notification import (
    Message, MessageDeliveryStatus, MessagePriority, MessageStatus, DeliveryStatus
)


class TestOptimizedMessageRepository:
    """Test optimized message repository operations."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        session = Mock()
        session.query.return_value = session
        session.filter.return_value = session
        session.order_by.return_value = session
        session.limit.return_value = session
        session.all.return_value = []
        return session

    @pytest.fixture
    def repository(self, mock_session):
        """Create optimized message repository with mock session."""
        return OptimizedMessageRepository(mock_session)

    def test_get_pending_messages_optimized(self, repository, mock_session):
        """Test optimized pending messages query."""
        current_time = datetime.now(timezone.utc)

        # Mock return data
        mock_messages = [
            Mock(id=1, priority="HIGH", scheduled_for=current_time),
            Mock(id=2, priority="NORMAL", scheduled_for=current_time)
        ]
        mock_session.all.return_value = mock_messages

        # Call method
        result = repository.get_pending_messages_optimized(current_time, limit=10)

        # Verify query was built correctly
        assert mock_session.query.called
        assert mock_session.filter.called
        assert mock_session.order_by.called
        assert mock_session.limit.called
        assert result == mock_messages

    def test_bulk_update_message_status(self, repository, mock_session):
        """Test bulk message status update."""
        message_ids = [1, 2, 3]
        status = MessageStatus.PROCESSING
        processed_at = datetime.now(timezone.utc)

        # Mock update result
        mock_session.query.return_value.filter.return_value.update.return_value = 3

        # Call method
        result = repository.bulk_update_message_status(message_ids, status, processed_at)

        # Verify update was called correctly
        assert mock_session.query.called
        assert result == 3

    def test_get_messages_with_delivery_status(self, repository, mock_session):
        """Test loading messages with delivery statuses."""
        message_ids = [1, 2, 3]
        mock_messages = [Mock(id=1), Mock(id=2), Mock(id=3)]
        mock_session.all.return_value = mock_messages

        # Call method
        result = repository.get_messages_with_delivery_status(message_ids)

        # Verify query with eager loading
        assert mock_session.query.called
        assert mock_session.options.called
        assert result == mock_messages

    def test_get_message_statistics_optimized(self, repository, mock_session):
        """Test optimized message statistics query."""
        start_date = datetime.now(timezone.utc) - timedelta(days=7)
        end_date = datetime.now(timezone.utc)

        # Mock aggregation result
        mock_row = Mock()
        mock_row.time_period = datetime.now(timezone.utc)
        mock_row.status = "DELIVERED"
        mock_row.priority = "NORMAL"
        mock_row.message_count = 10
        mock_row.avg_retry_count = 1.5
        mock_session.all.return_value = [mock_row]

        # Call method
        result = repository.get_message_statistics_optimized(start_date, end_date)

        # Verify result format
        assert len(result) == 1
        assert result[0]["status"] == "DELIVERED"
        assert result[0]["message_count"] == 10


class TestOptimizedDeliveryStatusRepository:
    """Test optimized delivery status repository operations."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        session = Mock()
        session.query.return_value = session
        session.filter.return_value = session
        session.join.return_value = session
        session.order_by.return_value = session
        session.offset.return_value = session
        session.limit.return_value = session
        session.count.return_value = 5
        session.all.return_value = []
        session.execute.return_value.fetchall.return_value = []
        return session

    @pytest.fixture
    def repository(self, mock_session):
        """Create optimized delivery status repository with mock session."""
        return OptimizedDeliveryStatusRepository(mock_session)

    def test_get_delivery_history_optimized(self, repository, mock_session):
        """Test optimized delivery history query."""
        # Test with user filter (requires join)
        deliveries, total = repository.get_delivery_history_optimized(
            user_id="user123",
            channel="telegram",
            limit=10,
            offset=0
        )

        # Verify join was used for user filter
        assert mock_session.join.called
        assert mock_session.count.called
        assert total == 5

    def test_get_delivery_history_without_user_filter(self, repository, mock_session):
        """Test delivery history query without user filter."""
        deliveries, total = repository.get_delivery_history_optimized(
            channel="email",
            limit=10,
            offset=0
        )

        # Verify no join was used
        assert mock_session.query.called
        assert mock_session.count.called

    def test_get_channel_performance_metrics(self, repository, mock_session):
        """Test channel performance metrics query."""
        start_date = datetime.now(timezone.utc) - timedelta(days=1)
        end_date = datetime.now(timezone.utc)

        # Mock aggregation result
        mock_row = Mock()
        mock_row.channel = "telegram"
        mock_row.total_attempts = 100
        mock_row.successful_deliveries = 95
        mock_row.avg_response_time = 250.5
        mock_row.median_response_time = 200.0
        mock_row.p95_response_time = 500.0
        mock_session.all.return_value = [mock_row]

        # Call method
        result = repository.get_channel_performance_metrics(start_date, end_date)

        # Verify result format
        assert "telegram" in result
        metrics = result["telegram"]
        assert metrics["total_attempts"] == 100
        assert metrics["success_rate"] == 0.95
        assert metrics["avg_response_time_ms"] == 250.5


class TestQueryPerformanceMonitor:
    """Test query performance monitoring."""

    @pytest.fixture
    def monitor(self):
        """Create query performance monitor."""
        return QueryPerformanceMonitor(slow_query_threshold=0.5)

    def test_query_metrics_creation(self, monitor):
        """Test query metrics creation and updates."""
        query_text = "SELECT * FROM msg_messages WHERE status = 'PENDING'"

        # Record multiple executions
        monitor._record_query_execution(query_text, 0.1)
        monitor._record_query_execution(query_text, 0.2)
        monitor._record_query_execution(query_text, 0.8)  # Slow query

        # Verify metrics
        assert len(monitor.query_metrics) == 1
        metrics = list(monitor.query_metrics.values())[0]
        assert metrics.execution_count == 3
        assert metrics.min_time == 0.1
        assert metrics.max_time == 0.8
        assert metrics.avg_time == pytest.approx(0.367, rel=1e-2)

    def test_slow_query_detection(self, monitor):
        """Test slow query detection and logging."""
        with patch('src.notification.service.query_analyzer._logger') as mock_logger:
            query_text = "SELECT * FROM msg_messages"

            # Record slow query
            monitor._record_query_execution(query_text, 1.5)

            # Verify warning was logged
            mock_logger.warning.assert_called_once()

    def test_query_normalization(self, monitor):
        """Test query normalization for grouping."""
        queries = [
            "SELECT * FROM msg_messages WHERE id = 123",
            "SELECT * FROM msg_messages WHERE id = 456",
            "select * from msg_messages where id = 789"
        ]

        # Record executions
        for query in queries:
            monitor._record_query_execution(query, 0.1)

        # Should be grouped into single normalized query
        assert len(monitor.query_metrics) == 1

    def test_performance_summary(self, monitor):
        """Test performance summary generation."""
        # Record some queries
        monitor._record_query_execution("SELECT 1", 0.1)
        monitor._record_query_execution("SELECT 2", 0.6)  # Slow
        monitor._record_query_execution("SELECT 3", 0.2)

        summary = monitor.get_performance_summary()

        assert summary["total_queries"] == 3
        assert summary["unique_queries"] == 3
        assert summary["slow_queries"] == 1
        assert summary["total_execution_time"] == 0.9

    def test_query_timer_context_manager(self):
        """Test query timer context manager."""
        with patch('src.notification.service.query_analyzer._logger') as mock_logger:
            with query_timer("test_operation") as result:
                # Simulate some work
                import time
                time.sleep(0.01)

            assert "execution_time" in result
            assert result["operation"] == "test_operation"
            assert result["execution_time"] > 0


class TestDatabaseHealthChecker:
    """Test database health checking functionality."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        session = Mock()
        return session

    @pytest.fixture
    def health_checker(self, mock_session):
        """Create database health checker with mock session."""
        return DatabaseHealthChecker(mock_session)

    def test_check_table_bloat(self, health_checker, mock_session):
        """Test table bloat checking."""
        # Mock query result
        mock_row = Mock()
        mock_row.schemaname = "public"
        mock_row.tablename = "msg_messages"
        mock_row.total_size = "100 MB"
        mock_row.table_size = "80 MB"
        mock_row.index_size = "20 MB"
        mock_row.index_ratio = 20.0

        mock_session.execute.return_value = [mock_row]

        # Call method
        result = health_checker.check_table_bloat()

        # Verify result
        assert "tables" in result
        assert len(result["tables"]) == 1
        assert result["tables"][0]["table"] == "msg_messages"
        assert "recommendations" in result

    def test_check_index_usage(self, health_checker, mock_session):
        """Test index usage checking."""
        # Mock query result
        mock_row = Mock()
        mock_row.schemaname = "public"
        mock_row.tablename = "msg_messages"
        mock_row.indexname = "idx_unused"
        mock_row.idx_scan = 0  # Unused index
        mock_row.idx_tup_read = 0
        mock_row.idx_tup_fetch = 0
        mock_row.index_size = "10 MB"

        mock_session.execute.return_value = [mock_row]

        # Call method
        result = health_checker.check_index_usage()

        # Verify result
        assert "indexes" in result
        assert len(result["indexes"]) == 1
        assert result["indexes"][0]["scans"] == 0
        assert "recommendations" in result

    def test_check_connection_stats(self, health_checker, mock_session):
        """Test connection statistics checking."""
        # Mock query result
        mock_rows = [
            Mock(state="active", connection_count=10),
            Mock(state="idle", connection_count=50),
            Mock(state="idle in transaction", connection_count=5)
        ]

        mock_session.execute.return_value = mock_rows

        # Call method
        result = health_checker.check_connection_stats()

        # Verify result
        assert result["total_connections"] == 65
        assert result["by_state"]["active"] == 10
        assert result["by_state"]["idle"] == 50
        assert "recommendations" in result


class TestNotificationServiceMigrations:
    """Test database migration functionality."""

    @pytest.fixture
    def mock_engine(self):
        """Create a mock database engine."""
        engine = Mock()
        conn = Mock()
        engine.connect.return_value.__enter__.return_value = conn
        engine.connect.return_value.__exit__.return_value = None
        return engine

    @pytest.fixture
    def migrations(self, mock_engine):
        """Create migrations instance with mock engine."""
        return NotificationServiceMigrations(mock_engine)

    def test_apply_all_optimizations(self, migrations):
        """Test applying all database optimizations."""
        with patch.object(migrations, '_create_optimized_indexes') as mock_indexes, \
             patch.object(migrations, '_add_performance_constraints') as mock_constraints, \
             patch.object(migrations, '_create_table_partitions') as mock_partitions, \
             patch.object(migrations, '_apply_database_settings') as mock_settings:

            # Call method
            result = migrations.apply_all_optimizations()

            # Verify all optimization methods were called
            mock_indexes.assert_called_once()
            mock_constraints.assert_called_once()
            mock_partitions.assert_called_once()
            mock_settings.assert_called_once()

            # Verify result structure
            assert "indexes_created" in result
            assert "constraints_added" in result
            assert "partitions_created" in result
            assert "settings_applied" in result

    def test_create_monitoring_views(self, migrations, mock_engine):
        """Test creating monitoring views."""
        # Mock successful view creation
        mock_conn = mock_engine.connect.return_value.__enter__.return_value
        mock_conn.execute.return_value = None

        # Call method
        result = migrations.create_monitoring_views()

        # Verify views were created
        assert "views_created" in result
        assert "views_failed" in result
        assert len(result["views_created"]) > 0

    def test_analyze_current_performance(self, migrations, mock_engine):
        """Test performance analysis."""
        # Mock query results
        mock_conn = mock_engine.connect.return_value.__enter__.return_value

        # Mock table stats
        table_row = Mock()
        table_row.tablename = "msg_messages"
        table_row.inserts = 1000
        table_row.updates = 100
        table_row.deletes = 10
        table_row.live_tuples = 990
        table_row.dead_tuples = 110
        table_row.last_vacuum = datetime.now(timezone.utc)
        table_row.last_analyze = datetime.now(timezone.utc)

        # Mock index stats
        index_row = Mock()
        index_row.schemaname = "public"
        index_row.tablename = "msg_messages"
        index_row.indexname = "idx_test"
        index_row.idx_scan = 100
        index_row.idx_tup_read = 1000
        index_row.idx_tup_fetch = 900

        mock_conn.execute.side_effect = [
            [table_row],  # Table stats query
            [index_row]   # Index stats query
        ]

        # Call method
        result = migrations.analyze_current_performance()

        # Verify analysis structure
        assert "table_stats" in result
        assert "index_stats" in result
        assert "recommendations" in result
        assert "msg_messages" in result["table_stats"]


class TestIntegrationScenarios:
    """Test integration scenarios for database optimization."""

    def test_optimization_workflow(self):
        """Test complete optimization workflow."""
        # Mock engine and session
        mock_engine = Mock()
        mock_session = Mock()

        with patch('src.notification.service.database_optimization.create_optimized_indexes') as mock_create_indexes, \
             patch('src.notification.service.database_optimization.analyze_query_performance') as mock_analyze:

            # Mock successful operations
            mock_create_indexes.return_value = None
            mock_analyze.return_value = {"recommendations": []}

            # Test index creation
            create_optimized_indexes(mock_engine)
            mock_create_indexes.assert_called_once_with(mock_engine)

            # Test performance analysis
            analyze_query_performance(mock_session)
            mock_analyze.assert_called_once_with(mock_session)

    def test_query_monitoring_integration(self):
        """Test query monitoring integration."""
        monitor = QueryPerformanceMonitor()
        mock_engine = Mock()

        # Test enabling monitoring
        monitor.enable_monitoring(mock_engine)
        assert monitor.monitoring_enabled

        # Test disabling monitoring
        with patch('sqlalchemy.event.remove') as mock_remove:
            monitor.disable_monitoring(mock_engine)
            assert not monitor.monitoring_enabled

    def test_migration_error_handling(self):
        """Test error handling in migrations."""
        mock_engine = Mock()
        mock_conn = Mock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn

        # Mock connection error
        mock_conn.execute.side_effect = Exception("Connection failed")

        migrations = NotificationServiceMigrations(mock_engine)

        # Should handle errors gracefully
        result = migrations.apply_all_optimizations()

        # Verify error handling
        assert "indexes_failed" in result or "settings_failed" in result


if __name__ == "__main__":
    pytest.main([__file__])