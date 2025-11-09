"""
Query Performance Analyzer for Notification Service

This module provides tools to analyze and monitor database query performance,
identify slow queries, and suggest optimizations.
"""

import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from contextlib import contextmanager
from dataclasses import dataclass, field
import statistics

from sqlalchemy.orm import Session
from sqlalchemy import event, text
from sqlalchemy.engine import Engine

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


@dataclass
class QueryMetrics:
    """Metrics for a database query."""
    query_hash: str
    query_text: str
    execution_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    last_executed: Optional[datetime] = None
    execution_times: List[float] = field(default_factory=list)

    def add_execution(self, execution_time: float):
        """Add a new execution time to the metrics."""
        self.execution_count += 1
        self.total_time += execution_time
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.avg_time = self.total_time / self.execution_count
        self.last_executed = datetime.now(timezone.utc)

        # Keep only last 100 execution times for percentile calculations
        self.execution_times.append(execution_time)
        if len(self.execution_times) > 100:
            self.execution_times.pop(0)

    @property
    def median_time(self) -> float:
        """Calculate median execution time."""
        if not self.execution_times:
            return 0.0
        return statistics.median(self.execution_times)

    @property
    def p95_time(self) -> float:
        """Calculate 95th percentile execution time."""
        if not self.execution_times:
            return 0.0
        return statistics.quantiles(self.execution_times, n=20)[18]  # 95th percentile

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "query_hash": self.query_hash,
            "query_text": self.query_text[:200] + "..." if len(self.query_text) > 200 else self.query_text,
            "execution_count": self.execution_count,
            "total_time": round(self.total_time, 4),
            "min_time": round(self.min_time, 4),
            "max_time": round(self.max_time, 4),
            "avg_time": round(self.avg_time, 4),
            "median_time": round(self.median_time, 4),
            "p95_time": round(self.p95_time, 4),
            "last_executed": self.last_executed.isoformat() if self.last_executed else None
        }


class QueryPerformanceMonitor:
    """Monitor and analyze database query performance."""

    def __init__(self, slow_query_threshold: float = 1.0):
        """
        Initialize the query performance monitor.

        Args:
            slow_query_threshold: Threshold in seconds for considering a query slow
        """
        self.slow_query_threshold = slow_query_threshold
        self.query_metrics: Dict[str, QueryMetrics] = {}
        self.monitoring_enabled = False

    def enable_monitoring(self, engine: Engine):
        """
        Enable query monitoring on the given engine.

        Args:
            engine: SQLAlchemy engine to monitor
        """
        if self.monitoring_enabled:
            return

        @event.listens_for(engine, "before_cursor_execute")
        def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            """Record query start time."""
            context._query_start_time = time.time()

        @event.listens_for(engine, "after_cursor_execute")
        def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            """Record query execution time and update metrics."""
            if hasattr(context, '_query_start_time'):
                execution_time = time.time() - context._query_start_time
                self._record_query_execution(statement, execution_time)

        self.monitoring_enabled = True
        _logger.info("Query performance monitoring enabled")

    def disable_monitoring(self, engine: Engine):
        """
        Disable query monitoring on the given engine.

        Args:
            engine: SQLAlchemy engine to stop monitoring
        """
        if not self.monitoring_enabled:
            return

        # Remove event listeners
        event.remove(engine, "before_cursor_execute", self.receive_before_cursor_execute)
        event.remove(engine, "after_cursor_execute", self.receive_after_cursor_execute)

        self.monitoring_enabled = False
        _logger.info("Query performance monitoring disabled")

    def _record_query_execution(self, statement: str, execution_time: float):
        """
        Record a query execution.

        Args:
            statement: SQL statement
            execution_time: Execution time in seconds
        """
        # Normalize query for grouping (remove parameters, extra whitespace)
        normalized_query = self._normalize_query(statement)
        query_hash = str(hash(normalized_query))

        if query_hash not in self.query_metrics:
            self.query_metrics[query_hash] = QueryMetrics(
                query_hash=query_hash,
                query_text=normalized_query
            )

        self.query_metrics[query_hash].add_execution(execution_time)

        # Log slow queries
        if execution_time > self.slow_query_threshold:
            _logger.warning(
                "Slow query detected (%.4fs): %s",
                execution_time,
                normalized_query[:100] + "..." if len(normalized_query) > 100 else normalized_query
            )

    def _normalize_query(self, statement: str) -> str:
        """
        Normalize a SQL statement for grouping.

        Args:
            statement: Raw SQL statement

        Returns:
            Normalized SQL statement
        """
        # Remove extra whitespace and normalize case
        normalized = ' '.join(statement.split()).upper()

        # Replace parameter placeholders with generic markers
        import re
        normalized = re.sub(r'\$\d+', '$N', normalized)  # PostgreSQL parameters
        normalized = re.sub(r'\?', '?', normalized)      # Generic parameters
        normalized = re.sub(r"'[^']*'", "'VALUE'", normalized)  # String literals
        normalized = re.sub(r'\b\d+\b', 'N', normalized)  # Numeric literals

        return normalized

    def get_slow_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the slowest queries.

        Args:
            limit: Maximum number of queries to return

        Returns:
            List of slow query metrics
        """
        slow_queries = [
            metrics for metrics in self.query_metrics.values()
            if metrics.avg_time > self.slow_query_threshold
        ]

        # Sort by average execution time
        slow_queries.sort(key=lambda x: x.avg_time, reverse=True)

        return [query.to_dict() for query in slow_queries[:limit]]

    def get_most_frequent_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most frequently executed queries.

        Args:
            limit: Maximum number of queries to return

        Returns:
            List of frequent query metrics
        """
        frequent_queries = sorted(
            self.query_metrics.values(),
            key=lambda x: x.execution_count,
            reverse=True
        )

        return [query.to_dict() for query in frequent_queries[:limit]]

    def get_total_time_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get queries with the highest total execution time.

        Args:
            limit: Maximum number of queries to return

        Returns:
            List of high total time query metrics
        """
        total_time_queries = sorted(
            self.query_metrics.values(),
            key=lambda x: x.total_time,
            reverse=True
        )

        return [query.to_dict() for query in total_time_queries[:limit]]

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of query performance.

        Returns:
            Dictionary with performance summary
        """
        if not self.query_metrics:
            return {
                "total_queries": 0,
                "unique_queries": 0,
                "slow_queries": 0,
                "avg_execution_time": 0.0,
                "total_execution_time": 0.0
            }

        total_executions = sum(m.execution_count for m in self.query_metrics.values())
        total_time = sum(m.total_time for m in self.query_metrics.values())
        slow_query_count = sum(
            1 for m in self.query_metrics.values()
            if m.avg_time > self.slow_query_threshold
        )

        return {
            "total_queries": total_executions,
            "unique_queries": len(self.query_metrics),
            "slow_queries": slow_query_count,
            "avg_execution_time": round(total_time / total_executions, 4) if total_executions > 0 else 0.0,
            "total_execution_time": round(total_time, 4),
            "monitoring_enabled": self.monitoring_enabled,
            "slow_query_threshold": self.slow_query_threshold
        }

    def reset_metrics(self):
        """Reset all collected metrics."""
        self.query_metrics.clear()
        _logger.info("Query performance metrics reset")

    def export_metrics(self) -> Dict[str, Any]:
        """
        Export all metrics for analysis.

        Returns:
            Dictionary with all query metrics
        """
        return {
            "summary": self.get_performance_summary(),
            "slow_queries": self.get_slow_queries(50),
            "frequent_queries": self.get_most_frequent_queries(50),
            "total_time_queries": self.get_total_time_queries(50),
            "export_timestamp": datetime.now(timezone.utc).isoformat()
        }


@contextmanager
def query_timer(operation_name: str):
    """
    Context manager to time database operations.

    Args:
        operation_name: Name of the operation being timed

    Yields:
        Dictionary to store timing results
    """
    start_time = time.time()
    result = {"operation": operation_name}

    try:
        yield result
    finally:
        execution_time = time.time() - start_time
        result["execution_time"] = execution_time

        if execution_time > 1.0:  # Log operations taking more than 1 second
            _logger.warning(
                "Slow operation '%s' took %.4fs",
                operation_name,
                execution_time
            )
        else:
            _logger.debug(
                "Operation '%s' completed in %.4fs",
                operation_name,
                execution_time
            )


class DatabaseHealthChecker:
    """Check database health and performance indicators."""

    def __init__(self, session: Session):
        """
        Initialize the database health checker.

        Args:
            session: SQLAlchemy database session
        """
        self.session = session

    def check_table_bloat(self) -> Dict[str, Any]:
        """
        Check for table bloat in notification tables.

        Returns:
            Dictionary with bloat information
        """
        bloat_query = text("""
            SELECT
                schemaname,
                tablename,
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as total_size,
                pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) as table_size,
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) - pg_relation_size(schemaname||'.'||tablename)) as index_size,
                ROUND(100 * (pg_total_relation_size(schemaname||'.'||tablename) - pg_relation_size(schemaname||'.'||tablename))::NUMERIC / pg_total_relation_size(schemaname||'.'||tablename), 2) as index_ratio
            FROM pg_tables
            WHERE tablename LIKE 'msg_%'
            ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
        """)

        try:
            result = self.session.execute(bloat_query)
            tables = []
            for row in result:
                tables.append({
                    "schema": row.schemaname,
                    "table": row.tablename,
                    "total_size": row.total_size,
                    "table_size": row.table_size,
                    "index_size": row.index_size,
                    "index_ratio": float(row.index_ratio) if row.index_ratio else 0.0
                })

            return {
                "tables": tables,
                "recommendations": self._generate_bloat_recommendations(tables)
            }
        except Exception as e:
            _logger.exception("Failed to check table bloat:")
            return {"error": str(e)}

    def check_index_usage(self) -> Dict[str, Any]:
        """
        Check index usage statistics.

        Returns:
            Dictionary with index usage information
        """
        index_usage_query = text("""
            SELECT
                schemaname,
                tablename,
                indexname,
                idx_scan,
                idx_tup_read,
                idx_tup_fetch,
                pg_size_pretty(pg_relation_size(indexrelid)) as index_size
            FROM pg_stat_user_indexes
            WHERE tablename LIKE 'msg_%'
            ORDER BY idx_scan DESC
        """)

        try:
            result = self.session.execute(index_usage_query)
            indexes = []
            for row in result:
                indexes.append({
                    "schema": row.schemaname,
                    "table": row.tablename,
                    "index": row.indexname,
                    "scans": row.idx_scan,
                    "tuples_read": row.idx_tup_read,
                    "tuples_fetched": row.idx_tup_fetch,
                    "size": row.index_size
                })

            return {
                "indexes": indexes,
                "recommendations": self._generate_index_recommendations(indexes)
            }
        except Exception as e:
            _logger.exception("Failed to check index usage:")
            return {"error": str(e)}

    def check_connection_stats(self) -> Dict[str, Any]:
        """
        Check database connection statistics.

        Returns:
            Dictionary with connection information
        """
        connection_query = text("""
            SELECT
                state,
                COUNT(*) as connection_count
            FROM pg_stat_activity
            WHERE datname = current_database()
            GROUP BY state
            ORDER BY connection_count DESC
        """)

        try:
            result = self.session.execute(connection_query)
            connections = {}
            total_connections = 0

            for row in result:
                connections[row.state or 'unknown'] = row.connection_count
                total_connections += row.connection_count

            return {
                "total_connections": total_connections,
                "by_state": connections,
                "recommendations": self._generate_connection_recommendations(connections, total_connections)
            }
        except Exception as e:
            _logger.exception("Failed to check connection stats:")
            return {"error": str(e)}

    def _generate_bloat_recommendations(self, tables: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on table bloat analysis."""
        recommendations = []

        for table in tables:
            if table["index_ratio"] > 50:
                recommendations.append(
                    f"Table {table['table']} has high index overhead ({table['index_ratio']}%). "
                    "Consider reviewing index usage and removing unused indexes."
                )

            # Check for very large tables that might need partitioning
            if "GB" in table["total_size"] and float(table["total_size"].split()[0]) > 10:
                recommendations.append(
                    f"Table {table['table']} is very large ({table['total_size']}). "
                    "Consider implementing table partitioning or archiving old data."
                )

        return recommendations

    def _generate_index_recommendations(self, indexes: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on index usage analysis."""
        recommendations = []

        for index in indexes:
            if index["scans"] == 0 and not index["index"].endswith("_pkey"):
                recommendations.append(
                    f"Index {index['index']} on table {index['table']} is never used. "
                    f"Consider dropping it to save {index['size']} of storage."
                )
            elif index["scans"] < 10 and not index["index"].endswith("_pkey"):
                recommendations.append(
                    f"Index {index['index']} on table {index['table']} is rarely used "
                    f"({index['scans']} scans). Review if it's still needed."
                )

        return recommendations

    def _generate_connection_recommendations(self, connections: Dict[str, int], total: int) -> List[str]:
        """Generate recommendations based on connection analysis."""
        recommendations = []

        if total > 100:
            recommendations.append(
                f"High number of database connections ({total}). "
                "Consider implementing connection pooling."
            )

        idle_connections = connections.get('idle', 0)
        if idle_connections > total * 0.5:
            recommendations.append(
                f"Many idle connections ({idle_connections}). "
                "Consider reducing connection pool size or implementing connection recycling."
            )

        return recommendations


# Global query monitor instance
query_monitor = QueryPerformanceMonitor()


def get_query_monitor() -> QueryPerformanceMonitor:
    """Get the global query monitor instance."""
    return query_monitor