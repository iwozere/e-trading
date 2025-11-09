"""
Database Migration Scripts for Notification Service Optimizations

This module contains migration scripts to add optimized indexes and constraints
for better query performance in the notification service.
"""

from typing import List, Dict, Any
from sqlalchemy import text, MetaData
from sqlalchemy.engine import Engine
from sqlalchemy.exc import ProgrammingError

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class NotificationServiceMigrations:
    """Database migrations for notification service optimizations."""

    def __init__(self, engine: Engine):
        """
        Initialize migrations with database engine.

        Args:
            engine: SQLAlchemy database engine
        """
        self.engine = engine
        self.metadata = MetaData()

    def apply_all_optimizations(self) -> Dict[str, Any]:
        """
        Apply all database optimizations.

        Returns:
            Dictionary with migration results
        """
        _logger.info("Starting notification service database optimizations...")

        results = {
            "indexes_created": [],
            "indexes_failed": [],
            "constraints_added": [],
            "constraints_failed": [],
            "partitions_created": [],
            "partitions_failed": [],
            "settings_applied": [],
            "settings_failed": []
        }

        # Apply optimizations in order
        self._create_optimized_indexes(results)
        self._add_performance_constraints(results)
        self._create_table_partitions(results)
        self._apply_database_settings(results)

        _logger.info("Completed notification service database optimizations")
        return results

    def _create_optimized_indexes(self, results: Dict[str, Any]):
        """Create optimized indexes for better query performance."""
        _logger.info("Creating optimized indexes...")

        # Define indexes to create
        indexes = [
            # Messages table - composite indexes for common query patterns
            {
                "name": "idx_msg_messages_status_scheduled_priority_optimized",
                "table": "msg_messages",
                "columns": ["status", "scheduled_for", "priority", "id"],
                "where": "status = 'PENDING'",
                "description": "Optimized index for pending message queries with priority ordering"
            },
            {
                "name": "idx_msg_messages_recipient_created_desc",
                "table": "msg_messages",
                "columns": ["recipient_id", "created_at DESC", "id DESC"],
                "description": "Index for user message history queries"
            },
            {
                "name": "idx_msg_messages_type_status_created",
                "table": "msg_messages",
                "columns": ["message_type", "status", "created_at"],
                "description": "Index for message type and status filtering"
            },
            {
                "name": "idx_msg_messages_retry_failed",
                "table": "msg_messages",
                "columns": ["status", "retry_count", "max_retries", "processed_at"],
                "where": "status = 'FAILED' AND retry_count < max_retries",
                "description": "Index for failed messages eligible for retry"
            },
            {
                "name": "idx_msg_messages_cleanup",
                "table": "msg_messages",
                "columns": ["status", "created_at"],
                "where": "status IN ('DELIVERED', 'CANCELLED')",
                "description": "Index for cleanup operations on old messages"
            },

            # Delivery status table - optimized for analytics and monitoring
            {
                "name": "idx_msg_delivery_status_channel_created_desc",
                "table": "msg_delivery_status",
                "columns": ["channel", "created_at DESC", "id DESC"],
                "description": "Index for channel-specific delivery history"
            },
            {
                "name": "idx_msg_delivery_status_analytics",
                "table": "msg_delivery_status",
                "columns": ["status", "channel", "created_at", "response_time_ms"],
                "description": "Composite index for analytics queries"
            },
            {
                "name": "idx_msg_delivery_status_message_lookup",
                "table": "msg_delivery_status",
                "columns": ["message_id", "channel", "status"],
                "description": "Index for message delivery status lookups"
            },
            {
                "name": "idx_msg_delivery_status_time_series",
                "table": "msg_delivery_status",
                "columns": ["created_at", "status", "channel"],
                "description": "Index for time series analytics"
            },
            {
                "name": "idx_msg_delivery_status_performance",
                "table": "msg_delivery_status",
                "columns": ["channel", "status", "response_time_ms"],
                "where": "status = 'DELIVERED' AND response_time_ms IS NOT NULL",
                "description": "Index for performance metrics calculation"
            },

            # Rate limits table - optimized for token bucket operations
            {
                "name": "idx_msg_rate_limits_refill_optimized",
                "table": "msg_rate_limits",
                "columns": ["last_refill", "tokens", "max_tokens"],
                "where": "tokens < max_tokens",
                "description": "Index for efficient token refill operations"
            },
            {
                "name": "idx_msg_rate_limits_user_channel_lookup",
                "table": "msg_rate_limits",
                "columns": ["user_id", "channel", "tokens", "last_refill"],
                "description": "Index for rate limit checks"
            },

            # Channel health table - optimized for monitoring
            {
                "name": "idx_msg_channel_health_monitoring",
                "table": "msg_channel_health",
                "columns": ["status", "checked_at DESC", "channel"],
                "description": "Index for health monitoring queries"
            },
            {
                "name": "idx_msg_channel_health_failures",
                "table": "msg_channel_health",
                "columns": ["failure_count", "last_failure", "status"],
                "where": "failure_count > 0",
                "description": "Index for tracking channel failures"
            },

            # Channel configs table - optimized for configuration lookups
            {
                "name": "idx_msg_channel_configs_enabled_lookup",
                "table": "msg_channel_configs",
                "columns": ["enabled", "channel", "updated_at"],
                "description": "Index for enabled channel lookups"
            }
        ]

        # Create each index
        with self.engine.connect() as conn:
            for index_def in indexes:
                try:
                    self._create_index(conn, index_def)
                    results["indexes_created"].append(index_def["name"])
                    _logger.info("Created index: %s", index_def["name"])
                except Exception as e:
                    results["indexes_failed"].append({
                        "name": index_def["name"],
                        "error": str(e)
                    })
                    _logger.error("Failed to create index %s: %s", index_def["name"], e)

    def _create_index(self, conn, index_def: Dict[str, Any]):
        """Create a single index."""
        columns_sql = ", ".join(index_def["columns"])
        where_clause = f" WHERE {index_def['where']}" if index_def.get("where") else ""

        sql = f"""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS {index_def["name"]}
        ON {index_def["table"]} ({columns_sql}){where_clause}
        """

        conn.execute(text(sql))

    def _add_performance_constraints(self, results: Dict[str, Any]):
        """Add performance-related constraints and triggers."""
        _logger.info("Adding performance constraints...")

        constraints = [
            # Add check constraint for reasonable retry counts
            {
                "name": "check_reasonable_retry_count",
                "table": "msg_messages",
                "constraint": "CHECK (retry_count <= 10)",
                "description": "Prevent excessive retry attempts"
            },

            # Add check constraint for reasonable response times
            {
                "name": "check_reasonable_response_time",
                "table": "msg_delivery_status",
                "constraint": "CHECK (response_time_ms IS NULL OR response_time_ms <= 300000)",
                "description": "Prevent unreasonable response time values (max 5 minutes)"
            },

            # Add check constraint for future scheduled times
            {
                "name": "check_reasonable_schedule_time",
                "table": "msg_messages",
                "constraint": "CHECK (scheduled_for <= created_at + INTERVAL '30 days')",
                "description": "Prevent scheduling messages too far in the future"
            }
        ]

        with self.engine.connect() as conn:
            for constraint_def in constraints:
                try:
                    sql = f"""
                    ALTER TABLE {constraint_def["table"]}
                    ADD CONSTRAINT {constraint_def["name"]} {constraint_def["constraint"]}
                    """
                    conn.execute(text(sql))
                    results["constraints_added"].append(constraint_def["name"])
                    _logger.info("Added constraint: %s", constraint_def["name"])
                except ProgrammingError as e:
                    if "already exists" in str(e).lower():
                        _logger.info("Constraint %s already exists", constraint_def["name"])
                    else:
                        results["constraints_failed"].append({
                            "name": constraint_def["name"],
                            "error": str(e)
                        })
                        _logger.error("Failed to add constraint %s: %s", constraint_def["name"], e)
                except Exception as e:
                    results["constraints_failed"].append({
                        "name": constraint_def["name"],
                        "error": str(e)
                    })
                    _logger.error("Failed to add constraint %s: %s", constraint_def["name"], e)

    def _create_table_partitions(self, results: Dict[str, Any]):
        """Create table partitions for large tables."""
        _logger.info("Creating table partitions...")

        # Check if tables are large enough to warrant partitioning
        with self.engine.connect() as conn:
            try:
                # Check message table size
                size_query = text("""
                    SELECT pg_total_relation_size('msg_messages') as size_bytes,
                           COUNT(*) as row_count
                    FROM msg_messages
                """)
                result = conn.execute(size_query).fetchone()

                if result and result.size_bytes > 1024 * 1024 * 1024:  # > 1GB
                    _logger.info("Messages table is large (%s bytes), considering partitioning", result.size_bytes)
                    # For now, just log the recommendation
                    results["partitions_created"].append({
                        "table": "msg_messages",
                        "recommendation": "Consider partitioning by created_at (monthly partitions)"
                    })

                # Check delivery status table size
                size_query = text("""
                    SELECT pg_total_relation_size('msg_delivery_status') as size_bytes,
                           COUNT(*) as row_count
                    FROM msg_delivery_status
                """)
                result = conn.execute(size_query).fetchone()

                if result and result.size_bytes > 1024 * 1024 * 1024:  # > 1GB
                    _logger.info("Delivery status table is large (%s bytes), considering partitioning", result.size_bytes)
                    results["partitions_created"].append({
                        "table": "msg_delivery_status",
                        "recommendation": "Consider partitioning by created_at (monthly partitions)"
                    })

            except Exception as e:
                results["partitions_failed"].append({
                    "error": str(e),
                    "description": "Failed to check table sizes for partitioning"
                })
                _logger.exception("Failed to check table sizes:")

    def _apply_database_settings(self, results: Dict[str, Any]):
        """Apply database-level performance settings."""
        _logger.info("Applying database performance settings...")

        settings = [
            # Optimize for notification service workload
            ("work_mem", "256MB", "Increase memory for complex queries"),
            ("maintenance_work_mem", "512MB", "Increase memory for maintenance operations"),
            ("effective_cache_size", "4GB", "Optimize for available system cache"),
            ("random_page_cost", "1.1", "Optimize for SSD storage"),
            ("seq_page_cost", "1.0", "Optimize for SSD storage"),
            ("checkpoint_completion_target", "0.9", "Spread checkpoint I/O"),
            ("wal_buffers", "16MB", "Increase WAL buffer size"),
            ("max_wal_size", "4GB", "Allow larger WAL files"),
            ("min_wal_size", "1GB", "Maintain minimum WAL size"),
            ("log_min_duration_statement", "1000", "Log slow queries (1 second+)"),
            ("log_checkpoints", "on", "Log checkpoint activity"),
            ("log_connections", "off", "Reduce log noise"),
            ("log_disconnections", "off", "Reduce log noise"),
            ("log_lock_waits", "on", "Log lock waits for debugging"),
            ("deadlock_timeout", "1s", "Quick deadlock detection"),
            ("max_connections", "200", "Reasonable connection limit"),
        ]

        with self.engine.connect() as conn:
            for setting_name, setting_value, description in settings:
                try:
                    # Check current value first
                    current_query = text(f"SHOW {setting_name}")
                    current_result = conn.execute(current_query).fetchone()
                    current_value = current_result[0] if current_result else "unknown"

                    # Apply setting (session-level for testing)
                    set_query = text(f"SET {setting_name} = :value")
                    conn.execute(set_query, {"value": setting_value})

                    results["settings_applied"].append({
                        "setting": setting_name,
                        "old_value": current_value,
                        "new_value": setting_value,
                        "description": description
                    })
                    _logger.info("Applied setting %s: %s -> %s", setting_name, current_value, setting_value)

                except Exception as e:
                    results["settings_failed"].append({
                        "setting": setting_name,
                        "value": setting_value,
                        "error": str(e)
                    })
                    _logger.warning("Failed to apply setting %s: %s", setting_name, e)

    def create_monitoring_views(self) -> Dict[str, Any]:
        """Create database views for monitoring and analytics."""
        _logger.info("Creating monitoring views...")

        results = {
            "views_created": [],
            "views_failed": []
        }

        views = [
            {
                "name": "v_msg_delivery_summary",
                "sql": """
                CREATE OR REPLACE VIEW v_msg_delivery_summary AS
                SELECT
                    channel,
                    status,
                    DATE_TRUNC('hour', created_at) as hour,
                    COUNT(*) as delivery_count,
                    AVG(response_time_ms) as avg_response_time,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY response_time_ms) as median_response_time,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) as p95_response_time
                FROM msg_delivery_status
                WHERE created_at >= NOW() - INTERVAL '7 days'
                GROUP BY channel, status, DATE_TRUNC('hour', created_at)
                ORDER BY hour DESC, channel, status
                """,
                "description": "Hourly delivery summary by channel and status"
            },
            {
                "name": "v_msg_channel_health_summary",
                "sql": """
                CREATE OR REPLACE VIEW v_msg_channel_health_summary AS
                SELECT
                    ch.channel,
                    ch.status as health_status,
                    ch.failure_count,
                    ch.avg_response_time_ms,
                    ch.last_success,
                    ch.last_failure,
                    ch.checked_at,
                    COALESCE(ds.recent_deliveries, 0) as recent_deliveries,
                    COALESCE(ds.recent_failures, 0) as recent_failures,
                    CASE
                        WHEN COALESCE(ds.recent_deliveries, 0) > 0
                        THEN ROUND(100.0 * COALESCE(ds.recent_failures, 0) / ds.recent_deliveries, 2)
                        ELSE 0
                    END as recent_failure_rate
                FROM msg_channel_health ch
                LEFT JOIN (
                    SELECT
                        channel,
                        COUNT(*) as recent_deliveries,
                        SUM(CASE WHEN status IN ('FAILED', 'BOUNCED') THEN 1 ELSE 0 END) as recent_failures
                    FROM msg_delivery_status
                    WHERE created_at >= NOW() - INTERVAL '1 hour'
                    GROUP BY channel
                ) ds ON ch.channel = ds.channel
                ORDER BY ch.channel
                """,
                "description": "Channel health summary with recent delivery statistics"
            },
            {
                "name": "v_msg_user_activity",
                "sql": """
                CREATE OR REPLACE VIEW v_msg_user_activity AS
                SELECT
                    m.recipient_id,
                    COUNT(*) as total_messages,
                    COUNT(CASE WHEN m.status = 'DELIVERED' THEN 1 END) as delivered_messages,
                    COUNT(CASE WHEN m.status = 'FAILED' THEN 1 END) as failed_messages,
                    MAX(m.created_at) as last_message_at,
                    AVG(ds.response_time_ms) as avg_response_time
                FROM msg_messages m
                LEFT JOIN msg_delivery_status ds ON m.id = ds.message_id AND ds.status = 'DELIVERED'
                WHERE m.created_at >= NOW() - INTERVAL '30 days'
                  AND m.recipient_id IS NOT NULL
                GROUP BY m.recipient_id
                HAVING COUNT(*) > 0
                ORDER BY total_messages DESC
                """,
                "description": "User activity summary for the last 30 days"
            }
        ]

        with self.engine.connect() as conn:
            for view_def in views:
                try:
                    conn.execute(text(view_def["sql"]))
                    results["views_created"].append({
                        "name": view_def["name"],
                        "description": view_def["description"]
                    })
                    _logger.info("Created view: %s", view_def["name"])
                except Exception as e:
                    results["views_failed"].append({
                        "name": view_def["name"],
                        "error": str(e)
                    })
                    _logger.error("Failed to create view %s: %s", view_def["name"], e)

        return results

    def analyze_current_performance(self) -> Dict[str, Any]:
        """Analyze current database performance."""
        _logger.info("Analyzing current database performance...")

        analysis = {
            "table_stats": {},
            "index_stats": {},
            "query_stats": {},
            "recommendations": []
        }

        with self.engine.connect() as conn:
            try:
                # Table statistics
                table_stats_query = text("""
                    SELECT
                        schemaname,
                        tablename,
                        n_tup_ins as inserts,
                        n_tup_upd as updates,
                        n_tup_del as deletes,
                        n_live_tup as live_tuples,
                        n_dead_tup as dead_tuples,
                        last_vacuum,
                        last_autovacuum,
                        last_analyze,
                        last_autoanalyze
                    FROM pg_stat_user_tables
                    WHERE tablename LIKE 'msg_%'
                    ORDER BY n_live_tup DESC
                """)

                result = conn.execute(table_stats_query)
                for row in result:
                    analysis["table_stats"][row.tablename] = {
                        "inserts": row.inserts,
                        "updates": row.updates,
                        "deletes": row.deletes,
                        "live_tuples": row.live_tuples,
                        "dead_tuples": row.dead_tuples,
                        "last_vacuum": row.last_vacuum.isoformat() if row.last_vacuum else None,
                        "last_analyze": row.last_analyze.isoformat() if row.last_analyze else None
                    }

                # Index statistics
                index_stats_query = text("""
                    SELECT
                        schemaname,
                        tablename,
                        indexname,
                        idx_scan,
                        idx_tup_read,
                        idx_tup_fetch
                    FROM pg_stat_user_indexes
                    WHERE tablename LIKE 'msg_%'
                    ORDER BY idx_scan DESC
                """)

                result = conn.execute(index_stats_query)
                for row in result:
                    table_key = f"{row.schemaname}.{row.tablename}"
                    if table_key not in analysis["index_stats"]:
                        analysis["index_stats"][table_key] = []

                    analysis["index_stats"][table_key].append({
                        "index_name": row.indexname,
                        "scans": row.idx_scan,
                        "tuples_read": row.idx_tup_read,
                        "tuples_fetched": row.idx_tup_fetch
                    })

                # Generate recommendations
                analysis["recommendations"] = self._generate_performance_recommendations(analysis)

            except Exception as e:
                _logger.exception("Failed to analyze performance:")
                analysis["error"] = str(e)

        return analysis

    def _generate_performance_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations based on analysis."""
        recommendations = []

        # Check for tables with high dead tuple ratio
        for table_name, stats in analysis["table_stats"].items():
            if stats["live_tuples"] > 0:
                dead_ratio = stats["dead_tuples"] / (stats["live_tuples"] + stats["dead_tuples"])
                if dead_ratio > 0.2:  # More than 20% dead tuples
                    recommendations.append(
                        f"Table {table_name} has high dead tuple ratio ({dead_ratio:.1%}). "
                        "Consider running VACUUM or adjusting autovacuum settings."
                    )

        # Check for unused indexes
        for table_key, indexes in analysis["index_stats"].items():
            for index_info in indexes:
                if index_info["scans"] == 0 and not index_info["index_name"].endswith("_pkey"):
                    recommendations.append(
                        f"Index {index_info['index_name']} is unused. Consider dropping it."
                    )

        return recommendations


def run_optimization_migration(engine: Engine) -> Dict[str, Any]:
    """
    Run the complete optimization migration.

    Args:
        engine: SQLAlchemy database engine

    Returns:
        Dictionary with migration results
    """
    migrations = NotificationServiceMigrations(engine)

    # Apply all optimizations
    results = migrations.apply_all_optimizations()

    # Create monitoring views
    view_results = migrations.create_monitoring_views()
    results["views"] = view_results

    # Analyze performance
    performance_analysis = migrations.analyze_current_performance()
    results["performance_analysis"] = performance_analysis

    return results