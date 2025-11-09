"""
Performance Dashboard for Notification Service

This script provides a real-time performance dashboard for monitoring
database query performance and system health.
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.data.db.core.database import get_database_url
from src.notification.docs.utilities.query_analyzer import (
    DatabaseHealthChecker,
    get_query_monitor
)
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class PerformanceDashboard:
    """Real-time performance dashboard for notification service."""

    def __init__(self):
        """Initialize the performance dashboard."""
        self.engine = None
        self.session_factory = None
        self.monitor = get_query_monitor()
        self.health_checker = None

    def initialize(self):
        """Initialize database connections."""
        try:
            database_url = get_database_url()
            self.engine = create_engine(database_url, echo=False)
            self.session_factory = sessionmaker(bind=self.engine)

            # Enable query monitoring
            self.monitor.enable_monitoring(self.engine)

            # Initialize health checker
            session = self.session_factory()
            self.health_checker = DatabaseHealthChecker(session)

            _logger.info("Performance dashboard initialized")
            return True

        except Exception:
            _logger.exception("Failed to initialize dashboard:")
            return False

    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time performance metrics."""
        metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query_performance": {},
            "database_health": {},
            "system_stats": {}
        }

        try:
            # Query performance metrics
            metrics["query_performance"] = self.monitor.get_performance_summary()
            metrics["query_performance"]["slow_queries"] = self.monitor.get_slow_queries(5)
            metrics["query_performance"]["frequent_queries"] = self.monitor.get_most_frequent_queries(5)

            # Database health metrics
            if self.health_checker:
                metrics["database_health"]["table_bloat"] = self.health_checker.check_table_bloat()
                metrics["database_health"]["index_usage"] = self.health_checker.check_index_usage()
                metrics["database_health"]["connections"] = self.health_checker.check_connection_stats()

            # System statistics
            with self.session_factory() as session:
                metrics["system_stats"] = self._get_system_statistics(session)

        except Exception as e:
            _logger.exception("Failed to get real-time metrics:")
            metrics["error"] = str(e)

        return metrics

    def _get_system_statistics(self, session) -> Dict[str, Any]:
        """Get system-level statistics."""
        from sqlalchemy import text

        stats = {}

        try:
            # Message queue depth
            result = session.execute(text("""
                SELECT
                    status,
                    COUNT(*) as count
                FROM msg_messages
                WHERE created_at >= NOW() - INTERVAL '1 hour'
                GROUP BY status
            """))

            queue_stats = {}
            for row in result:
                queue_stats[row.status] = row.count
            stats["message_queue"] = queue_stats

            # Delivery rates
            result = session.execute(text("""
                SELECT
                    channel,
                    status,
                    COUNT(*) as count,
                    AVG(response_time_ms) as avg_response_time
                FROM msg_delivery_status
                WHERE created_at >= NOW() - INTERVAL '1 hour'
                GROUP BY channel, status
                ORDER BY channel, status
            """))

            delivery_stats = {}
            for row in result:
                if row.channel not in delivery_stats:
                    delivery_stats[row.channel] = {}
                delivery_stats[row.channel][row.status] = {
                    "count": row.count,
                    "avg_response_time": float(row.avg_response_time) if row.avg_response_time else None
                }
            stats["delivery_rates"] = delivery_stats

            # Rate limiting stats
            result = session.execute(text("""
                SELECT
                    channel,
                    COUNT(*) as users,
                    AVG(tokens) as avg_tokens,
                    SUM(CASE WHEN tokens = 0 THEN 1 ELSE 0 END) as rate_limited_users
                FROM msg_rate_limits
                GROUP BY channel
            """))

            rate_limit_stats = {}
            for row in result:
                rate_limit_stats[row.channel] = {
                    "total_users": row.users,
                    "avg_tokens": float(row.avg_tokens) if row.avg_tokens else 0,
                    "rate_limited_users": row.rate_limited_users
                }
            stats["rate_limiting"] = rate_limit_stats

        except Exception as e:
            _logger.exception("Failed to get system statistics:")
            stats["error"] = str(e)

        return stats

    def display_dashboard(self, refresh_interval: int = 30):
        """Display real-time dashboard in terminal."""
        print("🚀 Notification Service Performance Dashboard")
        print("=" * 80)
        print(f"Refresh interval: {refresh_interval} seconds")
        print("Press Ctrl+C to exit\n")

        try:
            while True:
                # Clear screen (works on most terminals)
                print("\033[2J\033[H", end="")

                # Get current metrics
                metrics = self.get_real_time_metrics()

                # Display timestamp
                print(f"📊 Dashboard Updated: {metrics['timestamp']}")
                print("=" * 80)

                # Query Performance Section
                self._display_query_performance(metrics.get("query_performance", {}))

                # Database Health Section
                self._display_database_health(metrics.get("database_health", {}))

                # System Statistics Section
                self._display_system_stats(metrics.get("system_stats", {}))

                # Wait for next refresh
                print(f"\n⏱️  Next refresh in {refresh_interval} seconds... (Ctrl+C to exit)")
                time.sleep(refresh_interval)

        except KeyboardInterrupt:
            print("\n\n👋 Dashboard stopped by user")
        except Exception as e:
            print(f"\n❌ Dashboard error: {e}")

    def _display_query_performance(self, perf_data: Dict[str, Any]):
        """Display query performance metrics."""
        print("\n🔍 QUERY PERFORMANCE")
        print("-" * 40)

        if not perf_data:
            print("   No performance data available")
            return

        # Summary stats
        print(f"   Total Queries: {perf_data.get('total_queries', 0)}")
        print(f"   Unique Queries: {perf_data.get('unique_queries', 0)}")
        print(f"   Slow Queries: {perf_data.get('slow_queries', 0)}")
        print(f"   Avg Execution Time: {perf_data.get('avg_execution_time', 0):.4f}s")
        print(f"   Total Execution Time: {perf_data.get('total_execution_time', 0):.4f}s")

        # Slow queries
        slow_queries = perf_data.get("slow_queries", [])
        if slow_queries:
            print("\n   🐌 Top Slow Queries:")
            for i, query in enumerate(slow_queries[:3], 1):
                print(f"      {i}. {query['avg_time']:.4f}s - {query['query_text'][:60]}...")

    def _display_database_health(self, health_data: Dict[str, Any]):
        """Display database health metrics."""
        print("\n💊 DATABASE HEALTH")
        print("-" * 40)

        if not health_data:
            print("   No health data available")
            return

        # Table bloat
        bloat_data = health_data.get("table_bloat", {})
        if bloat_data.get("tables"):
            print("   📊 Table Sizes:")
            for table in bloat_data["tables"][:3]:
                print(f"      • {table['table']}: {table['total_size']} (Index: {table.get('index_ratio', 0):.1f}%)")

        # Index usage
        index_data = health_data.get("index_usage", {})
        if index_data.get("indexes"):
            unused_indexes = [idx for idx in index_data["indexes"] if idx["scans"] == 0]
            if unused_indexes:
                print(f"   ⚠️  Unused Indexes: {len(unused_indexes)}")

        # Connections
        conn_data = health_data.get("connections", {})
        if conn_data:
            print(f"   🔗 Total Connections: {conn_data.get('total_connections', 0)}")
            by_state = conn_data.get("by_state", {})
            if by_state:
                print(f"      Active: {by_state.get('active', 0)}, Idle: {by_state.get('idle', 0)}")

    def _display_system_stats(self, stats_data: Dict[str, Any]):
        """Display system statistics."""
        print("\n📈 SYSTEM STATISTICS")
        print("-" * 40)

        if not stats_data:
            print("   No system data available")
            return

        # Message queue
        queue_data = stats_data.get("message_queue", {})
        if queue_data:
            print("   📬 Message Queue (Last Hour):")
            for status, count in queue_data.items():
                print(f"      • {status}: {count}")

        # Delivery rates
        delivery_data = stats_data.get("delivery_rates", {})
        if delivery_data:
            print("   📤 Delivery Rates (Last Hour):")
            for channel, statuses in delivery_data.items():
                total = sum(s["count"] for s in statuses.values())
                delivered = statuses.get("DELIVERED", {}).get("count", 0)
                success_rate = (delivered / total * 100) if total > 0 else 0
                print(f"      • {channel}: {success_rate:.1f}% success ({delivered}/{total})")

        # Rate limiting
        rate_data = stats_data.get("rate_limiting", {})
        if rate_data:
            print("   🚦 Rate Limiting:")
            for channel, limits in rate_data.items():
                limited = limits.get("rate_limited_users", 0)
                total = limits.get("total_users", 0)
                if total > 0:
                    print(f"      • {channel}: {limited}/{total} users rate limited")

    def export_metrics(self, filename: str = None) -> str:
        """Export current metrics to JSON file."""
        if not filename:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"performance_metrics_{timestamp}.json"

        try:
            metrics = self.get_real_time_metrics()

            with open(filename, "w") as f:
                json.dump(metrics, f, indent=2, default=str)

            _logger.info("Metrics exported to %s", filename)
            return filename

        except Exception:
            _logger.exception("Failed to export metrics:")
            raise

    def generate_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate performance report for the specified time period."""
        report = {
            "period_hours": hours,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": {},
            "recommendations": []
        }

        try:
            with self.session_factory() as session:
                # Query performance over time period
                from sqlalchemy import text

                # Message volume trends
                result = session.execute(text(f"""
                    SELECT
                        DATE_TRUNC('hour', created_at) as hour,
                        status,
                        COUNT(*) as count
                    FROM msg_messages
                    WHERE created_at >= NOW() - INTERVAL '{hours} hours'
                    GROUP BY DATE_TRUNC('hour', created_at), status
                    ORDER BY hour DESC
                """))

                message_trends = []
                for row in result:
                    message_trends.append({
                        "hour": row.hour.isoformat(),
                        "status": row.status,
                        "count": row.count
                    })

                report["message_trends"] = message_trends

                # Delivery performance
                result = session.execute(text(f"""
                    SELECT
                        channel,
                        COUNT(*) as total_attempts,
                        SUM(CASE WHEN status = 'DELIVERED' THEN 1 ELSE 0 END) as successful,
                        AVG(CASE WHEN status = 'DELIVERED' THEN response_time_ms END) as avg_response_time,
                        PERCENTILE_CONT(0.95) WITHIN GROUP (
                            ORDER BY CASE WHEN status = 'DELIVERED' THEN response_time_ms END
                        ) as p95_response_time
                    FROM msg_delivery_status
                    WHERE created_at >= NOW() - INTERVAL '{hours} hours'
                    GROUP BY channel
                """))

                delivery_performance = []
                for row in result:
                    success_rate = (row.successful / row.total_attempts) if row.total_attempts > 0 else 0
                    delivery_performance.append({
                        "channel": row.channel,
                        "total_attempts": row.total_attempts,
                        "successful": row.successful,
                        "success_rate": success_rate,
                        "avg_response_time": float(row.avg_response_time) if row.avg_response_time else None,
                        "p95_response_time": float(row.p95_response_time) if row.p95_response_time else None
                    })

                report["delivery_performance"] = delivery_performance

                # Generate recommendations
                report["recommendations"] = self._generate_report_recommendations(
                    message_trends, delivery_performance
                )

        except Exception as e:
            _logger.exception("Failed to generate report:")
            report["error"] = str(e)

        return report

    def _generate_report_recommendations(
        self,
        message_trends: List[Dict],
        delivery_performance: List[Dict]
    ) -> List[str]:
        """Generate recommendations based on report data."""
        recommendations = []

        # Check for channels with low success rates
        for perf in delivery_performance:
            if perf["success_rate"] < 0.95:  # Less than 95% success
                recommendations.append(
                    f"Channel '{perf['channel']}' has low success rate ({perf['success_rate']:.1%}). "
                    "Investigate delivery issues."
                )

        # Check for high response times
        for perf in delivery_performance:
            if perf.get("avg_response_time", 0) > 5000:  # More than 5 seconds
                recommendations.append(
                    f"Channel '{perf['channel']}' has high response time "
                    f"({perf['avg_response_time']:.0f}ms). Consider optimization."
                )

        # Check for high message volume
        total_messages = sum(trend["count"] for trend in message_trends)
        if total_messages > 10000:  # More than 10k messages in period
            recommendations.append(
                f"High message volume ({total_messages} messages). "
                "Consider implementing message batching or rate limiting."
            )

        return recommendations

    def cleanup(self):
        """Cleanup resources."""
        if self.monitor:
            self.monitor.disable_monitoring(self.engine)

        if self.engine:
            self.engine.dispose()

        _logger.info("Performance dashboard cleaned up")


def main():
    """Main dashboard application."""
    import argparse

    parser = argparse.ArgumentParser(description="Notification Service Performance Dashboard")
    parser.add_argument("--refresh", type=int, default=30, help="Refresh interval in seconds")
    parser.add_argument("--export", action="store_true", help="Export metrics to JSON and exit")
    parser.add_argument("--report", type=int, help="Generate report for specified hours and exit")

    args = parser.parse_args()

    # Initialize dashboard
    dashboard = PerformanceDashboard()
    if not dashboard.initialize():
        print("❌ Failed to initialize dashboard")
        return 1

    try:
        if args.export:
            # Export metrics and exit
            filename = dashboard.export_metrics()
            print(f"✅ Metrics exported to: {filename}")
            return 0

        elif args.report:
            # Generate report and exit
            report = dashboard.generate_report(args.report)

            # Save report
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            report_file = f"performance_report_{timestamp}.json"

            with open(report_file, "w") as f:
                json.dump(report, f, indent=2, default=str)

            print(f"✅ Performance report generated: {report_file}")

            # Display summary
            print(f"\n📊 PERFORMANCE REPORT ({args.report} hours)")
            print("=" * 50)

            if report.get("delivery_performance"):
                print("\n📤 Delivery Performance:")
                for perf in report["delivery_performance"]:
                    print(f"   • {perf['channel']}: {perf['success_rate']:.1%} success rate")

            if report.get("recommendations"):
                print(f"\n💡 Recommendations ({len(report['recommendations'])}):")
                for i, rec in enumerate(report["recommendations"], 1):
                    print(f"   {i}. {rec}")

            return 0

        else:
            # Run interactive dashboard
            dashboard.display_dashboard(args.refresh)
            return 0

    finally:
        dashboard.cleanup()


if __name__ == "__main__":
    sys.exit(main())