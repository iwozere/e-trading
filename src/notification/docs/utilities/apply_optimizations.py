"""
Apply Database Optimizations Script

This script applies all database optimizations for the notification service,
including indexes, constraints, and performance settings.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from sqlalchemy import create_engine
from src.data.db.core.database import get_database_url
from src.notification.docs.utilities.database_migrations import run_optimization_migration
from src.notification.docs.utilities.query_analyzer import QueryPerformanceMonitor, get_query_monitor
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def main():
    """Apply all database optimizations."""
    _logger.info("Starting notification service database optimization...")

    try:
        # Get database connection
        database_url = get_database_url()
        engine = create_engine(database_url, echo=False)

        _logger.info("Connected to database: %s", database_url.split('@')[1] if '@' in database_url else "local")

        # Run optimization migration
        _logger.info("Applying database optimizations...")
        results = run_optimization_migration(engine)

        # Print results summary
        print("\n" + "="*60)
        print("DATABASE OPTIMIZATION RESULTS")
        print("="*60)

        # Indexes
        if results.get("indexes_created"):
            print(f"\n✅ INDEXES CREATED ({len(results['indexes_created'])})")
            for index_name in results["indexes_created"]:
                print(f"   • {index_name}")

        if results.get("indexes_failed"):
            print(f"\n❌ INDEXES FAILED ({len(results['indexes_failed'])})")
            for failed in results["indexes_failed"]:
                print(f"   • {failed['name']}: {failed['error']}")

        # Constraints
        if results.get("constraints_added"):
            print(f"\n✅ CONSTRAINTS ADDED ({len(results['constraints_added'])})")
            for constraint_name in results["constraints_added"]:
                print(f"   • {constraint_name}")

        if results.get("constraints_failed"):
            print(f"\n❌ CONSTRAINTS FAILED ({len(results['constraints_failed'])})")
            for failed in results["constraints_failed"]:
                print(f"   • {failed['name']}: {failed['error']}")

        # Database settings
        if results.get("settings_applied"):
            print(f"\n✅ SETTINGS APPLIED ({len(results['settings_applied'])})")
            for setting in results["settings_applied"]:
                print(f"   • {setting['setting']}: {setting['old_value']} → {setting['new_value']}")

        if results.get("settings_failed"):
            print(f"\n❌ SETTINGS FAILED ({len(results['settings_failed'])})")
            for failed in results["settings_failed"]:
                print(f"   • {failed['setting']}: {failed['error']}")

        # Views
        if results.get("views", {}).get("views_created"):
            print(f"\n✅ MONITORING VIEWS CREATED ({len(results['views']['views_created'])})")
            for view in results["views"]["views_created"]:
                print(f"   • {view['name']}: {view['description']}")

        # Performance analysis
        if results.get("performance_analysis"):
            analysis = results["performance_analysis"]
            if analysis.get("recommendations"):
                print(f"\n💡 PERFORMANCE RECOMMENDATIONS ({len(analysis['recommendations'])})")
                for i, rec in enumerate(analysis["recommendations"], 1):
                    print(f"   {i}. {rec}")

        # Partitioning recommendations
        if results.get("partitions_created"):
            print(f"\n💡 PARTITIONING RECOMMENDATIONS ({len(results['partitions_created'])})")
            for partition in results["partitions_created"]:
                if isinstance(partition, dict) and "recommendation" in partition:
                    print(f"   • {partition['table']}: {partition['recommendation']}")

        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETE")
        print("="*60)

        # Save detailed results to file
        results_file = Path("optimization_results.json")
        with open(results_file, "w") as f:
            # Convert datetime objects to strings for JSON serialization
            json_results = json.loads(json.dumps(results, default=str))
            json.dump(json_results, f, indent=2)

        print(f"\nDetailed results saved to: {results_file.absolute()}")

        # Enable query monitoring for future analysis
        monitor = get_query_monitor()
        monitor.enable_monitoring(engine)
        _logger.info("Query performance monitoring enabled")

        print("\n🔍 Query performance monitoring is now enabled.")
        print("   Use the monitoring views or query analyzer to track performance.")

        return True

    except Exception as e:
        _logger.exception("Failed to apply optimizations:")
        print(f"\n❌ OPTIMIZATION FAILED: {e}")
        return False

    finally:
        if 'engine' in locals():
            engine.dispose()


def show_monitoring_queries():
    """Show useful monitoring queries."""
    print("\n" + "="*60)
    print("MONITORING QUERIES")
    print("="*60)

    queries = [
        {
            "name": "Channel Performance Summary",
            "sql": "SELECT * FROM v_msg_delivery_summary WHERE hour >= NOW() - INTERVAL '24 hours' ORDER BY hour DESC, channel;"
        },
        {
            "name": "Channel Health Status",
            "sql": "SELECT * FROM v_msg_channel_health_summary ORDER BY recent_failure_rate DESC;"
        },
        {
            "name": "User Activity Summary",
            "sql": "SELECT * FROM v_msg_user_activity ORDER BY total_messages DESC LIMIT 10;"
        },
        {
            "name": "Slow Query Detection",
            "sql": """
                SELECT query, calls, total_time, mean_time, rows
                FROM pg_stat_statements
                WHERE query LIKE '%msg_%' AND mean_time > 1000
                ORDER BY mean_time DESC LIMIT 10;
            """
        },
        {
            "name": "Index Usage Statistics",
            "sql": """
                SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read
                FROM pg_stat_user_indexes
                WHERE tablename LIKE 'msg_%'
                ORDER BY idx_scan DESC;
            """
        }
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{i}. {query['name']}:")
        print("   " + query['sql'].strip().replace('\n', '\n   '))

    print("\n" + "="*60)


if __name__ == "__main__":
    success = main()

    if success:
        show_monitoring_queries()
        print("\n✅ Database optimization completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Database optimization failed!")
        sys.exit(1)