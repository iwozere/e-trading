#!/usr/bin/env python3
"""
Test System Monitoring Service
------------------------------

Simple script to test the monitoring service functionality.
"""

import asyncio
import json
from src.web_ui.backend.services.monitoring_service import SystemMonitoringService

def test_monitoring_service():
    """Test the monitoring service."""
    print("🔍 Testing System Monitoring Service")
    print("=" * 50)

    # Initialize service
    monitoring = SystemMonitoringService()

    # Get comprehensive metrics
    print("📊 Getting system metrics...")
    metrics = monitoring.get_comprehensive_metrics()

    print(f"CPU Usage: {metrics['cpu']['usage_percent']}%")
    print(f"Memory Usage: {metrics['memory']['usage_percent']}%")
    print(f"Temperature: {metrics['temperature']['average_celsius']}°C")

    # Show disk usage
    print("\n💾 Disk Usage:")
    for device, disk_info in metrics['disk']['partitions'].items():
        print(f"  {device}: {disk_info['usage_percent']}% ({disk_info['used_gb']:.1f}GB / {disk_info['total_gb']:.1f}GB)")

    # Show alerts
    print(f"\n🚨 Alerts: {len(monitoring.get_alerts())}")
    for alert in monitoring.get_alerts():
        print(f"  {alert['severity'].upper()}: {alert['message']}")

    # Service status
    status = monitoring.get_service_status()
    print(f"\n✅ Service Status: {status['status']}")
    print(f"   Uptime: {status['uptime_seconds']:.1f} seconds")
    print(f"   Total Alerts: {status['total_alerts']}")

    print("\n🎉 Monitoring service test completed!")

if __name__ == "__main__":
    test_monitoring_service()