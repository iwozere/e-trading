#!/usr/bin/env python3
"""
Health Monitoring Integration Example
Demonstrates how to integrate the health monitoring system with the notification service.
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.notification.service.health_monitor import (
    HealthMonitor, HealthCheckConfig, HealthStatus, HealthCheckType,
    create_default_health_config, create_strict_health_config,
    health_monitor
)
from src.notification.service.delivery_tracker import MessageDeliveryStatus
from src.data.db.models.model_notification import MessagePriority
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


async def health_status_changed(channel: str, old_status: HealthStatus, new_status: HealthStatus):
    """Callback for health status changes."""
    _logger.info(
        "Channel %s health changed: %s -> %s",
        channel, old_status.value, new_status.value
    )

    # In a real implementation, you might:
    # - Send alerts to administrators
    # - Update monitoring dashboards
    # - Trigger failover procedures
    # - Log to external monitoring systems


async def health_alert_received(channel: str, alert_type: str, message: str):
    """Callback for health alerts."""
    _logger.warning("Health Alert [%s] %s: %s", channel, alert_type, message)

    # In a real implementation, you might:
    # - Send notifications to operations team
    # - Create incident tickets
    # - Trigger automated recovery procedures
    # - Update status pages


async def simulate_delivery_integration():
    """Simulate integration between health monitoring and delivery tracking."""
    print("=== Health Monitoring Integration Example ===\n")

    # Start health monitor
    monitor = HealthMonitor()
    await monitor.start()

    # Register callbacks
    monitor.add_status_change_callback(health_status_changed)
    monitor.add_health_alert_callback(health_alert_received)

    # Configure health monitoring for different channels
    channels = {
        "telegram_channel": create_default_health_config("telegram_channel", check_interval_seconds=2),
        "email_channel": create_strict_health_config("email_channel", check_interval_seconds=2),
        "sms_channel": create_default_health_config("sms_channel", check_interval_seconds=2),
    }

    for channel, config in channels.items():
        monitor.configure_channel(config)
        print(f"✓ Configured health monitoring for {channel}")

    print("\n--- Monitoring Channel Health ---")

    # Let the health monitoring run for a while
    await asyncio.sleep(5)

    # Get health summary
    summary = monitor.get_health_summary()
    print(f"\nHealth Summary:")
    print(f"  Total Channels: {summary['total_channels']}")
    print(f"  Healthy: {summary['healthy_percentage']:.1f}%")
    print(f"  Status Distribution: {summary['status_distribution']}")
    print(f"  Total Health Checks: {summary['statistics']['total_checks']}")

    # Show individual channel status
    print(f"\nIndividual Channel Status:")
    for channel in channels.keys():
        status = monitor.get_channel_status(channel)
        if status:
            print(f"  {channel}: {status.overall_status.value} "
                  f"(Uptime: {status.uptime_percentage:.1f}%, "
                  f"Avg Response: {status.average_response_time_ms:.1f}ms)")

    # Simulate manual channel management
    print(f"\n--- Manual Channel Management ---")

    # Manually disable a channel
    result = monitor.manually_disable_channel("sms_channel", "Maintenance window")
    if result:
        print("✓ Manually disabled SMS channel for maintenance")

    # Check status after disable
    status = monitor.get_channel_status("sms_channel")
    print(f"SMS Channel Status: Enabled={status.is_enabled}, Auto-disabled={status.auto_disabled}")

    # Re-enable the channel
    result = monitor.manually_enable_channel("sms_channel", "Maintenance complete")
    if result:
        print("✓ Re-enabled SMS channel after maintenance")

    # Demonstrate health metrics retrieval
    print(f"\n--- Health Metrics Analysis ---")

    for channel in ["telegram_channel", "email_channel"]:
        metrics = monitor.get_channel_metrics(channel, HealthCheckType.RESPONSE_TIME)
        if metrics:
            avg_response = sum(m.value for m in metrics) / len(metrics)
            print(f"{channel} - Average Response Time: {avg_response:.1f}ms ({len(metrics)} samples)")

        connectivity_metrics = monitor.get_channel_metrics(channel, HealthCheckType.CONNECTIVITY)
        if connectivity_metrics:
            avg_connectivity = sum(m.value for m in connectivity_metrics) / len(connectivity_metrics)
            print(f"{channel} - Average Connectivity: {avg_connectivity:.1f}% ({len(connectivity_metrics)} samples)")

    # Demonstrate integration with delivery status
    print(f"\n--- Delivery Integration Example ---")

    # Create a sample delivery status
    delivery_status = MessageDeliveryStatus(
        message_id=12345,
        message_type="alert_notification",
        priority=MessagePriority.HIGH,
        channels=["telegram_channel", "email_channel"],
        recipient_id="user123",
        created_at=datetime.now(timezone.utc)
    )

    print(f"Created delivery status for message {delivery_status.message_id}")

    # In a real implementation, you would:
    # 1. Check channel health before attempting delivery
    # 2. Skip disabled channels or use fallbacks
    # 3. Update health metrics based on delivery results
    # 4. Trigger health alerts on delivery failures

    # Example health check before delivery
    healthy_channels = []
    for channel in delivery_status.channels:
        channel_status = monitor.get_channel_status(channel)
        if channel_status and channel_status.is_enabled and channel_status.overall_status in [
            HealthStatus.HEALTHY, HealthStatus.DEGRADED
        ]:
            healthy_channels.append(channel)
        else:
            print(f"⚠️  Skipping {channel} - not healthy for delivery")

    print(f"✓ Healthy channels for delivery: {healthy_channels}")

    # Demonstrate testing a callback configuration
    print(f"\n--- Health Check Testing ---")

    # Test each channel's health configuration
    for channel in channels.keys():
        # Show the current status and recent metrics
        status = monitor.get_channel_status(channel)
        recent_metrics = monitor.get_channel_metrics(channel)
        print(f"{channel} status: {status.overall_status.value}, "
              f"Response={status.average_response_time_ms:.1f}ms, "
              f"Metrics collected={len(recent_metrics)}")

    print(f"\n--- Cleanup ---")

    # Stop health monitoring
    await monitor.stop()
    print("✓ Health monitoring stopped")

    print(f"\n=== Integration Example Complete ===")


async def main():
    """Run the health monitoring integration example."""
    try:
        await simulate_delivery_integration()
        return 0
    except Exception as e:
        _logger.exception("Integration example failed:")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)