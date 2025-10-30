#!/usr/bin/env python3
"""
Test script for Channel Health Monitoring System.
Tests health checks, status evaluation, auto-disable/enable, and monitoring features.
"""
import asyncio
import sys
from pathlib import Path
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.notification.service.health_monitor import (
    HealthMonitor, HealthCheckConfig, HealthStatus, HealthCheckType,
    HealthThreshold, HealthMetric, ChannelHealthStatus,
    create_default_health_config, create_strict_health_config, create_lenient_health_config,
    health_monitor
)
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


async def test_basic_health_monitoring():
    """Test basic health monitoring functionality."""
    print("Testing Basic Health Monitoring...")
    try:
        # Create health monitor
        monitor = HealthMonitor()
        await monitor.start()

        # Configure a test channel
        config = create_default_health_config(
            "test_channel",
            check_interval_seconds=1,  # Fast interval for testing
            enabled_checks={HealthCheckType.CONNECTIVITY, HealthCheckType.RESPONSE_TIME}
        )
        monitor.configure_channel(config)
        print("✓ Configured health monitoring for test channel")

        # Wait for a few health checks
        await asyncio.sleep(3)

        # Check that status was created and updated
        status = monitor.get_channel_status("test_channel")
        assert status is not None
        assert status.channel == "test_channel"
        assert status.overall_status != HealthStatus.UNKNOWN
        print(f"✓ Channel status: {status.overall_status.value}")

        # Check that metrics were collected
        metrics = monitor.get_channel_metrics("test_channel")
        assert len(metrics) > 0
        print(f"✓ Collected {len(metrics)} health metrics")

        # Check specific metric types
        connectivity_metrics = monitor.get_channel_metrics("test_channel", HealthCheckType.CONNECTIVITY)
        response_time_metrics = monitor.get_channel_metrics("test_channel", HealthCheckType.RESPONSE_TIME)
        assert len(connectivity_metrics) > 0
        assert len(response_time_metrics) > 0
        print("✓ Both connectivity and response time metrics collected")

        await monitor.stop()
        return True

    except Exception as e:
        print(f"✗ Basic health monitoring test failed: {e}")
        return False


async def test_health_threshold_evaluation():
    """Test health threshold evaluation logic."""
    print("\nTesting Health Threshold Evaluation...")
    try:
        # Create threshold for success rate
        threshold = HealthThreshold(
            metric_type=HealthCheckType.SUCCESS_RATE,
            healthy_min=95.0,
            degraded_min=85.0,
            unhealthy_min=70.0,
            critical_min=50.0
        )

        # Test different values
        test_cases = [
            (98.0, HealthStatus.HEALTHY),
            (90.0, HealthStatus.DEGRADED),
            (65.0, HealthStatus.UNHEALTHY),  # Below unhealthy_min of 70.0
            (40.0, HealthStatus.CRITICAL),
            (30.0, HealthStatus.CRITICAL)
        ]

        for value, expected_status in test_cases:
            actual_status = threshold.evaluate_status(value)
            assert actual_status == expected_status, f"Value {value} should be {expected_status.value}, got {actual_status.value}"

        print("✓ Success rate threshold evaluation works correctly")

        # Test response time threshold
        response_threshold = HealthThreshold(
            metric_type=HealthCheckType.RESPONSE_TIME,
            healthy_max=1000.0,
            degraded_max=3000.0,
            unhealthy_max=10000.0,
            critical_max=30000.0
        )

        response_test_cases = [
            (500.0, HealthStatus.HEALTHY),
            (2000.0, HealthStatus.DEGRADED),
            (15000.0, HealthStatus.UNHEALTHY),  # Above unhealthy_max of 10000.0
            (45000.0, HealthStatus.CRITICAL)
        ]

        for value, expected_status in response_test_cases:
            actual_status = response_threshold.evaluate_status(value)
            assert actual_status == expected_status, f"Response time {value}ms should be {expected_status.value}, got {actual_status.value}"

        print("✓ Response time threshold evaluation works correctly")
        return True

    except Exception as e:
        print(f"✗ Health threshold evaluation test failed: {e}")
        return False


async def test_auto_disable_enable():
    """Test automatic channel disable and enable functionality."""
    print("\nTesting Auto Disable/Enable...")
    try:
        # Create health monitor
        monitor = HealthMonitor()
        await monitor.start()

        # Configure channel with strict auto-disable settings
        config = HealthCheckConfig(
            channel="auto_test_channel",
            check_interval_seconds=1,
            auto_disable_threshold=3,  # Disable after 3 failures
            auto_enable_threshold=2,   # Enable after 2 successes
            enabled_checks={HealthCheckType.SUCCESS_RATE},
            thresholds={
                HealthCheckType.SUCCESS_RATE: HealthThreshold(
                    metric_type=HealthCheckType.SUCCESS_RATE,
                    healthy_min=95.0,
                    critical_min=50.0
                )
            }
        )
        monitor.configure_channel(config)

        # Get initial status
        status = monitor.get_channel_status("auto_test_channel")
        assert status.is_enabled == True
        assert status.auto_disabled == False
        print("✓ Channel initially enabled")

        # Simulate failures by injecting bad metrics
        for i in range(4):  # More than threshold
            bad_metric = HealthMetric(
                metric_type=HealthCheckType.SUCCESS_RATE,
                value=30.0,  # Critical value
                timestamp=datetime.now(timezone.utc),
                channel="auto_test_channel"
            )

            # Manually update status to simulate failures
            with monitor._lock:
                status = monitor._status["auto_test_channel"]
                status.metrics[HealthCheckType.SUCCESS_RATE] = bad_metric
                status.overall_status = HealthStatus.CRITICAL
                status.consecutive_failures += 1
                status.failure_count += 1

                # Check auto-disable logic
                await monitor._check_auto_disable_enable(status, config)

        # Check if channel was auto-disabled
        status = monitor.get_channel_status("auto_test_channel")
        assert status.is_enabled == False
        assert status.auto_disabled == True
        print("✓ Channel auto-disabled after consecutive failures")

        # Simulate recovery by injecting good metrics
        for i in range(3):  # More than enable threshold
            good_metric = HealthMetric(
                metric_type=HealthCheckType.SUCCESS_RATE,
                value=98.0,  # Healthy value
                timestamp=datetime.now(timezone.utc),
                channel="auto_test_channel"
            )

            # Manually update status to simulate recovery
            with monitor._lock:
                status = monitor._status["auto_test_channel"]
                status.metrics[HealthCheckType.SUCCESS_RATE] = good_metric
                status.overall_status = HealthStatus.HEALTHY
                status.consecutive_failures = 0

                # Add to metric history for auto-enable check
                monitor._metric_history["auto_test_channel"][HealthCheckType.SUCCESS_RATE].append(good_metric)

                # Check auto-enable logic
                await monitor._check_auto_disable_enable(status, config)

        # Check if channel was auto-enabled
        status = monitor.get_channel_status("auto_test_channel")
        assert status.is_enabled == True
        assert status.auto_disabled == False
        print("✓ Channel auto-enabled after recovery")

        await monitor.stop()
        return True

    except Exception as e:
        print(f"✗ Auto disable/enable test failed: {e}")
        return False


async def test_manual_disable_enable():
    """Test manual channel disable and enable functionality."""
    print("\nTesting Manual Disable/Enable...")
    try:
        # Create health monitor
        monitor = HealthMonitor()
        await monitor.start()

        # Configure a test channel
        config = create_default_health_config("manual_test_channel")
        monitor.configure_channel(config)

        # Check initial state
        status = monitor.get_channel_status("manual_test_channel")
        assert status.is_enabled == True
        print("✓ Channel initially enabled")

        # Manually disable
        result = monitor.manually_disable_channel("manual_test_channel", "Testing manual disable")
        assert result == True

        status = monitor.get_channel_status("manual_test_channel")
        assert status.is_enabled == False
        assert status.auto_disabled == False  # Manual, not auto
        print("✓ Channel manually disabled")

        # Manually enable
        result = monitor.manually_enable_channel("manual_test_channel", "Testing manual enable")
        assert result == True

        status = monitor.get_channel_status("manual_test_channel")
        assert status.is_enabled == True
        assert status.auto_disabled == False
        print("✓ Channel manually enabled")

        # Test with non-existent channel
        result = monitor.manually_disable_channel("non_existent", "Should fail")
        assert result == False
        print("✓ Manual operations correctly handle non-existent channels")

        await monitor.stop()
        return True

    except Exception as e:
        print(f"✗ Manual disable/enable test failed: {e}")
        return False


async def test_health_callbacks():
    """Test health status change and alert callbacks."""
    print("\nTesting Health Callbacks...")
    try:
        # Create health monitor
        monitor = HealthMonitor()
        await monitor.start()

        # Track callback invocations
        status_changes = []
        health_alerts = []

        def status_change_callback(channel, old_status, new_status):
            status_changes.append((channel, old_status.value, new_status.value))

        def health_alert_callback(channel, alert_type, message):
            health_alerts.append((channel, alert_type, message))

        # Register callbacks
        monitor.add_status_change_callback(status_change_callback)
        monitor.add_health_alert_callback(health_alert_callback)
        print("✓ Registered health callbacks")

        # Configure channel
        config = create_default_health_config("callback_test_channel")
        monitor.configure_channel(config)

        # Simulate status changes by manually updating status
        with monitor._lock:
            status = monitor._status["callback_test_channel"]
            old_status = status.overall_status

            # Trigger status change callback
            await monitor._trigger_status_change_callbacks(
                "callback_test_channel",
                old_status,
                HealthStatus.DEGRADED
            )

            # Trigger health alert callback
            await monitor._trigger_health_alert_callbacks(
                "callback_test_channel",
                "test_alert",
                "This is a test alert"
            )

        # Check callbacks were invoked
        assert len(status_changes) == 1
        assert status_changes[0][0] == "callback_test_channel"
        assert status_changes[0][2] == "DEGRADED"
        print("✓ Status change callback invoked")

        assert len(health_alerts) == 1
        assert health_alerts[0][0] == "callback_test_channel"
        assert health_alerts[0][1] == "test_alert"
        print("✓ Health alert callback invoked")

        await monitor.stop()
        return True

    except Exception as e:
        print(f"✗ Health callbacks test failed: {e}")
        return False


async def test_health_summary_and_statistics():
    """Test health summary and statistics functionality."""
    print("\nTesting Health Summary and Statistics...")
    try:
        # Create health monitor
        monitor = HealthMonitor()
        await monitor.start()

        # Configure multiple channels with different statuses
        channels = ["healthy_channel", "degraded_channel", "unhealthy_channel"]
        statuses = [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]

        for channel, target_status in zip(channels, statuses):
            config = create_default_health_config(channel)
            monitor.configure_channel(config)

            # Manually set status for testing
            with monitor._lock:
                status = monitor._status[channel]
                status.overall_status = target_status

        print("✓ Configured multiple channels with different statuses")

        # Get health summary
        summary = monitor.get_health_summary()

        assert summary["total_channels"] == 3
        assert "status_distribution" in summary
        assert "healthy_percentage" in summary
        assert "statistics" in summary
        assert "timestamp" in summary

        # Check status distribution
        distribution = summary["status_distribution"]
        assert distribution.get("HEALTHY", 0) == 1
        assert distribution.get("DEGRADED", 0) == 1
        assert distribution.get("UNHEALTHY", 0) == 1

        print(f"✓ Health summary: {summary['healthy_percentage']:.1f}% healthy")
        print(f"✓ Status distribution: {distribution}")

        # Test individual channel status retrieval
        all_statuses = monitor.get_all_statuses()
        assert len(all_statuses) == 3
        assert "healthy_channel" in all_statuses
        assert "degraded_channel" in all_statuses
        assert "unhealthy_channel" in all_statuses
        print("✓ Individual channel status retrieval works")

        await monitor.stop()
        return True

    except Exception as e:
        print(f"✗ Health summary and statistics test failed: {e}")
        return False


async def test_metric_history_and_trends():
    """Test metric history tracking and trend analysis."""
    print("\nTesting Metric History and Trends...")
    try:
        # Create health monitor
        monitor = HealthMonitor()
        await monitor.start()

        # Configure channel
        config = create_default_health_config(
            "history_test_channel",
            metric_history_size=50
        )
        monitor.configure_channel(config)

        # Simulate metric collection over time
        test_metrics = []
        for i in range(10):
            # Create metrics with varying values
            connectivity_metric = HealthMetric(
                metric_type=HealthCheckType.CONNECTIVITY,
                value=90.0 + (i % 3) * 5.0,  # Values between 90-100
                timestamp=datetime.now(timezone.utc),
                channel="history_test_channel"
            )

            response_time_metric = HealthMetric(
                metric_type=HealthCheckType.RESPONSE_TIME,
                value=500.0 + i * 100.0,  # Increasing response times
                timestamp=datetime.now(timezone.utc),
                channel="history_test_channel"
            )

            # Add to history
            monitor._metric_history["history_test_channel"][HealthCheckType.CONNECTIVITY].append(connectivity_metric)
            monitor._metric_history["history_test_channel"][HealthCheckType.RESPONSE_TIME].append(response_time_metric)

            test_metrics.extend([connectivity_metric, response_time_metric])

        print("✓ Simulated metric collection over time")

        # Test metric retrieval
        all_metrics = monitor.get_channel_metrics("history_test_channel")
        assert len(all_metrics) == 20  # 10 connectivity + 10 response time
        print(f"✓ Retrieved {len(all_metrics)} total metrics")

        # Test specific metric type retrieval
        connectivity_metrics = monitor.get_channel_metrics("history_test_channel", HealthCheckType.CONNECTIVITY)
        response_time_metrics = monitor.get_channel_metrics("history_test_channel", HealthCheckType.RESPONSE_TIME)

        assert len(connectivity_metrics) == 10
        assert len(response_time_metrics) == 10
        print("✓ Metric type filtering works correctly")

        # Test metric ordering (should be sorted by timestamp)
        timestamps = [m.timestamp for m in all_metrics]
        assert timestamps == sorted(timestamps), "Metrics should be sorted by timestamp"
        print("✓ Metrics are properly sorted by timestamp")

        # Test uptime calculation
        with monitor._lock:
            status = monitor._status["history_test_channel"]
            await monitor._update_uptime_percentage(status)

            # Should have reasonable uptime since all connectivity values are > 50%
            assert status.uptime_percentage >= 50.0
            print(f"✓ Uptime calculation: {status.uptime_percentage:.1f}%")

        # Test average response time calculation
        with monitor._lock:
            status = monitor._status["history_test_channel"]
            await monitor._update_average_response_time(status, "history_test_channel")

            # Should reflect the increasing response times
            assert status.average_response_time_ms > 500.0
            print(f"✓ Average response time: {status.average_response_time_ms:.1f}ms")

        await monitor.stop()
        return True

    except Exception as e:
        print(f"✗ Metric history and trends test failed: {e}")
        return False


async def test_configuration_presets():
    """Test different health configuration presets."""
    print("\nTesting Configuration Presets...")
    try:
        # Test default configuration
        default_config = create_default_health_config("default_channel")
        assert default_config.channel == "default_channel"
        assert default_config.auto_disable_threshold == 5
        assert default_config.auto_enable_threshold == 3
        assert HealthCheckType.SUCCESS_RATE in default_config.thresholds
        print("✓ Default configuration created correctly")

        # Test strict configuration
        strict_config = create_strict_health_config("strict_channel")
        assert strict_config.channel == "strict_channel"
        assert strict_config.auto_disable_threshold == 3  # More aggressive
        assert strict_config.auto_enable_threshold == 5   # More conservative

        # Check stricter thresholds
        success_threshold = strict_config.thresholds[HealthCheckType.SUCCESS_RATE]
        assert success_threshold.healthy_min == 98.0  # Stricter than default 95.0
        print("✓ Strict configuration has tighter thresholds")

        # Test lenient configuration
        lenient_config = create_lenient_health_config("lenient_channel")
        assert lenient_config.channel == "lenient_channel"
        assert lenient_config.auto_disable_threshold == 10  # Less aggressive
        assert lenient_config.auto_enable_threshold == 2    # Less conservative

        # Check more relaxed thresholds
        success_threshold = lenient_config.thresholds[HealthCheckType.SUCCESS_RATE]
        assert success_threshold.healthy_min == 85.0  # More relaxed than default 95.0
        print("✓ Lenient configuration has relaxed thresholds")

        # Test threshold evaluation differences
        test_value = 90.0  # 90% success rate

        default_status = default_config.thresholds[HealthCheckType.SUCCESS_RATE].evaluate_status(test_value)
        strict_status = strict_config.thresholds[HealthCheckType.SUCCESS_RATE].evaluate_status(test_value)
        lenient_status = lenient_config.thresholds[HealthCheckType.SUCCESS_RATE].evaluate_status(test_value)

        # 90% should be degraded for default/strict, but healthy for lenient
        assert default_status == HealthStatus.DEGRADED
        assert strict_status == HealthStatus.DEGRADED
        assert lenient_status == HealthStatus.HEALTHY
        print("✓ Configuration presets evaluate thresholds differently")

        return True

    except Exception as e:
        print(f"✗ Configuration presets test failed: {e}")
        return False


async def test_channel_management():
    """Test channel configuration and removal."""
    print("\nTesting Channel Management...")
    try:
        # Create health monitor
        monitor = HealthMonitor()
        await monitor.start()

        # Configure multiple channels
        channels = ["channel1", "channel2", "channel3"]
        for channel in channels:
            config = create_default_health_config(channel)
            monitor.configure_channel(config)

        # Check all channels are configured
        all_statuses = monitor.get_all_statuses()
        assert len(all_statuses) == 3
        for channel in channels:
            assert channel in all_statuses
        print("✓ Multiple channels configured successfully")

        # Remove a channel
        result = monitor.remove_channel("channel2")
        assert result == True

        # Check channel was removed
        all_statuses = monitor.get_all_statuses()
        assert len(all_statuses) == 2
        assert "channel1" in all_statuses
        assert "channel2" not in all_statuses
        assert "channel3" in all_statuses
        print("✓ Channel removal works correctly")

        # Try to remove non-existent channel
        result = monitor.remove_channel("non_existent")
        assert result == False
        print("✓ Removing non-existent channel handled correctly")

        # Reset metrics for remaining channel
        result = monitor.reset_channel_metrics("channel1")
        assert result == True

        status = monitor.get_channel_status("channel1")
        assert status.failure_count == 0
        assert status.consecutive_failures == 0
        assert status.uptime_percentage == 100.0
        print("✓ Channel metrics reset successfully")

        await monitor.stop()
        return True

    except Exception as e:
        print(f"✗ Channel management test failed: {e}")
        return False


async def test_concurrent_monitoring():
    """Test concurrent health monitoring for multiple channels."""
    print("\nTesting Concurrent Monitoring...")
    try:
        # Create health monitor
        monitor = HealthMonitor()
        await monitor.start()

        # Configure multiple channels with fast check intervals
        channels = [f"concurrent_channel_{i}" for i in range(5)]
        for channel in channels:
            config = create_default_health_config(
                channel,
                check_interval_seconds=0.5  # Very fast for testing
            )
            monitor.configure_channel(config)

        print(f"✓ Configured {len(channels)} channels for concurrent monitoring")

        # Wait for multiple check cycles
        await asyncio.sleep(3)

        # Check that all channels have been monitored
        for channel in channels:
            status = monitor.get_channel_status(channel)
            assert status is not None
            assert status.overall_status != HealthStatus.UNKNOWN

            metrics = monitor.get_channel_metrics(channel)
            assert len(metrics) > 0  # Should have collected some metrics

        print("✓ All channels monitored concurrently")

        # Check health summary
        summary = monitor.get_health_summary()
        assert summary["total_channels"] == len(channels)
        assert summary["statistics"]["total_checks"] > 0
        print(f"✓ Health summary shows {summary['statistics']['total_checks']} total checks")

        await monitor.stop()
        return True

    except Exception as e:
        print(f"✗ Concurrent monitoring test failed: {e}")
        return False


async def main():
    """Run all health monitor tests."""
    print("=== Channel Health Monitoring Tests ===\n")

    tests = [
        ("Basic Health Monitoring", test_basic_health_monitoring),
        ("Health Threshold Evaluation", test_health_threshold_evaluation),
        ("Auto Disable/Enable", test_auto_disable_enable),
        ("Manual Disable/Enable", test_manual_disable_enable),
        ("Health Callbacks", test_health_callbacks),
        ("Health Summary and Statistics", test_health_summary_and_statistics),
        ("Metric History and Trends", test_metric_history_and_trends),
        ("Configuration Presets", test_configuration_presets),
        ("Channel Management", test_channel_management),
        ("Concurrent Monitoring", test_concurrent_monitoring),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"Running {test_name} test...")
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))
        print()

    # Summary
    print("=== Test Results ===")
    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("🎉 All health monitoring tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)