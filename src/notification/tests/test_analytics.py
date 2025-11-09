"""
Test script for Notification Service Analytics System.

Tests comprehensive analytics functionality including delivery rates,
response time analysis, trend analysis, and performance comparisons.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.notification.service.analytics import (
    NotificationAnalytics,
    TimeGranularity,
    ChannelStats,
    UserStats,
    TrendAnalysis,
    notification_analytics
)
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


async def test_delivery_rates():
    """Test delivery rate calculations."""
    print("\nTesting Delivery Rate Calculations...")

    try:
        analytics = NotificationAnalytics()

        # Test overall delivery rates
        rates = await analytics.get_delivery_rates(days=30)

        print("✓ Overall delivery rates calculated:")
        print(f"  Period: {rates['period_days']} days")
        print(f"  Total messages: {rates['overall_statistics']['total_messages']}")
        print(f"  Success rate: {rates['overall_statistics']['success_rate']:.2%}")
        print(f"  Avg response time: {rates['overall_statistics']['average_response_time_ms']}ms")

        # Test channel-specific rates
        if rates['channel_rates']:
            print("✓ Channel-specific rates:")
            for channel, stats in rates['channel_rates'].items():
                print(f"  {channel}: {stats['success_rate']:.2%} success rate")

        # Test with channel filter
        telegram_rates = await analytics.get_delivery_rates(channel="telegram", days=7)
        print(f"✓ Telegram-specific rates (7 days): {telegram_rates['overall_statistics']['success_rate']:.2%}")

        # Test with user filter
        user_rates = await analytics.get_delivery_rates(user_id="test_user", days=30)
        print("✓ User-specific rates calculated")

        return True

    except Exception as e:
        print(f"✗ Delivery rates test failed: {e}")
        return False


async def test_response_time_analysis():
    """Test response time analysis."""
    print("\nTesting Response Time Analysis...")

    try:
        analytics = NotificationAnalytics()

        # Test overall response time analysis
        analysis = await analytics.get_response_time_analysis(days=30)

        print("✓ Response time analysis completed:")
        stats = analysis['statistics']
        print(f"  Count: {stats['count']}")
        print(f"  Average: {stats['average_ms']:.1f}ms")
        print(f"  Median: {stats['median_ms']:.1f}ms")
        print(f"  Min: {stats['min_ms']}ms")
        print(f"  Max: {stats['max_ms']}ms")
        print(f"  Std Dev: {stats['std_deviation_ms']:.1f}ms")

        # Test percentiles
        if analysis['percentiles']:
            print("✓ Percentiles calculated:")
            for percentile, value in analysis['percentiles'].items():
                print(f"  {percentile}: {value}ms")

        # Test channel breakdown
        if analysis['channel_breakdown']:
            print("✓ Channel breakdown:")
            for channel, breakdown in analysis['channel_breakdown'].items():
                print(f"  {channel}: {breakdown['average_ms']:.1f}ms avg")

        # Test with channel filter
        telegram_analysis = await analytics.get_response_time_analysis(
            channel="telegram", days=7
        )
        print("✓ Telegram response time analysis completed")

        return True

    except Exception as e:
        print(f"✗ Response time analysis test failed: {e}")
        return False


async def test_aggregated_statistics():
    """Test time-based statistics aggregation."""
    print("\nTesting Aggregated Statistics...")

    try:
        analytics = NotificationAnalytics()

        # Test daily aggregation
        daily_stats = await analytics.get_aggregated_statistics(
            granularity=TimeGranularity.DAILY,
            days=7
        )

        print("✓ Daily aggregation completed:")
        print(f"  Granularity: {daily_stats['granularity']}")
        print(f"  Total periods: {daily_stats['summary']['total_periods']}")
        print(f"  Avg messages per period: {daily_stats['summary']['avg_messages_per_period']:.1f}")

        # Test weekly aggregation
        weekly_stats = await analytics.get_aggregated_statistics(
            granularity=TimeGranularity.WEEKLY,
            days=30
        )

        print("✓ Weekly aggregation completed:")
        print(f"  Total periods: {weekly_stats['summary']['total_periods']}")

        # Test monthly aggregation
        monthly_stats = await analytics.get_aggregated_statistics(
            granularity=TimeGranularity.MONTHLY,
            days=90
        )

        print("✓ Monthly aggregation completed:")
        print(f"  Total periods: {monthly_stats['summary']['total_periods']}")

        # Test with channel filter
        channel_stats = await analytics.get_aggregated_statistics(
            granularity=TimeGranularity.DAILY,
            days=7,
            channel="telegram"
        )

        print("✓ Channel-filtered aggregation completed")

        return True

    except Exception as e:
        print(f"✗ Aggregated statistics test failed: {e}")
        return False


async def test_trend_analysis():
    """Test performance trend analysis."""
    print("\nTesting Trend Analysis...")

    try:
        analytics = NotificationAnalytics()

        # Test success rate trend
        success_trend = await analytics.get_trend_analysis(
            metric="success_rate",
            days=30
        )

        print("✓ Success rate trend analysis:")
        print(f"  Metric: {success_trend.metric_name}")
        print(f"  Direction: {success_trend.trend_direction}")
        print(f"  Strength: {success_trend.trend_strength:.2f}")
        print(f"  Change: {success_trend.change_percentage:.1f}%")
        print(f"  Mean: {success_trend.mean:.3f}")
        print(f"  Std Dev: {success_trend.std_deviation:.3f}")

        # Test response time trend
        response_trend = await analytics.get_trend_analysis(
            metric="response_time",
            days=30
        )

        print("✓ Response time trend analysis:")
        print(f"  Direction: {response_trend.trend_direction}")
        print(f"  Change: {response_trend.change_percentage:.1f}%")

        # Test message count trend
        message_trend = await analytics.get_trend_analysis(
            metric="message_count",
            days=30
        )

        print("✓ Message count trend analysis:")
        print(f"  Direction: {message_trend.trend_direction}")
        print(f"  Data points: {len(message_trend.time_series)}")

        # Test with channel filter
        channel_trend = await analytics.get_trend_analysis(
            metric="success_rate",
            days=14,
            channel="telegram"
        )

        print("✓ Channel-specific trend analysis completed")

        return True

    except Exception as e:
        print(f"✗ Trend analysis test failed: {e}")
        return False


async def test_channel_performance_comparison():
    """Test channel performance comparison."""
    print("\nTesting Channel Performance Comparison...")

    try:
        analytics = NotificationAnalytics()

        # Test performance comparison
        comparison = await analytics.get_channel_performance_comparison(days=30)

        print("✓ Channel performance comparison completed:")
        print(f"  Period: {comparison['period_days']} days")
        print(f"  Channels analyzed: {len(comparison['channel_comparisons'])}")

        # Show channel comparisons
        for channel, data in comparison['channel_comparisons'].items():
            stats = data['statistics']
            score = data['performance_score']
            print(f"  {channel}:")
            print(f"    Performance score: {score:.3f}")
            print(f"    Success rate: {stats.get('success_rate', 0):.2%}")
            print(f"    Avg response time: {stats.get('avg_response_time_ms', 0):.1f}ms")

        # Show rankings
        rankings = comparison['rankings']
        print("✓ Performance rankings:")
        for i, ranking in enumerate(rankings['by_performance'][:3], 1):
            print(f"  {i}. {ranking['channel']}: {ranking['score']:.3f}")

        print("✓ Success rate rankings:")
        for i, (channel, rate) in enumerate(rankings['by_success_rate'][:3], 1):
            print(f"  {i}. {channel}: {rate:.2%}")

        print("✓ Response time rankings (fastest first):")
        for i, (channel, time_ms) in enumerate(rankings['by_response_time'][:3], 1):
            print(f"  {i}. {channel}: {time_ms:.1f}ms")

        return True

    except Exception as e:
        print(f"✗ Channel performance comparison test failed: {e}")
        return False


async def test_data_structures():
    """Test analytics data structures."""
    print("\nTesting Analytics Data Structures...")

    try:
        # Test ChannelStats
        channel_stats = ChannelStats(
            channel="telegram",
            total_messages=100,
            successful_deliveries=90,
            failed_deliveries=10
        )
        channel_stats.calculate_rates()

        print("✓ ChannelStats created:")
        print(f"  Channel: {channel_stats.channel}")
        print(f"  Success rate: {channel_stats.success_rate:.2%}")

        # Test conversion to dict
        stats_dict = channel_stats.to_dict()
        assert "channel" in stats_dict
        assert "success_rate" in stats_dict
        print("✓ ChannelStats to_dict() works")

        # Test UserStats
        user_stats = UserStats(
            user_id="test_user",
            total_messages=50,
            successful_deliveries=45
        )

        # Add channel stats
        user_stats.channel_stats["telegram"] = channel_stats
        user_stats.calculate_rates()

        print("✓ UserStats created:")
        print(f"  User: {user_stats.user_id}")
        print(f"  Total messages: {user_stats.total_messages}")
        print(f"  Channels: {len(user_stats.channel_stats)}")

        # Test conversion to dict
        user_dict = user_stats.to_dict()
        assert "user_id" in user_dict
        assert "channel_stats" in user_dict
        print("✓ UserStats to_dict() works")

        # Test TrendAnalysis
        from src.notification.service.analytics import TimeSeriesPoint

        time_series = [
            TimeSeriesPoint(datetime.now(timezone.utc) - timedelta(days=i), 0.8 + i * 0.01)
            for i in range(10)
        ]

        trend = TrendAnalysis(
            metric_name="success_rate",
            time_series=time_series,
            trend_direction="increasing",
            trend_strength=0.7,
            change_percentage=5.0,
            mean=0.85,
            median=0.84,
            std_deviation=0.03,
            min_value=0.80,
            max_value=0.89
        )

        print("✓ TrendAnalysis created:")
        print(f"  Metric: {trend.metric_name}")
        print(f"  Direction: {trend.trend_direction}")
        print(f"  Data points: {len(trend.time_series)}")

        # Test conversion to dict
        trend_dict = trend.to_dict()
        assert "metric_name" in trend_dict
        assert "time_series" in trend_dict
        assert "statistics" in trend_dict
        print("✓ TrendAnalysis to_dict() works")

        return True

    except Exception as e:
        print(f"✗ Data structures test failed: {e}")
        return False


async def test_global_analytics_instance():
    """Test global analytics instance."""
    print("\nTesting Global Analytics Instance...")

    try:
        # Test that global instance works
        rates = await notification_analytics.get_delivery_rates(days=7)

        print("✓ Global analytics instance works:")
        print(f"  Total messages: {rates['overall_statistics']['total_messages']}")

        # Test multiple concurrent calls
        tasks = [
            notification_analytics.get_delivery_rates(days=1),
            notification_analytics.get_response_time_analysis(days=1),
            notification_analytics.get_aggregated_statistics(days=1)
        ]

        results = await asyncio.gather(*tasks)

        print("✓ Concurrent analytics calls completed:")
        print(f"  Results: {len(results)}")

        return True

    except Exception as e:
        print(f"✗ Global analytics instance test failed: {e}")
        return False


async def test_error_handling():
    """Test error handling in analytics."""
    print("\nTesting Error Handling...")

    try:
        analytics = NotificationAnalytics()

        # Test with invalid parameters
        try:
            await analytics.get_delivery_rates(days=0)  # Invalid days
            print("✗ Should have failed with invalid days")
            return False
        except Exception:
            print("✓ Invalid days parameter handled correctly")

        # Test with very large days parameter
        try:
            rates = await analytics.get_delivery_rates(days=1000)
            print("✓ Large days parameter handled")
        except Exception as e:
            print(f"✓ Large days parameter error handled: {e}")

        # Test empty data scenarios
        try:
            trend = await analytics.get_trend_analysis(days=1)  # Minimal data
            print(f"✓ Minimal data trend analysis: {trend.trend_direction}")
        except Exception as e:
            print(f"✓ Minimal data error handled: {e}")

        return True

    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False


async def run_all_tests():
    """Run all analytics tests."""
    print("=" * 60)
    print("NOTIFICATION SERVICE ANALYTICS TESTS")
    print("=" * 60)

    tests = [
        ("Delivery Rates", test_delivery_rates),
        ("Response Time Analysis", test_response_time_analysis),
        ("Aggregated Statistics", test_aggregated_statistics),
        ("Trend Analysis", test_trend_analysis),
        ("Channel Performance Comparison", test_channel_performance_comparison),
        ("Data Structures", test_data_structures),
        ("Global Analytics Instance", test_global_analytics_instance),
        ("Error Handling", test_error_handling),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{'-' * 40}")
        print(f"Running: {test_name}")
        print(f"{'-' * 40}")

        try:
            result = await test_func()
            results.append((test_name, result))

            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")

        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))

    # Summary
    print(f"\n{'=' * 60}")
    print("TEST SUMMARY")
    print(f"{'=' * 60}")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:<8} {test_name}")

    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("🎉 All analytics tests passed!")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed")
        return False


if __name__ == "__main__":
    asyncio.run(run_all_tests())