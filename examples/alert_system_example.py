#!/usr/bin/env python3
"""
Smart Alert System Example
==========================

Demonstrates how to use the smart alert system with:
- Custom alert rules
- Performance metrics integration
- Alert aggregation
- Alert management
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.notification.alert_system import (
    SmartAlertSystem, AlertRule, AlertSeverity, AlertChannel
)
from src.notification.async_notification_manager import AsyncNotificationManager


async def main():
    """Main example function"""
    print("🚨 Smart Alert System Example")
    print("=" * 50)
    
    # Initialize notification manager (mock for demo)
    notification_manager = AsyncNotificationManager()
    
    # Initialize smart alert system
    alert_system = SmartAlertSystem(notification_manager)
    
    print("✅ Alert system initialized with default rules")
    
    # Add custom alert rule
    custom_rule = AlertRule(
        name="custom_volatility_alert",
        condition="volatility > 50",
        severity=AlertSeverity.WARNING,
        channels=[AlertChannel.TELEGRAM],
        cooldown="15m",
        description="Alert when volatility exceeds 50%",
        template="📊 High Volatility: {volatility:.1f}%"
    )
    alert_system.add_alert_rule(custom_rule)
    print("✅ Added custom volatility alert rule")
    
    # Simulate performance metrics updates
    print("\n📈 Simulating performance metrics updates...")
    
    # Update 1: Normal conditions
    metrics_1 = {
        "max_drawdown_pct": 8.5,
        "daily_pnl": 2.3,
        "max_consecutive_losses": 2,
        "sharpe_ratio": 1.2,
        "api_errors": 1,
        "volatility": 25.0
    }
    alert_system.update_performance_metrics(metrics_1)
    await alert_system.evaluate_alerts()
    print("📊 Metrics 1: Normal conditions - No alerts triggered")
    
    # Update 2: High drawdown
    metrics_2 = {
        "max_drawdown_pct": 18.5,
        "daily_pnl": -1.2,
        "max_consecutive_losses": 4,
        "sharpe_ratio": 0.8,
        "api_errors": 2,
        "volatility": 35.0
    }
    alert_system.update_performance_metrics(metrics_2)
    await alert_system.evaluate_alerts()
    print("📊 Metrics 2: High drawdown & consecutive losses - Alerts triggered")
    
    # Update 3: Critical conditions
    metrics_3 = {
        "max_drawdown_pct": 22.0,
        "daily_pnl": -5.8,
        "max_consecutive_losses": 6,
        "sharpe_ratio": 0.5,
        "api_errors": 8,
        "volatility": 65.0
    }
    alert_system.update_performance_metrics(metrics_3)
    await alert_system.evaluate_alerts()
    print("📊 Metrics 3: Critical conditions - Multiple alerts triggered")
    
    # Display alert information
    print("\n📋 Alert Information:")
    print("-" * 30)
    
    # Active alerts
    active_alerts = alert_system.get_active_alerts()
    print(f"Active Alerts: {len(active_alerts)}")
    for alert in active_alerts:
        print(f"  • {alert.rule_name}: {alert.message}")
    
    # Alert history
    history = alert_system.get_alert_history(hours=1)
    print(f"\nAlert History (last hour): {len(history)}")
    for alert in history:
        print(f"  • {alert.timestamp.strftime('%H:%M:%S')} - {alert.rule_name}: {alert.severity.value}")
    
    # Alert statistics
    stats = alert_system.get_alert_statistics()
    print(f"\nAlert Statistics:")
    print(f"  • Total Alerts: {stats['total_alerts']}")
    print(f"  • Active Alerts: {stats['active_alerts']}")
    print(f"  • Severity Distribution: {stats['severity_distribution']}")
    print(f"  • Most Frequent Rule: {stats['most_frequent_rule']}")
    
    # Demonstrate alert acknowledgment
    if active_alerts:
        first_alert = active_alerts[0]
        alert_system.acknowledge_alert(first_alert.rule_name)
        print(f"\n✅ Acknowledged alert: {first_alert.rule_name}")
    
    # Export configuration
    config = alert_system.export_configuration()
    print(f"\n📄 Configuration exported: {len(config['alert_rules'])} rules")
    
    # Demonstrate alert aggregation
    print("\n🔄 Testing alert aggregation...")
    
    # Trigger multiple similar alerts quickly
    for i in range(5):
        metrics = {
            "max_drawdown_pct": 20.0 + i,
            "daily_pnl": -2.0,
            "max_consecutive_losses": 3,
            "sharpe_ratio": 0.7,
            "api_errors": 3,
            "volatility": 40.0
        }
        alert_system.update_performance_metrics(metrics)
        await alert_system.evaluate_alerts()
        await asyncio.sleep(0.1)  # Small delay
    
    print("✅ Alert aggregation test completed")
    
    # Final statistics
    final_stats = alert_system.get_alert_statistics()
    print(f"\n📊 Final Statistics:")
    print(f"  • Total Alerts: {final_stats['total_alerts']}")
    print(f"  • Active Alerts: {final_stats['active_alerts']}")
    
    print("\n🎉 Smart Alert System example completed!")


if __name__ == "__main__":
    asyncio.run(main()) 