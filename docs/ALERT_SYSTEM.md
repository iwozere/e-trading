# Smart Alert System Documentation

## Overview

The Smart Alert System provides intelligent, configurable alerting for the crypto trading platform. It integrates with the existing notification infrastructure to deliver timely, relevant alerts based on performance metrics, risk indicators, and system events.

## Features

### 🎯 **Smart Alert Rules Engine**
- **Configurable Conditions**: Python expressions for flexible alert triggers
- **Multiple Severity Levels**: Info, Warning, High, Critical
- **Channel Support**: Telegram, Email, SMS, Webhook
- **Cooldown Management**: Prevent alert spam with configurable delays

### 🔄 **Alert Aggregation & Filtering**
- **Smart Aggregation**: Groups similar alerts to reduce noise
- **Time-based Filtering**: Configurable aggregation windows
- **Priority-based Processing**: Handle alerts by severity level

### 📈 **Performance-Based Alerts**
- **Drawdown Monitoring**: Alert on excessive drawdowns
- **Profit/Loss Tracking**: Notify on profit targets and losses
- **Risk Metrics**: Monitor Sharpe ratio, consecutive losses
- **System Health**: Track API errors and system issues

### 📊 **Alert Management**
- **Alert History**: Complete audit trail of all alerts
- **Statistics**: Alert frequency and distribution analysis
- **Acknowledgment System**: Mark alerts as handled
- **Configuration Export/Import**: Backup and restore alert rules

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Performance     │───▶│ Smart Alert      │───▶│ Notification    │
│ Metrics         │    │ System           │    │ Manager         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │ Alert History    │
                       │ & Statistics     │
                       └──────────────────┘
```

## Components

### AlertRule
Defines alert conditions and behavior:

```python
@dataclass
class AlertRule:
    name: str                    # Unique rule identifier
    condition: str               # Python expression (e.g., "max_drawdown > 15")
    severity: AlertSeverity      # Info, Warning, High, Critical
    channels: List[AlertChannel] # Telegram, Email, SMS, Webhook
    cooldown: str               # "5m", "1h", "1d"
    enabled: bool               # Enable/disable rule
    description: str            # Human-readable description
    template: str               # Message template with placeholders
```

### Alert
Represents an individual alert instance:

```python
@dataclass
class Alert:
    rule_name: str              # Source rule name
    severity: AlertSeverity     # Alert severity
    message: str                # Formatted alert message
    data: Dict[str, Any]        # Associated data/metrics
    timestamp: datetime         # When alert was triggered
    acknowledged: bool          # Whether alert was acknowledged
    escalated_level: int        # Escalation level (0 = not escalated)
```

### AlertAggregator
Reduces alert noise by grouping similar alerts:

- **Grouping Logic**: Alerts with same rule and severity
- **Aggregation Window**: Configurable time window (default: 5 minutes)
- **Threshold**: Minimum alerts before aggregation (default: 3)
- **Message Formatting**: Creates aggregated message with count

### SmartAlertSystem
Main orchestrator for the alert system:

- **Rule Management**: Add, remove, enable/disable rules
- **Metric Updates**: Receive performance metrics for evaluation
- **Alert Evaluation**: Check conditions and trigger alerts
- **Notification Dispatch**: Send alerts to configured channels
- **History Management**: Track and query alert history

## Default Alert Rules

The system comes with pre-configured alert rules:

### 1. Drawdown Alert
```python
AlertRule(
    name="drawdown_alert",
    condition="max_drawdown > 15",
    severity=AlertSeverity.HIGH,
    channels=[AlertChannel.TELEGRAM, AlertChannel.EMAIL],
    cooldown="1h",
    description="Alert when max drawdown exceeds 15%",
    template="🚨 High Drawdown Alert: {max_drawdown:.2f}%"
)
```

### 2. Profit Target
```python
AlertRule(
    name="profit_target",
    condition="daily_pnl > 5",
    severity=AlertSeverity.INFO,
    channels=[AlertChannel.TELEGRAM],
    cooldown="4h",
    description="Alert when daily PnL exceeds 5%",
    template="💰 Profit Target Hit: {daily_pnl:.2f}%"
)
```

### 3. Consecutive Losses
```python
AlertRule(
    name="consecutive_losses",
    condition="consecutive_losses >= 3",
    severity=AlertSeverity.WARNING,
    channels=[AlertChannel.TELEGRAM, AlertChannel.EMAIL],
    cooldown="30m",
    description="Alert when 3+ consecutive losses",
    template="⚠️ Consecutive Losses: {consecutive_losses} losses in a row"
)
```

### 4. Sharpe Ratio Drop
```python
AlertRule(
    name="sharpe_ratio_drop",
    condition="sharpe_ratio < 1.0",
    severity=AlertSeverity.WARNING,
    channels=[AlertChannel.TELEGRAM],
    cooldown="2h",
    description="Alert when Sharpe ratio drops below 1.0",
    template="📉 Sharpe Ratio Alert: {sharpe_ratio:.2f} (target: 1.0)"
)
```

### 5. API Error Alert
```python
AlertRule(
    name="api_error",
    condition="api_errors > 5",
    severity=AlertSeverity.CRITICAL,
    channels=[AlertChannel.TELEGRAM, AlertChannel.EMAIL],
    cooldown="10m",
    description="Alert when API errors exceed threshold",
    template="🚨 API Error Alert: {api_errors} errors in last hour"
)
```

## Usage Examples

### Basic Integration

```python
from src.notification.alert_system import SmartAlertSystem
from src.notification.async_notification_manager import AsyncNotificationManager

# Initialize
notification_manager = AsyncNotificationManager()
alert_system = SmartAlertSystem(notification_manager)

# Update metrics
metrics = {
    "max_drawdown_pct": 18.5,
    "daily_pnl": 2.3,
    "max_consecutive_losses": 4,
    "sharpe_ratio": 0.8,
    "api_errors": 2
}
alert_system.update_performance_metrics(metrics)

# Evaluate alerts
await alert_system.evaluate_alerts()
```

### Custom Alert Rule

```python
from src.notification.alert_system import AlertRule, AlertSeverity, AlertChannel

# Create custom rule
volatility_rule = AlertRule(
    name="high_volatility",
    condition="volatility > 50",
    severity=AlertSeverity.WARNING,
    channels=[AlertChannel.TELEGRAM],
    cooldown="15m",
    description="Alert when volatility exceeds 50%",
    template="📊 High Volatility: {volatility:.1f}%"
)

# Add to system
alert_system.add_alert_rule(volatility_rule)
```

### Alert Management

```python
# Get active alerts
active_alerts = alert_system.get_active_alerts()
for alert in active_alerts:
    print(f"{alert.rule_name}: {alert.message}")

# Acknowledge alert
alert_system.acknowledge_alert("drawdown_alert")

# Get statistics
stats = alert_system.get_alert_statistics()
print(f"Total alerts: {stats['total_alerts']}")
print(f"Active alerts: {stats['active_alerts']}")
print(f"Severity distribution: {stats['severity_distribution']}")
```

### Configuration Management

```python
# Export configuration
config = alert_system.export_configuration()
with open('alert_config.json', 'w') as f:
    json.dump(config, f, indent=2)

# Import configuration
with open('alert_config.json', 'r') as f:
    config = json.load(f)
alert_system.import_configuration(config)
```

## Performance Metrics

The alert system expects these performance metrics:

| Metric | Description | Example Value |
|--------|-------------|---------------|
| `max_drawdown_pct` | Maximum drawdown percentage | 15.5 |
| `daily_pnl` | Daily profit/loss percentage | 2.3 |
| `max_consecutive_losses` | Maximum consecutive losses | 3 |
| `sharpe_ratio` | Sharpe ratio | 1.2 |
| `api_errors` | Number of API errors | 2 |
| `volatility` | Volatility percentage | 25.0 |

## Alert Channels

### Telegram
- **Format**: Text message with emojis
- **Rate Limiting**: Handled by notification manager
- **Best For**: Real-time alerts, mobile notifications

### Email
- **Format**: HTML email with subject and body
- **Rate Limiting**: Configurable per email provider
- **Best For**: Detailed reports, non-urgent alerts

### SMS
- **Format**: Short text message
- **Rate Limiting**: Carrier-dependent
- **Best For**: Critical alerts, emergency notifications

### Webhook
- **Format**: JSON payload
- **Rate Limiting**: Configurable
- **Best For**: Integration with external systems

## Alert Severity Levels

### Info
- **Color**: Blue
- **Icon**: ℹ️
- **Use Case**: Informational updates, positive events
- **Example**: Profit targets hit, successful trades

### Warning
- **Color**: Yellow
- **Icon**: ⚠️
- **Use Case**: Attention needed, potential issues
- **Example**: Sharpe ratio drop, consecutive losses

### High
- **Color**: Orange
- **Icon**: 🚨
- **Use Case**: Important issues requiring action
- **Example**: High drawdown, significant losses

### Critical
- **Color**: Red
- **Icon**: 🚨
- **Use Case**: Immediate attention required
- **Example**: API failures, system errors

## Cooldown Management

Cooldowns prevent alert spam by limiting how frequently alerts can be triggered:

```python
# Examples
"5m"   # 5 minutes
"1h"   # 1 hour
"4h"   # 4 hours
"1d"   # 1 day
```

## Alert Aggregation

The system automatically aggregates similar alerts:

- **Grouping**: Alerts with same rule and severity
- **Window**: 5-minute aggregation window
- **Threshold**: 3 alerts before aggregation
- **Message**: "Multiple {rule_name} alerts (X occurrences)"

## Best Practices

### 1. Rule Design
- **Be Specific**: Use precise conditions
- **Set Appropriate Cooldowns**: Balance responsiveness with noise
- **Use Descriptive Names**: Make rules easy to identify
- **Include Templates**: Provide meaningful alert messages

### 2. Channel Selection
- **Critical Alerts**: Use multiple channels (Telegram + Email)
- **Info Alerts**: Use single channel (Telegram only)
- **Consider Timing**: Email for detailed reports, Telegram for real-time

### 3. Performance Monitoring
- **Regular Updates**: Update metrics frequently
- **Comprehensive Coverage**: Include all relevant metrics
- **Error Handling**: Handle evaluation errors gracefully

### 4. Maintenance
- **Regular Cleanup**: Clear old alerts periodically
- **Monitor Statistics**: Track alert frequency and effectiveness
- **Review Rules**: Periodically review and adjust alert rules

## Troubleshooting

### Common Issues

1. **Alerts Not Triggering**
   - Check if rule is enabled
   - Verify condition syntax
   - Ensure metrics are being updated
   - Check cooldown periods

2. **Too Many Alerts**
   - Increase cooldown periods
   - Adjust aggregation thresholds
   - Review condition logic
   - Use more specific conditions

3. **Missing Notifications**
   - Check notification manager configuration
   - Verify channel settings
   - Review rate limiting
   - Check network connectivity

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.getLogger('src.notification.alert_system').setLevel(logging.DEBUG)
```

## Integration with Trading Bots

### Live Trading Bot Integration

```python
# In your trading bot main loop
async def trading_cycle():
    # ... trading logic ...
    
    # Update performance metrics
    metrics = calculate_performance_metrics()
    alert_system.update_performance_metrics(metrics)
    
    # Evaluate alerts
    await alert_system.evaluate_alerts()
    
    # Check for active alerts
    active_alerts = alert_system.get_active_alerts()
    if active_alerts:
        logger.warning(f"Active alerts: {len(active_alerts)}")
```

### Analytics Pipeline Integration

```python
# After analytics calculations
def process_analytics_results(analytics):
    # Extract metrics
    metrics = {
        "max_drawdown_pct": analytics.metrics.max_drawdown_pct,
        "daily_pnl": analytics.metrics.total_return_pct,
        "max_consecutive_losses": analytics.metrics.max_consecutive_losses,
        "sharpe_ratio": analytics.metrics.sharpe_ratio,
    }
    
    # Update alert system
    alert_system.update_performance_metrics(metrics)
    
    # Evaluate alerts
    asyncio.run(alert_system.evaluate_alerts())
```

## Future Enhancements

### Planned Features

1. **Machine Learning Integration**
   - Predictive alerting
   - Anomaly detection
   - Alert effectiveness optimization

2. **Advanced Escalation**
   - Multi-level escalation rules
   - Escalation timeouts
   - Escalation acknowledgments

3. **Alert Templates**
   - Rich message templates
   - Dynamic content generation
   - Multi-language support

4. **Alert Snoozing**
   - Temporary alert suppression
   - Snooze scheduling
   - Snooze history

5. **Alert Analytics**
   - Alert effectiveness metrics
   - Response time analysis
   - Alert pattern recognition

---

*Last Updated: December 2024*
*Version: 1.0.0*
