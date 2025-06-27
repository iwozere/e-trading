"""
Smart Alert System
=================

Advanced alert system with:
- Smart alert rules engine
- Alert aggregation and filtering
- Escalation system
- Performance-based alerts
- Alert history and management
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import re
from collections import defaultdict

from .async_notification_manager import AsyncNotificationManager


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert channels"""
    TELEGRAM = "telegram"
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"


@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    condition: str  # Python expression or function name
    severity: AlertSeverity
    channels: List[AlertChannel]
    cooldown: str  # e.g., "5m", "1h", "1d"
    enabled: bool = True
    description: str = ""
    template: str = ""
    
    def __post_init__(self):
        """Convert cooldown string to timedelta"""
        if isinstance(self.cooldown, str):
            self.cooldown = self._parse_cooldown(self.cooldown)
    
    def _parse_cooldown(self, cooldown_str: str) -> timedelta:
        """Parse cooldown string to timedelta"""
        match = re.match(r'(\d+)([mhd])', cooldown_str.lower())
        if not match:
            return timedelta(minutes=5)  # Default 5 minutes
        
        value, unit = match.groups()
        value = int(value)
        
        if unit == 'm':
            return timedelta(minutes=value)
        elif unit == 'h':
            return timedelta(hours=value)
        elif unit == 'd':
            return timedelta(days=value)
        
        return timedelta(minutes=5)


@dataclass
class Alert:
    """Alert instance"""
    rule_name: str
    severity: AlertSeverity
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    escalated_level: int = 0
    
    @property
    def age(self) -> timedelta:
        """Calculate alert age"""
        return datetime.now() - self.timestamp


@dataclass
class EscalationRule:
    """Escalation rule configuration"""
    name: str
    levels: List[Dict[str, Any]] = field(default_factory=list)
    enabled: bool = True
    
    def add_level(self, delay: str, channels: List[AlertChannel], message_template: str = ""):
        """Add escalation level"""
        self.levels.append({
            "delay": self._parse_cooldown(delay),
            "channels": channels,
            "message_template": message_template
        })


class AlertAggregator:
    """Aggregates similar alerts to reduce noise"""
    
    def __init__(self, aggregation_window: timedelta = timedelta(minutes=5)):
        self.aggregation_window = aggregation_window
        self.alert_groups: Dict[str, List[Alert]] = defaultdict(list)
    
    def add_alert(self, alert: Alert) -> Optional[Alert]:
        """Add alert and return aggregated alert if group is ready"""
        group_key = self._get_group_key(alert)
        
        # Remove old alerts from group
        self.alert_groups[group_key] = [
            a for a in self.alert_groups[group_key]
            if a.age < self.aggregation_window
        ]
        
        # Add new alert
        self.alert_groups[group_key].append(alert)
        
        # Check if group should be aggregated
        if len(self.alert_groups[group_key]) >= 3:  # Threshold for aggregation
            return self._create_aggregated_alert(group_key)
        
        return None
    
    def _get_group_key(self, alert: Alert) -> str:
        """Generate group key for alert aggregation"""
        return f"{alert.rule_name}_{alert.severity.value}"
    
    def _create_aggregated_alert(self, group_key: str) -> Alert:
        """Create aggregated alert from group"""
        alerts = self.alert_groups[group_key]
        rule_name, severity_str = group_key.split('_', 1)
        
        # Create aggregated message
        if len(alerts) == 1:
            message = alerts[0].message
        else:
            message = f"Multiple {rule_name} alerts ({len(alerts)} occurrences)"
        
        # Use highest severity
        severity = max(alerts, key=lambda a: a.severity.value).severity
        
        return Alert(
            rule_name=f"{rule_name}_aggregated",
            severity=severity,
            message=message,
            data={"original_alerts": len(alerts)},
            timestamp=alerts[0].timestamp
        )


class SmartAlertSystem:
    """
    Smart alert system with rules engine, aggregation, and escalation
    """
    
    def __init__(self, notification_manager: AsyncNotificationManager):
        self.notification_manager = notification_manager
        self.logger = logging.getLogger(__name__)
        
        # Alert rules and instances
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Aggregation and escalation
        self.aggregator = AlertAggregator()
        self.escalation_rules: Dict[str, EscalationRule] = {}
        
        # Cooldown tracking
        self.last_alert_times: Dict[str, datetime] = {}
        
        # Performance metrics for alert conditions
        self.performance_metrics: Dict[str, Any] = {}
        
        # Initialize default rules
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default alert rules"""
        default_rules = [
            AlertRule(
                name="drawdown_alert",
                condition="max_drawdown > 15",
                severity=AlertSeverity.HIGH,
                channels=[AlertChannel.TELEGRAM, AlertChannel.EMAIL],
                cooldown="1h",
                description="Alert when max drawdown exceeds 15%",
                template="🚨 High Drawdown Alert: {max_drawdown:.2f}%"
            ),
            AlertRule(
                name="profit_target",
                condition="daily_pnl > 5",
                severity=AlertSeverity.INFO,
                channels=[AlertChannel.TELEGRAM],
                cooldown="4h",
                description="Alert when daily PnL exceeds 5%",
                template="💰 Profit Target Hit: {daily_pnl:.2f}%"
            ),
            AlertRule(
                name="consecutive_losses",
                condition="consecutive_losses >= 3",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.TELEGRAM, AlertChannel.EMAIL],
                cooldown="30m",
                description="Alert when 3+ consecutive losses",
                template="⚠️ Consecutive Losses: {consecutive_losses} losses in a row"
            ),
            AlertRule(
                name="sharpe_ratio_drop",
                condition="sharpe_ratio < 1.0",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.TELEGRAM],
                cooldown="2h",
                description="Alert when Sharpe ratio drops below 1.0",
                template="📉 Sharpe Ratio Alert: {sharpe_ratio:.2f} (target: 1.0)"
            ),
            AlertRule(
                name="api_error",
                condition="api_errors > 5",
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.TELEGRAM, AlertChannel.EMAIL],
                cooldown="10m",
                description="Alert when API errors exceed threshold",
                template="🚨 API Error Alert: {api_errors} errors in last hour"
            )
        ]
        
        for rule in default_rules:
            self.add_alert_rule(rule)
    
    def add_alert_rule(self, rule: AlertRule):
        """Add a new alert rule"""
        self.alert_rules[rule.name] = rule
        self.logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule"""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            self.logger.info(f"Removed alert rule: {rule_name}")
    
    def update_performance_metrics(self, metrics: Dict[str, Any]):
        """Update performance metrics for alert evaluation"""
        self.performance_metrics.update(metrics)
        self.logger.debug(f"Updated performance metrics: {list(metrics.keys())}")
    
    async def evaluate_alerts(self):
        """Evaluate all alert rules against current metrics"""
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            # Check cooldown
            if not self._check_cooldown(rule_name, rule.cooldown):
                continue
            
            # Evaluate condition
            if self._evaluate_condition(rule.condition):
                await self._trigger_alert(rule)
    
    def _check_cooldown(self, rule_name: str, cooldown: timedelta) -> bool:
        """Check if enough time has passed since last alert"""
        last_time = self.last_alert_times.get(rule_name)
        if last_time is None:
            return True
        
        return datetime.now() - last_time >= cooldown
    
    def _evaluate_condition(self, condition: str) -> bool:
        """Evaluate alert condition against current metrics"""
        try:
            # Create safe evaluation context
            context = {
                **self.performance_metrics,
                'max_drawdown': self.performance_metrics.get('max_drawdown_pct', 0),
                'daily_pnl': self.performance_metrics.get('daily_pnl', 0),
                'consecutive_losses': self.performance_metrics.get('max_consecutive_losses', 0),
                'sharpe_ratio': self.performance_metrics.get('sharpe_ratio', 0),
                'api_errors': self.performance_metrics.get('api_errors', 0),
            }
            
            # Evaluate condition
            return eval(condition, {"__builtins__": {}}, context)
        
        except Exception as e:
            self.logger.error(f"Error evaluating condition '{condition}': {e}")
            return False
    
    async def _trigger_alert(self, rule: AlertRule):
        """Trigger an alert"""
        # Create alert message
        message = self._format_alert_message(rule)
        
        # Create alert instance
        alert = Alert(
            rule_name=rule.name,
            severity=rule.severity,
            message=message,
            data=self.performance_metrics.copy()
        )
        
        # Check for aggregation
        aggregated_alert = self.aggregator.add_alert(alert)
        if aggregated_alert:
            alert = aggregated_alert
        
        # Store alert
        self.active_alerts[rule.name] = alert
        self.alert_history.append(alert)
        self.last_alert_times[rule.name] = datetime.now()
        
        # Send notifications
        await self._send_alert_notifications(alert, rule)
        
        self.logger.info(f"Triggered alert: {rule.name} - {message}")
    
    def _format_alert_message(self, rule: AlertRule) -> str:
        """Format alert message using template"""
        if not rule.template:
            return f"Alert: {rule.name}"
        
        try:
            return rule.template.format(**self.performance_metrics)
        except Exception as e:
            self.logger.error(f"Error formatting alert template: {e}")
            return f"Alert: {rule.name}"
    
    async def _send_alert_notifications(self, alert: Alert, rule: AlertRule):
        """Send alert notifications to configured channels"""
        for channel in rule.channels:
            try:
                if channel == AlertChannel.TELEGRAM:
                    await self.notification_manager.send_telegram_message(
                        alert.message,
                        notification_type="alert"
                    )
                elif channel == AlertChannel.EMAIL:
                    await self.notification_manager.send_email(
                        subject=f"Trading Alert: {alert.rule_name}",
                        body=alert.message,
                        notification_type="alert"
                    )
                elif channel == AlertChannel.WEBHOOK:
                    await self.notification_manager.send_webhook(
                        url=self.performance_metrics.get('webhook_url'),
                        data={
                            "alert": alert.rule_name,
                            "severity": alert.severity.value,
                            "message": alert.message,
                            "timestamp": alert.timestamp.isoformat()
                        }
                    )
            
            except Exception as e:
                self.logger.error(f"Error sending alert to {channel.value}: {e}")
    
    def acknowledge_alert(self, rule_name: str):
        """Acknowledge an alert"""
        if rule_name in self.active_alerts:
            self.active_alerts[rule_name].acknowledged = True
            self.logger.info(f"Acknowledged alert: {rule_name}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.alert_history
            if alert.timestamp >= cutoff_time
        ]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        total_alerts = len(self.alert_history)
        active_alerts = len(self.active_alerts)
        
        severity_counts = defaultdict(int)
        for alert in self.alert_history:
            severity_counts[alert.severity.value] += 1
        
        return {
            "total_alerts": total_alerts,
            "active_alerts": active_alerts,
            "severity_distribution": dict(severity_counts),
            "most_frequent_rule": self._get_most_frequent_rule()
        }
    
    def _get_most_frequent_rule(self) -> Optional[str]:
        """Get the most frequently triggered alert rule"""
        rule_counts = defaultdict(int)
        for alert in self.alert_history:
            rule_counts[alert.rule_name] += 1
        
        if rule_counts:
            return max(rule_counts.items(), key=lambda x: x[1])[0]
        return None
    
    def clear_old_alerts(self, days: int = 7):
        """Clear alerts older than specified days"""
        cutoff_time = datetime.now() - timedelta(days=days)
        self.alert_history = [
            alert for alert in self.alert_history
            if alert.timestamp >= cutoff_time
        ]
        self.logger.info(f"Cleared alerts older than {days} days")
    
    def export_configuration(self) -> Dict[str, Any]:
        """Export alert system configuration"""
        return {
            "alert_rules": {
                name: {
                    "name": rule.name,
                    "condition": rule.condition,
                    "severity": rule.severity.value,
                    "channels": [c.value for c in rule.channels],
                    "cooldown": str(rule.cooldown),
                    "enabled": rule.enabled,
                    "description": rule.description,
                    "template": rule.template
                }
                for name, rule in self.alert_rules.items()
            },
            "escalation_rules": {
                name: {
                    "name": rule.name,
                    "levels": rule.levels,
                    "enabled": rule.enabled
                }
                for name, rule in self.escalation_rules.items()
            }
        }
    
    def import_configuration(self, config: Dict[str, Any]):
        """Import alert system configuration"""
        # Clear existing rules
        self.alert_rules.clear()
        self.escalation_rules.clear()
        
        # Import alert rules
        for name, rule_config in config.get("alert_rules", {}).items():
            rule = AlertRule(
                name=rule_config["name"],
                condition=rule_config["condition"],
                severity=AlertSeverity(rule_config["severity"]),
                channels=[AlertChannel(c) for c in rule_config["channels"]],
                cooldown=rule_config["cooldown"],
                enabled=rule_config["enabled"],
                description=rule_config["description"],
                template=rule_config["template"]
            )
            self.alert_rules[name] = rule
        
        # Import escalation rules
        for name, esc_config in config.get("escalation_rules", {}).items():
            rule = EscalationRule(
                name=esc_config["name"],
                levels=esc_config["levels"],
                enabled=esc_config["enabled"]
            )
            self.escalation_rules[name] = rule
        
        self.logger.info("Imported alert system configuration")
