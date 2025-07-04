"""
Notification and alerting models.

Includes:
- Notification, alert, and escalation rule dataclasses
- Notification types, priorities, alert severity, and channels enums
"""
import re
from enum import Enum
from typing import Any, Dict, List
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

class NotificationType(Enum):
    """Types of notifications"""
    TRADE_ENTRY = "trade_entry"
    TRADE_EXIT = "trade_exit"
    TRADE_UPDATE = "trade_update"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    SYSTEM = "system"
    PERFORMANCE = "performance"


class NotificationPriority(Enum):
    """Notification priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

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
class Notification:
    """Notification data structure"""
    type: NotificationType
    priority: NotificationPriority
    title: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "trading_bot"
    retry_count: int = 0
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert notification to dictionary"""
        return {
            "type": self.type.value,
            "priority": self.priority.value,
            "title": self.title,
            "message": self.message,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries
        }

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
