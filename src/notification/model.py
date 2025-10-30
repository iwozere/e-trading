"""
Notification Model Classes
-------------------------

Data models and enums for the notification system.
"""

from enum import Enum


class NotificationType(Enum):
    """Notification types for categorization."""
    TRADE_ENTRY = "trade_entry"
    TRADE_EXIT = "trade_exit"
    ERROR = "error"
    STATUS = "status"
    ALERT = "alert"
    REPORT = "report"
    SYSTEM = "system"
    INFO = "info"


class NotificationPriority(Enum):
    """Notification priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"
    URGENT = "urgent"


# Backward compatibility aliases
MessageType = NotificationType
MessagePriority = NotificationPriority