"""
Consolidated Database Models
---------------------------

Enhanced database models for the consolidated database system
combining web UI and Telegram functionality with proper table prefixing.
"""

from .consolidated_models import (
    Base,
    User,
    WebUIAuditLog,
    WebUIStrategyTemplate,
    WebUISystemConfig,
    WebUIPerformanceSnapshot,
    TelegramAlert,
    TelegramSchedule,
    TelegramFeedback
)

__all__ = [
    # Base and core models
    'Base',
    'User',

    # Web UI models (webui_ prefix)
    'WebUIAuditLog',
    'WebUIStrategyTemplate',
    'WebUISystemConfig',
    'WebUIPerformanceSnapshot',

    # Telegram models (telegram_ prefix)
    'TelegramAlert',
    'TelegramSchedule',
    'TelegramFeedback'
]