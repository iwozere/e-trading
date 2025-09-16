"""
Telegram Database Service
------------------------

Clean, unified interface for all telegram bot database operations.
Uses the repository pattern with proper session management.
"""

from typing import Optional, List, Dict, Any
from src.data.database_service import get_database_service
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def init_db():
    """Initialize telegram database."""
    from src.data.database_service import init_databases
    init_databases()


# --- USER MANAGEMENT ---
def set_user_email(telegram_user_id: str, email: str, code: str, sent_time: int,
                   language: Optional[str] = None, is_admin: Optional[bool] = None):
    """Set or update user's email and verification code."""
    db_service = get_database_service()
    with db_service.get_telegram_repo() as repo:
        fields = {
            'email': email,
            'verification_code': code,
            'code_sent_time': sent_time,
            'max_alerts': 5,
            'max_schedules': 5
        }
        if language is not None:
            fields['language'] = language
        if is_admin is not None:
            fields['is_admin'] = bool(is_admin)

        repo.upsert_user(telegram_user_id, **fields)
        repo.set_verification_code(telegram_user_id, code, sent_time)


def get_user_status(telegram_user_id: str) -> Optional[Dict[str, Any]]:
    """Get user's status including email, verification, and approval."""
    db_service = get_database_service()
    with db_service.get_telegram_repo() as repo:
        return repo.get_user_status(telegram_user_id)


def verify_code(telegram_user_id: str, code: str, expiry_seconds: int = 3600) -> bool:
    """Verify user's email verification code."""
    import time
    db_service = get_database_service()
    with db_service.get_telegram_repo() as repo:
        user = repo.get_user(telegram_user_id)
        if not user or not user.verification_code or user.code_sent_time is None:
            return False

        if (user.verification_code == code and
            (time.time() - int(user.code_sent_time)) <= expiry_seconds):
            repo.upsert_user(telegram_user_id, verified=True)
            return True
        return False


def approve_user(telegram_user_id: str) -> bool:
    """Approve user for access."""
    db_service = get_database_service()
    with db_service.get_telegram_repo() as repo:
        return repo.approve_user(telegram_user_id, True)


def reject_user(telegram_user_id: str) -> bool:
    """Reject user's approval request."""
    db_service = get_database_service()
    with db_service.get_telegram_repo() as repo:
        return repo.approve_user(telegram_user_id, False)


def get_pending_approvals() -> List[Dict[str, Any]]:
    """Get users pending approval."""
    db_service = get_database_service()
    with db_service.get_telegram_repo() as repo:
        return repo.list_pending_approvals()


def update_user_email(telegram_user_id: str, email: str):
    """Update user's email address."""
    db_service = get_database_service()
    with db_service.get_telegram_repo() as repo:
        repo.upsert_user(telegram_user_id, email=email)


def update_user_verification(telegram_user_id: str, verified: bool):
    """Update user's verification status."""
    db_service = get_database_service()
    with db_service.get_telegram_repo() as repo:
        repo.upsert_user(telegram_user_id, verified=bool(verified))


def list_users() -> List[Dict[str, Any]]:
    """List all users."""
    db_service = get_database_service()
    with db_service.get_telegram_repo() as repo:
        users = repo.list_users()
        return [
            {
                'telegram_user_id': u.telegram_user_id,
                'email': u.email,
                'verified': bool(u.verified),
                'approved': bool(u.approved),
                'language': u.language,
                'is_admin': bool(u.is_admin),
                'max_alerts': u.max_alerts,
                'max_schedules': u.max_schedules,
            }
            for u in users
        ]


def get_admin_user_ids() -> List[str]:
    """Get all admin user IDs."""
    db_service = get_database_service()
    with db_service.get_telegram_repo() as repo:
        return repo.get_admin_user_ids()


# --- ALERTS ---
def add_alert(user_id: str, ticker: str, price: float, condition: str, email: bool = False) -> int:
    """Add price alert."""
    db_service = get_database_service()
    with db_service.get_telegram_repo() as repo:
        return repo.add_alert(user_id, ticker, price, condition, email)


def add_indicator_alert(user_id: str, ticker: str, config_json: str,
                       alert_action: str = "notify", timeframe: str = "15m",
                       email: bool = False) -> int:
    """Add indicator-based alert."""
    db_service = get_database_service()
    with db_service.get_telegram_repo() as repo:
        return repo.add_indicator_alert(user_id, ticker, config_json, alert_action, timeframe, email)


def get_alert(alert_id: int) -> Optional[Dict[str, Any]]:
    """Get alert by ID."""
    db_service = get_database_service()
    with db_service.get_telegram_repo() as repo:
        alert = repo.get_alert(alert_id)
        if not alert:
            return None
        return _alert_to_dict(alert)


def list_alerts(user_id: str) -> List[Dict[str, Any]]:
    """List user's alerts."""
    db_service = get_database_service()
    with db_service.get_telegram_repo() as repo:
        alerts = repo.list_alerts(user_id)
        return [_alert_to_dict(alert) for alert in alerts]


def update_alert(alert_id: int, **kwargs) -> bool:
    """Update alert."""
    if not kwargs:
        return False
    db_service = get_database_service()
    with db_service.get_telegram_repo() as repo:
        return repo.update_alert(alert_id, **kwargs)


def delete_alert(alert_id: int) -> bool:
    """Delete alert."""
    db_service = get_database_service()
    with db_service.get_telegram_repo() as repo:
        return repo.delete_alert(alert_id)


def get_active_alerts() -> List[Dict[str, Any]]:
    """Get all active alerts."""
    db_service = get_database_service()
    with db_service.get_telegram_repo() as repo:
        alerts = repo.get_active_alerts()
        return [_alert_to_dict(alert) for alert in alerts]


def get_alerts_by_type(alert_type: str = None) -> List[Dict[str, Any]]:
    """Get alerts by type."""
    db_service = get_database_service()
    with db_service.get_telegram_repo() as repo:
        alerts = repo.get_alerts_by_type(alert_type)
        return [_alert_to_dict(alert) for alert in alerts]


# --- SCHEDULES ---
def add_schedule(user_id: str, ticker: str, scheduled_time: str, period: str = None,
                email: int = 0, indicators: str = None, interval: str = None,
                provider: str = None) -> int:
    """Add schedule."""
    db_service = get_database_service()
    with db_service.get_telegram_repo() as repo:
        return repo.add_schedule(user_id, ticker, scheduled_time, period,
                               bool(email), indicators, interval, provider)


def create_schedule(schedule_data: Dict[str, Any]) -> int:
    """Create schedule from data dictionary."""
    db_service = get_database_service()
    with db_service.get_telegram_repo() as repo:
        return repo.create_schedule(schedule_data)


def add_json_schedule(user_id: str, config_json: str, schedule_config: str = "advanced") -> int:
    """Add JSON-based schedule."""
    db_service = get_database_service()
    with db_service.get_telegram_repo() as repo:
        return repo.add_json_schedule(user_id, config_json, schedule_config)


def get_schedule(schedule_id: int) -> Optional[Dict[str, Any]]:
    """Get schedule by ID."""
    db_service = get_database_service()
    with db_service.get_telegram_repo() as repo:
        schedule = repo.get_schedule(schedule_id)
        if not schedule:
            return None
        return _schedule_to_dict(schedule)


def get_schedule_by_id(schedule_id: int) -> Optional[Dict[str, Any]]:
    """Alias for get_schedule."""
    return get_schedule(schedule_id)


def list_schedules(user_id: str) -> List[Dict[str, Any]]:
    """List user's schedules."""
    db_service = get_database_service()
    with db_service.get_telegram_repo() as repo:
        schedules = repo.list_schedules(user_id)
        return [_schedule_to_dict(schedule) for schedule in schedules]


def update_schedule(schedule_id: int, **kwargs) -> bool:
    """Update schedule."""
    if not kwargs:
        return False
    db_service = get_database_service()
    with db_service.get_telegram_repo() as repo:
        return repo.update_schedule(schedule_id, **kwargs)


def delete_schedule(schedule_id: int) -> bool:
    """Delete schedule."""
    db_service = get_database_service()
    with db_service.get_telegram_repo() as repo:
        return repo.delete_schedule(schedule_id)


def get_active_schedules() -> List[Dict[str, Any]]:
    """Get all active schedules."""
    db_service = get_database_service()
    with db_service.get_telegram_repo() as repo:
        schedules = repo.get_active_schedules()
        return [_schedule_to_dict(schedule) for schedule in schedules]


def get_schedules_by_config(schedule_config: str = None) -> List[Dict[str, Any]]:
    """Get schedules by configuration type."""
    db_service = get_database_service()
    with db_service.get_telegram_repo() as repo:
        schedules = repo.get_schedules_by_config(schedule_config)
        return [_schedule_to_dict(schedule) for schedule in schedules]


# --- SETTINGS ---
def set_setting(key: str, value: str):
    """Set global setting."""
    db_service = get_database_service()
    with db_service.get_telegram_repo() as repo:
        repo.set_setting(key, value)


def get_setting(key: str) -> Optional[str]:
    """Get global setting."""
    db_service = get_database_service()
    with db_service.get_telegram_repo() as repo:
        return repo.get_setting(key)


def set_global_setting(key: str, value: str):
    """Alias for set_setting."""
    set_setting(key, value)


# --- USER LIMITS ---
def set_user_limit(telegram_user_id: str, limit_type: str, value: int):
    """Set user limit."""
    db_service = get_database_service()
    with db_service.get_telegram_repo() as repo:
        repo.set_user_limit(telegram_user_id, limit_type, value)


def get_user_limit(telegram_user_id: str, limit_type: str) -> Optional[int]:
    """Get user limit."""
    db_service = get_database_service()
    with db_service.get_telegram_repo() as repo:
        user = repo.get_user(telegram_user_id)
        if not user:
            return None
        return getattr(user, limit_type, None)


def set_user_max_alerts(telegram_user_id: str, max_alerts: int):
    """Set user's maximum alerts."""
    db_service = get_database_service()
    with db_service.get_telegram_repo() as repo:
        repo.set_user_limit(telegram_user_id, 'max_alerts', max_alerts)


def set_user_max_schedules(telegram_user_id: str, max_schedules: int):
    """Set user's maximum schedules."""
    db_service = get_database_service()
    with db_service.get_telegram_repo() as repo:
        repo.set_user_limit(telegram_user_id, 'max_schedules', max_schedules)


# --- FEEDBACK ---
def add_feedback(user_id: str, feedback_type: str, message: str) -> int:
    """Add user feedback."""
    db_service = get_database_service()
    with db_service.get_telegram_repo() as repo:
        return repo.add_feedback(user_id, feedback_type, message)


def list_feedback(feedback_type: str = None) -> List[Dict[str, Any]]:
    """List feedback entries."""
    db_service = get_database_service()
    with db_service.get_telegram_repo() as repo:
        feedback_entries = repo.list_feedback(feedback_type)
        return [
            {
                'id': f.id,
                'user_id': f.user_id,
                'type': f.type,
                'message': f.message,
                'created': f.created,
                'status': f.status
            }
            for f in feedback_entries
        ]


def update_feedback_status(feedback_id: int, status: str) -> bool:
    """Update feedback status."""
    db_service = get_database_service()
    with db_service.get_telegram_repo() as repo:
        return repo.update_feedback_status(feedback_id, status)


# --- COMMAND AUDIT ---
def log_command_audit(telegram_user_id: str, command: str, full_message: str = None,
                     is_registered_user: bool = False, user_email: str = None,
                     success: bool = True, error_message: str = None,
                     response_time_ms: int = None) -> int:
    """Log command audit."""
    db_service = get_database_service()
    with db_service.get_telegram_repo() as repo:
        return repo.log_command_audit(telegram_user_id, command, full_message,
                                    is_registered_user, user_email, success,
                                    error_message, response_time_ms)


def get_user_command_history(telegram_user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Get user's command history."""
    db_service = get_database_service()
    with db_service.get_telegram_repo() as repo:
        history = repo.get_user_command_history(telegram_user_id, limit)
        return [
            {
                'id': r.id,
                'command': r.command,
                'full_message': r.full_message,
                'is_registered_user': bool(r.is_registered_user),
                'user_email': r.user_email,
                'success': bool(r.success),
                'error_message': r.error_message,
                'response_time_ms': r.response_time_ms,
                'created': r.created
            }
            for r in history
        ]


def get_all_command_audit(limit: int = 100, offset: int = 0,
                         user_id: str = None, command: str = None,
                         success_only: bool = None,
                         start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
    """Get command audit with filters."""
    db_service = get_database_service()
    with db_service.get_telegram_repo() as repo:
        audit_entries = repo.get_all_command_audit(limit, offset, user_id, command,
                                                 success_only, start_date, end_date)
        return [
            {
                'id': r.id,
                'telegram_user_id': r.telegram_user_id,
                'command': r.command,
                'full_message': r.full_message,
                'is_registered_user': bool(r.is_registered_user),
                'user_email': r.user_email,
                'success': bool(r.success),
                'error_message': r.error_message,
                'response_time_ms': r.response_time_ms,
                'created': r.created
            }
            for r in audit_entries
        ]


def get_command_audit_stats() -> Dict[str, Any]:
    """Get command audit statistics."""
    db_service = get_database_service()
    with db_service.get_telegram_repo() as repo:
        return repo.get_command_audit_stats()


def get_unique_users_command_history() -> List[Dict[str, Any]]:
    """Get unique users command history summary."""
    db_service = get_database_service()
    with db_service.get_telegram_repo() as repo:
        return repo.get_unique_users_command_history()


# --- VERIFICATION CODES ---
def get_verification_code(telegram_user_id: str) -> tuple[Optional[str], Optional[int]]:
    """Get user's verification code and sent time."""
    db_service = get_database_service()
    with db_service.get_telegram_repo() as repo:
        user = repo.get_user(telegram_user_id)
        if user:
            return (user.verification_code, user.code_sent_time)
        return (None, None)


def count_codes_last_hour(telegram_user_id: str) -> int:
    """Count verification codes sent in last hour for rate limiting."""
    db_service = get_database_service()
    with db_service.get_telegram_repo() as repo:
        return repo.count_codes_last_hour(telegram_user_id)


# --- HELPER FUNCTIONS ---
def _alert_to_dict(alert) -> Dict[str, Any]:
    """Convert alert model to dictionary."""
    return {
        'id': alert.id,
        'ticker': alert.ticker,
        'user_id': alert.user_id,
        'price': float(alert.price) if alert.price is not None else None,
        'condition': alert.condition,
        'active': bool(alert.active),
        'email': bool(alert.email),
        'created': alert.created,
        'alert_type': alert.alert_type,
        'timeframe': alert.timeframe,
        'config_json': alert.config_json,
        'alert_action': alert.alert_action
    }


def _schedule_to_dict(schedule) -> Dict[str, Any]:
    """Convert schedule model to dictionary."""
    return {
        'id': schedule.id,
        'ticker': schedule.ticker,
        'user_id': schedule.user_id,
        'scheduled_time': schedule.scheduled_time,
        'period': schedule.period,
        'active': bool(schedule.active),
        'email': bool(schedule.email),
        'indicators': schedule.indicators,
        'interval': schedule.interval,
        'provider': schedule.provider,
        'created': schedule.created,
        'schedule_type': schedule.schedule_type,
        'list_type': schedule.list_type,
        'config_json': schedule.config_json,
        'schedule_config': schedule.schedule_config
    }