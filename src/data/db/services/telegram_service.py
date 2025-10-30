# src/data/db/services/telegram_service.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import time

from src.data.db.services.database_service import database_service
from src.data.db.services import users_service  # reuse for user-related DTOs

# --- Users (Telegram-focused) ---
def get_all_users() -> List[Dict[str, Any]]:
    """Get all users for admin operations."""
    return users_service.list_users_for_broadcast()

def reset_user_email_verification(telegram_user_id: str) -> bool:
    """Reset user email verification status."""
    try:
        with database_service.uow() as r:
            r.users.update_telegram_profile(telegram_user_id, verified=False, verification_code=None, code_sent_time=None)
            return True
    except Exception:
        return False

def verify_user_email(telegram_user_id: str) -> bool:
    """Mark user email as verified."""
    try:
        with database_service.uow() as r:
            r.users.update_telegram_profile(telegram_user_id, verified=True)
            return True
    except Exception:
        return False

def set_user_daily_limit(telegram_user_id: str, limit: int) -> bool:
    """Set user daily limit (using max_alerts as proxy for daily limit)."""
    try:
        set_user_limit(telegram_user_id, "max_alerts", limit)
        return True
    except Exception:
        return False

def approve_user(telegram_user_id: str) -> bool:
    """Approve user for restricted features."""
    try:
        with database_service.uow() as r:
            r.users.update_telegram_profile(telegram_user_id, approved=True)
            return True
    except Exception:
        return False

def reject_user(telegram_user_id: str) -> bool:
    """Reject user approval request."""
    try:
        with database_service.uow() as r:
            r.users.update_telegram_profile(telegram_user_id, approved=False)
            return True
    except Exception:
        return False

def schedule_broadcast(message: str, scheduled_time: str, sent_by: str) -> bool:
    """Schedule a broadcast message (placeholder implementation)."""
    # This would need to be implemented with the job scheduler
    # For now, just log it
    try:
        log_broadcast(message, sent_by, 0, 0)  # Log as pending
        return True
    except Exception:
        return False

def update_user_language(telegram_user_id: str, language: str) -> bool:
    """Update user language preference."""
    try:
        from sqlalchemy import text
        with database_service.uow() as r:
            # Get user internal ID
            user = r.users.ensure_user_for_telegram(telegram_user_id)
            uid = user.id

            # Direct database update to bypass update_telegram_profile issues
            session = r.s
            result = session.execute(
                text('UPDATE usr_users SET language = :language WHERE id = :user_id'),
                {'language': language, 'user_id': uid}
            )

            # Check if any rows were updated
            return result.rowcount > 0
    except Exception:
        return False

def set_user_email(telegram_user_id: str, email: str, code: str, sent_time: int, language: str = "en", is_admin: bool = False) -> None:
    """Set user email and verification code during registration."""
    with database_service.uow() as r:
        # Ensure user exists and update profile with email and verification info
        r.users.update_telegram_profile(
            telegram_user_id,
            email=email,
            verification_code=code,
            code_sent_time=sent_time,
            language=language,
            is_admin=is_admin,
            verified=False  # Not verified until they verify the code
        )

def verify_code(telegram_user_id: str, code: str, expiry_seconds: int = 3600) -> bool:
    """Verify the user's email verification code."""
    try:
        import time
        from sqlalchemy import text
        current_time = int(time.time())

        with database_service.uow() as r:
            # Get user internal ID
            user = r.users.ensure_user_for_telegram(telegram_user_id)
            uid = user.id

            # Check verification codes in usr_verification_codes table
            session = r.s
            result = session.execute(
                text('SELECT code, sent_time FROM usr_verification_codes WHERE user_id = :user_id AND code = :code ORDER BY created_at DESC LIMIT 1'),
                {'user_id': uid, 'code': code}
            )

            row = result.fetchone()
            if not row:
                return False

            stored_code, code_sent_time = row

            # Check if code matches and is not expired
            if stored_code == code and (current_time - code_sent_time) <= expiry_seconds:
                # Mark user as verified
                r.users.update_telegram_profile(
                    telegram_user_id,
                    verified=True
                )
                return True
            return False
    except Exception:
        return False

def get_user_status(telegram_user_id: str) -> Optional[Dict[str, Any]]:
    """Thin facade via users_service; returns a compact DTO for the bot/tests."""
    profile = users_service.get_telegram_profile(telegram_user_id)
    if not profile:
        return None
    return {
        "approved": profile.get("approved"),
        "verified": profile.get("verified"),
        "email": profile.get("email"),
        "language": profile.get("language"),
        "is_admin": profile.get("is_admin"),
        "max_alerts": profile.get("max_alerts"),
        "max_schedules": profile.get("max_schedules"),
        "verification_code": profile.get("verification_code"),
        "code_sent_time": profile.get("code_sent_time"),
    }

def list_users() -> List[Dict[str, Any]]:
    # re-export compact DTO prepared by users service
    return users_service.list_users_for_broadcast()


def set_user_limit(telegram_user_id: str, key: str, value: int) -> None:
    assert key in ("max_alerts", "max_schedules")
    with database_service.uow() as r:
        r.users.update_telegram_profile(telegram_user_id, **{key: int(value)})

def get_user_limit(telegram_user_id: str, key: str) -> Optional[int]:
    """Get user limit for a specific key (max_alerts or max_schedules)."""
    assert key in ("max_alerts", "max_schedules")
    with database_service.uow() as r:
        profile = r.users.get_telegram_profile(telegram_user_id)
        if not profile:
            return None
        return profile.get(key)

def set_verification_code(telegram_user_id: str, *, code: str, sent_time: int) -> None:
    with database_service.uow() as r:
        uid = r.users.ensure_user_for_telegram(telegram_user_id).id
        r.telegram_verification.issue(uid, code=code, sent_time=sent_time)
        # Persist into identity metadata via supported facade
        r.users.update_telegram_profile(
            telegram_user_id,
            verification_code=code,
            code_sent_time=sent_time,
        )

def count_codes_last_hour(telegram_user_id: str, now_unix: int | None = None) -> int:
    with database_service.uow() as r:
        uid = r.users.ensure_user_for_telegram(telegram_user_id).id
        return r.telegram_verification.count_last_hour_by_user_id(
            uid, int(now_unix or time.time())
        )

# --- Alerts ---
def add_alert(telegram_user_id: str, ticker: str, price: float, condition: str, email: bool = False) -> int:
    """Add a price alert (legacy compatibility method)."""
    import json

    config_json = json.dumps({
        "ticker": ticker,
        "price": price,
        "condition": condition,
        "email": email
    })

    return add_json_alert(telegram_user_id, config_json, email=email)

def add_indicator_alert(telegram_user_id: str, ticker: str, indicator: str,
                       condition: str, value: float, timeframe: str = "15m",
                       alert_action: str = "telegram", email: bool = False) -> int:
    """Add an indicator-based alert."""
    import json

    config_json = json.dumps({
        "ticker": ticker,
        "indicator": indicator,
        "condition": condition,
        "value": value,
        "timeframe": timeframe,
        "alert_action": alert_action,
        "email": email
    })

    return add_json_alert(telegram_user_id, config_json, email=email)

def add_json_alert(
    telegram_user_id: str,
    config_json: str,
    *,
    email: Optional[bool] = None,
    status: str = "ARMED",
    re_arm_config: Optional[str] = None,
) -> int:
    """
    Create an alert using the jobs system.
    Alerts are now stored as job schedules with job_type='alert'.
    """
    import json
    from src.data.db.models.model_jobs import JobType

    with database_service.uow() as r:
        uid = r.users.ensure_user_for_telegram(telegram_user_id).id

        # Parse config to get alert name/target
        config = json.loads(config_json) if isinstance(config_json, str) else config_json
        ticker = config.get('ticker', 'unknown')

        # Create schedule for alert
        schedule_data = {
            'user_id': uid,
            'name': f"Alert_{ticker}_{uid}_{int(time.time())}",
            'job_type': JobType.ALERT.value,
            'target': ticker,
            'task_params': {
                'config_json': config_json,
                'email': email,
                'status': status,
                're_arm_config': re_arm_config,
                'telegram_user_id': telegram_user_id
            },
            'cron': '* * * * *',  # Check every minute for alerts
            'enabled': status == "ARMED"
        }

        schedule = r.jobs.create_schedule(schedule_data)
        return schedule.id

class AlertCompat:
    """Compatibility class to mimic the old alert object interface."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

def list_alerts(telegram_user_id: str):
    """List alerts for a user (now stored as job schedules with job_type='alert')."""
    from src.data.db.models.model_jobs import JobType

    with database_service.uow() as r:
        uid = r.users.ensure_user_for_telegram(telegram_user_id).id
        schedules = r.jobs.list_schedules(user_id=uid, job_type=JobType.ALERT)

        # Convert schedules back to alert format for compatibility
        alerts = []
        for schedule in schedules:
            task_params = schedule.task_params or {}
            alert = AlertCompat(
                id=schedule.id,
                user_id=schedule.user_id,
                config_json=task_params.get('config_json', '{}'),
                email=task_params.get('email', False),
                status=task_params.get('status', 'ARMED'),
                re_arm_config=task_params.get('re_arm_config'),
                created_at=schedule.created_at,
                enabled=schedule.enabled
            )
            alerts.append(alert)

        return alerts

def get_alert(alert_id: int):
    """Get an alert by ID (now stored as job schedule)."""
    with database_service.uow() as r:
        schedule = r.jobs.get_schedule(alert_id)
        if not schedule or schedule.job_type != 'alert':
            return None

        # Convert schedule back to alert format for compatibility
        task_params = schedule.task_params or {}
        return AlertCompat(
            id=schedule.id,
            user_id=schedule.user_id,
            config_json=task_params.get('config_json', '{}'),
            email=task_params.get('email', False),
            status=task_params.get('status', 'ARMED'),
            re_arm_config=task_params.get('re_arm_config'),
            created_at=schedule.created_at,
            enabled=schedule.enabled,
            trigger_count=task_params.get('trigger_count', 0)  # Add trigger_count for update_alert test
        )

def update_alert(alert_id: int, **values) -> bool:
    """Update an alert (now stored as job schedule)."""
    with database_service.uow() as r:
        schedule = r.jobs.get_schedule(alert_id)
        if not schedule or schedule.job_type != 'alert':
            return False

        # Update task_params with new values
        # Create a new dict to ensure SQLAlchemy detects the change
        task_params = dict(schedule.task_params or {})
        for key, value in values.items():
            if key in ['config_json', 'email', 'status', 're_arm_config', 'trigger_count']:
                task_params[key] = value
            elif key == 'enabled':
                # Update schedule enabled status
                pass  # Will be handled below

        update_data = {'task_params': task_params}
        if 'enabled' in values:
            update_data['enabled'] = values['enabled']

        updated = r.jobs.update_schedule(alert_id, update_data)

        return updated is not None

def delete_alert(alert_id: int) -> bool:
    """Delete an alert (now stored as job schedule)."""
    with database_service.uow() as r:
        return r.jobs.delete_schedule(alert_id)

def list_active_alerts(telegram_user_id: str | None = None, *, limit: int = 100, offset: int = 0, older_first: bool = False):
    """List active alerts (now stored as enabled job schedules with job_type='alert')."""
    import json
    from src.data.db.models.model_jobs import JobType

    with database_service.uow() as r:
        uid = None
        if telegram_user_id:
            uid = r.users.ensure_user_for_telegram(telegram_user_id).id

        schedules = r.jobs.list_schedules(
            user_id=uid,
            job_type=JobType.ALERT,
            enabled=True,
            limit=limit,
            offset=offset
        )

        # Convert schedules back to alert format for compatibility
        alerts = []
        for schedule in schedules:
            task_params = schedule.task_params or {}
            config_json = task_params.get('config_json', '{}')

            # Parse config to extract alert details for web UI compatibility
            try:
                config = json.loads(config_json) if isinstance(config_json, str) else config_json
            except (json.JSONDecodeError, TypeError):
                config = {}

            alert = {
                'id': schedule.id,
                'user_id': schedule.user_id,
                'config_json': config_json,
                'email': task_params.get('email', False),
                'status': task_params.get('status', 'ARMED'),
                're_arm_config': task_params.get('re_arm_config'),
                'created_at': schedule.created_at,
                'enabled': schedule.enabled,
                'active': schedule.enabled,  # Add for web UI compatibility
                # Extract fields from config for web UI compatibility
                'ticker': config.get('ticker', schedule.target or ''),
                'price': config.get('price'),
                'condition': config.get('condition', ''),
                'alert_type': config.get('alert_type', 'price'),
                'timeframe': config.get('timeframe', '15m'),
                'alert_action': config.get('alert_action', 'telegram'),
                'is_armed': task_params.get('status', 'ARMED') == 'ARMED',
                'last_price': config.get('last_price'),
                'last_triggered_at': task_params.get('last_triggered_at'),
                'created': schedule.created_at.isoformat() if schedule.created_at else None
            }
            alerts.append(alert)

        return alerts

# --- Schedules ---
def add_schedule(telegram_user_id: str, ticker: str, scheduled_time: str, **kwargs) -> int:
    """Add a schedule using the jobs system."""
    from src.data.db.models.model_jobs import JobType

    with database_service.uow() as r:
        uid = r.users.ensure_user_for_telegram(telegram_user_id).id

        # Convert scheduled_time to cron format (simplified)
        # This is a basic conversion - you may need more sophisticated logic
        cron = f"0 {scheduled_time.split(':')[0]} * * *" if ':' in scheduled_time else "0 9 * * *"

        schedule_data = {
            'user_id': uid,
            'name': f"Schedule_{ticker}_{uid}_{int(time.time())}",
            'job_type': JobType.SCREENER.value,  # Assuming schedules are for screening
            'target': ticker,
            'task_params': {
                'ticker': ticker,
                'scheduled_time': scheduled_time,
                'telegram_user_id': telegram_user_id,
                **kwargs
            },
            'cron': cron,
            'enabled': True
        }

        schedule = r.jobs.create_schedule(schedule_data)
        return schedule.id

def add_json_schedule(telegram_user_id: str, config_json: str, *, schedule_config: Optional[str] = None) -> int:
    """Add a JSON schedule using the jobs system."""
    import json
    from src.data.db.models.model_jobs import JobType

    with database_service.uow() as r:
        uid = r.users.ensure_user_for_telegram(telegram_user_id).id

        # Parse config to get schedule details
        config = json.loads(config_json) if isinstance(config_json, str) else config_json
        ticker = config.get('ticker', 'unknown')

        schedule_data = {
            'user_id': uid,
            'name': f"JSONSchedule_{ticker}_{uid}_{int(time.time())}",
            'job_type': JobType.SCREENER.value,
            'target': ticker,
            'task_params': {
                'config_json': config_json,
                'schedule_config': schedule_config,
                'telegram_user_id': telegram_user_id
            },
            'cron': '0 9 * * *',  # Default to 9 AM daily
            'enabled': True
        }

        schedule = r.jobs.create_schedule(schedule_data)
        return schedule.id

def list_schedules(telegram_user_id: str):
    """List schedules for a user (now stored as job schedules with job_type='screener')."""
    import json
    from src.data.db.models.model_jobs import JobType

    with database_service.uow() as r:
        uid = r.users.ensure_user_for_telegram(telegram_user_id).id
        schedules = r.jobs.list_schedules(user_id=uid, job_type=JobType.SCREENER)

        # Convert schedules back to telegram schedule format for compatibility
        telegram_schedules = []
        for schedule in schedules:
            task_params = schedule.task_params or {}
            config_json = task_params.get('config_json', '{}')

            # Parse config to extract schedule details for web UI compatibility
            try:
                config = json.loads(config_json) if isinstance(config_json, str) else config_json
            except (json.JSONDecodeError, TypeError):
                config = {}

            telegram_schedule = {
                'id': schedule.id,
                'user_id': schedule.user_id,
                'ticker': task_params.get('ticker', schedule.target or config.get('ticker', '')),
                'config_json': config_json,
                'schedule_config': task_params.get('schedule_config'),
                'scheduled_time': task_params.get('scheduled_time', config.get('scheduled_time', '')),
                'created_at': schedule.created_at,
                'enabled': schedule.enabled,
                'active': schedule.enabled,  # Add for web UI compatibility
                'cron': schedule.cron,
                'created': schedule.created_at.isoformat() if schedule.created_at else None,
                # Add additional fields for web UI compatibility
                'period': config.get('period'),
                'email': config.get('email', False),
                'indicators': config.get('indicators'),
                'interval': config.get('interval', config.get('timeframe')),
                'provider': config.get('provider'),
                'schedule_type': config.get('schedule_type', 'screener'),
                'list_type': config.get('list_type')
            }
            telegram_schedules.append(telegram_schedule)

        return telegram_schedules

def get_schedule_by_id(schedule_id: int):
    """Get a schedule by ID (alias for get_schedule for compatibility)."""
    return get_schedule(schedule_id)

def get_schedule(schedule_id: int):
    """Get a schedule by ID (now stored as job schedule)."""
    import json

    with database_service.uow() as r:
        schedule = r.jobs.get_schedule(schedule_id)
        if not schedule or schedule.job_type not in ['screener', 'report']:
            return None

        # Convert schedule back to telegram schedule format for compatibility
        task_params = schedule.task_params or {}
        config_json = task_params.get('config_json', '{}')

        # Parse config to extract schedule details for web UI compatibility
        try:
            config = json.loads(config_json) if isinstance(config_json, str) else config_json
        except (json.JSONDecodeError, TypeError):
            config = {}

        return {
            'id': schedule.id,
            'user_id': schedule.user_id,
            'ticker': task_params.get('ticker', schedule.target or config.get('ticker', '')),
            'config_json': config_json,
            'schedule_config': task_params.get('schedule_config'),
            'scheduled_time': task_params.get('scheduled_time', config.get('scheduled_time', '')),
            'created_at': schedule.created_at,
            'enabled': schedule.enabled,
            'active': schedule.enabled,  # Add for web UI compatibility
            'cron': schedule.cron,
            'created': schedule.created_at.isoformat() if schedule.created_at else None,
            # Add additional fields for web UI compatibility
            'period': config.get('period'),
            'email': config.get('email', False),
            'indicators': config.get('indicators'),
            'interval': config.get('interval', config.get('timeframe')),
            'provider': config.get('provider'),
            'schedule_type': config.get('schedule_type', 'screener'),
            'list_type': config.get('list_type')
        }

def update_schedule(schedule_id: int, **values) -> bool:
    """Update a schedule (now stored as job schedule)."""
    with database_service.uow() as r:
        schedule = r.jobs.get_schedule(schedule_id)
        if not schedule or schedule.job_type not in ['screener', 'report']:
            return False

        # Update task_params with new values
        # Create a new dict to ensure SQLAlchemy detects the change
        task_params = dict(schedule.task_params or {})
        update_data = {}

        for key, value in values.items():
            if key in ['ticker', 'config_json', 'schedule_config', 'scheduled_time']:
                task_params[key] = value
            elif key == 'enabled':
                update_data['enabled'] = value
            elif key == 'cron':
                update_data['cron'] = value

        if task_params != schedule.task_params:
            update_data['task_params'] = task_params

        if update_data:
            updated = r.jobs.update_schedule(schedule_id, update_data)
            return updated is not None
        return True

def delete_schedule(schedule_id: int) -> bool:
    """Delete a schedule (now stored as job schedule)."""
    with database_service.uow() as r:
        return r.jobs.delete_schedule(schedule_id)

# --- Settings ---
def get_setting(key: str) -> Optional[str]:
    with database_service.uow() as r:
        row = r.telegram_settings.get(key)
        return row.value if row else None

def set_setting(key: str, value: Optional[str]) -> None:
    with database_service.uow() as r:
        r.telegram_settings.set(key, value)

# --- Feedback ---
def add_feedback(telegram_user_id: str, type_: str, message: str) -> int:
    with database_service.uow() as r:
        uid = r.users.ensure_user_for_telegram(telegram_user_id).id
        row = r.telegram_feedback.create(uid, type_, message)
        return row.id

def list_feedback(type_: Optional[str] = None):
    with database_service.uow() as r:
        return list(r.telegram_feedback.list(type_))

def update_feedback_status(feedback_id: int, status: str) -> bool:
    with database_service.uow() as r:
        return r.telegram_feedback.set_status(feedback_id, status)

# --- Command audit ---
def log_command_audit(telegram_user_id: str, command: str, **kwargs) -> int:
    with database_service.uow() as r:
        row = r.telegram_audit.log(str(telegram_user_id), command, **kwargs)
        return row.id

def get_user_command_history(telegram_user_id: str, limit: int = 20):
    with database_service.uow() as r:
        logs = r.telegram_audit.last_commands(str(telegram_user_id), limit=limit)

        # Convert to dictionaries while session is active
        return [
            {
                "id": log.id,
                "telegram_user_id": log.telegram_user_id,
                "command": log.command,
                "full_message": log.full_message,
                "is_registered_user": log.is_registered_user,
                "user_email": log.user_email,
                "success": log.success,
                "error_message": log.error_message,
                "response_time_ms": log.response_time_ms,
                "created": log.created_at.isoformat() if log.created_at else None
            }
            for log in logs
        ]

def get_all_command_audit(*, limit: int = 100, offset: int = 0,
                          user_id: Optional[str] = None, command: Optional[str] = None,
                          success_only: Optional[bool] = None,
                          start_date: Optional[str] = None, end_date: Optional[str] = None):
    with database_service.uow() as r:
        logs = r.telegram_audit.list(
            limit=limit, offset=offset, user_id=user_id, command=command,
            success_only=success_only, start_date=start_date, end_date=end_date
        )

        # Convert to dictionaries while session is active
        return [
            {
                "id": log.id,
                "telegram_user_id": log.telegram_user_id,
                "command": log.command,
                "full_message": log.full_message,
                "is_registered_user": log.is_registered_user,
                "user_email": log.user_email,
                "success": log.success,
                "error_message": log.error_message,
                "response_time_ms": log.response_time_ms,
                "created": log.created_at.isoformat() if log.created_at else None
            }
            for log in logs
        ]

def get_command_audit_stats() -> Dict[str, Any]:
    with database_service.uow() as r:
        return r.telegram_audit.stats()

# --- Broadcast logs ---
def log_broadcast(message: str, sent_by: str, success_count: int, total_count: int) -> int:
    """Log a broadcast message with delivery statistics."""
    with database_service.uow() as r:
        row = r.telegram_broadcast.create(
            message=message,
            sent_by=sent_by,
            success_count=success_count,
            total_count=total_count
        )
        return row.id

def get_broadcast_history(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    """Get broadcast history with pagination."""
    with database_service.uow() as r:
        logs = r.telegram_broadcast.list(limit=limit, offset=offset)

        # Convert to dictionaries while session is active
        return [
            {
                "id": log.id,
                "message": log.message,
                "sent_by": log.sent_by,
                "total_recipients": log.total_count,
                "successful_deliveries": log.success_count,
                "failed_deliveries": (log.total_count - log.success_count) if log.total_count and log.success_count else 0,
                "delivery_status": "completed" if log.success_count is not None else "pending",
                "sent_at": log.created_at.isoformat() if log.created_at else None
            }
            for log in logs
        ]

def get_broadcast_stats() -> Dict[str, Any]:
    """Get broadcast statistics."""
    with database_service.uow() as r:
        return r.telegram_broadcast.stats()

def get_active_alerts():
    """Get all active alerts for admin panel."""
    return list_active_alerts()

def get_active_schedules():
    """Get all active schedules for admin panel."""
    import json
    from src.data.db.models.model_jobs import JobType

    with database_service.uow() as r:
        schedules = r.jobs.list_schedules(
            job_type=JobType.SCREENER,
            enabled=True
        )

        # Convert schedules to telegram schedule format for compatibility
        telegram_schedules = []
        for schedule in schedules:
            task_params = schedule.task_params or {}
            config_json = task_params.get('config_json', '{}')

            # Parse config to extract schedule details for web UI compatibility
            try:
                config = json.loads(config_json) if isinstance(config_json, str) else config_json
            except (json.JSONDecodeError, TypeError):
                config = {}

            telegram_schedule = {
                'id': schedule.id,
                'user_id': schedule.user_id,
                'ticker': task_params.get('ticker', schedule.target or config.get('ticker', '')),
                'config_json': config_json,
                'schedule_config': task_params.get('schedule_config'),
                'scheduled_time': task_params.get('scheduled_time', config.get('scheduled_time', '')),
                'created_at': schedule.created_at,
                'enabled': schedule.enabled,
                'active': schedule.enabled,  # Add for web UI compatibility
                'cron': schedule.cron,
                'created': schedule.created_at.isoformat() if schedule.created_at else None,
                # Add additional fields for web UI compatibility
                'period': config.get('period'),
                'email': config.get('email', False),
                'indicators': config.get('indicators'),
                'interval': config.get('interval', config.get('timeframe')),
                'provider': config.get('provider'),
                'schedule_type': config.get('schedule_type', 'screener'),
                'list_type': config.get('list_type')
            }
            telegram_schedules.append(telegram_schedule)

        return telegram_schedules
