# src/data/db/services/telegram_service.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import time

from src.data.db.services.database_service import database_service
from src.data.db.services import users_service  # reuse for user-related DTOs

# --- Users (Telegram-focused) ---
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
def add_json_alert(
    telegram_user_id: str,
    config_json: str,
    *,
    email: Optional[bool] = None,
    status: str = "ARMED",
    re_arm_config: Optional[str] = None,
) -> int:
    """
    Create an alert using the new schema.
    Example config_json:
      {"ticker":"AAPL","rule":{"price_above":170}}
    Example re_arm_config:
      {"rearm_on_cross_below":170}
    """
    with database_service.uow() as r:
        uid = r.users.ensure_user_for_telegram(telegram_user_id).id
        row = r.telegram_alerts.create(
            uid,
            config_json=config_json,
            email=email,
            status=status,
            re_arm_config=re_arm_config,
        )
        return row.id

def list_alerts(telegram_user_id: str):
    with database_service.uow() as r:
        uid = r.users.ensure_user_for_telegram(telegram_user_id).id
        return list(r.telegram_alerts.list_for_user(uid))

def get_alert(alert_id: int):
    with database_service.uow() as r:
        return r.telegram_alerts.get(alert_id)

def update_alert(alert_id: int, **values) -> bool:
    with database_service.uow() as r:
        return r.telegram_alerts.update(alert_id, **values)

def delete_alert(alert_id: int) -> bool:
    with database_service.uow() as r:
        r.telegram_alerts.delete(alert_id)
        return True

# --- Schedules ---
def add_schedule(telegram_user_id: str, ticker: str, scheduled_time: str, **kwargs) -> int:
    with database_service.uow() as r:
        uid = r.users.ensure_user_for_telegram(telegram_user_id).id
        row = r.telegram_schedules.upsert({"user_id": uid, "ticker": ticker, "scheduled_time": scheduled_time, **kwargs})
        return row.id

def add_json_schedule(telegram_user_id: str, config_json: str, *, schedule_config: Optional[str] = None) -> int:
    with database_service.uow() as r:
        uid = r.users.ensure_user_for_telegram(telegram_user_id).id
        row = r.telegram_schedules.upsert({"user_id": uid, "config_json": config_json, "schedule_config": schedule_config})
        return row.id

def list_schedules(telegram_user_id: str):
    with database_service.uow() as r:
        uid = r.users.ensure_user_for_telegram(telegram_user_id).id
        return list(r.telegram_schedules.list_for_user(uid))

def get_schedule(schedule_id: int):
    with database_service.uow() as r:
        return r.telegram_schedules.get(schedule_id)

def update_schedule(schedule_id: int, **values) -> bool:
    with database_service.uow() as r:
        return r.telegram_schedules.update(schedule_id, **values)

def delete_schedule(schedule_id: int) -> bool:
    with database_service.uow() as r:
        r.telegram_schedules.delete(schedule_id)
        return True

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
        return list(r.telegram_audit.last_commands(str(telegram_user_id), limit=limit))

def get_all_command_audit(*, limit: int = 100, offset: int = 0,
                          user_id: Optional[str] = None, command: Optional[str] = None,
                          success_only: Optional[bool] = None,
                          start_date: Optional[str] = None, end_date: Optional[str] = None):
    with database_service.uow() as r:
        return list(r.telegram_audit.list(
            limit=limit, offset=offset, user_id=user_id, command=command,
            success_only=success_only, start_date=start_date, end_date=end_date
        ))

def get_command_audit_stats() -> Dict[str, Any]:
    with database_service.uow() as r:
        return r.telegram_audit.stats()
