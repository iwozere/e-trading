# src/data/db/services/telegram_service.py
"""
Telegram Service
---------------
Service layer for Telegram bot operations. Provides high-level business logic
for managing Telegram users, alerts, schedules, feedback, and command audit logs.

All operations use a single Unit-of-Work:
    with uow() as r:
        ... r.telegram_*
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import time
import json

from src.data.db.services.base_service import BaseDBService, with_uow, handle_db_error
from src.data.db.services.users_service import users_service


class TelegramService(BaseDBService):
    """Service layer for Telegram bot operations.

    Provides high-level business logic for managing Telegram users, alerts,
    schedules, feedback, and command audit logs. Methods use the Unit of Work
    via decorators for consistent transaction/error handling.
    """

    def __init__(self) -> None:
        super().__init__()

    # --- Users ---
    @with_uow
    @handle_db_error
    def list_users(self) -> List[Dict[str, Any]]:
        return users_service.list_users_for_broadcast()

    @with_uow
    @handle_db_error
    def reset_user_email_verification(self, telegram_user_id: str) -> bool:
        users_service.update_telegram_profile(
            telegram_user_id,
            verified=False,
            verification_code=None,
            code_sent_time=None,
        )
        return True

    @with_uow
    @handle_db_error
    def verify_user_email(self, telegram_user_id: str) -> bool:
        users_service.update_telegram_profile(telegram_user_id, verified=True)
        return True

    @with_uow
    @handle_db_error
    def set_user_limit(self, telegram_user_id: str, key: str, value: int) -> None:
        assert key in ("max_alerts", "max_schedules")
        users_service.update_telegram_profile(telegram_user_id, **{key: int(value)})

    @with_uow
    @handle_db_error
    def get_user_limit(self, telegram_user_id: str, key: str) -> Optional[int]:
        assert key in ("max_alerts", "max_schedules")
        profile = users_service.get_telegram_profile(telegram_user_id)
        if not profile:
            return None
        return profile.get(key)

    @with_uow
    @handle_db_error
    def get_user_status(self, telegram_user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user status including approval, verification, and limit information.

        Args:
            telegram_user_id: Telegram user ID

        Returns:
            Dictionary with user status or None if user not found
        """
        profile = users_service.get_telegram_profile(telegram_user_id)
        if not profile:
            return None

        return {
            "telegram_user_id": telegram_user_id,
            "email": profile.get("email"),
            "approved": profile.get("approved", False),
            "verified": profile.get("verified", False),
            "max_alerts": profile.get("max_alerts"),
            "max_schedules": profile.get("max_schedules"),
            "language": profile.get("language", "en"),
            "is_admin": profile.get("is_admin", False),
        }

    @with_uow
    @handle_db_error
    def approve_user(self, telegram_user_id: str) -> bool:
        users_service.update_telegram_profile(telegram_user_id, approved=True)
        return True

    @with_uow
    @handle_db_error
    def reject_user(self, telegram_user_id: str) -> bool:
        users_service.update_telegram_profile(telegram_user_id, approved=False)
        return True

    @with_uow
    @handle_db_error
    def update_user_language(self, telegram_user_id: str, language: str) -> bool:
        from sqlalchemy import text

        uid = users_service.ensure_user_for_telegram(telegram_user_id)

        session = self.repos.s
        result = session.execute(
            text("UPDATE usr_users SET language = :language WHERE id = :user_id"),
            {"language": language, "user_id": uid},
        )
        return result.rowcount > 0

    @with_uow
    @handle_db_error
    def set_user_email(self, telegram_user_id: str, email: str, code: str, sent_time: int, language: str = "en", is_admin: bool = False) -> None:
        users_service.update_telegram_profile(
            telegram_user_id,
            email=email,
            verification_code=code,
            code_sent_time=sent_time,
            language=language,
            is_admin=is_admin,
            verified=False,
        )

    @with_uow
    @handle_db_error
    def verify_code(self, telegram_user_id: str, code: str, expiry_seconds: int = 3600) -> bool:
        from sqlalchemy import text

        current_time = int(time.time())
        uid = users_service.ensure_user_for_telegram(telegram_user_id)

        session = self.repos.s
        result = session.execute(
            text("SELECT code, sent_time FROM usr_verification_codes WHERE user_id = :user_id AND code = :code ORDER BY created_at DESC LIMIT 1"),
            {"user_id": uid, "code": code},
        )

        row = result.fetchone()
        if not row:
            return False

        stored_code, code_sent_time = row
        if stored_code == code and (current_time - code_sent_time) <= expiry_seconds:
            users_service.update_telegram_profile(telegram_user_id, verified=True)
            return True
        return False

    # --- Verification helpers ---
    @with_uow
    @handle_db_error
    def set_verification_code(self, telegram_user_id: str, *, code: str, sent_time: int) -> None:
        uid = users_service.ensure_user_for_telegram(telegram_user_id)
        self.repos.telegram_verification.issue(uid, code=code, sent_time=sent_time)
        users_service.update_telegram_profile(telegram_user_id, verification_code=code, code_sent_time=sent_time)

    @with_uow
    @handle_db_error
    def count_codes_last_hour(self, telegram_user_id: str, now_unix: Optional[int] = None) -> int:
        uid = users_service.ensure_user_for_telegram(telegram_user_id)
        return self.repos.telegram_verification.count_last_hour_by_user_id(uid, int(now_unix or time.time()))

    # --- Alerts / Jobs ---
    @with_uow
    @handle_db_error
    def add_json_alert(self, telegram_user_id: str, config_json: str, *, email: Optional[bool] = None, status: str = "ARMED", re_arm_config: Optional[str] = None) -> int:
        from src.data.db.models.model_jobs import JobType

        uid = users_service.ensure_user_for_telegram(telegram_user_id)
        config = json.loads(config_json) if isinstance(config_json, str) else config_json
        ticker = config.get("ticker", "unknown")

        schedule_data = {
            "user_id": uid,
            "name": f"Alert_{ticker}_{uid}_{int(time.time())}",
            "job_type": JobType.ALERT.value,
            "target": ticker,
            "task_params": {
                "config_json": config_json,
                "email": email,
                "status": status,
                "re_arm_config": re_arm_config,
                "telegram_user_id": telegram_user_id,
            },
            "cron": "* * * * *",
            "enabled": status == "ARMED",
        }

        schedule = self.repos.jobs.create_schedule(schedule_data)
        return schedule.id

    @with_uow
    @handle_db_error
    def add_alert(self, telegram_user_id: str, ticker: str, price: float, condition: str, email: bool = False) -> int:
        config_json = json.dumps({"ticker": ticker, "price": price, "condition": condition, "email": email})
        return self.add_json_alert(telegram_user_id, config_json, email=email)

    @with_uow
    @handle_db_error
    def add_indicator_alert(self, telegram_user_id: str, ticker: str, indicator: str, condition: str, value: float, timeframe: str = "15m", alert_action: str = "telegram", email: bool = False) -> int:
        config_json = json.dumps({
            "ticker": ticker,
            "indicator": indicator,
            "condition": condition,
            "value": value,
            "timeframe": timeframe,
            "alert_action": alert_action,
            "email": email,
        })
        return self.add_json_alert(telegram_user_id, config_json, email=email)

    @with_uow
    @handle_db_error
    def list_alerts(self, telegram_user_id: str) -> List[Dict[str, Any]]:
        """
        List all alerts for a specific telegram user.

        Args:
            telegram_user_id: Telegram user ID

        Returns:
            List of alert dictionaries with alert details
        """
        from src.data.db.models.model_jobs import JobType

        uid = users_service.ensure_user_for_telegram(telegram_user_id)

        # Get all alert-type schedules for this user
        schedules = self.repos.jobs.list_schedules(
            user_id=uid,
            job_type=JobType.ALERT
        )

        alerts: List[Dict[str, Any]] = []
        for schedule in schedules:
            task_params = schedule.task_params or {}
            config_json = task_params.get("config_json", "{}")

            try:
                config = json.loads(config_json) if isinstance(config_json, str) else config_json
            except (json.JSONDecodeError, TypeError):
                config = {}

            alert = {
                "id": schedule.id,
                "user_id": schedule.user_id,
                "ticker": config.get("ticker", schedule.target or ""),
                "active": schedule.enabled,
                "email": task_params.get("email", config.get("email", False)),
                "status": task_params.get("status", "ARMED" if schedule.enabled else "DISARMED"),
                "created_at": schedule.created_at,
                "updated_at": schedule.updated_at,
            }

            # Add alert-type specific fields
            if "price" in config:
                alert.update({
                    "alert_type": "price",
                    "price": config.get("price"),
                    "condition": config.get("condition"),
                })
            elif "indicator" in config:
                alert.update({
                    "alert_type": "indicator",
                    "indicator": config.get("indicator"),
                    "condition": config.get("condition"),
                    "value": config.get("value"),
                    "timeframe": config.get("timeframe", "15m"),
                    "alert_action": config.get("alert_action", "telegram"),
                })
            else:
                # Generic alert type
                alert.update({
                    "alert_type": "custom",
                    "config": config,
                })

            alerts.append(alert)

        return alerts

    # --- Schedules ---
    @with_uow
    @handle_db_error
    def add_schedule(self, telegram_user_id: str, ticker: str, scheduled_time: str, **kwargs) -> int:
        from src.data.db.models.model_jobs import JobType

        uid = users_service.ensure_user_for_telegram(telegram_user_id)
        cron = f"0 {scheduled_time.split(':')[0]} * * *" if ":" in scheduled_time else "0 9 * * *"

        schedule_data = {
            "user_id": uid,
            "name": f"Schedule_{ticker}_{uid}_{int(time.time())}",
            "job_type": JobType.SCREENER.value,
            "target": ticker,
            "task_params": {"ticker": ticker, "scheduled_time": scheduled_time, "telegram_user_id": telegram_user_id, **kwargs},
            "cron": cron,
            "enabled": True,
        }

        schedule = self.repos.jobs.create_schedule(schedule_data)
        return schedule.id

    @with_uow
    @handle_db_error
    def add_json_schedule(self, telegram_user_id: str, config_json: str, *, schedule_config: Optional[str] = None) -> int:
        from src.data.db.models.model_jobs import JobType

        uid = users_service.ensure_user_for_telegram(telegram_user_id)
        config = json.loads(config_json) if isinstance(config_json, str) else config_json
        ticker = config.get("ticker", "unknown")

        schedule_data = {
            "user_id": uid,
            "name": f"JSONSchedule_{ticker}_{uid}_{int(time.time())}",
            "job_type": JobType.SCREENER.value,
            "target": ticker,
            "task_params": {"config_json": config_json, "schedule_config": schedule_config, "telegram_user_id": telegram_user_id},
            "cron": "0 9 * * *",
            "enabled": True,
        }
        schedule = self.repos.jobs.create_schedule(schedule_data)
        return schedule.id

    @with_uow
    @handle_db_error
    def list_schedules(self, telegram_user_id: str):
        from src.data.db.models.model_jobs import JobType

        uid = users_service.ensure_user_for_telegram(telegram_user_id)
        # Get all schedules except alerts (which are handled by /alerts)
        all_schedules = self.repos.jobs.list_schedules(user_id=uid)

        # Filter out ALERT type schedules
        schedules = [s for s in all_schedules if s.job_type != JobType.ALERT.value]

        telegram_schedules: List[Dict[str, Any]] = []
        for schedule in schedules:
            task_params = schedule.task_params or {}
            config_json = task_params.get("config_json", "{}")
            try:
                config = json.loads(config_json) if isinstance(config_json, str) else config_json
            except (json.JSONDecodeError, TypeError):
                config = {}

            telegram_schedule = {
                "id": schedule.id,
                "user_id": schedule.user_id,
                "name": schedule.name,
                "job_type": schedule.job_type,
                "ticker": task_params.get("ticker", schedule.target or config.get("ticker", "")),
                "config_json": config_json,
                "schedule_config": task_params.get("schedule_config"),
                "scheduled_time": task_params.get("scheduled_time", config.get("scheduled_time", "")),
                "created_at": schedule.created_at,
                "enabled": schedule.enabled,
                "active": schedule.enabled,
                "cron": schedule.cron,
                "created": schedule.created_at.isoformat() if schedule.created_at else None,
                "period": config.get("period"),
                "email": config.get("email", False),
                "indicators": config.get("indicators"),
                "interval": config.get("interval", config.get("timeframe")),
                "provider": config.get("provider"),
                "schedule_type": config.get("schedule_type", "screener"),
                "list_type": config.get("list_type"),
            }
            telegram_schedules.append(telegram_schedule)

        return telegram_schedules

    @with_uow
    @handle_db_error
    def get_schedule(self, schedule_id: int):
        schedule = self.repos.jobs.get_schedule(schedule_id)
        if not schedule or schedule.job_type not in ["screener", "report"]:
            return None

        task_params = schedule.task_params or {}
        config_json = task_params.get("config_json", "{}")
        try:
            config = json.loads(config_json) if isinstance(config_json, str) else config_json
        except (json.JSONDecodeError, TypeError):
            config = {}

        return {
            "id": schedule.id,
            "user_id": schedule.user_id,
            "ticker": task_params.get("ticker", schedule.target or config.get("ticker", "")),
            "config_json": config_json,
            "schedule_config": task_params.get("schedule_config"),
            "scheduled_time": task_params.get("scheduled_time", config.get("scheduled_time", "")),
            "created_at": schedule.created_at,
            "enabled": schedule.enabled,
            "active": schedule.enabled,
            "cron": schedule.cron,
            "created": schedule.created_at.isoformat() if schedule.created_at else None,
            "period": config.get("period"),
            "email": config.get("email", False),
            "indicators": config.get("indicators"),
            "interval": config.get("interval", config.get("timeframe")),
            "provider": config.get("provider"),
            "schedule_type": config.get("schedule_type", "screener"),
            "list_type": config.get("list_type"),
        }

    @with_uow
    @handle_db_error
    def update_schedule(self, schedule_id: int, **values) -> bool:
        schedule = self.repos.jobs.get_schedule(schedule_id)
        if not schedule or schedule.job_type not in ["screener", "report"]:
            return False

        task_params = dict(schedule.task_params or {})
        update_data: Dict[str, Any] = {}
        for key, value in values.items():
            if key in ["ticker", "config_json", "schedule_config", "scheduled_time"]:
                task_params[key] = value
            elif key == "enabled":
                update_data["enabled"] = value
            elif key == "cron":
                update_data["cron"] = value

        if task_params != schedule.task_params:
            update_data["task_params"] = task_params

        if update_data:
            updated = self.repos.jobs.update_schedule(schedule_id, update_data)
            return updated is not None
        return True

    @with_uow
    @handle_db_error
    def delete_schedule(self, schedule_id: int) -> bool:
        return self.repos.jobs.delete_schedule(schedule_id)

    # --- Settings ---
    @with_uow
    @handle_db_error
    def get_setting(self, key: str) -> Optional[str]:
        row = self.repos.telegram_settings.get(key)
        return row.value if row else None

    @with_uow
    @handle_db_error
    def set_setting(self, key: str, value: Optional[str]) -> None:
        self.repos.telegram_settings.set(key, value)

    # --- Feedback ---
    @with_uow
    @handle_db_error
    def add_feedback(self, telegram_user_id: str, type_: str, message: str) -> int:
        uid = users_service.ensure_user_for_telegram(telegram_user_id)
        row = self.repos.telegram_feedback.create(uid, type_, message)
        return row.id

    @with_uow
    @handle_db_error
    def list_feedback(self, type_: Optional[str] = None):
        return list(self.repos.telegram_feedback.list(type_))

    @with_uow
    @handle_db_error
    def update_feedback_status(self, feedback_id: int, status: str) -> bool:
        return self.repos.telegram_feedback.set_status(feedback_id, status)

    # --- Command audit ---
    @with_uow
    @handle_db_error
    def log_command_audit(self, telegram_user_id: str, command: str, **kwargs) -> int:
        row = self.repos.telegram_audit.log(str(telegram_user_id), command, **kwargs)
        return row.id

    @with_uow
    @handle_db_error
    def get_user_command_history(self, telegram_user_id: str, limit: int = 20):
        logs = self.repos.telegram_audit.last_commands(str(telegram_user_id), limit=limit)
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
                "created": log.created_at.isoformat() if log.created_at else None,
            }
            for log in logs
        ]

    @with_uow
    @handle_db_error
    def get_all_command_audit(self, *, limit: int = 100, offset: int = 0, user_id: Optional[str] = None, command: Optional[str] = None, success_only: Optional[bool] = None, start_date: Optional[str] = None, end_date: Optional[str] = None):
        logs = self.repos.telegram_audit.list(limit=limit, offset=offset, user_id=user_id, command=command, success_only=success_only, start_date=start_date, end_date=end_date)
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
                "created": log.created_at.isoformat() if log.created_at else None,
            }
            for log in logs
        ]

    @with_uow
    @handle_db_error
    def get_command_audit_stats(self) -> Dict[str, Any]:
        return self.repos.telegram_audit.stats()

    # --- Broadcast logs ---
    @with_uow
    @handle_db_error
    def log_broadcast(self, message: str, sent_by: str, success_count: int, total_count: int) -> int:
        row = self.repos.telegram_broadcast.create(message=message, sent_by=sent_by, success_count=success_count, total_count=total_count)
        return row.id

    @with_uow
    @handle_db_error
    def get_broadcast_history(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        logs = self.repos.telegram_broadcast.list(limit=limit, offset=offset)
        return [
            {
                "id": log.id,
                "message": log.message,
                "sent_by": log.sent_by,
                "total_recipients": log.total_count,
                "successful_deliveries": log.success_count,
                "failed_deliveries": (log.total_count - log.success_count) if log.total_count and log.success_count else 0,
                "delivery_status": "completed" if log.success_count is not None else "pending",
                "sent_at": log.created_at.isoformat() if log.created_at else None,
            }
            for log in logs
        ]

    @with_uow
    @handle_db_error
    def get_broadcast_stats(self) -> Dict[str, Any]:
        return self.repos.telegram_broadcast.stats()

    @with_uow
    @handle_db_error
    def get_active_alerts(self):
        return self.repos.jobs.active_alerts()

    @with_uow
    @handle_db_error
    def get_active_schedules(self):
        from src.data.db.models.model_jobs import JobType

        schedules = self.repos.jobs.list_schedules(job_type=JobType.SCREENER, enabled=True)
        telegram_schedules: List[Dict[str, Any]] = []
        for schedule in schedules:
            task_params = schedule.task_params or {}
            config_json = task_params.get("config_json", "{}")
            try:
                config = json.loads(config_json) if isinstance(config_json, str) else config_json
            except (json.JSONDecodeError, TypeError):
                config = {}

            telegram_schedule = {
                "id": schedule.id,
                "user_id": schedule.user_id,
                "ticker": task_params.get("ticker", schedule.target or config.get("ticker", "")),
                "config_json": config_json,
                "schedule_config": task_params.get("schedule_config"),
                "scheduled_time": task_params.get("scheduled_time", config.get("scheduled_time", "")),
                "created_at": schedule.created_at,
                "enabled": schedule.enabled,
                "active": schedule.enabled,
                "cron": schedule.cron,
                "created": schedule.created_at.isoformat() if schedule.created_at else None,
                "period": config.get("period"),
                "email": config.get("email", False),
                "indicators": config.get("indicators"),
                "interval": config.get("interval", config.get("timeframe")),
                "provider": config.get("provider"),
                "schedule_type": config.get("schedule_type", "screener"),
                "list_type": config.get("list_type"),
            }
            telegram_schedules.append(telegram_schedule)

        return telegram_schedules


# Global service instance
telegram_service = TelegramService()
