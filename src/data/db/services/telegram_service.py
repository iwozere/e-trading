# src/data/db/services/telegram_service.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import time

from sqlalchemy import select
from sqlalchemy.orm import Session

# low-level repos stay where they are
from src.data.db.repos.repo_telegram import (
    AlertsRepo, SchedulesRepo, FeedbackRepo, CommandAuditRepo,
    BroadcastRepo, SettingsRepo, VerificationRepo,
    TelegramAlert, TelegramSchedule, TelegramFeedback, TelegramCommandAudit
)

# cross-domain models live here (service layer is allowed to import multiple domains)
from src.data.db.models.model_users import User, AuthIdentity


@dataclass
class TelegramUserDTO:
    telegram_user_id: str
    user_id: int
    email: Optional[str]
    verified: Optional[bool] = None
    approved: Optional[bool] = None
    language: Optional[str] = None
    is_admin: Optional[bool] = None
    max_alerts: Optional[int] = None
    max_schedules: Optional[int] = None
    verification_code: Optional[str] = None
    code_sent_time: Optional[int] = None


class TelegramRepository:
    """High-level telegram facade: user/identity + alerts/schedules/settings/feedback/audit."""
    def __init__(self, s: Session) -> None:
        self.s = s
        self.alerts = AlertsRepo(s)
        self.schedules = SchedulesRepo(s)
        self.feedback = FeedbackRepo(s)
        self.command_audit = CommandAuditRepo(s)
        self.broadcasts = BroadcastRepo(s)
        self.settings = SettingsRepo(s)
        self.verification = VerificationRepo(s)

    # -------------------------- identity helpers --------------------------
    def _get_identity(self, tg_id: str) -> Optional[AuthIdentity]:
        q = select(AuthIdentity).where(
            AuthIdentity.provider == "telegram",
            AuthIdentity.external_id == str(tg_id),
        )
        return self.s.execute(q).scalars().first()

    def _ensure_user_and_identity(self, tg_id: str, email: Optional[str] = None) -> tuple[User, AuthIdentity]:
        ident = self._get_identity(tg_id)
        if ident:
            user = self.s.get(User, ident.user_id)
            if email and user and user.email != email:
                user.email = email
            return user, ident

        user = User(email=email)
        self.s.add(user); self.s.flush()
        ident = AuthIdentity(user_id=user.id, provider="telegram", external_id=str(tg_id), identity_metadata={})
        self.s.add(ident); self.s.flush()
        return user, ident

    def _update_meta(self, ident: AuthIdentity, **fields) -> None:
        md = dict(ident.identity_metadata or {})
        for k, v in fields.items():
            if v is not None:
                md[k] = v
        ident.identity_metadata = md

    def _resolve_user_id(self, tg_id: str, *, autocreate: bool = False, email: Optional[str] = None) -> Optional[int]:
        ident = self._get_identity(tg_id)
        if ident:
            return ident.user_id
        if not autocreate:
            return None
        user, _ = self._ensure_user_and_identity(tg_id, email=email)
        return user.id

    # ------------------------------ users -----------------------------------
    def upsert_user(self, tg_id: str, *, email: Optional[str] = None,
                    verified: Optional[bool] = None, approved: Optional[bool] = None,
                    language: Optional[str] = None, is_admin: Optional[bool] = None,
                    max_alerts: Optional[int] = None, max_schedules: Optional[int] = None) -> None:
        user, ident = self._ensure_user_and_identity(tg_id, email=email)
        self._update_meta(
            ident,
            verified=verified, approved=approved, language=language, is_admin=is_admin,
            max_alerts=max_alerts, max_schedules=max_schedules,
        )

    def get_user(self, tg_id: str) -> Optional[TelegramUserDTO]:
        ident = self._get_identity(tg_id)
        if not ident:
            return None
        user = self.s.get(User, ident.user_id)
        md = ident.identity_metadata or {}
        return TelegramUserDTO(
            telegram_user_id=str(tg_id),
            user_id=ident.user_id,
            email=user.email if user else None,
            verified=md.get("verified"),
            approved=md.get("approved"),
            language=md.get("language"),
            is_admin=md.get("is_admin"),
            max_alerts=md.get("max_alerts"),
            max_schedules=md.get("max_schedules"),
            verification_code=md.get("verification_code"),
            code_sent_time=md.get("code_sent_time"),
        )

    def approve_user(self, tg_id: str, approved: bool) -> bool:
        ident = self._get_identity(tg_id)
        if not ident:
            return False
        self._update_meta(ident, approved=approved)
        return True

    def get_user_status(self, tg_id: str) -> Optional[dict]:
        ident = self._get_identity(tg_id)
        if not ident:
            return None
        md = ident.identity_metadata or {}
        return {"approved": md.get("approved"), "verified": md.get("verified")}

    def list_pending_approvals(self) -> list[dict]:
        rows = self.s.execute(
            select(AuthIdentity, User.email)
            .join(User, User.id == AuthIdentity.user_id)
            .where(AuthIdentity.provider == "telegram")
        ).all()
        out = []
        for ident, email in rows:
            md = ident.identity_metadata or {}
            if md.get("approved") is False:
                out.append({"telegram_user_id": ident.external_id, "email": email})
        return out

    def get_admin_user_ids(self) -> list[str]:
        rows = self.s.execute(
            select(AuthIdentity).where(AuthIdentity.provider == "telegram")
        ).scalars().all()
        return [r.external_id for r in rows if (r.identity_metadata or {}).get("is_admin") is True]

    def set_user_limit(self, tg_id: str, key: str, value: int) -> None:
        assert key in ("max_alerts", "max_schedules")
        ident = self._get_identity(tg_id)
        if not ident:
            _, ident = self._ensure_user_and_identity(tg_id)
        self._update_meta(ident, **{key: int(value)})

    # ----------------------- verification codes -----------------------------
    def set_verification_code(self, tg_id: str, *, code: str, sent_time: int) -> None:
        uid = self._resolve_user_id(tg_id, autocreate=True)
        self.verification.issue(uid, code=code, sent_time=sent_time)
        ident = self._get_identity(tg_id)
        self._update_meta(ident, verification_code=code, code_sent_time=sent_time)

    def count_codes_last_hour(self, tg_id: str) -> int:
        uid = self._resolve_user_id(tg_id, autocreate=True)
        now_unix = int(time.time())
        return self.verification.count_last_hour_by_user_id(uid, now_unix)

    # -------------------------------- alerts --------------------------------
    def add_alert(self, tg_id: str, ticker: str, price: float, condition: str, *, email: Optional[bool] = None) -> int:
        uid = self._resolve_user_id(tg_id, autocreate=True)
        row = self.alerts.create(uid, ticker, price=price, condition=condition,
                                 alert_type="price", email=email, is_armed=True, active=True)
        return row.id

    def add_indicator_alert(self, tg_id: str, ticker: str, config_json: str, *,
                            alert_action: Optional[str] = None, timeframe: Optional[str] = None,
                            email: Optional[bool] = None) -> int:
        uid = self._resolve_user_id(tg_id, autocreate=True)
        row = self.alerts.create(uid, ticker, alert_type="indicator", config_json=config_json,
                                 alert_action=alert_action, timeframe=timeframe, email=email,
                                 is_armed=True, active=True)
        return row.id

    def get_alert(self, alert_id: int) -> Optional[TelegramAlert]:
        return self.alerts.get(alert_id)

    def list_alerts(self, tg_id: str) -> list[TelegramAlert]:
        uid = self._resolve_user_id(tg_id, autocreate=True)
        return list(self.alerts.list_for_user(uid))

    def get_active_alerts(self) -> list[TelegramAlert]:
        return list(self.alerts.list_active_global())

    def get_alerts_by_type(self, alert_type: str) -> list[TelegramAlert]:
        return list(self.alerts.list_by_type(alert_type))

    def update_alert(self, alert_id: int, **values) -> bool:
        return self.alerts.update(alert_id, **values)

    def delete_alert(self, alert_id: int) -> bool:
        self.alerts.delete(alert_id)
        return True

    # ------------------------------- schedules ------------------------------
    def add_schedule(self, tg_id: str, ticker: str, scheduled_time: str, **kwargs) -> int:
        uid = self._resolve_user_id(tg_id, autocreate=True)
        row = self.schedules.upsert({"user_id": uid, "ticker": ticker, "scheduled_time": scheduled_time, **kwargs})
        return row.id

    def add_json_schedule(self, tg_id: str, config_json: str, *, schedule_config: Optional[str] = None) -> int:
        uid = self._resolve_user_id(tg_id, autocreate=True)
        row = self.schedules.upsert({"user_id": uid, "config_json": config_json, "schedule_config": schedule_config})
        return row.id

    def create_schedule(self, data: dict) -> int:
        uid = data.get("user_id")
        if not isinstance(uid, int):
            uid = self._resolve_user_id(str(uid), autocreate=True)
        row = self.schedules.upsert({**data, "user_id": uid})
        return row.id

    def list_schedules(self, tg_id: str) -> list[TelegramSchedule]:
        uid = self._resolve_user_id(tg_id, autocreate=True)
        return list(self.schedules.list_for_user(uid))

    def get_schedule(self, schedule_id: int) -> Optional[TelegramSchedule]:
        return self.schedules.get(schedule_id)

    def update_schedule(self, schedule_id: int, **values) -> bool:
        return self.schedules.update(schedule_id, **values)

    def get_active_schedules(self) -> list[TelegramSchedule]:
        return list(self.schedules.list_active_global())

    def get_schedules_by_config(self, schedule_config: str) -> list[TelegramSchedule]:
        return list(self.schedules.list_by_config(schedule_config))

    def delete_schedule(self, schedule_id: int) -> bool:
        self.schedules.delete(schedule_id)
        return True

    # ------------------------------- settings --------------------------------
    def get_setting(self, key: str) -> Optional[str]:
        row = self.settings.get(key)
        return row.value if row else None

    def set_setting(self, key: str, value: Optional[str]) -> None:
        self.settings.set(key, value)

    # ------------------------------- feedback --------------------------------
    def add_feedback(self, tg_id: str, type_: str, message: str) -> int:
        uid = self._resolve_user_id(tg_id, autocreate=True)
        row = self.feedback.create(uid, type_, message)
        return row.id

    def list_feedback(self, type_: Optional[str] = None) -> list[TelegramFeedback]:
        return list(self.feedback.list(type_))

    def update_feedback_status(self, feedback_id: int, status: str) -> bool:
        return self.feedback.set_status(feedback_id, status)

    # ----------------------------- command audit -----------------------------
    def log_command_audit(self, tg_id: str, command: str, **kwargs) -> int:
        row = self.command_audit.log(str(tg_id), command, **kwargs)
        return row.id

    def get_user_command_history(self, tg_id: str, limit: int = 20) -> list[TelegramCommandAudit]:
        return list(self.command_audit.last_commands(str(tg_id), limit=limit))

    def get_all_command_audit(self, *, limit: int = 100, offset: int = 0,
                              user_id: Optional[str] = None, command: Optional[str] = None,
                              success_only: Optional[bool] = None,
                              start_date: Optional[str] = None, end_date: Optional[str] = None) -> list[TelegramCommandAudit]:
        return list(self.command_audit.list(
            limit=limit, offset=offset, user_id=user_id, command=command,
            success_only=success_only, start_date=start_date, end_date=end_date
        ))

    def get_command_audit_stats(self) -> dict:
        return self.command_audit.stats()

    def get_unique_users_command_history(self) -> list[dict]:
        return self.command_audit.unique_users_summary()

