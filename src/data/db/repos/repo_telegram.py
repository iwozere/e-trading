# src/data/db/repos/repo_telegram.py
from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional, Sequence, List, Dict, Any
from sqlalchemy import select, update, delete, func
from sqlalchemy.orm import Session

from src.data.db.models.model_users import User, AuthIdentity
from src.data.db.models.model_telegram import (
    TelegramSetting,
    TelegramFeedback,
    TelegramCommandAudit,
    TelegramBroadcastLog,
)

UTC = timezone.utc
utcnow = lambda: datetime.now(UTC)


# -------------------- Alerts --------------------
class AlertsRepo:
    def __init__(self, s: Session) -> None:
        self.s = s

    def create(
        self,
        user_id: int,
        *,
        config_json: str,
        email: Optional[bool] = None,
        status: str = "ARMED",
        re_arm_config: Optional[str] = None,
    ) -> TelegramAlert:
        """
        Create an alert aligned with the new schema:
          - rules live in config_json (e.g. {"ticker":"AAPL","rule":{"price_above":170}})
          - rearm rules live in re_arm_config (e.g. {"rearm_on_cross_below":170})
          - status: ARMED | TRIGGERED | INACTIVE
        """
        row = TelegramAlert(
            user_id=user_id,
            status=status,
            email=bool(email) if email is not None else False,
            created_at=utcnow(),
            config_json=config_json,
            re_arm_config=re_arm_config,
            trigger_count=0,
        )
        self.s.add(row)
        self.s.flush()
        return row

    def get(self, alert_id: int) -> Optional[TelegramAlert]:
        return self.s.get(TelegramAlert, alert_id)

    def list_for_user(self, user_id: int) -> Sequence[TelegramAlert]:
        q = select(TelegramAlert).where(TelegramAlert.user_id == user_id)
        return list(self.s.execute(q).scalars())

    def list_by_status(self, status: str = "ARMED") -> Sequence[TelegramAlert]:
        q = select(TelegramAlert).where(TelegramAlert.status == status)
        return list(self.s.execute(q).scalars())

    def list_active(
        self,
        user_id: int | None = None,
        *,
        limit: int | None = None,
        offset: int = 0,
        older_first: bool = False,
    ):
        q = select(TelegramAlert).where(TelegramAlert.status.in_(("ARMED", "TRIGGERED")))
        if user_id is not None:
            q = q.where(TelegramAlert.user_id == user_id)
        q = q.order_by(TelegramAlert.id.asc() if older_first else TelegramAlert.id.desc())
        if offset:
            q = q.offset(offset)
        if limit:
            q = q.limit(limit)
        return list(self.s.execute(q).scalars())

    def update(self, alert_id: int, **values) -> bool:
        if not values:
            return False
        # Allowed keys include: status, email, config_json, re_arm_config,
        # trigger_count, last_trigger_condition, last_triggered_at, etc.
        res = self.s.execute(
            update(TelegramAlert)
            .where(TelegramAlert.id == alert_id)
            .values(**values)
        )
        return (res.rowcount or 0) > 0

    def delete(self, alert_id: int) -> None:
        self.s.execute(delete(TelegramAlert).where(TelegramAlert.id == alert_id))
        self.s.flush()


# -------------------- Schedules --------------------
class SchedulesRepo:
    def __init__(self, s: Session) -> None:
        self.s = s

    def upsert(self, data: dict) -> TelegramSchedule:
        # Copy so we don't mutate caller dict
        data = dict(data)
        # Accept alias 'schedule_time' -> 'scheduled_time'
        if "schedule_time" in data and "scheduled_time" not in data:
            data["scheduled_time"] = data.pop("schedule_time")
        # Enforce NOT NULL scheduled_time default
        if not data.get("scheduled_time"):
            data["scheduled_time"] = "09:00"
        # Normalize booleans and set robust defaults (don’t rely on DB defaults)
        if "email" in data:
            data["email"] = bool(data["email"])
        if "active" not in data:
            data["active"] = True

        row = TelegramSchedule(**data)
        self.s.add(row); self.s.flush()
        return row

    def get(self, schedule_id: int) -> Optional[TelegramSchedule]:
        return self.s.get(TelegramSchedule, schedule_id)

    def list_for_user(self, user_id: int) -> Sequence[TelegramSchedule]:
        q = select(TelegramSchedule).where(TelegramSchedule.user_id == user_id)
        return list(self.s.execute(q).scalars())

    def list_by_config(self, schedule_config: str) -> Sequence[TelegramSchedule]:
        q = select(TelegramSchedule).where(TelegramSchedule.schedule_config == schedule_config)
        return list(self.s.execute(q).scalars())

    def update(self, schedule_id: int, **values) -> bool:
        if "schedule_time" in values and "scheduled_time" not in values:
            values = {**values, "scheduled_time": values.pop("schedule_time")}
        if "scheduled_time" in values and not values["scheduled_time"]:
            values["scheduled_time"] = "09:00"
        res = self.s.execute(update(TelegramSchedule).where(TelegramSchedule.id == schedule_id).values(**values))
        return (res.rowcount or 0) > 0

    def delete(self, schedule_id: int) -> None:
        self.s.execute(delete(TelegramSchedule).where(TelegramSchedule.id == schedule_id))
        self.s.flush()


# -------------------- Settings --------------------
class SettingsRepo:
    def __init__(self, s: Session) -> None:
        self.s = s

    def get(self, key: str) -> Optional[TelegramSetting]:
        return self.s.get(TelegramSetting, key)

    def set(self, key: str, value: Optional[str]) -> None:
        row = self.s.get(TelegramSetting, key)
        if row is None:
            row = TelegramSetting(key=key, value=value)
            self.s.add(row)
        else:
            row.value = value
        self.s.flush()


# -------------------- Feedback --------------------
class FeedbackRepo:
    def __init__(self, s: Session) -> None:
        self.s = s

    def create(self, user_id: int, type_: str, message: str) -> TelegramFeedback:
        row = TelegramFeedback(user_id=user_id, type=type_, message=message, status="open", created=utcnow())
        self.s.add(row); self.s.flush(); return row

    def list(self, type_: Optional[str] = None) -> Sequence[TelegramFeedback]:
        q = select(TelegramFeedback)
        if type_:
            q = q.where(TelegramFeedback.type == type_)
        return list(self.s.execute(q).scalars())

    def set_status(self, feedback_id: int, status: str) -> bool:
        res = self.s.execute(update(TelegramFeedback).where(TelegramFeedback.id == feedback_id).values(status=status))
        return (res.rowcount or 0) > 0


# -------------------- Broadcasts --------------------
class BroadcastRepo:
    def __init__(self, s: Session) -> None:
        self.s = s

    def create(self, message: str, sent_by: str, success_count: int = 0, total_count: int = 0) -> TelegramBroadcastLog:
        """Create a new broadcast log entry."""
        row = TelegramBroadcastLog(
            message=message,
            sent_by=sent_by,
            success_count=success_count,
            total_count=total_count,
            created_at=utcnow()
        )
        self.s.add(row)
        self.s.flush()
        return row

    def log(self, message: str, sent_by: str, success_count: int = 0, total_count: int = 0) -> TelegramBroadcastLog:
        """Legacy method - alias for create."""
        return self.create(message, sent_by, success_count, total_count)

    def list(self, limit: int = 50, offset: int = 0) -> List[TelegramBroadcastLog]:
        """Get broadcast history with pagination."""
        q = (
            select(TelegramBroadcastLog)
            .order_by(TelegramBroadcastLog.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(self.s.execute(q).scalars().all())

    def stats(self) -> Dict[str, Any]:
        """Get broadcast statistics."""
        # Total broadcasts
        total_q = select(func.count(TelegramBroadcastLog.id))
        total_broadcasts = self.s.execute(total_q).scalar_one() or 0

        # Total recipients and successful deliveries
        recipients_q = select(
            func.sum(TelegramBroadcastLog.total_count),
            func.sum(TelegramBroadcastLog.success_count)
        )
        result = self.s.execute(recipients_q).first()
        total_recipients = result[0] or 0
        successful_deliveries = result[1] or 0

        # Recent activity (last 24 hours)
        from datetime import datetime, timedelta
        yesterday = datetime.utcnow() - timedelta(days=1)
        recent_q = select(func.count(TelegramBroadcastLog.id)).where(
            TelegramBroadcastLog.created_at >= yesterday
        )
        recent_broadcasts = self.s.execute(recent_q).scalar_one() or 0

        return {
            "total_broadcasts": total_broadcasts,
            "total_recipients": total_recipients,
            "successful_deliveries": successful_deliveries,
            "failed_deliveries": total_recipients - successful_deliveries,
            "recent_broadcasts_24h": recent_broadcasts,
            "average_delivery_rate": (successful_deliveries / total_recipients * 100) if total_recipients > 0 else 0
        }


# -------------------- Verification --------------------
class VerificationRepo:
    def __init__(self, s: Session) -> None:
        self.s = s

    def issue(self, user_id: int, *, code: str, sent_time: int) -> TelegramVerificationCode:
        row = TelegramVerificationCode(user_id=user_id, code=code, sent_time=sent_time)
        self.s.add(row); self.s.flush(); return row

    def count_last_hour_by_user_id(self, user_id: int, now_unix: int) -> int:
        cutoff = now_unix - 3600
        q = select(func.count(TelegramVerificationCode.id)).where(
            TelegramVerificationCode.user_id == user_id,
            TelegramVerificationCode.sent_time >= cutoff,
        )
        return int(self.s.execute(q).scalar_one() or 0)


# -------------------- Command audit --------------------
class CommandAuditRepo:
    def __init__(self, s: Session) -> None:
        self.s = s

    def log(self, telegram_user_id: str, command: str, **kwargs) -> TelegramCommandAudit:
        row = TelegramCommandAudit(
            telegram_user_id=telegram_user_id,
            command=command,
            full_message=kwargs.get("full_message"),
            is_registered_user=bool(kwargs.get("is_registered_user")),
            user_email=kwargs.get("user_email"),
            success=bool(kwargs.get("success")),
            error_message=kwargs.get("error_message"),
            response_time_ms=int(kwargs.get("response_time_ms") or 0),
            created_at=utcnow(),
        )
        self.s.add(row); self.s.flush(); return row

    def last_commands(self, telegram_user_id: str, *, limit: int = 20) -> Sequence[TelegramCommandAudit]:
        q = (
            select(TelegramCommandAudit)
            .where(TelegramCommandAudit.telegram_user_id == telegram_user_id)
            .order_by(TelegramCommandAudit.id.desc())
            .limit(limit)
        )
        return list(self.s.execute(q).scalars())

    def list(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        user_id: Optional[str] = None,
        command: Optional[str] = None,
        success_only: Optional[bool] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Sequence[TelegramCommandAudit]:
        q = select(TelegramCommandAudit)
        if user_id:
            q = q.where(TelegramCommandAudit.telegram_user_id == user_id)
        if command:
            q = q.where(TelegramCommandAudit.command == command)
        if success_only:
            q = q.where(TelegramCommandAudit.success.is_(True))
        if start_date:
            q = q.where(TelegramCommandAudit.created >= start_date)
        if end_date:
            q = q.where(TelegramCommandAudit.created <= end_date)
        q = q.order_by(TelegramCommandAudit.id.desc()).offset(offset).limit(limit)
        return list(self.s.execute(q).scalars())

    def stats(self) -> dict:
        total = int(self.s.execute(select(func.count(TelegramCommandAudit.id))).scalar_one() or 0)
        rows = self.s.execute(
            select(TelegramCommandAudit.command, func.count(TelegramCommandAudit.id)).group_by(TelegramCommandAudit.command)
        ).all()
        success = int(self.s.execute(select(func.count(TelegramCommandAudit.id)).where(TelegramCommandAudit.success.is_(True))).scalar_one() or 0)
        return {"total": total, "by_command": {cmd: int(cnt) for cmd, cnt in rows}, "success_rate": (success / total) if total else None}

    def unique_users_summary(self) -> list[dict]:
        rows = self.s.execute(select(TelegramCommandAudit.telegram_user_id).distinct()).all()
        return [{"telegram_user_id": r[0]} for r in rows]
