# src/data/db/repos/repo_telegram.py
from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional, Sequence, List, Dict, Any
from sqlalchemy import select, update, func
from sqlalchemy.orm import Session

from src.data.db.models.model_telegram import (
    TelegramSetting,
    TelegramFeedback,
    TelegramCommandAudit,
    TelegramBroadcastLog,
)

UTC = timezone.utc
utcnow = lambda: datetime.now(UTC)



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
        row = TelegramFeedback(user_id=user_id, type=type_, message=message, status="open", created_at=utcnow())
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
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
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
