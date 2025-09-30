# model_telegram.py  (aligned to DB)

from __future__ import annotations
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import (
    String, Integer, Boolean, DateTime, Text, ForeignKey, Numeric, Float, Index, text
)
from src.data.db.core.base import Base

# telegram_alerts
class TelegramAlert(Base):
    __tablename__ = "telegram_alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))

    # ARMED, TRIGGERED, INACTIVE
    status: Mapped[str | None] = mapped_column(String(50))

    # Should we notify user by email?
    email: Mapped[bool | None] = mapped_column(Boolean)
    created_at: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True))
    config_json: Mapped[str | None] = mapped_column(Text)
    re_arm_config: Mapped[str | None] = mapped_column(Text)
    state_json: Mapped[str | None] = mapped_column(Text)
    trigger_count: Mapped[int | None] = mapped_column(Integer)

    # Price, indicators values etc. at the time, when it was triggered last time
    last_trigger_condition: Mapped[str | None] = mapped_column(Text)
    last_triggered_at: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("ix_telegram_alerts_user_id", "user_id"),
    )

# telegram_broadcast_logs
class TelegramBroadcastLog(Base):
    __tablename__ = "telegram_broadcast_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    message: Mapped[str] = mapped_column(Text)
    sent_by: Mapped[str] = mapped_column(String(255))
    success_count: Mapped[int | None] = mapped_column(Integer)
    total_count: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True))

# telegram_command_audits
class TelegramCommandAudit(Base):
    __tablename__ = "telegram_command_audits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    telegram_user_id: Mapped[str] = mapped_column(String(255))
    command: Mapped[str] = mapped_column(String(255))
    full_message: Mapped[str | None] = mapped_column(Text)
    is_registered_user: Mapped[bool | None] = mapped_column(Boolean)
    user_email: Mapped[str | None] = mapped_column(String(255))
    success: Mapped[bool | None] = mapped_column(Boolean)
    error_message: Mapped[str | None] = mapped_column(Text)
    response_time_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("ix_telegram_command_audits_telegram_user_id", "telegram_user_id"),
        Index("ix_telegram_command_audits_success", "success"),
        Index("ix_telegram_command_audits_command", "command"),
        Index("ix_telegram_command_audits_created", "created_at"),
    )

# telegram_feedbacks
class TelegramFeedback(Base):
    __tablename__ = "telegram_feedbacks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    type: Mapped[str] = mapped_column(String(50))
    message: Mapped[str] = mapped_column(Text)
    created: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str | None] = mapped_column(String(20))

# telegram_schedules
class TelegramSchedule(Base):
    __tablename__ = "telegram_schedules"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str | None] = mapped_column(String(50))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    scheduled_time: Mapped[str] = mapped_column(String(20))
    period: Mapped[str | None] = mapped_column(String(20))
    active: Mapped[bool | None] = mapped_column(Boolean)
    email: Mapped[bool | None] = mapped_column(Boolean)
    indicators: Mapped[str | None] = mapped_column(Text)
    interval: Mapped[str | None] = mapped_column(String(10))
    provider: Mapped[str | None] = mapped_column(String(20))
    created: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True))
    schedule_type: Mapped[str | None] = mapped_column(String(20))
    list_type: Mapped[str | None] = mapped_column(String(50))
    config_json: Mapped[str | None] = mapped_column(Text)
    schedule_config: Mapped[str | None] = mapped_column(String(20))

    __table_args__ = (
        Index("ix_telegram_schedules_user_id", "user_id"),
    )

# telegram_settings
class TelegramSetting(Base):
    __tablename__ = "telegram_settings"
    key: Mapped[str] = mapped_column("key", String(100), primary_key=True)
    value: Mapped[str | None] = mapped_column(Text)

# telegram_verification_codes
class TelegramVerificationCode(Base):
    __tablename__ = "telegram_verification_codes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    code: Mapped[str] = mapped_column(String(32))
    sent_time: Mapped[int] = mapped_column(Integer)
