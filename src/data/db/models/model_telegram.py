# model_telegram.py  (aligned to DB)

from __future__ import annotations
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import (
    String, Integer, Boolean, DateTime, Text, ForeignKey, Numeric, Float, Index, func
)
from src.data.db.core.base import Base

# telegram_broadcast_logs
class TelegramBroadcastLog(Base):
    __tablename__ = "telegram_broadcast_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    message: Mapped[str] = mapped_column(Text)
    sent_by: Mapped[str] = mapped_column(String(255))
    success_count: Mapped[int | None] = mapped_column(Integer)
    total_count: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True), default=func.now())

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
    created_at: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True), default=func.now())

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
    user_id: Mapped[int] = mapped_column(ForeignKey("usr_users.id", ondelete="CASCADE"))
    type: Mapped[str | None] = mapped_column(String(50))
    message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True), default=func.now())
    status: Mapped[str | None] = mapped_column(String(20))

# telegram_settings
class TelegramSetting(Base):
    __tablename__ = "telegram_settings"
    key: Mapped[str] = mapped_column("key", String(100), primary_key=True)
    value: Mapped[str | None] = mapped_column(Text)

