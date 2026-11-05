# model_webui.py  (aligned to DB)

from sqlalchemy import Boolean, DateTime, ForeignKey, Identity, Index, Integer, String, Text, func, text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql.elements import conv

from src.data.db.core.base import Base
from src.data.db.core.json_types import JsonType


class WebUIAuditLog(Base):
    __tablename__ = "webui_audit_logs"
    id: Mapped[int] = mapped_column(Integer, Identity(always=True), primary_key=True)
    # Drop ondelete to match DB
    user_id: Mapped[int] = mapped_column(ForeignKey("usr_users.id", name=conv("webui_audit_logs_user_id_fkey")))
    action: Mapped[str] = mapped_column(String(100))
    resource_type: Mapped[str | None] = mapped_column(String(50))
    resource_id: Mapped[str | None] = mapped_column(String(100))
    details: Mapped[dict | None] = mapped_column(JsonType())
    ip_address: Mapped[str | None] = mapped_column(String(45))
    user_agent: Mapped[str | None] = mapped_column(String(500))
    created_at: Mapped[DateTime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("ix_webui_audit_logs_user_id", "user_id"),
        Index("ix_webui_audit_logs_action", "action"),
    )


class WebUIPerformanceSnapshot(Base):
    __tablename__ = "webui_performance_snapshots"
    id: Mapped[int] = mapped_column(Integer, Identity(always=True), primary_key=True)
    strategy_id: Mapped[str] = mapped_column(String(100))
    timestamp: Mapped[DateTime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    pnl: Mapped[dict] = mapped_column(JsonType())
    positions: Mapped[dict | None] = mapped_column(JsonType())
    trades_count: Mapped[int | None] = mapped_column(Integer, server_default=text("0"))
    win_rate: Mapped[dict | None] = mapped_column(JsonType())
    drawdown: Mapped[dict | None] = mapped_column(JsonType())
    metrics: Mapped[dict | None] = mapped_column(JsonType())

    __table_args__ = (Index("ix_webui_performance_snapshots_strategy_id", "strategy_id"),)


class WebUIStrategyTemplate(Base):
    __tablename__ = "webui_strategy_templates"
    id: Mapped[int] = mapped_column(Integer, Identity(always=True), primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    description: Mapped[str | None] = mapped_column(Text)
    template_data: Mapped[dict] = mapped_column(JsonType())
    is_public: Mapped[bool | None] = mapped_column(Boolean, server_default=text("FALSE"))
    created_by: Mapped[int] = mapped_column(ForeignKey("usr_users.id", name=conv("webui_strategy_templates_created_by_fkey")))
    created_at: Mapped[DateTime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[DateTime | None] = mapped_column(DateTime())  # timestamp WITHOUT time zone in prod

    __table_args__ = (Index("ix_webui_strategy_templates_created_by", "created_by"),)


class WebUISystemConfig(Base):
    __tablename__ = "webui_system_config"
    id: Mapped[int] = mapped_column(Integer, Identity(always=True), primary_key=True)
    key: Mapped[str] = mapped_column(String(100), unique=True)  # remove index=True to match DB
    value: Mapped[dict] = mapped_column(JsonType())
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[DateTime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[DateTime | None] = mapped_column(DateTime())  # timestamp WITHOUT time zone in prod
