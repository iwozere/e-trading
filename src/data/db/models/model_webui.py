# model_webui.py  (aligned to DB)

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Integer, DateTime, JSON, ForeignKey, Index, text, Boolean
from src.data.db.core.base import Base

class WebUIAuditLog(Base):
    __tablename__ = "webui_audit_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Drop ondelete to match DB
    user_id: Mapped[int] = mapped_column(ForeignKey("usr_users.id"))
    action: Mapped[str] = mapped_column(String(100))
    resource_type: Mapped[str | None] = mapped_column(String(50))
    resource_id: Mapped[str | None] = mapped_column(String(100))
    details: Mapped[dict | None] = mapped_column(JSON)
    ip_address: Mapped[str | None] = mapped_column(String(45))
    user_agent: Mapped[str | None] = mapped_column(String(500))
    created_at: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))

    __table_args__ = (
        Index("ix_webui_audit_logs_user_id", "user_id"),
        Index("ix_webui_audit_logs_action", "action"),
    )

class WebUIPerformanceSnapshot(Base):
    __tablename__ = "webui_performance_snapshots"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    strategy_id: Mapped[str] = mapped_column(String(100))
    timestamp: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))
    pnl: Mapped[dict] = mapped_column(JSON)
    positions: Mapped[dict | None] = mapped_column(JSON)
    trades_count: Mapped[int | None] = mapped_column(Integer, server_default=text("0"))
    win_rate: Mapped[dict | None] = mapped_column(JSON)
    drawdown: Mapped[dict | None] = mapped_column(JSON)
    metrics: Mapped[dict | None] = mapped_column(JSON)

    __table_args__ = (
        Index("ix_webui_performance_snapshots_strategy_id", "strategy_id"),
    )

class WebUIStrategyTemplate(Base):
    __tablename__ = "webui_strategy_templates"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    description: Mapped[str | None]
    template_data: Mapped[dict] = mapped_column(JSON)
    is_public: Mapped[bool | None] = mapped_column(Boolean, server_default=text("FALSE"))
    created_by: Mapped[int] = mapped_column(ForeignKey("usr_users.id"))
    created_at: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))
    updated_at: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("ix_webui_strategy_templates_created_by", "created_by"),
    )

class WebUISystemConfig(Base):
    __tablename__ = "webui_system_config"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key: Mapped[str] = mapped_column(String(100), unique=True)  # remove index=True to match DB
    value: Mapped[dict] = mapped_column(JSON)
    description: Mapped[str | None]
    created_at: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))
    updated_at: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True))
