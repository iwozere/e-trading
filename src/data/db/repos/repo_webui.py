from __future__ import annotations
from typing import Sequence
from sqlalchemy import select
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from src.data.db.models.model_webui import (
    WebUIAuditLog, WebUIPerformanceSnapshot, WebUIStrategyTemplate, WebUISystemConfig
)

UTC = timezone.utc

class AuditRepo:
    def __init__(self, s: Session) -> None:
        self.s = s

    def log(self, user_id: int, action: str, **kw) -> WebUIAuditLog:
        row = WebUIAuditLog(user_id=user_id, action=action, **kw)
        self.s.add(row); self.s.flush()
        return row


class SnapshotRepo:
    def __init__(self, s: Session) -> None:
        self.s = s

    def add(self, snapshot: dict) -> WebUIPerformanceSnapshot:
        # ensure a timestamp so ordering is deterministic
        if not snapshot.get("timestamp"):
            snapshot = {**snapshot, "timestamp": datetime.now(UTC)}
        row = WebUIPerformanceSnapshot(**snapshot)
        self.s.add(row); self.s.flush()
        return row

    def latest(self, strategy_id: str, limit: int = 50) -> Sequence[WebUIPerformanceSnapshot]:
        q = (
            select(WebUIPerformanceSnapshot)
            .where(WebUIPerformanceSnapshot.strategy_id == strategy_id)
            .order_by(
                WebUIPerformanceSnapshot.timestamp.desc(),
                WebUIPerformanceSnapshot.id.desc(),
            )
            .limit(limit)
        )
        return list(self.s.execute(q).scalars())


class StrategyTemplateRepo:
    def __init__(self, s: Session) -> None:
        self.s = s

    def create(self, t: dict) -> WebUIStrategyTemplate:
        row = WebUIStrategyTemplate(**t)
        self.s.add(row); self.s.flush()
        return row

    def by_author(self, user_id: int) -> Sequence[WebUIStrategyTemplate]:
        q = select(WebUIStrategyTemplate).where(WebUIStrategyTemplate.created_by == user_id)
        return list(self.s.execute(q).scalars())


class SystemConfigRepo:
    def __init__(self, s: Session) -> None:
        self.s = s

    def get(self, key: str) -> WebUISystemConfig | None:
        q = select(WebUISystemConfig).where(WebUISystemConfig.key == key)
        return self.s.execute(q).scalar_one_or_none()

    def set(self, key: str, value: dict, description: str | None = None) -> WebUISystemConfig:
        row = self.s.execute(select(WebUISystemConfig).where(WebUISystemConfig.key == key)).scalar_one_or_none()
        if row:
            row.value = value
            if description is not None:
                row.description = description
            return row
        row = WebUISystemConfig(key=key, value=value, description=description)
        self.s.add(row); self.s.flush()
        return row
