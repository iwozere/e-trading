"""
WebUI Service
-------------
Thin service over WebUI repos (audit logs, performance snapshots,
strategy templates, system config). Returns plain dicts/primitives.

All operations use a single Unit-of-Work:
    with db.uow() as r:
        ... r.webui_*
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

from src.data.db.services.base_service import BaseDBService, with_uow, handle_db_error

# ---------- DTO helpers ----------
def _config_to_dict(c) -> Dict[str, Any]:
    return {
        "id": c.id,
        "key": c.key,
        "value": c.value,
        "description": c.description,
        "created_at": c.created_at,
        "updated_at": c.updated_at,
    }

def _template_to_dict(t) -> Dict[str, Any]:
    return {
        "id": t.id,
        "name": t.name,
        "description": t.description,
        "template_data": t.template_data,
        "is_public": bool(t.is_public) if t.is_public is not None else False,
        "created_by": t.created_by,
        "created_at": t.created_at,
        "updated_at": t.updated_at,
    }

def _snapshot_to_dict(s) -> Dict[str, Any]:
    return {
        "id": s.id,
        "strategy_id": s.strategy_id,
        "timestamp": s.timestamp,
        "pnl": s.pnl,
        "positions": s.positions,
        "trades_count": s.trades_count,
        "win_rate": s.win_rate,
        "drawdown": s.drawdown,
        "metrics": s.metrics,
    }


class WebUIService(BaseDBService):
    """Service layer for Web UI related DB operations.

    Wraps webui repositories (config, templates, snapshots, audit) and
    exposes high-level methods. Methods are decorated to provide a Unit
    of Work and centralized DB error handling.
    """

    def __init__(self) -> None:
        super().__init__()

    # ---------- System Config ----------
    @with_uow
    @handle_db_error
    def set_config(self, key: str, value: Dict[str, Any], description: Optional[str] = None) -> Dict[str, Any]:
        row = self.repos.webui_config.set(key, value, description=description)
        return self._config_to_dict(row)

    @with_uow
    @handle_db_error
    def get_config(self, key: str) -> Optional[Dict[str, Any]]:
        row = self.repos.webui_config.get(key)
        return self._config_to_dict(row) if row else None

    # ---------- Strategy Templates ----------
    @with_uow
    @handle_db_error
    def create_template(self, template_data: Dict[str, Any]) -> Dict[str, Any]:
        row = self.repos.webui_templates.create(template_data)
        return self._template_to_dict(row)

    @with_uow
    @handle_db_error
    def get_templates_by_author(self, user_id: int) -> List[Dict[str, Any]]:
        rows = self.repos.webui_templates.by_author(user_id)
        return [self._template_to_dict(t) for t in rows]

    # ---------- Performance Snapshots ----------
    @with_uow
    @handle_db_error
    def add_snapshot(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        row = self.repos.webui_snapshots.add(snapshot)
        return self._snapshot_to_dict(row)

    @with_uow
    @handle_db_error
    def latest_snapshots(self, strategy_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        rows = self.repos.webui_snapshots.latest(strategy_id, limit=limit)
        return [self._snapshot_to_dict(s) for s in rows]

    # ---------- Audit Logs ----------
    @with_uow
    @handle_db_error
    def audit_log(
        self,
        user_id: int,
        action: str,
        *,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> int:
        row = self.repos.webui_audit.log(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        return int(row.id)

    # Keep DTO helpers as instance methods for convenience
    def _config_to_dict(self, c) -> Dict[str, Any]:
        return _config_to_dict(c)

    def _template_to_dict(self, t) -> Dict[str, Any]:
        return _template_to_dict(t)

    def _snapshot_to_dict(self, s) -> Dict[str, Any]:
        return _snapshot_to_dict(s)


# Global service instance
webui_service = WebUIService()
