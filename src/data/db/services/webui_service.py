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

from src.data.db.services.database_service import get_database_service

# Resolve the singleton once per module; avoids repeated lookups.
def _db():
    # resolve dynamically so tests' monkeypatch (engine/SessionLocal) is honored
    return get_database_service()

# ---------- System Config ----------
def set_config(key: str, value: Dict[str, Any], description: Optional[str] = None) -> Dict[str, Any]:
    with _db().uow() as r:
        row = r.webui_config.set(key, value, description=description)
        return _config_to_dict(row)

def get_config(key: str) -> Optional[Dict[str, Any]]:
    with _db().uow() as r:
        row = r.webui_config.get(key)
        return _config_to_dict(row) if row else None


# ---------- Strategy Templates ----------
def create_template(template_data: Dict[str, Any]) -> Dict[str, Any]:
    with _db().uow() as r:
        row = r.webui_templates.create(template_data)
        return _template_to_dict(row)

def get_templates_by_author(user_id: int) -> List[Dict[str, Any]]:
    with _db().uow() as r:
        rows = r.webui_templates.by_author(user_id)
        return [_template_to_dict(t) for t in rows]


# ---------- Performance Snapshots ----------
def add_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    with _db().uow() as r:
        row = r.webui_snapshots.add(snapshot)
        return _snapshot_to_dict(row)

def latest_snapshots(strategy_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    with _db().uow() as r:
        rows = r.webui_snapshots.latest(strategy_id, limit=limit)
        return [_snapshot_to_dict(s) for s in rows]


# ---------- Audit Logs ----------
def audit_log(
    user_id: int,
    action: str,
    *,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> int:
    with _db().uow() as r:
        row = r.webui_audit.log(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        return int(row.id)


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
