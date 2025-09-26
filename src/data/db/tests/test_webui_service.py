from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker

# services
from src.data.db.services import database_service as ds
from src.data.db.services.webui_service import (
    set_config, get_config,
    create_template, get_templates_by_author,
    add_snapshot, latest_snapshots,
    audit_log,
)

# models
from src.data.db.models.model_users import Base as UsersBase, User
from src.data.db.models.model_webui import Base as WebUIBase, WebUIAuditLog, WebUIPerformanceSnapshot

@pytest.fixture()
def setup_inmemory_db(monkeypatch):
    # In-memory engine with FKs ON
    eng = create_engine("sqlite+pysqlite:///:memory:", future=True)
    @event.listens_for(eng, "connect")
    def _fk_on(dbapi_con, _):
        cur = dbapi_con.cursor()
        cur.execute("PRAGMA foreign_keys=ON;")
        cur.close()
    SessionLocal = sessionmaker(bind=eng, autoflush=False, autocommit=False, future=True)

    # Patch database_service
    monkeypatch.setattr(ds, "engine", eng, raising=False)
    monkeypatch.setattr(ds, "SessionLocal", SessionLocal, raising=False)
    ds._db_service_singleton = None

    # Create tables together so FKs resolve
    from sqlalchemy import MetaData
    merged = MetaData()
    for md in (UsersBase.metadata, WebUIBase.metadata):
        for t in md.tables.values():
            t.to_metadata(merged)
    merged.create_all(eng)

    # Insert a dummy user for audit FK
    s = SessionLocal()
    u = User(email="tester@example.com", role="trader", is_active=True)
    s.add(u); s.flush()
    user_id = u.id
    s.commit(); s.close()

    yield eng, SessionLocal, user_id


def test_system_config_and_templates(setup_inmemory_db):
    eng, SessionLocal, user_id = setup_inmemory_db

    # System config
    out = set_config("theme", {"dark": True})
    assert out["key"] == "theme" and out["value"] == {"dark": True}
    out2 = set_config("theme", {"dark": False}, description="toggle")
    cfg = get_config("theme")
    assert cfg and cfg["value"] == {"dark": False} and cfg["description"] == "toggle"

    # Templates
    t1 = create_template({"name": "SMA", "template_data": {"fast": 10}, "created_by": user_id})
    t2 = create_template({"name": "RSI", "template_data": {"rsi": 30}, "created_by": user_id})

    # Another author
    s = SessionLocal()
    other = User(email="other@example.com", role="trader", is_active=True)
    s.add(other); s.flush()
    create_template({"name": "Other", "template_data": {}, "created_by": other.id})
    s.close()

    mine = get_templates_by_author(user_id)
    names = {t["name"] for t in mine}
    assert names == {"SMA", "RSI"}


def test_snapshots_and_audit(setup_inmemory_db):
    eng, SessionLocal, user_id = setup_inmemory_db

    # Snapshots
    s1 = add_snapshot({"strategy_id": "str-1", "pnl": {"d": 1}})
    s2 = add_snapshot({"strategy_id": "str-1", "pnl": {"d": 2}})
    s3 = add_snapshot({"strategy_id": "str-1", "pnl": {"d": 3}})

    latest = latest_snapshots("str-1", limit=2)
    assert [x["id"] for x in latest] == [s3["id"], s2["id"]]

    # Audit log
    log_id = audit_log(user_id=user_id, action="login", details={"ip": "127.0.0.1"})
    assert isinstance(log_id, int)

    # Optional: verify persisted via direct query
    SessionLocal2 = SessionLocal
    s = SessionLocal2()
    try:
        row = s.execute(select(WebUIAuditLog).where(WebUIAuditLog.id == log_id)).scalar_one()
        assert row.details == {"ip": "127.0.0.1"}
        count = s.execute(select(WebUIPerformanceSnapshot).where(WebUIPerformanceSnapshot.strategy_id == "str-1")).scalars().all()
        assert len(count) == 3
    finally:
        s.close()

# ----------------------------- Negative cases ---------------------------------
def test_audit_log_fk_violation(setup_inmemory_db):
    # FK: user_id must exist in users table
    from sqlalchemy.exc import IntegrityError
    from src.data.db.services.webui_service import audit_log

    _, _, _user_id = setup_inmemory_db
    with pytest.raises(IntegrityError):
        audit_log(user_id=999999, action="login")  # non-existent user


def test_create_template_missing_author(setup_inmemory_db):
    # created_by is NOT NULL -> None should violate constraint
    from sqlalchemy.exc import IntegrityError
    from src.data.db.services.webui_service import create_template

    with pytest.raises(IntegrityError):
        create_template({"name": "NoAuthor", "template_data": {}, "created_by": None})


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-q", "-rA"]))