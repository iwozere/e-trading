from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

import pytest
from sqlalchemy import create_engine, event, select, MetaData
from sqlalchemy.orm import sessionmaker

from src.data.db.models.model_webui import Base as WebUIBase, WebUIAuditLog, WebUIPerformanceSnapshot
from src.data.db.models.model_users import Base as UsersBase, User
from src.data.db.repos.repo_webui import (
    AuditRepo, SnapshotRepo, StrategyTemplateRepo, SystemConfigRepo
)

# ---------------- in-memory DB with FKs + tables (users + webui) ----------------
@pytest.fixture()
def engine():
    eng = create_engine("sqlite+pysqlite:///:memory:", future=True)
    @event.listens_for(eng, "connect")
    def _fk_on(dbapi_con, _):
        cur = dbapi_con.cursor()
        cur.execute("PRAGMA foreign_keys=ON;")
        cur.close()
    # create Users + WebUI together so FKs resolve
    merged = MetaData()
    for md in (UsersBase.metadata, WebUIBase.metadata):
        for t in md.tables.values():
            t.to_metadata(merged)
    merged.create_all(eng)
    return eng

@pytest.fixture()
def dbsess(engine):
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    s = Session()
    # Insert a dummy user to satisfy FK for audit logs
    u = User(email="tester@example.com", role="trader", is_active=True)
    s.add(u); s.flush()
    try:
        yield s
        s.rollback()
    finally:
        s.close()

@pytest.fixture()
def repos(dbsess):
    return {
        "audit": AuditRepo(dbsess),
        "snaps": SnapshotRepo(dbsess),
        "templates": StrategyTemplateRepo(dbsess),
        "cfg": SystemConfigRepo(dbsess),
        "session": dbsess,
    }

# ------------------------------- SystemConfigRepo ------------------------------
def test_system_config_set_get(repos):
    cfg = repos["cfg"]

    assert cfg.get("theme") is None
    row = cfg.set("theme", {"dark": True})
    assert row.key == "theme" and row.value == {"dark": True}

    # update path
    row2 = cfg.set("theme", {"dark": False}, description="toggle")
    got = cfg.get("theme")
    assert got and got.value == {"dark": False} and got.description == "toggle"

# -------------------------------- StrategyTemplateRepo -------------------------
def test_strategy_templates(repos):
    templates = repos["templates"]
    sess = repos["session"]

    # author = the dummy user we inserted in fixture (id should be 1)
    author_id = sess.execute(select(User.id).limit(1)).scalar_one()

    t1 = templates.create({"name": "SMA Crossover", "template_data": {"fast": 10, "slow": 20}, "created_by": author_id})
    t2 = templates.create({"name": "RSI Revert", "template_data": {"rsi": 30}, "created_by": author_id})
    # another author
    other = User(email="other@example.com", role="trader", is_active=True)
    sess.add(other); sess.flush()
    templates.create({"name": "Other Strat", "template_data": {}, "created_by": other.id})

    mine = templates.by_author(author_id)
    names = {t.name for t in mine}
    assert names == {"SMA Crossover", "RSI Revert"}
    assert all(t.created_by == author_id for t in mine)

# --------------------------------- SnapshotRepo --------------------------------
def test_performance_snapshots_latest(repos):
    snaps = repos["snaps"]
    sess = repos["session"]

    s1 = snaps.add({"strategy_id": "str-1", "pnl": {"day": 1}})
    s2 = snaps.add({"strategy_id": "str-1", "pnl": {"day": 2}})
    s3 = snaps.add({"strategy_id": "str-1", "pnl": {"day": 3}})

    latest_two = snaps.latest("str-1", limit=2)
    ids = [row.id for row in latest_two]
    # Expect most recent first (by timestamp desc)
    assert ids == [s3.id, s2.id]

    # Sanity: they’re actually in the table
    count = sess.execute(select(WebUIPerformanceSnapshot).where(WebUIPerformanceSnapshot.strategy_id == "str-1")).scalars().all()
    assert len(count) == 3

# ---------------------------------- AuditRepo ----------------------------------
def test_audit_log(repos):
    audit = repos["audit"]
    sess = repos["session"]

    user_id = sess.execute(select(User.id).limit(1)).scalar_one()
    row = audit.log(user_id=user_id, action="login", resource_type=None, resource_id=None, details={"ip": "127.0.0.1"})
    assert row.id is not None and row.action == "login" and row.user_id == user_id

    # Verify persisted fields
    stored = sess.execute(select(WebUIAuditLog).where(WebUIAuditLog.id == row.id)).scalar_one()
    assert stored.details == {"ip": "127.0.0.1"}

if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-q", "-rA"]))