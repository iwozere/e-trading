from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError

from src.data.db.services.database_service import database_service as ds
from src.data.db.models.model_users import Base as UsersBase, User
from src.data.db.models.model_webui import Base as WebUIBase, WebUIStrategyTemplate, WebUIPerformanceSnapshot


@pytest.fixture()
def setup_inmemory_db(monkeypatch):
    eng = create_engine("sqlite+pysqlite:///:memory:", future=True)
    @event.listens_for(eng, "connect")
    def _fk_on(dbapi_con, _):
        cur = dbapi_con.cursor()
        cur.execute("PRAGMA foreign_keys=ON;")
        cur.close()
    SessionLocal = sessionmaker(bind=eng, autoflush=False, autocommit=False, future=True)

    # patch DS to use in-memory engine
    monkeypatch.setattr(ds, "engine", eng, raising=False)
    monkeypatch.setattr(ds, "SessionLocal", SessionLocal, raising=False)
    ds._db_service_singleton = None

    # create only users + webui tables
    UsersBase.metadata.create_all(eng)
    WebUIBase.metadata.create_all(eng)

    # seed one valid user
    s = SessionLocal()
    u = User(email="valid@example.com", role="trader", is_active=True)
    s.add(u); s.flush()
    user_id = u.id
    s.commit(); s.close()

    yield eng, SessionLocal, user_id


def test_uow_rollback_across_repos(setup_inmemory_db):
    eng, SessionLocal, user_id = setup_inmemory_db
    db = ds.get_database_service()

    # Attempt a multi-repo transaction: create template + snapshot, then cause FK error -> rollback all
    with pytest.raises(IntegrityError):
        with db.uow() as r:
            r.webui_templates.create({"name": "Temp1", "template_data": {"a": 1}, "created_by": user_id})
            r.webui_snapshots.add({"strategy_id": "str-rollback", "pnl": {"x": 1}})
            # trigger failure (FK violation)
            r.webui_audit.log(user_id=999999, action="oops")  # non-existent user -> IntegrityError

    # Verify rollback: nothing persisted
    s = SessionLocal()
    try:
        n_templates = s.execute(select(WebUIStrategyTemplate)).scalars().all()
        n_snaps = s.execute(select(WebUIPerformanceSnapshot)).scalars().all()
        assert n_templates == []
        assert n_snaps == []
    finally:
        s.close()


def test_uow_commit_success(setup_inmemory_db):
    eng, SessionLocal, user_id = setup_inmemory_db
    db = ds.get_database_service()

    with db.uow() as r:
        r.webui_templates.create({"name": "TempOK", "template_data": {"ok": True}, "created_by": user_id})
        r.webui_snapshots.add({"strategy_id": "str-ok", "pnl": {"p": 1}})
        r.webui_audit.log(user_id=user_id, action="ok")  # valid

    # Everything committed
    s = SessionLocal()
    try:
        t = s.execute(select(WebUIStrategyTemplate).where(WebUIStrategyTemplate.name == "TempOK")).scalar_one_or_none()
        snaps = s.execute(select(WebUIPerformanceSnapshot).where(WebUIPerformanceSnapshot.strategy_id == "str-ok")).scalars().all()
        assert t is not None
        assert len(snaps) == 1
    finally:
        s.close()


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-q", "-rA"]))