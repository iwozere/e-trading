from __future__ import annotations
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from src.data.db.repos.repo_webui import AuditRepo, SnapshotRepo, StrategyTemplateRepo, SystemConfigRepo
from src.data.db.repos.repo_users import UsersRepo


UTC = timezone.utc


def test_webui_repos(db_session: Session):
    users = UsersRepo(db_session)
    u = users.ensure_user_for_telegram("9090", defaults_user={"email": "web@example.com"})

    # audit log
    audits = AuditRepo(db_session)
    a = audits.log(u.id, "login", resource_type="user", resource_id=str(u.id), details={"ip": "127.0.0.1"})
    assert a.id is not None

    # performance snapshots
    snaps = SnapshotRepo(db_session)
    s1 = snaps.add({"strategy_id": "s1", "timestamp": datetime.now(UTC), "pnl": {"net": 0}})
    s2 = snaps.add({"strategy_id": "s1", "pnl": {"net": 1}})
    latest = snaps.latest("s1", limit=2)
    assert len(latest) >= 2

    # strategy templates
    st = StrategyTemplateRepo(db_session)
    t = st.create({
        "name": "basic",
        "description": "desc",
        "template_data": {"k": 1},
        "created_by": u.id,
    })
    assert t.id is not None
    mine = st.by_author(u.id)
    assert any(x.id == t.id for x in mine)

    # system config
    sc = SystemConfigRepo(db_session)
    sc.set("feature_x", {"enabled": True}, description="flag")
    cfg = sc.get("feature_x")
    assert cfg and cfg.value.get("enabled") is True
