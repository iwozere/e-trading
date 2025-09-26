from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

import time
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from src.data.db.models.model_users import Base as UsersBase
from src.data.db.models.model_telegram import Base as TelegramBase
from src.data.db.services.telegram_service import TelegramRepository


# ---------- local fixtures: in-memory DB with FKs ON, tables created ----------
@pytest.fixture()
def engine():
    eng = create_engine("sqlite+pysqlite:///:memory:", future=True)
    @event.listens_for(eng, "connect")
    def _fk_on(dbapi_con, _):
        cur = dbapi_con.cursor()
        cur.execute("PRAGMA foreign_keys=ON;")
        cur.close()
    # Only users + telegram bases needed for this repo
    UsersBase.metadata.create_all(eng)
    TelegramBase.metadata.create_all(eng)
    return eng

@pytest.fixture()
def dbsess(engine):
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    s = Session()
    try:
        yield s
        s.rollback()
    finally:
        s.close()

@pytest.fixture()
def repo(dbsess):
    return TelegramRepository(dbsess)


# ----------------------------- users / identity -------------------------------
def test_user_resolution_and_upsert(repo: TelegramRepository):
    # not created yet
    assert repo.get_user("tg_1") is None

    # upsert creates user + identity and sets fields
    repo.upsert_user("tg_1", email="u1@example.com", verified=True,
                     language="en", is_admin=True, max_alerts=5, max_schedules=3)
    dto = repo.get_user("tg_1")
    assert dto and dto.email == "u1@example.com" and dto.verified is True
    assert dto.language == "en" and dto.is_admin is True
    assert dto.max_alerts == 5 and dto.max_schedules == 3

    # approval + status
    assert repo.approve_user("tg_1", True) is True
    status = repo.get_user_status("tg_1")
    assert status and status["approved"] is True

    # pending approvals: only users explicitly set to False should appear
    repo.upsert_user("tg_pending", email="p@example.com", approved=False)
    pending = repo.list_pending_approvals()
    ids = {p["telegram_user_id"] for p in pending}
    assert "tg_pending" in ids

    # admins list
    admins = repo.get_admin_user_ids()
    assert "tg_1" in admins

    # limits update
    repo.set_user_limit("tg_1", "max_alerts", 10)
    dto2 = repo.get_user("tg_1")
    assert dto2 and dto2.max_alerts == 10


# ----------------------------- verification codes ----------------------------
def test_verification_codes_rate_limit(repo: TelegramRepository):
    now = int(time.time())
    # set an old code (older than 1h) and a recent one
    repo.set_verification_code("tg_rate", code="OLD", sent_time=now - 4000)
    repo.set_verification_code("tg_rate", code="NEW", sent_time=now - 100)
    # count should include only the recent one
    assert repo.count_codes_last_hour("tg_rate") == 1
    # user DTO reflects the *last* code set
    dto = repo.get_user("tg_rate")
    assert dto and dto.verification_code == "NEW" and dto.code_sent_time == now - 100


# ---------------------------------- alerts -----------------------------------
def test_alerts_crud_and_filters(repo: TelegramRepository):
    # Ensure user exists
    repo.upsert_user("tg_alerts", email="a@b.com")

    # Price alert
    aid = repo.add_alert("tg_alerts", "AAPL", 200.0, "above", email=True)
    a = repo.get_alert(aid)
    assert a and a.ticker == "AAPL" and a.alert_type == "price" and a.active is True

    # Update & list
    assert repo.update_alert(aid, active=False) is True
    my_alerts = repo.list_alerts("tg_alerts")
    assert any(x.id == aid for x in my_alerts)

    # Active filter excludes it now
    only_active = repo.get_active_alerts()
    assert all(x.id != aid for x in only_active)

    # Indicator alert
    iid = repo.add_indicator_alert("tg_alerts", "MSFT", "{}", alert_action="notify", timeframe="15m", email=False)
    ind = repo.get_alert(iid)
    assert ind and ind.alert_type == "indicator" and ind.is_armed is True

    # By type
    types = repo.get_alerts_by_type("indicator")
    assert any(x.id == iid for x in types)

    # Delete
    assert repo.delete_alert(aid) is True
    remaining = repo.list_alerts("tg_alerts")
    assert all(x.id != aid for x in remaining)


# -------------------------------- schedules ----------------------------------
def test_schedules_crud_and_filters(repo: TelegramRepository):
    repo.upsert_user("tg_sched", email="s@x.com")

    sid_plain = repo.add_schedule("tg_sched", "AAPL", "09:30", period="daily",
                                  email=True, indicators="rsi", interval="1h", provider="yfinance")
    sid_json = repo.add_json_schedule("tg_sched", "{}", schedule_config="advanced")
    sid_via_dict = repo.create_schedule({
        "user_id": "tg_sched", "ticker": "NVDA", "scheduled_time": "10:00",
        "period": "weekly", "email": False, "indicators": None, "interval": "4h",
        "provider": "alpaca", "schedule_type": "plain", "list_type": None,
        "config_json": None, "schedule_config": None
    })

    # list + get
    all_scheds = repo.list_schedules("tg_sched")
    assert {s.id for s in all_scheds} == {sid_plain, sid_json, sid_via_dict}
    got = repo.get_schedule(sid_plain)
    assert got and got.ticker == "AAPL"

    # update & active filter
    assert repo.update_schedule(sid_plain, active=False) is True
    active = repo.get_active_schedules()
    assert all(s.id != sid_plain for s in active)

    # by config: only the JSON one
    advanced_only = repo.get_schedules_by_config("advanced")
    assert len(advanced_only) == 1 and advanced_only[0].id == sid_json

    # delete
    assert repo.delete_schedule(sid_via_dict) is True
    left = repo.list_schedules("tg_sched")
    assert len(left) == 2


# -------------------------------- settings -----------------------------------
def test_settings_set_and_get(repo: TelegramRepository):
    assert repo.get_setting("theme") is None
    repo.set_setting("theme", "dark")
    assert repo.get_setting("theme") == "dark"
    # update path
    repo.set_setting("theme", "light")
    assert repo.get_setting("theme") == "light"


# -------------------------------- feedback -----------------------------------
def test_feedback_flow(repo: TelegramRepository):
    repo.upsert_user("tg_fb", email="fb@x.com")
    fid = repo.add_feedback("tg_fb", "bug", "something broke")
    items = repo.list_feedback("bug")
    assert any(f.id == fid for f in items)
    assert repo.update_feedback_status(fid, "closed") is True


# ------------------------------ command audit --------------------------------
def test_command_audit(repo: TelegramRepository):
    # two commands for tg_ca, one success, one fail
    repo.log_command_audit("tg_ca", "/start", full_message="hello", is_registered_user=True,
                           user_email="u@x.com", success=True, error_message=None, response_time_ms=100)
    repo.log_command_audit("tg_ca", "/help", full_message="help", is_registered_user=False,
                           user_email=None, success=False, error_message="err", response_time_ms=200)

    # user history (limit 1 -> last only)
    hist = repo.get_user_command_history("tg_ca", limit=1)
    assert len(hist) == 1 and hist[0].command in ("/help", "/start")

    # filtered list
    all_cmds = repo.get_all_command_audit(limit=10, user_id="tg_ca", command="/start", success_only=True)
    assert all(c.command == "/start" and c.success for c in all_cmds)

    # stats
    stats = repo.get_command_audit_stats()
    assert stats["total"] >= 2 and "by_command" in stats and stats["success_rate"] is not None

    # unique users summary (we only used one tg id here)
    uniq = repo.get_unique_users_command_history()
    assert any(row["telegram_user_id"] == "tg_ca" for row in uniq)


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-q", "-rA"]))