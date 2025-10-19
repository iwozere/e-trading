from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

import time
from datetime import datetime, timezone
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

# Models (create only what we need)
from src.data.db.models.model_users import Base as UsersBase, User
from src.data.db.models.model_telegram import Base as TelegramBase, TelegramSetting

# Repos under test (only import ones that work with existing models)
from src.data.db.repos.repo_telegram import (
    SettingsRepo,
    FeedbackRepo,
    BroadcastRepo,
    CommandAuditRepo,
)
from src.data.db.repos.repo_users import VerificationRepo


# ------------------------ In-memory DB (FKs ON) + tables ------------------------
@pytest.fixture()
def engine():
    eng = create_engine("sqlite+pysqlite:///:memory:", future=True)

    @event.listens_for(eng, "connect")
    def _fk_on(dbapi_con, _):
        cur = dbapi_con.cursor()
        cur.execute("PRAGMA foreign_keys=ON;")
        cur.close()

    # Create minimal schema for these repos
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


# --------------------------------- helpers ---------------------------------
def _mk_user(dbsess, email: str = None) -> int:
    """Create a minimal user and return user_id."""
    u = User(email=email)
    dbsess.add(u)
    dbsess.flush()
    return u.id


# -------------------------------- AlertsRepo --------------------------------
# NOTE: AlertsRepo is not yet implemented in repo_telegram.py - skipping test
@pytest.mark.skip(reason="AlertsRepo not implemented yet")
def test_alerts_crud_and_filters(dbsess):
    uid = _mk_user(dbsess, "a@example.com")
    alerts = AlertsRepo(dbsess)

    # create JSON-based alert
    a = alerts.create(
        uid,
        config_json='{"ticker":"AAPL","rule":{"price_above":170}}',
        email=True,
        status="ARMED",
        re_arm_config='{"rearm_on_cross_below":170}',
    )
    assert a.id is not None and a.status == "ARMED" and a.trigger_count == 0

    # get + list_for_user
    got = alerts.get(a.id)
    assert got and got.id == a.id
    my = alerts.list_for_user(uid)
    assert any(x.id == a.id for x in my)

    # list_by_status (ARMED)
    armed = alerts.list_by_status("ARMED")
    assert any(x.id == a.id for x in armed)

    # update -> triggered once, then deactivate
    ok = alerts.update(
        a.id,
        status="TRIGGERED",
        trigger_count=1,
        last_trigger_condition='{"price":171.0}',
        last_triggered_at=datetime.now(timezone.utc),
    )
    assert ok is True
    still = alerts.get(a.id)
    assert still and still.status == "TRIGGERED" and still.trigger_count == 1

    ok = alerts.update(a.id, status="INACTIVE")
    assert ok is True
    assert alerts.get(a.id).status == "INACTIVE"

    # delete
    alerts.delete(a.id)
    assert alerts.get(a.id) is None


# ------------------------------- SchedulesRepo -------------------------------
# NOTE: SchedulesRepo is not yet implemented in repo_telegram.py - skipping test
@pytest.mark.skip(reason="SchedulesRepo not implemented yet")
def test_schedules_upsert_update_delete(dbsess):
    uid = _mk_user(dbsess, "s@example.com")
    scheds = SchedulesRepo(dbsess)

    # upsert with alias schedule_time (maps to scheduled_time) and ensure defaulting works
    s1 = scheds.upsert({
        "user_id": uid,
        "ticker": "MSFT",
        "scheduled_time": "08:30",  # alias accepted
        "period": "daily",
        "email": True,
        "interval": "1h",
        "provider": "yfinance",
        "indicators": "rsi",
    })
    assert s1.id is not None and s1.scheduled_time == "08:30"

    # upsert with missing scheduled_time -> defaults to "09:00"
    s2 = scheds.upsert({"user_id": uid, "ticker": "AAPL"})
    assert s2.scheduled_time == "09:00"

    # list_for_user
    mine = scheds.list_for_user(uid)
    ids = {x.id for x in mine}
    assert {s1.id, s2.id} <= ids

    # Update and assert persisted state (don't rely on rowcount True/False)
    _ = scheds.update(s1.id, scheduled_time="09:15", active=False)
    got = scheds.get(s1.id)
    assert got is not None
    assert got.scheduled_time == "09:15"
    assert got.active is False

    # list_by_config (none set above; ensure it’s empty)
    assert scheds.list_by_config("advanced") == []

    # delete
    scheds.delete(s2.id)
    assert scheds.get(s2.id) is None


# ------------------------------- SettingsRepo --------------------------------
def test_settings_set_and_get(dbsess):
    settings = SettingsRepo(dbsess)

    assert settings.get("theme") is None
    settings.set("theme", "dark")
    row = settings.get("theme")
    assert isinstance(row, TelegramSetting) and row.value == "dark"

    # overwrite + set None (allowed)
    settings.set("theme", "light")
    assert settings.get("theme").value == "light"
    settings.set("theme", None)
    assert settings.get("theme").value is None


# ------------------------------- FeedbackRepo --------------------------------
def test_feedback_flow(dbsess):
    uid = _mk_user(dbsess, "f@example.com")
    fb = FeedbackRepo(dbsess)

    row = fb.create(uid, "bug", "something is wrong")
    assert row.id is not None and row.status == "open"

    items = fb.list("bug")
    assert any(x.id == row.id for x in items)

    ok = fb.set_status(row.id, "closed")
    assert ok is True


# ------------------------------ BroadcastRepo --------------------------------
def test_broadcast_log(dbsess):
    br = BroadcastRepo(dbsess)
    row = br.log("hello world", "tester", success_count=2, total_count=3)
    assert row.id is not None and row.message == "hello world" and row.success_count == 2 and row.total_count == 3


# ------------------------------ VerificationRepo -----------------------------
def test_verification_issue_and_count(dbsess):
    uid = _mk_user(dbsess, "v@example.com")
    ver = VerificationRepo(dbsess)

    now = int(time.time())
    old = now - 4000

    # issue 2 codes: one old, one recent
    ver.issue(uid, code="OLD", sent_time=old)
    ver.issue(uid, code="NEW", sent_time=now - 100)

    # only recent should be counted within last hour
    cnt = ver.count_last_hour_by_user_id(uid, now_unix=now)
    assert cnt == 1


# ------------------------------ CommandAuditRepo -----------------------------
def test_command_audit_log_list_stats(dbsess):
    ca = CommandAuditRepo(dbsess)

    ca.log("tg1", "/start", full_message="hi", is_registered_user=True, user_email="a@x.com", success=True, response_time_ms=100)
    ca.log("tg1", "/help", full_message="help", is_registered_user=False, user_email=None, success=False, response_time_ms=200)
    ca.log("tg2", "/start", full_message="hi2", is_registered_user=True, user_email="b@x.com", success=True, response_time_ms=150)

    # last_commands (ordered desc by id, limit)
    last = ca.last_commands("tg1", limit=1)
    assert len(last) == 1 and last[0].command in ("/start", "/help")

    # list with filters
    only_start = ca.list(limit=10, user_id="tg2", command="/start", success_only=True)
    assert len(only_start) == 1 and only_start[0].telegram_user_id == "tg2" and only_start[0].command == "/start"

    # stats + unique users
    stats = ca.stats()
    assert "total" in stats and "by_command" in stats and "success_rate" in stats
    uniq = ca.unique_users_summary()
    ids = {r["telegram_user_id"] for r in uniq}
    assert {"tg1", "tg2"} <= ids


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-q", "-rA"]))
