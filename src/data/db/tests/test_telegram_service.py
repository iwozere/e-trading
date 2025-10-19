from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

import time
import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

# services under test
from src.data.db.services import database_service as ds
from src.data.db.services import telegram_service as tgs

# models for table creation / optional asserts
from src.data.db.models.model_users import Base as UsersBase, User, AuthIdentity
from src.data.db.models.model_telegram import Base as TelegramBase


@pytest.fixture()
def setup_inmemory_db(monkeypatch):
    """
    Same approach as test_trading_service.py:
      - in-memory SQLite + FK ON
      - monkeypatch database_service engine/SessionLocal
      - create only the bases we need here: Users + Telegram
    """
    eng = create_engine("sqlite+pysqlite:///:memory:", future=True)

    @event.listens_for(eng, "connect")
    def _fk_on(dbapi_con, _):
        cur = dbapi_con.cursor()
        cur.execute("PRAGMA foreign_keys=ON;")
        cur.close()

    SessionLocal = sessionmaker(bind=eng, autoflush=False, autocommit=False, future=True, expire_on_commit=False)

    monkeypatch.setattr(ds, "engine", eng, raising=False)
    monkeypatch.setattr(ds, "SessionLocal", SessionLocal, raising=False)
    ds._db_service_singleton = None  # reset singleton

    # Create the necessary tables
    UsersBase.metadata.create_all(eng)
    TelegramBase.metadata.create_all(eng)

    yield eng, SessionLocal


# ----------------------------- helpers for tests -----------------------------

def _seed_user(SessionLocal, telegram_user_id: str, *, email: Optional[str] = None,
               approved: Optional[bool] = None, verified: Optional[bool] = None):
    """
    Create a user + telegram identity with optional metadata.
    UsersRepo.ensure_user_for_telegram() is used implicitly by service calls,
    but for list_users() test we need a deterministic email/metadata preset.
    """
    s = SessionLocal()
    try:
        # minimal user
        u = User(email=email or None)
        s.add(u); s.flush()
        # telegram identity
        ident = AuthIdentity(
            user_id=u.id,
            provider="telegram",
            external_id=str(telegram_user_id),
            identity_metadata={
                "approved": approved,
                "verified": verified,
            }
        )
        s.add(ident); s.flush()
        s.commit()
        return u.id
    finally:
        s.close()


# ----------------------------------- tests -----------------------------------

def test_get_user_status_and_limits_flow(setup_inmemory_db):
    eng, SessionLocal = setup_inmemory_db
    tg_id = "859865894"

    # Initially, there is no identity -> None
    assert tgs.get_user_status(tg_id) is None

    # Setting a limit should ensure user/identity exist
    tgs.set_user_limit(tg_id, "max_alerts", 5)
    status = tgs.get_user_status(tg_id)
    assert status is not None
    assert status.get("max_alerts") == 5
    # Other optional fields should be present (possibly None/False)
    assert "approved" in status and "verified" in status and "email" in status


def test_verification_flow(setup_inmemory_db, monkeypatch):
    eng, SessionLocal = setup_inmemory_db
    tg_id = "user123"

    now_unix = int(time.time())
    # Issue code
    tgs.set_verification_code(tg_id, code="999999", sent_time=now_unix)

    # Should count >=1 within last hour
    cnt = tgs.count_codes_last_hour(tg_id, now_unix=now_unix)
    assert isinstance(cnt, int) and cnt >= 1

    # Sanity: if we fake "now" 2 hours later, count should be 0
    cnt_late = tgs.count_codes_last_hour(tg_id, now_unix=now_unix + 2 * 3600)
    assert cnt_late == 0


def test_alerts_crud(setup_inmemory_db):
    _, _ = setup_inmemory_db
    tg_id = "tgA"

    # Create JSON-based alert
    alert_id = tgs.add_json_alert(
        tg_id,
        config_json='{"ticker":"AAPL","rule":{"price_above":170}}',
        email=True,
        status="ARMED",
        re_arm_config='{"rearm_on_cross_below":170}',
    )
    assert isinstance(alert_id, int)

    all_alerts = tgs.list_alerts(tg_id)
    assert len(all_alerts) == 1
    cfg = json.loads(all_alerts[0].config_json or "{}")
    assert cfg.get("ticker") == "AAPL"

    # Update alert
    ok = tgs.update_alert(alert_id, status="TRIGGERED", trigger_count=1)
    assert ok is True
    a = tgs.get_alert(alert_id)
    assert a and a.status == "TRIGGERED" and a.trigger_count == 1

    # Delete alert
    ok = tgs.delete_alert(alert_id)
    assert ok is True
    assert tgs.list_alerts(tg_id) == []


def test_schedules_crud(setup_inmemory_db):
    _, _ = setup_inmemory_db
    tg_id = "tgSched"

    sched_id = tgs.add_schedule(tg_id, ticker="MSFT", scheduled_time="08:30", schedule_config="daily")
    assert isinstance(sched_id, int)

    rows = tgs.list_schedules(tg_id)
    assert len(rows) == 1 and rows[0]['ticker'] == "MSFT"

    # Update
    assert tgs.update_schedule(sched_id, scheduled_time="09:00") is True
    row = tgs.get_schedule(sched_id)
    assert row and row['scheduled_time'] == "09:00"

    # Delete
    assert tgs.delete_schedule(sched_id) is True
    assert tgs.list_schedules(tg_id) == []


def test_settings_feedback_and_audit(setup_inmemory_db):
    _, _ = setup_inmemory_db
    tg_id = "tgF"

    # Settings
    assert tgs.get_setting("market_open") is None
    tgs.set_setting("market_open", "09:30")
    assert tgs.get_setting("market_open") == "09:30"
    tgs.set_setting("market_open", None)
    assert tgs.get_setting("market_open") is None

    # Feedback
    fid = tgs.add_feedback(tg_id, "bug", "Something is wrong")
    assert isinstance(fid, int)
    fb_list = tgs.list_feedback()
    assert any(f.id == fid for f in fb_list)
    assert tgs.update_feedback_status(fid, "closed") is True

    # Command audit
    aid = tgs.log_command_audit(tg_id, "/help", full_message="/help", is_registered_user=True, success=True)
    assert isinstance(aid, int)
    hist = tgs.get_user_command_history(tg_id, limit=5)
    assert len(hist) >= 1
    all_rows = tgs.get_all_command_audit(limit=10)
    assert isinstance(all_rows, list) and len(all_rows) >= 1
    stats = tgs.get_command_audit_stats()
    assert "total" in stats and "by_command" in stats


def test_list_users_compact_dto(setup_inmemory_db):
    eng, SessionLocal = setup_inmemory_db
    # Prepare identities so list_users() has something to return
    _seed_user(SessionLocal, "100", email="a@example.com", approved=True, verified=True)
    _seed_user(SessionLocal, "200", email="b@example.com", approved=False, verified=False)

    rows = tgs.list_users()
    # Should be a compact DTO list
    assert isinstance(rows, list)
    assert all(isinstance(x, dict) for x in rows)
    sample = rows[0]
    assert set(sample.keys()) >= {"telegram_user_id", "email", "approved", "verified"}
