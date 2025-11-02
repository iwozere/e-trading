from __future__ import annotations
from sqlalchemy.orm import Session

from src.data.db.repos.repo_telegram import SettingsRepo, FeedbackRepo, BroadcastRepo, CommandAuditRepo
from src.data.db.repos.repo_users import UsersRepo


def test_settings_and_feedback_and_broadcasts(db_session: Session):
    users = UsersRepo(db_session)
    u = users.ensure_user_for_telegram("7070", defaults_user={"email": "tg@example.com"})

    # Settings
    settings = SettingsRepo(db_session)
    settings.set("welcome_message", "Hello!")
    assert settings.get("welcome_message").value == "Hello!"

    # Feedback
    fb = FeedbackRepo(db_session)
    row = fb.create(u.id, "bug", "It broke")
    assert row.id is not None
    assert any(x.id == row.id for x in fb.list("bug"))
    assert fb.set_status(row.id, "closed") is True

    # Broadcasts
    br = BroadcastRepo(db_session)
    log = br.create("msg", sent_by="tester", success_count=3, total_count=4)
    assert log.id is not None
    lst = br.list(limit=10)
    assert any(x.id == log.id for x in lst)
    stats = br.stats()
    assert stats["total_broadcasts"] >= 1


def test_command_audit(db_session: Session):
    ca = CommandAuditRepo(db_session)
    a1 = ca.log("u123", "start", success=True)
    a2 = ca.log("u123", "help", success=False)
    assert a1.id and a2.id
    last = ca.last_commands("u123", limit=10)
    assert len(last) >= 2
    listing = ca.list(user_id="u123", limit=5)
    assert len(listing) >= 2
    s = ca.stats()
    assert s["total"] >= 2
    uniq = ca.unique_users_summary()
    assert any(r["telegram_user_id"] == "u123" for r in uniq)
