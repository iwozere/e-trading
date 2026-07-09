from __future__ import annotations

import time

from sqlalchemy.orm import Session

from src.data.db.repos.repo_users import UsersRepo, VerificationRepo
from src.data.db.tests.repos.conftest import ensure_user_for_telegram


def test_users_repo_telegram_flow(db_session: Session):
    r = UsersRepo(db_session)

    # ensure user by telegram id
    u = ensure_user_for_telegram(r, "1001", defaults_user={"email": "alice@example.com"})
    assert u.id > 0
    assert u.email == "alice@example.com"

    # profile update and fetch (metadata merge, as UsersService.update_telegram_profile does)
    ident = r.get_identity(provider="telegram", external_id="1001")
    assert ident is not None
    meta = dict(ident.identity_metadata or {})
    meta.update({"verified": True, "approved": False, "language": "en", "is_admin": True})
    ident.identity_metadata = meta
    r.create_identity(ident)
    profile = r.get_telegram_profile("1001")
    assert profile is not None
    assert profile["user_id"] == u.id
    assert profile["verified"] is True
    assert profile["approved"] is False
    assert profile["language"] == "en"
    assert profile["is_admin"] is True

    # listings
    all_users = r.list_telegram_users_dto()
    assert any(row["telegram_user_id"] == "1001" for row in all_users)
    pending = r.list_pending_telegram_approvals()
    assert any(row["telegram_user_id"] == "1001" for row in pending)
    admins = r.get_admin_telegram_user_ids()
    assert "1001" in admins


def test_verification_repo_issue_and_count(db_session: Session):
    urepo = UsersRepo(db_session)
    user = ensure_user_for_telegram(urepo, "2002", defaults_user={"email": "bob@example.com"})

    vrepo = VerificationRepo(db_session)
    now = int(time.time())
    v1 = vrepo.issue(user.id, code="ABC123", sent_time=now - 10)
    v2 = vrepo.issue(user.id, code="XYZ999", sent_time=now - 300)
    assert v1.id and v2.id

    count = vrepo.count_last_hour_by_user_id(user.id, now_unix=now)
    assert count >= 2
