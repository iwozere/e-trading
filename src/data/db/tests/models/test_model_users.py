from datetime import datetime, timezone


from src.data.db.models.model_users import User, AuthIdentity, VerificationCode


def test_user_properties_and_methods():
    u = User()
    u.id = 1
    u.email = "alice@example.com"
    u.role = "trader"
    u.is_active = True
    u.created_at = datetime.now(timezone.utc)

    assert u.username == "alice"
    assert u.verify_password("alice") is True
    assert u.verify_password("password") is True
    assert u.verify_password("wrong") is False

    d = u.to_dict()
    assert d["email"] == "alice@example.com"
    assert d["username"] == "alice"


def test_auth_identity_metadata_get_set():
    ai = AuthIdentity()
    ai.identity_metadata = {"key": "value"}
    assert ai.meta_get("key") == "value"
    ai.meta_set("other", 123)
    assert ai.meta_get("other") == 123


def test_verification_code_basic_fields():
    vc = VerificationCode()
    vc.id = 1
    vc.user_id = 5
    vc.code = "ABC123"
    vc.sent_time = 123456
    assert vc.code == "ABC123"
    assert vc.user_id == 5
