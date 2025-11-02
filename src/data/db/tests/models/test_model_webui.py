from src.data.db.models.model_webui import WebUIAuditLog, WebUIStrategyTemplate


def test_webui_audit_and_template_basic():
    w = WebUIAuditLog()
    w.user_id = 1
    w.action = "login"
    w.resource_type = "session"
    assert w.action == "login"

    t = WebUIStrategyTemplate()
    t.name = "template1"
    t.template_data = {"k": "v"}
    assert t.name == "template1"
