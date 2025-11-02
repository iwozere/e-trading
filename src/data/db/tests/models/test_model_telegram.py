from src.data.db.models.model_telegram import TelegramBroadcastLog, TelegramCommandAudit, TelegramSetting


def test_telegram_models_basic():
    b = TelegramBroadcastLog()
    b.message = "hi"
    b.sent_by = "bot"
    assert b.message == "hi"

    ca = TelegramCommandAudit()
    ca.telegram_user_id = "123"
    ca.command = "/start"
    assert ca.command == "/start"

    s = TelegramSetting()
    s.key = "opt"
    s.value = "v"
    assert s.key == "opt"
