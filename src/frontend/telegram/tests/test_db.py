import os
import sqlite3
import pytest
import time
from src.data.db import telegram_service as db_module
from datetime import datetime

test_db_path = "test_telegram_screener.sqlite3"

@pytest.fixture(autouse=True)
def setup_and_teardown_db(monkeypatch):
    # Patch DB_PATH to use a test file
    monkeypatch.setattr(db_module, "DB_PATH", test_db_path)
    db_module.init_db()
    yield
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

def test_user_crud():
    telegram_user_id = "testuser"
    email = "test@example.com"
    code = "123456"
    sent_time = int(time.time())
    db_module.set_user_email(telegram_user_id, email, code, sent_time)
    status = db_module.get_user_status(telegram_user_id)
    assert status["email"] == email
    assert not status["verified"]
    assert db_module.verify_code(telegram_user_id, code)
    status = db_module.get_user_status(telegram_user_id)
    assert status["verified"]

def test_alerts_crud():
    user_id = "user1"
    ticker = "AAPL"
    price = 150.0
    condition = "above"
    alert_id = db_module.add_alert(user_id, ticker, price, condition)
    alert = db_module.get_alert(alert_id)
    assert alert["ticker"] == ticker
    assert alert["price"] == price
    assert alert["condition"] == condition
    alerts = db_module.list_alerts(user_id)
    assert any(a["id"] == alert_id for a in alerts)
    db_module.update_alert(alert_id, price=200.0, active=0)
    alert = db_module.get_alert(alert_id)
    assert alert["price"] == 200.0
    assert alert["active"] == 0
    db_module.delete_alert(alert_id)
    assert db_module.get_alert(alert_id) is None

def test_schedules_crud():
    user_id = "user2"
    ticker = "BTCUSDT"
    scheduled_time = "09:00"
    period = "daily"
    email = 1
    indicators = "RSI,MACD"
    interval = "1d"
    provider = "bnc"
    schedule_id = db_module.add_schedule(user_id, ticker, scheduled_time, period, email, indicators, interval, provider)
    schedule = db_module.get_schedule(schedule_id)
    assert schedule["ticker"] == ticker
    assert schedule["scheduled_time"] == scheduled_time
    assert schedule["period"] == period
    assert schedule["email"] == email
    assert schedule["indicators"] == indicators
    assert schedule["interval"] == interval
    assert schedule["provider"] == provider
    schedules = db_module.list_schedules(user_id)
    assert any(s["id"] == schedule_id for s in schedules)
    db_module.update_schedule(schedule_id, period="weekly", active=0)
    schedule = db_module.get_schedule(schedule_id)
    assert schedule["period"] == "weekly"
    assert schedule["active"] == 0
    db_module.delete_schedule(schedule_id)
    assert db_module.get_schedule(schedule_id) is None

def test_user_language_and_admin():
    telegram_user_id = "adminuser"
    email = "admin@example.com"
    code = "654321"
    sent_time = int(time.time())
    # Register as admin with language 'ru'
    db_module.set_user_email(telegram_user_id, email, code, sent_time, language="ru", is_admin=True)
    status = db_module.get_user_status(telegram_user_id)
    assert status["email"] == email
    assert status["language"] == "ru"
    assert status["is_admin"] is True
    # Update language
    conn = sqlite3.connect(db_module.DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE users SET language=? WHERE telegram_user_id=?", ("en", telegram_user_id))
    conn.commit()
    conn.close()
    status = db_module.get_user_status(telegram_user_id)
    assert status["language"] == "en"
    # Register as non-admin
    db_module.set_user_email("normaluser", "normal@example.com", "111111", sent_time, language="en", is_admin=False)
    status = db_module.get_user_status("normaluser")
    assert status["is_admin"] is False

def test_settings_and_limits():
    # Test global settings
    db_module.set_setting("max_alerts", "10")
    db_module.set_setting("max_schedules", "5")
    assert db_module.get_setting("max_alerts") == "10"
    assert db_module.get_setting("max_schedules") == "5"
    # Test per-user limits
    telegram_user_id = "limituser"
    email = "limit@example.com"
    code = "222222"
    sent_time = int(time.time())
    db_module.set_user_email(telegram_user_id, email, code, sent_time)
    db_module.set_user_limit(telegram_user_id, "max_alerts", 7)
    db_module.set_user_limit(telegram_user_id, "max_schedules", 3)
    assert db_module.get_user_limit(telegram_user_id, "max_alerts") == 7
    assert db_module.get_user_limit(telegram_user_id, "max_schedules") == 3
    # Test user listing
    users = db_module.list_users()
    assert any(u[0] == telegram_user_id and u[1] == email for u in users)
