import pytest
import time
from unittest.mock import Mock

@pytest.fixture
def mock_telegram_service():
    """Create a mock telegram service for testing."""
    mock_service = Mock()

    # Mock user data storage
    mock_service._users = {}
    mock_service._alerts = {}
    mock_service._schedules = {}
    mock_service._settings = {}
    mock_service._user_limits = {}

    # Mock service methods
    def mock_set_user_email(telegram_user_id, email, code, sent_time, language="en", is_admin=False):
        mock_service._users[telegram_user_id] = {
            "email": email,
            "code": code,
            "sent_time": sent_time,
            "verified": False,
            "language": language,
            "is_admin": is_admin
        }

    def mock_get_user_status(telegram_user_id):
        return mock_service._users.get(telegram_user_id)

    def mock_verify_code(telegram_user_id, code):
        user = mock_service._users.get(telegram_user_id)
        if user and user["code"] == code:
            user["verified"] = True
            return True
        return False

    def mock_add_alert(user_id, ticker, price, condition):
        alert_id = len(mock_service._alerts) + 1
        mock_service._alerts[alert_id] = {
            "id": alert_id,
            "user_id": user_id,
            "ticker": ticker,
            "price": price,
            "condition": condition,
            "active": 1
        }
        return alert_id

    def mock_get_alert(alert_id):
        return mock_service._alerts.get(alert_id)

    def mock_list_alerts(user_id):
        return [alert for alert in mock_service._alerts.values() if alert["user_id"] == user_id]

    def mock_update_alert(alert_id, **kwargs):
        if alert_id in mock_service._alerts:
            mock_service._alerts[alert_id].update(kwargs)

    def mock_delete_alert(alert_id):
        if alert_id in mock_service._alerts:
            del mock_service._alerts[alert_id]

    def mock_add_schedule(user_id, ticker, scheduled_time, period, email, indicators, interval, provider):
        schedule_id = len(mock_service._schedules) + 1
        mock_service._schedules[schedule_id] = {
            "id": schedule_id,
            "user_id": user_id,
            "ticker": ticker,
            "scheduled_time": scheduled_time,
            "period": period,
            "email": email,
            "indicators": indicators,
            "interval": interval,
            "provider": provider,
            "active": 1
        }
        return schedule_id

    def mock_get_schedule(schedule_id):
        return mock_service._schedules.get(schedule_id)

    def mock_list_schedules(user_id):
        return [schedule for schedule in mock_service._schedules.values() if schedule["user_id"] == user_id]

    def mock_update_schedule(schedule_id, **kwargs):
        if schedule_id in mock_service._schedules:
            mock_service._schedules[schedule_id].update(kwargs)

    def mock_delete_schedule(schedule_id):
        if schedule_id in mock_service._schedules:
            del mock_service._schedules[schedule_id]

    def mock_set_setting(key, value):
        mock_service._settings[key] = value

    def mock_get_setting(key):
        return mock_service._settings.get(key)

    def mock_set_user_limit(telegram_user_id, key, value):
        if telegram_user_id not in mock_service._user_limits:
            mock_service._user_limits[telegram_user_id] = {}
        mock_service._user_limits[telegram_user_id][key] = value

    def mock_get_user_limit(telegram_user_id, key):
        return mock_service._user_limits.get(telegram_user_id, {}).get(key)

    def mock_list_users():
        return [(user_id, user_data["email"]) for user_id, user_data in mock_service._users.items()]

    # Assign mock methods
    mock_service.set_user_email = mock_set_user_email
    mock_service.get_user_status = mock_get_user_status
    mock_service.verify_code = mock_verify_code
    mock_service.add_alert = mock_add_alert
    mock_service.get_alert = mock_get_alert
    mock_service.list_alerts = mock_list_alerts
    mock_service.update_alert = mock_update_alert
    mock_service.delete_alert = mock_delete_alert
    mock_service.add_schedule = mock_add_schedule
    mock_service.get_schedule = mock_get_schedule
    mock_service.list_schedules = mock_list_schedules
    mock_service.update_schedule = mock_update_schedule
    mock_service.delete_schedule = mock_delete_schedule
    mock_service.set_setting = mock_set_setting
    mock_service.get_setting = mock_get_setting
    mock_service.set_user_limit = mock_set_user_limit
    mock_service.get_user_limit = mock_get_user_limit
    mock_service.list_users = mock_list_users

    return mock_service

def test_user_crud(mock_telegram_service):
    telegram_user_id = "testuser"
    email = "test@example.com"
    code = "123456"
    sent_time = int(time.time())
    mock_telegram_service.set_user_email(telegram_user_id, email, code, sent_time)
    status = mock_telegram_service.get_user_status(telegram_user_id)
    assert status["email"] == email
    assert not status["verified"]
    assert mock_telegram_service.verify_code(telegram_user_id, code)
    status = mock_telegram_service.get_user_status(telegram_user_id)
    assert status["verified"]

def test_alerts_crud(mock_telegram_service):
    user_id = "user1"
    ticker = "AAPL"
    price = 150.0
    condition = "above"
    alert_id = mock_telegram_service.add_alert(user_id, ticker, price, condition)
    alert = mock_telegram_service.get_alert(alert_id)
    assert alert["ticker"] == ticker
    assert alert["price"] == price
    assert alert["condition"] == condition
    alerts = mock_telegram_service.list_alerts(user_id)
    assert any(a["id"] == alert_id for a in alerts)
    mock_telegram_service.update_alert(alert_id, price=200.0, active=0)
    alert = mock_telegram_service.get_alert(alert_id)
    assert alert["price"] == 200.0
    assert alert["active"] == 0
    mock_telegram_service.delete_alert(alert_id)
    assert mock_telegram_service.get_alert(alert_id) is None

def test_schedules_crud(mock_telegram_service):
    user_id = "user2"
    ticker = "BTCUSDT"
    scheduled_time = "09:00"
    period = "daily"
    email = 1
    indicators = "RSI,MACD"
    interval = "1d"
    provider = "bnc"
    schedule_id = mock_telegram_service.add_schedule(user_id, ticker, scheduled_time, period, email, indicators, interval, provider)
    schedule = mock_telegram_service.get_schedule(schedule_id)
    assert schedule["ticker"] == ticker
    assert schedule["scheduled_time"] == scheduled_time
    assert schedule["period"] == period
    assert schedule["email"] == email
    assert schedule["indicators"] == indicators
    assert schedule["interval"] == interval
    assert schedule["provider"] == provider
    schedules = mock_telegram_service.list_schedules(user_id)
    assert any(s["id"] == schedule_id for s in schedules)
    mock_telegram_service.update_schedule(schedule_id, period="weekly", active=0)
    schedule = mock_telegram_service.get_schedule(schedule_id)
    assert schedule["period"] == "weekly"
    assert schedule["active"] == 0
    mock_telegram_service.delete_schedule(schedule_id)
    assert mock_telegram_service.get_schedule(schedule_id) is None

def test_user_language_and_admin(mock_telegram_service):
    telegram_user_id = "adminuser"
    email = "admin@example.com"
    code = "654321"
    sent_time = int(time.time())
    # Register as admin with language 'ru'
    mock_telegram_service.set_user_email(telegram_user_id, email, code, sent_time, language="ru", is_admin=True)
    status = mock_telegram_service.get_user_status(telegram_user_id)
    assert status["email"] == email
    assert status["language"] == "ru"
    assert status["is_admin"] is True
    # Register as non-admin
    mock_telegram_service.set_user_email("normaluser", "normal@example.com", "111111", sent_time, language="en", is_admin=False)
    status = mock_telegram_service.get_user_status("normaluser")
    assert status["is_admin"] is False

def test_settings_and_limits(mock_telegram_service):
    # Test global settings
    mock_telegram_service.set_setting("max_alerts", "10")
    mock_telegram_service.set_setting("max_schedules", "5")
    assert mock_telegram_service.get_setting("max_alerts") == "10"
    assert mock_telegram_service.get_setting("max_schedules") == "5"
    # Test per-user limits
    telegram_user_id = "limituser"
    email = "limit@example.com"
    code = "222222"
    sent_time = int(time.time())
    mock_telegram_service.set_user_email(telegram_user_id, email, code, sent_time)
    mock_telegram_service.set_user_limit(telegram_user_id, "max_alerts", 7)
    mock_telegram_service.set_user_limit(telegram_user_id, "max_schedules", 3)
    assert mock_telegram_service.get_user_limit(telegram_user_id, "max_alerts") == 7
    assert mock_telegram_service.get_user_limit(telegram_user_id, "max_schedules") == 3
    # Test user listing
    users = mock_telegram_service.list_users()
    assert any(u[0] == telegram_user_id and u[1] == email for u in users)
