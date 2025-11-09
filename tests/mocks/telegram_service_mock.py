"""
Mock implementation of telegram_service for testing.

This mock provides all the methods used by TelegramBusinessLogic
with configurable responses and behavior tracking.
"""

from typing import Any, Dict, List, Optional
import time
import json

from src.notification.logger import setup_logger
_logger = setup_logger(__name__)


class TelegramServiceMock:
    """
    Mock implementation of telegram_service for testing.

    Provides configurable responses and tracks method calls for verification.
    """

    def __init__(self):
        """Initialize mock with default data and call tracking."""
        self.call_log = []
        self.users = {}
        self.alerts = {}
        self.schedules = {}
        self.settings = {}
        self.feedback = {}
        self.audit_logs = {}
        self.broadcast_logs = {}
        self.verification_codes = {}

        # Configuration for mock behavior
        self.should_raise_errors = {}
        self.custom_responses = {}
        self.rate_limits = {}

        # Default user data
        self.default_user = {
            "approved": True,
            "verified": True,
            "email": "test@example.com",
            "language": "en",
            "is_admin": False,
            "max_alerts": 10,
            "max_schedules": 5,
            "verification_code": None,
            "code_sent_time": None
        }

    def _log_call(self, method_name: str, *args, **kwargs):
        """Log method calls for verification in tests."""
        self.call_log.append({
            "method": method_name,
            "args": args,
            "kwargs": kwargs,
            "timestamp": time.time()
        })
        _logger.debug("Mock call: %s(*%s, **%s)", method_name, args, kwargs)

    def _check_error_config(self, method_name: str):
        """Check if method should raise an error based on configuration."""
        if method_name in self.should_raise_errors:
            error = self.should_raise_errors[method_name]
            if callable(error):
                raise error()
            else:
                raise error

    def _get_custom_response(self, method_name: str, default_response):
        """Get custom response if configured, otherwise return default."""
        return self.custom_responses.get(method_name, default_response)

    # User Management Methods

    def get_user_status(self, telegram_user_id: str) -> Optional[Dict[str, Any]]:
        """Get user status with mock data."""
        self._log_call("get_user_status", telegram_user_id)
        self._check_error_config("get_user_status")

        if telegram_user_id in self.users:
            return self._get_custom_response("get_user_status", self.users[telegram_user_id])
        else:
            # Return default user or None based on configuration
            default = self.default_user.copy() if "return_default_user" in self.custom_responses else None
            return self._get_custom_response("get_user_status", default)

    def get_all_users(self) -> List[Dict[str, Any]]:
        """Get all users for admin operations."""
        self._log_call("get_all_users")
        self._check_error_config("get_all_users")

        users_list = [{"telegram_user_id": uid, **data} for uid, data in self.users.items()]
        return self._get_custom_response("get_all_users", users_list)

    def list_users(self) -> List[Dict[str, Any]]:
        """List users for broadcast operations."""
        self._log_call("list_users")
        self._check_error_config("list_users")

        return self.get_all_users()

    def verify_user_email(self, telegram_user_id: str) -> bool:
        """Mark user email as verified."""
        self._log_call("verify_user_email", telegram_user_id)
        self._check_error_config("verify_user_email")

        if telegram_user_id in self.users:
            self.users[telegram_user_id]["verified"] = True
            self.users[telegram_user_id]["verification_code"] = None
            self.users[telegram_user_id]["code_sent_time"] = None
        else:
            # Create user if doesn't exist
            self.users[telegram_user_id] = self.default_user.copy()
            self.users[telegram_user_id]["verified"] = True

        return self._get_custom_response("verify_user_email", True)

    def approve_user(self, telegram_user_id: str) -> bool:
        """Approve user for restricted features."""
        self._log_call("approve_user", telegram_user_id)
        self._check_error_config("approve_user")

        if telegram_user_id in self.users:
            self.users[telegram_user_id]["approved"] = True
        else:
            self.users[telegram_user_id] = self.default_user.copy()
            self.users[telegram_user_id]["approved"] = True

        return self._get_custom_response("approve_user", True)

    def reject_user(self, telegram_user_id: str) -> bool:
        """Reject user approval request."""
        self._log_call("reject_user", telegram_user_id)
        self._check_error_config("reject_user")

        if telegram_user_id in self.users:
            self.users[telegram_user_id]["approved"] = False

        return self._get_custom_response("reject_user", True)

    def update_user_language(self, telegram_user_id: str, language: str) -> bool:
        """Update user language preference."""
        self._log_call("update_user_language", telegram_user_id, language)
        self._check_error_config("update_user_language")

        if telegram_user_id in self.users:
            self.users[telegram_user_id]["language"] = language
        else:
            self.users[telegram_user_id] = self.default_user.copy()
            self.users[telegram_user_id]["language"] = language

        return self._get_custom_response("update_user_language", True)

    def set_user_limit(self, telegram_user_id: str, key: str, value: int) -> None:
        """Set user limit for alerts or schedules."""
        self._log_call("set_user_limit", telegram_user_id, key, value)
        self._check_error_config("set_user_limit")

        if telegram_user_id not in self.users:
            self.users[telegram_user_id] = self.default_user.copy()

        self.users[telegram_user_id][key] = value

    def get_user_limit(self, telegram_user_id: str, key: str) -> Optional[int]:
        """Get user limit for a specific key."""
        self._log_call("get_user_limit", telegram_user_id, key)
        self._check_error_config("get_user_limit")

        if telegram_user_id in self.users:
            return self.users[telegram_user_id].get(key)
        return self._get_custom_response("get_user_limit", None)

    def set_user_daily_limit(self, telegram_user_id: str, limit: int) -> bool:
        """Set user daily limit."""
        self._log_call("set_user_daily_limit", telegram_user_id, limit)
        self._check_error_config("set_user_daily_limit")

        self.set_user_limit(telegram_user_id, "max_alerts", limit)
        return self._get_custom_response("set_user_daily_limit", True)

    # Verification Code Methods

    def set_verification_code(self, telegram_user_id: str, *, code: str, sent_time: int) -> None:
        """Set verification code for user."""
        self._log_call("set_verification_code", telegram_user_id, code=code, sent_time=sent_time)
        self._check_error_config("set_verification_code")

        if telegram_user_id not in self.users:
            self.users[telegram_user_id] = self.default_user.copy()

        self.users[telegram_user_id]["verification_code"] = code
        self.users[telegram_user_id]["code_sent_time"] = sent_time

        # Track for rate limiting
        if telegram_user_id not in self.verification_codes:
            self.verification_codes[telegram_user_id] = []
        self.verification_codes[telegram_user_id].append(sent_time)

    def count_codes_last_hour(self, telegram_user_id: str, now_unix: int = None) -> int:
        """Count verification codes sent in the last hour."""
        self._log_call("count_codes_last_hour", telegram_user_id, now_unix)
        self._check_error_config("count_codes_last_hour")

        if now_unix is None:
            now_unix = int(time.time())

        if telegram_user_id not in self.verification_codes:
            return 0

        hour_ago = now_unix - 3600
        recent_codes = [t for t in self.verification_codes[telegram_user_id] if t > hour_ago]

        return self._get_custom_response("count_codes_last_hour", len(recent_codes))

    # Alert Management Methods

    def add_alert(self, telegram_user_id: str, ticker: str, price: float, condition: str, email: bool = False) -> int:
        """Add a price alert."""
        self._log_call("add_alert", telegram_user_id, ticker, price, condition, email)
        self._check_error_config("add_alert")

        alert_id = len(self.alerts) + 1
        self.alerts[alert_id] = {
            "id": alert_id,
            "user_id": telegram_user_id,
            "ticker": ticker,
            "price": price,
            "condition": condition,
            "email": email,
            "status": "ARMED",
            "created_at": time.time(),
            "enabled": True
        }

        return self._get_custom_response("add_alert", alert_id)

    def add_json_alert(self, telegram_user_id: str, config_json: str, *, email: Optional[bool] = None,
                      status: str = "ARMED", re_arm_config: Optional[str] = None) -> int:
        """Add a JSON-configured alert."""
        self._log_call("add_json_alert", telegram_user_id, config_json, email=email, status=status, re_arm_config=re_arm_config)
        self._check_error_config("add_json_alert")

        alert_id = len(self.alerts) + 1
        config = json.loads(config_json) if isinstance(config_json, str) else config_json

        self.alerts[alert_id] = {
            "id": alert_id,
            "user_id": telegram_user_id,
            "config_json": config_json,
            "email": email,
            "status": status,
            "re_arm_config": re_arm_config,
            "created_at": time.time(),
            "enabled": status == "ARMED",
            "ticker": config.get("ticker", "unknown")
        }

        return self._get_custom_response("add_json_alert", alert_id)

    def list_alerts(self, telegram_user_id: str):
        """List alerts for a user."""
        self._log_call("list_alerts", telegram_user_id)
        self._check_error_config("list_alerts")

        user_alerts = [alert for alert in self.alerts.values() if alert["user_id"] == telegram_user_id]
        return self._get_custom_response("list_alerts", user_alerts)

    def get_alert(self, alert_id: int):
        """Get an alert by ID."""
        self._log_call("get_alert", alert_id)
        self._check_error_config("get_alert")

        alert = self.alerts.get(alert_id)
        return self._get_custom_response("get_alert", alert)

    def update_alert(self, alert_id: int, **values) -> bool:
        """Update an alert."""
        self._log_call("update_alert", alert_id, **values)
        self._check_error_config("update_alert")

        if alert_id in self.alerts:
            self.alerts[alert_id].update(values)
            return self._get_custom_response("update_alert", True)

        return self._get_custom_response("update_alert", False)

    def delete_alert(self, alert_id: int) -> bool:
        """Delete an alert."""
        self._log_call("delete_alert", alert_id)
        self._check_error_config("delete_alert")

        if alert_id in self.alerts:
            del self.alerts[alert_id]
            return self._get_custom_response("delete_alert", True)

        return self._get_custom_response("delete_alert", False)

    def list_active_alerts(self, telegram_user_id: str = None, *, limit: int = 100, offset: int = 0, older_first: bool = False):
        """List active alerts."""
        self._log_call("list_active_alerts", telegram_user_id, limit=limit, offset=offset, older_first=older_first)
        self._check_error_config("list_active_alerts")

        active_alerts = [alert for alert in self.alerts.values() if alert.get("enabled", True)]
        if telegram_user_id:
            active_alerts = [alert for alert in active_alerts if alert["user_id"] == telegram_user_id]

        return self._get_custom_response("list_active_alerts", active_alerts[offset:offset+limit])

    # Schedule Management Methods

    def add_schedule(self, telegram_user_id: str, ticker: str, scheduled_time: str, **kwargs) -> int:
        """Add a schedule."""
        self._log_call("add_schedule", telegram_user_id, ticker, scheduled_time, **kwargs)
        self._check_error_config("add_schedule")

        schedule_id = len(self.schedules) + 1
        self.schedules[schedule_id] = {
            "id": schedule_id,
            "user_id": telegram_user_id,
            "ticker": ticker,
            "scheduled_time": scheduled_time,
            "created_at": time.time(),
            "enabled": True,
            **kwargs
        }

        return self._get_custom_response("add_schedule", schedule_id)

    def add_json_schedule(self, telegram_user_id: str, config_json: str, *, schedule_config: Optional[str] = None) -> int:
        """Add a JSON-configured schedule."""
        self._log_call("add_json_schedule", telegram_user_id, config_json, schedule_config=schedule_config)
        self._check_error_config("add_json_schedule")

        schedule_id = len(self.schedules) + 1
        config = json.loads(config_json) if isinstance(config_json, str) else config_json

        self.schedules[schedule_id] = {
            "id": schedule_id,
            "user_id": telegram_user_id,
            "config_json": config_json,
            "schedule_config": schedule_config,
            "created_at": time.time(),
            "enabled": True,
            "ticker": config.get("ticker", "unknown")
        }

        return self._get_custom_response("add_json_schedule", schedule_id)

    def list_schedules(self, telegram_user_id: str):
        """List schedules for a user."""
        self._log_call("list_schedules", telegram_user_id)
        self._check_error_config("list_schedules")

        user_schedules = [schedule for schedule in self.schedules.values() if schedule["user_id"] == telegram_user_id]
        return self._get_custom_response("list_schedules", user_schedules)

    def get_schedule(self, schedule_id: int):
        """Get a schedule by ID."""
        self._log_call("get_schedule", schedule_id)
        self._check_error_config("get_schedule")

        schedule = self.schedules.get(schedule_id)
        return self._get_custom_response("get_schedule", schedule)

    def get_schedule_by_id(self, schedule_id: int):
        """Get a schedule by ID (alias)."""
        return self.get_schedule(schedule_id)

    def update_schedule(self, schedule_id: int, **values) -> bool:
        """Update a schedule."""
        self._log_call("update_schedule", schedule_id, **values)
        self._check_error_config("update_schedule")

        if schedule_id in self.schedules:
            self.schedules[schedule_id].update(values)
            return self._get_custom_response("update_schedule", True)

        return self._get_custom_response("update_schedule", False)

    def delete_schedule(self, schedule_id: int) -> bool:
        """Delete a schedule."""
        self._log_call("delete_schedule", schedule_id)
        self._check_error_config("delete_schedule")

        if schedule_id in self.schedules:
            del self.schedules[schedule_id]
            return self._get_custom_response("delete_schedule", True)

        return self._get_custom_response("delete_schedule", False)

    # Settings Management Methods

    def get_setting(self, key: str) -> Optional[str]:
        """Get a setting value."""
        self._log_call("get_setting", key)
        self._check_error_config("get_setting")

        return self._get_custom_response("get_setting", self.settings.get(key))

    def set_setting(self, key: str, value: Optional[str]) -> None:
        """Set a setting value."""
        self._log_call("set_setting", key, value)
        self._check_error_config("set_setting")

        if value is None:
            self.settings.pop(key, None)
        else:
            self.settings[key] = value

    # Feedback Methods

    def add_feedback(self, telegram_user_id: str, type_: str, message: str) -> int:
        """Add user feedback."""
        self._log_call("add_feedback", telegram_user_id, type_, message)
        self._check_error_config("add_feedback")

        feedback_id = len(self.feedback) + 1
        self.feedback[feedback_id] = {
            "id": feedback_id,
            "user_id": telegram_user_id,
            "type": type_,
            "message": message,
            "created_at": time.time(),
            "status": "pending"
        }

        return self._get_custom_response("add_feedback", feedback_id)

    def list_feedback(self, type_: Optional[str] = None):
        """List feedback entries."""
        self._log_call("list_feedback", type_)
        self._check_error_config("list_feedback")

        feedback_list = list(self.feedback.values())
        if type_:
            feedback_list = [f for f in feedback_list if f["type"] == type_]

        return self._get_custom_response("list_feedback", feedback_list)

    def update_feedback_status(self, feedback_id: int, status: str) -> bool:
        """Update feedback status."""
        self._log_call("update_feedback_status", feedback_id, status)
        self._check_error_config("update_feedback_status")

        if feedback_id in self.feedback:
            self.feedback[feedback_id]["status"] = status
            return self._get_custom_response("update_feedback_status", True)

        return self._get_custom_response("update_feedback_status", False)

    # Audit and Logging Methods

    def log_command_audit(self, telegram_user_id: str, command: str, **kwargs) -> int:
        """Log command audit entry."""
        self._log_call("log_command_audit", telegram_user_id, command, **kwargs)
        self._check_error_config("log_command_audit")

        audit_id = len(self.audit_logs) + 1
        self.audit_logs[audit_id] = {
            "id": audit_id,
            "telegram_user_id": telegram_user_id,
            "command": command,
            "created_at": time.time(),
            **kwargs
        }

        return self._get_custom_response("log_command_audit", audit_id)

    def get_user_command_history(self, telegram_user_id: str, limit: int = 20):
        """Get user command history."""
        self._log_call("get_user_command_history", telegram_user_id, limit)
        self._check_error_config("get_user_command_history")

        user_logs = [log for log in self.audit_logs.values() if log["telegram_user_id"] == telegram_user_id]
        user_logs.sort(key=lambda x: x["created_at"], reverse=True)

        return self._get_custom_response("get_user_command_history", user_logs[:limit])

    def log_broadcast(self, message: str, sent_by: str, success_count: int, total_count: int) -> int:
        """Log broadcast message."""
        self._log_call("log_broadcast", message, sent_by, success_count, total_count)
        self._check_error_config("log_broadcast")

        broadcast_id = len(self.broadcast_logs) + 1
        self.broadcast_logs[broadcast_id] = {
            "id": broadcast_id,
            "message": message,
            "sent_by": sent_by,
            "success_count": success_count,
            "total_count": total_count,
            "created_at": time.time()
        }

        return self._get_custom_response("log_broadcast", broadcast_id)

    # Test Helper Methods

    def configure_error(self, method_name: str, error: Exception):
        """Configure a method to raise an error."""
        self.should_raise_errors[method_name] = error

    def configure_response(self, method_name: str, response: Any):
        """Configure a custom response for a method."""
        self.custom_responses[method_name] = response

    def clear_errors(self):
        """Clear all error configurations."""
        self.should_raise_errors.clear()

    def clear_responses(self):
        """Clear all custom response configurations."""
        self.custom_responses.clear()

    def reset_mock(self):
        """Reset all mock data and configurations."""
        self.call_log.clear()
        self.users.clear()
        self.alerts.clear()
        self.schedules.clear()
        self.settings.clear()
        self.feedback.clear()
        self.audit_logs.clear()
        self.broadcast_logs.clear()
        self.verification_codes.clear()
        self.should_raise_errors.clear()
        self.custom_responses.clear()
        self.rate_limits.clear()

    def get_call_count(self, method_name: str) -> int:
        """Get the number of times a method was called."""
        return len([call for call in self.call_log if call["method"] == method_name])

    def get_last_call(self, method_name: str) -> Optional[Dict[str, Any]]:
        """Get the last call to a specific method."""
        calls = [call for call in self.call_log if call["method"] == method_name]
        return calls[-1] if calls else None

    def was_called_with(self, method_name: str, *args, **kwargs) -> bool:
        """Check if a method was called with specific arguments."""
        for call in self.call_log:
            if (call["method"] == method_name and
                call["args"] == args and
                call["kwargs"] == kwargs):
                return True
        return False

    def add_test_user(self, telegram_user_id: str, **user_data):
        """Add a test user with specific data."""
        user = self.default_user.copy()
        user.update(user_data)
        self.users[telegram_user_id] = user
        return user