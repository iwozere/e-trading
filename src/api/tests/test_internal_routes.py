"""
Tests for internal routes (log-alert forwarding from Vector).
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.internal_routes import receive_log_alert


def _make_request(body: dict, host: str = "127.0.0.1") -> MagicMock:
    """Build a duck-typed Request stand-in — the handler only touches
    request.client.host, request.headers, and request.body()."""
    request = MagicMock()
    request.client.host = host
    request.headers = {}
    request.body = AsyncMock(return_value=json.dumps(body).encode())
    return request


@pytest.mark.asyncio
async def test_log_alert_queues_only_telegram_linked_admins():
    """
    Regression: log-alert must resolve admins via the Telegram-linked admin
    list, not every role='admin' row. A local-only admin account with no
    Telegram identity has nowhere to deliver a telegram-channel message to
    and will always fail delivery (see monitoring.txt incident).
    """
    request = _make_request({"text": "[systemd/foo] ERROR: boom", "source": "foo.service"})

    with (
        patch("src.api.internal_routes.settings.internal_api_token", ""),
        patch("src.api.internal_routes.telegram_service") as mock_telegram_service,
        patch("src.api.internal_routes.NotificationService") as mock_notification_service,
    ):
        mock_telegram_service.get_admin_user_ids.return_value = ["111"]
        mock_svc = mock_notification_service.return_value

        result = await receive_log_alert(request)

    assert result == {"ok": True}
    mock_telegram_service.get_admin_user_ids.assert_called_once()
    mock_svc.create_message.assert_called_once()
    queued = mock_svc.create_message.call_args[0][0]
    assert queued["recipient_id"] == "111"
    assert queued["channels"] == ["telegram"]
    assert queued["content"]["source"] == "foo.service"


@pytest.mark.asyncio
async def test_log_alert_no_telegram_admins_skips_delivery():
    """No admin has a Telegram identity — nothing should be queued."""
    request = _make_request({"text": "ERROR: boom", "source": "foo.service"})

    with (
        patch("src.api.internal_routes.settings.internal_api_token", ""),
        patch("src.api.internal_routes.telegram_service") as mock_telegram_service,
        patch("src.api.internal_routes.NotificationService") as mock_notification_service,
    ):
        mock_telegram_service.get_admin_user_ids.return_value = []

        result = await receive_log_alert(request)

    assert result == {"ok": True, "warning": "no admin users found"}
    mock_notification_service.return_value.create_message.assert_not_called()


@pytest.mark.asyncio
async def test_log_alert_rejects_non_localhost():
    request = _make_request({"text": "ERROR: boom", "source": "foo.service"}, host="203.0.113.1")

    with pytest.raises(Exception) as exc_info:
        await receive_log_alert(request)

    assert getattr(exc_info.value, "status_code", None) == 403
