"""The client puts the HTML body into `content.html`, where the email channel reads it."""

import asyncio
from typing import Any, Dict, List

import pytest

from src.notification.service.client import NotificationServiceClient


def _captured(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> Dict[str, Any]:
    client = NotificationServiceClient(service_url="database://")
    sent: List[Dict[str, Any]] = []

    async def fake_db(message_data: Dict[str, Any]) -> None:
        sent.append(message_data)

    monkeypatch.setattr(client, "_send_notification_direct_to_db", fake_db)
    assert asyncio.run(client.send_notification(notification_type="report", title="T", message="plain", channels=["email"], **kwargs))
    (message,) = sent
    return message


def test_html_is_stored_in_content(monkeypatch: pytest.MonkeyPatch) -> None:
    message = _captured(monkeypatch, html="<b>table</b>", data={"rows": 3})

    assert message["content"]["html"] == "<b>table</b>"
    assert message["content"]["message"] == "plain"
    assert message["message_metadata"] == {"rows": 3}


def test_no_html_leaves_content_plain(monkeypatch: pytest.MonkeyPatch) -> None:
    message = _captured(monkeypatch)

    assert "html" not in message["content"]
