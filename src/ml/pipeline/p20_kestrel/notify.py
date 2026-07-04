"""
P20 Kestrel — Push notification helper.

Single fail-soft wrapper around NotificationServiceClient so alerting
modules (calendar_sync, risk_checker) send real pushes in addition to
their k20_alerts_log audit rows. A send failure never breaks the job.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def send_push(title: str, message: str) -> bool:
    """
    Send a push notification to admins. Fail-soft: never raises.

    Args:
        title: Notification title.
        message: Notification body.

    Returns:
        True if the send call completed without error, else False.
    """
    try:
        from src.notification.service.client import NotificationServiceClient

        client = NotificationServiceClient()
        asyncio.run(client.send_to_admins(title=title, message=message))
        return True
    except Exception:
        _logger.exception("Push notification failed: %s", title)
        return False
