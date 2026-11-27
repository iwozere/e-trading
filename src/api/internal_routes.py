"""Internal routes for system-to-system communication — no auth, localhost only."""

import hmac
import json
import re

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from src.api.config import settings
from src.data.db.services.notification_service import NotificationService
from src.data.db.services.telegram_service import telegram_service
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

router = APIRouter(prefix="/internal", tags=["internal"])

_LOCALHOST = {"127.0.0.1", "::1"}
_TOKEN_HEADER = "X-Internal-Token"

# Vector's `final_shape` transform already prefixes alert.text with "[type/source] "
# (see docs/monitoring-setup.md). Strip that off before re-wrapping in "[Monitoring/...]"
# below, otherwise the source shows up twice, e.g.
# "[Monitoring/session-1.scope] [systemd/session-1.scope] Read error ...".
_SOURCE_PREFIX_RE = re.compile(r"^\[[^\]]+\]\s*")

# Same keywords Vector's `errors_only` filter matches on (docs/monitoring-setup.md).
_ERROR_KEYWORD_RE = re.compile(r"(?i)(error|exception|critical|traceback|fatal)")

# Python-logging line at a non-alerting level, e.g. "2026-11-27 01:09:23,982 INFO name: ...".
_LOW_LEVEL_LINE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}[,.\d]*\s+(DEBUG|INFO)\b")

# Zero-count summaries such as "0 error(s)", "0 errors", "0 exceptions".
_ZERO_COUNT_RE = re.compile(r"(?i)\b0\s+(errors?|error\(s\)|exceptions?|exception\(s\)|fatals?)")


def _is_noise(text: str) -> bool:
    """
    Tell whether a forwarded log line only looked like an error to Vector's keyword filter.

    Vector keeps any line containing "error" etc., so INFO progress lines such as
    "... 37 request(s), 364 phrase(s) stored, 0 error(s)" were raised as alerts.

    Args:
        text: Alert text as sent by Vector (may carry a "[type/source] " prefix).

    Returns:
        True if the line should be dropped instead of alerted on.
    """
    line = _SOURCE_PREFIX_RE.sub("", text.splitlines()[0] if text else "", count=1)
    if _LOW_LEVEL_LINE_RE.match(line):
        return True
    return not _ERROR_KEYWORD_RE.search(_ZERO_COUNT_RE.sub("", line))


def _check_internal_access(request: Request) -> None:
    """Verify request is from localhost and carries the correct shared secret."""
    if not request.client or request.client.host not in _LOCALHOST:
        raise HTTPException(status_code=403, detail="Forbidden")
    if settings.internal_api_token:
        incoming = request.headers.get(_TOKEN_HEADER, "")
        if not hmac.compare_digest(incoming, settings.internal_api_token):
            raise HTTPException(status_code=403, detail="Forbidden")


class LogAlertRequest(BaseModel):
    text: str
    source: str


@router.post("/log-alert", include_in_schema=False)
async def receive_log_alert(request: Request) -> dict:
    """Receive a log error alert from Vector. Restricted to localhost + shared secret."""
    _check_internal_access(request)

    raw = await request.body()
    try:
        data = json.loads(raw.decode())
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"Invalid JSON: {e}") from e

    # Vector HTTP sink sends a JSON array by default; NDJSON framing sends a single object.
    if isinstance(data, list):
        if not data:
            return {"ok": True}
        payload = data[0]
    else:
        payload = data

    try:
        alert = LogAlertRequest(**payload)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    if _is_noise(alert.text):
        _logger.debug("Dropped non-error log alert: source=%s", alert.source)
        return {"ok": True, "dropped": "noise"}

    # Only admins with a linked Telegram identity can receive this — channels is
    # hardcoded to ["telegram"] below, and a role='admin' row with no Telegram
    # identity (e.g. a local-only admin account) has nowhere to deliver to.
    admin_ids = telegram_service.get_admin_user_ids()

    if not admin_ids:
        _logger.warning("No Telegram-linked admin users found — log alert not delivered: %s", alert.source)
        return {"ok": True, "warning": "no admin users found"}

    # Build a subject that shows the source and the first meaningful part of the error line.
    # alert.text format from Vector: "[systemd/service] ERROR: ..." — strip that leading
    # "[type/source]" tag so it isn't shown twice alongside our own "[Monitoring/source]".
    first_line = alert.text.splitlines()[0] if alert.text else alert.source
    first_line = _SOURCE_PREFIX_RE.sub("", first_line, count=1)
    subject = f"[Monitoring/{alert.source}] {first_line[:120]}"

    svc = NotificationService()
    for admin_id in admin_ids:
        svc.create_message(
            {
                "message_type": "system_alert",
                "channels": ["telegram"],
                "recipient_id": admin_id,
                "content": {"title": subject, "message": alert.text, "source": alert.source},
                "priority": "HIGH",
            }
        )

    _logger.info("Log alert queued for %d admin(s): source=%s", len(admin_ids), alert.source)
    return {"ok": True}
