"""Internal routes for system-to-system communication — no auth, localhost only."""

import hmac
import json

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from src.api.config import settings
from src.data.db.models.model_users import User
from src.data.db.services.database_service import get_database_service
from src.data.db.services.notification_service import NotificationService
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

router = APIRouter(prefix="/internal", tags=["internal"])

_LOCALHOST = {"127.0.0.1", "::1"}
_TOKEN_HEADER = "X-Internal-Token"


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

    db_service = get_database_service()
    with db_service.uow() as uow:
        admin_ids = [str(u.id) for u in uow.s.query(User).filter(User.role == "admin").all()]

    if not admin_ids:
        _logger.warning("No admin users found — log alert not delivered: %s", alert.source)
        return {"ok": True, "warning": "no admin users found"}

    # Build a subject that shows the source and the first meaningful part of the error line.
    # alert.text format from Vector: "[systemd/service] ERROR: ..."
    first_line = alert.text.splitlines()[0] if alert.text else alert.source
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
