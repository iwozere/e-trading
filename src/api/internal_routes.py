"""Internal routes for system-to-system communication — no auth, localhost only."""
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

from src.data.db.models.model_users import User
from src.data.db.services.database_service import get_database_service
from src.data.db.services.notification_service import NotificationService
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

router = APIRouter(prefix="/internal", tags=["internal"])

_LOCALHOST = {"127.0.0.1", "::1"}


class LogAlertRequest(BaseModel):
    text: str
    source: str


@router.post("/log-alert", include_in_schema=False)
async def receive_log_alert(request: Request, body: LogAlertRequest) -> dict:
    """Receive a log error alert from Vector. Restricted to localhost."""
    if not request.client or request.client.host not in _LOCALHOST:
        raise HTTPException(status_code=403, detail="Forbidden")

    db_service = get_database_service()
    with db_service.uow() as uow:
        admin_ids = [
            str(u.id) for u in uow.s.query(User).filter(User.role == "admin").all()
        ]

    if not admin_ids:
        _logger.warning("No admin users found — log alert not delivered: %s", body.source)
        return {"ok": True, "warning": "no admin users found"}

    svc = NotificationService()
    for admin_id in admin_ids:
        svc.create_message({
            "message_type": "system_alert",
            "channels": ["telegram"],
            "recipient_id": admin_id,
            "content": {"message": body.text, "source": body.source},
            "priority": "HIGH",
        })

    _logger.info("Log alert queued for %d admin(s): source=%s", len(admin_ids), body.source)
    return {"ok": True}
