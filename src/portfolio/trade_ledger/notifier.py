"""
Notifier for the trade-ledger report.

Queues the monthly email (HTML body + XLSX attachment) or a short failure
email in the notification DB, the same way P05 / P17 do: `content.html` and
`content.attachments` are what the email channel reads. The XLSX travels as
base64 inside the message, so the notification bot does not need read access
to the scheduler's data folder.
"""

import base64
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.notification.logger import setup_logger
from src.portfolio.trade_ledger.report_builder import ReportView, format_html_body, format_plain_text

_logger = setup_logger(__name__)

SOURCE = "portfolio.trade_ledger"
# Gmail clips HTML bodies above ~102 KB ("[Message clipped]").
GMAIL_CLIP_BYTES = 102_000

MessageCreator = Callable[[Dict[str, Any]], Any]


def _default_creator() -> MessageCreator:
    """Return `NotificationService().create_message` (imported lazily so tests need no DB)."""
    from src.data.db.services.notification_service import NotificationService

    return NotificationService().create_message


def _queue(message: Dict[str, Any], create_message: Optional[MessageCreator]) -> bool:
    """Queue one message; log and return False on any DB error."""
    creator = create_message or _default_creator()
    try:
        creator(message)
    except Exception:
        _logger.exception("Failed to queue trade-ledger email %r", message["content"]["title"])
        return False
    return True


def build_report_message(view: ReportView, xlsx_path: Path, recipient_id: Optional[int], channels: Sequence[str]) -> Dict[str, Any]:
    """
    Build the notification-DB message for the monthly report.

    Args:
        view: Report content.
        xlsx_path: Workbook to attach.
        recipient_id: User ID receiving the email.
        channels: Delivery channels.

    Returns:
        Message dict for `NotificationService.create_message`.
    """
    html = format_html_body(view)
    html_bytes = len(html.encode("utf-8"))
    if html_bytes > GMAIL_CLIP_BYTES:
        _logger.warning("Trade-ledger HTML body is %d bytes; Gmail clips above %d", html_bytes, GMAIL_CLIP_BYTES)
    data = xlsx_path.read_bytes()
    return {
        "message_type": "REPORT",
        "channels": list(channels),
        "recipient_id": str(recipient_id) if recipient_id is not None else None,
        "content": {
            "title": view.title,
            "message": format_plain_text(view),
            "html": html,
            "attachments": {xlsx_path.name: {"type": "base64", "data": base64.b64encode(data).decode("ascii"), "size": len(data)}},
            "source": SOURCE,
        },
        "priority": "NORMAL",
        "message_metadata": {"source": SOURCE, "reconciliation": view.reconciliation.status, "executions": len(view.executions)},
    }


def build_failure_message(month_label: str, errors: List[str], recipient_id: Optional[int], channels: Sequence[str]) -> Dict[str, Any]:
    """
    Build a short failure email.

    Args:
        month_label: Report month, e.g. "2026-10".
        errors: What went wrong (one line each).
        recipient_id: User ID receiving the email.
        channels: Delivery channels.

    Returns:
        Message dict for `NotificationService.create_message`.
    """
    text = "\n".join([f"The trade-ledger report for {month_label} could not be built.", "", *errors])
    return {
        "message_type": "REPORT",
        "channels": list(channels),
        "recipient_id": str(recipient_id) if recipient_id is not None else None,
        "content": {"title": f"Trade Ledger — {month_label} FAILED", "message": text, "source": SOURCE},
        "priority": "HIGH",
        "message_metadata": {"source": SOURCE, "failed": True},
    }


def send_report(
    view: ReportView,
    xlsx_path: Path,
    recipient_id: Optional[int],
    channels: Sequence[str],
    create_message: Optional[MessageCreator] = None,
) -> bool:
    """
    Queue the monthly report email.

    Args:
        view: Report content.
        xlsx_path: Workbook to attach.
        recipient_id: User ID receiving the email.
        channels: Delivery channels.
        create_message: Message creator (defaults to `NotificationService.create_message`).

    Returns:
        True when the message was queued.
    """
    ok = _queue(build_report_message(view, xlsx_path, recipient_id, channels), create_message)
    if ok:
        _logger.info("Queued trade-ledger report %r to user %s", view.title, recipient_id)
    return ok


def send_failure(
    month_label: str,
    errors: List[str],
    recipient_id: Optional[int],
    channels: Sequence[str],
    create_message: Optional[MessageCreator] = None,
) -> bool:
    """
    Queue a short failure email.

    Args:
        month_label: Report month, e.g. "2026-10".
        errors: What went wrong.
        recipient_id: User ID receiving the email.
        channels: Delivery channels.
        create_message: Message creator (defaults to `NotificationService.create_message`).

    Returns:
        True when the message was queued.
    """
    ok = _queue(build_failure_message(month_label, errors, recipient_id, channels), create_message)
    if ok:
        _logger.info("Queued trade-ledger failure email for %s to user %s", month_label, recipient_id)
    return ok
