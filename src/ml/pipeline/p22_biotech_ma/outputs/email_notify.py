"""
P22 — queue an HTML email in the notification DB (2026-11-09).

Same path as the trade-ledger and P05/P17 emails: the email channel sends `content.html` when
present and `content.message` otherwise. Used by the weekly backtest and label-review digest jobs,
which build their own HTML instead of the scheduler's generic key/value notification.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pipeline.base_plugin import SYSTEM_USER_ID
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

SOURCE = "ml.pipeline.p22_biotech_ma"

MessageCreator = Callable[[Dict[str, Any]], Any]

_TABLE_STYLE = "border-collapse:collapse;font-family:Arial,sans-serif;font-size:13px"
_CELL_STYLE = "border:1px solid #ccc;padding:3px 8px"


def html_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """A plain bordered HTML table; every cell is escaped (pass `RawHtml` to keep markup)."""
    head = "".join(f'<th style="{_CELL_STYLE};background:#f0f0f0">{html.escape(h)}</th>' for h in headers)
    body = "".join("<tr>" + "".join(f'<td style="{_CELL_STYLE}">{_cell(v)}</td>' for v in row) + "</tr>" for row in rows)
    return f'<table style="{_TABLE_STYLE}"><tr>{head}</tr>{body}</table>'


class RawHtml(str):
    """A cell value that is already HTML (e.g. a link) and must not be escaped."""


def link(url: Optional[str], text: str) -> RawHtml:
    """An `<a>` cell, or the escaped text alone when there is no URL."""
    if not url:
        return RawHtml(html.escape(text))
    return RawHtml(f'<a href="{html.escape(url, quote=True)}">{html.escape(text)}</a>')


def _cell(value: Any) -> str:
    if isinstance(value, RawHtml):
        return str(value)
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return html.escape(str(value))


def _default_creator() -> MessageCreator:
    """`NotificationService().create_message`, imported lazily so tests need no DB."""
    from src.data.db.services.notification_service import NotificationService

    return NotificationService().create_message


def build_email(title: str, text: str, body_html: str, user_id: Optional[int] = None) -> Dict[str, Any]:
    """
    The notification-DB message for one email.

    Args:
        title: Subject line.
        text: Plain-text fallback.
        body_html: HTML body.
        user_id: Recipient; the scheduler's system user (the P22 schedules' owner) when None.
    """
    recipient = user_id if user_id is not None else SYSTEM_USER_ID
    return {
        "message_type": "REPORT",
        "channels": ["email"],
        "recipient_id": str(recipient),
        "content": {"title": title, "message": text, "html": body_html, "source": SOURCE},
        "priority": "NORMAL",
        "message_metadata": {"source": SOURCE},
    }


def queue_email(title: str, text: str, body_html: str, user_id: Optional[int] = None,
                create_message: Optional[MessageCreator] = None) -> bool:
    """Queue one email; `False` (logged) when the notification DB write fails."""
    creator = create_message or _default_creator()
    try:
        creator(build_email(title, text, body_html, user_id))
    except Exception:
        _logger.exception("Failed to queue P22 email %r", title)
        return False
    _logger.info("Queued P22 email %r", title)
    return True
