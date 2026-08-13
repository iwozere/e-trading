"""
Notifier.

Formats and sends the earnings-triggered stop-loss coverage reminder digest.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
from typing import Any, List, Optional, Sequence

from src.notification.logger import setup_logger
from src.portfolio.management.coverage import CoverageRow, CoverageStatus
from src.portfolio.management.earnings_window import TRIGGER_T_MINUS_1_DAY, EarningsEvent

_logger = setup_logger(__name__)

_TRIGGER_LABELS = {
    TRIGGER_T_MINUS_1_DAY: "earnings in ~1 day",
    "t_minus_1_hour": "earnings in ~1 hour",
}

_STATUS_TEXT = {
    CoverageStatus.COVERED: "covered",
    CoverageStatus.PARTIALLY_COVERED: "PARTIALLY covered",
    CoverageStatus.UNCOVERED: "UNCOVERED — no live stop order found",
}


@dataclass(frozen=True)
class TriggeredReminder:
    """One held ticker whose T-1day/T-1hour earnings trigger just fired."""

    event: EarningsEvent
    trigger: str  # TRIGGER_T_MINUS_1_DAY | TRIGGER_T_MINUS_1_HOUR
    coverage: CoverageRow


def _fmt_qty(value: float) -> str:
    """Format a share quantity without a trailing `.0` for whole shares."""
    return f"{value:g}"


def format_plain_text(reminders: Sequence[TriggeredReminder], as_of: Optional[datetime] = None) -> str:
    """
    Build the plain-text body used for Telegram (and the email fallback).

    Args:
        reminders: Triggered reminders for this run.
        as_of: Timestamp shown in the header. Defaults to now (UTC).

    Returns:
        Plain-text message.
    """
    when = as_of or datetime.now(UTC)
    header = (
        f"Earnings Stop-Loss Reminder — {when.strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"{len(reminders)} ticker(s) with earnings coming up"
    )

    if not reminders:
        return header

    lines = [header, ""]
    for r in reminders:
        session = f" ({r.event.session.upper()})" if r.event.session != "unknown" else ""
        lines.append(
            f"{r.event.ticker:<6} {_TRIGGER_LABELS[r.trigger]} — {r.event.earnings_date}{session} — "
            f"{_STATUS_TEXT[r.coverage.status]} "
            f"({_fmt_qty(r.coverage.protected_qty)}/{_fmt_qty(r.coverage.position_qty)} shares protected)"
        )

    return "\n".join(lines)


def format_html(reminders: Sequence[TriggeredReminder], as_of: Optional[datetime] = None) -> str:
    """
    Build an HTML body for the email channel.

    Args:
        reminders: Triggered reminders for this run.
        as_of: Timestamp shown in the header. Defaults to now (UTC).

    Returns:
        HTML string suitable for the email body.
    """
    when = as_of or datetime.now(UTC)
    header = (
        f"<h2>Earnings Stop-Loss Reminder &mdash; {escape(when.strftime('%Y-%m-%d %H:%M UTC'))}</h2>"
        f"<p>{len(reminders)} ticker(s) with earnings coming up</p>"
    )

    if not reminders:
        return header

    table_rows = []
    for r in reminders:
        session = r.event.session.upper() if r.event.session != "unknown" else "-"
        table_rows.append(
            "<tr>"
            f"<td><b>{escape(r.event.ticker)}</b></td>"
            f"<td>{escape(_TRIGGER_LABELS[r.trigger])}</td>"
            f"<td>{r.event.earnings_date}</td>"
            f"<td>{escape(session)}</td>"
            f"<td>{escape(_STATUS_TEXT[r.coverage.status])}</td>"
            f"<td>{_fmt_qty(r.coverage.protected_qty)}/{_fmt_qty(r.coverage.position_qty)}</td>"
            "</tr>"
        )

    table = (
        "<table border='1' cellpadding='6' cellspacing='0' "
        "style='border-collapse:collapse;font-family:monospace;'>"
        "<thead><tr>"
        "<th>Ticker</th><th>Trigger</th><th>Earnings date</th><th>Session</th>"
        "<th>Coverage</th><th>Protected/Held</th>"
        "</tr></thead>"
        "<tbody>" + "".join(table_rows) + "</tbody>"
        "</table>"
    )

    return header + table


async def send_reminder(
    reminders: List[TriggeredReminder],
    channels: Sequence[str],
    recipient_id: Optional[int] = None,
    client: Optional[Any] = None,
    dry_run: bool = False,
    as_of: Optional[datetime] = None,
) -> bool:
    """
    Dispatch the reminder to the configured channels.

    Args:
        reminders: Triggered reminders for this run.
        channels: Channels to notify (any subset of "telegram", "email").
        recipient_id: User ID used to resolve both the email address and
            Telegram chat ID for delivery.
        client: Optional pre-built `NotificationServiceClient`. A default one
            is created when not supplied.
        dry_run: If True, format the message and log it but don't send.
        as_of: Timestamp used for the header.

    Returns:
        True if delivery was successful (or not attempted in `dry_run`).
    """
    if not reminders:
        _logger.info("No triggered reminders this run; skipping notification")
        return True

    plain = format_plain_text(reminders, as_of=as_of)
    html = format_html(reminders, as_of=as_of)
    uncovered = sum(1 for r in reminders if r.coverage.status != CoverageStatus.COVERED)
    title = f"Earnings Stop-Loss Reminder — {len(reminders)} ticker(s), {uncovered} not fully covered"

    if dry_run:
        _logger.info("Dry run enabled; reminder message below\n%s", plain)
        return True

    if client is None:
        from src.notification.service.client import NotificationServiceClient

        # POST /api/notifications requires a user JWT (Depends(get_current_user)),
        # which a scheduled backend job never has — HTTP mode would 401 on every
        # call and silently fall back to this same DB path anyway, just noisier.
        client = NotificationServiceClient(service_url="database://")

    ok = await client.send_notification(
        notification_type="portfolio_stop_loss_reminder",
        title=title,
        message=plain,
        priority="normal",
        channels=list(channels),
        source="portfolio.management",
        data={"html": html, "reminder_count": len(reminders), "uncovered_count": uncovered},
        recipient_id=str(recipient_id) if recipient_id is not None else None,
    )

    if ok:
        _logger.info("Stop-loss reminder queued to channels: %s", list(channels))
    else:
        _logger.error("Stop-loss reminder delivery returned False")
    return bool(ok)
