"""
Notifier.

Formats the PnL alert digest and dispatches it via `NotificationServiceClient`
to all configured channels (Telegram, Email).
"""

from datetime import UTC, datetime
from html import escape
from typing import Any, List, Optional, Sequence

from src.notification.logger import setup_logger
from src.portfolio.pnl_alert.pnl_evaluator import AlertRow

_logger = setup_logger(__name__)


def _format_money(value: float) -> str:
    """Format a signed USD amount, e.g. `+$364.00` / `-$12.50`."""
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):,.2f}"


def _format_pct(value: float) -> str:
    """Format a signed percentage, e.g. `+30.33%`."""
    sign = "+" if value >= 0 else "-"
    return f"{sign}{abs(value) * 100:.2f}%"


def format_plain_text(
    rows: Sequence[AlertRow],
    threshold_pct: float,
    as_of: datetime | None = None,
) -> str:
    """
    Build the plain-text body used for Telegram (and the email fallback).

    Args:
        rows: Rows already filtered and sorted by the evaluator.
        threshold_pct: Threshold used, for the header text.
        as_of: Timestamp shown in the header. Defaults to now (UTC).

    Returns:
        Plain-text message.
    """
    when = as_of or datetime.now(UTC)
    header = (
        f"Portfolio PnL Alert - {when.strftime('%Y-%m-%d')}\n"
        f"{len(rows)} position(s) above "
        f"+{threshold_pct * 100:.2f}% threshold"
    )

    if not rows:
        return header

    lines = [header, ""]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            f"{rank}. {row.symbol:<6} "
            f"avg ${row.avg_price:,.2f}   "
            f"now ${row.current_price:,.2f}   "
            f"PnL {_format_money(row.pnl_abs)}  ({_format_pct(row.pnl_pct)})"
        )

    sources_summary = _sources_summary(rows)
    if sources_summary:
        lines.append("")
        lines.append(sources_summary)

    return "\n".join(lines)


def format_html(
    rows: Sequence[AlertRow],
    threshold_pct: float,
    as_of: datetime | None = None,
) -> str:
    """
    Build an HTML body for the email channel.

    Args:
        rows: Rows already filtered and sorted by the evaluator.
        threshold_pct: Threshold used, for the header text.
        as_of: Timestamp shown in the header. Defaults to now (UTC).

    Returns:
        HTML string suitable for the email body.
    """
    when = as_of or datetime.now(UTC)
    header = (
        f"<h2>Portfolio PnL Alert &mdash; {escape(when.strftime('%Y-%m-%d'))}</h2>"
        f"<p>{len(rows)} position(s) above "
        f"+{threshold_pct * 100:.2f}% threshold</p>"
    )

    if not rows:
        return header

    table_rows = []
    for rank, row in enumerate(rows, start=1):
        table_rows.append(
            "<tr>"
            f"<td>{rank}</td>"
            f"<td><b>{escape(row.symbol)}</b></td>"
            f"<td>${row.avg_price:,.2f}</td>"
            f"<td>${row.current_price:,.2f}</td>"
            f"<td>{escape(_format_money(row.pnl_abs))}</td>"
            f"<td>{escape(_format_pct(row.pnl_pct))}</td>"
            f"<td>{escape(row.source)}</td>"
            "</tr>"
        )

    table = (
        "<table border='1' cellpadding='6' cellspacing='0' "
        "style='border-collapse:collapse;font-family:monospace;'>"
        "<thead><tr>"
        "<th>#</th><th>Ticker</th><th>Avg</th><th>Now</th>"
        "<th>PnL</th><th>PnL %</th><th>Source</th>"
        "</tr></thead>"
        "<tbody>" + "".join(table_rows) + "</tbody>"
        "</table>"
    )

    summary = _sources_summary(rows)
    summary_html = f"<p>{escape(summary)}</p>" if summary else ""
    return header + table + summary_html


def _sources_summary(rows: Sequence[AlertRow]) -> str:
    """Build the `Sources: ibkr=X, watchlist=Y` footer line."""
    if not rows:
        return ""
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.source] = counts.get(row.source, 0) + 1
    parts = [f"{src}={n}" for src, n in sorted(counts.items())]
    return "Sources: " + ", ".join(parts)


async def send_alert(
    rows: List[AlertRow],
    channels: Sequence[str],
    threshold_pct: float,
    recipient_id: int | None = None,
    client: Optional[Any] = None,
    dry_run: bool = False,
    as_of: datetime | None = None,
) -> bool:
    """
    Dispatch the alert to the configured channels.

    Args:
        rows: Rows from the evaluator.
        channels: Channels to notify (any subset of "telegram", "email").
        threshold_pct: Threshold used, for the header text.
        recipient_id: User ID used to resolve both the email address and
            Telegram chat ID for delivery.
        client: Optional pre-built `NotificationServiceClient`. A default one
            is created when not supplied.
        dry_run: If True, format the message and log it but don't send.
        as_of: Timestamp used for the header.

    Returns:
        True if delivery was successful (or not attempted in `dry_run`).
    """
    if not rows:
        _logger.info("No alert rows above threshold; skipping notification")
        return True

    plain = format_plain_text(rows, threshold_pct, as_of=as_of)
    html = format_html(rows, threshold_pct, as_of=as_of)
    title = f"Portfolio PnL Alert - {len(rows)} above +{threshold_pct * 100:.2f}%"

    if dry_run:
        _logger.info("Dry run enabled; alert message below\n%s", plain)
        return True

    if client is None:
        from src.notification.service.client import NotificationServiceClient

        client = NotificationServiceClient()

    ok = await client.send_notification(
        notification_type="portfolio_pnl_alert",
        title=title,
        message=plain,
        priority="normal",
        channels=list(channels),
        source="portfolio.pnl_alert",
        data={"html": html, "row_count": len(rows), "threshold_pct": threshold_pct},
        recipient_id=str(recipient_id) if recipient_id is not None else None,
    )

    if ok:
        _logger.info("PnL alert queued to channels: %s", list(channels))
    else:
        _logger.error("PnL alert delivery returned False")
    return bool(ok)
