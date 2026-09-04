"""
Notifier.

Formats the PnL digest and dispatches it via `NotificationServiceClient` to
all configured channels (Telegram, Email). Every priced holding is included,
sorted by PnL% descending; rows at or above the threshold are highlighted.
Recent insider (Form 4) activity is attached under each held ticker that has
any in the trailing window.
"""

from datetime import UTC, datetime
from html import escape
from typing import Any, Dict, List, Optional, Sequence

from src.notification.logger import setup_logger
from src.portfolio.pnl_alert.insider_activity import InsiderTransaction
from src.portfolio.pnl_alert.pnl_evaluator import AlertRow

_logger = setup_logger(__name__)

# Human-readable labels for the SEC Form 4 transaction codes we're likely to
# see among non-derivative transactions. Falls back to acquired_disposed_code
# ("Acquired"/"Disposed") for anything not listed here.
_CODE_LABELS = {
    "P": "Buy",
    "S": "Sell",
    "A": "Grant/Award",
    "D": "Sale to Issuer",
    "F": "Tax Withholding",
    "M": "Option Exercise",
    "C": "Conversion",
    "G": "Gift",
    "I": "Discretionary",
}


def _format_money(value: float) -> str:
    """Format a signed USD amount, e.g. `+$364.00` / `-$12.50`."""
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):,.2f}"


def _format_pct(value: float) -> str:
    """Format a signed percentage, e.g. `+30.33%`."""
    sign = "+" if value >= 0 else "-"
    return f"{sign}{abs(value) * 100:.2f}%"


def _describe_txn_code(transaction_code: str, acquired_disposed_code: str) -> str:
    """Human label for a Form 4 transaction code, e.g. `"P"` -> `"Buy"`."""
    label = _CODE_LABELS.get(transaction_code)
    if label:
        return label
    if acquired_disposed_code == "A":
        return "Acquired"
    if acquired_disposed_code == "D":
        return "Disposed"
    return transaction_code or "?"


def _format_insider_txn_plain(txn: InsiderTransaction) -> str:
    """One indented plain-text line describing a single insider transaction."""
    label = _describe_txn_code(txn.transaction_code, txn.acquired_disposed_code)
    amount = f"{txn.shares:,} sh @ ${txn.price_per_share:,.2f} = ${txn.total_value_usd:,.2f}" if txn.price_per_share else f"{txn.shares:,} sh"
    plan_suffix = "  [10b5-1 plan]" if txn.is_10b5_1_plan else ""
    return f"      {txn.transaction_date}  {label:<12} {txn.insider_name} ({txn.role}) — {amount}{plan_suffix}"


def _insider_section_plain(insider_by_ticker: Dict[str, List[InsiderTransaction]], symbol: str) -> List[str]:
    """Plain-text lines for one ticker's insider activity, or `[]` if none."""
    txns = insider_by_ticker.get(symbol)
    if not txns:
        return []
    lines = ["      Insider activity (30d):"]
    lines.extend(_format_insider_txn_plain(t) for t in txns)
    return lines


def format_plain_text(
    rows: Sequence[AlertRow],
    threshold_pct: float,
    as_of: datetime | None = None,
    insider_by_ticker: Optional[Dict[str, List[InsiderTransaction]]] = None,
) -> str:
    """
    Build the plain-text body used for Telegram (and the email fallback).

    Args:
        rows: Every priced holding, sorted by PnL% desc (`flagged=True` rows
            are highlighted).
        threshold_pct: Threshold used, for the header text and highlighting.
        as_of: Timestamp shown in the header. Defaults to now (UTC).
        insider_by_ticker: Optional trailing-window Form 4 activity per held
            ticker (see `insider_activity.load_insider_activity`). Tickers
            absent from the mapping show no insider section.

    Returns:
        Plain-text message.
    """
    when = as_of or datetime.now(UTC)
    insider_by_ticker = insider_by_ticker or {}
    flagged_count = sum(1 for r in rows if r.flagged)
    header = (
        f"Portfolio PnL Digest - {when.strftime('%Y-%m-%d')}\n"
        f"{len(rows)} position(s), {flagged_count} flagged >= "
        f"+{threshold_pct * 100:.2f}% threshold"
    )

    if not rows:
        return header

    lines = [header, ""]
    for rank, row in enumerate(rows, start=1):
        marker = "\U0001f53a " if row.flagged else "   "  # 🔺
        lines.append(
            f"{marker}{rank}. {row.symbol:<6} "
            f"avg ${row.avg_price:,.2f}   "
            f"now ${row.current_price:,.2f}   "
            f"PnL {_format_money(row.pnl_abs)}  ({_format_pct(row.pnl_pct)})"
        )
        lines.extend(_insider_section_plain(insider_by_ticker, row.symbol))

    sources_summary = _sources_summary(rows)
    if sources_summary:
        lines.append("")
        lines.append(sources_summary)

    return "\n".join(lines)


def _insider_section_html(insider_by_ticker: Dict[str, List[InsiderTransaction]], symbol: str) -> str:
    """Nested `<tr>` with a small insider-activity table for one ticker, or `""` if none."""
    txns = insider_by_ticker.get(symbol)
    if not txns:
        return ""

    txn_rows = []
    for t in txns:
        label = _describe_txn_code(t.transaction_code, t.acquired_disposed_code)
        amount = f"{t.shares:,} sh @ ${t.price_per_share:,.2f} = ${t.total_value_usd:,.2f}" if t.price_per_share else f"{t.shares:,} sh"
        plan_badge = " <i>(10b5-1 plan)</i>" if t.is_10b5_1_plan else ""
        txn_rows.append(
            "<tr>"
            f"<td>{escape(t.transaction_date)}</td>"
            f"<td>{escape(label)}</td>"
            f"<td>{escape(t.insider_name)} ({escape(t.role)})</td>"
            f"<td>{escape(amount)}{plan_badge}</td>"
            "</tr>"
        )

    return (
        "<tr><td colspan='7' style='padding-left:2em;'>"
        "<table border='1' cellpadding='4' cellspacing='0' "
        "style='border-collapse:collapse;font-family:monospace;font-size:0.9em;'>"
        "<tbody>" + "".join(txn_rows) + "</tbody>"
        "</table>"
        "</td></tr>"
    )


def format_html(
    rows: Sequence[AlertRow],
    threshold_pct: float,
    as_of: datetime | None = None,
    insider_by_ticker: Optional[Dict[str, List[InsiderTransaction]]] = None,
) -> str:
    """
    Build an HTML body for the email channel.

    Args:
        rows: Every priced holding, sorted by PnL% desc (`flagged=True` rows
            are highlighted).
        threshold_pct: Threshold used, for the header text and highlighting.
        as_of: Timestamp shown in the header. Defaults to now (UTC).
        insider_by_ticker: Optional trailing-window Form 4 activity per held
            ticker (see `insider_activity.load_insider_activity`).

    Returns:
        HTML string suitable for the email body.
    """
    when = as_of or datetime.now(UTC)
    insider_by_ticker = insider_by_ticker or {}
    flagged_count = sum(1 for r in rows if r.flagged)
    header = (
        f"<h2>Portfolio PnL Digest &mdash; {escape(when.strftime('%Y-%m-%d'))}</h2>"
        f"<p>{len(rows)} position(s), {flagged_count} flagged &ge; "
        f"+{threshold_pct * 100:.2f}% threshold</p>"
    )

    if not rows:
        return header

    table_rows = []
    for rank, row in enumerate(rows, start=1):
        row_style = " style='font-weight:bold;background:#fff8e1;'" if row.flagged else ""
        flag_cell = "\U0001f53a" if row.flagged else ""  # 🔺
        table_rows.append(
            f"<tr{row_style}>"
            f"<td>{rank}</td>"
            f"<td>{flag_cell}</td>"
            f"<td><b>{escape(row.symbol)}</b></td>"
            f"<td>${row.avg_price:,.2f}</td>"
            f"<td>${row.current_price:,.2f}</td>"
            f"<td>{escape(_format_money(row.pnl_abs))}</td>"
            f"<td>{escape(_format_pct(row.pnl_pct))}</td>"
            "</tr>"
        )
        table_rows.append(_insider_section_html(insider_by_ticker, row.symbol))

    table = (
        "<table border='1' cellpadding='6' cellspacing='0' "
        "style='border-collapse:collapse;font-family:monospace;'>"
        "<thead><tr>"
        "<th>#</th><th></th><th>Ticker</th><th>Avg</th><th>Now</th>"
        "<th>PnL</th><th>PnL %</th>"
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
    insider_by_ticker: Optional[Dict[str, List[InsiderTransaction]]] = None,
) -> bool:
    """
    Dispatch the digest to the configured channels.

    Args:
        rows: Every priced holding from the evaluator.
        channels: Channels to notify (any subset of "telegram", "email").
        threshold_pct: Threshold used, for the header text and highlighting.
        recipient_id: User ID used to resolve both the email address and
            Telegram chat ID for delivery.
        client: Optional pre-built `NotificationServiceClient`. A default one
            is created when not supplied.
        dry_run: If True, format the message and log it but don't send.
        as_of: Timestamp used for the header.
        insider_by_ticker: Optional trailing-window Form 4 activity per held
            ticker.

    Returns:
        True if delivery was successful (or not attempted in `dry_run`).
    """
    if not rows:
        _logger.info("No priced holdings; skipping notification")
        return True

    plain = format_plain_text(rows, threshold_pct, as_of=as_of, insider_by_ticker=insider_by_ticker)
    html = format_html(rows, threshold_pct, as_of=as_of, insider_by_ticker=insider_by_ticker)
    flagged_count = sum(1 for r in rows if r.flagged)
    title = f"Portfolio PnL Digest - {len(rows)} position(s), {flagged_count} above +{threshold_pct * 100:.2f}%"

    if dry_run:
        _logger.info("Dry run enabled; digest message below\n%s", plain)
        return True

    if client is None:
        from src.notification.service.client import NotificationServiceClient

        # POST /api/notifications requires a user JWT (Depends(get_current_user)),
        # which a scheduled backend job never has — HTTP mode would 401 on every
        # call and silently fall back to this same DB path anyway, just noisier.
        client = NotificationServiceClient(service_url="database://")

    ok = await client.send_notification(
        notification_type="portfolio_pnl_alert",
        title=title,
        message=plain,
        priority="normal",
        channels=list(channels),
        source="portfolio.pnl_alert",
        data={"html": html, "row_count": len(rows), "flagged_count": flagged_count, "threshold_pct": threshold_pct},
        recipient_id=str(recipient_id) if recipient_id is not None else None,
    )

    if ok:
        _logger.info("PnL digest queued to channels: %s", list(channels))
    else:
        _logger.error("PnL digest delivery returned False")
    return bool(ok)
