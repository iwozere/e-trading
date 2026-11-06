"""
P24 Falling Knife — scan notifications (spec §9.5).

One message per run, only when there are new signals or paper exits. Telegram gets a short text plus one photo message
per chart (its sender would repeat a long text as the caption of every attachment); email gets one HTML message with
the charts attached. Messages are queued through ``NotificationService`` like P05; the notification processor
delivers them. Failures are logged, never raised: the results are already on disk.
"""

from __future__ import annotations

import html
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.ml.pipeline.p24_falling_knife.paper import PaperPosition
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

SOURCE = "p24_falling_knife"
CAVEAT = ("Paper only. The 2024+ holdout failed the go/no-go gate (mean +4.1%, CI includes 0; no clear edge over "
          "buying every 40% drop). Review the chart before acting.")


def _pct(value: Any) -> str:
    return f"{float(value):+.1%}" if value is not None else "—"


def _signal_line(row: Dict[str, Any]) -> str:
    flag = " ⚠ suspect data" if row.get("suspect_gap") else ""
    return (f"{row['ticker']} {row['timeframe']}: close {row['close']:.2f}, drop {_pct(-float(row['drop_pct'] or 0))} "
            f"from {row['peak_date']}, CCI {row['cci_3']}→{row['cci_2']}, score {row['score']}, entry next open{flag}")


def _exit_line(pos: PaperPosition, pending: bool) -> str:
    if pending:
        return (f"{pos.ticker} {pos.timeframe}: EXIT next open ({pos.exit_reason}), entered {pos.entry_date} at "
                f"{pos.entry_price:.2f}, now {_pct(pos.net_return)}")
    return (f"{pos.ticker} {pos.timeframe}: exited {pos.exit_date} at {pos.exit_price:.2f} ({pos.exit_reason}), "
            f"{_pct(pos.net_return)} after {pos.bars_held} bars")


def compose(timeframe: str, session: date, signals: Sequence[Dict[str, Any]], exit_pending: Sequence[PaperPosition],
            exited: Sequence[PaperPosition], forming_count: int) -> Optional[Dict[str, str]]:
    """
    Build the message, or None when there is nothing to send.

    Args:
        timeframe: ``"daily"`` or ``"weekly"``.
        session: Session scanned.
        signals: New signal rows.
        exit_pending: Positions whose exit rule fired on this bar.
        exited: Positions whose exit filled on this bar's open.
        forming_count: Size of the forming watchlist (context only).

    Returns:
        ``{"title", "text", "html"}`` or None.
    """
    if not (signals or exit_pending or exited):
        return None
    title = f"P24 Falling Knife {timeframe} — {session}: {len(signals)} signal(s), {len(exit_pending) + len(exited)} exit(s)"
    lines: List[str] = [title, ""]
    if signals:
        lines += ["New signals:"] + [f"• {_signal_line(r)}" for r in signals] + [""]
    if exit_pending or exited:
        lines += ["Paper exits:"] + [f"• {_exit_line(p, True)}" for p in exit_pending]
        lines += [f"• {_exit_line(p, False)}" for p in exited] + [""]
    lines += [f"Forming watchlist: {forming_count} setup(s).", "", CAVEAT]
    text = "\n".join(lines)
    body = "".join(f"<p>{html.escape(line)}</p>" if line else "" for line in lines[1:])
    page = f"<html><body><h3>{html.escape(title)}</h3>{body}</body></html>"
    return {"title": title, "text": text, "html": page}


def send(message: Dict[str, str], charts: Sequence[Path], user_id: Optional[str], max_charts: int) -> int:
    """
    Queue the message on the user's channels.

    Args:
        message: ``compose`` output.
        charts: Chart PNGs (signals first, then exits).
        user_id: Scheduler-injected user id; nothing is sent without it.
        max_charts: Charts attached at most.

    Returns:
        Number of messages queued.
    """
    if not user_id:
        _logger.warning("P24 notify: no user id, nothing sent (%s)", message["title"])
        return 0
    try:
        from src.data.db.services.notification_service import NotificationService
        from src.data.db.services.users_service import UsersService

        channels = UsersService().get_user_notification_channels(int(user_id)) or {}
        service = NotificationService()
    except (ValueError, TypeError):
        _logger.warning("P24 notify: invalid user id %r", user_id)
        return 0
    except Exception:
        _logger.exception("P24 notify: cannot resolve channels for user %s", user_id)
        return 0
    files = [str(p) for p in list(charts)[:max_charts] if p.exists()]
    queued = 0

    def _queue(channel: str, content: Dict[str, Any]) -> None:
        nonlocal queued
        try:
            service.create_message({"message_type": "REPORT", "channels": [channel], "recipient_id": str(user_id),
                                    "content": dict(content, source=SOURCE), "priority": "NORMAL",
                                    "message_metadata": {"source": SOURCE}})
            queued += 1
        except Exception:
            _logger.exception("P24 notify: failed to queue %s message", channel)

    if channels.get("telegram_chat_id"):
        _queue("telegram", {"title": message["title"], "message": message["text"]})
        for path in files:
            _queue("telegram", {"title": Path(path).stem, "message": Path(path).stem.replace("_", " "),
                                "attachments": {Path(path).name: path}})
    if channels.get("email"):
        _queue("email", {"title": message["title"], "message": message["text"], "html": message["html"],
                         "attachments": {"files": files}})
    _logger.info("P24 notify: %d message(s) queued for user %s", queued, user_id)
    return queued
