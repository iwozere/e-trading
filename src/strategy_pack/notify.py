"""Format pack signals and send via ``NotificationServiceClient``."""

from __future__ import annotations

from typing import List

from src.notification.logger import setup_logger
from src.notification.service.client import MessagePriority, MessageType, NotificationServiceClient
from src.strategy_pack.io import DedupStore
from src.strategy_pack.models import PackSignal

_logger = setup_logger(__name__)


def format_signal_message(sig: PackSignal) -> str:
    lines = [
        f"*Strategy:* {sig.strategy_id} ({sig.variant})",
        f"*Symbol:* {sig.symbol}",
        f"*Signal:* {sig.signal}",
        f"*Bar:* {sig.bar_timeframe} @ {sig.bar_close_ts}",
        f"*Price:* {sig.price:.6g}" if sig.price else "*Price:* n/a",
        f"*Reason:* {sig.reason_code}",
    ]
    if sig.metadata:
        lines.append("")
        lines.append("*Metadata:*")
        for k, v in list(sig.metadata.items())[:25]:
            lines.append(f"• {k}: {v}")
    return "\n".join(lines)


async def send_pack_notifications(
    client: NotificationServiceClient,
    signals: List[PackSignal],
    dedup: DedupStore,
    *,
    source: str = "strategy_pack",
) -> int:
    """Send alerts for signals with ``notify_recommended`` and passing dedup. Returns send count."""
    sent = 0
    for sig in signals:
        if not sig.notify_recommended:
            continue
        if not dedup.should_notify(sig.idempotency_key):
            _logger.debug("Skip duplicate notification %s", sig.idempotency_key)
            continue
        title = f"[SIGNAL] {sig.strategy_id} — {sig.symbol} — {sig.signal}"
        body = format_signal_message(sig)
        ok = await client.send_notification(
            notification_type=MessageType.ALERT,
            title=title,
            message=body,
            priority=MessagePriority.HIGH,
            source=source,
            data=sig.to_jsonl_dict(),
        )
        if ok:
            dedup.mark_sent(sig.idempotency_key)
            sent += 1
        else:
            _logger.warning("Notification failed for %s %s", sig.strategy_id, sig.symbol)
    return sent
