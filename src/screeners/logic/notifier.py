import asyncio
from typing import Dict, Any, List
from datetime import datetime

from src.notification.service.client import NotificationServiceClient, MessageType, MessagePriority
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

class SignalNotifier:
    """
    Formats and broadcasts screener signals to notification channels (Telegram, etc.).
    """

    def __init__(self, client: NotificationServiceClient):
        self.client = client

    async def notify_signals(self, signals: List[Dict[str, Any]]):
        """Sends notifications for a batch of signals."""
        for signal in signals:
            # We only notify if there's a clear 'buy' or 'sell' signal
            # This can be customized based on user preference
            if signal.get("signal") in ["buy", "sell"]:
                await self._send_single_signal(signal)

    async def _send_single_signal(self, signal: Dict[str, Any]):
        """Formats and sends a single signal alert."""
        symbol = signal.get("symbol", "UNKNOWN")
        side = signal.get("signal", "NEUTRAL").upper()
        price = signal.get("price", 0.0)

        # Format the Message (Markdown)
        title = f"🚀 SCREENER ALERT: {side} {symbol}"

        # Build Body
        body = [
            f"*Symbol:* {symbol}",
            f"*Price:* ${price:.2f}",
            f"*Signal:* {side}",
            "",
            "*ML Analysis:*"
        ]

        ml = signal.get("ml_analysis", {})
        if ml:
            body.append(f"• Regime: {ml.get('regime', 'N/A')}")
            body.append(f"• Confidence: {ml.get('confidence', 0.0) * 100:.1f}%")
            body.append(f"• Prediction: {ml.get('prediction', 0.0) * 100:+.2f}%")
        else:
            body.append("• No ML data available")

        body.append("")
        body.append("*Technical Indicators:*")
        inds = signal.get("indicators", {})
        for alias, val in inds.items():
            if val is not None:
                body.append(f"• {alias}: {val:.2f}")
            else:
                body.append(f"• {alias}: N/A")

        full_message = "\n".join(body)

        # Send via client
        success = await self.client.send_notification(
            notification_type=MessageType.ALERT,
            title=title,
            message=full_message,
            priority=MessagePriority.HIGH,
            source="ibkr_screener"
        )

        if success:
            _logger.info("Signal alert sent for %s", symbol)
        else:
            _logger.error("Failed to send signal alert for %s", symbol)
