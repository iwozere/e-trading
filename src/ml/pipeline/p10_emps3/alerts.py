"""
Alert Module for P10 EMPS3
Phase 1.5 / Pre-Breakout Notifications
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import asyncio

from src.notification.logger import setup_logger
from src.notification.service.client import NotificationServiceClient

_logger = setup_logger(__name__)

class EMPS3AlertSender:
    def __init__(self, user_id: Optional[str] = None):
        self.client = None
        self.user_id = user_id
        try:
            self.client = NotificationServiceClient()
            _logger.info("Notification service client initialized")
        except Exception:
            _logger.warning("Notification service client not available", exc_info=True)

    def send_phase1_5_alert(
        self,
        phase1_5_df: pd.DataFrame,
        csv_path: Optional[Path] = None
    ) -> None:
        if phase1_5_df.empty:
            return

        if not self.client:
            _logger.warning("Notification client not available, skipping alerts")
            return

        count = len(phase1_5_df)
        top_tickers = phase1_5_df['ticker'].head(10).tolist()

        title = f"⚠️ EMPS3 Phase 1.5 Alert - {count} Early Warning Candidate{'s' if count > 1 else ''}"
        message = f"""Detected {count} ticker{'s' if count > 1 else ''} hitting Phase 1.5 (Early Warning)

Top candidates: {', '.join(top_tickers)}

These tickers appeared 3+ times in the last 5 days with:
- ATR trending downwards
- Volume Z-Score trending upwards

See attached CSV."""

        attachments = {}
        if csv_path and csv_path.exists():
            attachments[csv_path.name] = str(csv_path)

        try:
            # Send to both channels in one run
            asyncio.run(self._send_notifications(title, message, attachments))
            _logger.info("Sent Phase 1.5 alerts for %d candidates", count)
        except Exception:
            _logger.exception("Failed to send Phase 1.5 alerts")

    def close(self) -> None:
        """Release the notification client. Call once at the end of the pipeline run."""
        if self.client:
            try:
                asyncio.run(self.client.close())
            except Exception:
                _logger.exception("Failed to close notification client")
            self.client = None

    async def _send_notifications(self, title: str, message: str, attachments: dict):
        await self.client.send_notification(
            notification_type="alert",
            title=title,
            message=message,
            priority="high",
            channels=['telegram', 'email'],
            recipient_id=self.user_id,
            attachments=attachments or None,
            source="emps3_pipeline"
        )
