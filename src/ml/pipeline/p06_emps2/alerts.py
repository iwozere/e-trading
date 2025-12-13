"""
Alert Module - Send notifications for Phase transitions

Integrates with the project's notification system to send alerts via:
- Telegram
- Email (with CSV attachments)
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import asyncio

from src.notification.logger import setup_logger
from src.notification.service.client import NotificationServiceClient

_logger = setup_logger(__name__)


class EMPS2AlertSender:
    """
    Send alerts via Telegram and Email for Phase 2 transitions.

    Integrates with:
    - src.notification.service.client.NotificationServiceClient
    """

    def __init__(self, user_id: Optional[str] = None):
        """Initialize alert sender with NotificationServiceClient."""
        self.client = None
        self.user_id = user_id

        # Try to initialize notification client
        try:
            self.client = NotificationServiceClient()
            _logger.info("Notification service client initialized")
        except Exception:
            _logger.warning("Notification service client not available", exc_info=True)

    def send_phase2_alert(
        self,
        phase2_df: pd.DataFrame,
        phase2_csv_path: Optional[Path] = None
    ) -> None:
        """
        Send alert for Phase 2 transitions.

        Args:
            phase2_df: Phase 2 candidates DataFrame
            phase2_csv_path: Path to phase2_alerts.csv (optional, for attachment)
        """
        if phase2_df.empty:
            _logger.info("No Phase 2 alerts to send")
            return

        if not self.client:
            _logger.warning("Notification client not available, skipping alerts")
            return

        count = len(phase2_df)
        top_tickers = phase2_df['ticker'].head(10).tolist()

        # Build message
        title = f"🔥 EMPS2 Phase 2 Alert - {count} Hot Candidate{'s' if count > 1 else ''}"
        message = f"""Detected {count} ticker{'s' if count > 1 else ''} transitioning to Phase 2 (Early Public Signal)

Top candidates:
{', '.join(top_tickers)}

These tickers showed persistent accumulation (5+ days) and are now showing:
- Volume acceleration (Z-Score >3.0)
- Sentiment rising or going viral

See attached CSV for full details."""

        # Prepare attachments if CSV path provided
        attachments = None
        if phase2_csv_path and phase2_csv_path.exists():
            attachments = {
                'files': [str(phase2_csv_path)]
            }

        # Send notification via both Telegram and Email
        try:
            # Run async notification
            asyncio.run(self._send_async_notification(
                title=title,
                message=message,
                attachments=attachments,
                channels=['telegram', 'email']
            ))
            _logger.info("Sent Phase 2 alert for %d candidates", count)
        except Exception:
            _logger.exception("Failed to send Phase 2 alert")

    async def _send_async_notification(
        self,
        title: str,
        message: str,
        attachments: Optional[dict] = None,
        channels: Optional[list] = None
    ) -> None:
        """
        Send async notification via notification service.

        Args:
            title: Notification title
            message: Notification message
            attachments: Optional attachments dict
            channels: List of channels to use
        """
        await self.client.send_notification(
            notification_type="alert",
            title=title,
            message=message,
            priority="high",
            channels=channels,
            recipient_id=self.user_id,
            attachments=attachments,
            source="emps2_pipeline"
        )

    def send_phase1_alert(
        self,
        phase1_df: pd.DataFrame,
        phase1_csv_path: Optional[Path] = None
    ) -> None:
        """
        Send alert for Phase 1 watchlist (optional).

        Args:
            phase1_df: Phase 1 candidates DataFrame
            phase1_csv_path: Path to phase1_watchlist.csv
        """
        if phase1_df.empty:
            _logger.info("No Phase 1 alerts to send")
            return

        if not self.client:
            _logger.warning("Notification client not available, skipping alerts")
            return

        count = len(phase1_df)
        top_tickers = phase1_df['ticker'].head(10).tolist()

        # Build message
        title = f"📊 EMPS2 Phase 1 Watchlist - {count} Candidate{'s' if count > 1 else ''}"
        message = f"""Detected {count} ticker{'s' if count > 1 else ''} in Phase 1 (Quiet Accumulation)

Top candidates:
{', '.join(top_tickers)}

These tickers appeared 5+ times in the last 10 days, showing persistent accumulation patterns.

Watchlist updated. See attached CSV for full details."""

        # Prepare attachments if CSV path provided
        attachments = None
        if phase1_csv_path and phase1_csv_path.exists():
            attachments = {
                'files': [str(phase1_csv_path)]
            }

        # Send notification via both Telegram and Email
        try:
            # Run async notification
            asyncio.run(self._send_async_notification(
                title=title,
                message=message,
                attachments=attachments,
                channels=['telegram', 'email']
            ))
            _logger.info("Sent Phase 1 alert for %d candidates", count)
        except Exception:
            _logger.exception("Failed to send Phase 1 alert")
