"""
Alert Module - Send notifications for Phase transitions

Integrates with the project's notification system to send alerts via:
- Telegram
- Email (with CSV attachments)
"""

from pathlib import Path
from typing import Optional
import pandas as pd

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class EMPS2AlertSender:
    """
    Send alerts via Telegram and Email for Phase 2 transitions.

    Integrates with:
    - src.notification.telegram_notifier
    - src.notification.email_notifier
    """

    def __init__(self):
        """Initialize alert sender with Telegram and Email notifiers."""
        self.telegram = None
        self.email = None

        # Try to import Telegram notifier
        try:
            from src.notification.telegram_notifier import TelegramNotifier
            self.telegram = TelegramNotifier()
            _logger.info("Telegram notifier initialized")
        except Exception:
            _logger.warning("Telegram notifier not available", exc_info=True)

        # Try to import Email notifier
        try:
            from src.notification.email_notifier import EmailNotifier
            self.email = EmailNotifier()
            _logger.info("Email notifier initialized")
        except Exception:
            _logger.warning("Email notifier not available", exc_info=True)

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

        count = len(phase2_df)
        top_tickers = phase2_df['ticker'].head(10).tolist()

        # Build message
        message = f"""
🔥 EMPS2 PHASE 2 ALERT 🔥

Detected {count} ticker{'s' if count > 1 else ''} transitioning to Phase 2 (Early Public Signal)

Top candidates:
{', '.join(top_tickers)}

These tickers showed persistent accumulation (5+ days) and are now showing:
- Volume acceleration (Z-Score >3.0)
- Sentiment rising or going viral

See attached CSV for full details.
"""

        # Send Telegram
        if self.telegram:
            try:
                self.telegram.send_message(message)
                _logger.info("Sent Telegram message for %d Phase 2 candidates", count)

                # Send file if path provided
                if phase2_csv_path and phase2_csv_path.exists():
                    self.telegram.send_file(
                        file_path=str(phase2_csv_path),
                        caption=f"Phase 2 Alerts - {count} candidates"
                    )
                    _logger.info("Sent Telegram file attachment")
            except Exception:
                _logger.exception("Failed to send Telegram alert")
        else:
            _logger.warning("Telegram not configured, skipping Telegram alert")

        # Send Email
        if self.email:
            try:
                attachments = []
                if phase2_csv_path and phase2_csv_path.exists():
                    attachments.append(str(phase2_csv_path))

                self.email.send_email(
                    subject=f"EMPS2 Phase 2 Alert - {count} Hot Candidate{'s' if count > 1 else ''}",
                    body=message,
                    attachments=attachments if attachments else None
                )
                _logger.info("Sent Email alert for %d Phase 2 candidates", count)
            except Exception:
                _logger.exception("Failed to send Email alert")
        else:
            _logger.warning("Email not configured, skipping Email alert")

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

        count = len(phase1_df)
        top_tickers = phase1_df['ticker'].head(10).tolist()

        # Build message
        message = f"""
📊 EMPS2 PHASE 1 UPDATE

Detected {count} ticker{'s' if count > 1 else ''} in Phase 1 (Quiet Accumulation)

Top candidates:
{', '.join(top_tickers)}

These tickers appeared 5+ times in the last 10 days, showing persistent accumulation patterns.

Watchlist updated. See attached CSV for full details.
"""

        # Send Telegram
        if self.telegram:
            try:
                self.telegram.send_message(message)
                _logger.info("Sent Telegram Phase 1 alert")

                if phase1_csv_path and phase1_csv_path.exists():
                    self.telegram.send_file(
                        file_path=str(phase1_csv_path),
                        caption=f"Phase 1 Watchlist - {count} candidates"
                    )
            except Exception:
                _logger.exception("Failed to send Telegram Phase 1 alert")

        # Send Email
        if self.email:
            try:
                attachments = []
                if phase1_csv_path and phase1_csv_path.exists():
                    attachments.append(str(phase1_csv_path))

                self.email.send_email(
                    subject=f"EMPS2 Phase 1 Watchlist - {count} Candidate{'s' if count > 1 else ''}",
                    body=message,
                    attachments=attachments if attachments else None
                )
                _logger.info("Sent Email Phase 1 alert")
            except Exception:
                _logger.exception("Failed to send Email Phase 1 alert")
