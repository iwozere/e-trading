"""
Sends trade notifications and alerts via email using SendGrid.

This module defines EmailNotifier, which can send trade and system notifications to configured email addresses for trading bots.

NOTE: The preferred way to send emails is now via the async notification system (AsyncNotificationManager).
Direct use of EmailNotifier is deprecated for new code. Use AsyncNotificationManager for all new notification logic.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

from config.donotshare.donotshare import SMTP_PASSWORD, SMTP_PORT, SMTP_SERVER, SMTP_USER
from src.notification.logger import setup_logger
_logger = setup_logger(__name__)

class EmailNotifier:
    """
    DEPRECATED: Use AsyncNotificationManager for all new notification logic.
    EmailNotifier provides synchronous email sending for legacy code only.
    """
    def __init__(self):
        self.smtp_server = SMTP_SERVER
        self.smtp_port = SMTP_PORT
        self.sender_email = SMTP_USER
        self.sender_password = SMTP_PASSWORD

    def send_email(self, to_addr: str, subject: str, body: str, from_name: str = None, attachments: list = None) -> bool:
        """Send an email with the given subject and body to the specified recipient."""
        msg = MIMEMultipart()
        if from_name:
            msg["From"] = f"{from_name} <{self.sender_email}>"
        else:
            msg["From"] = self.sender_email
        msg["To"] = to_addr
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "html"))
        # Attach files if provided
        if attachments:
            for file_path in attachments:
                try:
                    with open(file_path, "rb") as f:
                        part = MIMEBase("application", "octet-stream")
                        part.set_payload(f.read())
                    encoders.encode_base64(part)
                    part.add_header(
                        "Content-Disposition",
                        f"attachment; filename={os.path.basename(file_path)}",
                    )
                    msg.attach(part)
                except Exception as e:
                    _logger.error("Failed to attach file %s: %s", file_path, e, exc_info=True)
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.ehlo()
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            _logger.info("Email sent to %s with subject: %s", to_addr, subject)
            return True
        except Exception as e:
            _logger.error("Failed to send email: %s", e, exc_info=True)
            return False

    def send_notification_email(self, buy_or_sell: str, symbol: str, price: float, amount: float, to_addr: str, body: str = ""):
        """Send a trade notification email with the provided trade data."""
        subject = f"Trade notification: {buy_or_sell} {symbol} at {price} with {amount}"
        self.send_email(to_addr, subject, body)
        _logger.debug("Email sent successfully: %s", subject)

    def send_email_with_mime(self, to_addr: str, subject: str, body: str, from_name: str = None, attachments: list = None) -> bool:
        """Send an email with the given subject, body, and MIME attachments to the specified recipient."""
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        msg = MIMEMultipart()
        if from_name:
            msg["From"] = f"{from_name} <{self.sender_email}>"
        else:
            msg["From"] = self.sender_email
        msg["To"] = to_addr
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "html"))
        # Attach MIME parts if provided
        if attachments:
            for filename, mime_part in attachments:
                try:
                    mime_part.add_header("Content-Disposition", f"attachment; filename={filename}")
                    msg.attach(mime_part)
                except Exception as e:
                    _logger.error("Failed to attach MIME part %s: %s", filename, e, exc_info=True)
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.ehlo()
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            _logger.info("Email sent to %s with subject: %s", to_addr, subject)
            return True
        except Exception as e:
            _logger.error("Failed to send email: %s", e, exc_info=True)
            return False

def send_email_alert(receiver_email: str, subject: str, message: str):
    """Send alert email

    Args:
        receiver_email (str): to address
        subject (str): email subject
        message (str): message body
    """
    try:
        notifier = EmailNotifier()
        notifier.send_email(receiver_email, subject, message)
        _logger.info("System alert email sent: %s", subject)
    except Exception as e:
        _logger.error("Failed to send system alert email: %s", e, exc_info=True)

def send_async_email(subject: str, body: str, to: str) -> None:
    """Send an email asynchronously with the given subject, body, and recipient."""
    # Implementation of send_async_email method
    # TODO: implement
    pass

#######################################
# Quick test. Keep commented out.
#######################################
# Example usage
# if __name__ == "__main__":
#     EmailNotifier().send_email(
#         "recipient@example.com", "TA test subject", "TA test body"
#     )
#     _logger.debug("Email sent successfully!")

if __name__ == "__main__":
    import os
    import smtplib
    from email.mime.text import MIMEText
    to_addr = input("Enter recipient email: ").strip()
    subject = "Gmail SMTP Test"
    body = "This is a test email sent using Gmail SMTP and an app password."
    msg = MIMEText(body, "plain")
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = to_addr
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, [to_addr], msg.as_string())
        print(f"Test email sent to {to_addr}")
    except Exception as e:
        print(f"Failed to send test email: {e}")
    finally:
        print("DONE")

