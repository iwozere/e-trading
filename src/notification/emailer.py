"""
Sends trade notifications and alerts via email using SendGrid.

This module defines EmailNotifier, which can send trade and system notifications to configured email addresses for trading bots.

NOTE: The preferred way to send emails is now via the async notification system (AsyncNotificationManager).
Direct use of EmailNotifier is deprecated for new code. Use AsyncNotificationManager for all new notification logic.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.notification.logger import setup_logger
from config.donotshare.donotshare import SMTP_PASSWORD, SMTP_PORT, SMTP_SERVER, SMTP_USER

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

_logger = setup_logger(__name__)

class EmailNotifier:
    """
    DEPRECATED: Use AsyncNotificationManager for all new notification logic.
    EmailNotifier provides synchronous email sending for legacy code only.
    """
    def __init__(self):
        self.logger = _logger
        self.smtp_server = SMTP_SERVER
        self.smtp_port = SMTP_PORT
        self.sender_email = SMTP_USER
        self.sender_password = SMTP_PASSWORD

    def send_email(self, to_addr: str, subject: str, body: str, from_name: str = None, attachments: list = None):
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
                    self.logger.error(f"Failed to attach file {file_path}: {e}")
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.ehlo()
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            self.logger.info(f"Email sent to {to_addr} with subject: {subject}")
        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")

    def send_notification_email(self, buy_or_sell: str, symbol: str, price: float, amount: float, to_addr: str, body: str = ""):
        subject = f"Trade notification: {buy_or_sell} {symbol} at {price} with {amount}"
        self.send_email(to_addr, subject, body)
        self.logger.debug(f"Email sent successfully: {subject}")

def send_email_alert(receiver_email: str, subject: str, message: str):
    try:
        notifier = EmailNotifier()
        notifier.send_email(receiver_email, subject, message)
        _logger.info(f"System alert email sent: {subject}")
    except Exception as e:
        _logger.error("Failed to send system alert email: %s", e, exc_info=True)

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
        print(f"DONE")
