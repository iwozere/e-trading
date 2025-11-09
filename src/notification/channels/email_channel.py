"""
Email Channel Plugin

Email notification channel implementation with SMTP support.
Supports HTML formatting, MIME attachments, and health monitoring.
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.application import MIMEApplication
from email.mime.base import MIMEBase
from email import encoders
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import aiosmtplib
from pathlib import Path

from src.notification.channels.base import (
    NotificationChannel, DeliveryResult, ChannelHealth, MessageContent,
    DeliveryStatus, ChannelHealthStatus
)
from src.notification.channels.config import (
    ConfigValidator, CommonValidationRules, validate_email
)
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class EmailChannel(NotificationChannel):
    """
    Email notification channel with SMTP support.

    Supports:
    - HTML and plain text emails
    - MIME attachments (images, documents, etc.)
    - SMTP authentication and TLS/SSL
    - Health monitoring via SMTP connection test
    - Dynamic recipient addresses
    - CC and BCC support
    """

    RATE_LIMIT_DEFAULT = 10  # emails per minute (conservative for SMTP)

    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate Email channel configuration.

        Args:
            config: Configuration dictionary to validate

        Raises:
            ValueError: If configuration is invalid
        """
        validator = ConfigValidator(self.channel_name)

        # SMTP server configuration
        validator.require_field(
            "smtp_host",
            str,
            description="SMTP server hostname",
            min_length=1
        )

        validator.require_field(
            "smtp_port",
            int,
            description="SMTP server port",
            min_value=1,
            max_value=65535
        )

        validator.require_field(
            "smtp_username",
            str,
            description="SMTP authentication username",
            custom_validator=validate_email
        )

        validator.require_field(
            "smtp_password",
            str,
            description="SMTP authentication password",
            min_length=1
        )

        # Sender configuration
        validator.require_field(
            "from_email",
            str,
            description="Sender email address",
            custom_validator=validate_email
        )

        validator.optional_field(
            "from_name",
            str,
            description="Sender display name"
        )

        validator.optional_field(
            "default_recipient",
            str,
            description="Default recipient email address",
            custom_validator=validate_email
        )

        # SMTP options
        validator.optional_field(
            "use_tls",
            bool,
            description="Use TLS encryption (STARTTLS)"
        )

        validator.optional_field(
            "use_ssl",
            bool,
            description="Use SSL encryption (implicit TLS)"
        )

        validator.optional_field(
            "validate_certs",
            bool,
            description="Validate SSL certificates"
        )

        # Email options
        validator.optional_field(
            "default_subject_prefix",
            str,
            description="Prefix to add to all email subjects"
        )

        validator.optional_field(
            "html_template",
            str,
            description="HTML template for email body"
        )

        validator.optional_field(
            "max_attachment_size_mb",
            int,
            description="Maximum attachment size in MB",
            min_value=1,
            max_value=100
        )

        # Add common validation rules
        validator.add_rule(CommonValidationRules.timeout_seconds(default=30))
        validator.add_rule(CommonValidationRules.rate_limit(default=self.RATE_LIMIT_DEFAULT))
        validator.add_rule(CommonValidationRules.max_retries(default=3))

        # Validate and update config
        validated_config = validator.validate(config)
        self.config.update(validated_config)

        # Set defaults
        if "use_tls" not in self.config and "use_ssl" not in self.config:
            # Default to TLS if neither is specified
            self.config["use_tls"] = True

        if "validate_certs" not in self.config:
            self.config["validate_certs"] = True

        if "max_attachment_size_mb" not in self.config:
            self.config["max_attachment_size_mb"] = 25  # Common email limit

    async def send_message(
        self,
        recipient: str,
        content: MessageContent,
        message_id: Optional[str] = None,
        priority: str = "NORMAL"
    ) -> DeliveryResult:
        """
        Send an email message.

        Args:
            recipient: Recipient email address (or comma-separated list)
            content: Message content to send
            message_id: Optional message ID for tracking
            priority: Message priority (affects retry behavior)

        Returns:
            DeliveryResult with delivery status and metadata
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Use recipient or fall back to default
            to_email = recipient or self.config.get("default_recipient")
            if not to_email:
                raise ValueError("No recipient email address provided")

            # Parse recipients (support comma-separated)
            recipients = [email.strip() for email in to_email.split(",")]

            # Validate all recipients
            for email_addr in recipients:
                validate_email(email_addr)

            # Extract metadata for email-specific options
            metadata = content.metadata or {}
            cc_emails = metadata.get("cc", [])
            bcc_emails = metadata.get("bcc", [])
            reply_to = metadata.get("reply_to")

            # Create email message
            msg = await self._create_email_message(
                recipients, content, cc_emails, bcc_emails, reply_to, message_id
            )

            # Send email
            await self._send_email_async(msg, recipients + cc_emails + bcc_emails)

            response_time = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)

            return DeliveryResult(
                success=True,
                status=DeliveryStatus.DELIVERED,
                external_id=msg.get("Message-ID", f"email_{int(start_time.timestamp())}"),
                response_time_ms=response_time,
                metadata={
                    "recipients": recipients,
                    "cc": cc_emails,
                    "bcc": bcc_emails,
                    "subject": msg.get("Subject"),
                    "message_id": msg.get("Message-ID"),
                    "attachment_count": len(content.attachments) if content.attachments else 0
                }
            )

        except smtplib.SMTPAuthenticationError as e:
            error_msg = f"SMTP authentication failed: {str(e)}"
            _logger.error("SMTP authentication error: %s", error_msg)

            return DeliveryResult(
                success=False,
                status=DeliveryStatus.FAILED,
                error_message=error_msg,
                response_time_ms=int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            )

        except smtplib.SMTPRecipientsRefused as e:
            error_msg = f"Recipients refused: {str(e)}"
            _logger.warning("SMTP recipients refused: %s", error_msg)

            return DeliveryResult(
                success=False,
                status=DeliveryStatus.BOUNCED,
                error_message=error_msg,
                response_time_ms=int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            )

        except (smtplib.SMTPException, aiosmtplib.SMTPException) as e:
            error_msg = f"SMTP error: {str(e)}"
            _logger.error("SMTP error sending email: %s", error_msg)

            return DeliveryResult(
                success=False,
                status=DeliveryStatus.FAILED,
                error_message=error_msg,
                response_time_ms=int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            )

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            _logger.exception("Unexpected error sending email: %s", error_msg)

            return DeliveryResult(
                success=False,
                status=DeliveryStatus.FAILED,
                error_message=error_msg,
                response_time_ms=int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            )

    async def _create_email_message(
        self,
        recipients: List[str],
        content: MessageContent,
        cc_emails: List[str],
        bcc_emails: List[str],
        reply_to: Optional[str],
        message_id: Optional[str]
    ) -> MIMEMultipart:
        """Create email message with attachments."""
        # Create message
        msg = MIMEMultipart("mixed")

        # Set headers
        from_name = self.config.get("from_name")
        if from_name:
            msg["From"] = f"{from_name} <{self.config['from_email']}>"
        else:
            msg["From"] = self.config["from_email"]

        msg["To"] = ", ".join(recipients)

        if cc_emails:
            msg["Cc"] = ", ".join(cc_emails)

        if reply_to:
            msg["Reply-To"] = reply_to

        # Subject with optional prefix
        subject = content.subject or "Notification"
        subject_prefix = self.config.get("default_subject_prefix")
        if subject_prefix:
            subject = f"{subject_prefix} {subject}"
        msg["Subject"] = subject

        # Set message ID for tracking
        if message_id:
            msg["Message-ID"] = f"<{message_id}@{self.config['smtp_host']}>"

        # Create body
        body_part = MIMEMultipart("alternative")

        # Plain text version
        text_part = MIMEText(content.text, "plain", "utf-8")
        body_part.attach(text_part)

        # HTML version
        if content.html:
            html_content = content.html
        else:
            # Convert plain text to HTML
            html_content = content.text.replace("\n", "<br>")

        # Apply HTML template if configured
        html_template = self.config.get("html_template")
        if html_template:
            html_content = html_template.format(
                content=html_content,
                subject=subject,
                from_name=from_name or self.config["from_email"]
            )
        else:
            # Simple HTML wrapper
            html_content = f"""
            <html>
                <body>
                    <div style="font-family: Arial, sans-serif; line-height: 1.6;">
                        {html_content}
                    </div>
                </body>
            </html>
            """

        html_part = MIMEText(html_content, "html", "utf-8")
        body_part.attach(html_part)

        msg.attach(body_part)

        # Add attachments
        if content.attachments:
            await self._add_attachments(msg, content.attachments)

        return msg

    async def _add_attachments(self, msg: MIMEMultipart, attachments: Dict[str, Any]):
        """Add attachments to email message."""
        max_size_bytes = self.config["max_attachment_size_mb"] * 1024 * 1024

        for filename, attachment_data in attachments.items():
            try:
                if isinstance(attachment_data, bytes):
                    # Bytes data
                    if len(attachment_data) > max_size_bytes:
                        _logger.warning(
                            "Attachment %s too large (%d bytes), skipping",
                            filename, len(attachment_data)
                        )
                        continue

                    # Determine MIME type based on file extension
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                        attachment = MIMEImage(attachment_data)
                        attachment.add_header("Content-Disposition", f"attachment; filename={filename}")
                    else:
                        attachment = MIMEApplication(attachment_data)
                        attachment.add_header("Content-Disposition", f"attachment; filename={filename}")

                    msg.attach(attachment)

                elif isinstance(attachment_data, str):
                    # File path
                    file_path = Path(attachment_data)
                    if not file_path.exists():
                        _logger.warning("Attachment file not found: %s", attachment_data)
                        continue

                    if file_path.stat().st_size > max_size_bytes:
                        _logger.warning(
                            "Attachment file %s too large (%d bytes), skipping",
                            attachment_data, file_path.stat().st_size
                        )
                        continue

                    with open(file_path, "rb") as f:
                        file_data = f.read()

                    # Determine MIME type
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                        attachment = MIMEImage(file_data)
                    elif filename.lower().endswith(('.txt', '.log')):
                        attachment = MIMEText(file_data.decode('utf-8', errors='ignore'))
                    else:
                        attachment = MIMEBase("application", "octet-stream")
                        attachment.set_payload(file_data)
                        encoders.encode_base64(attachment)

                    attachment.add_header("Content-Disposition", f"attachment; filename={filename}")
                    msg.attach(attachment)

            except Exception as e:
                _logger.error("Failed to attach file %s: %s", filename, e)

    async def _send_email_async(self, msg: MIMEMultipart, all_recipients: List[str]):
        """Send email using aiosmtplib for async operation."""
        smtp_kwargs = {
            "hostname": self.config["smtp_host"],
            "port": self.config["smtp_port"],
            "username": self.config["smtp_username"],
            "password": self.config["smtp_password"],
            "timeout": self.config.get("timeout_seconds", 30)
        }

        # SSL/TLS configuration
        if self.config.get("use_ssl"):
            smtp_kwargs["use_tls"] = True
            smtp_kwargs["tls_context"] = ssl.create_default_context()
            if not self.config.get("validate_certs", True):
                smtp_kwargs["tls_context"].check_hostname = False
                smtp_kwargs["tls_context"].verify_mode = ssl.CERT_NONE
        elif self.config.get("use_tls"):
            smtp_kwargs["start_tls"] = True
            smtp_kwargs["tls_context"] = ssl.create_default_context()
            if not self.config.get("validate_certs", True):
                smtp_kwargs["tls_context"].check_hostname = False
                smtp_kwargs["tls_context"].verify_mode = ssl.CERT_NONE

        # Send email
        await aiosmtplib.send(
            msg,
            recipients=all_recipients,
            **smtp_kwargs
        )

    async def check_health(self) -> ChannelHealth:
        """
        Check Email channel health by testing SMTP connection.

        Returns:
            ChannelHealth with current status and metrics
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Test SMTP connection
            smtp_kwargs = {
                "hostname": self.config["smtp_host"],
                "port": self.config["smtp_port"],
                "timeout": self.config.get("timeout_seconds", 30)
            }

            # SSL/TLS configuration
            if self.config.get("use_ssl"):
                smtp_kwargs["use_tls"] = True
                smtp_kwargs["tls_context"] = ssl.create_default_context()
                if not self.config.get("validate_certs", True):
                    smtp_kwargs["tls_context"].check_hostname = False
                    smtp_kwargs["tls_context"].verify_mode = ssl.CERT_NONE
            elif self.config.get("use_tls"):
                smtp_kwargs["start_tls"] = True
                smtp_kwargs["tls_context"] = ssl.create_default_context()
                if not self.config.get("validate_certs", True):
                    smtp_kwargs["tls_context"].check_hostname = False
                    smtp_kwargs["tls_context"].verify_mode = ssl.CERT_NONE

            # Connect and authenticate
            async with aiosmtplib.SMTP(**smtp_kwargs) as smtp:
                await smtp.login(self.config["smtp_username"], self.config["smtp_password"])

            response_time = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)

            return ChannelHealth(
                status=ChannelHealthStatus.HEALTHY,
                last_check=datetime.now(timezone.utc),
                response_time_ms=response_time,
                metadata={
                    "smtp_host": self.config["smtp_host"],
                    "smtp_port": self.config["smtp_port"],
                    "use_tls": self.config.get("use_tls", False),
                    "use_ssl": self.config.get("use_ssl", False),
                    "from_email": self.config["from_email"]
                }
            )

        except (smtplib.SMTPAuthenticationError, aiosmtplib.SMTPAuthenticationError) as e:
            response_time = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            error_msg = f"SMTP authentication failed: {str(e)}"

            return ChannelHealth(
                status=ChannelHealthStatus.DOWN,
                last_check=datetime.now(timezone.utc),
                response_time_ms=response_time,
                error_message=error_msg
            )

        except (smtplib.SMTPException, aiosmtplib.SMTPException) as e:
            response_time = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            error_msg = f"SMTP error: {str(e)}"

            # Determine if this is temporary or permanent
            if "timeout" in str(e).lower() or "connection" in str(e).lower():
                status = ChannelHealthStatus.DEGRADED
            else:
                status = ChannelHealthStatus.DOWN

            return ChannelHealth(
                status=status,
                last_check=datetime.now(timezone.utc),
                response_time_ms=response_time,
                error_message=error_msg
            )

        except Exception as e:
            response_time = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            error_msg = f"Unexpected error: {str(e)}"

            return ChannelHealth(
                status=ChannelHealthStatus.DOWN,
                last_check=datetime.now(timezone.utc),
                response_time_ms=response_time,
                error_message=error_msg
            )

    def get_rate_limit(self) -> int:
        """
        Get rate limit for Email channel.

        Returns:
            Rate limit in messages per minute
        """
        return self.config.get("rate_limit_per_minute", self.RATE_LIMIT_DEFAULT)

    def supports_feature(self, feature: str) -> bool:
        """
        Check if Email channel supports a feature.

        Args:
            feature: Feature name to check

        Returns:
            True if feature is supported
        """
        supported_features = {
            "html": True,
            "attachments": True,
            "images": True,
            "documents": True,
            "cc": True,
            "bcc": True,
            "reply_to": True,
            "formatting": True,
            "templates": True,
            "priority": False,  # SMTP doesn't have native priority
            "read_receipts": True,  # Can request but not guaranteed
            "delivery_receipts": False,  # Not reliable
            "encryption": True,  # TLS/SSL support
            "large_attachments": True,
            "bulk_sending": True
        }

        return supported_features.get(feature, False)

    def get_max_message_length(self) -> Optional[int]:
        """
        Get maximum message length for Email.

        Returns:
            None (no practical limit for email body)
        """
        return None  # Email doesn't have a practical message length limit

    def format_message(self, content: MessageContent) -> MessageContent:
        """
        Format message for Email (ensure HTML is available).

        Args:
            content: Original message content

        Returns:
            Formatted message content with HTML version
        """
        # If no HTML provided, create simple HTML version
        if not content.html and content.text:
            html_content = content.text.replace("\n", "<br>")

            return MessageContent(
                text=content.text,
                subject=content.subject,
                html=html_content,
                attachments=content.attachments,
                metadata=content.metadata
            )

        return content