"""
SMS Channel Plugin Template

SMS notification channel implementation with Twilio support.
Provides a template for SMS providers with message length validation and delivery tracking.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timezone
import asyncio
import aiohttp

from src.notification.channels.base import (
    NotificationChannel, DeliveryResult, ChannelHealth, MessageContent,
    DeliveryStatus, ChannelHealthStatus
)
from src.notification.channels.config import (
    ConfigValidator, CommonValidationRules, validate_phone
)
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class SMSChannel(NotificationChannel):
    """
    SMS notification channel with Twilio support.

    Supports:
    - Message length validation and splitting
    - Delivery confirmation tracking
    - Multiple SMS providers (Twilio, AWS SNS, etc.)
    - International phone number support
    - Health monitoring via API test
    """

    # SMS limits
    MAX_SMS_LENGTH = 160  # Standard SMS length
    MAX_CONCATENATED_LENGTH = 1600  # Typical limit for concatenated SMS
    RATE_LIMIT_DEFAULT = 5  # SMS per minute (conservative)

    # Supported providers
    SUPPORTED_PROVIDERS = ["twilio", "aws_sns", "nexmo", "messagebird"]

    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate SMS channel configuration.

        Args:
            config: Configuration dictionary to validate

        Raises:
            ValueError: If configuration is invalid
        """
        validator = ConfigValidator(self.channel_name)

        # Provider selection
        validator.require_field(
            "provider",
            str,
            description="SMS provider (twilio, aws_sns, nexmo, messagebird)",
            allowed_values=self.SUPPORTED_PROVIDERS
        )

        # Common fields
        validator.optional_field(
            "default_from_number",
            str,
            description="Default sender phone number",
            custom_validator=validate_phone
        )

        validator.optional_field(
            "default_recipient",
            str,
            description="Default recipient phone number",
            custom_validator=validate_phone
        )

        # Provider-specific configuration
        provider = config.get("provider", "twilio")

        if provider == "twilio":
            self._add_twilio_validation(validator)
        elif provider == "aws_sns":
            self._add_aws_sns_validation(validator)
        elif provider == "nexmo":
            self._add_nexmo_validation(validator)
        elif provider == "messagebird":
            self._add_messagebird_validation(validator)

        # SMS options
        validator.optional_field(
            "enable_delivery_reports",
            bool,
            description="Enable delivery status callbacks"
        )

        validator.optional_field(
            "max_message_parts",
            int,
            description="Maximum number of SMS parts for long messages",
            min_value=1,
            max_value=10
        )

        validator.optional_field(
            "validity_period_hours",
            int,
            description="Message validity period in hours",
            min_value=1,
            max_value=72
        )

        # Add common validation rules
        validator.add_rule(CommonValidationRules.timeout_seconds(default=30))
        validator.add_rule(CommonValidationRules.rate_limit(default=self.RATE_LIMIT_DEFAULT))
        validator.add_rule(CommonValidationRules.max_retries(default=3))

        # Validate and update config
        validated_config = validator.validate(config)
        self.config.update(validated_config)

        # Set defaults
        if "enable_delivery_reports" not in self.config:
            self.config["enable_delivery_reports"] = True

        if "max_message_parts" not in self.config:
            self.config["max_message_parts"] = 5

    def _add_twilio_validation(self, validator: ConfigValidator):
        """Add Twilio-specific validation rules."""
        validator.require_field(
            "account_sid",
            str,
            description="Twilio Account SID",
            min_length=30,
            max_length=40
        )

        validator.require_field(
            "auth_token",
            str,
            description="Twilio Auth Token",
            min_length=30,
            max_length=40
        )

        validator.optional_field(
            "messaging_service_sid",
            str,
            description="Twilio Messaging Service SID (alternative to from_number)"
        )

    def _add_aws_sns_validation(self, validator: ConfigValidator):
        """Add AWS SNS-specific validation rules."""
        validator.require_field(
            "aws_access_key_id",
            str,
            description="AWS Access Key ID",
            min_length=16,
            max_length=32
        )

        validator.require_field(
            "aws_secret_access_key",
            str,
            description="AWS Secret Access Key",
            min_length=20
        )

        validator.require_field(
            "aws_region",
            str,
            description="AWS Region",
            min_length=2
        )

    def _add_nexmo_validation(self, validator: ConfigValidator):
        """Add Nexmo/Vonage-specific validation rules."""
        validator.require_field(
            "api_key",
            str,
            description="Nexmo API Key",
            min_length=8
        )

        validator.require_field(
            "api_secret",
            str,
            description="Nexmo API Secret",
            min_length=8
        )

    def _add_messagebird_validation(self, validator: ConfigValidator):
        """Add MessageBird-specific validation rules."""
        validator.require_field(
            "access_key",
            str,
            description="MessageBird Access Key",
            min_length=20
        )

    async def send_message(
        self,
        recipient: str,
        content: MessageContent,
        message_id: Optional[str] = None,
        priority: str = "NORMAL"
    ) -> DeliveryResult:
        """
        Send an SMS message.

        Args:
            recipient: Recipient phone number
            content: Message content to send
            message_id: Optional message ID for tracking
            priority: Message priority (affects retry behavior)

        Returns:
            DeliveryResult with delivery status and metadata
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Use recipient or fall back to default
            to_number = recipient or self.config.get("default_recipient")
            if not to_number:
                raise ValueError("No recipient phone number provided")

            # Validate phone number
            validate_phone(to_number)

            # Get sender number
            from_number = self.config.get("default_from_number")
            messaging_service_sid = self.config.get("messaging_service_sid")

            if not from_number and not messaging_service_sid:
                raise ValueError("No sender phone number or messaging service configured")

            # Handle long messages
            message_parts = self._split_sms_message(content.text)

            if len(message_parts) > self.config["max_message_parts"]:
                raise ValueError(f"Message too long: {len(message_parts)} parts (max: {self.config['max_message_parts']})")

            # Send message(s) based on provider
            provider = self.config["provider"]

            if provider == "twilio":
                result = await self._send_twilio_sms(
                    to_number, from_number, messaging_service_sid, message_parts, message_id
                )
            elif provider == "aws_sns":
                result = await self._send_aws_sns_sms(
                    to_number, message_parts, message_id
                )
            elif provider == "nexmo":
                result = await self._send_nexmo_sms(
                    to_number, from_number, message_parts, message_id
                )
            elif provider == "messagebird":
                result = await self._send_messagebird_sms(
                    to_number, from_number, message_parts, message_id
                )
            else:
                raise ValueError(f"Unsupported SMS provider: {provider}")

            response_time = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            result.response_time_ms = response_time

            return result

        except ValueError as e:
            error_msg = str(e)
            _logger.warning("SMS validation error: %s", error_msg)

            return DeliveryResult(
                success=False,
                status=DeliveryStatus.FAILED,
                error_message=error_msg,
                response_time_ms=int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            )

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            _logger.exception("Unexpected error sending SMS: %s", error_msg)

            return DeliveryResult(
                success=False,
                status=DeliveryStatus.FAILED,
                error_message=error_msg,
                response_time_ms=int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            )

    def _split_sms_message(self, text: str) -> list[str]:
        """
        Split long SMS message into parts.

        Args:
            text: Message text to split

        Returns:
            List of message parts
        """
        if len(text) <= self.MAX_SMS_LENGTH:
            return [text]

        # For concatenated SMS, each part is slightly shorter due to headers
        part_length = 153  # Standard for concatenated SMS
        parts = []

        while text:
            if len(text) <= part_length:
                parts.append(text)
                break

            # Find a good break point (space, punctuation)
            break_point = part_length
            for i in range(part_length - 1, max(0, part_length - 20), -1):
                if text[i] in [' ', '.', ',', ';', '!', '?', '\n']:
                    break_point = i
                    break

            parts.append(text[:break_point].strip())
            text = text[break_point:].strip()

        return parts

    async def _send_twilio_sms(
        self,
        to_number: str,
        from_number: Optional[str],
        messaging_service_sid: Optional[str],
        message_parts: list[str],
        message_id: Optional[str]
    ) -> DeliveryResult:
        """Send SMS via Twilio API."""
        try:
            import base64

            # Twilio API credentials
            account_sid = self.config["account_sid"]
            auth_token = self.config["auth_token"]

            # Create auth header
            credentials = f"{account_sid}:{auth_token}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()

            headers = {
                "Authorization": f"Basic {encoded_credentials}",
                "Content-Type": "application/x-www-form-urlencoded"
            }

            url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"

            sent_messages = []

            async with aiohttp.ClientSession() as session:
                for i, part in enumerate(message_parts):
                    # Prepare form data
                    data = {
                        "To": to_number,
                        "Body": part
                    }

                    if messaging_service_sid:
                        data["MessagingServiceSid"] = messaging_service_sid
                    elif from_number:
                        data["From"] = from_number

                    if self.config.get("enable_delivery_reports"):
                        data["StatusCallback"] = "https://your-webhook-url.com/sms/status"  # Configure as needed

                    # Send request
                    async with session.post(url, headers=headers, data=data) as response:
                        if response.status == 201:
                            result = await response.json()
                            sent_messages.append(result["sid"])
                        else:
                            error_text = await response.text()
                            raise Exception(f"Twilio API error {response.status}: {error_text}")

                    # Small delay between parts
                    if i < len(message_parts) - 1:
                        await asyncio.sleep(1)

            return DeliveryResult(
                success=True,
                status=DeliveryStatus.SENT,  # Twilio returns 'sent' initially
                external_id=",".join(sent_messages),
                metadata={
                    "provider": "twilio",
                    "to_number": to_number,
                    "from_number": from_number,
                    "messaging_service_sid": messaging_service_sid,
                    "message_parts": len(message_parts),
                    "message_sids": sent_messages
                }
            )

        except Exception as e:
            return DeliveryResult(
                success=False,
                status=DeliveryStatus.FAILED,
                error_message=f"Twilio error: {str(e)}"
            )

    async def _send_aws_sns_sms(
        self,
        to_number: str,
        message_parts: list[str],
        message_id: Optional[str]
    ) -> DeliveryResult:
        """Send SMS via AWS SNS (placeholder implementation)."""
        # This is a template - implement actual AWS SNS integration
        _logger.warning("AWS SNS SMS implementation not yet available")

        return DeliveryResult(
            success=False,
            status=DeliveryStatus.FAILED,
            error_message="AWS SNS SMS provider not implemented"
        )

    async def _send_nexmo_sms(
        self,
        to_number: str,
        from_number: Optional[str],
        message_parts: list[str],
        message_id: Optional[str]
    ) -> DeliveryResult:
        """Send SMS via Nexmo/Vonage (placeholder implementation)."""
        # This is a template - implement actual Nexmo integration
        _logger.warning("Nexmo SMS implementation not yet available")

        return DeliveryResult(
            success=False,
            status=DeliveryStatus.FAILED,
            error_message="Nexmo SMS provider not implemented"
        )

    async def _send_messagebird_sms(
        self,
        to_number: str,
        from_number: Optional[str],
        message_parts: list[str],
        message_id: Optional[str]
    ) -> DeliveryResult:
        """Send SMS via MessageBird (placeholder implementation)."""
        # This is a template - implement actual MessageBird integration
        _logger.warning("MessageBird SMS implementation not yet available")

        return DeliveryResult(
            success=False,
            status=DeliveryStatus.FAILED,
            error_message="MessageBird SMS provider not implemented"
        )

    async def check_health(self) -> ChannelHealth:
        """
        Check SMS channel health by testing provider API.

        Returns:
            ChannelHealth with current status and metrics
        """
        start_time = datetime.now(timezone.utc)

        try:
            provider = self.config["provider"]

            if provider == "twilio":
                health = await self._check_twilio_health()
            elif provider == "aws_sns":
                health = await self._check_aws_sns_health()
            elif provider == "nexmo":
                health = await self._check_nexmo_health()
            elif provider == "messagebird":
                health = await self._check_messagebird_health()
            else:
                raise ValueError(f"Unsupported provider: {provider}")

            response_time = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            health.response_time_ms = response_time
            health.last_check = datetime.now(timezone.utc)

            return health

        except Exception as e:
            response_time = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)

            return ChannelHealth(
                status=ChannelHealthStatus.DOWN,
                last_check=datetime.now(timezone.utc),
                response_time_ms=response_time,
                error_message=f"Health check failed: {str(e)}"
            )

    async def _check_twilio_health(self) -> ChannelHealth:
        """Check Twilio API health."""
        try:
            import base64

            account_sid = self.config["account_sid"]
            auth_token = self.config["auth_token"]

            credentials = f"{account_sid}:{auth_token}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()

            headers = {
                "Authorization": f"Basic {encoded_credentials}"
            }

            url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}.json"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        account_info = await response.json()

                        return ChannelHealth(
                            status=ChannelHealthStatus.HEALTHY,
                            last_check=datetime.now(timezone.utc),
                            metadata={
                                "provider": "twilio",
                                "account_sid": account_sid,
                                "account_status": account_info.get("status"),
                                "account_type": account_info.get("type")
                            }
                        )
                    else:
                        error_text = await response.text()
                        return ChannelHealth(
                            status=ChannelHealthStatus.DOWN,
                            last_check=datetime.now(timezone.utc),
                            error_message=f"Twilio API error {response.status}: {error_text}"
                        )

        except Exception as e:
            return ChannelHealth(
                status=ChannelHealthStatus.DOWN,
                last_check=datetime.now(timezone.utc),
                error_message=f"Twilio health check error: {str(e)}"
            )

    async def _check_aws_sns_health(self) -> ChannelHealth:
        """Check AWS SNS health (placeholder)."""
        return ChannelHealth(
            status=ChannelHealthStatus.DOWN,
            last_check=datetime.now(timezone.utc),
            error_message="AWS SNS health check not implemented"
        )

    async def _check_nexmo_health(self) -> ChannelHealth:
        """Check Nexmo health (placeholder)."""
        return ChannelHealth(
            status=ChannelHealthStatus.DOWN,
            last_check=datetime.now(timezone.utc),
            error_message="Nexmo health check not implemented"
        )

    async def _check_messagebird_health(self) -> ChannelHealth:
        """Check MessageBird health (placeholder)."""
        return ChannelHealth(
            status=ChannelHealthStatus.DOWN,
            last_check=datetime.now(timezone.utc),
            error_message="MessageBird health check not implemented"
        )

    def get_rate_limit(self) -> int:
        """
        Get rate limit for SMS channel.

        Returns:
            Rate limit in messages per minute
        """
        return self.config.get("rate_limit_per_minute", self.RATE_LIMIT_DEFAULT)

    def supports_feature(self, feature: str) -> bool:
        """
        Check if SMS channel supports a feature.

        Args:
            feature: Feature name to check

        Returns:
            True if feature is supported
        """
        supported_features = {
            "splitting": True,
            "delivery_reports": True,
            "international": True,
            "concatenated_sms": True,
            "priority": False,  # SMS doesn't have native priority
            "attachments": False,  # Standard SMS doesn't support attachments
            "html": False,  # SMS is plain text only
            "formatting": False,  # Limited formatting in SMS
            "read_receipts": False,  # Not supported in SMS
            "bulk_sending": True,
            "scheduling": False  # Would need provider-specific implementation
        }

        return supported_features.get(feature, False)

    def get_max_message_length(self) -> int:
        """
        Get maximum message length for SMS.

        Returns:
            Maximum message length in characters
        """
        return self.MAX_CONCATENATED_LENGTH

    def format_message(self, content: MessageContent) -> MessageContent:
        """
        Format message for SMS (plain text only).

        Args:
            content: Original message content

        Returns:
            Formatted message content (plain text)
        """
        # SMS only supports plain text, strip HTML if present
        text = content.text

        # If HTML is provided, try to extract plain text
        if content.html and not text:
            # Simple HTML to text conversion
            import re
            text = re.sub(r'<[^>]+>', '', content.html)
            text = text.replace('&nbsp;', ' ').replace('&amp;', '&')
            text = text.replace('&lt;', '<').replace('&gt;', '>')

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return MessageContent(
            text=text,
            subject=None,  # SMS doesn't have subjects
            html=None,     # SMS doesn't support HTML
            attachments=None,  # SMS doesn't support attachments
            metadata=content.metadata
        )