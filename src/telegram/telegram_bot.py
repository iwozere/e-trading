import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import time
from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.filters import Command
import asyncio
import random
from src.notification.service.client import NotificationServiceClient, MessageType, MessagePriority
from config.donotshare.donotshare import TELEGRAM_BOT_TOKEN
from src.telegram.screener.notifications import (
    process_report_command, process_screener_command
)
# Immediate handlers moved inline since immediate_handlers.py was removed
from src.telegram.screener import business_logic

# Service layer imports
# telegram_service is imported in initialize_services() to get the instance
from src.indicators.service import IndicatorService

# Configure logging
from src.notification.logger import setup_logger, set_logging_context
_logger = setup_logger("telegram_screener_bot")

# Inline immediate handlers (replacing removed immediate_handlers.py)
async def process_info_command_immediate(user_id: str, message):
    """Info command handler with immediate response."""
    try:
        # Get service instances
        telegram_svc, _ = get_service_instances()

        if not telegram_svc:
            await message.reply("❌ Service temporarily unavailable. Please try again later.")
            return

        # Get user status
        status = telegram_svc.get_user_status(user_id)

        if status:
            email = status["email"] or "(not set)"
            verified = "✅ Yes" if status["verified"] else "❌ No"
            approved = "✅ Yes" if status["approved"] else "❌ No"
            admin = "✅ Yes" if status["is_admin"] else "❌ No"
            language = status["language"] or "(not set)"

            info_text = f"""ℹ️ **Your Account Information**

📧 **Email:** {email}
✅ **Verified:** {verified}
👤 **Approved:** {approved}
🔧 **Admin:** {admin}
🌐 **Language:** {language}

Use /help to see available commands."""
        else:
            info_text = """ℹ️ **Your Account Information**

📧 **Email:** (not set)
✅ **Verified:** ❌ No
👤 **Approved:** ❌ No
🔧 **Admin:** ❌ No
🌐 **Language:** (not set)

Use /register your@email.com to get started."""

        await message.reply(info_text, parse_mode="Markdown")

    except Exception:
        _logger.exception("Error in info command")
        await message.reply("❌ An error occurred while retrieving your information. Please try again.")

async def process_register_command_immediate(user_id: str, args, message):
    """Register command handler with immediate response."""
    try:
        # Get service instances
        telegram_svc, _ = get_service_instances()

        if not telegram_svc:
            await message.reply("❌ Service temporarily unavailable. Please try again later.")
            return

        # Parse email address
        if len(args) < 2:
            await message.reply("❌ Please provide your email address.\n\nUsage: `/register your@email.com`", parse_mode="Markdown")
            return

        email = args[1].strip()
        language = args[2].strip().lower() if len(args) > 2 else "en"

        # Basic email validation
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            await message.reply("❌ Please provide a valid email address.")
            return

        # Check rate limiting
        try:
            codes_sent = telegram_svc.count_codes_last_hour(user_id)
            if codes_sent >= 5:
                await message.reply("❌ **Rate Limit Exceeded**\n\nToo many verification codes sent. Please wait an hour before requesting another.", parse_mode="Markdown")
                return
        except Exception as e:
            _logger.warning("Rate limit check failed: %s", e)
            # Continue with registration but log the issue

        # Generate verification code
        import random
        verification_code = f"{random.randint(100000, 999999):06d}"
        sent_time = int(time.time())

        # Check if user already exists
        existing_user = telegram_svc.get_user_status(user_id)
        if existing_user:
            # Check if the email is the same as the current one
            current_email = existing_user.get("email")
            if current_email and current_email.lower() == email.lower():
                # Same email - check if verified
                if existing_user.get("verified"):
                    # Already verified - no action needed
                    await message.reply(f"ℹ️ **Email Already Registered & Verified**\n\n📧 Your email **{email}** is already registered and verified\n✅ Status: Verified\n\n💡 No changes needed. Use `/info` to see your account details.", parse_mode="Markdown")
                    return
                else:
                    # Same email but not verified - send new verification code
                    try:
                        from src.data.db.services.database_service import database_service
                        with database_service.uow() as r:
                            # Direct update to bypass constraint issues
                            r.users.update_telegram_profile(
                                user_id,
                                verification_code=verification_code,
                                code_sent_time=sent_time,
                                verified=False
                            )
                        _logger.info("Successfully updated verification code for user %s: %s", user_id, verification_code)
                        await message.reply(f"📧 **Resending Verification Code**\n\nYour email **{email}** is already registered but not verified.\n📨 Sending a new verification code...", parse_mode="Markdown")
                    except Exception as e:
                        _logger.error("Failed to update verification code directly: %s", e)
                        # Fallback to original method
                        telegram_svc.set_user_email(user_id, email, verification_code, sent_time, language)
                        await message.reply(f"📧 **Resending Verification Code**\n\nYour email **{email}** is already registered but not verified.\n📨 Sending a new verification code...", parse_mode="Markdown")
            else:
                # Different email - always require verification for new email address
                # Try direct database update as workaround for constraint issues
                try:
                    from src.data.db.services.database_service import database_service
                    with database_service.uow() as r:
                        # Direct update to bypass constraint issues
                        r.users.update_telegram_profile(
                            user_id,
                            email=email,
                            verification_code=verification_code,
                            code_sent_time=sent_time,
                            verified=False,
                            language=language
                        )
                    _logger.info("Successfully updated user %s with new email %s and verification code %s", user_id, email, verification_code)
                    await message.reply(f"✅ **Email Updated!**\n\n📧 Your email has been updated to **{email}**\n📨 A new verification code has been sent\n\nUse `/verify CODE` to verify your new email address.", parse_mode="Markdown")
                except Exception as e:
                    _logger.error("Failed to update user email directly: %s", e)
                    # Fallback to original method
                    telegram_svc.set_user_email(user_id, email, verification_code, sent_time, language)
                    await message.reply(f"✅ **Email Updated!**\n\n📧 Your email has been updated to **{email}**\n📨 A new verification code has been sent\n\nUse `/verify CODE` to verify your new email address.", parse_mode="Markdown")
        else:
            # Register new user
            telegram_svc.set_user_email(user_id, email, verification_code, sent_time, language)
            await message.reply(f"✅ **Registration Successful!**\n\n📧 A verification code has been sent to **{email}**\n\nUse `/verify CODE` to verify your email.", parse_mode="Markdown")

        # Ensure verification code is set correctly (as backup)
        telegram_svc.set_verification_code(user_id, code=verification_code, sent_time=sent_time)

        # Send verification email via notification service (non-blocking)
        try:
            client = await get_notification_client()
            if client:
                await client.send_notification(
                    notification_type="INFO",
                    title="Your Alkotrader Email Verification Code",
                    message=f"Hello,\n\nThank you for registering your email with the Alkotrader Telegram bot.\n\nYour verification code is: {verification_code}\n\nThis code is valid for 1 hour. If you did not request this, please ignore this email.\n\nBest regards,\nAlkotrader Team",
                    priority="NORMAL",
                    channels=["email"],
                    recipient_id=email
                )
        except Exception as e:
            _logger.warning("Failed to send verification email: %s", e)
            # Don't fail the registration if email sending fails

    except Exception:
        _logger.exception("Error in register command")
        await message.reply("❌ An error occurred during registration. Please try again.")

async def process_verify_command_immediate(user_id: str, args, message):
    """Verify command handler with immediate response."""
    try:
        # Get service instances
        telegram_svc, _ = get_service_instances()

        if not telegram_svc:
            await message.reply("❌ Service temporarily unavailable. Please try again later.")
            return

        # Parse verification code
        if len(args) < 2:
            await message.reply("❌ Please provide the verification code.\n\nUsage: `/verify CODE`", parse_mode="Markdown")
            return

        code = args[1]

        # Validate code format
        if not code.isdigit() or len(code) != 6:
            await message.reply("❌ Verification code must be a 6-digit number.")
            return

        # Get user status
        user_status = telegram_svc.get_user_status(user_id)
        if not user_status:
            await message.reply("❌ User not found. Please register first using `/register your@email.com`", parse_mode="Markdown")
            return

        # Verify the code using the service method
        success = telegram_svc.verify_code(user_id, code, expiry_seconds=3600)

        if success:
            await message.reply("✅ **Email Verified Successfully!**\n\nYour email has been verified. You can now use all bot features including email reports.", parse_mode="Markdown")
        else:
            await message.reply("❌ **Invalid or Expired Code**\n\nPlease check the code or request a new one with `/register`", parse_mode="Markdown")

    except Exception:
        _logger.exception("Error in verify command")
        await message.reply("❌ An error occurred while processing verification. Please try again.")

async def process_language_command_immediate(user_id: str, args, message):
    """Language command handler with immediate response."""
    try:
        # Get service instances
        telegram_svc, _ = get_service_instances()

        if not telegram_svc:
            await message.reply("❌ Service temporarily unavailable. Please try again later.")
            return

        # Parse language code
        if len(args) < 2:
            await message.reply("❌ Please provide a language code.\n\nUsage: `/language en` (supported: en, ru)", parse_mode="Markdown")
            return

        language = args[1].strip().lower()

        # Validate language
        supported_languages = ["en", "ru"]
        if language not in supported_languages:
            await message.reply(f"❌ Language '{language}' not supported.\n\nSupported languages: {', '.join(supported_languages)}")
            return

        # Check if user is approved
        user_status = telegram_svc.get_user_status(user_id)
        if not user_status:
            await message.reply("❌ Please register first using `/register your@email.com`", parse_mode="Markdown")
            return

        if not user_status.get("approved", False):
            await message.reply("❌ **Access Restricted**\n\nThis feature requires admin approval. Please use `/request_approval` first.", parse_mode="Markdown")
            return

        # Update language preference
        success = telegram_svc.update_user_language(user_id, language)
        if success:
            await message.reply(f"✅ **Language Updated**\n\nYour language preference has been updated to **{language.upper()}**.", parse_mode="Markdown")
        else:
            await message.reply("❌ Unable to update language preference. Please try again later.")

    except Exception:
        _logger.exception("Error in language command")
        await message.reply("❌ An error occurred while updating language. Please try again.")

async def process_admin_command_immediate(user_id: str, args, message):
    """Admin command handler using proper business logic."""
    from src.telegram.screener.notifications import process_admin_command
    client = await get_notification_client()
    await process_admin_command(message, user_id, args, client)

async def process_alerts_command_immediate(user_id: str, args, message):
    """Alert management command handler."""
    try:
        from src.data.db.services.alerts_service import AlertsService
        from src.data.data_manager import DataManager
        from src.data.db.services.jobs_service import JobsService

        # Get service instances
        telegram_svc, indicator_svc = get_service_instances()
        if not telegram_svc:
            await message.reply("❌ Service temporarily unavailable. Please try again later.")
            return

        # Initialize services for alert operations
        data_manager = DataManager()
        jobs_service = JobsService()
        alerts_service = AlertsService(jobs_service, data_manager, indicator_svc)

        # Parse command arguments
        if len(args) < 2:
            # List user's alerts
            alerts = alerts_service.get_user_alerts(int(user_id), active_only=True)
            if not alerts:
                await message.reply("📋 You have no active alerts.")
                return

            response = "🔔 **Your Active Alerts:**\n\n"
            for alert in alerts[:10]:  # Limit to 10 alerts
                status = "✅" if alert["enabled"] else "⏸️"
                response += f"{status} **{alert['ticker']}** ({alert['timeframe']})\n"
                response += f"   ID: {alert['id']} | Created: {alert['created_at'].strftime('%Y-%m-%d')}\n\n"

            if len(alerts) > 10:
                response += f"... and {len(alerts) - 10} more alerts\n\n"

            response += "Use `/alerts add TICKER PRICE above/below` to add alerts\n"
            response += "Use `/alerts delete ID` to remove alerts"

            await message.reply(response, parse_mode="Markdown")
            return

        action = args[1].lower()

        if action == "add" and len(args) >= 5:
            # Add new alert: /alerts add TICKER PRICE above/below
            ticker = args[2].upper()
            try:
                price = float(args[3])
                condition = args[4].lower()

                if condition not in ["above", "below"]:
                    await message.reply("❌ Condition must be 'above' or 'below'")
                    return

                # Create alert configuration
                alert_config = {
                    "ticker": ticker,
                    "timeframe": "15m",  # Default timeframe
                    "rule": {
                        "type": "price",
                        "condition": condition,
                        "value": price
                    },
                    "notify": {
                        "telegram": True,
                        "email": "-email" in message.text
                    }
                }

                result = await alerts_service.create_alert(int(user_id), alert_config)

                if result["success"]:
                    email_text = " and email" if "-email" in message.text else ""
                    await message.reply(f"✅ Alert created for **{ticker}** {condition} ${price:,.2f}\n"
                                      f"You'll be notified via Telegram{email_text} when triggered.",
                                      parse_mode="Markdown")
                else:
                    await message.reply(f"❌ Failed to create alert: {result.get('error', 'Unknown error')}")

            except ValueError:
                await message.reply("❌ Invalid price. Please use a number.")

        elif action == "delete" and len(args) >= 3:
            # Delete alert: /alerts delete ID
            try:
                alert_id = int(args[2])
                success = alerts_service.delete_alert(alert_id)

                if success:
                    await message.reply(f"✅ Alert {alert_id} deleted successfully.")
                else:
                    await message.reply(f"❌ Failed to delete alert {alert_id}. Check the ID and try again.")

            except ValueError:
                await message.reply("❌ Invalid alert ID. Please use a number.")

        elif action == "evaluate":
            # Evaluate user's alerts: /alerts evaluate
            await message.reply("🔄 Evaluating your alerts...")

            results = await alerts_service.evaluate_user_alerts(int(user_id))

            response = "📊 **Alert Evaluation Results:**\n\n"
            response += f"✅ Evaluated: {results['total_evaluated']}\n"
            response += f"🔥 Triggered: {results['triggered']}\n"
            response += f"🔄 Rearmed: {results['rearmed']}\n"
            response += f"❌ Errors: {results['errors']}\n"

            if results['triggered'] > 0:
                response += "\n🚨 **Triggered Alerts:**\n"
                for result in results['results']:
                    if result['triggered']:
                        response += f"• {result['ticker']}\n"

            await message.reply(response, parse_mode="Markdown")

        else:
            # Show help
            help_text = ("🔔 **Alert Commands:**\n\n"
                        "`/alerts` - List your alerts\n"
                        "`/alerts add TICKER PRICE above/below` - Add price alert\n"
                        "`/alerts delete ID` - Delete alert\n"
                        "`/alerts evaluate` - Check your alerts now\n\n"
                        "**Examples:**\n"
                        "`/alerts add BTCUSDT 65000 above`\n"
                        "`/alerts add AAPL 150 below -email`\n"
                        "`/alerts delete 123`")
            await message.reply(help_text, parse_mode="Markdown")

    except Exception:
        _logger.exception("Error processing alerts command for user %s", user_id)
        await message.reply("❌ An error occurred while processing your alert command. Please try again.")

async def process_schedules_command_immediate(user_id: str, args, message):
    """Schedules command handler using proper business logic."""
    from src.telegram.screener.notifications import process_schedules_command
    client = await get_notification_client()
    await process_schedules_command(message, user_id, args, client)

async def process_feedback_command_immediate(user_id: str, args, message):
    """Feedback command handler with immediate response."""
    try:
        if len(args) < 2:
            await message.reply("❌ Please provide your feedback message.\n\nUsage: `/feedback Your message here`", parse_mode="Markdown")
            return

        feedback_text = " ".join(args[1:])

        # Log feedback for admin review
        _logger.info("User feedback from %s: %s", user_id, feedback_text)

        # Send immediate confirmation to user
        await message.reply("✅ **Feedback Received**\n\nThank you for your feedback! It has been forwarded to the development team.", parse_mode="Markdown")

        # Store feedback in database (non-blocking)
        try:
            telegram_svc, _ = get_service_instances()
            if telegram_svc:
                telegram_svc.add_feedback(user_id, "feedback", feedback_text)
        except Exception as e:
            _logger.warning("Failed to store feedback in database: %s", e)
            # Don't fail the command if database storage fails

    except Exception:
        _logger.exception("Error in feedback command")
        await message.reply("❌ An error occurred while processing your feedback. Please try again.")

async def process_feature_command_immediate(user_id: str, args, message):
    """Feature command handler using proper business logic."""
    from src.telegram.screener.notifications import process_feature_command
    client = await get_notification_client()
    await process_feature_command(message, user_id, args, client)

async def process_request_approval_command_immediate(user_id: str, args, message):
    """Request approval command handler with immediate response."""
    try:
        # Get service instances
        telegram_svc, _ = get_service_instances()

        if not telegram_svc:
            await message.reply("❌ Service temporarily unavailable. Please try again later.")
            return

        # Check user status
        status = telegram_svc.get_user_status(user_id)

        if not status:
            await message.reply("❌ Please register first using `/register your@email.com`", parse_mode="Markdown")
            return

        if not status.get("verified", False):
            await message.reply("❌ Please verify your email first using `/verify CODE`", parse_mode="Markdown")
            return

        if status.get("approved", False):
            await message.reply("✅ You are already approved for restricted features!")
            return

        # Send immediate response to user
        await message.reply("✅ **Approval Request Submitted**\n\nYour request has been submitted to the administrators. You will be notified when your request is reviewed.", parse_mode="Markdown")

        # Notify admins via notification service (non-blocking)
        try:
            client = await get_notification_client()
            if client:
                await client.send_notification(
                    notification_type="INFO",
                    title="New Approval Request",
                    message=f"User {user_id} ({status.get('email')}) has requested approval for restricted features.",
                    priority="HIGH",
                    channels=["telegram"],
                    recipient_id="admin"  # This would need to be configured for admin notifications
                )
        except Exception as e:
            _logger.warning("Failed to notify admins about approval request: %s", e)
            # Don't fail the request if admin notification fails

    except Exception:
        _logger.exception("Error in request approval command")
        await message.reply("❌ An error occurred while processing your approval request. Please try again.")

async def process_unknown_command_immediate(user_id: str, message, help_text):
    """Simple unknown command handler."""
    await message.reply(f"❓ Unknown command. {help_text}")

# Global variables
notification_client = None

async def get_notification_client():
    """Lazily initialize notification client when needed."""
    global notification_client

    if notification_client is None:
        try:
            import os
            # For database-centric architecture, use a special URL to indicate database-only mode
            notification_service_url = os.getenv("NOTIFICATION_SERVICE_URL", "database://localhost")
            notification_client = NotificationServiceClient(
                service_url=notification_service_url,
                timeout=30,
                max_retries=3
            )
            _logger.info("Notification service client initialized for heavy commands")
        except Exception as e:
            _logger.warning("Could not initialize notification service client: %s", e)
            notification_client = None

    return notification_client

# ARCHITECTURE NOTE:
# Interactive commands (info, register, verify, admin, alerts, schedules, feedback, feature, etc.)
# are now processed immediately without using the notification service queue.
# Only heavy processing commands (report, screener) and background notifications
# (email verification, admin broadcasts, scheduled reports) use the notification service.

# HTTP API support
from aiohttp import web

# Service initialization and health check functions

async def send_email_notification_if_requested(message: Message, response_text: str, command: str):
    """
    Send email notification if -email flag is present in the command.
    The NotificationServiceClient handles fallback logic automatically.

    Args:
        message: Telegram message object
        response_text: The response text to send via email
        command: The command name for logging
    """
    try:
        # Parse the command to check for -email flag
        from src.telegram.command_parser import parse_command
        parsed = parse_command(message.text)

        if parsed.args.get("email", False):
            # Get user information for email
            telegram_svc, _ = get_service_instances()
            if telegram_svc:
                user_status = telegram_svc.get_user_status(str(message.from_user.id))
                if user_status and user_status.get('email'):
                    user_email = user_status['email']

                    # Send email notification via notification service
                    # The client automatically handles HTTP API + database fallback
                    client = await get_notification_client()
                    if client:
                        success = await client.send_notification(
                            notification_type="telegram_command_response",
                            title=f"Telegram Bot - {command.upper()} Command Response",
                            message=response_text,
                            priority="normal",
                            channels=["email"],
                            email_receiver=user_email,
                            recipient_id=str(message.from_user.id),
                            data={
                                "command": command,
                                "telegram_user_id": str(message.from_user.id),
                                "source": "telegram_bot"
                            }
                        )
                        if success:
                            _logger.info("Email notification sent for %s command to user %s", command, message.from_user.id)
                        else:
                            _logger.warning("Email notification failed for %s command to user %s", command, message.from_user.id)
                    else:
                        _logger.warning("Notification client not available for email notification")
                else:
                    # Send Telegram message about missing email
                    await message.answer("📧 Email notification requested but no verified email found. Use /register to set up email notifications.")
            else:
                _logger.warning("Telegram service not available for email notification")

    except Exception as e:
        _logger.error("Error sending email notification for %s command: %s", command, e)
        # Don't fail the main command if email fails
        # Don't fail the main command if email fails


async def send_email_notification_with_attachments(message: Message, response_text: str, command: str, attachments: dict = None):
    """
    Send email notification with support for attachments (pictures, documents).

    Args:
        message: Telegram message object
        response_text: The response text to send via email
        command: The command name for logging
        attachments: Dictionary of filename -> file_data (bytes or file path)
    """
    try:
        # Parse the command to check for -email flag
        from src.telegram.command_parser import parse_command
        parsed = parse_command(message.text)

        if parsed.args.get("email", False):
            # Get user information for email
            telegram_svc, _ = get_service_instances()
            if telegram_svc:
                user_status = telegram_svc.get_user_status(str(message.from_user.id))
                if user_status and user_status.get('email'):
                    user_email = user_status['email']

                    # Send email notification with attachments via notification service
                    # The client automatically handles HTTP API + database fallback
                    client = await get_notification_client()
                    if client:
                        success = await client.send_notification(
                            notification_type="telegram_command_response",
                            title=f"Telegram Bot - {command.upper()} Command Response",
                            message=response_text,
                            priority="normal",
                            channels=["email"],
                            email_receiver=user_email,
                            recipient_id=str(message.from_user.id),
                            attachments=attachments,
                            data={
                                "command": command,
                                "telegram_user_id": str(message.from_user.id),
                                "source": "telegram_bot",
                                "has_attachments": bool(attachments)
                            }
                        )
                        if success:
                            _logger.info("Email notification with attachments sent for %s command to user %s", command, message.from_user.id)
                        else:
                            _logger.warning("Email notification with attachments failed for %s command to user %s", command, message.from_user.id)
                    else:
                        _logger.warning("Notification client not available for email notification")
                else:
                    # Send Telegram message about missing email
                    await message.answer("📧 Email notification requested but no verified email found. Use /register to set up email notifications.")
            else:
                _logger.warning("Telegram service not available for email notification")

    except Exception as e:
        _logger.error("Error sending email notification with attachments for %s command: %s", command, e)



async def extract_attachments_from_telegram_message(message: Message) -> dict:
    """
    Extract attachments (photos, documents) from a Telegram message.

    Args:
        message: Telegram message object

    Returns:
        Dictionary of filename -> file_data (bytes)
    """
    attachments = {}

    try:
        # Handle photos
        if message.photo:
            # Get the largest photo size
            photo = message.photo[-1]
            file_info = await bot.get_file(photo.file_id)
            file_data = await bot.download_file(file_info.file_path)

            # Generate filename
            filename = f"photo_{photo.file_id}.jpg"
            attachments[filename] = file_data.read()

        # Handle documents
        if message.document:
            file_info = await bot.get_file(message.document.file_id)
            file_data = await bot.download_file(file_info.file_path)

            # Use original filename or generate one
            filename = message.document.file_name or f"document_{message.document.file_id}"
            attachments[filename] = file_data.read()

        # Handle stickers (as images)
        if message.sticker:
            file_info = await bot.get_file(message.sticker.file_id)
            file_data = await bot.download_file(file_info.file_path)

            filename = f"sticker_{message.sticker.file_id}.webp"
            attachments[filename] = file_data.read()

    except Exception:
        _logger.exception("Error extracting attachments from Telegram message:")

    return attachments
async def initialize_services() -> bool:
    """
    Initialize telegram_service and indicator_service instances.

    Returns:
        bool: True if all services initialized successfully, False otherwise
    """
    global telegram_service_instance, indicator_service_instance

    try:
        _logger.info("Initializing service layer...")

        # Initialize telegram service instance from module
        try:
            # Import the actual service instance (not the module)
            from src.data.db.services.telegram_service import telegram_service as telegram_service_instance

            # Validate telegram service has required methods
            required_methods = ['get_user_status', 'set_user_limit']
            for method in required_methods:
                if not hasattr(telegram_service_instance, method):
                    raise RuntimeError(f"Telegram service missing required method: {method}")

            _logger.info("Telegram service initialized and validated successfully")
        except Exception:
            _logger.exception("Failed to initialize telegram service:")
            return False

        # Initialize indicator service with default configuration and enhanced error handling
        try:
            indicator_service_instance = IndicatorService()

            # Validate indicator service initialization
            if not hasattr(indicator_service_instance, 'compute_for_ticker'):
                raise RuntimeError("IndicatorService missing required method: compute_for_ticker")

            # Test that adapters are available
            if hasattr(indicator_service_instance, 'adapters') and not indicator_service_instance.adapters:
                _logger.warning("IndicatorService has no adapters available - some functionality may be limited")

            _logger.info("Indicator service initialized and validated successfully")
        except Exception:
            _logger.exception("Failed to initialize indicator service:")
            # For now, continue without indicator service as some commands don't require it
            indicator_service_instance = None
            _logger.warning("Continuing without IndicatorService - indicator-based commands will be limited")

        # Set service instances in business logic layer with enhanced error handling
        try:
            business_logic.set_service_instances(telegram_service_instance, indicator_service_instance)
            _logger.info("Service instances set in business logic layer successfully")
        except Exception:
            _logger.exception("Failed to set service instances in business logic layer:")
            return False

        # Perform health checks with enhanced error reporting
        try:
            if await perform_service_health_checks():
                _logger.info("All services initialized and health checks passed")
                return True
            else:
                _logger.error("Service health checks failed - some functionality may be limited")
                # Return True anyway if telegram service is working, as basic functionality can still work
                if telegram_service_instance:
                    _logger.info("Continuing with limited functionality - telegram service is available")
                    return True
                else:
                    _logger.error("Critical services failed - cannot start bot")
                    return False
        except Exception as health_error:
            _logger.error("Error during health checks: %s", health_error)
            # If health checks fail but services are initialized, continue with limited functionality
            if telegram_service_instance:
                _logger.warning("Health checks failed but telegram service available - continuing with limited functionality")
                return True
            else:
                return False

    except Exception as e:
        _logger.exception("Unexpected error during service initialization: %s", e)
        return False

async def perform_service_health_checks() -> bool:
    """
    Perform health checks on all initialized services.

    Returns:
        bool: True if all health checks pass, False otherwise
    """
    try:
        _logger.info("Performing service health checks...")

        # Health check for telegram service
        if not await check_telegram_service_health():
            _logger.error("Telegram service health check failed")
            return False

        # Health check for indicator service
        if not await check_indicator_service_health():
            _logger.error("Indicator service health check failed")
            return False

        _logger.info("All service health checks passed")
        return True

    except Exception as e:
        _logger.exception("Error during service health checks: %s", e)
        return False

async def check_telegram_service_health() -> bool:
    """
    Check telegram service health by testing basic operations.

    Returns:
        bool: True if service is healthy, False otherwise
    """
    try:
        # Test basic service functionality
        # Try to get a setting (this tests database connectivity)
        test_setting = telegram_service_instance.get_setting("health_check_test")
        _logger.debug("Telegram service health check: setting retrieval successful")

        # Test user operations (this tests core functionality)
        # This should not fail even if user doesn't exist
        test_user_status = telegram_service_instance.get_user_status("health_check_test_user")
        _logger.debug("Telegram service health check: user status check successful")

        return True

    except Exception:
        _logger.exception("Telegram service health check failed:")
        return False

async def check_indicator_service_health() -> bool:
    """
    Check indicator service health by testing basic operations.

    Returns:
        bool: True if service is healthy, False otherwise
    """
    try:
        # Test that the service can be instantiated and has required adapters
        if not hasattr(indicator_service_instance, 'adapters'):
            _logger.error("Indicator service missing adapters attribute")
            return False

        # Check that required adapters are available
        required_adapters = ["ta-lib", "pandas-ta", "fundamentals"]
        for adapter_name in required_adapters:
            if adapter_name not in indicator_service_instance.adapters:
                _logger.error("Indicator service missing required adapter: %s", adapter_name)
                return False

        _logger.debug("Indicator service health check: all required adapters available")

        # Test basic functionality - check if service can handle indicator metadata
        from src.indicators.registry import INDICATOR_META
        if not INDICATOR_META:
            _logger.error("Indicator service health check: no indicator metadata available")
            return False

        _logger.debug("Indicator service health check: indicator metadata available")
        return True

    except Exception:
        _logger.exception("Indicator service health check failed:")
        return False

def get_service_instances() -> tuple:
    """
    Get the initialized service instances from business logic.

    Returns:
        tuple: (telegram_service_instance, indicator_service_instance)
    """
    try:
        return business_logic.get_service_instances()
    except Exception as e:
        _logger.debug("Service instances not available: %s", e)
        return None, None


# Initialize bot and dispatcher
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

# Global service instances
telegram_service_instance = None
indicator_service_instance = None

# HTTP API routes
async def api_send_message(request: web.Request) -> web.Response:
    """API endpoint to send message to specific user"""
    try:
        data = await request.json()
        user_id = data.get('user_id')
        message = data.get('message')
        title = data.get('title', 'Alkotrader Notification')

        if not user_id or not message:
            return web.json_response({
                'success': False,
                'error': 'Missing user_id or message'
            }, status=400)

        # Use notification service client to send message
        client = await get_notification_client()
        if not client:
            return web.json_response({'success': False, 'error': 'Notification service unavailable'}, status=503)
        success = await client.send_notification(
            notification_type=MessageType.INFO,
            title=title,
            message=message,
            priority=MessagePriority.NORMAL,
            channels=["telegram"],
            recipient_id=str(user_id)
        )

        return web.json_response({
            'success': success,
            'message': 'Message queued for delivery' if success else 'Failed to queue message'
        })

    except Exception as e:
        _logger.exception("Error in api_send_message: ")
        return web.json_response({
            'success': False,
            'error': str(e)
        }, status=500)

async def api_broadcast(request: web.Request) -> web.Response:
    """API endpoint to broadcast message to all users"""
    try:
        data = await request.json()
        message = data.get('message')
        title = data.get('title', 'Alkotrader Announcement')

        if not message:
            return web.json_response({
                'success': False,
                'error': 'Missing message'
            }, status=400)

        # Get all registered users using service layer
        telegram_svc, _ = get_service_instances()
        if not telegram_svc:
            return web.json_response({
                'success': False,
                'error': 'Service not available'
            }, status=503)

        users = telegram_svc.list_users()
        if not users:
            return web.json_response({
                'success': False,
                'error': 'No registered users found'
            }, status=404)

        success_count = 0
        total_count = len(users)

        # Queue messages for all users
        client = await get_notification_client()
        if not client:
            return web.json_response({'success': False, 'error': 'Notification service unavailable'}, status=503)

        for user in users:
            user_id = user["telegram_user_id"]
            if user_id and user_id.isdigit():
                success = await client.send_notification(
                    notification_type=MessageType.INFO,
                    title=title,
                    message=message,
                    priority=MessagePriority.NORMAL,
                    channels=["telegram"],
                    recipient_id=str(user_id)
                )
                if success:
                    success_count += 1

        return web.json_response({
            'success': True,
            'message': f'Broadcast queued for {success_count}/{total_count} users',
            'success_count': success_count,
            'total_count': total_count
        })

    except Exception as e:
        _logger.exception("Error in api_broadcast: ")
        return web.json_response({
            'success': False,
            'error': str(e)
        }, status=500)

async def api_notify(request: web.Request) -> web.Response:
    """API endpoint for sending notifications from scheduler service"""
    try:
        data = await request.json()

        # Extract required fields
        notification_type = data.get('notification_type', 'INFO')
        title = data.get('title', 'Alert Notification')
        message = data.get('message')
        priority = data.get('priority', 'NORMAL')
        telegram_chat_id = data.get('telegram_chat_id')

        if not message:
            return web.json_response({
                'success': False,
                'error': 'Missing message field'
            }, status=400)

        if not telegram_chat_id:
            return web.json_response({
                'success': False,
                'error': 'Missing telegram_chat_id field'
            }, status=400)

        # Use notification service client to send notification
        client = await get_notification_client()
        success = await client.send_notification(
            notification_type=notification_type,
            title=title,
            message=message,
            priority=priority,
            channels=["telegram"],
            telegram_chat_id=int(telegram_chat_id),
            recipient_id=str(telegram_chat_id),
            data=data.get('data', {})
        )

        return web.json_response({
            'success': success,
            'message': 'Notification queued for delivery' if success else 'Failed to queue notification'
        })

    except Exception as e:
        _logger.exception("Error in api_notify: ")
        return web.json_response({
            'success': False,
            'error': str(e)
        }, status=500)

async def api_status(request: web.Request) -> web.Response:
    """API endpoint for health check and status"""
    try:
        # Get notification client stats
        stats = notification_client.get_stats() if notification_client else {}

        # Get user count using service layer
        telegram_svc, _ = get_service_instances()
        if telegram_svc:
            users = telegram_svc.list_users()
            user_count = len(users)
        else:
            user_count = 0

        # Check service health
        service_health = await perform_service_health_checks()

        # Get service instances status
        telegram_svc, indicator_svc = get_service_instances()

        service_status = {
            'telegram_service': {
                'initialized': telegram_svc is not None,
                'healthy': service_health
            },
            'indicator_service': {
                'initialized': indicator_svc is not None,
                'healthy': service_health,
                'adapters': list(indicator_svc.adapters.keys()) if indicator_svc else []
            }
        }

        overall_status = 'healthy' if service_health else 'degraded'

        return web.json_response({
            'success': True,
            'status': overall_status,
            'services': service_status,
            'notification_stats': stats,
            'user_count': user_count,
            'queue_size': 0  # Queue is managed by notification service
        })

    except Exception as e:
        _logger.exception("Error in api_status: ")
        return web.json_response({
            'success': False,
            'status': 'error',
            'error': str(e)
        }, status=500)

# Create HTTP app
api_app = web.Application()
api_app.router.add_post('/api/send_message', api_send_message)
api_app.router.add_post('/api/broadcast', api_broadcast)
api_app.router.add_post('/api/notify', api_notify)
api_app.router.add_get('/api/status', api_status)
api_app.router.add_get('/api/test', lambda r: web.json_response({'status': 'ok', 'message': 'Bot API is working!'}))

HELP_TEXT = (
    "Welcome to the Telegram Screener Bot!\n\n"
    "📧 Email Notifications: Add -email to any command to receive the response via email as well as Telegram.\n"
    "Example: /help -email, /info -email, /alerts -email\n\n"
    "Basic Commands:\n"
    "/start - Show welcome message\n"
    "/help - Show this help message\n"
    "/info - Show your registered email and verification status\n"
    "/register email@example.com - Register or update your email for reports\n"
    "/verify CODE - Verify your email with the code sent\n"
    "/request_approval - Request admin approval after email verification (required for restricted features)\n"
    "/language LANG - Update language preference (en, ru)\n\n"

    "Report Commands:\n"
    "/report TICKER1 TICKER2 ... [flags] - Get a report for specified tickers\n"
    "Flags:\n"
    "  -email: Send report to your registered email\n"
    "  -indicators=RSI,MACD,MA50,PE,EPS: Specify technical indicators\n"
    "  -period=3mo,1y,2y: Data period (default: 2y)\n"
    "  -interval=1d,15m,1h: Data interval (default: 1d)\n"
    "  -provider=yf,bnc: Data provider (yf=Yahoo, bnc=Binance)\n"
    "  -config=JSON_STRING: Use JSON configuration for advanced options\n\n"
    "JSON Configuration Examples:\n"
    "  /report -config='{\"report_type\":\"analysis\",\"tickers\":[\"AAPL\",\"MSFT\"],\"period\":\"1y\",\"indicators\":[\"RSI\",\"MACD\"],\"email\":true}'\n"
    "  /report -config='{\"report_type\":\"analysis\",\"tickers\":[\"TSLA\"],\"period\":\"6mo\",\"interval\":\"1h\",\"indicators\":[\"RSI\",\"MACD\",\"BollingerBands\"],\"include_fundamentals\":false}'\n\n"

    "Screener Commands:\n"
    "/screener JSON_CONFIG [-email] - Run enhanced screener immediately\n"
    "  JSON_CONFIG: Screener configuration in JSON format\n"
    "  Example: /screener '{\"screener_type\":\"hybrid\",\"list_type\":\"us_medium_cap\",\"fmp_criteria\":{\"marketCapMoreThan\":200000000,\"peRatioLessThan\":20},\"fundamental_criteria\":[{\"indicator\":\"Revenue_Growth\",\"operator\":\"min\",\"value\":0.05}],\"max_results\":5,\"min_score\":2.0}'\n"
    "  Example: /screener '{\"screener_type\":\"fundamental\",\"list_type\":\"us_small_cap\",\"fmp_strategy\":\"conservative_value\",\"max_results\":10}' -email\n"
    "Flags:\n"
    "  -email: Send results to your registered email\n\n"

    "Alert Commands:\n"
    "/alerts - List all your active price alerts\n"
    "/alerts add TICKER PRICE above/below [flags] - Add price alert\n"
    "  Example: /alerts add BTCUSDT 65000 above -email\n"
    "/alerts delete ALERT_ID - Delete alert by ID\n"
    "/alerts evaluate - Check your alerts now\n"
    "Flags:\n"
    "  -email: Send alert notifications to email\n\n"

    "Schedule Commands:\n"
    "/schedules - List all your scheduled reports\n"
    "/schedules add TICKER TIME [flags] - Schedule daily report\n"
    "  TIME: HH:MM format (24h UTC)\n"
    "  Example: /schedules add AAPL 09:00 -email\n"
    "Flags:\n"
    "  -email: Send report to email\n"
    "  -indicators=RSI,MACD: Specify indicators\n"
    "  -period=1y: Data period\n"
    "  -interval=1d: Data interval\n"
    "  -provider=yf: Data provider\n"

    "/schedules screener LIST_TYPE TIME [flags] - Schedule fundamental screener report\n"
    "  LIST_TYPE: us_small_cap, us_medium_cap, us_large_cap, swiss_shares, or custom list name\n"
    "  TIME: HH:MM format (24h UTC)\n"
    "  Example: /schedules screener us_small_cap 08:00 -email\n"
    "  Example: /schedules screener my_custom_list 09:30 -indicators=PE,PB,ROE\n"
    "Flags:\n"
    "  -email: Send screener report to email\n"
    "  -indicators=PE,PB,ROE: Specify fundamental indicators to include\n"
    "Screener finds undervalued stocks using fundamental analysis (P/E, P/B, ROE, DCF, etc.)\n"
    "/schedules edit SCHEDULE_ID [TIME] [flags] - Edit schedule\n"
    "/schedules delete SCHEDULE_ID - Delete schedule\n"
    "/schedules pause SCHEDULE_ID - Pause schedule\n"
    "/schedules resume SCHEDULE_ID - Resume schedule\n\n"

    "Feedback Commands:\n"
    "/feedback MESSAGE - Send feedback or bug report\n"
    "/feature MESSAGE - Suggest a new feature\n\n"

    "Note: Some commands require admin approval. Use /request_approval after email verification."
)

ADMIN_HELP_TEXT = (
    "Admin Commands:\n"
    "/admin users - List all registered users\n"
    "/admin listusers - List users as telegram_user_id - email pairs\n"
    "/admin pending - List users waiting for approval\n"
    "/admin approve TELEGRAM_USER_ID - Approve user for restricted features\n"
    "/admin reject TELEGRAM_USER_ID - Reject user's approval request\n"
    "/admin verify TELEGRAM_USER_ID - Manually verify user's email\n"
    "/admin resetemail TELEGRAM_USER_ID - Reset user's email\n"
    "/admin setlimit alerts N [USER_ID] - Set max alerts (global or per-user)\n"
    "/admin setlimit schedules N [USER_ID] - Set max schedules (global or per-user)\n"
    "/admin broadcast MESSAGE - Send broadcast message to all users\n"
)

def generate_code():
    return f"{random.randint(100000, 999999):06d}"

async def audit_command_wrapper(message: Message, command_func, *args, **kwargs):
    """
    Wrapper function to audit all commands with service layer error handling.

    This wrapper implements the dependency injection pattern by:
    1. Ensuring service instances are available before processing commands
    2. Auditing all commands using the telegram_service layer
    3. Handling service layer errors gracefully
    4. Providing consistent service access to command handlers through business logic

    Command handlers access services through:
    - business_logic.handle_command() which uses global service instances
    - business_logic.get_service_instances() for direct service access

    Service instances are set during bot initialization via:
    - business_logic.set_service_instances(telegram_service, indicator_service)
    """
    start_time = time.time()
    telegram_user_id = str(message.from_user.id)
    command = message.text.split()[0] if message.text else ""
    full_message = message.text

    # Check service health before processing command with enhanced error handling
    try:
        telegram_svc, indicator_svc = get_service_instances()
        if not telegram_svc:
            _logger.error("Telegram service not initialized for command %s from user %s", command, telegram_user_id)
            try:
                await message.answer("Service temporarily unavailable. Please try again later.")
            except Exception as msg_error:
                _logger.error("Failed to send error message to user %s: %s", telegram_user_id, msg_error)
            return
    except Exception as service_error:
        _logger.error("Error getting service instances for command %s from user %s: %s",
                     command, telegram_user_id, service_error)
        try:
            await message.answer("Service temporarily unavailable. Please try again later.")
        except Exception as msg_error:
            _logger.error("Failed to send error message to user %s: %s", telegram_user_id, msg_error)
        return

    try:
        # Check if user is registered using service layer with enhanced error handling
        try:
            user_status = telegram_svc.get_user_status(telegram_user_id)
            is_registered_user = user_status is not None
            user_email = user_status.get('email') if user_status else None
        except Exception as user_status_error:
            _logger.warning("Failed to get user status for %s during command %s: %s",
                          telegram_user_id, command, user_status_error)
            # Continue with unknown user status
            user_status = None
            is_registered_user = False
            user_email = None

        # Execute the command - command handlers will use service instances through business logic
        result = await command_func(*args, **kwargs)

        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)

        # Log successful command using service layer with enhanced context
        try:
            _logger.info("Command executed successfully: user=%s, command=%s, response_time=%dms, registered=%s",
                        telegram_user_id, command, response_time_ms, is_registered_user)

            telegram_svc.log_command_audit(
                telegram_user_id=telegram_user_id,
                command=command,
                full_message=full_message,
                is_registered_user=is_registered_user,
                user_email=user_email,
                success=True,
                response_time_ms=response_time_ms
            )

            _logger.debug("Command audit logged successfully for user %s, command %s", telegram_user_id, command)

        except Exception as audit_error:
            _logger.warning("Failed to log successful command audit for user %s, command %s: %s",
                          telegram_user_id, command, audit_error)

        return result

    except Exception as e:
        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)

        # Try to get user status for error logging with enhanced error handling
        try:
            user_status = telegram_svc.get_user_status(telegram_user_id)
            is_registered_user = user_status is not None
            user_email = user_status.get('email') if user_status else None
        except Exception as user_error:
            _logger.warning("Failed to get user status for error logging (user %s, command %s): %s",
                          telegram_user_id, command, user_error)
            is_registered_user = False
            user_email = None

        # Log failed command using service layer with enhanced context
        try:
            _logger.error("Command failed: user=%s, command=%s, response_time=%dms, registered=%s, error=%s",
                         telegram_user_id, command, response_time_ms, is_registered_user, str(e))

            telegram_svc.log_command_audit(
                telegram_user_id=telegram_user_id,
                command=command,
                full_message=full_message,
                is_registered_user=is_registered_user,
                user_email=user_email,
                success=False,
                error_message=str(e),
                response_time_ms=response_time_ms
            )

            _logger.debug("Command failure audit logged successfully for user %s, command %s", telegram_user_id, command)

        except Exception as audit_error:
            _logger.warning("Failed to log failed command audit for user %s, command %s: %s",
                          telegram_user_id, command, audit_error)

        # Re-raise the exception
        raise

@dp.message(Command("start"))
async def cmd_start(message: Message):
    _logger.info("Received /start command from user %s", message.from_user.id)
    try:
        # Handle /start command immediately without notification service
        welcome_text = f"Welcome to the Alkotrader Bot! 🤖\n\n{HELP_TEXT}"
        await message.answer(welcome_text)

        # Send email notification if -email flag is present (optional, won't block if service unavailable)
        try:
            await send_email_notification_if_requested(message, welcome_text, "start")
        except Exception as email_error:
            _logger.debug("Email notification not available for /start command: %s", email_error)

        # Audit the command for logging purposes (optional, won't block if service unavailable)
        try:
            async def start_command_func(*args, **kwargs):
                return {"status": "ok", "message": "Start command processed"}
            await audit_command_wrapper(message, start_command_func, str(message.from_user.id))
        except Exception as audit_error:
            _logger.debug("Audit not available for /start command: %s", audit_error)

        _logger.info("Successfully processed /start command for user %s", message.from_user.id)
    except Exception:
        _logger.exception("Error processing /start command for user %s", message.from_user.id)
        # Send a simple error message directly
        await message.answer("Sorry, there was an error processing your command. Please try again.")

@dp.message(Command("help"))
async def cmd_help(message: Message):
    try:
        # Handle /help command immediately without notification service
        await message.answer(HELP_TEXT)

        # Send email notification if -email flag is present (optional, won't block if service unavailable)
        try:
            await send_email_notification_if_requested(message, HELP_TEXT, "help")
        except Exception as email_error:
            _logger.debug("Email notification not available for /help command: %s", email_error)

        # Audit the command for logging purposes (optional, won't block if service unavailable)
        try:
            async def help_command_func(*args, **kwargs):
                return {"status": "ok", "message": "Help command processed"}
            await audit_command_wrapper(message, help_command_func, str(message.from_user.id))
        except Exception as audit_error:
            _logger.debug("Audit not available for /help command: %s", audit_error)

    except Exception:
        _logger.exception("Error processing /help command for user %s", message.from_user.id)
        # Final fallback: send built-in help text directly
        try:
            await message.answer(HELP_TEXT)
        except Exception:
            await message.answer("Sorry, there was an error processing your command. Please try again.")

@dp.message(Command("info"))
async def cmd_info(message: Message):
    await audit_command_wrapper(message, process_info_command_immediate, str(message.from_user.id), message)

@dp.message(Command("register"))
async def cmd_register(message: Message):
    args = message.text.split()
    await audit_command_wrapper(message, process_register_command_immediate, str(message.from_user.id), args, message)

@dp.message(Command("verify"))
async def cmd_verify(message: Message):
    args = message.text.split()
    await audit_command_wrapper(message, process_verify_command_immediate, str(message.from_user.id), args, message)

@dp.message(Command("request_approval"))
async def cmd_request_approval(message: Message):
    args = message.text.split()
    await audit_command_wrapper(message, process_request_approval_command_immediate, str(message.from_user.id), args, message)

@dp.message(Command("language"))
async def cmd_language(message: Message):
    args = message.text.split()
    await audit_command_wrapper(message, process_language_command_immediate, str(message.from_user.id), args, message)

@dp.message(Command("admin"))
async def cmd_admin(message: Message):
    args = message.text.split()
    await audit_command_wrapper(message, process_admin_command_immediate, str(message.from_user.id), args, message)

@dp.message(Command("report"))
async def cmd_report(message: Message):
    args = message.text.split()
    # Report command still uses notification service for heavy processing and email support
    client = await get_notification_client()
    await audit_command_wrapper(message, process_report_command, message, str(message.from_user.id), args, client)

@dp.message(Command("alerts"))
async def cmd_alerts(message: Message):
    args = message.text.split()
    await audit_command_wrapper(message, process_alerts_command_immediate, str(message.from_user.id), args, message)

@dp.message(Command("schedules"))
async def cmd_schedules(message: Message):
    args = message.text.split()
    await audit_command_wrapper(message, process_schedules_command_immediate, str(message.from_user.id), args, message)

@dp.message(Command("screener"))
async def cmd_screener(message: Message):
    args = message.text.split()
    # Screener command still uses notification service for heavy processing and email support
    client = await get_notification_client()
    await audit_command_wrapper(message, process_screener_command, message, str(message.from_user.id), args, client)

@dp.message(Command("feedback"))
async def cmd_feedback(message: Message):
    args = message.text.split(maxsplit=1)
    await audit_command_wrapper(message, process_feedback_command_immediate, str(message.from_user.id), args, message)

@dp.message(Command("feature"))
async def cmd_feature(message: Message):
    args = message.text.split(maxsplit=1)
    await audit_command_wrapper(message, process_feature_command_immediate, str(message.from_user.id), args, message)

@dp.message(lambda message: message.text and message.text.startswith("/"))
async def unknown_command(message: Message):
    _logger.info("Received unknown command: %s from user %s", message.text, message.from_user.id)

    # Handle case-insensitive commands
    command_text = message.text.strip()
    command_name = command_text.split()[0].lstrip("/").lower()

    # Map common case variations to their handlers
    command_handlers = {
        "info": cmd_info,
        "help": cmd_help,
        "start": cmd_start,
        "report": cmd_report,
        "alerts": cmd_alerts,
        "schedules": cmd_schedules,
        "screener": cmd_screener,
        "admin": cmd_admin,
        "register": cmd_register,
        "verify": cmd_verify,
        "request_approval": cmd_request_approval,
        "language": cmd_language,
        "feedback": cmd_feedback,
        "feature": cmd_feature,
    }

    if command_name in command_handlers:
        _logger.info("Handling case-insensitive command: %s -> %s", command_text, command_name)
        try:
            await command_handlers[command_name](message)
            return
        except Exception:
            _logger.exception("Error processing case-insensitive command %s for user %s", command_name, message.from_user.id)
            await message.answer("Sorry, there was an error processing your command. Please try again.")
            return

    # If not a case variation, process as unknown command
    try:
        await audit_command_wrapper(message, process_unknown_command_immediate, str(message.from_user.id), message, HELP_TEXT)
    except Exception:
        _logger.exception("Error processing unknown command for user %s", message.from_user.id)
        await message.answer("Sorry, there was an error processing your command. Please try again.")

@dp.message()
async def all_messages(message: Message):
    """Catch all messages for debugging and provide help for non-command messages"""
    _logger.info("Received message: %s from user %s", message.text, message.from_user.id)

    # For non-command messages, provide help
    if message.text and not message.text.startswith("/"):
        try:
            # Handle non-command messages immediately
            help_message = f"❓ I don't understand '{message.text}'\n\nPlease use /help to see all available commands."
            await message.answer(help_message)

            # Audit the interaction for logging purposes
            async def non_command_func(*args, **kwargs):
                return {"status": "ok", "message": "Non-command message processed"}

            await audit_command_wrapper(message, non_command_func, str(message.from_user.id))

        except Exception:
            _logger.exception("Error processing non-command message for user %s", message.from_user.id)
            await message.answer("Please use /help to see available commands.")

async def main():
    global notification_client

    _logger.info("Starting bot initialization...")
    if TELEGRAM_BOT_TOKEN:
        _logger.info("Bot token: %s...", TELEGRAM_BOT_TOKEN[:10])
    else:
        _logger.error("Bot token is None!")
        return

    # Set logging context so that notification service client logs go to telegram bot log file
    set_logging_context("telegram_screener_bot")

    # Initialize service layer (optional for basic commands)
    _logger.info("Initializing service layer...")
    try:
        services_initialized = await initialize_services()
        if services_initialized:
            _logger.info("Service layer initialized successfully")
        else:
            _logger.warning("Service layer initialization failed - bot will run with limited functionality")
    except Exception as e:
        _logger.warning("Service layer initialization error: %s - bot will run with limited functionality", e)

    # Notification service client will be initialized lazily when needed
    _logger.info("Notification service client will be initialized when needed for heavy commands")
    notification_client = None

    # Initialize heartbeat manager
    _logger.info("Initializing heartbeat manager...")
    try:
        from src.common.heartbeat_manager import HeartbeatManager

        def telegram_bot_health_check():
            """Health check function for telegram bot."""
            try:
                # Check if bot is initialized properly
                bot_healthy = bot is not None and TELEGRAM_BOT_TOKEN is not None

                # For lazy initialization, we consider the notification system healthy
                # if we can potentially initialize it (no need to check if it's already initialized)
                notification_system_available = True  # We use lazy initialization

                if bot_healthy and notification_system_available:
                    return {
                        'status': 'HEALTHY',
                        'metadata': {
                            'bot_token_present': bool(TELEGRAM_BOT_TOKEN),
                            'notification_system': 'lazy_initialization',
                            'last_check': time.time()
                        }
                    }
                elif bot_healthy:
                    return {
                        'status': 'DEGRADED',
                        'error_message': 'Notification system not available',
                        'metadata': {
                            'bot_token_present': True,
                            'notification_system': 'unavailable'
                        }
                    }
                else:
                    return {
                        'status': 'DOWN',
                        'error_message': 'Bot not properly initialized',
                        'metadata': {
                            'bot_token_present': bool(TELEGRAM_BOT_TOKEN),
                            'notification_system': 'unknown'
                        }
                    }
            except Exception as e:
                return {
                    'status': 'DOWN',
                    'error_message': f'Health check failed: {str(e)}'
                }

        # Create and start heartbeat manager
        heartbeat_manager = HeartbeatManager(
            system='telegram_bot',
            interval_seconds=30
        )
        heartbeat_manager.set_health_check_function(telegram_bot_health_check)
        heartbeat_manager.start_heartbeat()

        _logger.info("Heartbeat manager started for telegram bot")

    except Exception:
        _logger.exception("Failed to initialize heartbeat manager:")

    # Initialize telegram queue processor for queued messages
    _logger.info("Initializing telegram queue processor...")
    try:
        from src.telegram.services.telegram_queue_processor import TelegramQueueProcessor

        # Create and start queue processor
        queue_processor = TelegramQueueProcessor(
            bot=bot,
            poll_interval=5  # Poll every 5 seconds
        )
        await queue_processor.start()

        _logger.info("Telegram queue processor started (polls database every 5 seconds)")

    except Exception:
        _logger.exception("Failed to initialize telegram queue processor:")

    _logger.info("Starting Telegram Screener Bot with HTTP API...")

    # Start both bot polling and HTTP API server
    try:
        bot_runner = web.AppRunner(api_app)
        await bot_runner.setup()

        # Start HTTP API server on port 8080
        api_site = web.TCPSite(bot_runner, 'localhost', 8080)
        await api_site.start()

        _logger.info("HTTP API server started on http://localhost:8080")
        _logger.info("Available endpoints:")
        _logger.info("  POST /api/send_message - Send message to specific user")
        _logger.info("  POST /api/broadcast - Broadcast message to all users")
        _logger.info("  GET  /api/status - Health check and status")

        # Start bot polling
        _logger.info("Starting bot polling...")
        await dp.start_polling(bot)
        _logger.info("Bot polling started successfully")

    except Exception as e:
        _logger.exception("Failed to start bot or HTTP API server: %s", e)
        return

if __name__ == "__main__":
    asyncio.run(main())
