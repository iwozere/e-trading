"""
handlers/account.py — User account commands: /info, /register, /verify, /language, /request_approval.

Command handlers register routes onto the supplied Dispatcher and delegate
business logic to the process_* functions defined here.
"""
import re
import random
import time
from typing import Optional

from aiogram import Dispatcher
from aiogram.types import Message
from aiogram.filters import Command

from src.notification.logger import setup_logger
from src.telegram.command_parser import parse_command
from src.model.telegram_bot import ParsedCommand
from src.telegram.lifecycle import get_service_instances, get_notification_client

_logger = setup_logger("telegram_screener_bot")


# ─── Processor functions ─────────────────────────────────────────────────────

async def process_info_command_immediate(user_id: str, message: Message) -> None:
    """Handle /info — display the user's account details."""
    try:
        telegram_svc, _ = get_service_instances()
        if not telegram_svc:
            await message.reply("❌ Service temporarily unavailable. Please try again later.")
            return

        status = telegram_svc.get_user_status(user_id)
        if status:
            email = status["email"] or "(not set)"
            verified = "✅ Yes" if status["verified"] else "❌ No"
            approved = "✅ Yes" if status["approved"] else "❌ No"
            admin = "✅ Yes" if status["is_admin"] else "❌ No"
            language = status["language"] or "(not set)"
            text = (
                f"ℹ️ **Your Account Information**\n\n"
                f"📧 **Email:** {email}\n"
                f"✅ **Verified:** {verified}\n"
                f"👤 **Approved:** {approved}\n"
                f"🔧 **Admin:** {admin}\n"
                f"🌐 **Language:** {language}\n\n"
                f"Use /help to see available commands."
            )
        else:
            text = (
                "ℹ️ **Your Account Information**\n\n"
                "📧 **Email:** (not set)\n"
                "✅ **Verified:** ❌ No\n"
                "👤 **Approved:** ❌ No\n"
                "🔧 **Admin:** ❌ No\n"
                "🌐 **Language:** (not set)\n\n"
                "Use /register your@email.com to get started."
            )
        await message.reply(text, parse_mode="Markdown")
    except Exception:
        _logger.exception("Error in /info command")
        await message.reply("❌ An error occurred while retrieving your information. Please try again.")


async def process_register_command_immediate(user_id: str, parsed: ParsedCommand, message: Message) -> None:
    """Handle /register <email> [language]."""
    try:
        telegram_svc, _ = get_service_instances()
        if not telegram_svc:
            await message.reply("❌ Service temporarily unavailable. Please try again later.")
            return

        email = parsed.args.get("email_address", "").strip()
        language = parsed.args.get("language", "en").strip().lower()

        if not email:
            await message.reply("❌ Please provide your email address.\n\nUsage: `/register your@email.com`", parse_mode="Markdown")
            return

        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            await message.reply("❌ Please provide a valid email address.")
            return

        try:
            codes_sent = telegram_svc.count_codes_last_hour(user_id)
            if codes_sent >= 5:
                await message.reply("❌ **Rate Limit Exceeded**\n\nToo many verification codes sent. Please wait an hour before requesting another.", parse_mode="Markdown")
                return
        except Exception as exc:
            _logger.warning("Rate limit check failed: %s", exc)

        verification_code = f"{random.randint(100000, 999999):06d}"
        sent_time = int(time.time())

        existing_user = telegram_svc.get_user_status(user_id)
        if existing_user:
            current_email = existing_user.get("email")
            if current_email and current_email.lower() == email.lower():
                if existing_user.get("verified"):
                    await message.reply(f"ℹ️ **Email Already Registered & Verified**\n\n📧 Your email **{email}** is already registered and verified\n✅ Status: Verified\n\n💡 No changes needed. Use `/info` to see your account details.", parse_mode="Markdown")
                    return
                else:
                    try:
                        telegram_svc.set_user_email(user_id, email, verification_code, sent_time, language)
                        _logger.info("Successfully updated verification code for user %s", user_id)
                        await message.reply(f"📧 **Resending Verification Code**\n\nYour email **{email}** is already registered but not verified.\n📨 Sending a new verification code...", parse_mode="Markdown")
                    except Exception as exc:
                        _logger.error("Failed to update verification code: %s", exc)
                        await message.reply("❌ An error occurred. Please try again later.")
            else:
                try:
                    telegram_svc.set_user_email(user_id, email, verification_code, sent_time, language)
                    _logger.info("Successfully updated user %s with new email %s", user_id, email)
                    await message.reply(f"✅ **Email Updated!**\n\n📧 Your email has been updated to **{email}**\n📨 A new verification code has been sent\n\nUse `/verify CODE` to verify your new email address.", parse_mode="Markdown")
                except Exception as exc:
                    _logger.error("Failed to update user email: %s", exc)
                    await message.reply("❌ An error occurred. Please try again later.")
        else:
            telegram_svc.set_user_email(user_id, email, verification_code, sent_time, language)
            await message.reply(f"✅ **Registration Successful!**\n\n📧 A verification code has been sent to **{email}**\n\nUse `/verify CODE` to verify your email.", parse_mode="Markdown")

        # NOTE: set_user_email already stores the verification code internally.

        try:
            client = await get_notification_client()
            if client:
                await client.send_notification(
                    notification_type="INFO",
                    title="Your Alkotrader Email Verification Code",
                    message=(
                        f"Hello,\n\nThank you for registering your email with the Alkotrader Telegram bot.\n\n"
                        f"Your verification code is: {verification_code}\n\n"
                        f"This code is valid for 1 hour. If you did not request this, please ignore this email.\n\n"
                        f"Best regards,\nAlkotrader Team"
                    ),
                    priority="NORMAL",
                    channels=["email"],
                    recipient_id=email,
                )
        except Exception as exc:
            _logger.warning("Failed to send verification email: %s", exc)

    except Exception:
        _logger.exception("Error in /register command")
        await message.reply("❌ An error occurred during registration. Please try again.")


async def process_verify_command_immediate(user_id: str, parsed: ParsedCommand, message: Message) -> None:
    """Handle /verify <code>."""
    try:
        telegram_svc, _ = get_service_instances()
        if not telegram_svc:
            await message.reply("❌ Service temporarily unavailable. Please try again later.")
            return

        code = parsed.args.get("verification_code", "")

        if not code:
            await message.reply("❌ Please provide the verification code.\n\nUsage: `/verify CODE`", parse_mode="Markdown")
            return
        if not code.isdigit() or len(code) != 6:
            await message.reply("❌ Verification code must be a 6-digit number.")
            return

        user_status = telegram_svc.get_user_status(user_id)
        if not user_status:
            await message.reply("❌ User not found. Please register first using `/register your@email.com`", parse_mode="Markdown")
            return

        success = telegram_svc.verify_code(user_id, code, expiry_seconds=3600)
        if success:
            await message.reply("✅ **Email Verified Successfully!**\n\nYour email has been verified. You can now use all bot features including email reports.", parse_mode="Markdown")
        else:
            await message.reply("❌ **Invalid or Expired Code**\n\nPlease check the code or request a new one with `/register`", parse_mode="Markdown")
    except Exception:
        _logger.exception("Error in /verify command")
        await message.reply("❌ An error occurred while processing verification. Please try again.")


async def process_language_command_immediate(user_id: str, parsed: ParsedCommand, message: Message) -> None:
    """Handle /language <code>."""
    try:
        telegram_svc, _ = get_service_instances()
        if not telegram_svc:
            await message.reply("❌ Service temporarily unavailable. Please try again later.")
            return

        language = parsed.args.get("language_code", "").strip().lower()

        if not language:
            await message.reply("❌ Please provide a language code.\n\nUsage: `/language en` (supported: en, ru)", parse_mode="Markdown")
            return
        supported_languages = ["en", "ru"]
        if language not in supported_languages:
            await message.reply(f"❌ Language '{language}' not supported.\n\nSupported languages: {', '.join(supported_languages)}")
            return

        user_status = telegram_svc.get_user_status(user_id)
        if not user_status:
            await message.reply("❌ Please register first using `/register your@email.com`", parse_mode="Markdown")
            return

        if not user_status.get("approved", False):
            await message.reply("❌ **Access Restricted**\n\nThis feature requires admin approval. Please use `/request_approval` first.", parse_mode="Markdown")
            return

        success = telegram_svc.update_user_language(user_id, language)
        if success:
            await message.reply(f"✅ **Language Updated**\n\nYour language preference has been updated to **{language.upper()}**.", parse_mode="Markdown")
        else:
            await message.reply("❌ Unable to update language preference. Please try again later.")
    except Exception:
        _logger.exception("Error in /language command")
        await message.reply("❌ An error occurred while updating language. Please try again.")


async def process_request_approval_command_immediate(user_id: str, parsed: ParsedCommand, message: Message) -> None:
    """Handle /request_approval."""
    try:
        telegram_svc, _ = get_service_instances()
        if not telegram_svc:
            await message.reply("❌ Service temporarily unavailable. Please try again later.")
            return

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

        await message.reply("✅ **Approval Request Submitted**\n\nYour request has been submitted to the administrators. You will be notified when your request is reviewed.", parse_mode="Markdown")

        try:
            client = await get_notification_client()
            if client:
                await client.send_notification(
                    notification_type="INFO",
                    title="New Approval Request",
                    message=f"User {user_id} ({status.get('email')}) has requested approval for restricted features.",
                    priority="HIGH",
                    channels=["telegram"],
                    recipient_id="admin",
                )
        except Exception as exc:
            _logger.warning("Failed to notify admins about approval request: %s", exc)
    except Exception:
        _logger.exception("Error in /request_approval command")
        await message.reply("❌ An error occurred while processing your approval request. Please try again.")


# ─── Route registration ───────────────────────────────────────────────────────

async def cmd_info(msg: Message):
    """Handle /info command."""
    from src.telegram.handlers.common import audit_command_wrapper
    await audit_command_wrapper(msg, process_info_command_immediate, str(msg.from_user.id), msg)


async def cmd_register(msg: Message):
    """Handle /register command."""
    from src.telegram.handlers.common import audit_command_wrapper
    parsed = parse_command(msg.text)
    await audit_command_wrapper(msg, process_register_command_immediate, str(msg.from_user.id), parsed, msg)


async def cmd_verify(msg: Message):
    """Handle /verify command."""
    from src.telegram.handlers.common import audit_command_wrapper
    parsed = parse_command(msg.text)
    await audit_command_wrapper(msg, process_verify_command_immediate, str(msg.from_user.id), parsed, msg)


async def cmd_language(msg: Message):
    """Handle /language command."""
    from src.telegram.handlers.common import audit_command_wrapper
    parsed = parse_command(msg.text)
    await audit_command_wrapper(msg, process_language_command_immediate, str(msg.from_user.id), parsed, msg)


async def cmd_request_approval(msg: Message):
    """Handle /request_approval command."""
    from src.telegram.handlers.common import audit_command_wrapper
    parsed = parse_command(msg.text)
    await audit_command_wrapper(msg, process_request_approval_command_immediate, str(msg.from_user.id), parsed, msg)


def register(dp: Dispatcher) -> None:
    """Register all account command handlers onto the Dispatcher."""
    dp.message(Command("info"))(cmd_info)
    dp.message(Command("register"))(cmd_register)
    dp.message(Command("verify"))(cmd_verify)
    dp.message(Command("language"))(cmd_language)
    dp.message(Command("request_approval"))(cmd_request_approval)
