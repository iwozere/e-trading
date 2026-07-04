"""
handlers/content.py — /report and /screener commands, and email/attachment helpers.

Heavy processing commands (report, screener) delegate to screener/notifications.py.
Email notification helpers are shared with other handler modules.
"""

from aiogram import Dispatcher
from aiogram.filters import Command
from aiogram.types import Message

from src.notification.logger import setup_logger
from src.telegram.command_parser import parse_command
from src.telegram.lifecycle import get_notification_client, get_service_instances

_logger = setup_logger("telegram_screener_bot")


# ─── Email / attachment helpers ───────────────────────────────────────────────


async def send_email_notification_if_requested(message: Message, response_text: str, command: str) -> None:
    """Send email copy of a command response if -email flag is present."""
    try:
        parsed = parse_command(message.text)
        if not parsed.args.get("email", False):
            return

        telegram_svc, _ = get_service_instances()
        if not telegram_svc:
            _logger.warning("TelegramService not available for email notification")
            return

        user_status = telegram_svc.get_user_status(str(message.from_user.id))
        if not user_status or not user_status.get("email"):
            await message.answer(
                "📧 Email notification requested but no verified email found. Use /register to set up email notifications."
            )
            return

        client = await get_notification_client()
        if not client:
            _logger.warning("Notification client not available for email notification")
            return

        success = await client.send_notification(
            notification_type="telegram_command_response",
            title=f"Telegram Bot - {command.upper()} Command Response",
            message=response_text,
            priority="normal",
            channels=["email"],
            email_receiver=user_status["email"],
            recipient_id=str(message.from_user.id),
            data={"command": command, "telegram_user_id": str(message.from_user.id), "source": "telegram_bot"},
        )
        if success:
            _logger.info("Email notification sent for %s to user %s", command, message.from_user.id)
        else:
            _logger.warning("Email notification failed for %s to user %s", command, message.from_user.id)

    except Exception as exc:
        _logger.error("Error sending email notification for %s: %s", command, exc)


async def send_email_notification_with_attachments(
    message: Message, response_text: str, command: str, attachments: dict = None
) -> None:
    """Send email copy with optional file attachments."""
    try:
        parsed = parse_command(message.text)
        if not parsed.args.get("email", False):
            return

        telegram_svc, _ = get_service_instances()
        if not telegram_svc:
            return

        user_status = telegram_svc.get_user_status(str(message.from_user.id))
        if not user_status or not user_status.get("email"):
            await message.answer(
                "📧 Email notification requested but no verified email found. Use /register to set up email notifications."
            )
            return

        client = await get_notification_client()
        if not client:
            return

        success = await client.send_notification(
            notification_type="telegram_command_response",
            title=f"Telegram Bot - {command.upper()} Command Response",
            message=response_text,
            priority="normal",
            channels=["email"],
            email_receiver=user_status["email"],
            recipient_id=str(message.from_user.id),
            attachments=attachments,
            data={
                "command": command,
                "telegram_user_id": str(message.from_user.id),
                "source": "telegram_bot",
                "has_attachments": bool(attachments),
            },
        )
        if success:
            _logger.info("Email+attachments sent for %s to user %s", command, message.from_user.id)
        else:
            _logger.warning("Email+attachments failed for %s to user %s", command, message.from_user.id)
    except Exception as exc:
        _logger.error("Error sending email with attachments for %s: %s", command, exc)


async def extract_attachments_from_telegram_message(bot, message: Message) -> dict:
    """
    Extract photo / document / sticker bytes from a Telegram message.

    Args:
        bot: The live aiogram Bot instance (passed explicitly to avoid import-time init).
        message: Telegram message object.

    Returns:
        Dict of filename -> bytes.
    """
    attachments = {}
    try:
        if message.photo:
            photo = message.photo[-1]
            file_info = await bot.get_file(photo.file_id)
            file_data = await bot.download_file(file_info.file_path)
            attachments[f"photo_{photo.file_id}.jpg"] = file_data.read()

        if message.document:
            file_info = await bot.get_file(message.document.file_id)
            file_data = await bot.download_file(file_info.file_path)
            filename = message.document.file_name or f"document_{message.document.file_id}"
            attachments[filename] = file_data.read()

        if message.sticker:
            file_info = await bot.get_file(message.sticker.file_id)
            file_data = await bot.download_file(file_info.file_path)
            attachments[f"sticker_{message.sticker.file_id}.webp"] = file_data.read()

    except Exception:
        _logger.exception("Error extracting attachments:")
    return attachments


# ─── Route registration ───────────────────────────────────────────────────────


def register(dp: Dispatcher) -> None:
    """Register /report and /screener handlers onto the Dispatcher."""
    from src.telegram.handlers.common import audit_command_wrapper
    from src.telegram.screener.notifications import process_report_command, process_screener_command

    @dp.message(Command("report"))
    async def cmd_report(msg: Message):
        parsed = parse_command(msg.text)
        client = await get_notification_client()
        await audit_command_wrapper(msg, process_report_command, msg, str(msg.from_user.id), parsed, client)

    @dp.message(Command("screener"))
    async def cmd_screener(msg: Message):
        parsed = parse_command(msg.text)
        client = await get_notification_client()
        await audit_command_wrapper(msg, process_screener_command, msg, str(msg.from_user.id), parsed, client)
