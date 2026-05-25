"""
handlers/misc.py — Miscellaneous commands: /start, /help, /schedules, /feedback, /feature,
                    unknown command fallback, and the catch-all non-command handler.
"""
from aiogram import Dispatcher
from aiogram.types import Message
from aiogram.filters import Command

from src.notification.logger import setup_logger
from src.telegram.command_parser import parse_command
from src.model.telegram_bot import ParsedCommand
from src.telegram.lifecycle import get_notification_client

_logger = setup_logger("telegram_screener_bot")


async def process_feedback_command_immediate(user_id: str, parsed: ParsedCommand, message: Message) -> None:
    """Handle /feedback <message>."""
    try:
        feedback_text = " ".join(parsed.positionals)

        if not feedback_text:
            await message.reply("❌ Please provide your feedback.\n\nUsage: `/feedback Your message here`", parse_mode="Markdown")
            return
        _logger.info("User feedback from %s: %s", user_id, feedback_text)
        await message.reply("✅ **Feedback Received**\n\nThank you for your feedback! It has been forwarded to the development team.", parse_mode="Markdown")

        try:
            from src.telegram.lifecycle import get_service_instances
            telegram_svc, _ = get_service_instances()
            if telegram_svc:
                telegram_svc.add_feedback(user_id, "feedback", feedback_text)
        except Exception as exc:
            _logger.warning("Failed to store feedback in database: %s", exc)
    except Exception:
        _logger.exception("Error in /feedback command")
        await message.reply("❌ An error occurred while processing your feedback. Please try again.")


async def process_feature_command_immediate(user_id: str, parsed: ParsedCommand, message: Message) -> None:
    """Handle /feature <message> via the notifications layer."""
    from src.telegram.screener.notifications import process_feature_command
    client = await get_notification_client()
    await process_feature_command(message, user_id, parsed, client)


async def process_unknown_command_immediate(user_id: str, message: Message, help_text: str) -> None:
    """Respond to unrecognised commands."""
    await message.reply(f"❓ Unknown command. {help_text}")


# ─── Route registration ───────────────────────────────────────────────────────

async def cmd_start(msg: Message):
    """Handle /start command."""
    from src.telegram.handlers.common import HELP_TEXT, audit_command_wrapper
    _logger.info("Received /start from user %s", msg.from_user.id)
    try:
        welcome_text = f"Welcome to the Alkotrader Bot! 🤖\n\n{HELP_TEXT}"
        await msg.answer(welcome_text)
        try:
            from src.telegram.handlers.content import send_email_notification_if_requested
            await send_email_notification_if_requested(msg, welcome_text, "start")
        except Exception as exc:
            _logger.debug("Email not available for /start: %s", exc)
        try:
            async def _audit(*a, **kw):
                return {"status": "ok"}
            await audit_command_wrapper(msg, _audit, str(msg.from_user.id))
        except Exception as exc:
            _logger.debug("Audit not available for /start: %s", exc)
    except Exception:
        _logger.exception("Error in /start for user %s", msg.from_user.id)
        await msg.answer("Sorry, there was an error. Please try again.")


async def cmd_help(msg: Message):
    """Handle /help command."""
    from src.telegram.handlers.common import HELP_TEXT, audit_command_wrapper
    try:
        await msg.answer(HELP_TEXT)
        try:
            from src.telegram.handlers.content import send_email_notification_if_requested
            await send_email_notification_if_requested(msg, HELP_TEXT, "help")
        except Exception as exc:
            _logger.debug("Email not available for /help: %s", exc)
        try:
            async def _audit(*a, **kw):
                return {"status": "ok"}
            await audit_command_wrapper(msg, _audit, str(msg.from_user.id))
        except Exception as exc:
            _logger.debug("Audit not available for /help: %s", exc)
    except Exception:
        _logger.exception("Error in /help for user %s", msg.from_user.id)
        try:
            await msg.answer(HELP_TEXT)
        except Exception:
            await msg.answer("Sorry, there was an error. Please try again.")


async def cmd_schedules(msg: Message):
    """Handle /schedules command."""
    from src.telegram.handlers.common import audit_command_wrapper
    from src.telegram.screener.notifications import process_schedules_command
    parsed = parse_command(msg.text)
    client = await get_notification_client()
    await audit_command_wrapper(msg, process_schedules_command, msg, str(msg.from_user.id), parsed, client)


async def cmd_feedback(msg: Message):
    """Handle /feedback command."""
    from src.telegram.handlers.common import audit_command_wrapper
    parsed = parse_command(msg.text)
    await audit_command_wrapper(msg, process_feedback_command_immediate, str(msg.from_user.id), parsed, msg)


async def cmd_feature(msg: Message):
    """Handle /feature command."""
    from src.telegram.handlers.common import audit_command_wrapper
    parsed = parse_command(msg.text)
    await audit_command_wrapper(msg, process_feature_command_immediate, str(msg.from_user.id), parsed, msg)


async def unknown_command(msg: Message):
    """
    Handle unknown slash commands.

    P2-TG-6: The manual case-insensitive dispatch table has been removed.
    All Command() filters now use ignore_case=True (see register() below),
    so aiogram handles case-insensitivity natively without a secondary
    lookup table that could diverge from the registered handlers.
    """
    from src.telegram.handlers.common import HELP_TEXT, audit_command_wrapper
    _logger.info("Unknown command: %s from user %s", msg.text, msg.from_user.id)
    try:
        await audit_command_wrapper(
            msg, process_unknown_command_immediate, str(msg.from_user.id), msg, HELP_TEXT
        )
    except Exception:
        _logger.exception("Error processing unknown command for user %s", msg.from_user.id)
        await msg.answer("Sorry, there was an error. Please try again.")


async def all_messages(msg: Message):
    """Handle non-command messages."""
    from src.telegram.handlers.common import audit_command_wrapper
    _logger.info("Received non-command message from user %s: %s", msg.from_user.id, msg.text)
    if msg.text and not msg.text.startswith("/"):
        try:
            help_msg = f"❓ I don't understand '{msg.text}'\n\nPlease use /help to see all available commands."
            await msg.answer(help_msg)
            async def _audit(*a, **kw):
                return {"status": "ok"}
            await audit_command_wrapper(msg, _audit, str(msg.from_user.id))
        except Exception:
            _logger.exception("Error processing non-command message for user %s", msg.from_user.id)
            await msg.answer("Please use /help to see available commands.")


def register(dp: Dispatcher) -> None:
    """Register all misc handlers onto the Dispatcher.

    All Command() filters use ignore_case=True (P2-TG-6) so aiogram handles
    case normalisation natively, removing the need for a secondary dispatch table.
    """
    dp.message(Command("start", ignore_case=True))(cmd_start)
    dp.message(Command("help", ignore_case=True))(cmd_help)
    dp.message(Command("schedules", ignore_case=True))(cmd_schedules)
    dp.message(Command("feedback", ignore_case=True))(cmd_feedback)
    dp.message(Command("feature", ignore_case=True))(cmd_feature)

    # Unknown slash command (anything that didn't match a registered Command filter)
    dp.message(lambda m: m.text and m.text.startswith("/"))(unknown_command)

    # Non-command messages
    dp.message()(all_messages)
