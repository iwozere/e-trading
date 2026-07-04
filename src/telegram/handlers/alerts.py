"""
handlers/alerts.py — /alerts command handler.

Routes through the AlertManager business-logic layer (P2-TG-1).
AlertManager.handle_alerts_add already enforces per-user alert quotas (P2-TG-5).
Services are cached at module level to avoid per-request instantiation.
"""

from aiogram import Dispatcher
from aiogram.filters import Command
from aiogram.types import Message

from src.model.telegram_bot import ParsedCommand
from src.notification.logger import setup_logger
from src.telegram.command_parser import parse_command

_logger = setup_logger("telegram_screener_bot")


async def process_alerts_command_immediate(user_id: str, parsed: ParsedCommand, message: Message) -> None:
    """Delegate /alerts to AlertManager, which enforces user quotas and owns all alert state."""
    try:
        from src.telegram.screener.business_logic import get_business_logic

        # Inject caller identity so AlertManager can authorise the operation
        parsed.args["telegram_user_id"] = user_id

        logic = get_business_logic()
        if logic is None:
            await message.reply("❌ Service temporarily unavailable. Please try again later.")
            return

        result = logic.alert_manager.handle_alerts(parsed)

        status = result.get("status", "error")
        title = result.get("title", "")
        body = result.get("message", "An error occurred")

        if status == "error":
            await message.reply(f"❌ {body}", parse_mode="Markdown")
        else:
            text = f"**{title}**\n\n{body}" if title else body
            await message.reply(text or "✅ Done", parse_mode="Markdown")

    except Exception:
        _logger.exception("Error processing /alerts for user %s", user_id)
        await message.reply("❌ An error occurred while processing your alert command. Please try again.")


def register(dp: Dispatcher) -> None:
    """Register the /alerts handler onto the Dispatcher."""
    from src.telegram.handlers.common import audit_command_wrapper

    @dp.message(Command("alerts", ignore_case=True))
    async def cmd_alerts(msg: Message):
        parsed = parse_command(msg.text or "")
        await audit_command_wrapper(msg, process_alerts_command_immediate, str(msg.from_user.id), parsed, msg)
