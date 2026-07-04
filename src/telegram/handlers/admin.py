"""
handlers/admin.py — /admin command handler.
Delegates all admin business logic to screener/notifications.py.
"""

from aiogram import Dispatcher
from aiogram.filters import Command
from aiogram.types import Message

from src.model.telegram_bot import ParsedCommand
from src.notification.logger import setup_logger
from src.telegram.command_parser import parse_command
from src.telegram.lifecycle import get_notification_client

_logger = setup_logger("telegram_screener_bot")


async def process_admin_command_immediate(user_id: str, parsed: ParsedCommand, message: Message) -> None:
    """Delegate admin operations to the notifications business logic layer."""
    from src.telegram.screener.notifications import process_admin_command

    client = await get_notification_client()
    await process_admin_command(message, user_id, parsed, client)


async def cmd_admin(msg: Message):
    """Handle /admin command."""
    from src.telegram.handlers.common import audit_command_wrapper

    parsed = parse_command(msg.text)
    await audit_command_wrapper(msg, process_admin_command_immediate, str(msg.from_user.id), parsed, msg)


def register(dp: Dispatcher) -> None:
    """Register the /admin handler onto the Dispatcher."""
    dp.message(Command("admin"))(cmd_admin)
