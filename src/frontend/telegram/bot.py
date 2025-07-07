import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import tempfile
from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.filters import Command, CommandObject
from aiogram.utils.keyboard import ReplyKeyboardBuilder
import asyncio
import random
from src.frontend.telegram import db
from src.notification.async_notification_manager import initialize_notification_manager, NotificationType, NotificationPriority
from config.donotshare.donotshare import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, SMTP_USER, SMTP_PASSWORD
from src.frontend.telegram.command_parser import parse_command, ParsedCommand
from src.frontend.telegram.screener.business_logic import handle_command, is_admin_user
from src.frontend.telegram.screener.notifications import (
    process_report_command, process_help_command, process_info_command, process_register_command, process_verify_command, process_language_command, process_admin_command, process_alerts_command, process_schedules_command, process_feedback_command, process_feature_command, process_unknown_command
)

# Configure logging
from src.notification.logger import setup_logger
logger = setup_logger("telegram_screener_bot")


# Initialize bot and dispatcher
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

HELP_TEXT = (
    "Welcome to the Telegram Screener Bot!\n\n"
    "/start - Show welcome message\n"
    "/help - Show this help message\n"
    "/info - Show your registered email and verification status\n"
    "/register email@example.com - Register or update your email for reports\n"
    "/verify CODE - Verify your email with the code sent\n"
    "/report TICKER1 TICKER2 ... [flags] - Get a report for specified tickers (stock or crypto). Flags:\n"
    "    -email: Send the report to your registered email\n"
    "    -indicators=...: Comma-separated list of indicators (e.g., RSI,MACD,MA50,PE,EPS)\n"
    "    -period=...: Data period (e.g., 3mo, 1y, 2y). Optional, default: 2y\n"
    "    -interval=...: Data interval (e.g., 1d, 15m). Optional, default: 1d\n"
    "    -provider=...: Data provider (e.g., yf for Yahoo, bnc for Binance). Optional. If not set: use yf for tickers <5 chars, bnc otherwise\n"
    "/alerts - List all your active price alerts\n"
    "/alerts add TICKER PRICE CONDITION - Add a new price alert (CONDITION: above/below)\n"
    "/alerts edit ALERT_ID [params] - Edit an existing alert\n"
    "/alerts delete ALERT_ID - Delete an alert by its ID\n"
    "/alerts pause ALERT_ID - Pause a specific alert\n"
    "/alerts resume ALERT_ID - Resume a paused alert\n"
    "/schedules - List all your scheduled reports\n"
    "/schedules add TICKER TIME [flags] - Schedule a report for a ticker at a specific time (24h UTC)\n"
    "/schedules edit SCHEDULE_ID [params] - Edit a scheduled report\n"
    "/schedules delete SCHEDULE_ID - Delete a scheduled report by its ID\n"
    "/schedules pause SCHEDULE_ID - Pause a scheduled report\n"
    "/schedules resume SCHEDULE_ID - Resume a paused scheduled report\n"
    "/feedback MESSAGE - Send feedback or bug report to the admin/developer\n"
    "/feature MESSAGE - Suggest a new feature\n"
    "/language LANG - Update language preference\n"
    "\nFor more details and examples, see the documentation or use /help."
)

ADMIN_HELP_TEXT = (
    "Admin Commands:\n"
    "/admin help - List all admin commands\n"
    "/admin listusers - List all users as telegram_user_id - email pairs\n"
    "/admin users - List all registered users and emails\n"
    "/admin setlimit alerts N - Set global default max alerts per user\n"
    "/admin setlimit alerts N TELEGRAM_USER_ID - Set per-user max alerts\n"
    "/admin setlimit schedules N - Set global default max schedules per user\n"
    "/admin setlimit schedules N TELEGRAM_USER_ID - Set per-user max schedules\n"
    "/admin resetemail TELEGRAM_USER_ID - Reset a user's email\n"
    "/admin verify TELEGRAM_USER_ID - Manually verify a user's email\n"
    "/admin broadcast MESSAGE - Send a broadcast message to all users\n"
)

def generate_code():
    return f"{random.randint(100000, 999999):06d}"

@dp.message(Command("start"))
async def cmd_start(message: Message):
    telegram_user_id = str(message.from_user.id)
    await process_help_command(message, telegram_user_id, notification_manager)

@dp.message(Command("help"))
async def cmd_help(message: Message):
    telegram_user_id = str(message.from_user.id)
    await process_help_command(message, telegram_user_id, notification_manager)

@dp.message(Command("info"))
async def cmd_info(message: Message):
    telegram_user_id = str(message.from_user.id)
    await process_info_command(message, telegram_user_id, notification_manager)

@dp.message(Command("register"))
async def cmd_register(message: Message):
    telegram_user_id = str(message.from_user.id)
    args = message.text.split()
    await process_register_command(message, telegram_user_id, args, notification_manager)

@dp.message(Command("verify"))
async def cmd_verify(message: Message):
    telegram_user_id = str(message.from_user.id)
    args = message.text.split()
    await process_verify_command(message, telegram_user_id, args, notification_manager)

@dp.message(Command("language"))
async def cmd_language(message: Message):
    telegram_user_id = str(message.from_user.id)
    args = message.text.split()
    await process_language_command(message, telegram_user_id, args, notification_manager)

@dp.message(Command("admin"))
async def cmd_admin(message: Message):
    telegram_user_id = str(message.from_user.id)
    args = message.text.split()
    await process_admin_command(message, telegram_user_id, args, notification_manager)

@dp.message(Command("report"))
async def cmd_report(message: Message):
    telegram_user_id = str(message.from_user.id)
    args = message.text.split()
    await process_report_command(message, telegram_user_id, args, notification_manager)

@dp.message(Command("alerts"))
async def cmd_alerts(message: Message):
    telegram_user_id = str(message.from_user.id)
    args = message.text.split()
    await process_alerts_command(message, telegram_user_id, args, notification_manager)

@dp.message(Command("schedules"))
async def cmd_schedules(message: Message):
    telegram_user_id = str(message.from_user.id)
    args = message.text.split()
    await process_schedules_command(message, telegram_user_id, args, notification_manager)

@dp.message(Command("feedback"))
async def cmd_feedback(message: Message):
    telegram_user_id = str(message.from_user.id)
    args = message.text.split(maxsplit=1)
    await process_feedback_command(message, telegram_user_id, args, notification_manager)

@dp.message(Command("feature"))
async def cmd_feature(message: Message):
    telegram_user_id = str(message.from_user.id)
    args = message.text.split(maxsplit=1)
    await process_feature_command(message, telegram_user_id, args, notification_manager)

@dp.message(lambda message: message.text and message.text.startswith("/"))
async def unknown_command(message: Message):
    telegram_user_id = str(message.from_user.id)
    await process_unknown_command(message, telegram_user_id, notification_manager, HELP_TEXT)

async def main():
    global notification_manager
    notification_manager = await initialize_notification_manager(
        telegram_token=TELEGRAM_BOT_TOKEN,
        telegram_chat_id=TELEGRAM_CHAT_ID,
        email_api_key=SMTP_PASSWORD,
        email_sender=SMTP_USER,
        email_receiver=SMTP_USER  # or dummy
    )
    logger.info("Starting Telegram Screener Bot...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())