import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import tempfile
from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.filters import Command, CommandObject
from aiogram.utils.keyboard import ReplyKeyboardBuilder
import asyncio
import random
from src.notification.async_notification_manager import initialize_notification_manager
from config.donotshare.donotshare import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, SMTP_USER, SMTP_PASSWORD
from src.frontend.telegram.screener.notifications import (
    process_report_command, process_help_command, process_info_command, process_register_command, process_verify_command, process_language_command, process_admin_command, process_alerts_command, process_schedules_command, process_feedback_command, process_feature_command, process_request_approval_command, process_unknown_command
)

# Configure logging
from src.notification.logger import setup_logger, set_logging_context
logger = setup_logger("telegram_screener_bot")


# Initialize bot and dispatcher
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

HELP_TEXT = (
    "Welcome to the Telegram Screener Bot!\n\n"
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
    "  -provider=yf,bnc: Data provider (yf=Yahoo, bnc=Binance)\n\n"

    "Alert Commands:\n"
    "/alerts - List all your active price alerts\n"
    "/alerts add TICKER PRICE CONDITION [flags] - Add price alert\n"
    "  CONDITION: above or below\n"
    "  Example: /alerts add BTCUSDT 65000 above -email\n"
    "Flags:\n"
    "  -email: Send alert notification to email\n"
    "/alerts edit ALERT_ID [PRICE] [CONDITION] [flags] - Edit alert\n"
    "  Example: /alerts edit 1 70000 below -email\n"
    "/alerts delete ALERT_ID - Delete alert\n"
    "/alerts pause ALERT_ID - Pause alert\n"
    "/alerts resume ALERT_ID - Resume alert\n\n"

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

@dp.message(Command("request_approval"))
async def cmd_request_approval(message: Message):
    telegram_user_id = str(message.from_user.id)
    args = message.text.split()
    await process_request_approval_command(message, telegram_user_id, args, notification_manager)

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

    # Set logging context so that notification manager logs go to telegram bot log file
    set_logging_context("telegram_screener_bot")

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