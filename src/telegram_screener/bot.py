import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.filters import Command, CommandObject
from aiogram.utils.keyboard import ReplyKeyboardBuilder
import asyncio
import random
import time
from src.telegram_screener import db
import sqlite3
from src.notification.logger import setup_logger

# Configure logging
setup_logger("telegram_screener_bot")
logger = setup_logger("telegram_screener_bot")

# Load token from environment or config (placeholder)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TOKEN_HERE")

# Initialize bot and dispatcher
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

HELP_TEXT = (
    "<b>Welcome to the Telegram Screener Bot!</b>\n\n"
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
    "/admin users - List all registered users and emails\n"
    "/admin resetemail TELEGRAM_USER_ID - Reset a user's email\n"
    "/admin verify TELEGRAM_USER_ID - Manually verify a user's email\n"
    "/admin setlimit alerts N - Set default max number of alerts per user\n"
    "/admin setlimit schedules N - Set default max number of scheduled reports per user\n"
    "/admin broadcast MESSAGE - Send a broadcast message to all users\n"
    "/feedback MESSAGE - Send feedback or bug report to the admin/developer\n"
    "/feature MESSAGE - Suggest a new feature\n"
    "/language LANG - Update language preference\n"
    "\nFor more details and examples, see the documentation or use /help."
)

ADMIN_HELP_TEXT = (
    "<b>Admin Commands:</b>\n"
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

def is_admin_user(telegram_user_id: str) -> bool:
    db.init_db()
    status = db.get_user_status(telegram_user_id)
    return status and status.get("is_admin", False)

@dp.message(Command("start"))
async def cmd_start(message: Message):
    """Handle /start command."""
    await message.answer(HELP_TEXT, parse_mode="HTML")

@dp.message(Command("help"))
async def cmd_help(message: Message):
    """Handle /help command."""
    await message.answer(HELP_TEXT, parse_mode="HTML")

@dp.message(Command("info"))
async def cmd_info(message: Message):
    telegram_user_id = str(message.from_user.id)
    db.init_db()
    status = db.get_user_status(telegram_user_id)
    if status:
        email = status["email"] or "(not set)"
        verified = "Yes" if status["verified"] else "No"
        language = status["language"] or "(not set)"
        await message.answer(f"<b>Your info:</b>\nEmail: {email}\nVerified: {verified}\nLanguage: {language}", parse_mode="HTML")
    else:
        await message.answer("<b>Your info:</b>\nEmail: (not set)\nVerified: No\nLanguage: (not set)", parse_mode="HTML")

@dp.message(Command("register"))
async def cmd_register(message: Message):
    """Handle /register email@example.com [lang] command."""
    args = message.text.split()
    if len(args) < 2 or "@" not in args[1]:
        await message.answer("Usage: /register email@example.com [lang]")
        return
    telegram_user_id = str(message.from_user.id)
    email = args[1].strip()
    language = args[2].strip().lower() if len(args) > 2 else None
    db.init_db()
    if db.count_codes_last_hour(telegram_user_id) >= 5:
        await message.answer("Too many verification attempts. Please try again later.")
        return
    code = generate_code()
    sent_time = int(time.time())
    db.set_user_email(telegram_user_id, email, code, sent_time, language)
    # TODO: Replace with real email sending
    logger.info(f"[MOCK EMAIL] Sent code {code} to {email} (lang={language})")
    await message.answer(f"Verification code sent to {email}. Please check your inbox and use /verify CODE in Telegram.")

@dp.message(Command("verify"))
async def cmd_verify(message: Message):
    """Handle /verify CODE command."""
    args = message.text.split()
    if len(args) != 2 or not args[1].isdigit():
        await message.answer("Usage: /verify CODE (6 digits)")
        return
    telegram_user_id = str(message.from_user.id)
    code = args[1]
    db.init_db()
    if db.verify_code(telegram_user_id, code):
        await message.answer("✅ Email verified successfully! You can now use -email flag to receive reports by email.")
    else:
        await message.answer("❌ Invalid or expired code. Please try again or /register again.")

@dp.message(Command("language"))
async def cmd_language(message: Message):
    """Handle /language LANG command to update language preference."""
    args = message.text.split()
    if len(args) != 2:
        await message.answer("Usage: /language LANG (e.g., en, ru)")
        return
    lang = args[1].strip().lower()
    telegram_user_id = str(message.from_user.id)
    db.init_db()
    status = db.get_user_status(telegram_user_id)
    if not status:
        await message.answer("You must register your email first with /register.")
        return
    # Update language
    conn = sqlite3.connect(db.DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE users SET language=? WHERE telegram_user_id=?", (lang, telegram_user_id))
    conn.commit()
    conn.close()
    await message.answer(f"Language updated to {lang}.")

@dp.message(Command("admin"))
async def cmd_admin(message: Message):
    args = message.text.split()
    telegram_user_id = str(message.from_user.id)
    if not is_admin_user(telegram_user_id):
        await message.answer("You are not authorized to use admin commands.")
        return
    if len(args) < 2:
        await message.answer(ADMIN_HELP_TEXT, parse_mode="HTML")
        return
    subcmd = args[1].lower()
    if subcmd == "help":
        await message.answer(ADMIN_HELP_TEXT, parse_mode="HTML")
    elif subcmd == "listusers":
        users = db.list_users()
        if not users:
            await message.answer("No users found.")
        else:
            text = "<b>Users:</b>\n" + "\n".join(f"{uid} - {email or '(no email)'}" for uid, email in users)
            await message.answer(text, parse_mode="HTML")
    elif subcmd == "users":
        users = db.list_users()
        if not users:
            await message.answer("No users found.")
        else:
            text = "<b>Users:</b>\n" + "\n".join(f"{uid} - {email or '(no email)'}" for uid, email in users)
            await message.answer(text, parse_mode="HTML")
    elif subcmd == "setlimit":
        if len(args) < 4:
            await message.answer("Usage: /admin setlimit [alerts|schedules] N [TELEGRAM_USER_ID]")
            return
        limit_type = args[2].lower()
        if limit_type not in ("alerts", "schedules"):
            await message.answer("Limit type must be 'alerts' or 'schedules'.")
            return
        try:
            n = int(args[3])
        except ValueError:
            await message.answer("N must be an integer.")
            return
        if len(args) == 5:
            # Per-user
            target_uid = args[4]
            db.set_user_limit(target_uid, f"max_{limit_type}", n)
            await message.answer(f"Set max_{limit_type} for user {target_uid} to {n}.")
        else:
            # Global
            db.set_setting(f"max_{limit_type}", str(n))
            await message.answer(f"Set global max_{limit_type} to {n}.")
    else:
        await message.answer(ADMIN_HELP_TEXT, parse_mode="HTML")

@dp.message(lambda message: message.text and message.text.startswith("/"))
async def unknown_command(message: Message):
    """Handle unknown commands by showing help."""
    await message.answer("Unknown command.\n" + HELP_TEXT, parse_mode="HTML")

async def main():
    logger.info("Starting Telegram Screener Bot...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())