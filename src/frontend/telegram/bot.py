import os
import sys
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.filters import Command, CommandObject
from aiogram.utils.keyboard import ReplyKeyboardBuilder
import asyncio
import random
import time
from src.frontend.telegram import db
import sqlite3
from src.notification.logger import setup_logger
from src.notification.async_notification_manager import initialize_notification_manager, NotificationType, NotificationPriority
from config.donotshare.donotshare import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, SMTP_USER, SMTP_PASSWORD
from src.frontend.telegram.command_parser import parse_command, ParsedCommand
from src.frontend.telegram.screener.business_logic import handle_command, is_admin_user

# Configure logging
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
    """Handle /start command."""
    print(">>> /start handler triggered")
    logger.info("Received /start command")
    try:
        telegram_user_id = str(message.from_user.id)

        # Use business logic to get help text
        parsed = ParsedCommand(
            command="help",
            args={"telegram_user_id": telegram_user_id}
        )
        result = handle_command(parsed)

        if result["status"] == "ok":
            await message.answer(result["help_text"], parse_mode="MarkdownV2")
        else:
            await message.answer(f"❌ Error: {result.get('message', 'Unknown error')}")

        print(">>> Sent message to user")
    except Exception as e:
        print(f"Exception sending message: {e}")
        logger.error(f"Exception sending message: {e}", exc_info=True)

@dp.message(Command("help"))
async def cmd_help(message: Message):
    """Handle /help command."""
    try:
        telegram_user_id = str(message.from_user.id)

        # Use business logic to get help text
        parsed = ParsedCommand(
            command="help",
            args={"telegram_user_id": telegram_user_id}
        )
        result = handle_command(parsed)

        if result["status"] == "ok":
            await message.answer(result["help_text"], parse_mode="MarkdownV2")
        else:
            await message.answer(f"❌ Error: {result.get('message', 'Unknown error')}")
    except Exception as e:
        logger.error(f"Error in help command: {e}", exc_info=True)
        await message.answer("❌ An error occurred while processing your request. Please try again.")

@dp.message(Command("info"))
async def cmd_info(message: Message):
    try:
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
    except Exception as e:
        logger.error(f"Error in info command: {e}", exc_info=True)
        await message.answer("❌ An error occurred while processing your request. Please try again.")

@dp.message(Command("register"))
async def cmd_register(message: Message):
    """Handle /register email@example.com [lang] command."""
    try:
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
    except Exception as e:
        logger.error(f"Error in register command: {e}", exc_info=True)
        await message.answer("❌ An error occurred while processing your request. Please try again.")

@dp.message(Command("verify"))
async def cmd_verify(message: Message):
    """Handle /verify CODE command."""
    try:
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
    except Exception as e:
        logger.error(f"Error in verify command: {e}", exc_info=True)
        await message.answer("❌ An error occurred while processing your request. Please try again.")

@dp.message(Command("language"))
async def cmd_language(message: Message):
    """Handle /language LANG command to update language preference."""
    try:
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
    except Exception as e:
        logger.error(f"Error in language command: {e}", exc_info=True)
        await message.answer("❌ An error occurred while processing your request. Please try again.")

@dp.message(Command("admin"))
async def cmd_admin(message: Message):
    try:
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
    except Exception as e:
        logger.error(f"Error in admin command: {e}", exc_info=True)
        await message.answer("❌ An error occurred while processing your request. Please try again.")

@dp.message(Command("report"))
async def cmd_report(message: Message):
    """Handle /report command for ticker analysis."""
    try:
        telegram_user_id = str(message.from_user.id)
        parsed = parse_command(message.text)

        # Check if user wants email and is verified
        if parsed.args.get("email", False):
            db.init_db()
            status = db.get_user_status(telegram_user_id)
            if not status or not status.get("verified", False):
                await message.answer("❌ You must verify your email first. Use /register and /verify commands.")
                return

        # Handle the report request
        result = handle_command(parsed)

        if result["status"] == "ok":
            analyses = result.get("analyses", [])
            if not analyses:
                await message.answer("❌ No data found for the specified tickers.")
                return

            # Send analysis results
            for analysis in analyses:
                ticker = analysis.ticker
                current_price = analysis.current_price
                change_pct = analysis.change_percentage

                # Basic analysis message
                msg = f"📊 <b>{ticker}</b>\n"
                msg += f"💰 Price: ${current_price:.2f}\n"
                msg += f"📈 Change: {change_pct:+.2f}%\n"

                # Add fundamental data if available
                if analysis.fundamentals:
                    fund = analysis.fundamentals
                    if fund.pe_ratio and fund.pe_ratio > 0:
                        msg += f"📊 P/E: {fund.pe_ratio:.2f}\n"
                    if fund.earnings_per_share and fund.earnings_per_share > 0:
                        msg += f"💵 EPS: ${fund.earnings_per_share:.2f}\n"
                    if fund.dividend_yield and fund.dividend_yield > 0:
                        msg += f"🎯 Dividend Yield: {fund.dividend_yield:.2f}%\n"

                # Add technical indicators if available
                if analysis.technicals:
                    tech = analysis.technicals
                    if tech.rsi:
                        msg += f"📉 RSI: {tech.rsi:.1f}\n"
                    if tech.macd:
                        msg += f"📊 MACD: {tech.macd:.3f}\n"

                await message.answer(msg, parse_mode="HTML")

            # Handle email sending if requested
            if result.get("email", False):
                await message.answer("📧 Report sent to your registered email address.")

        else:
            await message.answer(f"❌ Error: {result.get('message', 'Unknown error')}")

    except Exception as e:
        logger.error(f"Error in report command: {e}", exc_info=True)
        await message.answer("❌ An error occurred while processing your request. Please try again.")

@dp.message(Command("alerts"))
async def cmd_alerts(message: Message):
    """Handle /alerts command for price alerts management."""
    try:
        args = message.text.split()
        telegram_user_id = str(message.from_user.id)

        if len(args) == 1:
            # List alerts
            db.init_db()
            alerts = db.list_alerts(telegram_user_id)
            if not alerts:
                await message.answer("📋 You have no active price alerts.")
            else:
                msg = "📋 <b>Your Active Alerts:</b>\n\n"
                for alert in alerts:
                    status = "✅ Active" if alert.get("active", True) else "⏸️ Paused"
                    msg += f"ID: {alert['id']} | {alert['ticker']} | ${alert['price']} | {alert['condition']} | {status}\n"
                await message.answer(msg, parse_mode="HTML")
        elif len(args) >= 4 and args[1] == "add":
            # Add alert
            ticker = args[2].upper()
            try:
                price = float(args[3])
                condition = args[4].lower() if len(args) > 4 else "above"
                if condition not in ["above", "below"]:
                    await message.answer("❌ Condition must be 'above' or 'below'")
                    return

                db.init_db()
                alert_id = db.add_alert(telegram_user_id, ticker, price, condition)
                await message.answer(f"✅ Alert added! ID: {alert_id}\n{ticker} {condition} ${price}")
            except ValueError:
                await message.answer("❌ Invalid price. Please provide a valid number.")
        elif len(args) >= 3 and args[1] == "delete":
            # Delete alert
            try:
                alert_id = int(args[2])
                db.init_db()
                if db.delete_alert(alert_id):
                    await message.answer(f"✅ Alert {alert_id} deleted successfully.")
                else:
                    await message.answer("❌ Alert not found or you don't have permission to delete it.")
            except ValueError:
                await message.answer("❌ Invalid alert ID. Please provide a valid number.")
        else:
            await message.answer("Usage:\n/alerts - List your alerts\n/alerts add TICKER PRICE [above|below] - Add alert\n/alerts delete ID - Delete alert")
    except Exception as e:
        logger.error(f"Error in alerts command: {e}", exc_info=True)
        await message.answer("❌ An error occurred while processing your request. Please try again.")

@dp.message(Command("schedules"))
async def cmd_schedules(message: Message):
    """Handle /schedules command for scheduled reports management."""
    try:
        args = message.text.split()
        telegram_user_id = str(message.from_user.id)

        if len(args) == 1:
            # List schedules
            db.init_db()
            schedules = db.list_schedules(telegram_user_id)
            if not schedules:
                await message.answer("📅 You have no scheduled reports.")
            else:
                msg = "📅 <b>Your Scheduled Reports:</b>\n\n"
                for schedule in schedules:
                    status = "✅ Active" if schedule.get("active", True) else "⏸️ Paused"
                    msg += f"ID: {schedule['id']} | {schedule['ticker']} | {schedule['scheduled_time']} | {status}\n"
                await message.answer(msg, parse_mode="HTML")
        elif len(args) >= 3 and args[1] == "add":
            # Add schedule
            ticker = args[2].upper()
            time_str = args[3] if len(args) > 3 else "09:00"

            # Validate time format (HH:MM)
            if not re.match(r"^([01]?[0-9]|2[0-3]):[0-5][0-9]$", time_str):
                await message.answer("❌ Invalid time format. Use HH:MM (24-hour format)")
                return

            db.init_db()
            schedule_id = db.add_schedule(telegram_user_id, ticker, time_str)
            await message.answer(f"✅ Schedule added! ID: {schedule_id}\n{ticker} at {time_str} UTC daily")
        elif len(args) >= 3 and args[1] == "delete":
            # Delete schedule
            try:
                schedule_id = int(args[2])
                db.init_db()
                if db.delete_schedule(schedule_id):
                    await message.answer(f"✅ Schedule {schedule_id} deleted successfully.")
                else:
                    await message.answer("❌ Schedule not found or you don't have permission to delete it.")
            except ValueError:
                await message.answer("❌ Invalid schedule ID. Please provide a valid number.")
        else:
            await message.answer("Usage:\n/schedules - List your schedules\n/schedules add TICKER [TIME] - Add schedule\n/schedules delete ID - Delete schedule")
    except Exception as e:
        logger.error(f"Error in schedules command: {e}", exc_info=True)
        await message.answer("❌ An error occurred while processing your request. Please try again.")

@dp.message(Command("feedback"))
async def cmd_feedback(message: Message):
    """Handle /feedback command for user feedback."""
    try:
        args = message.text.split(maxsplit=1)
        if len(args) < 2:
            await message.answer("Usage: /feedback YOUR_MESSAGE")
            return

        feedback_text = args[1]
        telegram_user_id = str(message.from_user.id)

        # Store feedback in database (placeholder - you may need to add this table)
        db.init_db()
        # TODO: Add feedback table and function
        # db.add_feedback(telegram_user_id, feedback_text)

        await message.answer("✅ Thank you for your feedback! We'll review it and get back to you if needed.")
    except Exception as e:
        logger.error(f"Error in feedback command: {e}", exc_info=True)
        await message.answer("❌ An error occurred while processing your request. Please try again.")

@dp.message(Command("feature"))
async def cmd_feature(message: Message):
    """Handle /feature command for feature requests."""
    try:
        args = message.text.split(maxsplit=1)
        if len(args) < 2:
            await message.answer("Usage: /feature YOUR_FEATURE_REQUEST")
            return

        feature_text = args[1]
        telegram_user_id = str(message.from_user.id)

        # Store feature request in database (placeholder - you may need to add this table)
        db.init_db()
        # TODO: Add feature_requests table and function
        # db.add_feature_request(telegram_user_id, feature_text)

        await message.answer("✅ Thank you for your feature request! We'll consider it for future updates.")
    except Exception as e:
        logger.error(f"Error in feature command: {e}", exc_info=True)
        await message.answer("❌ An error occurred while processing your request. Please try again.")

@dp.message(lambda message: message.text and message.text.startswith("/"))
async def unknown_command(message: Message):
    """Handle unknown commands by showing help."""
    try:
        await message.answer("Unknown command.\n" + HELP_TEXT, parse_mode="MarkdownV2")
    except Exception as e:
        logger.error(f"Error in unknown command handler: {e}", exc_info=True)
        await message.answer("❌ An error occurred while processing your request. Please try again.")

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