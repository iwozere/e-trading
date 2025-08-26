import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import tempfile
import time
from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.filters import Command, CommandObject
from aiogram.utils.keyboard import ReplyKeyboardBuilder
import asyncio
import random
from src.notification.async_notification_manager import initialize_notification_manager
from config.donotshare.donotshare import TELEGRAM_BOT_TOKEN, SMTP_USER, SMTP_PASSWORD
from src.frontend.telegram.screener.notifications import (
    process_report_command, process_help_command, process_info_command, process_register_command, process_verify_command, process_language_command, process_admin_command, process_alerts_command, process_schedules_command, process_screener_command, process_feedback_command, process_feature_command, process_request_approval_command, process_unknown_command
)
from src.frontend.telegram import db

# Configure logging
from src.notification.logger import setup_logger, set_logging_context
_logger = setup_logger("telegram_screener_bot")

# HTTP API support
from aiohttp import web
import json
from typing import Dict, Any, Optional


# Initialize bot and dispatcher
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

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

        # Use notification manager to send message
        success = await notification_manager.send_notification(
            notification_type="INFO",
            title=title,
            message=message,
            priority="NORMAL",
            channels=["telegram"],
            telegram_chat_id=int(user_id)
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

        # Get all registered users
        users = db.list_users()
        if not users:
            return web.json_response({
                'success': False,
                'error': 'No registered users found'
            }, status=404)

        success_count = 0
        total_count = len(users)

        # Queue messages for all users
        for user in users:
            user_id = user["telegram_user_id"]
            if user_id and user_id.isdigit():
                success = await notification_manager.send_notification(
                    notification_type="INFO",
                    title=title,
                    message=message,
                    priority="NORMAL",
                    channels=["telegram"],
                    telegram_chat_id=int(user_id)
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

async def api_status(request: web.Request) -> web.Response:
    """API endpoint for health check and status"""
    try:
        # Get notification manager stats
        stats = notification_manager.stats if notification_manager else {}

        # Get user count
        users = db.list_users()
        user_count = len(users)

        return web.json_response({
            'success': True,
            'status': 'healthy',
            'notification_stats': stats,
            'user_count': user_count,
            'queue_size': notification_manager.notification_queue.qsize() if notification_manager else 0
        })

    except Exception as e:
        _logger.exception("Error in api_status: ")
        return web.json_response({
            'success': False,
            'error': str(e)
        }, status=500)

# Create HTTP app
api_app = web.Application()
api_app.router.add_post('/api/send_message', api_send_message)
api_app.router.add_post('/api/broadcast', api_broadcast)
api_app.router.add_get('/api/status', api_status)
api_app.router.add_get('/api/test', lambda r: web.json_response({'status': 'ok', 'message': 'Bot API is working!'}))

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

async def audit_command_wrapper(message: Message, command_func, *args, **kwargs):
    """Wrapper function to audit all commands."""
    start_time = time.time()
    telegram_user_id = str(message.from_user.id)
    command = message.text.split()[0] if message.text else ""
    full_message = message.text

    # Check if user is registered
    user_status = db.get_user_status(telegram_user_id)
    is_registered_user = user_status is not None
    user_email = user_status.get('email') if user_status else None

    try:
        # Execute the command
        result = await command_func(message, *args, **kwargs)

        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)

        # Log successful command
        db.log_command_audit(
            telegram_user_id=telegram_user_id,
            command=command,
            full_message=full_message,
            is_registered_user=is_registered_user,
            user_email=user_email,
            success=True,
            response_time_ms=response_time_ms
        )

        return result

    except Exception as e:
        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)

        # Log failed command
        db.log_command_audit(
            telegram_user_id=telegram_user_id,
            command=command,
            full_message=full_message,
            is_registered_user=is_registered_user,
            user_email=user_email,
            success=False,
            error_message=str(e),
            response_time_ms=response_time_ms
        )

        # Re-raise the exception
        raise

@dp.message(Command("start"))
async def cmd_start(message: Message):
    _logger.info("Received /start command from user %s", message.from_user.id)
    try:
        await audit_command_wrapper(message, process_help_command, str(message.from_user.id), message.text, notification_manager)
        _logger.info("Successfully processed /start command for user %s", message.from_user.id)
    except Exception as e:
        _logger.exception("Error processing /start command for user %s", message.from_user.id)
        # Send a simple error message directly
        await message.answer("Sorry, there was an error processing your command. Please try again.")

@dp.message(Command("help"))
async def cmd_help(message: Message):
    await audit_command_wrapper(message, process_help_command, str(message.from_user.id), message.text, notification_manager)

@dp.message(Command("info"))
async def cmd_info(message: Message):
    await audit_command_wrapper(message, process_info_command, str(message.from_user.id), notification_manager)

@dp.message(Command("register"))
async def cmd_register(message: Message):
    args = message.text.split()
    await audit_command_wrapper(message, process_register_command, str(message.from_user.id), args, notification_manager)

@dp.message(Command("verify"))
async def cmd_verify(message: Message):
    args = message.text.split()
    await audit_command_wrapper(message, process_verify_command, str(message.from_user.id), args, notification_manager)

@dp.message(Command("request_approval"))
async def cmd_request_approval(message: Message):
    args = message.text.split()
    await audit_command_wrapper(message, process_request_approval_command, str(message.from_user.id), args, notification_manager)

@dp.message(Command("language"))
async def cmd_language(message: Message):
    args = message.text.split()
    await audit_command_wrapper(message, process_language_command, str(message.from_user.id), args, notification_manager)

@dp.message(Command("admin"))
async def cmd_admin(message: Message):
    args = message.text.split()
    await audit_command_wrapper(message, process_admin_command, str(message.from_user.id), args, notification_manager)

@dp.message(Command("report"))
async def cmd_report(message: Message):
    args = message.text.split()
    await audit_command_wrapper(message, process_report_command, str(message.from_user.id), args, notification_manager)

@dp.message(Command("alerts"))
async def cmd_alerts(message: Message):
    args = message.text.split()
    await audit_command_wrapper(message, process_alerts_command, str(message.from_user.id), args, notification_manager)

@dp.message(Command("schedules"))
async def cmd_schedules(message: Message):
    args = message.text.split()
    await audit_command_wrapper(message, process_schedules_command, str(message.from_user.id), args, notification_manager)

@dp.message(Command("screener"))
async def cmd_screener(message: Message):
    args = message.text.split()
    await audit_command_wrapper(message, process_screener_command, str(message.from_user.id), args, notification_manager)

@dp.message(Command("feedback"))
async def cmd_feedback(message: Message):
    args = message.text.split(maxsplit=1)
    await audit_command_wrapper(message, process_feedback_command, str(message.from_user.id), args, notification_manager)

@dp.message(Command("feature"))
async def cmd_feature(message: Message):
    args = message.text.split(maxsplit=1)
    await audit_command_wrapper(message, process_feature_command, str(message.from_user.id), args, notification_manager)

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
        except Exception as e:
            _logger.exception("Error processing case-insensitive command %s for user %s", command_name, message.from_user.id)
            await message.answer("Sorry, there was an error processing your command. Please try again.")
            return

    # If not a case variation, process as unknown command
    try:
        await audit_command_wrapper(message, process_unknown_command, str(message.from_user.id), notification_manager, HELP_TEXT)
    except Exception as e:
        _logger.exception("Error processing unknown command for user %s", message.from_user.id)
        await message.answer("Sorry, there was an error processing your command. Please try again.")

@dp.message()
async def all_messages(message: Message):
    """Catch all messages for debugging"""
    _logger.info("Received message: %s from user %s", message.text, message.from_user.id)

async def main():
    global notification_manager

    _logger.info("Starting bot initialization...")
    if TELEGRAM_BOT_TOKEN:
        _logger.info("Bot token: %s...", TELEGRAM_BOT_TOKEN[:10])
    else:
        _logger.error("Bot token is None!")

    # Set logging context so that notification manager logs go to telegram bot log file
    set_logging_context("telegram_screener_bot")

    # Initialize notification manager with a dummy chat ID for Telegram channel creation
    # The actual chat ID will be provided dynamically in each message
    _logger.info("Initializing notification manager...")
    notification_manager = await initialize_notification_manager(
        telegram_token=TELEGRAM_BOT_TOKEN,
        telegram_chat_id="0",  # Dummy chat ID - actual chat ID provided dynamically
        email_api_key=SMTP_PASSWORD,
        email_sender=SMTP_USER,
        email_receiver=SMTP_USER  # or dummy
    )

    # Disable batching for immediate processing in bot context
    notification_manager.batch_size = 1
    notification_manager.batch_timeout = 0.1
    _logger.info("Notification manager initialized successfully")

    _logger.info("Starting Telegram Screener Bot with HTTP API...")

    # Start both bot polling and HTTP API server
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

if __name__ == "__main__":
    asyncio.run(main())
