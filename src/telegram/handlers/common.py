"""
handlers/common.py — Shared utilities used across all handler modules.

Contains:
- audit_command_wrapper: wraps command execution with timing, logging, DB audit.
- HELP_TEXT / ADMIN_HELP_TEXT: canonical help strings.
"""
import time
from aiogram.types import Message

from src.notification.logger import setup_logger
from src.telegram.lifecycle import get_service_instances

_logger = setup_logger("telegram_screener_bot")


HELP_TEXT = (
    "Welcome to the Telegram Screener Bot!\n\n"
    "📧 Email Notifications: Add -email to any command to receive the response via email as well as Telegram.\n"
    "Example: /help -email, /info -email, /alerts -email\n\n"
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

    "Screener Commands:\n"
    "/screener JSON_CONFIG [-email] - Run enhanced screener immediately\n"
    "Flags:\n"
    "  -email: Send results to your registered email\n\n"

    "Alert Commands:\n"
    "/alerts - List all your active price alerts\n"
    "/alerts add TICKER PRICE above/below [flags] - Add price alert\n"
    "/alerts delete ALERT_ID - Delete alert by ID\n"
    "/alerts evaluate - Check your alerts now\n\n"

    "Schedule Commands:\n"
    "/schedules - List all your scheduled reports\n"
    "/schedules add TICKER TIME [flags] - Schedule daily report\n"
    "/schedules delete SCHEDULE_ID - Delete schedule\n\n"

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


async def audit_command_wrapper(message: Message, command_func, *args, **kwargs):
    """
    Wrap a command handler with service availability check, timing, and DB audit logging.

    Handlers access services via lifecycle.get_service_instances(), which in turn
    delegates to business_logic.get_service_instances().
    """
    start_time = time.time()
    telegram_user_id = str(message.from_user.id)
    command = message.text.split()[0] if message.text else ""
    full_message = message.text

    # Guard: service must be available
    try:
        telegram_svc, _ = get_service_instances()
        if not telegram_svc:
            _logger.error("TelegramService not initialised for command %s from user %s", command, telegram_user_id)
            try:
                await message.answer("Service temporarily unavailable. Please try again later.")
            except Exception:
                pass
            return
    except Exception as exc:
        _logger.error("Error getting service instances for %s from %s: %s", command, telegram_user_id, exc)
        try:
            await message.answer("Service temporarily unavailable. Please try again later.")
        except Exception:
            pass
        return

    # Resolve calling user once before execution
    try:
        user_status = telegram_svc.get_user_status(telegram_user_id)
        is_registered_user = user_status is not None
        user_email = user_status.get("email") if user_status else None
    except Exception as exc:
        _logger.warning("Could not get user status for %s before %s: %s", telegram_user_id, command, exc)
        user_status = None
        is_registered_user = False
        user_email = None

    try:
        result = await command_func(*args, **kwargs)
        response_time_ms = int((time.time() - start_time) * 1000)

        try:
            _logger.info(
                "Command OK: user=%s command=%s time=%dms registered=%s",
                telegram_user_id, command, response_time_ms, is_registered_user,
            )
            telegram_svc.log_command_audit(
                telegram_user_id=telegram_user_id,
                command=command,
                full_message=full_message,
                is_registered_user=is_registered_user,
                user_email=user_email,
                success=True,
                response_time_ms=response_time_ms,
            )
        except Exception as audit_exc:
            _logger.warning("Failed to log success audit for %s/%s: %s", telegram_user_id, command, audit_exc)

        return result

    except Exception as exc:
        response_time_ms = int((time.time() - start_time) * 1000)

        try:
            _logger.error(
                "Command FAILED: user=%s command=%s time=%dms registered=%s error=%s",
                telegram_user_id, command, response_time_ms, is_registered_user, exc,
            )
            telegram_svc.log_command_audit(
                telegram_user_id=telegram_user_id,
                command=command,
                full_message=full_message,
                is_registered_user=is_registered_user,
                user_email=user_email,
                success=False,
                error_message=str(exc),
                response_time_ms=response_time_ms,
            )
        except Exception as audit_exc:
            _logger.warning("Failed to log failure audit for %s/%s: %s", telegram_user_id, command, audit_exc)

        raise
