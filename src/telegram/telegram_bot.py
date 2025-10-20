import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import tempfile
import time
from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.filters import Command, CommandObject
from aiogram.utils.keyboard import ReplyKeyboardBuilder
import asyncio
import random
from src.notification.service.client import NotificationServiceClient, MessageType, MessagePriority
from config.donotshare.donotshare import TELEGRAM_BOT_TOKEN, SMTP_USER, SMTP_PASSWORD
from src.telegram.screener.notifications import (
    process_report_command, process_help_command, process_info_command, process_register_command, process_verify_command, process_language_command, process_admin_command, process_alerts_command, process_schedules_command, process_screener_command, process_feedback_command, process_feature_command, process_request_approval_command, process_unknown_command
)
from src.telegram.screener import business_logic

# Service layer imports
from src.data.db.services import telegram_service
from src.indicators.service import IndicatorService

# Configure logging
from src.notification.logger import setup_logger, set_logging_context
_logger = setup_logger("telegram_screener_bot")

# Global variables
notification_client = None

# HTTP API support
from aiohttp import web
import json
from typing import Dict, Any, Optional

# Service initialization and health check functions
async def initialize_services() -> bool:
    """
    Initialize telegram_service and indicator_service instances.

    Returns:
        bool: True if all services initialized successfully, False otherwise
    """
    global telegram_service_instance, indicator_service_instance

    try:
        _logger.info("Initializing service layer...")

        # Initialize telegram service (it's a module with functions, not a class)
        try:
            telegram_service_instance = telegram_service

            # Validate telegram service has required methods
            required_methods = ['get_user_status', 'set_user_limit', 'add_alert', 'list_alerts']
            for method in required_methods:
                if not hasattr(telegram_service_instance, method):
                    raise RuntimeError(f"Telegram service missing required method: {method}")

            _logger.info("Telegram service initialized and validated successfully")
        except Exception as e:
            _logger.error("Failed to initialize telegram service: %s", e)
            return False

        # Initialize indicator service with default configuration and enhanced error handling
        try:
            indicator_service_instance = IndicatorService()

            # Validate indicator service initialization
            if not hasattr(indicator_service_instance, 'compute_for_ticker'):
                raise RuntimeError("IndicatorService missing required method: compute_for_ticker")

            # Test that adapters are available
            if hasattr(indicator_service_instance, 'adapters') and not indicator_service_instance.adapters:
                _logger.warning("IndicatorService has no adapters available - some functionality may be limited")

            _logger.info("Indicator service initialized and validated successfully")
        except Exception as e:
            _logger.error("Failed to initialize indicator service: %s", e)
            # For now, continue without indicator service as some commands don't require it
            indicator_service_instance = None
            _logger.warning("Continuing without IndicatorService - indicator-based commands will be limited")

        # Set service instances in business logic layer with enhanced error handling
        try:
            business_logic.set_service_instances(telegram_service_instance, indicator_service_instance)
            _logger.info("Service instances set in business logic layer successfully")
        except Exception as e:
            _logger.error("Failed to set service instances in business logic layer: %s", e)
            return False

        # Perform health checks with enhanced error reporting
        try:
            if await perform_service_health_checks():
                _logger.info("All services initialized and health checks passed")
                return True
            else:
                _logger.error("Service health checks failed - some functionality may be limited")
                # Return True anyway if telegram service is working, as basic functionality can still work
                if telegram_service_instance:
                    _logger.info("Continuing with limited functionality - telegram service is available")
                    return True
                else:
                    _logger.error("Critical services failed - cannot start bot")
                    return False
        except Exception as health_error:
            _logger.error("Error during health checks: %s", health_error)
            # If health checks fail but services are initialized, continue with limited functionality
            if telegram_service_instance:
                _logger.warning("Health checks failed but telegram service available - continuing with limited functionality")
                return True
            else:
                return False

    except Exception as e:
        _logger.exception("Unexpected error during service initialization: %s", e)
        return False

async def perform_service_health_checks() -> bool:
    """
    Perform health checks on all initialized services.

    Returns:
        bool: True if all health checks pass, False otherwise
    """
    try:
        _logger.info("Performing service health checks...")

        # Health check for telegram service
        if not await check_telegram_service_health():
            _logger.error("Telegram service health check failed")
            return False

        # Health check for indicator service
        if not await check_indicator_service_health():
            _logger.error("Indicator service health check failed")
            return False

        _logger.info("All service health checks passed")
        return True

    except Exception as e:
        _logger.exception("Error during service health checks: %s", e)
        return False

async def check_telegram_service_health() -> bool:
    """
    Check telegram service health by testing basic operations.

    Returns:
        bool: True if service is healthy, False otherwise
    """
    try:
        # Test basic service functionality
        # Try to get a setting (this tests database connectivity)
        test_setting = telegram_service_instance.get_setting("health_check_test")
        _logger.debug("Telegram service health check: setting retrieval successful")

        # Test user operations (this tests core functionality)
        # This should not fail even if user doesn't exist
        test_user_status = telegram_service_instance.get_user_status("health_check_test_user")
        _logger.debug("Telegram service health check: user status check successful")

        return True

    except Exception as e:
        _logger.error("Telegram service health check failed: %s", e)
        return False

async def check_indicator_service_health() -> bool:
    """
    Check indicator service health by testing basic operations.

    Returns:
        bool: True if service is healthy, False otherwise
    """
    try:
        # Test that the service can be instantiated and has required adapters
        if not hasattr(indicator_service_instance, 'adapters'):
            _logger.error("Indicator service missing adapters attribute")
            return False

        # Check that required adapters are available
        required_adapters = ["ta-lib", "pandas-ta", "fundamentals"]
        for adapter_name in required_adapters:
            if adapter_name not in indicator_service_instance.adapters:
                _logger.error("Indicator service missing required adapter: %s", adapter_name)
                return False

        _logger.debug("Indicator service health check: all required adapters available")

        # Test basic functionality - check if service can handle indicator metadata
        from src.indicators.registry import INDICATOR_META
        if not INDICATOR_META:
            _logger.error("Indicator service health check: no indicator metadata available")
            return False

        _logger.debug("Indicator service health check: indicator metadata available")
        return True

    except Exception as e:
        _logger.error("Indicator service health check failed: %s", e)
        return False

def get_service_instances() -> tuple:
    """
    Get the initialized service instances.

    Returns:
        tuple: (telegram_service_instance, indicator_service_instance)
    """
    return telegram_service_instance, indicator_service_instance


# Initialize bot and dispatcher
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

# Global service instances
telegram_service_instance = None
indicator_service_instance = None

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

        # Use notification service client to send message
        success = await notification_client.send_notification(
            notification_type=MessageType.INFO,
            title=title,
            message=message,
            priority=MessagePriority.NORMAL,
            channels=["telegram"],
            recipient_id=str(user_id)
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

        # Get all registered users using service layer
        telegram_svc, _ = get_service_instances()
        if not telegram_svc:
            return web.json_response({
                'success': False,
                'error': 'Service not available'
            }, status=503)

        users = telegram_svc.list_users()
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
                success = await notification_client.send_notification(
                    notification_type=MessageType.INFO,
                    title=title,
                    message=message,
                    priority=MessagePriority.NORMAL,
                    channels=["telegram"],
                    recipient_id=str(user_id)
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

async def api_notify(request: web.Request) -> web.Response:
    """API endpoint for sending notifications from scheduler service"""
    try:
        data = await request.json()

        # Extract required fields
        notification_type = data.get('notification_type', 'INFO')
        title = data.get('title', 'Alert Notification')
        message = data.get('message')
        priority = data.get('priority', 'NORMAL')
        telegram_chat_id = data.get('telegram_chat_id')

        if not message:
            return web.json_response({
                'success': False,
                'error': 'Missing message field'
            }, status=400)

        if not telegram_chat_id:
            return web.json_response({
                'success': False,
                'error': 'Missing telegram_chat_id field'
            }, status=400)

        # Use notification service client to send notification
        success = await notification_client.send_notification(
            notification_type=notification_type,
            title=title,
            message=message,
            priority=priority,
            channels=["telegram"],
            recipient_id=str(telegram_chat_id),
            data=data.get('data', {})
        )

        return web.json_response({
            'success': success,
            'message': 'Notification queued for delivery' if success else 'Failed to queue notification'
        })

    except Exception as e:
        _logger.exception("Error in api_notify: ")
        return web.json_response({
            'success': False,
            'error': str(e)
        }, status=500)

async def api_status(request: web.Request) -> web.Response:
    """API endpoint for health check and status"""
    try:
        # Get notification client stats
        stats = notification_client.get_stats() if notification_client else {}

        # Get user count using service layer
        telegram_svc, _ = get_service_instances()
        if telegram_svc:
            users = telegram_svc.list_users()
            user_count = len(users)
        else:
            user_count = 0

        # Check service health
        service_health = await perform_service_health_checks()

        # Get service instances status
        telegram_svc, indicator_svc = get_service_instances()

        service_status = {
            'telegram_service': {
                'initialized': telegram_svc is not None,
                'healthy': service_health
            },
            'indicator_service': {
                'initialized': indicator_svc is not None,
                'healthy': service_health,
                'adapters': list(indicator_svc.adapters.keys()) if indicator_svc else []
            }
        }

        overall_status = 'healthy' if service_health else 'degraded'

        return web.json_response({
            'success': True,
            'status': overall_status,
            'services': service_status,
            'notification_stats': stats,
            'user_count': user_count,
            'queue_size': 0  # Queue is managed by notification service
        })

    except Exception as e:
        _logger.exception("Error in api_status: ")
        return web.json_response({
            'success': False,
            'status': 'error',
            'error': str(e)
        }, status=500)

# Create HTTP app
api_app = web.Application()
api_app.router.add_post('/api/send_message', api_send_message)
api_app.router.add_post('/api/broadcast', api_broadcast)
api_app.router.add_post('/api/notify', api_notify)
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
    """
    Wrapper function to audit all commands with service layer error handling.

    This wrapper implements the dependency injection pattern by:
    1. Ensuring service instances are available before processing commands
    2. Auditing all commands using the telegram_service layer
    3. Handling service layer errors gracefully
    4. Providing consistent service access to command handlers through business logic

    Command handlers access services through:
    - business_logic.handle_command() which uses global service instances
    - business_logic.get_service_instances() for direct service access

    Service instances are set during bot initialization via:
    - business_logic.set_service_instances(telegram_service, indicator_service)
    """
    start_time = time.time()
    telegram_user_id = str(message.from_user.id)
    command = message.text.split()[0] if message.text else ""
    full_message = message.text

    # Check service health before processing command with enhanced error handling
    try:
        telegram_svc, indicator_svc = get_service_instances()
        if not telegram_svc:
            _logger.error("Telegram service not initialized for command %s from user %s", command, telegram_user_id)
            try:
                await message.answer("Service temporarily unavailable. Please try again later.")
            except Exception as msg_error:
                _logger.error("Failed to send error message to user %s: %s", telegram_user_id, msg_error)
            return
    except Exception as service_error:
        _logger.error("Error getting service instances for command %s from user %s: %s",
                     command, telegram_user_id, service_error)
        try:
            await message.answer("Service temporarily unavailable. Please try again later.")
        except Exception as msg_error:
            _logger.error("Failed to send error message to user %s: %s", telegram_user_id, msg_error)
        return

    try:
        # Check if user is registered using service layer with enhanced error handling
        try:
            user_status = telegram_svc.get_user_status(telegram_user_id)
            is_registered_user = user_status is not None
            user_email = user_status.get('email') if user_status else None
        except Exception as user_status_error:
            _logger.warning("Failed to get user status for %s during command %s: %s",
                          telegram_user_id, command, user_status_error)
            # Continue with unknown user status
            user_status = None
            is_registered_user = False
            user_email = None

        # Execute the command - command handlers will use service instances through business logic
        result = await command_func(message, *args, **kwargs)

        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)

        # Log successful command using service layer with enhanced context
        try:
            _logger.info("Command executed successfully: user=%s, command=%s, response_time=%dms, registered=%s",
                        telegram_user_id, command, response_time_ms, is_registered_user)

            telegram_svc.log_command_audit(
                telegram_user_id=telegram_user_id,
                command=command,
                full_message=full_message,
                is_registered_user=is_registered_user,
                user_email=user_email,
                success=True,
                response_time_ms=response_time_ms
            )

            _logger.debug("Command audit logged successfully for user %s, command %s", telegram_user_id, command)

        except Exception as audit_error:
            _logger.warning("Failed to log successful command audit for user %s, command %s: %s",
                          telegram_user_id, command, audit_error)

        return result

    except Exception as e:
        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)

        # Try to get user status for error logging with enhanced error handling
        try:
            user_status = telegram_svc.get_user_status(telegram_user_id)
            is_registered_user = user_status is not None
            user_email = user_status.get('email') if user_status else None
        except Exception as user_error:
            _logger.warning("Failed to get user status for error logging (user %s, command %s): %s",
                          telegram_user_id, command, user_error)
            is_registered_user = False
            user_email = None

        # Log failed command using service layer with enhanced context
        try:
            _logger.error("Command failed: user=%s, command=%s, response_time=%dms, registered=%s, error=%s",
                         telegram_user_id, command, response_time_ms, is_registered_user, str(e))

            telegram_svc.log_command_audit(
                telegram_user_id=telegram_user_id,
                command=command,
                full_message=full_message,
                is_registered_user=is_registered_user,
                user_email=user_email,
                success=False,
                error_message=str(e),
                response_time_ms=response_time_ms
            )

            _logger.debug("Command failure audit logged successfully for user %s, command %s", telegram_user_id, command)

        except Exception as audit_error:
            _logger.warning("Failed to log failed command audit for user %s, command %s: %s",
                          telegram_user_id, command, audit_error)

        # Re-raise the exception
        raise

@dp.message(Command("start"))
async def cmd_start(message: Message):
    _logger.info("Received /start command from user %s", message.from_user.id)
    try:
        await audit_command_wrapper(message, process_help_command, str(message.from_user.id), message.text, notification_client)
        _logger.info("Successfully processed /start command for user %s", message.from_user.id)
    except Exception as e:
        _logger.exception("Error processing /start command for user %s", message.from_user.id)
        # Send a simple error message directly
        await message.answer("Sorry, there was an error processing your command. Please try again.")

@dp.message(Command("help"))
async def cmd_help(message: Message):
    try:
        await audit_command_wrapper(message, process_help_command, str(message.from_user.id), message.text, notification_client)
    except Exception as e:
        _logger.exception("Error processing /help command for user %s", message.from_user.id)
        # Fallback: send built-in help text directly
        try:
            await message.answer(HELP_TEXT)
        except Exception:
            await message.answer("Sorry, there was an error processing your command. Please try again.")

@dp.message(Command("info"))
async def cmd_info(message: Message):
    await audit_command_wrapper(message, process_info_command, str(message.from_user.id), notification_client)

@dp.message(Command("register"))
async def cmd_register(message: Message):
    args = message.text.split()
    await audit_command_wrapper(message, process_register_command, str(message.from_user.id), args, notification_client)

@dp.message(Command("verify"))
async def cmd_verify(message: Message):
    args = message.text.split()
    await audit_command_wrapper(message, process_verify_command, str(message.from_user.id), args, notification_client)

@dp.message(Command("request_approval"))
async def cmd_request_approval(message: Message):
    args = message.text.split()
    await audit_command_wrapper(message, process_request_approval_command, str(message.from_user.id), args, notification_client)

@dp.message(Command("language"))
async def cmd_language(message: Message):
    args = message.text.split()
    await audit_command_wrapper(message, process_language_command, str(message.from_user.id), args, notification_client)

@dp.message(Command("admin"))
async def cmd_admin(message: Message):
    args = message.text.split()
    await audit_command_wrapper(message, process_admin_command, str(message.from_user.id), args, notification_client)

@dp.message(Command("report"))
async def cmd_report(message: Message):
    args = message.text.split()
    await audit_command_wrapper(message, process_report_command, str(message.from_user.id), args, notification_client)

@dp.message(Command("alerts"))
async def cmd_alerts(message: Message):
    args = message.text.split()
    await audit_command_wrapper(message, process_alerts_command, str(message.from_user.id), args, notification_client)

@dp.message(Command("schedules"))
async def cmd_schedules(message: Message):
    args = message.text.split()
    await audit_command_wrapper(message, process_schedules_command, str(message.from_user.id), args, notification_client)

@dp.message(Command("screener"))
async def cmd_screener(message: Message):
    args = message.text.split()
    await audit_command_wrapper(message, process_screener_command, str(message.from_user.id), args, notification_client)

@dp.message(Command("feedback"))
async def cmd_feedback(message: Message):
    args = message.text.split(maxsplit=1)
    await audit_command_wrapper(message, process_feedback_command, str(message.from_user.id), args, notification_client)

@dp.message(Command("feature"))
async def cmd_feature(message: Message):
    args = message.text.split(maxsplit=1)
    await audit_command_wrapper(message, process_feature_command, str(message.from_user.id), args, notification_client)

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
        await audit_command_wrapper(message, process_unknown_command, str(message.from_user.id), notification_client, HELP_TEXT)
    except Exception as e:
        _logger.exception("Error processing unknown command for user %s", message.from_user.id)
        await message.answer("Sorry, there was an error processing your command. Please try again.")

@dp.message()
async def all_messages(message: Message):
    """Catch all messages for debugging"""
    _logger.info("Received message: %s from user %s", message.text, message.from_user.id)

async def main():
    global notification_client

    _logger.info("Starting bot initialization...")
    if TELEGRAM_BOT_TOKEN:
        _logger.info("Bot token: %s...", TELEGRAM_BOT_TOKEN[:10])
    else:
        _logger.error("Bot token is None!")
        return

    # Set logging context so that notification service client logs go to telegram bot log file
    set_logging_context("telegram_screener_bot")

    # Initialize service layer first
    _logger.info("Initializing service layer...")
    if not await initialize_services():
        _logger.error("Failed to initialize services. Bot cannot start.")
        return

    # Initialize notification service client
    _logger.info("Initializing notification service client...")
    try:
        import os
        notification_service_url = os.getenv("NOTIFICATION_SERVICE_URL", "http://localhost:8000")
        notification_client = NotificationServiceClient(
            service_url=notification_service_url,
            timeout=30,
            max_retries=3
        )

        # Test connectivity
        health = await notification_client.get_health_status()
        _logger.info("Notification service health: %s", health.get('status', 'unknown'))

    except Exception as e:
        _logger.exception("Failed to initialize notification service client: %s", e)
        notification_client = None

    _logger.info("Starting Telegram Screener Bot with HTTP API...")

    # Start both bot polling and HTTP API server
    try:
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

    except Exception as e:
        _logger.exception("Failed to start bot or HTTP API server: %s", e)
        return

if __name__ == "__main__":
    asyncio.run(main())
