import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

import sqlite3
import time
from typing import Any, Dict, List
from src.frontend.telegram.command_parser import ParsedCommand
from src.common import get_ohlcv
from src.common.fundamentals import get_fundamentals
from src.common.technicals import calculate_technicals_from_df
from src.model.telegram_bot import TickerAnalysis
from src.frontend.telegram import db
from src.common.ticker_analyzer import format_ticker_report

from src.notification.logger import setup_logger
_logger = setup_logger(__name__)

def handle_command(parsed: ParsedCommand) -> Dict[str, Any]:
    """
    Main business logic handler. Dispatches based on command and parameters.

    Args:
        parsed: ParsedCommand object containing command and arguments

    Returns:
        Dict with result/status/data for notification manager
    """
    if parsed.command == "report":
        return handle_report(parsed)
    elif parsed.command == "help":
        return handle_help(parsed)
    elif parsed.command == "info":
        return handle_info(parsed)
    elif parsed.command == "register":
        return handle_register(parsed)
    elif parsed.command == "verify":
        return handle_verify(parsed)
    elif parsed.command == "request_approval":
        return handle_request_approval(parsed)
    elif parsed.command == "language":
        return handle_language(parsed)
    elif parsed.command == "admin":
        return handle_admin(parsed)
    elif parsed.command == "alerts":
        return handle_alerts(parsed)
    elif parsed.command == "schedules":
        return handle_schedules(parsed)
    elif parsed.command == "screener":
        return handle_screener(parsed)
    elif parsed.command == "feedback":
        return handle_feedback(parsed)
    elif parsed.command == "feature":
        return handle_feature(parsed)
    # Add more command handlers as needed
    return {"status": "error", "message": f"Unknown command: {parsed.command}"}


def handle_help(parsed: ParsedCommand) -> Dict[str, Any]:
    """
    Business logic for /help and /start commands.
    Returns appropriate help text based on user admin status.
    """
    try:
        telegram_user_id = parsed.args.get("telegram_user_id")
        if not telegram_user_id:
            return {"status": "error", "message": "No telegram_user_id provided"}

        db.init_db()
        is_admin = is_admin_user(telegram_user_id)

        # Import help texts here to avoid circular imports
        from src.frontend.telegram.bot import HELP_TEXT, ADMIN_HELP_TEXT

        # Show regular help text
        help_text = HELP_TEXT

        # Add admin commands if user is admin
        if is_admin:
            help_text += "\n\n" + ADMIN_HELP_TEXT

        return {
            "status": "ok",
            "help_text": help_text,
            "is_admin": is_admin
        }
    except Exception as e:
        _logger.exception("Error generating help")
        return {"status": "error", "message": f"Error generating help: {str(e)}"}


def is_admin_user(telegram_user_id: str) -> bool:
    """Check if user is an admin."""
    db.init_db()
    status = db.get_user_status(telegram_user_id)
    return status and status.get("is_admin", False)

def is_approved_user(telegram_user_id: str) -> bool:
    """Check if user is approved for restricted features."""
    db.init_db()
    status = db.get_user_status(telegram_user_id)
    return status and status.get("approved", False)

def check_admin_access(telegram_user_id: str) -> Dict[str, Any]:
    """Check if user has admin access. Returns error dict if not."""
    if not is_admin_user(telegram_user_id):
        return {"status": "error", "message": "Access denied. Admin privileges required."}
    return {"status": "ok"}

def check_approved_access(telegram_user_id: str) -> Dict[str, Any]:
    """Check if user has approved access for restricted features. Returns error dict if not."""
    if not is_approved_user(telegram_user_id):
        return {"status": "error", "message": "Access denied, please contact chat's admin or send request for approval using command /request_approval"}
    return {"status": "ok"}


def handle_request_approval(parsed: ParsedCommand) -> Dict[str, Any]:
    """
    Business logic for /request_approval command.
    User requests admin approval after email verification.
    """
    try:
        telegram_user_id = parsed.args.get("telegram_user_id")
        if not telegram_user_id:
            return {"status": "error", "message": "No telegram_user_id provided"}

        db.init_db()
        status = db.get_user_status(telegram_user_id)

        if not status:
            return {"status": "error", "message": "Please register first using /register email@example.com"}

        if not status.get("verified", False):
            return {"status": "error", "message": "Please verify your email first using /verify CODE"}

        if status.get("approved", False):
            return {"status": "error", "message": "You are already approved for restricted features"}

        # Check if user already has a pending request (optional - could add a separate table for requests)
        # For now, we'll just notify admins about the request

        return {
            "status": "ok",
            "message": "Your approval request has been submitted. Admins will review your request and notify you of the decision.",
            "user_id": telegram_user_id,
            "email": status.get("email"),
            "notify_admins": True
        }
    except Exception as e:
        _logger.exception("Error processing approval request")
        return {"status": "error", "message": f"Error processing approval request: {str(e)}"}

def handle_report(parsed: ParsedCommand) -> Dict[str, Any]:
    """
    Business logic for /report command.
    For each ticker:
      - Use analyze_ticker_business for unified analysis logic
      - Use format_ticker_report to generate message and chart
    """
    args = parsed.args

    # Check if user has approved access
    telegram_user_id = args.get("telegram_user_id")
    access_check = check_approved_access(telegram_user_id)
    if access_check["status"] != "ok":
        return access_check

    # Check if JSON configuration is provided
    config_json = args.get("config")
    if config_json:
        # Validate and parse JSON configuration
        try:
            from src.frontend.telegram.screener.report_config_parser import ReportConfigParser
            is_valid, errors = ReportConfigParser.validate_report_config(config_json)
            if not is_valid:
                return {"status": "error", "title": "Report Error", "message": f"Invalid report configuration: {'; '.join(errors)}"}

            report_config = ReportConfigParser.parse_report_config(config_json)
            if not report_config:
                return {"status": "error", "title": "Report Error", "message": "Failed to parse report configuration"}

            # Use configuration from JSON
            tickers = [t.upper() for t in report_config.tickers]
            period = report_config.period
            interval = report_config.interval
            provider = report_config.provider
            indicators = ",".join(report_config.indicators) if report_config.indicators else None
            email = report_config.email

        except Exception as e:
            _logger.exception("Error processing JSON configuration: %s", e)
            return {"status": "error", "title": "Report Error", "message": f"Error processing JSON configuration: {str(e)}"}
    else:
        # Use traditional command-line parameters
        tickers_raw = args.get("tickers")
        if isinstance(tickers_raw, str):
            tickers = [tickers_raw.upper()]
        elif isinstance(tickers_raw, list):
            tickers = [t.upper() for t in tickers_raw]
        else:
            tickers = [t.upper() for t in parsed.positionals]

        if not tickers:
            return {"status": "error", "title": "Report Error", "message": "No tickers specified"}

        period = args.get("period") or "2y"
        interval = args.get("interval") or "1d"
        provider = args.get("provider")
        indicators = args.get("indicators")
        email = args.get("email", False)

    reports = []

    # Fetch the registered email for the current user
    telegram_user_id = args.get("telegram_user_id")
    user_email = None
    if telegram_user_id:
        db.init_db()
        status = db.get_user_status(telegram_user_id)
        if status and status.get("email"):
            user_email = status["email"]

    all_failed = True
    for ticker in tickers:
        analysis = analyze_ticker_business(
            ticker=ticker,
            provider=provider,
            period=period,
            interval=interval
        )
        report = format_ticker_report(analysis)
        report['ticker'] = ticker
        report['error'] = analysis.error if analysis.error else None
        reports.append(report)
        if not analysis.error:
            all_failed = False
    # If all analyses failed due to missing keys
    if all_failed and any(report['error'] and any(key in report['error'] for key in ["Alpha Vantage API key", "Finnhub API key", "Twelve Data API key", "Polygon.io API key"]) for report in reports):
        return {
            "status": "error",
            "title": "Report Error",
            "message": f"No data could be retrieved for {', '.join(tickers)}. Missing or invalid API keys for 1 or more providers. Please check your API keys in donotshare.py."
        }
    # If all analyses failed for any reason
    if all_failed:
        return {
            "status": "error",
            "title": "Report Error",
            "message": f"No data could be retrieved for {', '.join(tickers)}. Please check your API keys or try a different provider/ticker."
        }
    # Otherwise, return reports for Telegram/email delivery
    return {
        "status": "ok",
        "reports": reports,
        "email": email,
        "user_email": user_email,
        "title": f"Report for {', '.join(tickers)}",
        "message": "Report generated successfully."
    }


def analyze_ticker_business(
    ticker: str,
    provider: str = None,
    period: str = "2y",
    interval: str = "1d"
) -> TickerAnalysis:
    """
    Business logic: fetch OHLCV for ticker/provider/period/interval, return TickerAnalysis.
    Uses common functions from src/common for data retrieval and analysis.
    """
    try:
        # Get OHLCV data using common function
        df = get_ohlcv(ticker, interval, period, provider)

        # Get fundamentals using common function
        fundamentals = get_fundamentals(ticker, provider)

        # Calculate technical indicators
        df_with_technicals, technicals = calculate_technicals_from_df(df)

        # Calculate current price and change percentage
        current_price = None
        change_percentage = None
        if df is not None and not df.empty:
            current_price = df['close'].iloc[-1]
            if len(df) > 1:
                prev_price = df['close'].iloc[-2]
                change_percentage = ((current_price - prev_price) / prev_price) * 100

        return TickerAnalysis(
            ticker=ticker.upper(),
            provider=provider or ("yf" if len(ticker) <= 5 else "bnc"),
            period=period,
            interval=interval,
            ohlcv=df_with_technicals,
            fundamentals=fundamentals,
            technicals=technicals,
            current_price=current_price,
            change_percentage=change_percentage,
            error=None,
            chart_image=None
        )
    except Exception as e:
        return TickerAnalysis(
            ticker=ticker.upper(),
            provider=provider or ("yf" if len(ticker) <= 5 else "bnc"),
            period=period,
            interval=interval,
            ohlcv=None,
            fundamentals=None,
            technicals=None,
            current_price=None,
            change_percentage=None,
            error=str(e),
            chart_image=None
        )


def handle_info(parsed: ParsedCommand) -> Dict[str, Any]:
    telegram_user_id = parsed.args.get("telegram_user_id")
    if not telegram_user_id:
        return {"status": "error", "message": "No telegram_user_id provided"}
    db.init_db()
    status = db.get_user_status(telegram_user_id)
    if status:
        email = status["email"] or "(not set)"
        verified = "Yes" if status["verified"] else "No"
        approved = "Yes" if status["approved"] else "No"
        admin = "Yes" if status["is_admin"] else "No"
        language = status["language"] or "(not set)"
        return {
            "status": "ok",
            "title": "Your Info",
            "message": f"Email: {email}\nVerified: {verified}\nApproved: {approved}\nAdmin: {admin}\nLanguage: {language}"
        }
    else:
        return {
            "status": "ok",
            "title": "Your Info",
            "message": "Email: (not set)\nVerified: No\nApproved: No\nAdmin: No\nLanguage: (not set)"
        }


def handle_admin(parsed: ParsedCommand) -> Dict[str, Any]:
    """
    Business logic for /admin commands.
    Handles user management, system settings, and administrative functions.
    """
    try:
        telegram_user_id = parsed.args.get("telegram_user_id")
        if not telegram_user_id:
            return {"status": "error", "message": "No telegram_user_id provided"}

        db.init_db()

        # Check if user has admin access
        access_check = check_admin_access(telegram_user_id)
        if access_check["status"] != "ok":
            return access_check

        # Get action and parameters from positionals
        action = parsed.positionals[0] if len(parsed.positionals) > 0 else None
        params = parsed.positionals[1:] if len(parsed.positionals) > 1 else []

        if not action:
            return {
                "status": "error",
                "title": "Admin Help",
                "message": ("Available admin commands:\n"
                           "/admin users - List all registered users\n"
                           "/admin listusers - List users as telegram_user_id - email pairs\n"
                           "/admin pending - List users waiting for approval\n"
                           "/admin approve USER_ID - Approve user for restricted features\n"
                           "/admin reject USER_ID - Reject user's approval request\n"
                           "/admin verify USER_ID - Manually verify user's email\n"
                           "/admin resetemail USER_ID - Reset user's email\n"
                           "/admin setlimit alerts N [USER_ID] - Set max alerts (global or per-user)\n"
                           "/admin setlimit schedules N [USER_ID] - Set max schedules (global or per-user)\n"
                           "/admin broadcast MESSAGE - Send broadcast message to all users")
            }

        if action == "users":
            return handle_admin_list_users(parsed)
        elif action == "listusers":
            return handle_admin_list_users(parsed)
        elif action == "resetemail" and len(params) >= 1:
            # Create a new parsed command with the user_id parameter
            new_parsed = ParsedCommand(
                command="admin",
                args={"telegram_user_id": telegram_user_id, "user_id": params[0]},
                positionals=[]
            )
            return handle_admin_reset_email(new_parsed)
        elif action == "verify" and len(params) >= 1:
            new_parsed = ParsedCommand(
                command="admin",
                args={"telegram_user_id": telegram_user_id, "user_id": params[0]},
                positionals=[]
            )
            return handle_admin_verify_user(new_parsed)
        elif action == "approve" and len(params) >= 1:
            new_parsed = ParsedCommand(
                command="admin",
                args={"telegram_user_id": telegram_user_id, "user_id": params[0]},
                positionals=[]
            )
            return handle_admin_approve_user(new_parsed)
        elif action == "reject" and len(params) >= 1:
            new_parsed = ParsedCommand(
                command="admin",
                args={"telegram_user_id": telegram_user_id, "user_id": params[0]},
                positionals=[]
            )
            return handle_admin_reject_user(new_parsed)
        elif action == "pending":
            return handle_admin_list_pending_approvals(parsed)
        elif action == "setlimit" and len(params) >= 2:
            new_parsed = ParsedCommand(
                command="admin",
                args={"telegram_user_id": telegram_user_id, "user_id": params[2] if len(params) > 2 else None, "limit": params[1]},
                positionals=[]
            )
            return handle_admin_set_limit(new_parsed)
        elif action == "broadcast" and len(params) >= 1:
            new_parsed = ParsedCommand(
                command="admin",
                args={"telegram_user_id": telegram_user_id, "message": " ".join(params), "scheduled_time": "now"},
                positionals=[]
            )
            return handle_admin_schedule_broadcast(new_parsed)
        else:
            return {"status": "error", "message": f"Unknown admin command: {action}"}

    except Exception as e:
        _logger.exception("Error in admin command")
        return {"status": "error", "message": f"Error in admin command: {str(e)}"}

def handle_admin_list_users(parsed: ParsedCommand) -> Dict[str, Any]:
    """List all users for admin review."""
    try:
        # Check admin access
        telegram_user_id = parsed.args.get("telegram_user_id")
        access_check = check_admin_access(telegram_user_id)
        if access_check["status"] != "ok":
            return access_check

        db.init_db()
        users = db.get_all_users()

        if not users:
            return {"status": "ok", "message": "No users found"}

        # Format user list
        user_list = []
        for user in users:
            status_text = "✅ Verified & Approved" if user.get("verified") and user.get("approved") else \
                         "✅ Verified" if user.get("verified") else "❌ Not Verified"
            user_list.append(f"• {user.get('email', 'N/A')} - {status_text}")

        return {
            "status": "ok",
            "message": f"**User List**\n\n" + "\n".join(user_list),
            "is_admin": True
        }

    except Exception as e:
        _logger.exception("Error listing users")
        return {"status": "error", "message": f"Error listing users: {str(e)}"}

def handle_admin_list_pending_approvals(parsed: ParsedCommand) -> Dict[str, Any]:
    """List users pending approval."""
    try:
        # Check admin access
        telegram_user_id = parsed.args.get("telegram_user_id")
        access_check = check_admin_access(telegram_user_id)
        if access_check["status"] != "ok":
            return access_check

        db.init_db()
        users = db.get_all_users()

        # Filter for verified but not approved users
        pending_users = [user for user in users if user.get("verified") and not user.get("approved")]

        if not pending_users:
            return {"status": "ok", "message": "No users pending approval"}

        # Format pending user list
        user_list = []
        for user in pending_users:
            user_list.append(f"• {user.get('email', 'N/A')} (ID: {user.get('telegram_user_id')})")

        return {
            "status": "ok",
            "message": f"**Users Pending Approval**\n\n" + "\n".join(user_list),
            "is_admin": True
        }

    except Exception as e:
        _logger.exception("Error listing users")
        return {"status": "error", "message": f"Error listing users: {str(e)}"}

def handle_admin_reset_email(parsed: ParsedCommand) -> Dict[str, Any]:
    """Reset user's email verification status."""
    try:
        # Check admin access
        telegram_user_id = parsed.args.get("telegram_user_id")
        access_check = check_admin_access(telegram_user_id)
        if access_check["status"] != "ok":
            return access_check

        user_id = parsed.args.get("user_id")
        if not user_id:
            return {"status": "error", "message": "No user_id provided"}

        db.init_db()
        success = db.reset_user_email_verification(user_id)

        if success:
            return {
                "status": "ok",
                "message": f"Email verification reset for user {user_id}",
                "is_admin": True
            }
        else:
            return {"status": "error", "message": f"Failed to reset email for user {user_id}"}

    except Exception as e:
        _logger.exception("Error resetting email")
        return {"status": "error", "message": f"Error resetting email: {str(e)}"}

def handle_admin_verify_user(parsed: ParsedCommand) -> Dict[str, Any]:
    """Verify a user's email."""
    try:
        # Check admin access
        telegram_user_id = parsed.args.get("telegram_user_id")
        access_check = check_admin_access(telegram_user_id)
        if access_check["status"] != "ok":
            return access_check

        user_id = parsed.args.get("user_id")
        if not user_id:
            return {"status": "error", "message": "No user_id provided"}

        db.init_db()
        success = db.verify_user_email(user_id)

        if success:
            return {
                "status": "ok",
                "message": f"User {user_id} verified successfully",
                "is_admin": True
            }
        else:
            return {"status": "error", "message": f"Failed to verify user {user_id}"}

    except Exception as e:
        _logger.exception("Error verifying user")
        return {"status": "error", "message": f"Error verifying user: {str(e)}"}

def handle_admin_set_limit(parsed: ParsedCommand) -> Dict[str, Any]:
    """Set user's daily request limit."""
    try:
        # Check admin access
        telegram_user_id = parsed.args.get("telegram_user_id")
        access_check = check_admin_access(telegram_user_id)
        if access_check["status"] != "ok":
            return access_check

        user_id = parsed.args.get("user_id")
        limit = parsed.args.get("limit")

        if not user_id or not limit:
            return {"status": "error", "message": "Both user_id and limit are required"}

        try:
            limit = int(limit)
        except ValueError:
            return {"status": "error", "message": "Limit must be a number"}

        db.init_db()
        success = db.set_user_daily_limit(user_id, limit)

        if success:
            return {
                "status": "ok",
                "message": f"Daily limit set to {limit} for user {user_id}",
                "is_admin": True
            }
        else:
            return {"status": "error", "message": f"Failed to set limit for user {user_id}"}

    except Exception as e:
        _logger.exception("Error setting limit")
        return {"status": "error", "message": f"Error setting limit: {str(e)}"}

def handle_admin_schedule_broadcast(parsed: ParsedCommand) -> Dict[str, Any]:
    """Schedule a broadcast message."""
    try:
        # Check admin access
        telegram_user_id = parsed.args.get("telegram_user_id")
        access_check = check_admin_access(telegram_user_id)
        if access_check["status"] != "ok":
            return access_check

        message = parsed.args.get("message")
        scheduled_time = parsed.args.get("scheduled_time")

        if not message or not scheduled_time:
            return {"status": "error", "message": "Both message and scheduled_time are required"}

        db.init_db()
        success = db.schedule_broadcast(message, scheduled_time, telegram_user_id)

        if success:
            return {
                "status": "ok",
                "message": f"Broadcast scheduled for {scheduled_time}",
                "is_admin": True
            }
        else:
            return {"status": "error", "message": "Failed to schedule broadcast"}

    except Exception as e:
        _logger.exception("Error scheduling broadcast")
        return {"status": "error", "message": f"Error scheduling broadcast: {str(e)}"}

def handle_admin_approve_user(parsed: ParsedCommand) -> Dict[str, Any]:
    """Approve a user for restricted features."""
    try:
        # Check admin access
        telegram_user_id = parsed.args.get("telegram_user_id")
        access_check = check_admin_access(telegram_user_id)
        if access_check["status"] != "ok":
            return access_check

        user_id = parsed.args.get("user_id")
        if not user_id:
            return {"status": "error", "message": "No user_id provided"}

        db.init_db()
        success = db.approve_user(user_id)

        if success:
            return {
                "status": "ok",
                "message": f"User {user_id} approved successfully",
                "is_admin": True
            }
        else:
            return {"status": "error", "message": f"Failed to approve user {user_id}"}

    except Exception as e:
        _logger.exception("Error approving user")
        return {"status": "error", "message": f"Error approving user: {str(e)}"}

def handle_admin_reject_user(parsed: ParsedCommand) -> Dict[str, Any]:
    """Reject a user's approval request."""
    try:
        # Check admin access
        telegram_user_id = parsed.args.get("telegram_user_id")
        access_check = check_admin_access(telegram_user_id)
        if access_check["status"] != "ok":
            return access_check

        user_id = parsed.args.get("user_id")
        if not user_id:
            return {"status": "error", "message": "No user_id provided"}

        db.init_db()
        success = db.reject_user(user_id)

        if success:
            return {
                "status": "ok",
                "message": f"User {user_id} rejected",
                "is_admin": True
            }
        else:
            return {"status": "error", "message": f"Failed to reject user {user_id}"}

    except Exception as e:
        _logger.exception("Error rejecting user")
        return {"status": "error", "message": f"Error rejecting user: {str(e)}"}

def handle_admin_list_pending_approvals(parsed: ParsedCommand) -> Dict[str, Any]:
    """List users pending approval."""
    try:
        # Check admin access
        telegram_user_id = parsed.args.get("telegram_user_id")
        access_check = check_admin_access(telegram_user_id)
        if access_check["status"] != "ok":
            return access_check

        db.init_db()
        users = db.get_all_users()

        # Filter for verified but not approved users
        pending_users = [user for user in users if user.get("verified") and not user.get("approved")]

        if not pending_users:
            return {"status": "ok", "message": "No users pending approval"}

        # Format pending user list
        user_list = []
        for user in pending_users:
            user_list.append(f"• {user.get('email', 'N/A')} (ID: {user.get('telegram_user_id')})")

        return {
            "status": "ok",
            "message": f"**Users Pending Approval**\n\n" + "\n".join(user_list),
            "is_admin": True
        }

    except Exception as e:
        _logger.exception("Error listing pending approvals")
        return {"status": "error", "message": f"Error listing pending approvals: {str(e)}"}


def handle_alerts(parsed: ParsedCommand) -> Dict[str, Any]:
    """
    Business logic for /alerts commands.
    Handles creating, listing, editing, and deleting price alerts.
    """
    try:
        telegram_user_id = parsed.args.get("telegram_user_id")
        if not telegram_user_id:
            return {"status": "error", "message": "No telegram_user_id provided"}

        db.init_db()

        # Check if user has approved access
        access_check = check_approved_access(telegram_user_id)
        if access_check["status"] != "ok":
            return access_check

        # Get action and parameters from positionals
        action = parsed.positionals[0] if len(parsed.positionals) > 0 else None
        params = parsed.positionals[1:] if len(parsed.positionals) > 1 else []

        if not action:
            # List all alerts for user
            return handle_alerts_list(telegram_user_id)

        if action == "add" and len(params) >= 3:
            ticker, price, condition = params[0], params[1], params[2]
            # Get email flag from parsed args
            email = parsed.args.get("email", False)
            return handle_alerts_add(telegram_user_id, ticker, price, condition, email)
        elif action == "add_indicator" and len(params) >= 2:
            ticker, config_json = params[0], params[1]
            # Get additional parameters from parsed args
            email = parsed.args.get("email", False)
            timeframe = parsed.args.get("timeframe", "15m")
            alert_action = parsed.args.get("action_type", "notify")
            return handle_alerts_add_indicator(telegram_user_id, ticker, config_json, timeframe, alert_action, email)
        elif action == "edit" and len(params) >= 1:
            alert_id = params[0]
            new_price = params[1] if len(params) > 1 else None
            new_condition = params[2] if len(params) > 2 else None
            # Get email flag from parsed args
            email = parsed.args.get("email")
            return handle_alerts_edit(telegram_user_id, alert_id, new_price, new_condition, email)
        elif action == "delete" and len(params) >= 1:
            alert_id = params[0]
            return handle_alerts_delete(telegram_user_id, alert_id)
        elif action == "pause" and len(params) >= 1:
            alert_id = params[0]
            return handle_alerts_pause(telegram_user_id, alert_id)
        elif action == "resume" and len(params) >= 1:
            alert_id = params[0]
            return handle_alerts_resume(telegram_user_id, alert_id)
        else:
            return {
                "status": "error",
                "title": "Alerts Help",
                "message": ("Available alert commands:\n"
                           "/alerts - List all alerts\n"
                           "/alerts add TICKER PRICE CONDITION [flags] - Add price alert\n"
                           "  CONDITION: above or below\n"
                           "  Example: /alerts add BTCUSDT 65000 above -email\n"
                           "/alerts add_indicator TICKER CONFIG_JSON [flags] - Add indicator alert\n"
                           "  Example: /alerts add_indicator AAPL '{\"type\":\"indicator\",\"indicator\":\"RSI\",\"parameters\":{\"period\":14},\"condition\":{\"operator\":\"<\",\"value\":30},\"alert_action\":\"BUY\",\"timeframe\":\"15m\"}' -email\n"
                           "Flags:\n"
                           "  -email: Send alert notification to email\n"
                           "  -timeframe=15m: Set timeframe (5m, 15m, 1h, 4h, 1d)\n"
                           "  -action_type=notify: Set action (BUY, SELL, HOLD, notify)\n"
                           "/alerts edit ALERT_ID [PRICE] [CONDITION] [flags] - Edit alert\n"
                           "  Example: /alerts edit 1 70000 below -email\n"
                           "/alerts delete ALERT_ID - Delete alert\n"
                           "/alerts pause ALERT_ID - Pause alert\n"
                           "/alerts resume ALERT_ID - Resume alert\n\n"
                           "Indicator Alert Examples:\n"
                           "• RSI oversold: {\"type\":\"indicator\",\"indicator\":\"RSI\",\"parameters\":{\"period\":14},\"condition\":{\"operator\":\"<\",\"value\":30}}\n"
                           "• Bollinger Bands: {\"type\":\"indicator\",\"indicator\":\"BollingerBands\",\"parameters\":{\"period\":20},\"condition\":{\"operator\":\"below_lower_band\"}}\n"
                           "• MACD crossover: {\"type\":\"indicator\",\"indicator\":\"MACD\",\"condition\":{\"operator\":\"crossover\"}}")
            }

    except Exception as e:
        _logger.exception("Error in alerts command: ")
        return {"status": "error", "message": f"Error processing alerts command: {str(e)}"}


def handle_alerts_list(telegram_user_id: str) -> Dict[str, Any]:
    """List all alerts for a user."""
    try:
        alerts = db.list_alerts(telegram_user_id)
        if not alerts:
            return {"status": "ok", "title": "Your Alerts", "message": "You have no active alerts."}

        alert_list = []
        for alert in alerts:
            status = "🟢 Active" if alert.get("active") else "🔴 Paused"
            email_flag = "📧" if alert.get("email") else "💬"

            # Handle different alert types
            alert_type = alert.get("alert_type", "price")
            if alert_type == "price":
                alert_list.append(
                    f"#{alert['id']}: {alert['ticker']} {alert['condition']} ${alert['price']:.2f} {email_flag} - {status}"
                )
            else:
                # Indicator alert
                from src.frontend.telegram.screener.alert_logic_evaluator import get_alert_summary
                summary = get_alert_summary(alert)
                alert_type_icon = "📊" if alert_type == "indicator" else "❓"
                timeframe = alert.get("timeframe", "15m")
                action = alert.get("alert_action", "notify")

                if "indicators" in summary:
                    indicators_text = ", ".join(summary["indicators"])
                    alert_list.append(
                        f"#{alert['id']}: {alert['ticker']} {alert_type_icon} {indicators_text} ({timeframe}, {action}) {email_flag} - {status}"
                    )
                else:
                    alert_list.append(
                        f"#{alert['id']}: {alert['ticker']} {alert_type_icon} Indicator Alert ({timeframe}, {action}) {email_flag} - {status}"
                    )

        message = f"Your alerts ({len(alerts)}):\n\n" + "\n".join(alert_list)
        return {"status": "ok", "title": "Your Alerts", "message": message}

    except Exception as e:
        _logger.exception("Error listing alerts: ")
        return {"status": "error", "message": f"Error listing alerts: {str(e)}"}


def handle_alerts_add(telegram_user_id: str, ticker: str, price_str: str, condition: str, email: bool = False) -> Dict[str, Any]:
    """Add a new price alert."""
    try:
        # Validate condition
        if condition.lower() not in ["above", "below"]:
            return {"status": "error", "message": "Condition must be 'above' or 'below'"}

        # Validate price
        try:
            price = float(price_str)
            if price <= 0:
                raise ValueError("Price must be positive")
        except ValueError:
            return {"status": "error", "message": "Price must be a positive number"}

        # Check user limits
        user_status = db.get_user_status(telegram_user_id)
        max_alerts = user_status.get("max_alerts", 5)
        current_alerts = len(db.list_alerts(telegram_user_id))

        if current_alerts >= max_alerts:
            return {
                "status": "error",
                "message": f"Alert limit reached ({max_alerts}). Delete some alerts first or contact admin."
            }

        # Add the alert (convert ticker to uppercase)
        alert_id = db.add_alert(telegram_user_id, ticker.upper(), price, condition.lower(), email)

        email_text = " and email" if email else ""
        return {
            "status": "ok",
            "title": "Alert Added",
            "message": f"Alert #{alert_id} created: {ticker.upper()} {condition.lower()} ${price:.2f}{email_text}"
        }

    except Exception as e:
        _logger.exception("Error adding alert: ")
        return {"status": "error", "message": f"Error adding alert: {str(e)}"}


def handle_alerts_edit(telegram_user_id: str, alert_id_str: str, new_price_str: str = None, new_condition: str = None, email: bool = None) -> Dict[str, Any]:
    """Edit an existing alert."""
    try:
        # Validate alert ID
        try:
            alert_id = int(alert_id_str)
        except ValueError:
            return {"status": "error", "message": "Alert ID must be a number"}

        # Check if alert exists and belongs to user
        alert = db.get_alert(alert_id)
        if not alert or alert.get("user_id") != telegram_user_id:
            return {"status": "error", "message": f"Alert #{alert_id} not found or access denied."}

        updates = {}

        # Validate and set new price
        if new_price_str:
            try:
                new_price = float(new_price_str)
                if new_price <= 0:
                    raise ValueError("Price must be positive")
                updates["price"] = new_price
            except ValueError:
                return {"status": "error", "message": "Price must be a positive number"}

        # Validate and set new condition
        if new_condition:
            if new_condition.lower() not in ["above", "below"]:
                return {"status": "error", "message": "Condition must be 'above' or 'below'"}
            updates["condition"] = new_condition.lower()

        # Validate and set email flag
        if email is not None:
            updates["email"] = 1 if email else 0

        if not updates:
            return {"status": "error", "message": "No updates provided. Specify new price, condition, and/or email flag."}

        # Update the alert
        db.update_alert(alert_id, **updates)

        # Get updated alert for confirmation
        updated_alert = db.get_alert(alert_id)

        return {
            "status": "ok",
            "title": "Alert Updated",
            "message": f"Alert #{alert_id} updated: {updated_alert['ticker']} {updated_alert['condition']} ${updated_alert['price']:.2f}"
        }

    except Exception as e:
        _logger.exception("Error editing alert: ")
        return {"status": "error", "message": f"Error editing alert: {str(e)}"}


def handle_alerts_add_indicator(telegram_user_id: str, ticker: str, config_json: str, timeframe: str = "15m",
                               alert_action: str = "notify", email: bool = False) -> Dict[str, Any]:
    """Add a new indicator-based alert."""
    try:
        # Validate ticker
        if not ticker or len(ticker.strip()) == 0:
            return {"status": "error", "message": "Ticker is required"}

        # Validate timeframe
        valid_timeframes = ["5m", "15m", "1h", "4h", "1d"]
        if timeframe not in valid_timeframes:
            return {"status": "error", "message": f"Invalid timeframe. Must be one of: {', '.join(valid_timeframes)}"}

        # Validate alert action
        valid_actions = ["BUY", "SELL", "HOLD", "notify"]
        if alert_action not in valid_actions:
            return {"status": "error", "message": f"Invalid action. Must be one of: {', '.join(valid_actions)}"}

        # Validate JSON configuration
        try:
            from src.frontend.telegram.screener.alert_config_parser import validate_alert_config
            is_valid, errors = validate_alert_config(config_json)
            if not is_valid:
                return {"status": "error", "message": f"Invalid alert configuration: {'; '.join(errors)}"}
        except Exception as e:
            return {"status": "error", "message": f"Error validating alert configuration: {str(e)}"}

        # Check user limits
        user_status = db.get_user_status(telegram_user_id)
        max_alerts = user_status.get("max_alerts", 5)
        current_alerts = len(db.list_alerts(telegram_user_id))

        if current_alerts >= max_alerts:
            return {
                "status": "error",
                "message": f"Alert limit reached ({max_alerts}). Delete some alerts first or contact admin."
            }

        # Add the indicator alert
        alert_id = db.add_indicator_alert(
            user_id=telegram_user_id,
            ticker=ticker.upper(),
            config_json=config_json,
            alert_action=alert_action,
            timeframe=timeframe,
            email=email
        )

        # Get alert summary for display
        from src.frontend.telegram.screener.alert_logic_evaluator import get_alert_summary
        alert_data = {
            "id": alert_id,
            "ticker": ticker.upper(),
            "alert_type": "indicator",
            "config_json": config_json,
            "timeframe": timeframe,
            "alert_action": alert_action
        }
        summary = get_alert_summary(alert_data)

        email_text = " and email" if email else ""
        return {
            "status": "ok",
            "title": "Indicator Alert Added",
            "message": f"Alert #{alert_id} created: {ticker.upper()} - {summary.get('type', 'Indicator Alert')} ({timeframe}){email_text}"
        }

    except Exception as e:
        _logger.exception("Error adding indicator alert: ")
        return {"status": "error", "message": f"Error adding indicator alert: {str(e)}"}


def handle_alerts_delete(telegram_user_id: str, alert_id_str: str) -> Dict[str, Any]:
    """Delete an alert."""
    try:
        # Validate alert ID
        try:
            alert_id = int(alert_id_str)
        except ValueError:
            return {"status": "error", "message": "Alert ID must be a number"}

        # Check if alert exists and belongs to user
        alert = db.get_alert(alert_id)
        if not alert or alert.get("user_id") != telegram_user_id:
            return {"status": "error", "message": f"Alert #{alert_id} not found or access denied."}

        # Delete the alert
        db.delete_alert(alert_id)

        return {
            "status": "ok",
            "title": "Alert Deleted",
            "message": f"Alert #{alert_id} for {alert['ticker']} has been deleted."
        }

    except Exception as e:
        _logger.exception("Error deleting alert: ")
        return {"status": "error", "message": f"Error deleting alert: {str(e)}"}


def handle_alerts_pause(telegram_user_id: str, alert_id_str: str) -> Dict[str, Any]:
    """Pause an alert."""
    try:
        # Validate alert ID
        try:
            alert_id = int(alert_id_str)
        except ValueError:
            return {"status": "error", "message": "Alert ID must be a number"}

        # Check if alert exists and belongs to user
        alert = db.get_alert(alert_id)
        if not alert or alert.get("user_id") != telegram_user_id:
            return {"status": "error", "message": f"Alert #{alert_id} not found or access denied."}

        # Pause the alert
        db.update_alert(alert_id, active=False)

        return {
            "status": "ok",
            "title": "Alert Paused",
            "message": f"Alert #{alert_id} for {alert['ticker']} has been paused."
        }

    except Exception as e:
        _logger.exception("Error pausing alert: ")
        return {"status": "error", "message": f"Error pausing alert: {str(e)}"}


def handle_alerts_resume(telegram_user_id: str, alert_id_str: str) -> Dict[str, Any]:
    """Resume a paused alert."""
    try:
        # Validate alert ID
        try:
            alert_id = int(alert_id_str)
        except ValueError:
            return {"status": "error", "message": "Alert ID must be a number"}

        # Check if alert exists and belongs to user
        alert = db.get_alert(alert_id)
        if not alert or alert.get("user_id") != telegram_user_id:
            return {"status": "error", "message": f"Alert #{alert_id} not found or access denied."}

        # Resume the alert
        db.update_alert(alert_id, active=True)

        return {
            "status": "ok",
            "title": "Alert Resumed",
            "message": f"Alert #{alert_id} for {alert['ticker']} has been resumed."
        }

    except Exception as e:
        _logger.exception("Error resuming alert: ")
        return {"status": "error", "message": f"Error resuming alert: {str(e)}"}


def handle_schedules(parsed: ParsedCommand) -> Dict[str, Any]:
    """
    Business logic for /schedules commands.
    Handles creating, listing, editing, and deleting scheduled reports.
    """
    try:
        telegram_user_id = parsed.args.get("telegram_user_id")
        if not telegram_user_id:
            return {"status": "error", "message": "No telegram_user_id provided"}

        db.init_db()

        # Check if user has approved access
        access_check = check_approved_access(telegram_user_id)
        if access_check["status"] != "ok":
            return access_check

        # Get action and parameters from positionals
        action = parsed.positionals[0] if len(parsed.positionals) > 0 else None
        params = parsed.positionals[1:] if len(parsed.positionals) > 1 else []

        if not action:
            # List all schedules for user
            return handle_schedules_list(telegram_user_id)

        if action == "screener" and len(params) >= 1:
            list_type = params[0]
            time = params[1] if len(params) > 1 else "09:00"  # Default time
            # Get flags from parsed args
            email = parsed.args.get("email", False)
            indicators = parsed.args.get("indicators")
            return handle_schedules_screener(telegram_user_id, list_type, time, email, indicators)
        elif action == "enhanced_screener" and len(params) >= 1:
            config_json = params[0]
            return handle_schedules_enhanced_screener(telegram_user_id, config_json)
        elif action == "add" and len(params) >= 2:
            ticker, time = params[0], params[1]
            # Get flags from parsed args
            email = parsed.args.get("email", False)
            indicators = parsed.args.get("indicators")
            period = parsed.args.get("period", "2y")
            interval = parsed.args.get("interval", "1d")
            provider = parsed.args.get("provider")
            return handle_schedules_add(telegram_user_id, ticker, time, email, indicators, period, interval, provider)
        elif action == "edit" and len(params) >= 1:
            schedule_id = params[0]
            new_time = params[1] if len(params) > 1 else None
            return handle_schedules_edit(telegram_user_id, schedule_id, new_time, parsed.args)
        elif action == "delete" and len(params) >= 1:
            schedule_id = params[0]
            return handle_schedules_delete(telegram_user_id, schedule_id)
        elif action == "pause" and len(params) >= 1:
            schedule_id = params[0]
            return handle_schedules_pause(telegram_user_id, schedule_id)
        elif action == "resume" and len(params) >= 1:
            schedule_id = params[0]
            return handle_schedules_resume(telegram_user_id, schedule_id)
        elif action == "add_json" and len(params) >= 1:
            config_json = params[0]
            return handle_schedules_add_json(telegram_user_id, config_json)
        else:
            return {
                "status": "error",
                "title": "Schedules Help",
                "message": ("Available schedule commands:\n"
                           "/schedules - List all schedules\n"
                           "/schedules add TICKER TIME [flags] - Schedule daily report\n"
                           "  TIME: HH:MM format (24h UTC)\n"
                           "  Example: /schedules add AAPL 09:00 -email\n"
                           "Flags:\n"
                           "  -email: Send report to email\n"
                           "  -indicators=RSI,MACD: Specify indicators\n"
                           "  -period=1y: Data period\n"
                           "  -interval=1d: Data interval\n"
                           "  -provider=yf: Data provider\n"
                           "/schedules add_json CONFIG_JSON - Add advanced schedule with JSON config\n"
                           "  Example: /schedules add_json '{\"type\":\"report\",\"ticker\":\"AAPL\",\"scheduled_time\":\"09:00\",\"period\":\"1y\",\"interval\":\"1d\",\"email\":true}'\n"
                           "/schedules screener LIST_TYPE [TIME] [flags] - Schedule fundamental screener\n"
                           "  LIST_TYPE: us_small_cap, us_medium_cap, us_large_cap, swiss_shares, custom_list\n"
                           "  TIME: HH:MM format (24h UTC)\n"
                           "  Example: /schedules screener us_small_cap 09:00 -email\n"
                           "  Example: /schedules screener us_large_cap -indicators=PE,PB,ROE\n"
                           "/schedules enhanced_screener CONFIG_JSON - Schedule enhanced screener with JSON config\n"
                           "  Example: /schedules enhanced_screener '{\"screener_type\":\"hybrid\",\"list_type\":\"us_medium_cap\",\"fmp_criteria\":{\"marketCapMoreThan\":2000000000,\"peRatioLessThan\":20,\"returnOnEquityMoreThan\":0.12,\"limit\":50},\"fundamental_criteria\":[{\"indicator\":\"PE\",\"operator\":\"max\",\"value\":15,\"weight\":1.0,\"required\":true}],\"technical_criteria\":[{\"indicator\":\"RSI\",\"parameters\":{\"period\":14},\"condition\":{\"operator\":\"<\",\"value\":70},\"weight\":0.6,\"required\":false}],\"max_results\":10,\"min_score\":7.0,\"email\":true}'\n"
                           "/schedules edit SCHEDULE_ID [TIME] [flags] - Edit schedule\n"
                           "/schedules delete SCHEDULE_ID - Delete schedule\n"
                           "/schedules pause SCHEDULE_ID - Pause schedule\n"
                           "/schedules resume SCHEDULE_ID - Resume schedule\n\n"
                           "JSON Schedule Examples:\n"
                           "• Single Report: {\"type\":\"report\",\"ticker\":\"AAPL\",\"scheduled_time\":\"09:00\",\"period\":\"1y\",\"interval\":\"1d\",\"email\":true}\n"
                           "• Multiple Reports: {\"type\":\"report\",\"tickers\":[\"AAPL\",\"MSFT\",\"GOOGL\"],\"scheduled_time\":\"09:00\",\"period\":\"1y\",\"interval\":\"1d\",\"indicators\":\"RSI,MACD\",\"email\":true}\n"
                           "• Advanced Report: {\"type\":\"report\",\"ticker\":\"TSLA\",\"scheduled_time\":\"16:30\",\"period\":\"6mo\",\"interval\":\"1h\",\"indicators\":\"RSI,MACD,BollingerBands\",\"email\":true}\n"
                           "• Screener: {\"type\":\"screener\",\"list_type\":\"us_small_cap\",\"scheduled_time\":\"08:00\",\"period\":\"1y\",\"interval\":\"1d\",\"indicators\":\"PE,PB,ROE\",\"email\":true}\n"
                           "• FMP Enhanced Screener: {\"screener_type\":\"hybrid\",\"list_type\":\"us_medium_cap\",\"fmp_criteria\":{\"marketCapMoreThan\":2000000000,\"peRatioLessThan\":20,\"returnOnEquityMoreThan\":0.12,\"limit\":50},\"fundamental_criteria\":[{\"indicator\":\"PE\",\"operator\":\"max\",\"value\":15,\"weight\":1.0,\"required\":true}],\"max_results\":10,\"min_score\":7.0,\"email\":true}\n"
                           "• FMP Strategy Screener: {\"screener_type\":\"hybrid\",\"list_type\":\"us_large_cap\",\"fmp_strategy\":\"conservative_value\",\"fundamental_criteria\":[{\"indicator\":\"ROE\",\"operator\":\"min\",\"value\":15,\"weight\":1.0,\"required\":true}],\"max_results\":15,\"min_score\":7.5,\"email\":true}")
            }

    except Exception as e:
        _logger.exception("Error in schedules command: ")
        return {"status": "error", "message": f"Error processing schedules command: {str(e)}"}


def handle_schedules_add_json(telegram_user_id: str, config_json: str) -> Dict[str, Any]:
    """Add a new JSON-based schedule with support for multiple tickers and report configurations."""
    try:
        # Parse JSON to determine schedule type
        import json
        config = json.loads(config_json)
        schedule_type = config.get("type", "report")

        # Validate JSON configuration based on type
        if schedule_type == "report":
            # Validate report-specific fields
            required_fields = ["scheduled_time"]
            missing_fields = [field for field in required_fields if field not in config]
            if missing_fields:
                return {"status": "error", "message": f"Missing required fields: {', '.join(missing_fields)}"}

            # Check for either 'ticker' (single) or 'tickers' (multiple)
            ticker = config.get("ticker")
            tickers = config.get("tickers", [])

            if not ticker and not tickers:
                return {"status": "error", "message": "Either 'ticker' (single) or 'tickers' (multiple) must be specified"}

            if ticker and tickers:
                return {"status": "error", "message": "Cannot specify both 'ticker' and 'tickers' - use one or the other"}

            # Validate scheduled_time format (HH:MM)
            scheduled_time = config.get("scheduled_time", "")
            if not scheduled_time or not isinstance(scheduled_time, str):
                return {"status": "error", "message": "scheduled_time must be a string in HH:MM format"}

            # Basic time format validation
            try:
                hour, minute = scheduled_time.split(":")
                if not (0 <= int(hour) <= 23 and 0 <= int(minute) <= 59):
                    raise ValueError("Invalid time")
            except:
                return {"status": "error", "message": "scheduled_time must be in HH:MM format (24h)"}

            # Validate tickers list if multiple
            if tickers and not isinstance(tickers, list):
                return {"status": "error", "message": "tickers must be a list"}

        else:
            # Use existing validation for other schedule types
            try:
                from src.frontend.telegram.screener.schedule_config_parser import validate_schedule_config
                is_valid, errors = validate_schedule_config(config_json)
                if not is_valid:
                    return {"status": "error", "message": f"Invalid schedule configuration: {'; '.join(errors)}"}
            except Exception as e:
                return {"status": "error", "message": f"Error validating schedule configuration: {str(e)}"}

        # Check user limits
        user_status = db.get_user_status(telegram_user_id)
        max_schedules = user_status.get("max_schedules", 5)
        current_schedules = len(db.list_schedules(telegram_user_id))

        if current_schedules >= max_schedules:
            return {
                "status": "error",
                "message": f"Schedule limit reached ({max_schedules}). Delete some schedules first or contact admin."
            }

        # Determine schedule_config based on type
        if schedule_type == "report":
            schedule_config = "report"
        elif schedule_type == "enhanced_screener":
            schedule_config = "enhanced_screener"
        else:
            schedule_config = "advanced"

        # Add the JSON schedule
        schedule_id = db.add_json_schedule(
            user_id=telegram_user_id,
            config_json=config_json,
            schedule_config=schedule_config
        )

        # Create success message based on type
        if schedule_type == "report":
            # Handle report schedule
            ticker = config.get("ticker")
            tickers = config.get("tickers", [])

            if ticker:
                tickers_str = ticker
                ticker_count = 1
            else:
                tickers_str = ", ".join(tickers)
                ticker_count = len(tickers)

            period = config.get("period", "2y")
            interval = config.get("interval", "1d")
            indicators = config.get("indicators", "")
            email_flag = " (with email)" if config.get("email", False) else ""

            message = f"Report schedule #{schedule_id} created for {tickers_str} at {scheduled_time} UTC{email_flag}"
            if indicators:
                message += f"\nIndicators: {indicators}"
            message += f"\nPeriod: {period}, Interval: {interval}"

            if ticker_count > 1:
                message += f"\n📊 Multiple tickers: {ticker_count} reports will be generated"

        else:
            # Use existing summary for other types
            from src.frontend.telegram.screener.schedule_config_parser import get_schedule_summary
            summary = get_schedule_summary(config_json)

            if "error" in summary:
                return {"status": "error", "message": f"Error creating schedule: {summary['error']}"}

            message = f"Schedule #{schedule_id} created: {summary.get('type', 'Unknown')} at {summary.get('scheduled_time', 'Unknown')}"

        return {
            "status": "ok",
            "title": "Schedule Added",
            "message": message
        }

    except json.JSONDecodeError:
        return {"status": "error", "message": "Invalid JSON format"}
    except Exception as e:
        _logger.exception("Error adding JSON schedule: ")
        return {"status": "error", "message": f"Error adding JSON schedule: {str(e)}"}



def handle_schedules_enhanced_screener(telegram_user_id: str, config_json: str) -> Dict[str, Any]:
    """Handle enhanced screener schedule creation with JSON configuration."""
    try:
        # Validate JSON configuration
        try:
            from src.frontend.telegram.screener.screener_config_parser import validate_screener_config
            is_valid, errors = validate_screener_config(config_json)
            if not is_valid:
                return {"status": "error", "message": f"Invalid screener configuration: {'; '.join(errors)}"}
        except Exception as e:
            return {"status": "error", "message": f"Error validating screener configuration: {str(e)}"}

        # Check user limits
        user_status = db.get_user_status(telegram_user_id)
        max_schedules = user_status.get("max_schedules", 5)
        current_schedules = len(db.list_schedules(telegram_user_id))

        if current_schedules >= max_schedules:
            return {
                "status": "error",
                "message": f"Schedule limit reached ({max_schedules}). Delete some schedules first or contact admin."
            }

        # Parse the configuration to get summary
        from src.frontend.telegram.screener.screener_config_parser import get_screener_summary
        summary = get_screener_summary(config_json)

        if "error" in summary:
            return {"status": "error", "message": f"Error parsing screener configuration: {summary['error']}"}

        # Add the enhanced screener schedule
        schedule_id = db.add_json_schedule(
            user_id=telegram_user_id,
            config_json=config_json,
            schedule_config="enhanced_screener"
        )

        # Create success message
        screener_type = summary.get("screener_type", "Unknown")
        list_type = summary.get("list_type", "Unknown")
        fundamental_count = summary.get("fundamental_criteria_count", 0)
        technical_count = summary.get("technical_criteria_count", 0)
        max_results = summary.get("max_results", 10)
        min_score = summary.get("min_score", 7.0)

        message = f"✅ Enhanced screener scheduled successfully!\n\n"
        message += f"📊 **Screener Type**: {screener_type.title()}\n"
        message += f"🔍 **List Type**: {list_type.replace('_', ' ').title()}\n"

        # Add FMP information if available
        fmp_criteria_count = summary.get("fmp_criteria_count", 0)
        fmp_strategy = summary.get("fmp_strategy")

        if fmp_criteria_count > 0:
            message += f"🚀 **FMP Pre-filtering**: {fmp_criteria_count} criteria\n"
        if fmp_strategy:
            message += f"📋 **FMP Strategy**: {fmp_strategy}\n"

        message += f"📈 **Fundamental Criteria**: {fundamental_count} indicators\n"
        message += f"📊 **Technical Criteria**: {technical_count} indicators\n"
        message += f"🎯 **Max Results**: {max_results}\n"
        message += f"📊 **Min Score**: {min_score}/10\n"
        message += f"🆔 **Schedule ID**: {schedule_id}\n\n"

        if screener_type == "fundamental":
            message += "This screener will analyze stocks based on fundamental metrics only."
        elif screener_type == "technical":
            message += "This screener will analyze stocks based on technical indicators only."
        elif screener_type == "hybrid":
            message += "This screener will combine fundamental and technical analysis for comprehensive screening."

        return {
            "status": "ok",
            "title": "Enhanced Screener Scheduled",
            "message": message
        }

    except Exception as e:
        _logger.exception("Error adding enhanced screener schedule: ")
        return {"status": "error", "message": f"Error adding enhanced screener schedule: {str(e)}"}


def handle_schedules_list(telegram_user_id: str) -> Dict[str, Any]:
    """List all schedules for a user."""
    try:
        schedules = db.list_schedules(telegram_user_id)
        if not schedules:
            return {"status": "ok", "title": "Your Schedules", "message": "You have no scheduled reports."}

        schedule_list = []
        for schedule in schedules:
            status = "🟢 Active" if schedule.get("active") else "🔴 Paused"
            email_flag = "📧" if schedule.get("email") else "💬"

            # Handle different schedule types
            schedule_config = schedule.get("schedule_config", "simple")
            if schedule_config == "simple":
                period = schedule.get("period", "daily")
                schedule_list.append(
                    f"#{schedule['id']}: {schedule['ticker']} at {schedule['scheduled_time']} ({period}) {email_flag} - {status}"
                )
            else:
                # JSON-based schedule
                from src.frontend.telegram.screener.schedule_config_parser import get_schedule_summary
                config_json = schedule.get("config_json")
                if config_json:
                    summary = get_schedule_summary(config_json)
                    if "error" not in summary:
                        schedule_type = summary.get("type", "Unknown")
                        scheduled_time = summary.get("scheduled_time", "Unknown")
                        ticker = summary.get("ticker", "")
                        list_type = summary.get("list_type", "")

                        if schedule_type == "report":
                            schedule_list.append(
                                f"#{schedule['id']}: 📊 {ticker} Report at {scheduled_time} {email_flag} - {status}"
                            )
                        elif schedule_type == "screener":
                            schedule_list.append(
                                f"#{schedule['id']}: 🔍 {list_type} Screener at {scheduled_time} {email_flag} - {status}"
                            )
                        else:
                            schedule_list.append(
                                f"#{schedule['id']}: ⚙️ {schedule_type} at {scheduled_time} {email_flag} - {status}"
                            )
                    else:
                        schedule_list.append(
                            f"#{schedule['id']}: ⚙️ JSON Schedule at {schedule.get('scheduled_time', 'Unknown')} {email_flag} - {status}"
                        )
                else:
                    schedule_list.append(
                        f"#{schedule['id']}: ⚙️ JSON Schedule at {schedule.get('scheduled_time', 'Unknown')} {email_flag} - {status}"
                    )

        message = f"Your scheduled reports ({len(schedules)}):\n\n" + "\n".join(schedule_list)
        return {"status": "ok", "title": "Your Schedules", "message": message}

    except Exception as e:
        _logger.exception("Error listing schedules: ")
        return {"status": "error", "message": f"Error listing schedules: {str(e)}"}


def handle_schedules_add(telegram_user_id: str, ticker: str, time: str, email: bool = False,
                        indicators: str = None, period: str = "2y", interval: str = "1d",
                        provider: str = None) -> Dict[str, Any]:
    """Add a new scheduled report."""
    try:
        # Validate time format (HH:MM)
        import re
        if not re.match(r'^([01]?[0-9]|2[0-3]):[0-5][0-9]$', time):
            return {"status": "error", "message": "Time must be in HH:MM format (24-hour, e.g., 09:00, 15:30)"}

        # Check user limits
        user_status = db.get_user_status(telegram_user_id)
        max_schedules = user_status.get("max_schedules", 5)
        current_schedules = len(db.list_schedules(telegram_user_id))

        if current_schedules >= max_schedules:
            return {
                "status": "error",
                "message": f"Schedule limit reached ({max_schedules}). Delete some schedules first or contact admin."
            }

        # Add the schedule (convert ticker to uppercase)
        schedule_id = db.add_schedule(
            telegram_user_id,
            ticker.upper(),
            time,
            period="daily",  # Default to daily for now
            email=email,
            indicators=indicators,
            interval=interval,
            provider=provider
        )

        email_text = " and email" if email else ""
        indicators_text = f" with indicators: {indicators}" if indicators else ""

        return {
            "status": "ok",
            "title": "Schedule Added",
            "message": f"Schedule #{schedule_id} created: {ticker.upper()} daily at {time} (UTC) via Telegram{email_text}{indicators_text}"
        }

    except Exception as e:
        _logger.exception("Error adding schedule: ")
        return {"status": "error", "message": f"Error adding schedule: {str(e)}"}


def handle_schedules_edit(telegram_user_id: str, schedule_id_str: str, new_time: str = None, args: dict = None) -> Dict[str, Any]:
    """Edit an existing schedule."""
    try:
        # Validate schedule ID
        try:
            schedule_id = int(schedule_id_str)
        except ValueError:
            return {"status": "error", "message": "Schedule ID must be a number"}

        # Check if schedule exists and belongs to user
        schedule = db.get_schedule_by_id(schedule_id)
        if not schedule or schedule.get("user_id") != telegram_user_id:
            return {"status": "error", "message": f"Schedule #{schedule_id} not found or access denied."}

        updates = {}

        # Validate and set new time
        if new_time:
            import re
            if not re.match(r'^([01]?[0-9]|2[0-3]):[0-5][0-9]$', new_time):
                return {"status": "error", "message": "Time must be in HH:MM format (24-hour, e.g., 09:00, 15:30)"}
            updates["scheduled_time"] = new_time

        # Update other flags if provided
        if args:
            if "email" in args:
                updates["email"] = args["email"]
            if "indicators" in args and args["indicators"]:
                updates["indicators"] = args["indicators"]
            if "period" in args and args["period"]:
                updates["period"] = args["period"]
            if "interval" in args and args["interval"]:
                updates["interval"] = args["interval"]
            if "provider" in args and args["provider"]:
                updates["provider"] = args["provider"]

        if not updates:
            return {"status": "error", "message": "No updates provided. Specify new time and/or flags."}

        # Update the schedule
        db.update_schedule(schedule_id, **updates)

        # Get updated schedule for confirmation
        updated_schedule = db.get_schedule_by_id(schedule_id)

        return {
            "status": "ok",
            "title": "Schedule Updated",
            "message": f"Schedule #{schedule_id} updated: {updated_schedule['ticker']} at {updated_schedule['scheduled_time']}"
        }

    except Exception as e:
        _logger.exception("Error editing schedule: ")
        return {"status": "error", "message": f"Error editing schedule: {str(e)}"}


def handle_schedules_delete(telegram_user_id: str, schedule_id_str: str) -> Dict[str, Any]:
    """Delete a schedule."""
    try:
        # Validate schedule ID
        try:
            schedule_id = int(schedule_id_str)
        except ValueError:
            return {"status": "error", "message": "Schedule ID must be a number"}

        # Check if schedule exists and belongs to user
        schedule = db.get_schedule_by_id(schedule_id)
        if not schedule or schedule.get("user_id") != telegram_user_id:
            return {"status": "error", "message": f"Schedule #{schedule_id} not found or access denied."}

        # Delete the schedule
        db.delete_schedule(schedule_id)

        return {
            "status": "ok",
            "title": "Schedule Deleted",
            "message": f"Schedule #{schedule_id} for {schedule['ticker']} has been deleted."
        }

    except Exception as e:
        _logger.exception("Error deleting schedule: ")
        return {"status": "error", "message": f"Error deleting schedule: {str(e)}"}


def handle_schedules_pause(telegram_user_id: str, schedule_id_str: str) -> Dict[str, Any]:
    """Pause a schedule."""
    try:
        # Validate schedule ID
        try:
            schedule_id = int(schedule_id_str)
        except ValueError:
            return {"status": "error", "message": "Schedule ID must be a number"}

        # Check if schedule exists and belongs to user
        schedule = db.get_schedule_by_id(schedule_id)
        if not schedule or schedule.get("user_id") != telegram_user_id:
            return {"status": "error", "message": f"Schedule #{schedule_id} not found or access denied."}

        # Pause the schedule
        db.update_schedule(schedule_id, active=False)

        return {
            "status": "ok",
            "title": "Schedule Paused",
            "message": f"Schedule #{schedule_id} for {schedule['ticker']} has been paused."
        }

    except Exception as e:
        _logger.exception("Error pausing schedule: ")
        return {"status": "error", "message": f"Error pausing schedule: {str(e)}"}


def handle_schedules_resume(telegram_user_id: str, schedule_id_str: str) -> Dict[str, Any]:
    """Resume a paused schedule."""
    try:
        # Validate schedule ID
        try:
            schedule_id = int(schedule_id_str)
        except ValueError:
            return {"status": "error", "message": "Schedule ID must be a number"}

        # Check if schedule exists and belongs to user
        schedule = db.get_schedule_by_id(schedule_id)
        if not schedule or schedule.get("user_id") != telegram_user_id:
            return {"status": "error", "message": f"Schedule #{schedule_id} not found or access denied."}

        # Resume the schedule
        db.update_schedule(schedule_id, active=True)

        return {
            "status": "ok",
            "title": "Schedule Resumed",
            "message": f"Schedule #{schedule_id} for {schedule['ticker']} has been resumed."
        }

    except Exception as e:
        _logger.exception("Error resuming schedule: ")
        return {"status": "error", "message": f"Error resuming schedule: {str(e)}"}


def handle_schedules_screener(telegram_user_id: str, list_type: str, time: str,
                            email: bool = False, indicators: str = None) -> Dict[str, Any]:
    """Handle screener schedule creation."""
    try:
        # Validate list type (case-insensitive)
        valid_list_types = ['us_small_cap', 'us_medium_cap', 'us_large_cap', 'swiss_shares', 'custom_list']
        if list_type.lower() not in [lt.lower() for lt in valid_list_types]:
            return {
                'status': 'error',
                'message': f"Invalid list type. Valid types: {', '.join(valid_list_types)}"
            }

        # Convert list_type to lowercase for consistency
        list_type = list_type.lower()

        # Validate time format
        try:
            # Simple time validation (HH:MM format)
            hour, minute = map(int, time.split(':'))
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                raise ValueError("Invalid time")
        except (ValueError, AttributeError):
            return {
                'status': 'error',
                'message': "Invalid time format. Use HH:MM (24-hour format, UTC)"
            }

        # Check user limits
        current_schedules = db.list_schedules(telegram_user_id)
        user_limit = db.get_user_limit(telegram_user_id, 'max_schedules')

        # Default to 5 if no limit is set
        if user_limit is None:
            user_limit = 5

        if len(current_schedules) >= user_limit:
            return {
                'status': 'error',
                'message': f"You have reached your limit of {user_limit} scheduled reports. Delete some schedules first."
            }

        # Create screener schedule
        schedule_data = {
            'telegram_user_id': telegram_user_id,
            'ticker': f"SCREENER_{list_type.upper()}",  # Special ticker format for screeners
            'scheduled_time': time,
            'email': email,
            'indicators': indicators,
            'period': 'daily',  # Screeners run daily
            'interval': '1d',
            'provider': 'yf',
            'active': True,
            'schedule_type': 'screener',  # New field to distinguish screeners
            'list_type': list_type  # Store the list type
        }

        schedule_id = db.create_schedule(schedule_data)

        if schedule_id:
            message = f"✅ Fundamental screener scheduled successfully!\n"
            message += f"📊 **List Type**: {list_type.replace('_', ' ').title()}\n"
            message += f"⏰ **Time**: {time} UTC (daily)\n"
            message += f"📧 **Email**: {'Yes' if email else 'No'}\n"
            if indicators:
                message += f"📈 **Indicators**: {indicators}\n"
            message += f"🆔 **Schedule ID**: {schedule_id}\n\n"
            message += "The screener will analyze stocks for undervaluation based on:\n"
            message += "• P/E < 15, P/B < 1.5, P/S < 1\n"
            message += "• ROE > 15%, Debt/Equity < 0.5\n"
            message += "• Positive Free Cash Flow\n"
            message += "• Composite scoring (0-10 scale)\n"
            message += "• DCF valuation analysis"

            return {
                'status': 'ok',
                'title': "Screener Scheduled",
                'message': message
            }
        else:
            return {
                'status': 'error',
                'message': "Failed to create screener schedule. Please try again."
            }

    except Exception as e:
        _logger.exception("Error creating screener schedule: ")
        return {
            'status': 'error',
            'message': f"Error creating screener schedule: {str(e)}"
        }


def handle_feedback(parsed: ParsedCommand) -> Dict[str, Any]:
    """
    Business logic for /feedback command.
    Collects user feedback and forwards to administrators.
    """
    try:
        telegram_user_id = parsed.args.get("telegram_user_id")
        feedback = parsed.args.get("feedback")

        if not telegram_user_id:
            return {"status": "error", "message": "No telegram_user_id provided"}

        if not feedback:
            return {"status": "error", "message": "Please provide feedback message. Usage: /feedback Your message here"}

        # Log feedback for admin review
        _logger.info("User feedback", extra={
            "user_id": telegram_user_id,
            "feedback": feedback,
            "type": "feedback"
        })

        # Store feedback in database for admin panel
        db.init_db()
        feedback_id = db.add_feedback(telegram_user_id, "feedback", feedback)

        return {
            "status": "ok",
            "title": "Feedback Received",
            "message": "Thank you for your feedback! It has been forwarded to the development team.",
            "admin_notification": {
                "type": "feedback",
                "user_id": telegram_user_id,
                "message": feedback
            }
        }

    except Exception as e:
        _logger.exception("Error processing feedback: ")
        return {"status": "error", "message": f"Error processing feedback: {str(e)}"}


def handle_feature(parsed: ParsedCommand) -> Dict[str, Any]:
    """
    Business logic for /feature command.
    Collects feature requests and forwards to administrators.
    """
    try:
        telegram_user_id = parsed.args.get("telegram_user_id")
        feature_request = parsed.args.get("feature")

        if not telegram_user_id:
            return {"status": "error", "message": "No telegram_user_id provided"}

        if not feature_request:
            return {"status": "error", "message": "Please provide feature request. Usage: /feature Your feature idea here"}

        # Log feature request for admin review
        _logger.info("Feature request", extra={
            "user_id": telegram_user_id,
            "feature_request": feature_request,
            "type": "feature_request"
        })

        # Store feature request in database for admin panel
        db.init_db()
        feature_id = db.add_feedback(telegram_user_id, "feature_request", feature_request)

        return {
            "status": "ok",
            "title": "Feature Request Received",
            "message": "Thank you for your feature request! It has been added to our development backlog.",
            "admin_notification": {
                "type": "feature_request",
                "user_id": telegram_user_id,
                "message": feature_request
            }
        }

    except Exception as e:
        _logger.exception("Error processing feature request: ")
        return {"status": "error", "message": f"Error processing feature request: {str(e)}"}


def handle_register(parsed: ParsedCommand) -> Dict[str, Any]:
    """
    Business logic for /register command.
    Register or update user email and send verification code.
    """
    try:
        telegram_user_id = parsed.args.get("telegram_user_id")
        email = parsed.args.get("email")
        language = parsed.args.get("language", "en")

        if not telegram_user_id:
            return {"status": "error", "message": "No telegram_user_id provided"}

        if not email:
            return {"status": "error", "message": "Please provide an email address. Usage: /register email@example.com [language]"}

        # Validate email format
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            return {"status": "error", "message": "Please provide a valid email address."}

        # Check rate limiting
        db.init_db()
        codes_sent = db.count_codes_last_hour(telegram_user_id)
        if codes_sent >= 5:
            return {"status": "error", "message": "Too many verification codes sent. Please wait an hour before requesting another."}

        # Generate verification code
        import random
        code = f"{random.randint(100000, 999999):06d}"
        sent_time = int(time.time())

        # Store user and code
        db.set_user_email(telegram_user_id, email, code, sent_time, language)

        # Send verification code via email
        # This will be handled by the notification system
        return {
            "status": "ok",
            "title": "Email Registration",
            "message": f"A 6-digit verification code has been sent to {email}. Use /verify CODE to verify your email.",
            "email_verification": {
                "email": email,
                "code": code,
                "user_id": telegram_user_id
            }
        }

    except Exception as e:
        _logger.exception("Error in register command: ")
        return {"status": "error", "message": f"Error registering email: {str(e)}"}


def handle_verify(parsed: ParsedCommand) -> Dict[str, Any]:
    """
    Business logic for /verify command.
    Verify user email with the provided code.
    """
    try:
        telegram_user_id = parsed.args.get("telegram_user_id")
        code = parsed.args.get("code")

        if not telegram_user_id:
            return {"status": "error", "message": "No telegram_user_id provided"}

        if not code:
            return {"status": "error", "message": "Please provide the verification code. Usage: /verify CODE"}

        # Validate code format
        if not code.isdigit() or len(code) != 6:
            return {"status": "error", "message": "Verification code must be a 6-digit number."}

        # Verify the code
        db.init_db()
        if db.verify_code(telegram_user_id, code, expiry_seconds=3600):
            return {
                "status": "ok",
                "title": "Email Verified",
                "message": "Your email has been successfully verified! You can now use all bot features including email reports."
            }
        else:
            return {
                "status": "error",
                "message": "Invalid or expired verification code. Please check the code or request a new one with /register."
            }

    except Exception as e:
        _logger.exception("Error in verify command: ")
        return {"status": "error", "message": f"Error verifying code: {str(e)}"}


def handle_language(parsed: ParsedCommand) -> Dict[str, Any]:
    """
    Business logic for /language command.
    Update user's language preference.
    """
    try:
        telegram_user_id = parsed.args.get("telegram_user_id")
        language = parsed.args.get("language")

        if not telegram_user_id:
            return {"status": "error", "message": "No telegram_user_id provided"}

        if not language:
            return {"status": "error", "message": "Please provide a language code. Usage: /language en (supported: en, ru)"}

        # Validate language
        supported_languages = ["en", "ru"]
        if language.lower() not in supported_languages:
            return {"status": "error", "message": f"Language '{language}' not supported. Supported languages: {', '.join(supported_languages)}"}

        # Check if user has approved access
        access_check = check_approved_access(telegram_user_id)
        if access_check["status"] != "ok":
            return access_check

        # Update user language
        db.init_db()
        user_status = db.get_user_status(telegram_user_id)
        if not user_status:
            return {"status": "error", "message": "Please register first using /register email@example.com"}

        # Update language in database
        conn = sqlite3.connect(db.DB_PATH)
        c = conn.cursor()
        c.execute("UPDATE users SET language=? WHERE telegram_user_id=?", (language.lower(), telegram_user_id))
        conn.commit()
        conn.close()

        return {
            "status": "ok",
            "title": "Language Updated",
            "message": f"Your language preference has been updated to {language.upper()}."
        }

    except Exception as e:
        _logger.exception("Error in language command: ")
        return {"status": "error", "message": f"Error updating language: {str(e)}"}


def handle_screener(parsed: ParsedCommand) -> Dict[str, Any]:
    """
    Business logic for /screener command for immediate screener execution.
    Supports both predefined screeners and custom JSON configuration.
    """
    try:
        # Extract parameters
        telegram_user_id = parsed.args.get("telegram_user_id")
        config_json = parsed.args.get("screener_name_or_config")
        send_email = parsed.args.get("email", False)

        if not telegram_user_id:
            return {"status": "error", "message": "No telegram_user_id provided"}

        if not config_json:
            return {"status": "error", "message": "Please provide screener name or configuration. Usage: /screener <SCREENER_NAME> [-email] or /screener <JSON_CONFIG> [-email]"}

        # Check if user has approved access
        access_check = check_approved_access(telegram_user_id)
        if access_check["status"] != "ok":
            return access_check

        # Import screener modules
        from src.frontend.telegram.screener.enhanced_screener import EnhancedScreener
        from src.frontend.telegram.screener.screener_config_parser import (
            parse_screener_config,
            validate_screener_config
        )

        # Check if config_json is a predefined screener name
        screener_config = None
        if config_json.startswith('{'):
            # It's a JSON configuration
            is_valid, errors = validate_screener_config(config_json)
            if not is_valid:
                return {"status": "error", "message": f"Invalid screener configuration: {errors}"}
            screener_config = parse_screener_config(config_json)
        else:
            # It's a predefined screener name
            screener_config = _get_predefined_screener_config(config_json)
            if not screener_config:
                return {"status": "error", "message": f"Unknown screener: {config_json}. Available screeners: {', '.join(_get_available_screeners())}"}

        # Run enhanced screener immediately
        enhanced_screener = EnhancedScreener()
        report = enhanced_screener.run_enhanced_screener(screener_config)

        if report.error:
            return {"status": "error", "message": report.error}

        # Format results
        message = enhanced_screener.format_enhanced_telegram_message(report, screener_config)

        # Send results
        if send_email:
            # Get user email
            db.init_db()
            user_status = db.get_user_status(telegram_user_id)
            if not user_status or not user_status.get("email"):
                return {"status": "error", "message": "Email not registered. Please use /register email@example.com first"}

            # Send via email
            from src.frontend.telegram.screener.notifications import send_screener_email
            send_screener_email(user_status["email"], report, screener_config)
            return {"status": "success", "message": "Screener results sent to your email"}
        else:
            # Return for Telegram display
            return {"status": "success", "message": message, "report": report}

    except Exception as e:
        _logger.exception("Error in screener command")
        return {"status": "error", "message": f"Screener error: {str(e)}"}


def _get_predefined_screener_config(screener_name: str):
    """
    Get predefined screener configuration by name.
    """
    try:
        import json
        from pathlib import Path
        from src.frontend.telegram.screener.screener_config_parser import ScreenerConfigParser

        # Load FMP screener criteria
        config_path = Path(__file__).resolve().parents[4] / "config" / "screener" / "fmp_screener_criteria.json"

        with open(config_path, 'r') as f:
            fmp_config = json.load(f)

        # Check if screener exists in predefined strategies
        if screener_name in fmp_config.get("predefined_strategies", {}):
            strategy = fmp_config["predefined_strategies"][screener_name]

            # Create screener configuration dictionary
            config_dict = {
                "screener_type": "hybrid",
                "list_type": _get_list_type_for_screener(screener_name),
                "fmp_criteria": strategy["criteria"],
                "fundamental_criteria": _get_fundamental_criteria_for_screener(screener_name),
                "technical_criteria": _get_technical_criteria_for_screener(screener_name),
                "max_results": strategy["criteria"].get("limit", 50),
                "min_score": 0.5,
                "period": "1y",
                "interval": "1d"
            }

            # Convert dictionary to ScreenerConfig object
            parser = ScreenerConfigParser()
            return parser._parse_config_dict(config_dict)

        return None

    except Exception as e:
        _logger.error("Error loading predefined screener config for %s: %s", screener_name, e)
        return None


def _get_list_type_for_screener(screener_name: str) -> str:
    """
    Determine the appropriate list type for a given screener.
    """
    if screener_name == "six_stocks":
        return "swiss_shares"
    elif screener_name in ["mid_cap_stocks", "large_cap_stocks", "extra_large_cap_stocks"]:
        return "us_large_cap"  # Will be filtered by FMP criteria
    else:
        return "us_medium_cap"  # Default fallback


def _get_fundamental_criteria_for_screener(screener_name: str):
    """
    Get fundamental criteria for a predefined screener.
    """
    # Base fundamental criteria for all screeners
    base_criteria = [
        {
            "indicator": "PE",
            "operator": "max",
            "value": 30,
            "weight": 1.0,
            "required": False
        },
        {
            "indicator": "PB",
            "operator": "max",
            "value": 3.0,
            "weight": 1.0,
            "required": False
        },
        {
            "indicator": "ROE",
            "operator": "min",
            "value": 0.10,
            "weight": 1.0,
            "required": False
        }
    ]

    # Adjust criteria based on screener type
    if screener_name == "conservative_value":
        base_criteria[0]["value"] = 12  # PE < 12
        base_criteria[1]["value"] = 1.2  # PB < 1.2
        base_criteria[2]["value"] = 0.15  # ROE > 15%
    elif screener_name == "deep_value":
        base_criteria[0]["value"] = 8   # PE < 8
        base_criteria[1]["value"] = 0.8  # PB < 0.8
        base_criteria[2]["value"] = 0.08  # ROE > 8%
    elif screener_name == "quality_growth":
        base_criteria[0]["value"] = 25  # PE < 25
        base_criteria[1]["value"] = 5.0  # PB < 5.0
        base_criteria[2]["value"] = 0.18  # ROE > 18%
    elif screener_name == "large_cap_stocks":
        base_criteria[0]["value"] = 35  # PE < 35
        base_criteria[1]["value"] = 5.0  # PB < 5.0
        base_criteria[2]["value"] = 0.15  # ROE > 15%
    elif screener_name == "extra_large_cap_stocks":
        base_criteria[0]["value"] = 40  # PE < 40
        base_criteria[1]["value"] = 6.0  # PB < 6.0
        base_criteria[2]["value"] = 0.15  # ROE > 15%
    elif screener_name == "six_stocks":
        base_criteria[0]["value"] = 20  # PE < 20
        base_criteria[1]["value"] = 2.5  # PB < 2.5
        base_criteria[2]["value"] = 0.08  # ROE > 8%

    return base_criteria


def _get_technical_criteria_for_screener(screener_name: str):
    """
    Get technical criteria for a predefined screener.
    """
    # Base technical criteria for all screeners
    return [
        {
            "indicator": "RSI",
            "parameters": {"period": 14},
            "condition": {"operator": "<", "value": 75},
            "weight": 0.5,
            "required": False
        }
    ]


def _get_available_screeners():
    """
    Get list of available predefined screeners.
    """
    try:
        import json
        from pathlib import Path

        config_path = Path(__file__).resolve().parents[4] / "config" / "screener" / "fmp_screener_criteria.json"

        with open(config_path, 'r') as f:
            fmp_config = json.load(f)

        return list(fmp_config.get("predefined_strategies", {}).keys())

    except Exception as e:
        _logger.error("Error loading available screeners: %s", e)
        return []
