import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from typing import Any, Dict, List
from src.frontend.telegram.command_parser import ParsedCommand
from src.common import get_ohlcv
from src.common.fundamentals import get_fundamentals
from src.common.technicals import calculate_technicals_from_df
from src.model.telegram_bot import TickerAnalysis
from src.frontend.telegram import db
from src.common.ticker_analyzer import format_ticker_report

from src.notification.logger import setup_logger
logger = setup_logger("telegram_screener_bot")

def handle_command(parsed: ParsedCommand) -> Dict[str, Any]:
    """
    Main business logic handler. Dispatches based on command and parameters.
    Returns a dict with result/status/data for notification manager.
    """
    if parsed.command == "report":
        return handle_report(parsed)
    elif parsed.command == "help":
        return handle_help(parsed)
    elif parsed.command == "info":
        return handle_info(parsed)
    elif parsed.command == "admin":
        return handle_admin(parsed)
    elif parsed.command == "alerts":
        return handle_alerts(parsed)
    elif parsed.command == "schedules":
        return handle_schedules(parsed)
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
        logger.error("Error: %s", e, exc_info=True)
        return {"status": "error", "message": f"Error generating help: {str(e)}"}


def is_admin_user(telegram_user_id: str) -> bool:
    """Check if user is an admin."""
    db.init_db()
    status = db.get_user_status(telegram_user_id)
    return status and status.get("is_admin", False)


def handle_report(parsed: ParsedCommand) -> Dict[str, Any]:
    """
    Business logic for /report command.
    For each ticker:
      - Use analyze_ticker_business for unified analysis logic
      - Use format_ticker_report to generate message and chart
    """
    args = parsed.args
    tickers_raw = args.get("tickers")
    if isinstance(tickers_raw, str):
        tickers = [tickers_raw]
    elif isinstance(tickers_raw, list):
        tickers = tickers_raw
    else:
        tickers = parsed.positionals
    if not tickers:
        return {"status": "error", "title": "Report Error", "message": "No tickers specified"}
    period = args.get("period") or "2y"
    interval = args.get("interval") or "1d"
    provider = args.get("provider")
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
        "email": args.get("email", False),
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
            provider=provider or ("yf" if len(ticker) < 5 else "bnc"),
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
            provider=provider or ("yf" if len(ticker) < 5 else "bnc"),
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
        language = status["language"] or "(not set)"
        return {
            "status": "ok",
            "title": "Your Info",
            "message": f"Email: {email}\nVerified: {verified}\nLanguage: {language}"
        }
    else:
        return {
            "status": "ok",
            "title": "Your Info",
            "message": "Email: (not set)\nVerified: No\nLanguage: (not set)"
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

        # Check if user is admin
        if not is_admin_user(telegram_user_id):
            return {"status": "error", "message": "Access denied. Admin privileges required."}

        # Get action and parameters from positionals
        action = parsed.positionals[0] if len(parsed.positionals) > 0 else None
        params = parsed.positionals[1:] if len(parsed.positionals) > 1 else []

        if not action:
            return {
                "status": "error",
                "title": "Admin Help",
                "message": ("Available admin commands:\n"
                           "/admin users - List all users\n"
                           "/admin listusers - List users with emails\n"
                           "/admin resetemail USER_ID - Reset user email\n"
                           "/admin verify USER_ID - Manually verify user\n"
                           "/admin setlimit alerts N [USER_ID] - Set alert limits\n"
                           "/admin setlimit schedules N [USER_ID] - Set schedule limits\n"
                           "/admin broadcast MESSAGE - Send broadcast message")
            }

        if action == "users":
            return handle_admin_users()
        elif action == "listusers":
            return handle_admin_listusers()
        elif action == "resetemail" and len(params) >= 1:
            return handle_admin_resetemail(params[0])
        elif action == "verify" and len(params) >= 1:
            return handle_admin_verify(params[0])
        elif action == "setlimit" and len(params) >= 2:
            return handle_admin_setlimit(params[0], params[1], params[2] if len(params) > 2 else None)
        elif action == "broadcast" and len(params) >= 1:
            return handle_admin_broadcast(" ".join(params))
        else:
            return {"status": "error", "message": f"Unknown admin command: {action}"}

    except Exception as e:
        logger.error("Error in admin command: %s", e, exc_info=True)
        return {"status": "error", "message": f"Error processing admin command: {str(e)}"}


def handle_admin_users() -> Dict[str, Any]:
    """List all registered users."""
    try:
        users = db.list_users()
        if not users:
            return {"status": "ok", "title": "Users", "message": "No users registered."}

        user_list = []
        for user in users:
            status = "✅ Verified" if user.get("verified") else "❌ Unverified"
            admin = "👑 Admin" if user.get("is_admin") else "👤 User"
            email = user.get("email", "(no email)")
            user_list.append(f"{admin} {user['telegram_user_id']}: {email} - {status}")

        message = f"Total users: {len(users)}\n\n" + "\n".join(user_list)
        return {"status": "ok", "title": "All Users", "message": message}

    except Exception as e:
        logger.error("Error listing users: %s", e, exc_info=True)
        return {"status": "error", "message": f"Error listing users: {str(e)}"}


def handle_admin_listusers() -> Dict[str, Any]:
    """List users in simple format."""
    try:
        users = db.list_users()
        if not users:
            return {"status": "ok", "title": "Users", "message": "No users registered."}

        user_list = []
        for user in users:
            email = user.get("email", "(no email)")
            user_list.append(f"{user['telegram_user_id']} - {email}")

        message = "\n".join(user_list)
        return {"status": "ok", "title": "User List", "message": message}

    except Exception as e:
        logger.error("Error listing users: %s", e, exc_info=True)
        return {"status": "error", "message": f"Error listing users: {str(e)}"}


def handle_admin_resetemail(user_id: str) -> Dict[str, Any]:
    """Reset a user's email."""
    try:
        db.init_db()
        user_status = db.get_user_status(user_id)
        if not user_status:
            return {"status": "error", "message": f"User {user_id} not found."}

        # Reset email and verification status
        db.update_user_email(user_id, None)
        db.update_user_verification(user_id, False)

        return {
            "status": "ok",
            "title": "Email Reset",
            "message": f"Email reset for user {user_id}. User must re-register their email."
        }

    except Exception as e:
        logger.error("Error resetting email: %s", e, exc_info=True)
        return {"status": "error", "message": f"Error resetting email: {str(e)}"}


def handle_admin_verify(user_id: str) -> Dict[str, Any]:
    """Manually verify a user's email."""
    try:
        db.init_db()
        user_status = db.get_user_status(user_id)
        if not user_status:
            return {"status": "error", "message": f"User {user_id} not found."}

        if not user_status.get("email"):
            return {"status": "error", "message": f"User {user_id} has no email to verify."}

        # Manually verify the user
        db.update_user_verification(user_id, True)

        return {
            "status": "ok",
            "title": "User Verified",
            "message": f"User {user_id} ({user_status['email']}) has been manually verified."
        }

    except Exception as e:
        logger.error("Error verifying user: %s", e, exc_info=True)
        return {"status": "error", "message": f"Error verifying user: {str(e)}"}


def handle_admin_setlimit(limit_type: str, limit_value: str, user_id: str = None) -> Dict[str, Any]:
    """Set limits for alerts or schedules."""
    try:
        if limit_type not in ["alerts", "schedules"]:
            return {"status": "error", "message": "Limit type must be 'alerts' or 'schedules'"}

        try:
            limit = int(limit_value)
            if limit < 0:
                raise ValueError("Limit must be non-negative")
        except ValueError:
            return {"status": "error", "message": "Limit value must be a non-negative integer"}

        db.init_db()

        if user_id:
            # Set user-specific limit
            user_status = db.get_user_status(user_id)
            if not user_status:
                return {"status": "error", "message": f"User {user_id} not found."}

            if limit_type == "alerts":
                db.set_user_max_alerts(user_id, limit)
            else:
                db.set_user_max_schedules(user_id, limit)

            return {
                "status": "ok",
                "title": "Limit Set",
                "message": f"Set max {limit_type} to {limit} for user {user_id}"
            }
        else:
            # Set global default limit
            db.set_global_setting(f"default_max_{limit_type}", str(limit))

            return {
                "status": "ok",
                "title": "Global Limit Set",
                "message": f"Set global default max {limit_type} to {limit}"
            }

    except Exception as e:
        logger.error("Error setting limit: %s", e, exc_info=True)
        return {"status": "error", "message": f"Error setting limit: {str(e)}"}


def handle_admin_broadcast(message_text: str) -> Dict[str, Any]:
    """Send broadcast message to all users."""
    try:
        db.init_db()
        users = db.list_users()
        if not users:
            return {"status": "error", "message": "No users to broadcast to."}

        # Store broadcast for processing by notification system
        # This will be handled by the notification manager
        return {
            "status": "ok",
            "title": "Broadcast Scheduled",
            "message": f"Broadcast message scheduled for {len(users)} users.",
            "broadcast": {
                "message": message_text,
                "user_count": len(users),
                "users": [user["telegram_user_id"] for user in users]
            }
        }

    except Exception as e:
        logger.error("Error scheduling broadcast: %s", e, exc_info=True)
        return {"status": "error", "message": f"Error scheduling broadcast: {str(e)}"}


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

        # Check if user is verified
        user_status = db.get_user_status(telegram_user_id)
        if not user_status or not user_status.get("verified"):
            return {"status": "error", "message": "Please verify your email first using /register and /verify commands."}

        # Get action and parameters from positionals
        action = parsed.positionals[0] if len(parsed.positionals) > 0 else None
        params = parsed.positionals[1:] if len(parsed.positionals) > 1 else []

        if not action:
            # List all alerts for user
            return handle_alerts_list(telegram_user_id)

        if action == "add" and len(params) >= 3:
            ticker, price, condition = params[0], params[1], params[2]
            return handle_alerts_add(telegram_user_id, ticker, price, condition)
        elif action == "edit" and len(params) >= 1:
            alert_id = params[0]
            new_price = params[1] if len(params) > 1 else None
            new_condition = params[2] if len(params) > 2 else None
            return handle_alerts_edit(telegram_user_id, alert_id, new_price, new_condition)
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
                           "/alerts add TICKER PRICE above|below - Add alert\n"
                           "/alerts edit ALERT_ID [PRICE] [CONDITION] - Edit alert\n"
                           "/alerts delete ALERT_ID - Delete alert\n"
                           "/alerts pause ALERT_ID - Pause alert\n"
                           "/alerts resume ALERT_ID - Resume alert")
            }

    except Exception as e:
        logger.error("Error in alerts command: %s", e, exc_info=True)
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
            alert_list.append(
                f"#{alert['id']}: {alert['ticker']} {alert['condition']} ${alert['price']:.2f} - {status}"
            )

        message = f"Your alerts ({len(alerts)}):\n\n" + "\n".join(alert_list)
        return {"status": "ok", "title": "Your Alerts", "message": message}

    except Exception as e:
        logger.error("Error listing alerts: %s", e, exc_info=True)
        return {"status": "error", "message": f"Error listing alerts: {str(e)}"}


def handle_alerts_add(telegram_user_id: str, ticker: str, price_str: str, condition: str) -> Dict[str, Any]:
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

        # Add the alert
        alert_id = db.add_alert(telegram_user_id, ticker.upper(), price, condition.lower())

        return {
            "status": "ok",
            "title": "Alert Added",
            "message": f"Alert #{alert_id} created: {ticker.upper()} {condition.lower()} ${price:.2f}"
        }

    except Exception as e:
        logger.error("Error adding alert: %s", e, exc_info=True)
        return {"status": "error", "message": f"Error adding alert: {str(e)}"}


def handle_alerts_edit(telegram_user_id: str, alert_id_str: str, new_price_str: str = None, new_condition: str = None) -> Dict[str, Any]:
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

        if not updates:
            return {"status": "error", "message": "No updates provided. Specify new price and/or condition."}

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
        logger.error("Error editing alert: %s", e, exc_info=True)
        return {"status": "error", "message": f"Error editing alert: {str(e)}"}


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
        logger.error("Error deleting alert: %s", e, exc_info=True)
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
        logger.error("Error pausing alert: %s", e, exc_info=True)
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
        logger.error("Error resuming alert: %s", e, exc_info=True)
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

        # Check if user is verified
        user_status = db.get_user_status(telegram_user_id)
        if not user_status or not user_status.get("verified"):
            return {"status": "error", "message": "Please verify your email first using /register and /verify commands."}

        # Get action and parameters from positionals
        action = parsed.positionals[0] if len(parsed.positionals) > 0 else None
        params = parsed.positionals[1:] if len(parsed.positionals) > 1 else []

        if not action:
            # List all schedules for user
            return handle_schedules_list(telegram_user_id)

        if action == "add" and len(params) >= 2:
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
        else:
            return {
                "status": "error",
                "title": "Schedules Help",
                "message": ("Available schedule commands:\n"
                           "/schedules - List all schedules\n"
                           "/schedules add TICKER TIME [flags] - Add schedule\n"
                           "/schedules edit SCHEDULE_ID [TIME] - Edit schedule\n"
                           "/schedules delete SCHEDULE_ID - Delete schedule\n"
                           "/schedules pause SCHEDULE_ID - Pause schedule\n"
                           "/schedules resume SCHEDULE_ID - Resume schedule\n\n"
                           "Flags: -email -indicators=RSI,MACD -period=1y -interval=1d -provider=yf")
            }

    except Exception as e:
        logger.error("Error in schedules command: %s", e, exc_info=True)
        return {"status": "error", "message": f"Error processing schedules command: {str(e)}"}


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
            period = schedule.get("period", "daily")
            schedule_list.append(
                f"#{schedule['id']}: {schedule['ticker']} at {schedule['scheduled_time']} ({period}) {email_flag} - {status}"
            )

        message = f"Your scheduled reports ({len(schedules)}):\n\n" + "\n".join(schedule_list)
        return {"status": "ok", "title": "Your Schedules", "message": message}

    except Exception as e:
        logger.error("Error listing schedules: %s", e, exc_info=True)
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

        # Add the schedule
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
        logger.error("Error adding schedule: %s", e, exc_info=True)
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
        logger.error("Error editing schedule: %s", e, exc_info=True)
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
        logger.error("Error deleting schedule: %s", e, exc_info=True)
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
        logger.error("Error pausing schedule: %s", e, exc_info=True)
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
        logger.error("Error resuming schedule: %s", e, exc_info=True)
        return {"status": "error", "message": f"Error resuming schedule: {str(e)}"}


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
        logger.info("User feedback", extra={
            "user_id": telegram_user_id,
            "feedback": feedback,
            "type": "feedback"
        })

        # Store feedback in database for admin panel
        # This would be processed by admin panel or notification system

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
        logger.error("Error processing feedback: %s", e, exc_info=True)
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
        logger.info("Feature request", extra={
            "user_id": telegram_user_id,
            "feature_request": feature_request,
            "type": "feature_request"
        })

        # Store feature request in database for admin panel
        # This would be processed by admin panel or notification system

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
        logger.error("Error processing feature request: %s", e, exc_info=True)
        return {"status": "error", "message": f"Error processing feature request: {str(e)}"}