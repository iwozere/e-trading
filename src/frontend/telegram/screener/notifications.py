import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.frontend.telegram.screener.business_logic import handle_command
from src.frontend.telegram.command_parser import ParsedCommand, parse_command
from src.frontend.telegram import db
from src.notification.logger import setup_logger
_logger = setup_logger(__name__)


async def process_report_notifications(result, notification_manager, message, user_email):
    channels = ["telegram"]
    if result.get("email", False):
        channels.append("email")

    _logger.info("Processing %d reports for channels: %s", len(result["reports"]), channels)

    # Email notifications
    if "email" in channels:
        for report in result["reports"]:
            attachments = None
            if report.get("chart_bytes"):
                attachments = {f"{report['ticker']}_chart.png": report["chart_bytes"]}
            if report.get("error"):
                await notification_manager.send_notification(
                    notification_type="ERROR",
                    title=f"Report Error for {report['ticker']}",
                    message=f"No data for {report['ticker']}: {report['error']}",
                    priority="NORMAL",
                    channels=["email"],
                    email_receiver=user_email
                )
            else:
                await notification_manager.send_notification(
                    notification_type="INFO",
                    title=f"Report for {report['ticker']}",
                    message=report["message"],
                    attachments=attachments,
                    priority="NORMAL",
                    channels=["email"],
                    email_receiver=user_email
                )
    # Telegram notifications - Send directly to avoid queue issues
    _logger.info("Starting Telegram notifications for %d reports", len(result["reports"]))
    for i, report in enumerate(result["reports"]):
        _logger.info("Sending Telegram notification %d/%d for ticker: %s", i+1, len(result["reports"]), report.get("ticker"))
        if "telegram" in channels:
            attachments = None
            if report.get("chart_bytes"):
                attachments = {f"{report['ticker']}_chart.png": report["chart_bytes"]}

            try:
                if report.get("error"):
                    # Send error notification directly
                    await notification_manager.channels["telegram"].send(
                        notification_manager._create_notification(
                            notification_type="ERROR",
                            title=f"Report Error for {report['ticker']}",
                            message=f"No data for {report['ticker']}: {report['error']}",
                            data={
                                "channels": ["telegram"],
                                "telegram_chat_id": message.chat.id,
                                "reply_to_message_id": message.message_id
                            }
                        )
                    )
                    _logger.info("Successfully sent error notification for %s", report['ticker'])
                else:
                    # Send success notification directly
                    await notification_manager.channels["telegram"].send(
                        notification_manager._create_notification(
                            notification_type="INFO",
                            title=f"Report for {report['ticker']}",
                            message=report["message"],
                            data={
                                "channels": ["telegram"],
                                "telegram_chat_id": message.chat.id,
                                "reply_to_message_id": message.message_id,
                                "attachments": attachments
                            }
                        )
                    )
                    _logger.info("Successfully sent Telegram notification for %s", report['ticker'])
            except Exception as e:
                _logger.exception("Error sending notification for %s", report['ticker'])
                # Try fallback without attachment
                try:
                    await notification_manager.channels["telegram"].send(
                        notification_manager._create_notification(
                            notification_type="INFO",
                            title=f"Report for {report['ticker']}",
                            message=report["message"] + "\n\n[Chart could not be sent due to an error.]",
                            data={
                                "channels": ["telegram"],
                                "telegram_chat_id": message.chat.id,
                                "reply_to_message_id": message.message_id
                            }
                        )
                    )
                    _logger.info("Successfully sent fallback Telegram notification for %s", report['ticker'])
                except Exception as e2:
                    _logger.exception("Error sending fallback notification for %s", report['ticker'])


def send_screener_email(email: str, report, config):
    """Send screener results via email."""
    try:
        from src.frontend.telegram.screener.enhanced_screener import EnhancedScreener

        subject = f"Screener Results - {config.list_type.replace('_', ' ').title()}"

        # Format email content using the enhanced screener's formatting
        enhanced_screener = EnhancedScreener()
        content = enhanced_screener.format_enhanced_telegram_message(report, config)

        # Convert markdown to HTML for email
        content_html = content.replace('**', '<strong>').replace('*', '<em>')
        content_html = content_html.replace('\n', '<br>')

        # Send email using existing email infrastructure
        from src.notification.emailer import EmailNotifier
        email_notifier = EmailNotifier()
        email_notifier.send_email(email, subject, content_html)

        _logger.info("Screener results sent via email to %s", email)

    except Exception as e:
        _logger.exception("Error sending screener email to %s", email)
        raise

async def process_report_command(message, telegram_user_id, args, notification_manager):
    try:
        # Use the proper command parser instead of manually creating ParsedCommand
        parsed = parse_command(message.text)
        # Add the telegram_user_id to the args
        parsed.args["telegram_user_id"] = telegram_user_id
        result = handle_command(parsed)
        user_email = result.get("user_email")
        if result.get("status") == "ok" and "reports" in result:
            await process_report_notifications(result, notification_manager, message, user_email)
        else:
            channels = ["telegram"]
            if result.get("email", False):
                channels.append("email")
            await notification_manager.send_notification(
                notification_type="ERROR",
                title=result.get("title", "Report"),
                message=result.get("message", "No message"),
                priority="NORMAL",
                channels=channels,
                telegram_chat_id=message.chat.id,
                reply_to_message_id=message.message_id,
                email_receiver=user_email if "email" in channels else None
            )
    except Exception as e:
        _logger.exception("Error in report command")
        return {
            "status": "error",
            "message": f"Error processing report command: {str(e)}"
        }

async def process_help_command(message, telegram_user_id, message_text=None, notification_manager=None):
    try:
        # Parse command to check for -email flag
        if message_text:
            parsed = parse_command(message_text)
            email_flag = parsed.args.get("email", False)
        else:
            email_flag = False

        # Get user info for email
        user_email = None
        if email_flag:
            user_status = db.get_user_status(telegram_user_id)
            if user_status and user_status.get("verified"):
                user_email = user_status.get("email")
            else:
                email_flag = False  # Don't send email if user not verified

        # Get help content
        help_content = get_comprehensive_help_content()

        channels = ["telegram"]
        if email_flag and user_email:
            channels.append("email")

        # Send Telegram notification
        await notification_manager.send_notification(
            notification_type="INFO",
            title="Alkotrader Bot - Complete Help Guide",
            message=help_content,
            priority="NORMAL",
            channels=["telegram"],
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )

        # Send email if requested and user is verified
        if email_flag and user_email:
            await notification_manager.send_notification(
                notification_type="INFO",
                title="Alkotrader Bot - Complete Help Guide",
                message=help_content,
                priority="NORMAL",
                channels=["email"],
                email_receiver=user_email
            )

    except Exception as e:
        _logger.exception("Error in help command")
        return {
            "status": "error",
            "message": f"Error processing help command: {str(e)}"
        }

def get_comprehensive_help_content():
    """Get comprehensive help content for the bot."""
    return """
🤖 ALKOTRADER BOT - COMPLETE COMMAND GUIDE

📋 QUICK START
1. Send /start to begin
2. Register with /register your@email.com
3. Verify your email with the code sent to you
4. Request approval with /request_approval
5. Start using the bot's features!

📊 REPORT COMMANDS
/report TICKER [flags] - Generate comprehensive analysis
Examples:
• /report AAPL - Basic report for Apple
• /report TSLA -email - Report sent to email
• /report BTCUSDT -indicators=RSI,MACD -period=6mo
• /report MSFT -interval=1h -provider=yf

JSON Configuration (Advanced):
• /report -config='{"report_type":"analysis","tickers":["AAPL","MSFT"],"period":"1y","indicators":["RSI","MACD"],"email":true}'
• /report -config='{"report_type":"analysis","tickers":["TSLA"],"period":"6mo","interval":"1h","indicators":["RSI","MACD","BollingerBands"],"include_fundamentals":false}'

Flags:
• -email - Send report to your verified email
• -indicators=RSI,MACD,BollingerBands - Specify indicators
• -period=1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
• -interval=1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
• -provider=yf,alpha_vantage,polygon
• -config=JSON_STRING - Use JSON configuration for advanced options

🚨 ALERT COMMANDS
Price Alerts:
• /alerts add TICKER PRICE CONDITION [flags]
• /alerts add AAPL 150.00 above -email
• /alerts add BTCUSDT 50000 below

Indicator Alerts (Advanced):
• /alerts add_indicator TICKER CONFIG_JSON [flags]
• /alerts add_indicator AAPL '{"type":"indicator","indicator":"RSI","parameters":{"period":14},"condition":{"operator":"<","value":30},"alert_action":"BUY","timeframe":"15m"}' -email

Alert Management:
• /alerts - List all alerts
• /alerts edit ALERT_ID [PRICE] [CONDITION] [flags]
• /alerts delete ALERT_ID
• /alerts pause ALERT_ID
• /alerts resume ALERT_ID

⏰ SCHEDULE COMMANDS
Simple Schedules:
• /schedules add TICKER TIME [flags]
• /schedules add AAPL 09:00 -email
• /schedules add TSLA 16:30 -indicators=RSI,MACD

JSON Schedules (Advanced):
• /schedules add_json CONFIG_JSON
• /schedules add_json '{"type":"report","ticker":"AAPL","scheduled_time":"09:00","period":"1y","interval":"1d","email":true}'

Screener Schedules:
• /schedules screener LIST_TYPE [TIME] [flags]
• /schedules screener us_small_cap 09:00 -email
• /schedules screener us_large_cap -indicators=PE,PB,ROE

Schedule Management:
• /schedules - List all schedules
• /schedules edit SCHEDULE_ID [TIME] [flags]
• /schedules delete SCHEDULE_ID
• /schedules pause SCHEDULE_ID
• /schedules resume SCHEDULE_ID

🔧 UTILITY COMMANDS
Account Management:
• /start - Start the bot
• /help - Show this help message
• /register your@email.com - Register email
• /verify CODE - Verify email
• /request_approval - Request admin approval
• /info - Show account information

📈 TECHNICAL INDICATORS
Available: RSI, MACD, Bollinger Bands, SMA, EMA, ADX, ATR, Stochastic, WilliamsR
Fundamental: P/E, P/B, ROE, ROA, Debt/Equity, Current Ratio, EPS, Revenue, Profit Margin

⚙️ ADVANCED FEATURES
JSON configurations for complex reports, alerts and schedules
Multiple timeframe support (5m, 15m, 1h, 4h, 1d)
Fundamental screening for undervalued stocks
Email notifications for all features

📞 SUPPORT
• Use /help command in the bot
• Contact admin for account issues
• Check admin panel for detailed help

🎯 PRO TIPS:
• Use -email flag for reports and alerts
• Combine multiple indicators for better analysis
• Set up daily schedules for regular monitoring
• Use fundamental screener for stock discovery
• Configure alerts with specific timeframes
• Use JSON configuration for complex report setups

For more detailed help, visit the admin panel help page!
"""

async def process_info_command(message, telegram_user_id, notification_manager):
    try:
        # Parse command to check for -email flag
        parsed = parse_command(message.text)
        email_flag = parsed.args.get("email", False)

        # Get user info for email
        user_email = None
        if email_flag:
            user_status = db.get_user_status(telegram_user_id)
            if user_status and user_status.get("verified"):
                user_email = user_status.get("email")
            else:
                email_flag = False  # Don't send email if user not verified

        # Add telegram_user_id to parsed command arguments
        parsed.args["telegram_user_id"] = telegram_user_id

        # Get info content
        result = handle_command(parsed)

        channels = ["telegram"]
        if email_flag and user_email:
            channels.append("email")

        # Send Telegram notification
        await notification_manager.send_notification(
            notification_type="INFO" if result["status"] == "ok" else "ERROR",
            title=result.get("title", "Info"),
            message=result.get("message", "No message"),
            priority="NORMAL",
            channels=["telegram"],
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )

        # Send email if requested and user is verified
        if email_flag and user_email:
            await notification_manager.send_notification(
                notification_type="INFO" if result["status"] == "ok" else "ERROR",
                title=result.get("title", "Info"),
                message=result.get("message", "No message"),
                priority="NORMAL",
                channels=["email"],
                email_receiver=user_email
            )

    except Exception as e:
        _logger.exception("Error in info command")
        return {
            "status": "error",
            "message": f"Error processing info command: {str(e)}"
        }

async def process_register_command(message, telegram_user_id, args, notification_manager):
    try:
        email = args[1].strip() if len(args) > 1 else None
        language = args[2].strip().lower() if len(args) > 2 else None
        parsed = ParsedCommand(command="register", args={"telegram_user_id": telegram_user_id, "email": email, "language": language})
        result = handle_command(parsed)
        channels = ["telegram"]

        # Send Telegram notification
        await notification_manager.send_notification(
            notification_type="INFO" if result["status"] == "ok" else "ERROR",
            title=result.get("title", "Register"),
            message=result.get("message", "No message"),
            priority="NORMAL",
            channels=["telegram"],
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )

        # Send verification email if registration was successful
        if result["status"] == "ok" and "email_verification" in result:
            verification_info = result["email_verification"]
            await notification_manager.send_notification(
                notification_type="INFO",
                title="Your Alkotrader Email Verification Code",
                message=f"Hello,\n\nThank you for registering your email with the Alkotrader Telegram bot.\n\nYour verification code is: {verification_info['code']}\n\nThis code is valid for 1 hour. If you did not request this, please ignore this email.\n\nBest regards,\nAlkotrader Team",
                priority="NORMAL",
                channels=["email"],
                email_receiver=verification_info["email"]
            )

    except Exception as e:
        _logger.exception("Error in register command")
        return {
            "status": "error",
            "message": f"Error processing register command: {str(e)}"
        }

async def process_verify_command(message, telegram_user_id, args, notification_manager):
    try:
        code = args[1] if len(args) > 1 else None
        parsed = ParsedCommand(command="verify", args={"telegram_user_id": telegram_user_id, "code": code})
        result = handle_command(parsed)
        channels = ["telegram"]
        if result.get("email", False):
            channels.append("email")
        await notification_manager.send_notification(
            notification_type="INFO" if result["status"] == "ok" else "ERROR",
            title=result.get("title", "Verify"),
            message=result.get("message", "No message"),
            priority="NORMAL",
            channels=channels,
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )
    except Exception as e:
        _logger.exception("Error in verify command: ")
        await notification_manager.send_notification(
            notification_type="ERROR",
            title="Verify Command Error",
            message="An error occurred while processing your request.",
            priority="CRITICAL",
            channels=["telegram"],
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )

async def process_request_approval_command(message, telegram_user_id, args, notification_manager):
    """Process /request_approval command"""
    try:
        # Parse command
        command_text = " ".join(args)
        parsed = parse_command(command_text)
        parsed.args["telegram_user_id"] = telegram_user_id

        # Process command
        result = handle_command(parsed)

        # Send response to user
        if result["status"] == "ok":
            await notification_manager.send_notification(
                notification_type="INFO",
                title=result.get("title", "Approval Request"),
                message=result["message"],
                priority="NORMAL",
                channels=["telegram"],
                telegram_chat_id=message.chat.id,
                reply_to_message_id=message.message_id
            )

            # Notify admins about the approval request
            if result.get("notify_admins"):
                await notification_manager.send_notification(
                    notification_type="INFO",
                    title="New Approval Request",
                    message=f"User {result['user_id']} ({result['email']}) has requested approval for restricted features.",
                    priority="HIGH",
                    channels=["telegram"],
                    telegram_chat_id=message.chat.id  # Send to admin chat
                )
        else:
            await notification_manager.send_notification(
                notification_type="ERROR",
                title="Approval Request Error",
                message=result["message"],
                priority="NORMAL",
                channels=["telegram"],
                telegram_chat_id=message.chat.id,
                reply_to_message_id=message.message_id
            )

    except Exception as e:
        _logger.exception("Error processing approval request: ")
        await notification_manager.send_notification(
            notification_type="ERROR",
            title="Error",
            message="An error occurred while processing your approval request.",
            priority="NORMAL",
            channels=["telegram"],
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )

async def process_language_command(message, telegram_user_id, args, notification_manager):
    try:
        lang = args[1].strip().lower() if len(args) > 1 else None
        parsed = ParsedCommand(command="language", args={"telegram_user_id": telegram_user_id, "language": lang})
        result = handle_command(parsed)
        channels = ["telegram"]
        if result.get("email", False):
            channels.append("email")
        await notification_manager.send_notification(
            notification_type="INFO" if result["status"] == "ok" else "ERROR",
            title=result.get("title", "Language"),
            message=result.get("message", "No message"),
            priority="NORMAL",
            channels=channels,
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )
    except Exception as e:
        _logger.exception("Error in language command: ")
        await notification_manager.send_notification(
            notification_type="ERROR",
            title="Language Command Error",
            message="An error occurred while processing your request.",
            priority="CRITICAL",
            channels=["telegram"],
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )

async def process_admin_command(message, telegram_user_id, args, notification_manager):
    try:
        # Use the proper command parser instead of manually creating ParsedCommand
        parsed = parse_command(message.text)
        # Add the telegram_user_id to the args
        parsed.args["telegram_user_id"] = telegram_user_id
        result = handle_command(parsed)
        channels = ["telegram"]
        if result.get("email", False):
            channels.append("email")

        # Send response to admin
        await notification_manager.send_notification(
            notification_type="INFO" if result["status"] == "ok" else "ERROR",
            title=result.get("title", "Admin"),
            message=result.get("message", "No message"),
            priority="NORMAL",
            channels=["telegram"],
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )

        # Handle broadcast functionality
        if result["status"] == "ok" and "broadcast" in result:
            broadcast_info = result["broadcast"]
            broadcast_message = broadcast_info["message"]
            user_ids = broadcast_info["users"]

            # Send broadcast to all users
            for user_id in user_ids:
                try:
                    await notification_manager.send_notification(
                        notification_type="INFO",
                        title="Alkotrader Announcement",
                        message=broadcast_message,
                        priority="NORMAL",
                        channels=["telegram"],
                        telegram_chat_id=int(user_id)
                    )
                except Exception as e:
                    _logger.error("Error sending broadcast to user %s: %s", user_id, e)

    except Exception as e:
        _logger.exception("Error in admin command: ")
        await notification_manager.send_notification(
            notification_type="ERROR",
            title="Admin Command Error",
            message="An error occurred while processing your request.",
            priority="CRITICAL",
            channels=["telegram"],
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )

async def process_alerts_command(message, telegram_user_id, args, notification_manager):
    try:
        # Use the proper command parser instead of manually creating ParsedCommand
        parsed = parse_command(message.text)
        # Add the telegram_user_id to the args
        parsed.args["telegram_user_id"] = telegram_user_id
        result = handle_command(parsed)
        channels = ["telegram"]
        if result.get("email", False):
            channels.append("email")
        await notification_manager.send_notification(
            notification_type="INFO" if result["status"] == "ok" else "ERROR",
            title=result.get("title", "Alerts"),
            message=result.get("message", "No message"),
            priority="NORMAL",
            channels=channels,
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )
    except Exception as e:
        _logger.exception("Error in alerts command: ")
        await notification_manager.send_notification(
            notification_type="ERROR",
            title="Alerts Command Error",
            message="An error occurred while processing your request.",
            priority="CRITICAL",
            channels=["telegram"],
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )

async def process_schedules_command(message, telegram_user_id, args, notification_manager):
    try:
        # Use the proper command parser instead of manually creating ParsedCommand
        parsed = parse_command(message.text)
        # Add the telegram_user_id to the args
        parsed.args["telegram_user_id"] = telegram_user_id
        result = handle_command(parsed)
        channels = ["telegram"]
        if result.get("email", False):
            channels.append("email")
        await notification_manager.send_notification(
            notification_type="INFO" if result["status"] == "ok" else "ERROR",
            title=result.get("title", "Schedules"),
            message=result.get("message", "No message"),
            priority="NORMAL",
            channels=channels,
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )
    except Exception as e:
        _logger.exception("Error in schedules command: ")
        await notification_manager.send_notification(
            notification_type="ERROR",
            title="Schedules Command Error",
            message="An error occurred while processing your request.",
            priority="CRITICAL",
            channels=["telegram"],
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )

async def process_feedback_command(message, telegram_user_id, args, notification_manager):
    try:
        feedback_text = args[1] if len(args) > 1 else None
        parsed = ParsedCommand(command="feedback", args={"telegram_user_id": telegram_user_id, "feedback": feedback_text})
        result = handle_command(parsed)
        channels = ["telegram"]
        if result.get("email", False):
            channels.append("email")
        await notification_manager.send_notification(
            notification_type="INFO" if result["status"] == "ok" else "ERROR",
            title=result.get("title", "Feedback"),
            message=result.get("message", "No message"),
            priority="NORMAL",
            channels=channels,
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )
    except Exception as e:
        _logger.exception("Error in feedback command: ")
        await notification_manager.send_notification(
            notification_type="ERROR",
            title="Feedback Command Error",
            message="An error occurred while processing your request.",
            priority="CRITICAL",
            channels=["telegram"],
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )

async def process_feature_command(message, telegram_user_id, args, notification_manager):
    """Process /feature command"""
    try:
        # Parse command
        parsed = parse_command(" ".join(args))
        parsed.args["telegram_user_id"] = telegram_user_id

        # Execute business logic
        result = handle_command(parsed)

        # Send response
        if result["status"] == "ok":
            await message.answer(result["message"])
        else:
            await message.answer(f"❌ {result['message']}")

    except Exception as e:
        _logger.exception("Error processing feature command")
        await message.answer("❌ Error processing feature request. Please try again.")


async def process_screener_command(message, telegram_user_id, args, notification_manager):
    """Process /screener command"""
    try:
        # Parse command
        parsed = parse_command(" ".join(args))
        parsed.args["telegram_user_id"] = telegram_user_id

        # Execute business logic
        result = handle_command(parsed)

        # Send response
        if result["status"] == "success":
            if "report" in result:
                # Send formatted message to Telegram
                await message.answer(result["message"], parse_mode='Markdown')
            else:
                # Email sent confirmation
                await message.answer(result["message"])
        else:
            await message.answer(f"❌ {result['message']}")

    except Exception as e:
        _logger.exception("Error processing screener command")
        await message.answer("❌ Error processing screener request. Please try again.")


async def process_unknown_command(message, telegram_user_id, notification_manager, help_text):
    try:
        parsed = ParsedCommand(command="unknown", args={"telegram_user_id": telegram_user_id, "text": message.text})

        # Create a user-friendly unknown command message
        unknown_command = message.text.split()[0] if message.text else "unknown"
        unknown_message = f"❓ Unknown command: {unknown_command}\n\nI don't recognize this command. Please use /help to see all available commands and their usage."

        await notification_manager.send_notification(
            notification_type="ERROR",
            title="Unknown Command",
            message=unknown_message,
            priority="NORMAL",
            channels=["telegram"],
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )
    except Exception as e:
        _logger.exception("Error in unknown command handler: ")
        await notification_manager.send_notification(
            notification_type="ERROR",
            title="Unknown Command Error",
            message="An error occurred while processing your request.",
            priority="CRITICAL",
            channels=["telegram"],
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )
