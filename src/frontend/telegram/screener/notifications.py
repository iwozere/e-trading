import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.frontend.telegram.screener.business_logic import handle_command
from src.frontend.telegram.command_parser import ParsedCommand, parse_command

logger = setup_logger("telegram_screener_bot")


async def process_report_notifications(result, notification_manager, message, user_email):
    channels = ["telegram"]
    if result.get("email", False):
        channels.append("email")

    logger.info("Processing %d reports for channels: %s", len(result["reports"]), channels)

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
    logger.info("Starting Telegram notifications for %d reports", len(result["reports"]))
    for i, report in enumerate(result["reports"]):
        logger.info("Sending Telegram notification %d/%d for ticker: %s", i+1, len(result["reports"]), report.get("ticker"))
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
                    logger.info("Successfully sent error notification for %s", report['ticker'])
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
                    logger.info("Successfully sent Telegram notification for %s", report['ticker'])
            except Exception as e:
                logger.exception("Error sending notification for %s: ", report['ticker'])
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
                    logger.info("Successfully sent fallback Telegram notification for %s", report['ticker'])
                except Exception as e2:
                    logger.exception("Error sending fallback notification for %s: ", report['ticker'])

            # Add a small delay between notifications to avoid rate limiting
            if i < len(result["reports"]) - 1:  # Don't delay after the last notification
                import asyncio
                logger.info("Waiting 500ms before next notification...")
                await asyncio.sleep(0.5)  # 500ms delay between notifications
                logger.info("Delay completed, continuing with next notification")

    logger.info("Completed processing all %d reports", len(result["reports"]))

    # Wait a bit to ensure all notifications are processed
    import asyncio
    await asyncio.sleep(1.0)
    logger.info("Final wait completed, all notifications should be sent")

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
        logger.exception("Error in report command: ")
        await notification_manager.send_notification(
            notification_type="ERROR",
            title="Report Command Error",
            message="An error occurred while processing your request.",
            priority="CRITICAL",
            channels=["telegram"],
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )

async def process_help_command(message, telegram_user_id, notification_manager):
    try:
        parsed = ParsedCommand(command="help", args={"telegram_user_id": telegram_user_id})
        result = handle_command(parsed)
        channels = ["telegram"]
        if result.get("email", False):
            channels.append("email")
        await notification_manager.send_notification(
            notification_type="INFO" if result["status"] == "ok" else "ERROR",
            title=result.get("title", "Help"),
            message=result.get("help_text", result.get("message", "No message")),
            priority="NORMAL",
            channels=channels,
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )
    except Exception as e:
        logger.exception("Error in help command: ")
        await notification_manager.send_notification(
            notification_type="ERROR",
            title="Help Command Error",
            message="An error occurred while processing your request.",
            priority="CRITICAL",
            channels=["telegram"],
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )

async def process_info_command(message, telegram_user_id, notification_manager):
    try:
        parsed = ParsedCommand(command="info", args={"telegram_user_id": telegram_user_id})
        result = handle_command(parsed)
        channels = ["telegram"]
        if result.get("email", False):
            channels.append("email")
        await notification_manager.send_notification(
            notification_type="INFO" if result["status"] == "ok" else "ERROR",
            title=result.get("title", "Info"),
            message=result.get("message", "No message"),
            priority="NORMAL",
            channels=channels,
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )
    except Exception as e:
        logger.exception("Error in info command: ")
        await notification_manager.send_notification(
            notification_type="ERROR",
            title="Info Command Error",
            message="An error occurred while processing your request.",
            priority="CRITICAL",
            channels=["telegram"],
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )

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
        logger.exception("Error in register command: ")
        await notification_manager.send_notification(
            notification_type="ERROR",
            title="Register Command Error",
            message="An error occurred while processing your request.",
            priority="CRITICAL",
            channels=["telegram"],
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )

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
        logger.exception("Error in verify command: ")
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
        logger.exception("Error processing approval request: ")
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
        logger.exception("Error in language command: ")
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
                    logger.error("Error sending broadcast to user %s: %s", user_id, e)

    except Exception as e:
        logger.exception("Error in admin command: ")
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
        logger.exception("Error in alerts command: ")
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
        logger.exception("Error in schedules command: ")
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
        logger.exception("Error in feedback command: ")
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
    try:
        feature_text = args[1] if len(args) > 1 else None
        parsed = ParsedCommand(command="feature", args={"telegram_user_id": telegram_user_id, "feature": feature_text})
        result = handle_command(parsed)
        channels = ["telegram"]
        if result.get("email", False):
            channels.append("email")
        await notification_manager.send_notification(
            notification_type="INFO" if result["status"] == "ok" else "ERROR",
            title=result.get("title", "Feature Request"),
            message=result.get("message", "No message"),
            priority="NORMAL",
            channels=channels,
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )
    except Exception as e:
        logger.exception("Error in feature command: ")
        await notification_manager.send_notification(
            notification_type="ERROR",
            title="Feature Command Error",
            message="An error occurred while processing your request.",
            priority="CRITICAL",
            channels=["telegram"],
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )

async def process_unknown_command(message, telegram_user_id, notification_manager, help_text):
    try:
        parsed = ParsedCommand(command="unknown", args={"telegram_user_id": telegram_user_id, "text": message.text})
        await notification_manager.send_notification(
            notification_type="ERROR",
            title="Unknown Command",
            message=help_text,
            priority="NORMAL",
            channels=["telegram"],
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )
    except Exception as e:
        logger.exception("Error in unknown command handler: ")
        await notification_manager.send_notification(
            notification_type="ERROR",
            title="Unknown Command Error",
            message="An error occurred while processing your request.",
            priority="CRITICAL",
            channels=["telegram"],
            telegram_chat_id=message.chat.id,
            reply_to_message_id=message.message_id
        )
