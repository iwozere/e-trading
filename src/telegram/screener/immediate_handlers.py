import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.telegram.screener.business_logic import handle_command, get_service_instances
from src.telegram.command_parser import ParsedCommand, parse_command
from src.notification.logger import setup_logger
from src.notification.service.client import MessageType, MessagePriority

_logger = setup_logger(__name__)


async def _send_email_notification_for_command(message, response_text: str, command: str, telegram_user_id: str):
    """
    Send email notification for a command response.

    Args:
        message: Telegram message object
        response_text: The response text to send via email
        command: The command name for logging
        telegram_user_id: Telegram user ID
    """
    try:
        # Get user information for email
        telegram_svc, _ = get_service_instances()
        if telegram_svc:
            user_status = telegram_svc.get_user_status(telegram_user_id)
            if user_status and user_status.get('email'):
                user_email = user_status['email']

                # Import notification client
                from src.notification.service.client import NotificationServiceClient

                # Create notification client if not exists
                notification_client = NotificationServiceClient()

                # Send email notification via notification service
                await notification_client.send_notification(
                    notification_type="telegram_command_response",
                    title=f"Telegram Bot - {command.upper()} Command Response",
                    message=response_text,
                    priority="normal",
                    channels=["email"],
                    email_receiver=user_email,
                    recipient_id=telegram_user_id,
                    data={
                        "command": command,
                        "telegram_user_id": telegram_user_id,
                        "source": "telegram_bot"
                    }
                )
                _logger.info("Email notification sent for %s command to user %s", command, telegram_user_id)
            else:
                # Send Telegram message about missing email
                await message.answer("📧 Email notification requested but no verified email found. Use /register to set up email notifications.")
        else:
            _logger.warning("Telegram service not available for email notification")

    except Exception as e:
        _logger.error("Error sending email notification for %s command: %s", command, e)
        # Don't fail the main command if email fails


async def process_info_command_immediate(telegram_user_id, message):
    """
    Process /info command immediately without notification service.

    Args:
        telegram_user_id: Telegram user ID
        message: Telegram message object

    Returns:
        Dict with command result
    """
    try:
        # Parse command to check for -email flag
        parsed = parse_command(message.text)
        parsed.args["telegram_user_id"] = telegram_user_id

        # Get info content from business logic
        result = await handle_command(parsed)

        # Send response immediately to Telegram
        response_text = result.get("message", "Info retrieved successfully")
        if result["status"] == "ok":
            await message.answer(response_text)
        else:
            response_text = f"❌ {result.get('message', 'Error retrieving info')}"
            await message.answer(response_text)

        # Send email notification if -email flag is present
        if parsed.args.get("email", False):
            await _send_email_notification_for_command(message, response_text, "info", telegram_user_id)

        return result

    except Exception as e:
        _logger.exception("Error in immediate info command")
        error_message = "❌ Error retrieving your information. Please try again."
        await message.answer(error_message)
        return {
            "status": "error",
            "message": f"Error processing info command: {str(e)}"
        }


async def process_register_command_immediate(telegram_user_id, args, message):
    """
    Process /register command immediately without notification service.

    Args:
        telegram_user_id: Telegram user ID
        args: Command arguments
        message: Telegram message object

    Returns:
        Dict with command result
    """
    try:
        email = args[1].strip() if len(args) > 1 else None
        language = args[2].strip().lower() if len(args) > 2 else None
        parsed = ParsedCommand(command="register", args={"telegram_user_id": telegram_user_id, "email": email, "language": language})

        result = await handle_command(parsed)

        # Send response immediately to Telegram
        response_text = result.get("message", "Registration successful")
        if result["status"] == "ok":
            await message.answer(response_text)

            # If email verification is needed, send verification email via notification service
            if "email_verification" in result:
                verification_info = result["email_verification"]
                # Import here to avoid circular imports
                from src.telegram.telegram_bot import notification_client
                if notification_client:
                    await notification_client.send_notification(
                        notification_type=MessageType.INFO,
                        title="Your Alkotrader Email Verification Code",
                        message=f"Hello,\n\nThank you for registering your email with the Alkotrader Telegram bot.\n\nYour verification code is: {verification_info['code']}\n\nThis code is valid for 1 hour. If you did not request this, please ignore this email.\n\nBest regards,\nAlkotrader Team",
                        priority=MessagePriority.NORMAL,
                        channels=["email"],
                        recipient_id=verification_info["email"]
                    )
        else:
            response_text = f"❌ {result.get('message', 'Registration failed')}"
            await message.answer(response_text)

        # Send email notification if -email flag is present (for registration confirmation)
        parsed_cmd = parse_command(message.text)
        if parsed_cmd.args.get("email", False):
            await _send_email_notification_for_command(message, response_text, "register", telegram_user_id)

        return result

    except Exception as e:
        _logger.exception("Error in immediate register command")
        error_message = "❌ Error processing registration. Please try again."
        await message.answer(error_message)
        return {
            "status": "error",
            "message": f"Error processing register command: {str(e)}"
        }


async def process_verify_command_immediate(telegram_user_id, args, message):
    """
    Process /verify command immediately without notification service.

    Args:
        telegram_user_id: Telegram user ID
        args: Command arguments
        message: Telegram message object

    Returns:
        Dict with command result
    """
    try:
        code = args[1] if len(args) > 1 else None
        parsed = ParsedCommand(command="verify", args={"telegram_user_id": telegram_user_id, "code": code})

        result = await handle_command(parsed)

        # Send response immediately to Telegram
        response_text = result.get("message", "Email verification successful")
        if result["status"] == "ok":
            await message.answer(response_text)
        else:
            response_text = f"❌ {result.get('message', 'Email verification failed')}"
            await message.answer(response_text)

        # Send email notification if -email flag is present
        parsed_cmd = parse_command(message.text)
        if parsed_cmd.args.get("email", False):
            await _send_email_notification_for_command(message, response_text, "verify", telegram_user_id)

        return result

    except Exception as e:
        _logger.exception("Error in immediate verify command")
        error_message = "❌ Error processing verification. Please try again."
        await message.answer(error_message)
        return {
            "status": "error",
            "message": f"Error processing verify command: {str(e)}"
        }


async def process_request_approval_command_immediate(telegram_user_id, args, message):
    """
    Process /request_approval command immediately without notification service.

    Args:
        telegram_user_id: Telegram user ID
        args: Command arguments
        message: Telegram message object

    Returns:
        Dict with command result
    """
    try:
        # Parse command
        command_text = " ".join(args)
        parsed = parse_command(command_text)
        parsed.args["telegram_user_id"] = telegram_user_id

        # Process command
        result = await handle_command(parsed)

        # Send response immediately to Telegram
        if result["status"] == "ok":
            await message.answer(result.get("message", "Approval request submitted"))

            # Notify admins about the approval request via notification service
            if result.get("notify_admins"):
                from src.telegram.telegram_bot import notification_client
                if notification_client:
                    await notification_client.send_notification(
                        notification_type=MessageType.INFO,
                        title="New Approval Request",
                        message=f"User {result['user_id']} ({result['email']}) has requested approval for restricted features.",
                        priority=MessagePriority.HIGH,
                        channels=["telegram"],
                        telegram_chat_id=message.chat.id  # Send to admin chat
                    )
        else:
            await message.answer(f"❌ {result.get('message', 'Approval request failed')}")

        return result

    except Exception as e:
        _logger.exception("Error in immediate request approval command")
        error_message = "❌ Error processing approval request. Please try again."
        await message.answer(error_message)
        return {
            "status": "error",
            "message": f"Error processing request approval command: {str(e)}"
        }


async def process_language_command_immediate(telegram_user_id, args, message):
    """
    Process /language command immediately without notification service.

    Args:
        telegram_user_id: Telegram user ID
        args: Command arguments
        message: Telegram message object

    Returns:
        Dict with command result
    """
    try:
        lang = args[1].strip().lower() if len(args) > 1 else None
        parsed = ParsedCommand(command="language", args={"telegram_user_id": telegram_user_id, "language": lang})

        result = await handle_command(parsed)

        # Send response immediately to Telegram
        if result["status"] == "ok":
            await message.answer(result.get("message", "Language updated successfully"))
        else:
            await message.answer(f"❌ {result.get('message', 'Language update failed')}")

        return result

    except Exception as e:
        _logger.exception("Error in immediate language command")
        error_message = "❌ Error updating language. Please try again."
        await message.answer(error_message)
        return {
            "status": "error",
            "message": f"Error processing language command: {str(e)}"
        }


async def process_admin_command_immediate(telegram_user_id, args, message):
    """
    Process /admin command immediately without notification service.

    Args:
        telegram_user_id: Telegram user ID
        args: Command arguments
        message: Telegram message object

    Returns:
        Dict with command result
    """
    try:
        # Use the proper command parser
        parsed = parse_command(message.text)
        parsed.args["telegram_user_id"] = telegram_user_id

        result = await handle_command(parsed)

        # Send response immediately to Telegram
        response_text = result.get("message", "Admin command executed")
        if result["status"] == "ok":
            await message.answer(response_text)

            # Handle broadcast functionality via notification service
            if "broadcast" in result:
                broadcast_info = result["broadcast"]
                broadcast_message = broadcast_info["message"]
                user_ids = broadcast_info["users"]

                # Send broadcast to all users via notification service
                from src.telegram.telegram_bot import notification_client
                if notification_client:
                    for user_id in user_ids:
                        try:
                            await notification_client.send_notification(
                                notification_type=MessageType.INFO,
                                title="Alkotrader Announcement",
                                message=broadcast_message,
                                priority=MessagePriority.NORMAL,
                                channels=["telegram"],
                                telegram_chat_id=int(user_id)
                            )
                        except Exception as e:
                            _logger.error("Error sending broadcast to user %s: %s", user_id, e)
        else:
            response_text = f"❌ {result.get('message', 'Admin command failed')}"
            await message.answer(response_text)

        # Send email notification if -email flag is present
        if parsed.args.get("email", False):
            await _send_email_notification_for_command(message, response_text, "admin", telegram_user_id)

        return result

    except Exception as e:
        _logger.exception("Error in immediate admin command")
        error_message = "❌ Error processing admin command. Please try again."
        await message.answer(error_message)
        return {
            "status": "error",
            "message": f"Error processing admin command: {str(e)}"
        }


async def process_alerts_command_immediate(telegram_user_id, args, message):
    """
    Process /alerts command immediately without notification service.

    Args:
        telegram_user_id: Telegram user ID
        args: Command arguments
        message: Telegram message object

    Returns:
        Dict with command result
    """
    try:
        # Use the proper command parser
        parsed = parse_command(message.text)
        parsed.args["telegram_user_id"] = telegram_user_id

        result = await handle_command(parsed)

        # Send response immediately to Telegram
        response_text = result.get("message", "Alerts command executed")
        if result["status"] == "ok":
            await message.answer(response_text)
        else:
            response_text = f"❌ {result.get('message', 'Alerts command failed')}"
            await message.answer(response_text)

        # Send email notification if -email flag is present
        if parsed.args.get("email", False):
            await _send_email_notification_for_command(message, response_text, "alerts", telegram_user_id)

        return result

    except Exception as e:
        _logger.exception("Error in immediate alerts command")
        error_message = "❌ Error processing alerts command. Please try again."
        await message.answer(error_message)
        return {
            "status": "error",
            "message": f"Error processing alerts command: {str(e)}"
        }


async def process_schedules_command_immediate(telegram_user_id, args, message):
    """
    Process /schedules command immediately without notification service.

    Args:
        telegram_user_id: Telegram user ID
        args: Command arguments
        message: Telegram message object

    Returns:
        Dict with command result
    """
    try:
        # Use the proper command parser
        parsed = parse_command(message.text)
        parsed.args["telegram_user_id"] = telegram_user_id

        result = await handle_command(parsed)

        # Send response immediately to Telegram
        response_text = result.get("message", "Schedules command executed")
        if result["status"] == "ok":
            await message.answer(response_text)
        else:
            response_text = f"❌ {result.get('message', 'Schedules command failed')}"
            await message.answer(response_text)

        # Send email notification if -email flag is present
        if parsed.args.get("email", False):
            await _send_email_notification_for_command(message, response_text, "schedules", telegram_user_id)

        return result

    except Exception as e:
        _logger.exception("Error in immediate schedules command")
        error_message = "❌ Error processing schedules command. Please try again."
        await message.answer(error_message)
        return {
            "status": "error",
            "message": f"Error processing schedules command: {str(e)}"
        }


async def process_feedback_command_immediate(telegram_user_id, args, message):
    """
    Process /feedback command immediately without notification service.

    Args:
        telegram_user_id: Telegram user ID
        args: Command arguments
        message: Telegram message object

    Returns:
        Dict with command result
    """
    try:
        feedback_text = args[1] if len(args) > 1 else None
        parsed = ParsedCommand(command="feedback", args={"telegram_user_id": telegram_user_id, "feedback": feedback_text})

        result = await handle_command(parsed)

        # Send response immediately to Telegram
        if result["status"] == "ok":
            await message.answer(result.get("message", "Feedback submitted successfully"))
        else:
            await message.answer(f"❌ {result.get('message', 'Feedback submission failed')}")

        return result

    except Exception as e:
        _logger.exception("Error in immediate feedback command")
        error_message = "❌ Error submitting feedback. Please try again."
        await message.answer(error_message)
        return {
            "status": "error",
            "message": f"Error processing feedback command: {str(e)}"
        }


async def process_feature_command_immediate(telegram_user_id, args, message):
    """
    Process /feature command immediately without notification service.

    Args:
        telegram_user_id: Telegram user ID
        args: Command arguments
        message: Telegram message object

    Returns:
        Dict with command result
    """
    try:
        # Parse command
        parsed = parse_command(" ".join(args))
        parsed.args["telegram_user_id"] = telegram_user_id

        # Execute business logic
        result = await handle_command(parsed)

        # Send response immediately to Telegram
        if result["status"] == "ok":
            await message.answer(result.get("message", "Feature request submitted"))
        else:
            await message.answer(f"❌ {result.get('message', 'Feature request failed')}")

        return result

    except Exception as e:
        _logger.exception("Error in immediate feature command")
        error_message = "❌ Error processing feature request. Please try again."
        await message.answer(error_message)
        return {
            "status": "error",
            "message": f"Error processing feature command: {str(e)}"
        }


async def process_unknown_command_immediate(telegram_user_id, message, help_text):
    """
    Process unknown command immediately without notification service.

    Args:
        telegram_user_id: Telegram user ID
        message: Telegram message object
        help_text: Help text to show

    Returns:
        Dict with command result
    """
    try:
        # Create a user-friendly unknown command message
        unknown_command = message.text.split()[0] if message.text else "unknown"
        unknown_message = f"❓ Unknown command: {unknown_command}\n\nI don't recognize this command. Please use /help to see all available commands."

        # Send response immediately to Telegram
        await message.answer(unknown_message)

        return {
            "status": "ok",
            "message": "Unknown command processed"
        }

    except Exception as e:
        _logger.exception("Error in immediate unknown command handler")
        error_message = "❌ Error processing command. Please try again."
        await message.answer(error_message)
        return {
            "status": "error",
            "message": f"Error processing unknown command: {str(e)}"
        }