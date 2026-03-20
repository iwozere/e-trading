import time
import re
import random
from typing import Any, Dict, Optional
from src.telegram.command_parser import ParsedCommand
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

class UserService:
    """
    Handles user-related business logic: registration, verification, info, language, and approvals.
    """

    def __init__(self, telegram_service):
        self.telegram_service = telegram_service

    def is_admin_user(self, telegram_user_id: str) -> bool:
        """Check if user is an admin."""
        if not self.telegram_service:
            return False
        status = self.telegram_service.get_user_status(telegram_user_id)
        return status and status.get("is_admin", False)

    def is_approved_user(self, telegram_user_id: str) -> bool:
        """Check if user is approved for restricted features."""
        if not self.telegram_service:
            return False
        status = self.telegram_service.get_user_status(telegram_user_id)
        return status and status.get("approved", False)

    def check_admin_access(self, telegram_user_id: str) -> Dict[str, Any]:
        """Check if user has admin access. Returns error dict if not."""
        if not self.is_admin_user(telegram_user_id):
            return {"status": "error", "message": "Access denied. Admin privileges required."}
        return {"status": "ok"}

    def check_approved_access(self, telegram_user_id: str) -> Dict[str, Any]:
        """Check if user has approved access for restricted features. Returns error dict if not."""
        if not self.is_approved_user(telegram_user_id):
            return {"status": "error", "message": "Access denied, please contact chat's admin or send request for approval using command /request_approval"}
        return {"status": "ok"}

    def handle_register(self, parsed: ParsedCommand) -> Dict[str, Any]:
        """Business logic for /register command."""
        try:
            telegram_user_id = parsed.args.get("telegram_user_id")
            email = parsed.args.get("email")
            language = parsed.args.get("language", "en")

            if not telegram_user_id:
                return {"status": "error", "message": "No telegram_user_id provided"}

            if not email:
                return {"status": "error", "message": "Please provide an email address. Usage: /register email@example.com [language]"}

            # Validate email format
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, email):
                return {"status": "error", "message": "Please provide a valid email address."}

            if not self.telegram_service:
                return {"status": "error", "message": "Service temporarily unavailable"}

            codes_sent = self.telegram_service.count_codes_last_hour(telegram_user_id)
            if codes_sent >= 5:
                return {"status": "error", "message": "Too many verification codes sent. Please wait an hour before requesting another."}

            # Generate verification code
            code = f"{random.randint(100000, 999999):06d}"
            sent_time = int(time.time())

            # Store user and code
            self.telegram_service.set_user_email(telegram_user_id, email, code, sent_time, language)

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
            _logger.exception("Error in register command")
            return {"status": "error", "message": f"Error registering email: {str(e)}"}

    def handle_verify(self, parsed: ParsedCommand) -> Dict[str, Any]:
        """Business logic for /verify command."""
        try:
            telegram_user_id = parsed.args.get("telegram_user_id")
            code = parsed.args.get("code")

            if not telegram_user_id:
                return {"status": "error", "message": "No telegram_user_id provided"}

            if not code:
                return {"status": "error", "message": "Please provide the verification code. Usage: /verify CODE"}

            if not code.isdigit() or len(code) != 6:
                return {"status": "error", "message": "Verification code must be a 6-digit number."}

            if not self.telegram_service:
                return {"status": "error", "message": "Service temporarily unavailable"}

            if self.telegram_service.verify_code(telegram_user_id, code, expiry_seconds=3600):
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
            _logger.exception("Error in verify command")
            return {"status": "error", "message": f"Error verifying code: {str(e)}"}

    def handle_info(self, parsed: ParsedCommand) -> Dict[str, Any]:
        """Business logic for /info command."""
        telegram_user_id = parsed.args.get("telegram_user_id")
        if not telegram_user_id:
            return {"status": "error", "message": "No telegram_user_id provided"}

        if not self.telegram_service:
            return {"status": "error", "message": "Service temporarily unavailable"}

        status = self.telegram_service.get_user_status(telegram_user_id)
        if status:
            email = status.get("email") or "(not set)"
            verified = "Yes" if status.get("verified") else "No"
            approved = "Yes" if status.get("approved") else "No"
            admin = "Yes" if status.get("is_admin") else "No"
            language = status.get("language") or "(not set)"
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

    def handle_language(self, parsed: ParsedCommand) -> Dict[str, Any]:
        """Business logic for /language command."""
        try:
            telegram_user_id = parsed.args.get("telegram_user_id")
            language = parsed.args.get("language")

            if not telegram_user_id:
                return {"status": "error", "message": "No telegram_user_id provided"}

            if not language:
                return {"status": "error", "message": "Please provide a language code. Usage: /language en (supported: en, ru)"}

            supported_languages = ["en", "ru"]
            if language.lower() not in supported_languages:
                return {"status": "error", "message": f"Language '{language}' not supported. Supported languages: {', '.join(supported_languages)}"}

            access_check = self.check_approved_access(telegram_user_id)
            if access_check["status"] != "ok":
                return access_check

            if not self.telegram_service:
                return {"status": "error", "message": "Service temporarily unavailable"}

            user_status = self.telegram_service.get_user_status(telegram_user_id)
            if not user_status:
                return {"status": "error", "message": "Please register first using /register email@example.com"}

            success = self.telegram_service.update_user_language(telegram_user_id, language.lower())
            if not success:
                return {"status": "error", "message": "Failed to update language preference"}

            return {
                "status": "ok",
                "title": "Language Updated",
                "message": f"Your language preference has been updated to {language.upper()}."
            }
        except Exception as e:
            _logger.exception("Error in language command")
            return {"status": "error", "message": f"Error updating language: {str(e)}"}

    def handle_request_approval(self, parsed: ParsedCommand) -> Dict[str, Any]:
        """Business logic for /request_approval command."""
        try:
            telegram_user_id = parsed.args.get("telegram_user_id")
            if not telegram_user_id:
                return {"status": "error", "message": "No telegram_user_id provided"}

            if not self.telegram_service:
                return {"status": "error", "message": "Service temporarily unavailable"}

            status = self.telegram_service.get_user_status(telegram_user_id)
            if not status:
                return {"status": "error", "message": "Please register first using /register your@email.com"}

            if not status.get("verified", False):
                return {"status": "error", "message": "Please verify your email first using /verify CODE"}

            if status.get("approved", False):
                return {"status": "error", "message": "You are already approved for restricted features"}

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

    # Admin Handlers
    def handle_admin_list_users(self, admin_telegram_user_id: str) -> Dict[str, Any]:
        """List all users for admin review."""
        try:
            if not self.telegram_service:
                return {"status": "error", "message": "Service temporarily unavailable"}

            users = self.telegram_service.get_all_users()
            if users is None:
                return {"status": "error", "message": "Unable to retrieve user list."}

            if not users:
                return {"status": "ok", "message": "No users found", "is_admin": True}

            user_list = []
            for user in users:
                email = user.get('email', 'N/A')
                verified = user.get("verified", False)
                approved = user.get("approved", False)
                status_text = "✅ Verified & Approved" if verified and approved else \
                             "✅ Verified" if verified else "❌ Not Verified"
                user_list.append(f"• {email} - {status_text}")

            return {
                "status": "ok",
                "message": f"**User List ({len(users)} users)**\n\n" + "\n".join(user_list),
                "is_admin": True
            }
        except Exception:
            _logger.exception("Error listing users")
            return {"status": "error", "message": "Unable to retrieve user list."}

    def handle_admin_list_pending_approvals(self, admin_telegram_user_id: str) -> Dict[str, Any]:
        """List users pending approval."""
        try:
            if not self.telegram_service:
                return {"status": "error", "message": "Service temporarily unavailable"}

            users = self.telegram_service.get_all_users()
            if not users:
                return {"status": "ok", "message": "No users found"}

            pending_users = [user for user in users if user.get("verified") and not user.get("approved")]
            if not pending_users:
                return {"status": "ok", "message": "No users pending approval"}

            user_list = []
            for user in pending_users:
                user_list.append(f"• {user.get('email', 'N/A')} (ID: {user.get('telegram_user_id')})")

            return {
                "status": "ok",
                "message": "**Users Pending Approval**\n\n" + "\n".join(user_list),
                "is_admin": True
            }
        except Exception:
            _logger.exception("Error listing pending approvals")
            return {"status": "error", "message": "Unable to retrieve pending approvals."}

    def handle_admin_reset_email(self, user_id: str) -> Dict[str, Any]:
        """Reset user's email verification status."""
        try:
            if self.telegram_service.reset_user_email_verification(user_id):
                return {"status": "ok", "message": f"Email verification reset for user {user_id}", "is_admin": True}
            return {"status": "error", "message": f"Failed to reset email for user {user_id}"}
        except Exception as e:
            _logger.exception("Error resetting email")
            return {"status": "error", "message": f"Error resetting email: {str(e)}"}

    def handle_admin_verify_user(self, user_id: str) -> Dict[str, Any]:
        """Verify a user's email."""
        try:
            if self.telegram_service.verify_user_email(user_id):
                return {"status": "ok", "message": f"User {user_id} verified successfully", "is_admin": True}
            return {"status": "error", "message": f"Failed to verify user {user_id}"}
        except Exception as e:
            _logger.exception("Error verifying user")
            return {"status": "error", "message": f"Error verifying user: {str(e)}"}

    def handle_admin_approve_user(self, user_id: str) -> Dict[str, Any]:
        """Approve a user for restricted features."""
        try:
            if self.telegram_service.approve_user(user_id):
                return {"status": "ok", "message": f"User {user_id} approved successfully", "is_admin": True}
            return {"status": "error", "message": f"Failed to approve user {user_id}"}
        except Exception as e:
            _logger.exception("Error approving user")
            return {"status": "error", "message": f"Error approving user: {str(e)}"}

    def handle_admin_reject_user(self, user_id: str) -> Dict[str, Any]:
        """Reject a user's approval request."""
        try:
            if self.telegram_service.reject_user(user_id):
                return {"status": "ok", "message": f"User {user_id} rejected", "is_admin": True}
            return {"status": "error", "message": f"Failed to reject user {user_id}"}
        except Exception as e:
            _logger.exception("Error rejecting user")
            return {"status": "error", "message": f"Error rejecting user: {str(e)}"}

    def handle_admin_set_limit(self, limit_type: str, limit_value: str, target_user_id: str = None) -> Dict[str, Any]:
        """Set user limits."""
        try:
            if limit_type not in ["alerts", "schedules"]:
                return {"status": "error", "message": "Limit type must be 'alerts' or 'schedules'"}
            try:
                limit = int(limit_value)
            except ValueError:
                return {"status": "error", "message": "Limit must be a number"}

            limit_key = f"max_{limit_type}"
            if target_user_id:
                self.telegram_service.set_user_limit(target_user_id, limit_key, limit)
                return {
                    "status": "ok",
                    "message": f"{limit_type.capitalize()} limit set to {limit} for user {target_user_id}",
                    "is_admin": True
                }
            return {"status": "error", "message": "Global limit setting not yet implemented"}
        except Exception as e:
            _logger.exception("Error setting limit")
            return {"status": "error", "message": f"Error setting limit: {str(e)}"}
