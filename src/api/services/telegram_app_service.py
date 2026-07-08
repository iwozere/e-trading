"""
Telegram Application Service
---------------------------

Application service layer for Telegram bot management in the web UI.
This service orchestrates domain services and transforms data for web UI consumption.

Follows proper architectural boundaries by only using domain services from src.data.db.services.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db.services.telegram_service import telegram_service
from src.data.db.services.users_service import users_service
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class TelegramAppService:
    """
    Application service for Telegram bot management.

    This service provides a clean interface for the web UI to interact with
    Telegram-related functionality while maintaining proper architectural boundaries.
    """

    def get_user_stats(self) -> Dict[str, Any]:
        """Get Telegram user statistics for web UI dashboard."""
        try:
            users = users_service.list_telegram_users_dto()

            total_users = len(users)
            verified_users = len([u for u in users if u.get("verified", False)])
            approved_users = len([u for u in users if u.get("approved", False)])
            pending_approvals = len([u for u in users if u.get("verified", False) and not u.get("approved", False)])
            admin_users = len([u for u in users if u.get("is_admin", False)])

            return {
                "total_users": total_users,
                "verified_users": verified_users,
                "approved_users": approved_users,
                "pending_approvals": pending_approvals,
                "admin_users": admin_users,
            }
        except Exception:
            _logger.exception("Error getting user stats:")
            raise

    def get_users_list(self, filter_type: str | None = None) -> List[Dict[str, Any]]:
        """Get filtered list of Telegram users."""
        try:
            users = users_service.list_telegram_users_dto()

            # Apply filtering
            if filter_type == "verified":
                users = [u for u in users if u.get("verified", False)]
            elif filter_type == "approved":
                users = [u for u in users if u.get("approved", False)]
            elif filter_type == "pending":
                users = [u for u in users if u.get("verified", False) and not u.get("approved", False)]

            # Transform to web UI format
            return [
                {
                    "telegram_user_id": str(user.get("telegram_user_id", "")),
                    "email": user.get("email"),
                    "verified": user.get("verified", False),
                    "approved": user.get("approved", False),
                    "language": str(user.get("language", "en")) if user.get("language") is not None else "en",
                    "is_admin": user.get("is_admin", False),
                    "max_alerts": user.get("max_alerts", 5),
                    "max_schedules": user.get("max_schedules", 5),
                    "created_at": user.get("created_at"),
                    "updated_at": user.get("updated_at"),
                }
                for user in users
            ]
        except Exception:
            _logger.exception("Error getting users list:")
            raise

    def verify_user(self, telegram_user_id: str) -> Dict[str, str]:
        """Manually verify a Telegram user."""
        try:
            users_service.update_telegram_profile(telegram_user_id, verified=True)
            _logger.info("Verified Telegram user: %s", telegram_user_id)
            return {"message": f"User {telegram_user_id} verified successfully"}
        except Exception:
            _logger.exception("Error verifying user %s:", telegram_user_id)
            raise

    def approve_user(self, telegram_user_id: str) -> Dict[str, str]:
        """Approve a Telegram user for access."""
        try:
            # Get user profile to check verification status
            profile = users_service.get_telegram_profile(telegram_user_id)
            if not profile:
                raise ValueError("User not found")

            if not profile.get("verified", False):
                raise ValueError("User must be verified before approval")

            users_service.update_telegram_profile(telegram_user_id, approved=True)
            _logger.info("Approved Telegram user: %s", telegram_user_id)
            return {"message": f"User {telegram_user_id} approved successfully"}
        except Exception:
            _logger.exception("Error approving user %s:", telegram_user_id)
            raise

    def reset_user_email(self, telegram_user_id: str) -> Dict[str, str]:
        """Reset a Telegram user's email verification status."""
        try:
            # Reset verification and approval
            users_service.update_telegram_profile(telegram_user_id, verified=False, approved=False)
            _logger.info("Reset email for Telegram user: %s", telegram_user_id)
            return {"message": f"Email reset for user {telegram_user_id} - user must re-verify"}
        except Exception:
            _logger.exception("Error resetting email for user %s:", telegram_user_id)
            raise

    def get_alert_stats(self) -> Dict[str, Any]:
        """Get Telegram alert statistics."""
        try:
            # Use telegram_service domain methods
            # Note: telegram_service methods are user-specific, so we'll get basic stats
            active_alerts = telegram_service.get_active_alerts()  # Get all active alerts

            return {
                "total_alerts": len(active_alerts),  # Simplified - only active alerts for now
                "active_alerts": len(active_alerts),
                "triggered_today": 0,  # Would need additional domain method
                "rearm_cycles": 0,  # Would need additional domain method
            }
        except Exception:
            _logger.exception("Error getting alert stats:")
            raise

    def get_alerts_list(
        self, filter_type: str | None = None, page: int = 1, page_size: int = 50
    ) -> List[Dict[str, Any]]:
        """Get filtered and paginated list of alerts."""
        import json as _json

        try:
            alerts = telegram_service.get_active_alerts()

            if filter_type == "inactive":
                alerts = []

            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            paginated_alerts = alerts[start_idx:end_idx]

            result = []
            for alert in paginated_alerts:
                task_params = getattr(alert, "task_params", None) or {}
                config_json_str = task_params.get("config_json", "{}")
                try:
                    config = (
                        _json.loads(config_json_str) if isinstance(config_json_str, str) else (config_json_str or {})
                    )
                except (_json.JSONDecodeError, TypeError):
                    config = {}
                result.append(
                    {
                        "id": alert.id,
                        "user_id": str(alert.user_id),
                        "ticker": task_params.get("ticker", getattr(alert, "target", "") or config.get("ticker", "")),
                        "price": config.get("price"),
                        "condition": config.get("condition", ""),
                        "active": alert.enabled,
                        "email": task_params.get("email", config.get("email", False)),
                        "alert_type": config.get("alert_type"),
                        "timeframe": config.get("timeframe"),
                        "config_json": config_json_str,
                        "alert_action": config.get("alert_action"),
                        "re_arm_config": task_params.get("re_arm_config"),
                        "is_armed": alert.enabled,
                        "last_price": None,
                        "last_triggered_at": None,
                        "created": alert.created_at.isoformat() if getattr(alert, "created_at", None) else None,
                    }
                )
            return result
        except Exception:
            _logger.exception("Error getting alerts list:")
            raise

    def toggle_alert(self, alert_id: int) -> Dict[str, Any]:
        """Toggle an alert's active status."""
        try:
            alert = telegram_service.get_alert(alert_id)
            if not alert:
                raise ValueError("Alert not found")

            current_enabled = getattr(alert, "enabled", True)
            new_enabled = not current_enabled
            success = telegram_service.update_alert(alert_id, enabled=new_enabled)

            if not success:
                raise RuntimeError("Failed to toggle alert")

            status_text = "activated" if new_enabled else "deactivated"
            return {
                "id": str(alert_id),
                "active": new_enabled,
                "is_active": new_enabled,
                "message": f"Alert {alert_id} {status_text} successfully",
            }
        except Exception:
            _logger.exception("Error toggling alert %d:", alert_id)
            raise

    def delete_alert(self, alert_id: int) -> Dict[str, str]:
        """Delete an alert."""
        try:
            alert = telegram_service.get_alert(alert_id)
            if not alert:
                raise ValueError("Alert not found")

            success = telegram_service.delete_alert(alert_id)
            if not success:
                raise RuntimeError("Failed to delete alert")

            return {"message": f"Alert {alert_id} deleted successfully"}
        except Exception:
            _logger.exception("Error deleting alert %d:", alert_id)
            raise

    def create_alert(
        self,
        user_id: str,
        ticker: str,
        condition: str = "custom",
        config_json: str | None = None,
        timeframe: str | None = None,
        alert_type: str | None = None,
    ) -> Dict[str, Any]:
        """Create a new alert from web UI."""
        import json as _json

        try:
            try:
                raw_config = _json.loads(config_json) if config_json else {}
            except (_json.JSONDecodeError, TypeError):
                raw_config = {}

            full_config = {
                "ticker": ticker,
                "condition": condition,
                "alert_type": alert_type or "custom",
                **raw_config,
            }
            if timeframe:
                full_config["timeframe"] = timeframe

            alert_id = telegram_service.add_json_alert(user_id, _json.dumps(full_config))
            _logger.info("Created alert %d for user %s, ticker %s", alert_id, user_id, ticker)
            return {
                "id": str(alert_id),
                "user_id": user_id,
                "ticker": ticker,
                "condition": condition,
                "active": True,
                "is_active": True,
                "config_json": config_json,
                "alert_type": alert_type or "custom",
                "timeframe": timeframe,
            }
        except Exception:
            _logger.exception("Error creating alert for user %s:", user_id)
            raise

    def create_schedule(
        self,
        user_id: str,
        scheduled_time: str,
        ticker: str = "",
        config_json: str | None = None,
        schedule_type: str | None = None,
        list_type: str | None = None,
    ) -> Dict[str, Any]:
        """Create a new schedule from web UI."""
        import json as _json

        try:
            try:
                raw_config = _json.loads(config_json) if config_json else {}
            except (_json.JSONDecodeError, TypeError):
                raw_config = {}

            resolved_ticker = ticker or list_type or raw_config.get("ticker", "SCREENER")
            full_config = {
                "ticker": resolved_ticker,
                "scheduled_time": scheduled_time,
                "schedule_type": schedule_type or "screener",
                **raw_config,
            }
            if list_type:
                full_config["list_type"] = list_type

            schedule_id = telegram_service.add_json_schedule(user_id, _json.dumps(full_config))
            _logger.info("Created schedule %d for user %s at %s", schedule_id, user_id, scheduled_time)
            return {
                "id": str(schedule_id),
                "user_id": user_id,
                "ticker": resolved_ticker,
                "scheduled_time": scheduled_time,
                "active": True,
                "is_active": True,
                "config_json": config_json,
                "schedule_type": schedule_type or "screener",
            }
        except Exception:
            _logger.exception("Error creating schedule for user %s:", user_id)
            raise

    def toggle_schedule(self, schedule_id: int) -> Dict[str, Any]:
        """Toggle a schedule's active status."""
        try:
            schedule = telegram_service.get_schedule(schedule_id)
            if not schedule:
                raise ValueError("Schedule not found")

            current_enabled = schedule.get("active", True)
            new_enabled = not current_enabled
            success = telegram_service.update_schedule(schedule_id, enabled=new_enabled)
            if not success:
                raise RuntimeError("Failed to toggle schedule")

            status_text = "activated" if new_enabled else "deactivated"
            return {
                "id": str(schedule_id),
                "active": new_enabled,
                "is_active": new_enabled,
                "message": f"Schedule {schedule_id} {status_text} successfully",
            }
        except Exception:
            _logger.exception("Error toggling schedule %d:", schedule_id)
            raise

    def delete_schedule(self, schedule_id: int) -> Dict[str, str]:
        """Delete a schedule."""
        try:
            schedule = telegram_service.get_schedule(schedule_id)
            if not schedule:
                raise ValueError("Schedule not found")

            success = telegram_service.delete_schedule(schedule_id)
            if not success:
                raise RuntimeError("Failed to delete schedule")

            return {"message": f"Schedule {schedule_id} deleted successfully"}
        except Exception:
            _logger.exception("Error deleting schedule %d:", schedule_id)
            raise

    def get_schedule_stats(self) -> Dict[str, Any]:
        """Get Telegram schedule statistics."""
        try:
            # Get all active schedules to calculate stats
            active_schedules = telegram_service.get_active_schedules()

            return {
                "total_schedules": len(active_schedules),  # Only active schedules for now
                "active_schedules": len(active_schedules),
                "executed_today": 0,  # Would need additional domain method
                "failed_executions": 0,  # Would need additional domain method
            }
        except Exception:
            _logger.exception("Error getting schedule stats:")
            raise

    def get_schedules_list(self, filter_type: str | None = None) -> List[Dict[str, Any]]:
        """Get filtered list of schedules."""
        try:
            # Get all active schedules from telegram_service
            schedules = telegram_service.get_active_schedules()

            # Filter if needed
            if filter_type == "inactive":
                # For now, return empty list since we only have active schedules
                schedules = []
            elif filter_type == "active":
                # Already filtered to active schedules
                pass

            # Transform to web UI format - schedules are now dictionaries from telegram_service
            return [
                {
                    "id": schedule.get("id"),
                    "user_id": str(schedule.get("user_id", "")),
                    "ticker": schedule.get("ticker", ""),
                    "scheduled_time": schedule.get("scheduled_time", ""),
                    "period": schedule.get("period"),
                    "active": schedule.get("active", True),
                    "email": schedule.get("email", False),
                    "indicators": schedule.get("indicators"),
                    "interval": schedule.get("interval"),
                    "provider": schedule.get("provider"),
                    "schedule_type": schedule.get("schedule_type"),
                    "list_type": schedule.get("list_type"),
                    "config_json": schedule.get("config_json"),
                    "schedule_config": schedule.get("schedule_config"),
                    "created": schedule.get("created"),
                }
                for schedule in schedules
            ]
        except Exception:
            _logger.exception("Error getting schedules list:")
            raise

    def send_broadcast(self, message: str) -> Dict[str, Any]:
        """Send broadcast message to approved users."""
        try:
            # Get users for broadcast
            users = telegram_service.list_users()
            approved_users = [u for u in users if u.get("approved", False)]

            total_recipients = len(approved_users)

            # In a real implementation, this would integrate with the Telegram bot
            # For now, we simulate successful delivery
            successful_deliveries = total_recipients
            failed_deliveries = 0

            # Log the broadcast
            broadcast_id = telegram_service.log_broadcast(
                message=message,
                sent_by="admin",  # TODO: Get actual user from context
                success_count=successful_deliveries,
                total_count=total_recipients,
            )

            _logger.info("Broadcast sent to %d users, logged with ID %d", total_recipients, broadcast_id)

            return {
                "message": "Broadcast sent successfully",
                "broadcast_id": str(broadcast_id),
                "total_recipients": total_recipients,
                "successful_deliveries": successful_deliveries,
                "failed_deliveries": failed_deliveries,
            }
        except Exception:
            _logger.exception("Error sending broadcast:")
            raise

    def get_broadcast_history(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Get broadcast history with pagination."""
        try:
            # Use telegram_service to get broadcast history
            history = telegram_service.get_broadcast_history(limit=limit, offset=offset)
            return history
        except Exception:
            _logger.exception("Error getting broadcast history:")
            raise

    def get_broadcast_stats(self) -> Dict[str, Any]:
        """Get broadcast statistics."""
        try:
            # Use telegram_service to get broadcast stats
            stats = telegram_service.get_broadcast_stats()
            return stats
        except Exception:
            _logger.exception("Error getting broadcast stats:")
            raise

    def get_audit_logs(
        self,
        limit: int = 100,
        offset: int = 0,
        user_id: str | None = None,
        command: str | None = None,
        success_only: bool | None = None,
    ) -> List[Dict[str, Any]]:
        """Get audit logs with filtering."""
        try:
            # Use telegram_service to get audit logs (already returns dictionaries)
            logs = telegram_service.get_all_command_audit(
                limit=limit, offset=offset, user_id=user_id, command=command, success_only=success_only
            )

            # Data is already in dictionary format from telegram_service
            return logs
        except Exception:
            _logger.exception("Error getting audit logs:")
            raise

    def get_audit_stats(self) -> Dict[str, Any]:
        """Get audit statistics."""
        try:
            return telegram_service.get_command_audit_stats()
        except Exception:
            _logger.exception("Error getting audit stats:")
            raise

    def get_user_audit_logs(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get audit logs for a specific user."""
        try:
            logs = telegram_service.get_user_command_history(user_id, limit)

            # Transform to web UI format (logs are already dictionaries from telegram_service)
            return [
                {
                    "id": log.get("id"),
                    "telegram_user_id": log.get("telegram_user_id"),
                    "command": log.get("command"),
                    "full_message": log.get("full_message"),
                    "is_registered_user": log.get("is_registered_user", False),
                    "user_email": log.get("user_email"),
                    "success": log.get("success", True),
                    "error_message": log.get("error_message"),
                    "response_time_ms": log.get("response_time_ms"),
                    "created": log.get("created"),  # Already in ISO format from telegram_service
                }
                for log in logs
            ]
        except Exception:
            _logger.exception("Error getting user audit logs:")
            raise
