import json
import re
from typing import Any, Dict, List, Optional
from src.telegram.command_parser import ParsedCommand
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

class ScheduleManager:
    """
    Handles lifecycle of scheduled reports and screeners: add, edit, list, delete, pause, resume.
    """

    def __init__(self, telegram_service):
        self.telegram_service = telegram_service

    def handle_schedules(self, parsed: ParsedCommand) -> Dict[str, Any]:
        """Business logic for /schedules command."""
        try:
            telegram_user_id = parsed.args.get("telegram_user_id")
            if not telegram_user_id:
                return {"status": "error", "message": "No telegram_user_id provided"}

            action = parsed.positionals[0] if len(parsed.positionals) > 0 else None
            params = parsed.positionals[1:] if len(parsed.positionals) > 1 else []

            if not action:
                return self.handle_schedules_list(telegram_user_id)

            if action == "add" and len(params) >= 2:
                ticker, time = params[0], params[1]
                email = parsed.args.get("email", False)
                indicators = parsed.args.get("indicators")
                period = parsed.args.get("period", "2y")
                interval = parsed.args.get("interval", "1d")
                provider = parsed.args.get("provider")
                return self.handle_schedules_add(telegram_user_id, ticker, time, email, indicators, period, interval, provider)
            elif action == "add_json" and len(params) >= 1:
                return self.handle_schedules_add_json(telegram_user_id, params[0])
            elif action == "screener" and len(params) >= 1:
                list_type = params[0]
                time = params[1] if len(params) > 1 else "09:00"
                email = parsed.args.get("email", False)
                indicators = parsed.args.get("indicators")
                return self.handle_schedules_screener(telegram_user_id, list_type, time, email, indicators)
            elif action == "enhanced_screener" and len(params) >= 1:
                return self.handle_schedules_enhanced_screener(telegram_user_id, params[0])
            elif action == "edit" and len(params) >= 1:
                return self.handle_schedules_edit(telegram_user_id, params[0], params[1] if len(params) > 1 else None, parsed.args)
            elif action == "delete" and len(params) >= 1:
                return self.handle_schedules_delete(telegram_user_id, params[0])
            elif action == "pause" and len(params) >= 1:
                return self.handle_schedules_pause(telegram_user_id, params[0])
            elif action == "resume" and len(params) >= 1:
                return self.handle_schedules_resume(telegram_user_id, params[0])
            else:
                return {"status": "error", "title": "Schedules Help", "message": self._get_schedules_help_text()}
        except Exception as e:
            _logger.exception("Error in schedules command")
            return {"status": "error", "message": f"Error processing schedules command: {str(e)}"}

    def handle_schedules_list(self, telegram_user_id: str) -> Dict[str, Any]:
        """List all schedules for a user."""
        try:
            if not self.telegram_service:
                return {"status": "error", "message": "Service unavailable"}
            schedules = self.telegram_service.list_schedules(telegram_user_id)
            if not schedules:
                return {"status": "ok", "title": "Your Schedules", "message": "No scheduled reports."}

            schedule_list = []
            for s in schedules:
                status = "🟢 Active" if s.get("active") else "🔴 Paused"
                icon = "📧" if s.get("email") else "💬"
                display_time = self._format_cron_time(s.get("cron", ""))
                name = s.get("name", "Schedule")
                ticker = s.get("ticker", "Report")
                schedule_list.append(f"#{s['id']} ({name}): {ticker} at {display_time} {icon} - {status}")

            return {"status": "ok", "title": "Your Schedules", "message": "\n".join(schedule_list)}
        except Exception as e:
            _logger.exception("Error listing schedules")
            return {"status": "error", "message": f"Error listing schedules: {str(e)}"}

    def handle_schedules_add(self, telegram_user_id: str, ticker: str, time: str, email: bool = False, indicators: str = None, period: str = "2y", interval: str = "1d", provider: str = None) -> Dict[str, Any]:
        """Add a new scheduled report."""
        try:
            if not re.match(r'^([01]?[0-9]|2[0-3]):[0-5][0-9]$', time):
                return {"status": "error", "message": "Time must be HH:MM format"}
            if not self.telegram_service: return {"status": "error", "message": "Service unavailable"}

            user_status = self.telegram_service.get_user_status(telegram_user_id)
            if len(self.telegram_service.list_schedules(telegram_user_id)) >= user_status.get("max_schedules", 5):
                return {"status": "error", "message": "Schedule limit reached."}

            schedule_id = self.telegram_service.add_schedule(telegram_user_id, ticker.upper(), time, period="daily", email=email, indicators=indicators, interval=interval, provider=provider)
            return {"status": "ok", "title": "Schedule Added", "message": f"Schedule #{schedule_id} created for {ticker.upper()} at {time} UTC."}
        except Exception as e:
            _logger.exception("Error adding schedule")
            return {"status": "error", "message": f"Error adding schedule: {str(e)}"}

    def handle_schedules_add_json(self, telegram_user_id: str, config_json: str) -> Dict[str, Any]:
        """Add a JSON-based schedule."""
        try:
            config = json.loads(config_json)
            if not self.telegram_service: return {"status": "error", "message": "Service unavailable"}
            user_status = self.telegram_service.get_user_status(telegram_user_id)
            if len(self.telegram_service.list_schedules(telegram_user_id)) >= user_status.get("max_schedules", 5):
                return {"status": "error", "message": "Schedule limit reached."}

            schedule_id = self.telegram_service.add_json_schedule(telegram_user_id, config_json, schedule_config="report")
            return {"status": "ok", "title": "Schedule Added", "message": f"JSON Schedule #{schedule_id} created."}
        except Exception as e:
            _logger.exception("Error adding JSON schedule")
            return {"status": "error", "message": "Failed to add JSON schedule."}

    def handle_schedules_screener(self, telegram_user_id: str, list_type: str, time: str, email: bool = False, indicators: str = None) -> Dict[str, Any]:
        """Handle fundamental screener schedule."""
        try:
            valid_types = ['us_small_cap', 'us_medium_cap', 'us_large_cap', 'swiss_shares', 'custom_list']
            if list_type.lower() not in valid_types:
                return {"status": "error", "message": f"Invalid list type. Valid: {', '.join(valid_types)}"}
            if not self.telegram_service: return {"status": "error", "message": "Service unavailable"}

            schedule_data = {
                'telegram_user_id': telegram_user_id,
                'ticker': f"SCREENER_{list_type.upper()}",
                'scheduled_time': time,
                'email': email,
                'indicators': indicators,
                'schedule_type': 'screener',
                'list_type': list_type.lower()
            }
            schedule_id = self.telegram_service.add_json_schedule(telegram_user_id, json.dumps(schedule_data))
            return {"status": "ok", "title": "Screener Scheduled", "message": f"Fundamental screener scheduled (#{schedule_id})."}
        except Exception as e:
            _logger.exception("Error scheduling screener")
            return {"status": "error", "message": "Failed to schedule screener."}

    def handle_schedules_enhanced_screener(self, telegram_user_id: str, config_json: str) -> Dict[str, Any]:
        """Handle enhanced screener schedule."""
        try:
            if not self.telegram_service: return {"status": "error", "message": "Service unavailable"}
            schedule_id = self.telegram_service.add_json_schedule(telegram_user_id, config_json, schedule_config="enhanced_screener")
            return {"status": "ok", "title": "Enhanced Screener Scheduled", "message": f"Enhanced screener scheduled (#{schedule_id})."}
        except Exception as e:
            _logger.exception("Error scheduling enhanced screener")
            return {"status": "error", "message": "Failed to schedule enhanced screener."}

    def handle_schedules_edit(self, telegram_user_id: str, schedule_id_str: str, new_time: str = None, args: dict = None) -> Dict[str, Any]:
        """Edit an existing schedule."""
        try:
            schedule_id = int(schedule_id_str)
            if not self.telegram_service: return {"status": "error", "message": "Service unavailable"}
            schedule = self.telegram_service.get_schedule_by_id(schedule_id)
            if not schedule or schedule.get("user_id") != telegram_user_id:
                return {"status": "error", "message": "Schedule not found."}

            updates = {}
            if new_time:
                if not re.match(r'^([01]?[0-9]|2[0-3]):[0-5][0-9]$', new_time):
                    return {"status": "error", "message": "Invalid time format."}
                updates["scheduled_time"] = new_time
            if args:
                for key in ["email", "indicators", "period", "interval", "provider"]:
                    if key in args: updates[key] = args[key]

            if not updates: return {"status": "error", "message": "No updates provided."}
            self.telegram_service.update_schedule(schedule_id, **updates)
            return {"status": "ok", "title": "Schedule Updated", "message": f"Schedule #{schedule_id} updated."}
        except Exception as e:
            _logger.exception("Error editing schedule")
            return {"status": "error", "message": "Failed to edit schedule."}

    def handle_schedules_delete(self, telegram_user_id: str, schedule_id_str: str) -> Dict[str, Any]:
        """Delete a schedule."""
        try:
            schedule_id = int(schedule_id_str)
            if not self.telegram_service: return {"status": "error", "message": "Service unavailable"}
            schedule = self.telegram_service.get_schedule_by_id(schedule_id)
            if not schedule or schedule.get("user_id") != telegram_user_id:
                return {"status": "error", "message": "Schedule not found."}
            self.telegram_service.delete_schedule(schedule_id)
            return {"status": "ok", "message": f"Schedule #{schedule_id} deleted."}
        except Exception as e:
            _logger.exception("Error deleting schedule")
            return {"status": "error", "message": "Failed to delete schedule."}

    def handle_schedules_pause(self, telegram_user_id: str, schedule_id_str: str) -> Dict[str, Any]:
        """Pause a schedule."""
        try:
            schedule_id = int(schedule_id_str)
            if not self.telegram_service: return {"status": "error", "message": "Service unavailable"}
            self.telegram_service.update_schedule(schedule_id, active=False)
            return {"status": "ok", "message": f"Schedule #{schedule_id} paused."}
        except Exception as e:
            _logger.exception("Error pausing schedule")
            return {"status": "error", "message": "Failed to pause schedule."}

    def handle_schedules_resume(self, telegram_user_id: str, schedule_id_str: str) -> Dict[str, Any]:
        """Resume a schedule."""
        try:
            schedule_id = int(schedule_id_str)
            if not self.telegram_service: return {"status": "error", "message": "Service unavailable"}
            self.telegram_service.update_schedule(schedule_id, active=True)
            return {"status": "ok", "message": f"Schedule #{schedule_id} resumed."}
        except Exception as e:
            _logger.exception("Error resuming schedule")
            return {"status": "error", "message": "Failed to resume schedule."}

    def _format_cron_time(self, cron: str) -> str:
        """Helper to format cron expression into readable time."""
        if not cron: return "Unknown"
        parts = cron.split()
        if len(parts) != 5: return cron
        try:
            minute, hour = parts[0], parts[1]
            return f"{int(hour):02d}:{int(minute):02d}"
        except: return cron

    def _get_schedules_help_text(self) -> str:
        return ("Available schedule commands:\n"
                "/schedules - List all schedules\n"
                "/schedules add TICKER TIME [flags] - Schedule daily report\n"
                "  Example: /schedules add AAPL 09:00 -email\n"
                "/schedules screener LIST_TYPE [TIME] [flags] - Schedule fundamental screener\n"
                "  Example: /schedules screener us_small_cap 09:00 -email\n"
                "/schedules delete SCHEDULE_ID - Delete schedule\n"
                "/schedules pause SCHEDULE_ID - Pause schedule\n"
                "/schedules resume SCHEDULE_ID - Resume schedule")
