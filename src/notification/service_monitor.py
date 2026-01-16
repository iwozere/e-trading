#!/usr/bin/env python3
"""
Linux Service Monitoring Script

Checks the status and logs of critical systemd services and sends notifications
via the existing notification system if issues are detected.
"""

import subprocess
import json
import os
import sys
import re
from datetime import datetime, timezone
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.notification.service.message_queue_client import MessageQueueClient
from src.data.db.services.database_service import DatabaseService
from src.data.db.models.model_notification import MessagePriority

_logger = setup_logger(__name__)

SERVICES_TO_MONITOR = [
    "ibgateway-docker.service",
    "notification_bot.service",
    "scheduler.service",
    "telegram_bot.service",
    "trading.service",
    "trading-webui.service",
    "trading-api.service"
]

STATE_FILE = PROJECT_ROOT / "logs" / "monitor_state.json"

class ServiceMonitor:
    def __init__(self, admin_user_id=1):
        self.admin_user_id = admin_user_id
        self.mq_client = MessageQueueClient()
        self.state = self._load_state()

    def _load_state(self):
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, "r") as f:
                    return json.load(f)
            except Exception as e:
                _logger.error(f"Failed to load state file: {e}")
        return {}

    def _save_state(self):
        try:
            with open(STATE_FILE, "w") as f:
                json.dump(self.state, f, indent=4)
        except Exception as e:
            _logger.error(f"Failed to save state file: {e}")

    def check_service_status(self, service_name):
        """Check if service is active using systemctl."""
        try:
            result = subprocess.run(
                ["systemctl", "is-active", service_name],
                capture_output=True,
                text=True
            )
            status = result.stdout.strip()
            return status == "active", status
        except Exception as e:
            _logger.error(f"Error checking status for {service_name}: {e}")
            return False, "error"

    def check_service_logs(self, service_name, minutes=10):
        """Check service logs for errors using journalctl."""
        try:
            result = subprocess.run(
                ["journalctl", "-u", service_name, "--since", f"{minutes} minutes ago", "--no-pager"],
                capture_output=True,
                text=True
            )
            logs = result.stdout

            # Look for common error patterns
            error_patterns = [
                r"Traceback",
                r"Exception",
                r"SyntaxError",
                r"ERROR",
                r"CRITICAL",
                r"failed",
                r"Stopped"
            ]

            found_errors = []
            for pattern in error_patterns:
                if re.search(pattern, logs, re.IGNORECASE):
                    # Extract a snippet around the first match
                    match = re.search(pattern, logs, re.IGNORECASE)
                    start = max(0, match.start() - 100)
                    end = min(len(logs), match.end() + 300)
                    snippet = logs[start:end].strip()
                    found_errors.append(f"Found match for '{pattern}':\n...{snippet}...")
                    break # Just take the first one found

            return found_errors
        except Exception as e:
            _logger.error(f"Error checking logs for {service_name}: {e}")
            return []

    def run(self):
        _logger.info("Starting service monitoring check...")
        alerts_triggered = []

        for service in SERVICES_TO_MONITOR:
            is_active, status = self.check_service_status(service)
            errors = self.check_service_logs(service)

            last_alert_time = self.state.get(service, {}).get("last_alert_at")
            should_alert = False
            reason = ""

            if not is_active:
                reason = f"Service '{service}' is {status} (expected active)."
                should_alert = True
            elif errors:
                reason = f"Service '{service}' has errors in logs:\n" + "\n".join(errors)
                should_alert = True

            if should_alert:
                # Throttling: only alert once every 1 hour if status remains same
                now = datetime.now(timezone.utc).timestamp()
                if last_alert_time is None or (now - last_alert_time > 3600):
                    _logger.warning(f"Alert triggered for {service}: {reason}")
                    self.send_notification(service, reason)
                    self.state[service] = {
                        "status": status,
                        "last_alert_at": now,
                        "reason": reason
                    }
                    alerts_triggered.append(service)
                else:
                    _logger.debug(f"Alert throttled for {service}")
            else:
                # Reset state if service is back to normal
                if service in self.state:
                    _logger.info(f"Service {service} is back to normal.")
                    del self.state[service]

        self._save_state()
        _logger.info(f"Monitoring check complete. Alerts sent for: {alerts_triggered}")

    def send_notification(self, service, reason):
        """Queue a notification message."""
        try:
            message_content = {
                "subject": f"🚨 Service Alert: {service}",
                "body": f"The monitoring script detected an issue with {service} on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\nReason:\n{reason}",
                "service": service,
                "timestamp": datetime.now().isoformat()
            }

            # Send to both Email and Telegram if configured
            self.mq_client.enqueue_message(
                message_type="SYSTEM_ALERT",
                content=message_content,
                channels=["TELEGRAM", "EMAIL"],
                recipient_id=str(self.admin_user_id),
                priority=MessagePriority.CRITICAL
            )
            _logger.info(f"Notification queued for {service}")
        except Exception as e:
            _logger.error(f"Failed to queue notification for {service}: {e}")

if __name__ == "__main__":
    monitor = ServiceMonitor(admin_user_id=1) # Default admin ID
    monitor.run()
