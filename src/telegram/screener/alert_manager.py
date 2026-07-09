from typing import Any, Dict

from src.notification.logger import setup_logger
from src.telegram.command_parser import ParsedCommand

_logger = setup_logger(__name__)


class AlertManager:
    """
    Handles price and indicator-based alert lifecycle: creation, listing, editing, and deletion.
    """

    def __init__(self, telegram_service):
        self.telegram_service = telegram_service

    def handle_alerts(self, parsed: ParsedCommand) -> Dict[str, Any]:
        """Business logic for /alerts command."""
        try:
            telegram_user_id = parsed.args.get("telegram_user_id")
            if not telegram_user_id:
                return {"status": "error", "message": "No telegram_user_id provided"}

            # Get action and parameters from positionals
            action = parsed.positionals[0] if len(parsed.positionals) > 0 else None
            params = parsed.positionals[1:] if len(parsed.positionals) > 1 else []

            if not action:
                return self.handle_alerts_list(telegram_user_id)

            if action == "add" and len(params) >= 3:
                ticker, price, condition = params[0], params[1], params[2]
                email = parsed.args.get("email", False)
                return self.handle_alerts_add(telegram_user_id, ticker, price, condition, email)
            elif action == "add_indicator" and len(params) >= 2:
                ticker, config_json = params[0], params[1]
                email = parsed.args.get("email", False)
                timeframe = parsed.args.get("timeframe", "15m")
                alert_action = parsed.args.get("action_type", "notify")
                return self.handle_alerts_add_indicator(
                    telegram_user_id, ticker, config_json, timeframe, alert_action, email
                )
            elif action == "edit" and len(params) >= 1:
                alert_id = params[0]
                new_price = params[1] if len(params) > 1 else None
                new_condition = params[2] if len(params) > 2 else None
                email = parsed.args.get("email")
                return self.handle_alerts_edit(telegram_user_id, alert_id, new_price, new_condition, email)
            elif action == "delete" and len(params) >= 1:
                alert_id = params[0]
                return self.handle_alerts_delete(telegram_user_id, alert_id)
            elif action == "pause" and len(params) >= 1:
                alert_id = params[0]
                return self.handle_alerts_pause(telegram_user_id, alert_id)
            elif action == "resume" and len(params) >= 1:
                alert_id = params[0]
                return self.handle_alerts_resume(telegram_user_id, alert_id)
            else:
                return {"status": "error", "title": "Alerts Help", "message": self._get_alerts_help_text()}
        except Exception as e:
            _logger.exception("Error in alerts command")
            return {"status": "error", "message": f"Error processing alerts command: {str(e)}"}

    def handle_alerts_list(self, telegram_user_id: str) -> Dict[str, Any]:
        """List all alerts for a user."""
        try:
            if not self.telegram_service:
                return {"status": "error", "message": "Service temporarily unavailable"}

            alerts = self.telegram_service.list_alerts(telegram_user_id)
            if not alerts:
                return {"status": "ok", "title": "Your Alerts", "message": "You have no active alerts."}

            alert_list = []
            for alert in alerts:
                status = "🟢 Active" if alert.get("active") else "🔴 Paused"
                email_flag = "📧" if alert.get("email") else "💬"
                alert_type = alert.get("alert_type", "price")
                if alert_type == "price":
                    alert_list.append(
                        f"#{alert['id']}: {alert['ticker']} {alert['condition']} ${alert['price']:.2f} {email_flag} - {status}"
                    )
                else:
                    alert_type_icon = "📊" if alert_type == "indicator" else "❓"
                    timeframe = alert.get("timeframe", "15m")
                    action = alert.get("alert_action", "notify")
                    condition_text = alert.get("condition", "Unknown condition")
                    alert_list.append(
                        f"#{alert['id']}: {alert['ticker']} {alert_type_icon} {condition_text} ({timeframe}, {action}) {email_flag} - {status}"
                    )

            message = f"Your alerts ({len(alerts)}):\n\n" + "\n".join(alert_list)
            return {"status": "ok", "title": "Your Alerts", "message": message}
        except Exception as e:
            _logger.exception("Error listing alerts")
            return {"status": "error", "message": f"Error listing alerts: {str(e)}"}

    def handle_alerts_add(
        self, telegram_user_id: str, ticker: str, price_str: str, condition: str, email: bool = False
    ) -> Dict[str, Any]:
        """Add a new price alert."""
        try:
            if condition.lower() not in ["above", "below"]:
                return {"status": "error", "message": "Condition must be 'above' or 'below'"}
            try:
                price = float(price_str)
                if price <= 0:
                    raise ValueError()
            except ValueError:
                return {"status": "error", "message": "Price must be a positive number"}

            if not self.telegram_service:
                return {"status": "error", "message": "Service temporarily unavailable"}

            user_status = self.telegram_service.get_user_status(telegram_user_id)
            max_alerts = user_status.get("max_alerts", 5)
            current_alerts = self.telegram_service.list_alerts(telegram_user_id)
            if len(current_alerts) >= max_alerts:
                return {"status": "error", "message": f"Alert limit reached ({max_alerts}). Delete some alerts first."}

            alert_id = self.telegram_service.add_alert(
                telegram_user_id, ticker.upper(), price, condition.lower(), email
            )
            if not alert_id:
                return {"status": "error", "message": "Failed to create alert."}

            email_text = " and email" if email else ""
            return {
                "status": "ok",
                "title": "Alert Added",
                "message": f"Alert #{alert_id} created: {ticker.upper()} {condition.lower()} ${price:.2f}{email_text}\n🔄 Re-arm enabled.",
            }
        except Exception as e:
            _logger.exception("Error adding alert")
            return {"status": "error", "message": f"Error adding alert: {str(e)}"}

    def handle_alerts_add_indicator(
        self,
        telegram_user_id: str,
        ticker: str,
        config_json: str,
        timeframe: str = "15m",
        alert_action: str = "notify",
        email: bool = False,
    ) -> Dict[str, Any]:
        """Add a new indicator-based alert."""
        try:
            if not ticker or len(ticker.strip()) == 0:
                return {"status": "error", "message": "Ticker is required"}

            valid_timeframes = ["5m", "15m", "1h", "4h", "1d"]
            if timeframe not in valid_timeframes:
                return {
                    "status": "error",
                    "message": f"Invalid timeframe. Must be one of: {', '.join(valid_timeframes)}",
                }

            valid_actions = ["BUY", "SELL", "HOLD", "notify"]
            if alert_action not in valid_actions:
                return {"status": "error", "message": f"Invalid action. Must be one of: {', '.join(valid_actions)}"}

            if not self.telegram_service:
                return {"status": "error", "message": "Service temporarily unavailable"}

            user_status = self.telegram_service.get_user_status(telegram_user_id)
            max_alerts = user_status.get("max_alerts", 5)
            current_alerts = len(self.telegram_service.list_alerts(telegram_user_id))
            if current_alerts >= max_alerts:
                return {"status": "error", "message": f"Alert limit reached ({max_alerts})."}

            alert_id = self.telegram_service.add_indicator_alert(
                telegram_user_id=telegram_user_id,
                ticker=ticker.upper(),
                indicator="custom",
                condition=config_json,
                value=0.0,
                timeframe=timeframe,
                alert_action=alert_action,
                email=email,
            )
            email_text = " and email" if email else ""
            return {
                "status": "ok",
                "title": "Indicator Alert Added",
                "message": f"Alert #{alert_id} created: {ticker.upper()} Indicator Alert ({timeframe}){email_text}",
            }
        except Exception as e:
            _logger.exception("Error adding indicator alert")
            return {"status": "error", "message": f"Error adding indicator alert: {str(e)}"}

    def handle_alerts_edit(
        self,
        telegram_user_id: str,
        alert_id_str: str,
        new_price_str: str | None = None,
        new_condition: str | None = None,
        email: bool | None = None,
    ) -> Dict[str, Any]:
        """Edit an existing alert."""
        try:
            alert_id = int(alert_id_str)
            if not self.telegram_service:
                return {"status": "error", "message": "Service temporarily unavailable"}

            alert = self.telegram_service.get_alert(alert_id)
            if not alert or alert.get("user_id") != telegram_user_id:
                return {"status": "error", "message": f"Alert #{alert_id} not found."}

            updates: Dict[str, Any] = {}
            if new_price_str:
                updates["price"] = float(new_price_str)
            if new_condition:
                updates["condition"] = new_condition.lower()
            if email is not None:
                updates["email"] = 1 if email else 0

            if not updates:
                return {"status": "error", "message": "No updates provided."}

            self.telegram_service.update_alert(alert_id, **updates)
            updated_alert = self.telegram_service.get_alert(alert_id)
            return {
                "status": "ok",
                "title": "Alert Updated",
                "message": f"Alert #{alert_id} updated: {updated_alert['ticker']} {updated_alert['condition']} ${updated_alert['price']:.2f}",
            }
        except Exception as e:
            _logger.exception("Error editing alert")
            return {"status": "error", "message": f"Error editing alert: {str(e)}"}

    def handle_alerts_delete(self, telegram_user_id: str, alert_id_str: str) -> Dict[str, Any]:
        """Delete an alert."""
        try:
            alert_id = int(alert_id_str)
            if not self.telegram_service:
                return {"status": "error", "message": "Service temporarily unavailable"}
            alert = self.telegram_service.get_alert(alert_id)
            if not alert or alert.get("user_id") != telegram_user_id:
                return {"status": "error", "message": f"Alert #{alert_id} not found."}

            self.telegram_service.delete_alert(alert_id)
            return {
                "status": "ok",
                "title": "Alert Deleted",
                "message": f"Alert #{alert_id} for {alert['ticker']} deleted.",
            }
        except Exception as e:
            _logger.exception("Error deleting alert")
            return {"status": "error", "message": f"Error deleting alert: {str(e)}"}

    def handle_alerts_pause(self, telegram_user_id: str, alert_id_str: str) -> Dict[str, Any]:
        """Pause an alert."""
        try:
            alert_id = int(alert_id_str)
            if not self.telegram_service:
                return {"status": "error", "message": "Service temporarily unavailable"}
            alert = self.telegram_service.get_alert(alert_id)
            if not alert or alert.get("user_id") != telegram_user_id:
                return {"status": "error", "message": f"Alert #{alert_id} not found."}

            self.telegram_service.update_alert(alert_id, active=False)
            return {"status": "ok", "title": "Alert Paused", "message": f"Alert #{alert_id} paused."}
        except Exception as e:
            _logger.exception("Error pausing alert")
            return {"status": "error", "message": f"Error pausing alert: {str(e)}"}

    def handle_alerts_resume(self, telegram_user_id: str, alert_id_str: str) -> Dict[str, Any]:
        """Resume a paused alert."""
        try:
            alert_id = int(alert_id_str)
            if not self.telegram_service:
                return {"status": "error", "message": "Service temporarily unavailable"}
            alert = self.telegram_service.get_alert(alert_id)
            if not alert or alert.get("user_id") != telegram_user_id:
                return {"status": "error", "message": f"Alert #{alert_id} not found."}

            self.telegram_service.update_alert(alert_id, active=True)
            return {"status": "ok", "title": "Alert Resumed", "message": f"Alert #{alert_id} resumed."}
        except Exception as e:
            _logger.exception("Error resuming alert")
            return {"status": "error", "message": f"Error resuming alert: {str(e)}"}

    def _get_alerts_help_text(self) -> str:
        return (
            "Available alert commands:\n"
            "/alerts - List all alerts\n"
            "/alerts add TICKER PRICE CONDITION [flags] - Add price alert\n"
            "  CONDITION: above or below\n"
            "  Example: /alerts add BTCUSDT 65000 above -email\n"
            "/alerts add_indicator TICKER CONFIG_JSON [flags] - Add indicator alert\n"
            '  Example: /alerts add_indicator AAPL \'{"type":"indicator","indicator":"RSI","condition":{"operator":"<","value":30}}\' -email\n'
            "Flags:\n"
            "  -email: Send alert notification to email\n"
            "  -timeframe=15m: Set timeframe (5m, 15m, 1h, 4h, 1d)\n"
            "  -action_type=notify: Set action (BUY, SELL, HOLD, notify)\n"
            "/alerts edit ALERT_ID [PRICE] [CONDITION] [flags] - Edit alert\n"
            "/alerts delete ALERT_ID - Delete alert\n"
            "/alerts pause ALERT_ID - Pause alert\n"
            "/alerts resume ALERT_ID - Resume alert"
        )
