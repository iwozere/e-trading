"""
handlers/alerts.py — /alerts command handler.
"""
from aiogram import Dispatcher
from aiogram.types import Message
from aiogram.filters import Command

from src.notification.logger import setup_logger
from src.telegram.command_parser import parse_command
from src.model.telegram_bot import ParsedCommand
from src.telegram.lifecycle import get_service_instances

_logger = setup_logger("telegram_screener_bot")


async def process_alerts_command_immediate(user_id: str, parsed: ParsedCommand, message: Message) -> None:
    """Handle /alerts [add|delete|evaluate] [params]."""
    try:
        from src.data.db.services.alerts_service import AlertsService
        from src.data.data_manager import DataManager
        from src.data.db.services.jobs_service import JobsService

        telegram_svc, indicator_svc = get_service_instances()
        if not telegram_svc:
            await message.reply("❌ Service temporarily unavailable. Please try again later.")
            return

        data_manager = DataManager()
        jobs_service = JobsService()
        alerts_service = AlertsService(jobs_service, data_manager, indicator_svc)

        action = parsed.args.get("action", "").lower()
        params = parsed.args.get("params", [])
        want_email = parsed.args.get("email", False)

        if not action:
            # List user's alerts
            alerts = alerts_service.get_user_alerts(int(user_id), active_only=True)
            if not alerts:
                await message.reply("📋 You have no active alerts.")
                return

            response = "🔔 **Your Active Alerts:**\n\n"
            for alert in alerts[:10]:
                status_icon = "✅" if alert["enabled"] else "⏸️"
                response += f"{status_icon} **{alert['ticker']}** ({alert['timeframe']})\n"
                response += f"   ID: {alert['id']} | Created: {alert['created_at'].strftime('%Y-%m-%d')}\n\n"
            if len(alerts) > 10:
                response += f"... and {len(alerts) - 10} more alerts\n\n"

            response += "Use `/alerts add TICKER PRICE above/below` to add alerts\n"
            response += "Use `/alerts delete ID` to remove alerts"
            await message.reply(response, parse_mode="Markdown")
            return

        if action == "add" and len(params) >= 3:
            # Add new alert: /alerts add TICKER PRICE above/below
            ticker = params[0].upper()
            try:
                price = float(params[1])
                condition = params[2].lower()

                if condition not in ["above", "below"]:
                    await message.reply("❌ Condition must be 'above' or 'below'")
                    return

                # Create alert configuration
                alert_config = {
                    "ticker": ticker,
                    "timeframe": "15m",  # Default timeframe
                    "rule": {
                        "type": "price",
                        "condition": condition,
                        "value": price
                    },
                    "notify": {
                        "telegram": True,
                        "email": want_email
                    }
                }

                result = await alerts_service.create_alert(int(user_id), alert_config)
                if result["success"]:
                    email_text = " and email" if want_email else ""
                    await message.reply(f"✅ Alert created for **{ticker}** {condition} ${price:,.2f}\n"
                                      f"You'll be notified via Telegram{email_text} when triggered.",
                                      parse_mode="Markdown")
                else:
                    await message.reply(f"❌ Failed to create alert: {result.get('error', 'Unknown error')}")

            except ValueError:
                await message.reply("❌ Invalid price. Please use a number.")

        elif action == "delete" and len(params) >= 1:
            # Delete alert: /alerts delete ID
            try:
                alert_id = int(params[0])
                success = alerts_service.delete_alert(alert_id)
                if success:
                    await message.reply(f"✅ Alert {alert_id} deleted successfully.")
                else:
                    await message.reply(f"❌ Failed to delete alert {alert_id}. Check the ID and try again.")
            except ValueError:
                await message.reply("❌ Invalid alert ID. Please use a number.")

        elif action == "evaluate":
            # Evaluate user's alerts: /alerts evaluate
            await message.reply("🔄 Evaluating your alerts...")
            results = await alerts_service.evaluate_user_alerts(int(user_id))

            response = "📊 **Alert Evaluation Results:**\n\n"
            response += f"✅ Evaluated: {results['total_evaluated']}\n"
            response += f"🔥 Triggered: {results['triggered']}\n"
            response += f"🔄 Rearmed: {results['rearmed']}\n"
            response += f"❌ Errors: {results['errors']}\n"

            if results['triggered'] > 0:
                response += "\n🚨 **Triggered Alerts:**\n"
                for result in results['results']:
                    if result['triggered']:
                        response += f"• {result['ticker']}\n"
            await message.reply(response, parse_mode="Markdown")

        else:
            # Show help
            help_text = ("🔔 **Alert Commands:**\n\n"
                        "`/alerts` - List your alerts\n"
                        "`/alerts add TICKER PRICE above/below` - Add price alert\n"
                        "`/alerts delete ID` - Delete alert\n"
                        "`/alerts evaluate` - Check your alerts now\n\n"
                        "**Examples:**\n"
                        "`/alerts add BTCUSDT 65000 above`\n"
                        "`/alerts add AAPL 150 below -email`\n"
                        "`/alerts delete 123`")
            await message.reply(help_text, parse_mode="Markdown")
    except Exception:
        _logger.exception("Error processing /alerts for user %s", user_id)
        await message.reply("❌ An error occurred while processing your alert command. Please try again.")


def register(dp: Dispatcher) -> None:
    """Register the /alerts handler onto the Dispatcher."""
    from src.telegram.handlers.common import audit_command_wrapper

    @dp.message(Command("alerts"))
    async def cmd_alerts(msg: Message):
        parsed = parse_command(msg.text)
        await audit_command_wrapper(msg, process_alerts_command_immediate, str(msg.from_user.id), parsed, msg)
