import asyncio
from typing import Any, Dict, Optional
from src.telegram.command_parser import ParsedCommand
from src.notification.logger import setup_logger

# Import specialized service modules
from src.telegram.screener.user_service import UserService
from src.telegram.screener.alert_manager import AlertManager
from src.telegram.screener.schedule_manager import ScheduleManager
from src.telegram.screener.report_engine import ReportEngine
from src.telegram.screener.screener_engine import ScreenerEngine
from src.telegram.lifecycle import get_service_instances, set_service_instances

_logger = setup_logger(__name__)

class TelegramBusinessLogic:
    """
    Thin facade for telegram bot business logic.
    Delegates to specialized service modules for specific functional areas.
    """

    def __init__(self, telegram_service, indicator_service=None):
        self.telegram_service = telegram_service
        self.indicator_service = indicator_service
        
        # Initialize specialized services
        self.user_service = UserService(telegram_service)
        self.alert_manager = AlertManager(telegram_service)
        self.schedule_manager = ScheduleManager(telegram_service)
        self.report_engine = ReportEngine(telegram_service, indicator_service)
        self.screener_engine = ScreenerEngine(telegram_service, indicator_service)

    async def handle_command(self, parsed: ParsedCommand) -> Dict[str, Any]:
        """Dispatches commands to the appropriate service handler."""
        try:
            cmd = parsed.command.lstrip('/')
            
            # User management commands
            if cmd == "register": return self.user_service.handle_register(parsed)
            elif cmd == "verify": return self.user_service.handle_verify(parsed)
            elif cmd == "info": return self.user_service.handle_info(parsed)
            elif cmd == "language": return self.user_service.handle_language(parsed)
            elif cmd == "request_approval": return self.user_service.handle_request_approval(parsed)
            
            # Alert commands
            elif cmd == "alerts": return self.alert_manager.handle_alerts(parsed)
            
            # Schedule commands
            elif cmd == "schedules": return self.schedule_manager.handle_schedules(parsed)
            
            # Report and Screener commands
            elif cmd == "report": return await self.report_engine.handle_report(parsed)
            elif cmd == "screener": return await self.screener_engine.handle_screener(parsed)
            
            # Admin commands
            elif cmd == "admin": return self.handle_admin(parsed)
            
            # Help
            elif cmd == "help": return self.handle_help(parsed)
            
            # Feedback / Feature (delegated to user_service or helper)
            elif cmd == "feedback": return self.handle_feedback(parsed)
            elif cmd == "feature": return self.handle_feature(parsed)
            
            return {"status": "error", "message": f"Unknown command: {cmd}"}
        except Exception as e:
            _logger.exception("Error handling command %s", parsed.command)
            return {"status": "error", "message": f"Internal error: {str(e)}"}

    def handle_help(self, parsed: ParsedCommand) -> Dict[str, Any]:
        """Business logic for /help command."""
        telegram_user_id = parsed.args.get("telegram_user_id")
        is_admin = self.user_service.is_admin_user(telegram_user_id)
        
        help_text = (
            "**Screener Bot Help**\n\n"
            "**Commands:**\n"
            "/info - Show your account status\n"
            "/register <email> - Register your email\n"
            "/verify <code> - Verify your email\n"
            "/report <ticker> - Get real-time technical analysis\n"
            "/screener <name/json> - Run stock screener\n"
            "/alerts - Manage price/indicator alerts\n"
            "/schedules - Manage scheduled reports\n"
            "/language <en/ru> - Set your language\n"
            "/feedback <text> - Send feedback\n"
            "/feature <text> - Request a feature\n"
            "/request_approval - Request access to restricted features\n"
            "/help - Show this help message"
        )
        
        if is_admin:
            help_text += "\n\n**Admin Commands:**\n/admin users - List users\n/admin pending - Pending approvals"
            
        return {"status": "ok", "title": "Help", "message": help_text}

    def handle_feedback(self, parsed: ParsedCommand) -> Dict[str, Any]:
        """Delegates feedback collection."""
        tid = parsed.args.get("telegram_user_id")
        try:
            msg = parsed.args.get("feedback") or " ".join(parsed.positionals)
            if self.telegram_service:
                self.telegram_service.add_feedback(tid, "feedback", msg)
            return {"status": "ok", "message": "Feedback received. Thank you!"}
        except Exception:
            _logger.exception("Error saving feedback for user %s", tid)
            return {"status": "error", "message": "Failed to save feedback."}

    def handle_feature(self, parsed: ParsedCommand) -> Dict[str, Any]:
        """Delegates feature request collection."""
        tid = parsed.args.get("telegram_user_id")
        try:
            msg = parsed.args.get("feature") or " ".join(parsed.positionals)
            if self.telegram_service:
                self.telegram_service.add_feedback(tid, "feature_request", msg)
            return {"status": "ok", "message": "Feature request received. Thank you!"}
        except Exception:
            _logger.exception("Error saving feature request for user %s", tid)
            return {"status": "error", "message": "Failed to save feature request."}

    def handle_admin(self, parsed: ParsedCommand) -> Dict[str, Any]:
        """Delegates admin commands to user_service."""
        access_check = self.user_service.check_admin_access(parsed.args.get("telegram_user_id"))
        if access_check["status"] != "ok": return access_check

        action = parsed.positionals[0] if parsed.positionals else None
        params = parsed.positionals[1:] if len(parsed.positionals) > 1 else []
        
        if action in ["users", "listusers"]: return self.user_service.handle_admin_list_users(parsed.args.get("telegram_user_id"))
        elif action == "pending": return self.user_service.handle_admin_list_pending_approvals(parsed.args.get("telegram_user_id"))
        elif action == "resetemail" and params: return self.user_service.handle_admin_reset_email(params[0])
        elif action == "verify" and params: return self.user_service.handle_admin_verify_user(params[0])
        elif action == "approve" and params: return self.user_service.handle_admin_approve_user(params[0])
        elif action == "reject" and params: return self.user_service.handle_admin_reject_user(params[0])
        elif action == "setlimit" and len(params) >= 2:
            return self.user_service.handle_admin_set_limit(params[0], params[1], params[2] if len(params) > 2 else None)
        
        return {"status": "error", "message": f"Unknown admin action: {action}"}

# Service instances are now managed in src.telegram.lifecycle

# ── Cached facade (P2-TG-2) ──────────────────────────────────────────────────
# Re-creating TelegramBusinessLogic (and all its sub-services) on every command
# call is wasteful.  This cached getter rebuilds only when the underlying service
# instances change (e.g. after a lifecycle restart).

_cached_logic: Optional["TelegramBusinessLogic"] = None
_cached_ts = None   # telegram_service identity used to detect staleness
_cached_ids = None  # indicator_service identity used to detect staleness


def get_business_logic() -> Optional["TelegramBusinessLogic"]:
    """Return a cached TelegramBusinessLogic, rebuilding if services have changed."""
    global _cached_logic, _cached_ts, _cached_ids
    ts, ids = get_service_instances()
    if ts is None:
        return None
    if _cached_logic is None or _cached_ts is not ts or _cached_ids is not ids:
        _cached_logic = TelegramBusinessLogic(ts, ids)
        _cached_ts = ts
        _cached_ids = ids
    return _cached_logic


async def handle_command(parsed: ParsedCommand) -> Dict[str, Any]:
    """Standalone entry point used by notifications.py."""
    logic = get_business_logic()
    if logic is None:
        return {"status": "error", "message": "Service temporarily unavailable"}
    return await logic.handle_command(parsed)

# Legacy / Exported functions for backward compatibility if needed
def is_admin_user(user_id: str) -> bool:
    ts, _ = get_service_instances()
    return UserService(ts).is_admin_user(user_id) if ts else False

def is_approved_user(user_id: str) -> bool:
    ts, _ = get_service_instances()
    return UserService(ts).is_approved_user(user_id) if ts else False

def check_approved_access(user_id: str) -> Dict[str, Any]:
    ts, _ = get_service_instances()
    return UserService(ts).check_approved_access(user_id) if ts else {"status": "error", "message": "Service unavailable"}

async def get_ohlcv(ticker: str, period: str = "1y", interval: str = "1d", provider: str = None) -> Any:
    """Wrapper for IndicatorService.get_ohlcv."""
    _, isvc = get_service_instances()
    if isvc:
        return await isvc.get_ohlcv(ticker, period, interval, provider)
    return None

async def get_indicator_data(ticker: str, indicators: list, period: str = "1y", interval: str = "1d") -> Dict[str, Any]:
    """Wrapper for IndicatorService.analyze_ticker."""
    _, isvc = get_service_instances()
    if isvc:
        return await isvc.analyze_ticker(ticker, indicators, period, interval)
    return {"status": "error", "message": "Service unavailable"}
async def get_fundamentals(ticker: str, provider: str = None) -> Any:
    """Wrapper for src.common.fundamentals.get_fundamentals."""
    from src.common.fundamentals import get_fundamentals as gf
    return gf(ticker, provider)

async def analyze_ticker_business(ticker: str, provider: str = None, period: str = "2y", interval: str = "1d") -> Any:
    """Legacy wrapper for TickerAnalyzer.analyze_ticker."""
    from src.common.ticker_analyzer import analyze_ticker
    return await analyze_ticker(ticker=ticker, provider=provider, period=period, interval=interval)
