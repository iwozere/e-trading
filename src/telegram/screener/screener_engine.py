import json
from pathlib import Path
from typing import Any, Dict

from src.notification.logger import setup_logger
from src.telegram.command_parser import ParsedCommand

_logger = setup_logger(__name__)


class ScreenerEngine:
    """
    Handles immediate and scheduled screener orchestration and predefined strategy configurations.
    """

    def __init__(self, telegram_service=None, indicator_service=None):
        self.telegram_service = telegram_service
        self.indicator_service = indicator_service

    async def handle_screener(self, parsed: ParsedCommand) -> Dict[str, Any]:
        """Business logic for /screener command for immediate execution."""
        try:
            telegram_user_id = parsed.args.get("telegram_user_id")
            config_json = parsed.args.get("screener_name_or_config")
            send_email = parsed.args.get("email", False)

            if not telegram_user_id:
                return {"status": "error", "message": "No telegram_user_id provided"}
            if not config_json:
                return {"status": "error", "message": "Please provide screener name or configuration."}

            from src.telegram.screener.enhanced_screener import EnhancedScreener
            from src.telegram.screener.screener_config_parser import parse_screener_config, validate_screener_config

            if config_json.startswith("{"):
                is_valid, errors = validate_screener_config(config_json)
                if not is_valid:
                    return {"status": "error", "message": f"Invalid config: {errors}"}
                screener_config = parse_screener_config(config_json)
            else:
                screener_config = self._get_predefined_screener_config(config_json)
                if not screener_config:
                    return {
                        "status": "error",
                        "message": f"Unknown screener: {config_json}. Available: {', '.join(self._get_available_screeners())}",
                    }

            if not self.indicator_service:
                from src.indicators.service import get_unified_indicator_service

                self.indicator_service = get_unified_indicator_service()

            enhanced_screener = EnhancedScreener(indicator_service=self.indicator_service)
            report = await enhanced_screener.run_enhanced_screener(screener_config)

            if report.error:
                return {"status": "error", "message": report.error}
            message = enhanced_screener.format_enhanced_telegram_message(report, screener_config)

            if send_email:
                if not self.telegram_service:
                    return {"status": "error", "message": "Service unavailable"}
                status = self.telegram_service.get_user_status(telegram_user_id)
                if not status or not status.get("email"):
                    return {"status": "error", "message": "Email not registered. Use /register first."}
                from src.telegram.screener.notifications import send_screener_email

                await send_screener_email(status["email"], report, screener_config, telegram_user_id)
                return {"status": "success", "message": "Screener results sent to your email"}
            else:
                return {"status": "success", "message": message, "report": report}
        except Exception as e:
            _logger.exception("Error in screener command")
            return {"status": "error", "message": f"Screener error: {str(e)}"}

    def _get_predefined_screener_config(self, screener_name: str):
        """Get predefined screener configuration by name."""
        try:
            from src.telegram.screener.screener_config_parser import ScreenerConfigParser

            config_path = Path(__file__).resolve().parents[4] / "config" / "screener" / "fmp_screener_criteria.json"
            if not config_path.exists():
                return None
            with open(config_path) as f:
                fmp_config = json.load(f)

            strategy = fmp_config.get("predefined_strategies", {}).get(screener_name)
            if not strategy:
                return None

            config_dict = {
                "screener_name": screener_name,
                "screener_type": "hybrid",
                "list_type": self._get_list_type_for_screener(screener_name),
                "fmp_criteria": strategy["criteria"],
                "fundamental_criteria": self._get_fundamental_criteria_for_screener(screener_name),
                "technical_criteria": self._get_technical_criteria_for_screener(screener_name),
                "max_results": strategy["criteria"].get("limit", 50),
                "min_score": 0.5,
                "period": "1y",
                "interval": "1d",
            }
            return ScreenerConfigParser()._parse_config_dict(config_dict)
        except Exception:
            _logger.exception(f"Error loading config for {screener_name}")
            return None

    def _get_list_type_for_screener(self, screener_name: str) -> str:
        if screener_name == "six_stocks":
            return "swiss_shares"
        elif screener_name == "mid_cap_stocks":
            return "us_medium_cap"
        elif screener_name in ["large_cap_stocks", "extra_large_cap_stocks"]:
            return "us_large_cap"
        return "us_medium_cap"

    def _get_fundamental_criteria_for_screener(self, screener_name: str):
        base_criteria = [
            {"indicator": "PE", "operator": "max", "value": 30, "weight": 1.0, "required": False},
            {"indicator": "PB", "operator": "max", "value": 3.0, "weight": 1.0, "required": False},
            {"indicator": "ROE", "operator": "min", "value": 0.10, "weight": 1.0, "required": False},
        ]
        if screener_name == "conservative_value":
            base_criteria[0]["value"], base_criteria[1]["value"], base_criteria[2]["value"] = 12, 1.2, 0.15
        elif screener_name == "deep_value":
            base_criteria[0]["value"], base_criteria[1]["value"], base_criteria[2]["value"] = 8, 0.8, 0.08
        elif screener_name == "quality_growth":
            base_criteria[0]["value"], base_criteria[1]["value"], base_criteria[2]["value"] = 25, 5.0, 0.18
        # ... other strategies (could be expanded)
        return base_criteria

    def _get_technical_criteria_for_screener(self, screener_name: str):
        return [
            {
                "indicator": "RSI",
                "parameters": {"period": 14},
                "condition": {"operator": "<", "value": 75},
                "weight": 0.5,
                "required": False,
            }
        ]

    def _get_available_screeners(self):
        try:
            config_path = Path(__file__).resolve().parents[4] / "config" / "screener" / "fmp_screener_criteria.json"
            if not config_path.exists():
                return []
            with open(config_path) as f:
                fmp_config = json.load(f)
            return list(fmp_config.get("predefined_strategies", {}).keys())
        except:
            return []
