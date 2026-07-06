from typing import Any, Dict

from src.common.ticker_analyzer import analyze_ticker, format_ticker_report
from src.model.telegram_bot import TickerAnalysis
from src.notification.logger import setup_logger
from src.telegram.command_parser import ParsedCommand
from src.telegram.screener.report_config_parser import ReportConfigParser

_logger = setup_logger(__name__)


class ReportEngine:
    """
    Handles ticker analysis and report generation logic.
    """

    def __init__(self, telegram_service=None, indicator_service=None):
        self.telegram_service = telegram_service
        self.indicator_service = indicator_service

    async def handle_report(self, parsed: ParsedCommand) -> Dict[str, Any]:
        """Business logic for /report command."""
        args = parsed.args
        telegram_user_id = args.get("telegram_user_id")

        # Config parsing
        config_json = args.get("config")
        if config_json:
            try:
                is_valid, errors = ReportConfigParser.validate_report_config(config_json)
                if not is_valid:
                    return {
                        "status": "error",
                        "title": "Report Error",
                        "message": f"Invalid config: {'; '.join(errors)}",
                    }
                report_config = ReportConfigParser.parse_report_config(config_json)
                if not report_config:
                    return {"status": "error", "message": "Failed to parse report configuration"}
                tickers = [t.upper() for t in report_config.tickers]
                period = report_config.period
                interval = report_config.interval
                provider: str | None = report_config.provider
                email = report_config.email
            except Exception as e:
                return {"status": "error", "message": f"JSON config error: {str(e)}"}
        else:
            tickers = [t.upper() for t in (args.get("tickers") or parsed.positionals)]
            if not tickers:
                return {"status": "error", "message": "No tickers specified"}
            period = args.get("period") or "2y"
            interval = args.get("interval") or "1d"
            provider = args.get("provider")
            email = args.get("email", False)

        # Get user email
        user_email = None
        if telegram_user_id and self.telegram_service:
            status = self.telegram_service.get_user_status(telegram_user_id)
            if status:
                user_email = status.get("email")

        reports = []
        all_failed = True
        for ticker in tickers:
            analysis = await self.analyze_ticker_business(ticker, provider, period, interval)
            report = format_ticker_report(analysis)
            report["ticker"] = ticker
            report["error"] = analysis.error
            reports.append(report)
            if not analysis.error:
                all_failed = False

        if all_failed:
            return {
                "status": "error",
                "title": "Report Error",
                "message": f"No data retrieved for {', '.join(tickers)}.",
            }

        return {
            "status": "ok",
            "reports": reports,
            "email": email,
            "user_email": user_email,
            "title": f"Report for {', '.join(tickers)}",
            "message": "Report generated successfully.",
        }

    async def analyze_ticker_business(
        self, ticker: str, provider: str | None = None, period: str = "2y", interval: str = "1d", force_refresh: bool = True
    ) -> TickerAnalysis:
        """Fetch OHLCV and return TickerAnalysis."""
        try:
            return await analyze_ticker(
                ticker=ticker, provider=provider, period=period, interval=interval, force_refresh=force_refresh
            )
        except Exception as e:
            _logger.exception(f"Error analyzing {ticker}")
            return TickerAnalysis(
                ticker=ticker.upper(), provider=provider or "unknown", period=period, interval=interval, error=str(e)
            )
