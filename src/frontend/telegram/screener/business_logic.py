from typing import Any, Dict, List
from src.frontend.telegram.command_parser import ParsedCommand
from src.common import get_ohlcv, analyze_period_interval
from src.common.fundamentals import get_fundamentals
from src.common.technicals import calculate_technicals_from_df
from src.model.telegram_bot import TickerAnalysis
from src.frontend.telegram import db
import os
from config.donotshare import donotshare
from src.common.ticker_analyzer import format_ticker_report

from src.notification.logger import setup_logger
logger = setup_logger("telegram_screener_bot")

def handle_command(parsed: ParsedCommand) -> Dict[str, Any]:
    """
    Main business logic handler. Dispatches based on command and parameters.
    Returns a dict with result/status/data for notification manager.
    """
    if parsed.command == "report":
        return handle_report(parsed)
    elif parsed.command == "help":
        return handle_help(parsed)
    elif parsed.command == "info":
        return handle_info(parsed)
    # Add more command handlers as needed
    return {"status": "error", "message": f"Unknown command: {parsed.command}"}


def handle_help(parsed: ParsedCommand) -> Dict[str, Any]:
    """
    Business logic for /help and /start commands.
    Returns appropriate help text based on user admin status.
    """
    try:
        telegram_user_id = parsed.args.get("telegram_user_id")
        if not telegram_user_id:
            return {"status": "error", "message": "No telegram_user_id provided"}

        db.init_db()
        is_admin = is_admin_user(telegram_user_id)

        # Import help texts here to avoid circular imports
        from src.frontend.telegram.bot import HELP_TEXT, ADMIN_HELP_TEXT

        # Show regular help text
        help_text = HELP_TEXT

        # Add admin commands if user is admin
        if is_admin:
            help_text += "\n\n" + ADMIN_HELP_TEXT

        return {
            "status": "ok",
            "help_text": help_text,
            "is_admin": is_admin
        }
    except Exception as e:
        logger.error("Error: %s", e, exc_info=True)
        return {"status": "error", "message": f"Error generating help: {str(e)}"}


def is_admin_user(telegram_user_id: str) -> bool:
    """Check if user is an admin."""
    db.init_db()
    status = db.get_user_status(telegram_user_id)
    return status and status.get("is_admin", False)


def handle_report(parsed: ParsedCommand) -> Dict[str, Any]:
    """
    Business logic for /report command.
    For each ticker:
      - Use analyze_ticker_business for unified analysis logic
      - Use format_ticker_report to generate message and chart
    """
    args = parsed.args
    tickers_raw = args.get("tickers")
    if isinstance(tickers_raw, str):
        tickers = [tickers_raw]
    elif isinstance(tickers_raw, list):
        tickers = tickers_raw
    else:
        tickers = parsed.positionals
    if not tickers:
        return {"status": "error", "title": "Report Error", "message": "No tickers specified"}
    period = args.get("period") or "2y"
    interval = args.get("interval") or "1d"
    provider = args.get("provider")
    reports = []
    missing_keys = []
    # Check for required API keys
    provider_keys = {
        "av": getattr(donotshare, "ALPHA_VANTAGE_KEY", None),
        "fh": getattr(donotshare, "FINNHUB_KEY", None),
        "td": getattr(donotshare, "TWELVE_DATA_KEY", None),
        "pg": getattr(donotshare, "POLYGON_KEY", None),
    }
    for k, v in provider_keys.items():
        if not v:
            missing_keys.append(k)
    all_failed = True
    for ticker in tickers:
        analysis = analyze_ticker_business(
            ticker=ticker,
            provider=provider,
            period=period,
            interval=interval
        )
        report = format_ticker_report(analysis)
        report['ticker'] = ticker
        report['error'] = analysis.error if analysis.error else None
        reports.append(report)
        if not analysis.error:
            all_failed = False
    # If all analyses failed due to missing keys
    if all_failed and any(report['error'] and any(key in report['error'] for key in ["Alpha Vantage API key", "Finnhub API key", "Twelve Data API key", "Polygon.io API key"]) for report in reports):
        return {
            "status": "error",
            "title": "Report Error",
            "message": f"No data could be retrieved for {', '.join(tickers)}. Missing or invalid API keys for providers: {', '.join(missing_keys)}. Please check your API keys in donotshare.py."
        }
    # If all analyses failed for any reason
    if all_failed:
        return {
            "status": "error",
            "title": "Report Error",
            "message": f"No data could be retrieved for {', '.join(tickers)}. Please check your API keys or try a different provider/ticker."
        }
    # Otherwise, return reports for Telegram/email delivery
    return {
        "status": "ok",
        "reports": reports,
        "email": args.get("email", False),
        "title": f"Report for {', '.join(tickers)}",
        "message": "Report generated successfully."
    }


def analyze_ticker_business(
    ticker: str,
    provider: str = None,
    period: str = "2y",
    interval: str = "1d"
) -> TickerAnalysis:
    """
    Business logic: fetch OHLCV for ticker/provider/period/interval, return TickerAnalysis.
    Uses common functions from src/common for data retrieval and analysis.
    """
    try:
        # Get OHLCV data using common function
        df = get_ohlcv(ticker, interval, period, provider)

        # Get fundamentals using common function
        fundamentals = get_fundamentals(ticker, provider)

        # Calculate technical indicators
        df_with_technicals, technicals = calculate_technicals_from_df(df)

        # Calculate current price and change percentage
        current_price = None
        change_percentage = None
        if df is not None and not df.empty:
            current_price = df['close'].iloc[-1]
            if len(df) > 1:
                prev_price = df['close'].iloc[-2]
                change_percentage = ((current_price - prev_price) / prev_price) * 100

        return TickerAnalysis(
            ticker=ticker.upper(),
            provider=provider or ("yf" if len(ticker) < 5 else "bnc"),
            period=period,
            interval=interval,
            ohlcv=df_with_technicals,
            fundamentals=fundamentals,
            technicals=technicals,
            current_price=current_price,
            change_percentage=change_percentage,
            error=None,
            chart_image=None
        )
    except Exception as e:
        return TickerAnalysis(
            ticker=ticker.upper(),
            provider=provider or ("yf" if len(ticker) < 5 else "bnc"),
            period=period,
            interval=interval,
            ohlcv=None,
            fundamentals=None,
            technicals=None,
            current_price=None,
            change_percentage=None,
            error=str(e),
            chart_image=None
        )


def handle_info(parsed: ParsedCommand) -> Dict[str, Any]:
    telegram_user_id = parsed.args.get("telegram_user_id")
    if not telegram_user_id:
        return {"status": "error", "message": "No telegram_user_id provided"}
    db.init_db()
    status = db.get_user_status(telegram_user_id)
    if status:
        email = status["email"] or "(not set)"
        verified = "Yes" if status["verified"] else "No"
        language = status["language"] or "(not set)"
        return {
            "status": "ok",
            "title": "Your Info",
            "message": f"Email: {email}\nVerified: {verified}\nLanguage: {language}"
        }
    else:
        return {
            "status": "ok",
            "title": "Your Info",
            "message": "Email: (not set)\nVerified: No\nLanguage: (not set)"
        }