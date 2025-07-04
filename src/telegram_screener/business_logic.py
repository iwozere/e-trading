from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import pandas as pd
from src.telegram_screener.command_parser import ParsedCommand
from src.telegram_screener.data_provider_factory import get_downloader
import datetime

# Import actual data downloaders and indicator factory in real implementation
# from src.data.yahoo_data_downloader import YahooDataDownloader
# from src.data.binance_data_downloader import BinanceDataDownloader
# from src.indicator.indicator_factory import IndicatorFactory

@dataclass
class TickerAnalysis:
    ticker: str
    provider: str
    period: str
    interval: str
    ohlcv: Optional[pd.DataFrame] = None
    error: Optional[str] = None
    # Add more fields as needed (e.g., indicators, fundamentals, etc.)

def handle_command(parsed: ParsedCommand) -> Dict[str, Any]:
    """
    Main business logic handler. Dispatches based on command and parameters.
    Returns a dict with result/status/data for notification manager.
    """
    if parsed.command == "report":
        return handle_report(parsed)
    # Add more command handlers as needed
    return {"status": "error", "message": f"Unknown command: {parsed.command}"}


def handle_report(parsed: ParsedCommand) -> Dict[str, Any]:
    """
    Business logic for /report command.
    For each ticker:
      - Use analyze_ticker_business for unified analysis logic
    """
    args = parsed.args
    tickers = [args.get("tickers")] if isinstance(args.get("tickers"), str) else args.get("tickers", [])
    period = args.get("period") or "2y"
    interval = args.get("interval") or "1d"
    provider = args.get("provider")
    analyses: List[TickerAnalysis] = []
    for ticker in tickers:
        analysis = analyze_ticker_business(
            ticker=ticker,
            provider=provider,
            period=period,
            interval=interval
        )
        analyses.append(analysis)
    return {
        "status": "ok",
        "analyses": analyses,
        "email": args.get("email", False),
        "indicators": args.get("indicators"),
    }

def analyze_period_interval(period: str = "2y", interval: str = "1d"):
    """
    Given a period (e.g., '2y', '6mo', '1w') and interval, return (start_date, end_date) as strings.
    Assumes the combination is already validated for the provider.
    """
    start_date = None
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    if period.endswith("y"):
        years = int(period[:-1])
        start_date = (datetime.datetime.now() - datetime.timedelta(days=365*years)).strftime("%Y-%m-%d")
    elif period.endswith("m"):
        months = int(period[:-1])
        start_date = (datetime.datetime.now() - datetime.timedelta(days=30*months)).strftime("%Y-%m-%d")
    elif period.endswith("mo"):
        months = int(period[:-2])
        start_date = (datetime.datetime.now() - datetime.timedelta(days=30*months)).strftime("%Y-%m-%d")
    elif period.endswith("w"):
        weeks = int(period[:-1])
        start_date = (datetime.datetime.now() - datetime.timedelta(days=7*weeks)).strftime("%Y-%m-%d")
    else:
        start_date = (datetime.datetime.now() - datetime.timedelta(days=730)).strftime("%Y-%m-%d")
    return start_date, end_date


def analyze_ticker_business(
    ticker: str,
    provider: str = None,
    period: str = "2y",
    interval: str = "1d"
) -> TickerAnalysis:
    """
    Business logic: fetch OHLCV for ticker/provider/period/interval, return TickerAnalysis (only ohlcv/error for now).
    Checks period/interval validity before date calculation.
    """
    prov = provider or ("yf" if len(ticker) < 5 else "bnc")
    try:
        downloader = get_downloader(prov)
        # Check period/interval validity BEFORE date calculation
        if not downloader.is_valid_period_interval(period, interval):
            return TickerAnalysis(
                ticker=ticker,
                provider=prov,
                period=period,
                interval=interval,
                ohlcv=None,
                error=f"Invalid period/interval for provider {prov}"
            )
        start_date, end_date = analyze_period_interval(period, interval)
        df = downloader.get_ohlcv(ticker, interval, start_date, end_date)
        return TickerAnalysis(
            ticker=ticker,
            provider=prov,
            period=period,
            interval=interval,
            ohlcv=df,
            error=None
        )
    except Exception as e:
        return TickerAnalysis(
            ticker=ticker,
            provider=prov,
            period=period,
            interval=interval,
            ohlcv=None,
            error=str(e)
        )