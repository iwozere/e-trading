from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import pandas as pd
from src.telegram_screener.command_parser import ParsedCommand
from src.common import get_ohlcv, analyze_period_interval
from src.common.fundamentals import get_fundamentals
from src.common.technicals import calculate_technicals_from_df
from src.model.telegram_bot import TickerAnalysis

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

        return TickerAnalysis(
            ticker=ticker.upper(),
            provider=provider or ("yf" if len(ticker) < 5 else "bnc"),
            period=period,
            interval=interval,
            ohlcv=df_with_technicals,
            fundamentals=fundamentals,
            technicals=technicals,
            error=None
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
            error=str(e)
        )