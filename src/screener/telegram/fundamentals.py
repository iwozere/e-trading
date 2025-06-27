import yfinance as yf
from src.notification.logger import setup_logger

logger = setup_logger("telegram_bot")


def get_fundamentals(ticker: str) -> dict:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        if not info:
            logger.error(f"No data returned from yfinance for ticker {ticker}")
            return {}

        logger.debug(
            f"Retrieved fundamentals for {ticker}: {info.get('shortName', 'Unknown')}"
        )

        return {
            "ticker": ticker.upper(),
            "company_name": info.get("shortName", "Unknown"),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "dividend_yield": info.get("dividendYield"),
            "earnings_per_share": info.get("trailingEps"),
        }
    except Exception as e:
        logger.error(f"Failed to get fundamentals for {ticker}: {str(e)}", exc_info=e)
        return {}
