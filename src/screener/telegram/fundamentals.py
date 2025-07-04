import yfinance as yf
from src.notification.logger import setup_logger
from src.model.model import Fundamentals

logger = setup_logger("telegram_bot")


def get_fundamentals(ticker: str) -> Fundamentals:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        if not info:
            logger.error("No data returned from yfinance for ticker %s", ticker, exc_info=True)
            return Fundamentals(
                ticker=ticker.upper(),
                company_name="Unknown",
                current_price=0.0,
                market_cap=0.0,
                pe_ratio=0.0,
                forward_pe=0.0,
                dividend_yield=0.0,
                earnings_per_share=0.0,
            )

        logger.debug(
            "Retrieved fundamentals for %s: %s", ticker, info.get('shortName', 'Unknown')
        )

        return Fundamentals(
            ticker=ticker.upper(),
            company_name=info.get("longName", "Unknown"),
            current_price=info.get("regularMarketPrice", 0.0),
            market_cap=info.get("marketCap", 0.0),
            pe_ratio=info.get("trailingPE", 0.0),
            forward_pe=info.get("forwardPE", 0.0),
            dividend_yield=info.get("dividendYield", 0.0),
            earnings_per_share=info.get("trailingEps", 0.0),
        )
    except Exception as e:
        logger.error("Failed to get fundamentals for %s: %s", ticker, e, exc_info=True)
        return Fundamentals(
            ticker=ticker.upper(),
            company_name="Unknown",
            current_price=0.0,
            market_cap=0.0,
            pe_ratio=0.0,
            forward_pe=0.0,
            dividend_yield=0.0,
            earnings_per_share=0.0,
        )
