"""
SIX (Swiss Exchange) stock screener.

Screens Swiss-listed stocks using fundamental + technical criteria.
Moved from src/util/six_stock_screener.py (UTIL-1).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.screeners.stock_screener import StockScreener
from src.util.tickers_list import get_six_tickers

_logger = setup_logger(__name__)


class SixStockScreener(StockScreener):
    def __init__(self):
        super().__init__(stock_data=None)


if __name__ == "__main__":
    _logger.info("Loading SIX tickers...")
    tickers = get_six_tickers()
    _logger.info("Found %d tickers", len(tickers))
    _logger.info("Screening by fundamental and technical criteria...")
    screener = SixStockScreener()
    df = screener.screen_stocks(tickers)
    _logger.info("Selected %d stocks", len(df))
    df.to_csv("six_selected_stocks.csv", index=False)
    _logger.info("Results saved to six_selected_stocks.csv")
