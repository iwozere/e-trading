"""
S&P 500 stock screener.

Screens S&P 500 stocks using fundamental + technical criteria.
Moved from src/util/sp500_stock_screener.py (UTIL-1).
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.screeners.stock_screener import StockScreener
from src.util.tickers_list import get_sp500_tickers_wikipedia

_logger = setup_logger(__name__)


class SP500StockScreener(StockScreener):
    def __init__(self):
        super().__init__(stock_data=None)


if __name__ == "__main__":
    _logger.info("Loading S&P 500 tickers...")
    tickers = get_sp500_tickers_wikipedia()
    _logger.info("Found %d tickers", len(tickers))
    _logger.info("Screening by fundamental and technical criteria...")
    screener = SP500StockScreener()
    df = screener.screen_stocks(tickers)
    _logger.info("Selected %d stocks", len(df))
    df.to_csv("sp500_selected_stocks.csv", index=False)
    _logger.info("Results saved to sp500_selected_stocks.csv")
    print(f"__SCHEDULER_RESULT__: {json.dumps({'result_count': len(df), 'tickers': list(df['Ticker'].head(10)) if not df.empty else []})}")
