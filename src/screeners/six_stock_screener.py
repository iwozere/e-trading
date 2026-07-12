"""
SIX (Swiss Exchange) stock screener.

Screens Swiss-listed stocks using fundamental + technical criteria.
Moved from src/util/six_stock_screener.py (UTIL-1).
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
    results_dir = PROJECT_ROOT / "results" / "screeners" / "six"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "six_selected_stocks.csv"
    df.to_csv(output_path, index=False)
    _logger.info("Results saved to %s", output_path)
    print(
        f"__SCHEDULER_RESULT__: {json.dumps({'result_count': len(df), 'tickers': list(df['Ticker'].head(10)) if not df.empty else []})}"
    )
