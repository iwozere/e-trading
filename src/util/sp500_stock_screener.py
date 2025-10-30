import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.util.stock_screener import StockScreener
from src.util.tickers_list import get_sp500_tickers_wikipedia

"""
Screening criteria:
- P/E < 25 — reasonable valuation
- ROE > 15% — capital efficiency
- Debt/Equity < 100% — moderate debt
- Free Cash Flow > 0
- Price > 50D > 200D — positive momentum
"""


class SP500StockScreener(StockScreener):
    def __init__(self):
        super().__init__(stock_data=None)


if __name__ == "__main__":
    print("Loading S&P 500 tickers...")
    tickers = get_sp500_tickers_wikipedia()
    print(f"Found {len(tickers)} tickers")
    print("Screening by fundamental and technical criteria...")
    screener = SP500StockScreener()
    df = screener.screen_stocks(tickers)
    print(f"\n=== Selected {len(df)} stocks ===")
    print(df)
    df.to_csv("sp500_selected_stocks.csv", index=False)
    print("Results saved to sp500_selected_stocks.csv")
