"""
StockScreener — fundamental + technical stock screening.

Screens stocks on:
- P/E < 25 — reasonable valuation
- ROE > 15% — capital efficiency
- Debt/Equity < 100% — moderate debt
- Free Cash Flow > 0
- Price > 50D > 200D — positive momentum

Moved from src/util/stock_screener.py (UTIL-1: screener logic belongs in src/screeners).
A backward-compatible re-export remains at src/util/stock_screener.py.
"""

import pandas as pd
import yfinance as yf

from typing import Any

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class StockScreener:
    def __init__(self, stock_data):
        self.stock_data = stock_data

    def filter_by_price(self, min_price, max_price):
        return [stock for stock in self.stock_data if min_price <= stock["price"] <= max_price]

    def filter_by_volume(self, min_volume):
        return [stock for stock in self.stock_data if stock["volume"] >= min_volume]

    def filter_by_market_cap(self, min_market_cap):
        return [stock for stock in self.stock_data if stock["market_cap"] >= min_market_cap]

    def screen_stocks(self, tickers, suffix=""):  # suffix for Swiss (SIX) shares is .SW
        results = []
        for ticker in tickers:
            try:
                symbol = f"{ticker}{suffix}"
                stock = yf.Ticker(symbol)
                info = stock.info
                pe = info.get("trailingPE", None)
                roe = info.get("returnOnEquity", None)
                debt_equity = info.get("debtToEquity", None)
                price = info.get("currentPrice", None)
                fifty_day = info.get("fiftyDayAverage", None)
                two_hundred_day = info.get("twoHundredDayAverage", None)
                fcf_info = self.get_fcf_growth(symbol)
                fcf_latest = None
                ocf_latest = None
                capex_latest = None
                if fcf_info and isinstance(fcf_info, dict) and "FCF (oldest → newest)" in fcf_info:
                    fcf_list = fcf_info["FCF (oldest → newest)"]
                    if fcf_list:
                        fcf_latest = fcf_list[-1]
                    if "OCF (oldest → newest)" in fcf_info:
                        ocf_latest = (
                            fcf_info["OCF (oldest → newest)"][-1] if fcf_info["OCF (oldest → newest)"] else None
                        )
                    if "CapEx (oldest → newest)" in fcf_info:
                        capex_latest = (
                            fcf_info["CapEx (oldest → newest)"][-1] if fcf_info["CapEx (oldest → newest)"] else None
                        )
                if (
                    pe is not None
                    and roe is not None
                    and debt_equity is not None
                    and price is not None
                    and fifty_day is not None
                    and two_hundred_day is not None
                    and fcf_latest is not None
                ):
                    if (
                        pe < 25
                        and roe > 0.15
                        and debt_equity < 100
                        and fcf_latest > 0
                        and price > fifty_day > two_hundred_day
                    ):
                        results.append(
                            {
                                "Ticker": ticker,
                                "P/E": round(pe, 2),
                                "ROE %": round(roe * 100, 1),
                                "D/E": round(debt_equity, 1),
                                "Price": price,
                                "50D Avg": round(fifty_day, 2),
                                "200D Avg": round(two_hundred_day, 2),
                                "OCF": ocf_latest,
                                "CapEx": capex_latest,
                                "FCF": fcf_latest,
                            }
                        )
            except Exception:
                _logger.exception("Error screening ticker %s", ticker)
        return pd.DataFrame(results)

    @staticmethod
    def get_fcf_growth(ticker: str) -> dict[str, Any] | None:
        try:
            t = yf.Ticker(ticker)
            cf = t.cashflow
            if (
                cf.empty
                or "Total Cash From Operating Activities" not in cf.index
                or "Capital Expenditures" not in cf.index
            ):
                return None

            ocf = cf.loc["Total Cash From Operating Activities"]
            capex = cf.loc["Capital Expenditures"]
            fcf = ocf + capex

            ocf = ocf.dropna().astype('float')[::-1]
            capex = capex.dropna().astype('float')[::-1]
            fcf = fcf.dropna().astype('float')[::-1]

            is_positive = bool((fcf > 0).all())
            fcf_list: list[float] = [float(x) for x in fcf]  # type: ignore
            is_growing = all(earlier <= later for earlier, later in zip(fcf_list, fcf_list[1:]))

            return {
                "Ticker": ticker,
                "FCF Positive": is_positive,
                "FCF Growing": is_growing,
                "FCF (oldest → newest)": list(fcf),
                "OCF (oldest → newest)": list(ocf),
                "CapEx (oldest → newest)": list(capex),
            }

        except Exception:
            _logger.exception("Error fetching FCF for %s", ticker)
            return None
