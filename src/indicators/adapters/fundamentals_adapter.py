# fundamentals_adapter.py
import pandas as pd

from src.common.fundamentals import get_fundamentals_unified
from src.indicators.adapters.base import BaseAdapter

from src.notification.logger import setup_logger
_logger = setup_logger(__name__)

class FundamentalsAdapter(BaseAdapter):
    FIELD_MAP = {
        "pe": "pe_ratio",
        "forward_pe": "forward_pe",
        "pb": "price_to_book",
        "ps": "price_to_sales",
        "peg": "peg_ratio",
        "roe": "return_on_equity",
        "roa": "return_on_assets",
        "de_ratio": "debt_to_equity",
        "current_ratio": "current_ratio",
        "quick_ratio": "quick_ratio",
        "div_yield": "dividend_yield",
        "payout_ratio": "payout_ratio",
        "market_cap": "market_cap",
        "enterprise_value": "enterprise_value",
    }

    def __init__(self, fundamentals_data=None):
        """
        fundamentals_data: Pre-fetched fundamental data object.
        Pass this in after fetching asynchronously at service level.
        """
        self._data = fundamentals_data

    def supports(self, name: str) -> bool:
        return name in self.FIELD_MAP

    async def compute(self, name, df, inputs, params):
        """Async compute for fundamentals"""
        try:
            ticker = params.get("ticker")
            provider = params.get("provider")

            if not ticker:
                raise ValueError("FundamentalsAdapter requires 'ticker' in params")

            # Fetch fundamentals asynchronously
            fund_data = await get_fundamentals_unified(ticker, provider)

            field = self.FIELD_MAP[name]
            value = getattr(fund_data, field, None)

            # Return as broadcasted series if df provided
            if df is not None and len(df) > 0:
                return {"value": pd.Series(value, index=df.index, name=name)}
            else:
                return {"value": pd.Series([value], name=name)}

        except Exception as e:
            _logger.warning("Error fetching fundamental %s for %s: %s", name, params.get('ticker'), e)
            # Return NaN series
            if df is not None and len(df) > 0:
                return {"value": pd.Series(index=df.index, dtype=float, name=name)}
            else:
                return {"value": pd.Series([None], name=name)}