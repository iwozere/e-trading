# fundamentals_adapter.py
import pandas as pd

from src.common.fundamentals import get_fundamentals_unified
from src.indicators.adapters.base import BaseAdapter
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class FundamentalsAdapter(BaseAdapter):
    # Keys MUST match the canonical fundamental indicator names in
    # `src.indicators.models.FUNDAMENTAL_INDICATORS` (the registry `INDICATOR_META`
    # is built from that dict) — `IndicatorService._select_provider()` looks an
    # adapter up by calling `supports(canonical_name)` directly, with no
    # short-name translation in between. Values are the corresponding
    # `src.model.telegram_bot.Fundamentals` field names.
    FIELD_MAP = {
        "pe_ratio": "pe_ratio",
        "forward_pe": "forward_pe",
        "pb_ratio": "price_to_book",
        "ps_ratio": "price_to_sales",
        "peg_ratio": "peg_ratio",
        "roe": "return_on_equity",
        "roa": "return_on_assets",
        "debt_to_equity": "debt_to_equity",
        "current_ratio": "current_ratio",
        "quick_ratio": "quick_ratio",
        "operating_margin": "operating_margin",
        "profit_margin": "profit_margin",
        "revenue_growth": "revenue_growth",
        "net_income_growth": "net_income_growth",
        "free_cash_flow": "free_cash_flow",
        "dividend_yield": "dividend_yield",
        "payout_ratio": "payout_ratio",
        "beta": "beta",
        "market_cap": "market_cap",
        "enterprise_value": "enterprise_value",
        "ev_to_ebitda": "enterprise_value_to_ebitda",
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
            # Use pre-fetched data if injected via the constructor (avoids a
            # redundant fetch when the caller already has fresh fundamentals,
            # and lets tests inject a mock instead of hitting the network).
            if self._data is not None:
                fund_data = self._data
            else:
                ticker = params.get("ticker")
                provider = params.get("provider")

                if not ticker:
                    raise ValueError("FundamentalsAdapter requires 'ticker' in params")

                fund_data = await get_fundamentals_unified(ticker, provider)

            field = self.FIELD_MAP[name]
            value = getattr(fund_data, field, None)

            # Return as broadcasted series if df provided
            if df is not None and len(df) > 0:
                return {"value": pd.Series(value, index=df.index, name=name)}
            else:
                return {"value": pd.Series([value], name=name)}

        except Exception as e:
            _logger.warning("Error fetching fundamental %s for %s: %s", name, params.get("ticker"), e)
            # Return NaN series
            if df is not None and len(df) > 0:
                return {"value": pd.Series(index=df.index, dtype=float, name=name)}
            else:
                return {"value": pd.Series([None], name=name)}
