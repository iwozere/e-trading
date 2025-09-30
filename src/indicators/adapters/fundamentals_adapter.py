# ---------------------------------------------------------------------------
# adapters/fundamentals_adapter.py
# ---------------------------------------------------------------------------
# Pulls normalized fundamentals via your existing DataManager/common.fundamentals,
# returns scalar values wrapped as single-row Series aligned to a synthetic index

import pandas as pd
from typing import Any

from src.indicators.adapters.base import BaseAdapter

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

    def __init__(self, fundamentals_getter=None):
        # Injected for testing; by default resolves from your codebase at runtime
        self._getter = fundamentals_getter

    def supports(self, name: str) -> bool:
        return name in self.FIELD_MAP

    def _get_fundamentals(self, ticker: str, provider: str | None) -> Any:
        if self._getter:
            return self._getter(ticker, provider)
        # Lazy import to avoid circulars
        from src.common.fundamentals import get_fundamentals_unified as _gf
        return asyncio.get_event_loop().run_until_complete(_gf(ticker, provider))

    def compute(self, name, df, inputs, params):
        # expects params: {"ticker": str, "provider": Optional[str]}
        ticker = params.get("ticker"); provider = params.get("provider")
        fun = self._get_fundamentals(ticker, provider)
        field = self.FIELD_MAP[name]
        value = getattr(fun, field, None)
        # Represent as a Series of length 1 (so the service can merge across outputs)
        s = pd.Series([value], index=[pd.Timestamp.utcnow()])
        return {"value": s}
