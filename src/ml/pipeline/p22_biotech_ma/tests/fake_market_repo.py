"""In-memory stand-in for the `P22Repo` reads `market_cap`/`price_quality`/`size_floor` use, with real as-of semantics."""

from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional


class FakeMarketRepo:
    """Prices, facts and splits for any number of companies, filtered by `as_of` like the real repo."""

    def __init__(self) -> None:
        self.prices: Dict[int, List[Dict[str, Any]]] = {}
        self.facts: Dict[tuple, List[Dict[str, Any]]] = {}
        self.splits: Dict[int, List[Dict[str, Any]]] = {}
        self.companies: Dict[str, Dict[str, Any]] = {}

    def add_price(self, company_id: int, trade_date: date, close: float) -> None:
        """A bar known on its own trade date (spec lag 0)."""
        self.prices.setdefault(company_id, []).append({"trade_date": trade_date, "close_raw": close, "vendor": "fmp"})

    def add_fact(self, company_id: int, metric: str, value: float, period_end: date, filed: date) -> None:
        """A fact for `period_end`, first known on `filed`."""
        self.facts.setdefault((company_id, metric), []).append(
            {"value": value, "period_end": period_end, "known_from": datetime(filed.year, filed.month, filed.day, tzinfo=timezone.utc)}
        )

    def add_split(self, company_id: int, ex_date: date, ratio: float) -> None:
        """A split known from its ex-date."""
        self.splits.setdefault(company_id, []).append({"ex_date": ex_date, "ratio": ratio})

    def get_latest_raw_close_as_of(self, company_id: int, as_of: date, vendor: Optional[str] = None) -> Optional[Dict[str, Any]]:
        del vendor
        visible = [p for p in self.prices.get(company_id, []) if p["trade_date"] <= as_of]
        return max(visible, key=lambda p: p["trade_date"]) if visible else None

    def get_financial_facts_as_of(self, company_id: int, metric: str, as_of_date: date) -> List[Dict[str, Any]]:
        visible = [f for f in self.facts.get((company_id, metric), []) if f["known_from"].date() <= as_of_date]
        return sorted(visible, key=lambda f: f["known_from"], reverse=True)

    def get_splits_as_of(self, company_id: int, as_of: date) -> List[Dict[str, Any]]:
        return sorted((s for s in self.splits.get(company_id, []) if s["ex_date"] <= as_of), key=lambda s: s["ex_date"])

    def get_company_by_cik(self, cik: str) -> Optional[Dict[str, Any]]:
        return self.companies.get(cik)
