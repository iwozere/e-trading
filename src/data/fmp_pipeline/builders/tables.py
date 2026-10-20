"""
Which non-price endpoint goes into which vintage table (spec §7.1). Splits, dividends and prices have dedicated
builders (``events.py``, ``prices.py``); everything else is a generic record table (``records.py``).

Table names follow spec §7.1 where it names them; the rest are named after their endpoint. An archived endpoint
that is in none of the tables is listed in ``build_info.json`` as ``unbuilt`` (its raw responses stay in the
archive), so nothing is dropped silently.
"""

from __future__ import annotations

from src.data.fmp_pipeline.builders.records import Field, TableSpec
from src.data.fmp_pipeline.planner.universe import CONSTITUENT_ENDPOINTS

SYMBOL = Field("symbol", "symbol")
DATE = Field("date", "date", "date")
STATEMENT_FIELDS = (SYMBOL, DATE, Field("period", "period"), Field("fiscalYear", "fiscal_year"),
                    Field("reportedCurrency", "reported_currency"), Field("cik", "cik"),
                    Field("filingDate", "filing_date", "date"), Field("acceptedDate", "accepted_date"))
STATEMENT_KEY = ("symbol", "date", "period")

_STATEMENTS = {
    "income-statement": "income_statement",
    "balance-sheet-statement": "balance_sheet",
    "cash-flow-statement": "cash_flow",
    "key-metrics": "key_metrics",
    "ratios": "ratios",
    "enterprise-values": "enterprise_values",
    "income-statement-as-reported": "income_statement_as_reported",
    "balance-sheet-statement-as-reported": "balance_sheet_as_reported",
    "cash-flow-statement-as-reported": "cash_flow_as_reported",
    "financial-statement-full-as-reported": "full_as_reported",
    "income-statement-growth": "income_statement_growth",
    "balance-sheet-statement-growth": "balance_sheet_growth",
    "cash-flow-statement-growth": "cash_flow_growth",
    "analyst-estimates": "analyst_estimates",
}


def _list(name: str, endpoint: str, fields: tuple[Field, ...] = (SYMBOL,), numeric: bool = False) -> TableSpec:
    return TableSpec(name, (endpoint,), per_symbol=False, fields=fields, numeric=numeric)


def _per_symbol(name: str, endpoint: str, fields: tuple[Field, ...] = (SYMBOL, DATE),
                dedup: tuple[str, ...] = ("symbol", "date"), numeric: bool = False) -> TableSpec:
    return TableSpec(name, (endpoint,), per_symbol=True, fields=fields, dedup=dedup, numeric=numeric)


def _index_label(endpoint: str) -> str:
    return endpoint.replace("historical-", "").replace("-constituent", "")


TABLES: tuple[TableSpec, ...] = (
    # --- reference
    _per_symbol("reference/profiles", "profile",
                (SYMBOL, Field("companyName", "name"), Field("exchange", "exchange"), Field("cik", "cik"),
                 Field("isin", "isin"), Field("cusip", "cusip"), Field("sector", "sector"),
                 Field("industry", "industry"), Field("country", "country"), Field("currency", "currency"),
                 Field("ipoDate", "ipo_date", "date"), Field("isEtf", "is_etf", "bool"),
                 Field("isFund", "is_fund", "bool"), Field("isAdr", "is_adr", "bool"),
                 Field("isActivelyTrading", "is_actively_trading", "bool")), dedup=("symbol",)),
    _list("reference/delisted", "delisted-companies",
          (SYMBOL, Field("companyName", "name"), Field("exchange", "exchange"), Field("ipoDate", "ipo_date", "date"),
           Field("delistedDate", "delisted_date", "date"))),
    _list("reference/stock_list", "stock-list", (SYMBOL, Field("companyName", "name"))),
    _list("reference/actively_trading", "actively-trading-list", (SYMBOL, Field("name", "name"))),
    _list("reference/etf_list", "etf-list", (SYMBOL, Field("name", "name"))),
    _list("reference/symbol_changes", "symbol-change",
          (DATE, Field("oldSymbol", "old_symbol"), Field("newSymbol", "new_symbol"),
           Field("companyName", "name"))),
    _list("reference/exchanges", "available-exchanges", (Field("exchange", "exchange"), Field("name", "name"))),
    _list("reference/sectors", "available-sectors", (Field("sector", "sector"),)),
    _list("reference/industries", "available-industries", (Field("industry", "industry"),)),
    _list("reference/countries", "available-countries", (Field("country", "country"),)),
    _list("reference/statement_symbols", "financial-statement-symbol-list"),
    _list("reference/cik_list", "cik-list", (Field("cik", "cik"), Field("companyName", "name"))),
    _list("reference/shares_float", "shares-float-all", (SYMBOL, DATE), numeric=True),
    _list("reference/index_list", "index-list", (SYMBOL, Field("name", "name"))),
    _list("reference/market_risk_premium", "market-risk-premium", (Field("country", "country"),), numeric=True),
    TableSpec("reference/holidays", ("holidays-by-exchange",), per_symbol=False,
              fields=(Field("exchange", "exchange"), DATE, Field("name", "name"))),
    TableSpec("reference/market_hours", ("exchange-market-hours",), per_symbol=False,
              fields=(Field("exchange", "exchange"), Field("name", "name"))),
    _per_symbol("reference/employee_count", "employee-count",
                (SYMBOL, Field("periodOfReport", "date", "date"), Field("filingDate", "filing_date", "date")),
                numeric=True),
    _per_symbol("reference/historical_employee_count", "historical-employee-count",
                (SYMBOL, Field("periodOfReport", "date", "date"), Field("filingDate", "filing_date", "date")),
                numeric=True),
    _per_symbol("reference/etf_info", "etf/info", (SYMBOL,), dedup=("symbol",)),
    _per_symbol("reference/etf_holdings", "etf/holdings", (SYMBOL, Field("asset", "asset")), dedup=()),
    # --- events (splits and dividends: events.py)
    _per_symbol("events/earnings", "earnings", numeric=True),
    _list("events/mna", "mergers-acquisitions-latest",
          (SYMBOL, Field("transactionDate", "date", "date"), Field("targetedSymbol", "targeted_symbol"))),
    # --- index membership (spec §5.4)
    TableSpec("index_membership/index_changes", tuple(e for e in CONSTITUENT_ENDPOINTS if e.startswith("historical")),
              per_symbol=False, fields=(DATE, SYMBOL), label="index",
              labels={e: _index_label(e) for e in CONSTITUENT_ENDPOINTS}),
    TableSpec("index_membership/index_members_current",
              tuple(e for e in CONSTITUENT_ENDPOINTS if not e.startswith("historical")), per_symbol=False,
              fields=(SYMBOL, Field("dateFirstAdded", "date_first_added", "date")), label="index",
              labels={e: _index_label(e) for e in CONSTITUENT_ENDPOINTS}),
    # --- prices: reconciliation sample (spec D8); daily/intraday raw: prices.py
    _per_symbol("prices/recon/daily_split_adjusted_sample", "historical-price-eod/full", numeric=True),
    _per_symbol("prices/recon/daily_div_adjusted_sample", "historical-price-eod/dividend-adjusted", numeric=True),
    # --- fundamentals
    *(_per_symbol(f"fundamentals/{name}", endpoint, STATEMENT_FIELDS, STATEMENT_KEY, numeric=True)
      for endpoint, name in _STATEMENTS.items()),
    _per_symbol("fundamentals/market_cap", "historical-market-capitalization", numeric=True),
    _per_symbol("fundamentals/financial_reports_dates", "financial-reports-dates",
                (SYMBOL, Field("fiscalYear", "fiscal_year"), Field("period", "period")),
                dedup=("symbol", "fiscal_year", "period")),
    _per_symbol("fundamentals/financial_scores", "financial-scores", (SYMBOL,), dedup=("symbol",), numeric=True),
    _per_symbol("fundamentals/owner_earnings", "owner-earnings", numeric=True),
    _per_symbol("fundamentals/revenue_product_segmentation", "revenue-product-segmentation",
                (SYMBOL, DATE, Field("period", "period")), dedup=STATEMENT_KEY),
    _per_symbol("fundamentals/revenue_geographic_segmentation", "revenue-geographic-segmentation",
                (SYMBOL, DATE, Field("period", "period")), dedup=STATEMENT_KEY),
    _per_symbol("fundamentals/grades_historical", "grades-historical", numeric=True),
    _per_symbol("fundamentals/ratings_historical", "ratings-historical", numeric=True),
    _per_symbol("fundamentals/grades", "grades",
                (SYMBOL, DATE, Field("gradingCompany", "grading_company"), Field("previousGrade", "previous_grade"),
                 Field("newGrade", "new_grade"), Field("action", "action"))),
    _per_symbol("events/earnings_call_transcript_dates", "earning-call-transcript-dates",
                (SYMBOL, DATE, Field("fiscalYear", "fiscal_year", "int"), Field("quarter", "quarter", "int")),
                dedup=("symbol", "fiscal_year", "quarter")),
    _per_symbol("events/earnings_call_transcripts", "earning-call-transcript",
                (SYMBOL, DATE, Field("year", "fiscal_year", "int"), Field("period", "period"),
                 Field("content", "content")), dedup=("symbol", "fiscal_year", "period")),
    # --- macro and calendars
    _list("macro/treasury_rates", "treasury-rates", (DATE,), numeric=True),
    TableSpec("macro/economic_indicators", ("economic-indicators",), per_symbol=False,
              fields=(Field("@name", "name"), DATE), dedup=("name", "date"), numeric=True),
    TableSpec("macro/calendars", ("splits-calendar", "dividends-calendar", "earnings-calendar", "ipos-calendar",
                                  "economic-calendar"), per_symbol=False, fields=(SYMBOL, DATE), label="calendar",
              labels={"splits-calendar": "splits", "dividends-calendar": "dividends",
                      "earnings-calendar": "earnings", "ipos-calendar": "ipos", "economic-calendar": "economic"}),
)

#: Endpoints with a dedicated builder.
DEDICATED = ("splits", "dividends", "historical-price-eod/non-split-adjusted")
INTRADAY_PREFIX = "historical-chart/"


def covered_endpoints() -> set[str]:
    """Return every endpoint some table or dedicated builder reads."""
    return {e for spec in TABLES for e in spec.endpoints} | set(DEDICATED)
