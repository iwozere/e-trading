"""
Pure conversions from Sharadar tables to the raw-basis layout the project's readers expect.

No I/O. Sharadar facts used here (docs/Findings.md §5):

- ``stocks``: ``open``/``high``/``low``/``close``/``volume`` are split-adjusted, ``closeunadj`` is as traded.
- ``actions``: ``split`` value = new shares per old share, dated on the ex-date; ``dividend`` value = cash per share
  **split-adjusted**, dated on the ex-date.
- ``tickers``: ``permaticker`` is the stable id; a delisted company whose ticker was reused has a number appended.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

#: Actions that end a price series, mapped to a short reason.
DELISTING_ACTIONS = {
    "acquisitionby": "acquired",
    "mergerto": "merged",
    "bankruptcyliquidation": "bankruptcy",
    "regulatorydelisting": "regulatory",
    "voluntarydelisting": "voluntary",
    "delisted": "delisted",
}
#: Most specific reason first: one security can carry ``delisted`` plus a specific row.
_REASON_RANK = {"acquired": 0, "merged": 1, "bankruptcy": 2, "regulatory": 3, "voluntary": 4, "delisted": 5}
CASH_ACTIONS = ("acquisitioncash", "acquisitioncashusd")
STOCK_ACTIONS = ("acquisitionstock",)
AR_DIMENSIONS = ("ARQ", "ART", "ARY")

_CIK_RE = re.compile(r"CIK=0*(\d+)")
_REUSE_SUFFIX_RE = re.compile(r"\d+$")


def extract_cik(secfilings: Optional[str]) -> Optional[int]:
    """Return the CIK inside a ``tickers.secfilings`` URL, or None."""
    if not isinstance(secfilings, str):
        return None
    match = _CIK_RE.search(secfilings)
    return int(match.group(1)) if match else None


def plain_symbol(ticker: str) -> str:
    """Return the ticker as traded: the number Sharadar appends to a reused delisted ticker is removed."""
    return _REUSE_SUFFIX_RE.sub("", ticker)


def instrument_type(category: Optional[str]) -> str:
    """
    Map a Sharadar issuer category to the vintage's ``instrument_type`` vocabulary.

    Domestic and Canadian common stock (single or primary class) is ``stock``; ADRs are ``adr``; secondary classes
    and preferreds get their own types so that ``is_common_stock`` leaves them out.
    """
    text = (category or "").lower()
    if "preferred" in text:
        return "preferred"
    if "secondary" in text:
        return "stock_secondary"
    if "common stock" in text:
        return "adr" if text.startswith("adr") else "stock"
    return "other"


def securities_frame(tickers: pd.DataFrame, table: str = "stocks") -> pd.DataFrame:
    """
    Build the security reference table from ``tickers`` rows of one price table.

    Args:
        tickers: Sharadar ``tickers`` table (all columns as strings).
        table: Which ``table`` value to keep (``stocks``; the SEP alias is accepted too).

    Returns:
        One row per ``permaticker`` with the columns of the FMP vintage's securities table that readers use
        (``security_id`` = ``permaticker``) plus Sharadar extras (``sharadar_ticker``, ``cik``, ``category``, …).
    """
    rows = tickers[tickers["table"].str.lower().isin({table, "sep"} if table == "stocks" else {table})].copy()
    rows["security_id"] = pd.to_numeric(rows["permaticker"], errors="coerce")
    rows = rows.dropna(subset=["security_id"])
    rows["security_id"] = rows["security_id"].astype("int64")
    for column in ("firstpricedate", "lastpricedate"):
        rows[column] = pd.to_datetime(rows[column].replace("", None), errors="coerce")
    rows = rows.sort_values(["security_id", "lastpricedate"]).drop_duplicates("security_id", keep="last")
    delisted = rows["isdelisted"].str.upper() == "Y"
    blank = rows.reindex(columns=["name", "exchange", "category", "cusips", "figi", "sector", "industry", "siccode",
                                  "famaindustry", "currency", "relatedtickers"]).replace("", None)
    out = pd.DataFrame({
        "security_id": rows["security_id"],
        "symbol": rows["ticker"].map(plain_symbol),
        "sharadar_ticker": rows["ticker"],
        "name": blank["name"],
        "exchange": blank["exchange"],
        "instrument_type": rows["category"].map(instrument_type),
        "category": blank["category"],
        "cik": rows["secfilings"].map(extract_cik).astype("Int64"),
        "cusips": blank["cusips"],
        "figi": blank["figi"],
        "sector": blank["sector"],
        "industry": blank["industry"],
        "siccode": blank["siccode"],
        "first_price_date": rows["firstpricedate"].dt.date,
        "last_price_date": rows["lastpricedate"].dt.date,
        "delisted_date": rows["lastpricedate"].dt.date.where(delisted, None),
        "is_actively_trading": ~delisted,
        "valid_from": None,
        "valid_to": None,
        "identity_uncertain": False,
    })
    return out.sort_values("security_id").reset_index(drop=True)


def raw_prices(stocks: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Sharadar daily bars to the raw (as-traded) basis.

    ``factor = closeunadj / close`` is the split factor of the day; prices are multiplied by it and volume divided.

    Args:
        stocks: Columns ``date``, ``open``, ``high``, ``low``, ``close``, ``volume``, ``closeadj``, ``closeunadj``
            (numeric or strings); other columns are passed through.

    Returns:
        Same rows minus those without a positive close, with raw ``open``/``high``/``low``/``close``/``volume`` and
        ``closeadj_vendor`` (Sharadar's fully adjusted close, kept only to check our own adjustment).
    """
    out = stocks.copy()
    for column in ("open", "high", "low", "close", "volume", "closeadj", "closeunadj"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out[(out["close"] > 0) & (out["closeunadj"] > 0)].copy()
    factor = (out["closeunadj"] / out["close"]).to_numpy(dtype=float)
    for column in ("open", "high", "low"):
        out[column] = out[column].to_numpy(dtype=float) * factor
    out["volume"] = out["volume"].to_numpy(dtype=float) / factor
    out["close"] = out["closeunadj"]
    out = out.rename(columns={"closeadj": "closeadj_vendor"}).drop(columns=["closeunadj"])
    return out


def _actions(actions: pd.DataFrame, names: tuple[str, ...]) -> pd.DataFrame:
    rows = actions[actions["action"].isin(names)].copy()
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce")
    rows["value"] = pd.to_numeric(rows["value"], errors="coerce")
    return rows.dropna(subset=["date"])


def splits_frame(actions: pd.DataFrame) -> pd.DataFrame:
    """
    Return split events in ``adjust()`` form.

    Returns:
        ``ticker``, ``date`` (ex-date), ``numerator`` (new shares per old), ``denominator`` (1.0).
    """
    rows = _actions(actions, ("split",))
    rows = rows[rows["value"] > 0]
    out = pd.DataFrame({"ticker": rows["ticker"], "date": rows["date"].dt.date, "numerator": rows["value"].astype(float),
                        "denominator": 1.0})
    return out.drop_duplicates(["ticker", "date"]).sort_values(["ticker", "date"]).reset_index(drop=True)


def dividends_frame(actions: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """
    Return cash dividends and spinoff dividends on the raw basis of their ex-date.

    Sharadar's amounts are split-adjusted; the raw amount is the adjusted one times the ratio of every split whose
    ex-date is **after** the dividend's ex-date.

    Args:
        actions: Sharadar ``actions`` table.
        splits: ``splits_frame`` output.

    Returns:
        ``ticker``, ``date`` (ex-date), ``amount`` (raw), ``amount_adj`` (as delivered), ``kind`` (``cash`` or
        ``spinoff``).
    """
    rows = _actions(actions, ("dividend", "spinoffdividend"))
    rows = rows[rows["value"] > 0]
    if rows.empty:
        return pd.DataFrame(columns=["ticker", "date", "amount", "amount_adj", "kind"])
    by_ticker = {ticker: (pd.to_datetime(group["date"]).to_numpy(), (group["numerator"] / group["denominator"]).to_numpy())
                 for ticker, group in splits.groupby("ticker")}
    later = np.ones(len(rows))
    dates = rows["date"].to_numpy()
    for position, ticker in enumerate(rows["ticker"].to_numpy()):
        if ticker in by_ticker:
            split_dates, ratios = by_ticker[ticker]
            later[position] = float(np.prod(ratios[split_dates > dates[position]]))
    out = pd.DataFrame({"ticker": rows["ticker"].to_numpy(), "date": rows["date"].dt.date.to_numpy(),
                        "amount": rows["value"].to_numpy(dtype=float) * later, "amount_adj": rows["value"].to_numpy(dtype=float),
                        "kind": np.where(rows["action"].to_numpy() == "dividend", "cash", "spinoff")})
    out = out.groupby(["ticker", "date", "kind"], as_index=False)[["amount", "amount_adj"]].sum()
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def delistings_frame(actions: pd.DataFrame) -> pd.DataFrame:
    """
    Return one row per delisted ticker: last trade date, reason and what a holder received.

    Returns:
        ``ticker``, ``date``, ``reason``, ``final_market_cap_musd``, ``acquirer`` (ticker or None),
        ``cash_per_share`` (USD, NaN if none or unknown), ``stock_ratio`` (acquirer shares per share) and
        ``stock_ticker``. Other consideration types (foreign cash, preferreds, elections) are not interpreted;
        ``other_consideration`` is True when any is present.
    """
    rows = _actions(actions, tuple(DELISTING_ACTIONS))
    columns = ["ticker", "date", "reason", "final_market_cap_musd", "acquirer", "cash_per_share", "stock_ratio",
               "stock_ticker", "other_consideration"]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    rows["reason"] = rows["action"].map(DELISTING_ACTIONS)
    rows["rank"] = rows["reason"].map(_REASON_RANK)
    rows["acquirer"] = rows["contraticker"].where(~rows["contraticker"].isin(["N/A", "", None]), None)
    main = rows.sort_values(["ticker", "date", "rank"]).drop_duplicates("ticker", keep="last")
    best = rows.sort_values(["ticker", "rank"]).drop_duplicates("ticker", keep="first")
    out = main[["ticker", "date"]].merge(best[["ticker", "reason", "acquirer"]], on="ticker")
    cap = rows.dropna(subset=["value"]).groupby("ticker")["value"].max().rename("final_market_cap_musd")
    out = out.merge(cap, on="ticker", how="left")
    cash = _actions(actions, CASH_ACTIONS).groupby("ticker")["value"].sum().rename("cash_per_share")
    stock = _actions(actions, STOCK_ACTIONS).sort_values("date").drop_duplicates("ticker", keep="last")
    stock = stock.set_index("ticker")[["value", "contraticker"]].rename(columns={"value": "stock_ratio", "contraticker": "stock_ticker"})
    known = set(CASH_ACTIONS) | set(STOCK_ACTIONS) | {"acquisitionby", "acquisitionof"}
    other = actions[actions["action"].str.startswith("acquisition") & ~actions["action"].isin(known)]["ticker"].unique()
    out = out.merge(cash, on="ticker", how="left").merge(stock, on="ticker", how="left")
    out["other_consideration"] = out["ticker"].isin(other)
    out["date"] = out["date"].dt.date
    return out[columns].sort_values("ticker").reset_index(drop=True)


def symbol_changes_frame(actions: pd.DataFrame) -> pd.DataFrame:
    """
    Return former symbols: ``ticker`` (current Sharadar ticker), ``old_symbol``, ``date`` (day of the change).
    """
    rows = _actions(actions, ("tickerchangefrom",))
    out = pd.DataFrame({"ticker": rows["ticker"], "old_symbol": rows["contraticker"], "date": rows["date"].dt.date})
    return out[~out["old_symbol"].isin(["N/A", ""])].drop_duplicates().sort_values(["ticker", "date"]).reset_index(drop=True)


def fundamentals_ar(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the as-reported dimensions and add the first day a row may be used.

    The most-recent-reported dimensions include later restatements and are dropped here, so they cannot leak into a
    backtest. ``available_from`` is the day after the filing date (``date``).

    Args:
        fundamentals: Sharadar ``fundamentals`` table.

    Returns:
        AR rows with ``date``, ``reportperiod`` and ``calendardate`` as dates, numeric indicators, ``available_from``.
    """
    rows = fundamentals[fundamentals["dimension"].isin(AR_DIMENSIONS)]
    keys = {"ticker", "dimension", "fiscalperiod"}
    converted = {}
    for column in rows.columns:
        if column in ("date", "reportperiod", "calendardate", "lastupdated"):
            converted[column] = pd.to_datetime(rows[column].replace("", None), errors="coerce")
        elif column in keys:
            converted[column] = rows[column]
        else:
            converted[column] = pd.to_numeric(rows[column], errors="coerce")
    out = pd.DataFrame(converted, index=rows.index)
    out = out.dropna(subset=["date"]).copy()
    out["available_from"] = out["date"] + pd.Timedelta(days=1)
    return out.sort_values(["ticker", "dimension", "date", "reportperiod"]).reset_index(drop=True)


def delisting_return(last_close: float, cash_per_share: Optional[float], stock_ratio: Optional[float] = None,
                     stock_price: Optional[float] = None) -> Optional[float]:
    """
    Return what a holder earned from the last close to the deal consideration.

    Args:
        last_close: Last raw close of the acquired stock.
        cash_per_share: Cash received per share, or None.
        stock_ratio: Acquirer shares received per share, or None.
        stock_price: Raw close of the acquirer on the same day; required when ``stock_ratio`` is set.

    Returns:
        ``consideration / last_close - 1``, or None if the consideration is unknown.
    """
    def _known(value: Optional[float]) -> bool:
        return value is not None and not np.isnan(value)

    if not last_close or last_close <= 0 or not (_known(cash_per_share) or _known(stock_ratio)):
        return None
    total = float(cash_per_share) if _known(cash_per_share) else 0.0  # type: ignore[arg-type]
    if _known(stock_ratio):
        if not _known(stock_price):
            return None
        total += float(stock_ratio) * float(stock_price)  # type: ignore[arg-type]
    return total / last_close - 1.0
