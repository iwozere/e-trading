"""
Map EODHD securities to FMP vintage securities by symbol, overlapping price history and company name.

EODHD gives no CIK, so a pair needs the same traded symbol and overlapping price ranges; the company name decides
between candidates and guards against a reused ticker. Our ids stay; the FMP id is an extra column.
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd

_MAPPABLE_FMP = {"stock", "adr"}
#: Names agree when their token overlap is at least this.
MIN_NAME_SIMILARITY = 0.5
#: Without a name match, a pair is accepted only when it is the only candidate and overlaps at least this long.
MIN_OVERLAP_DAYS_WITHOUT_NAME = 365

_NAME_NOISE = {"inc", "corp", "corporation", "co", "company", "ltd", "limited", "plc", "holdings", "holding", "group", "the", "sa",
               "nv", "ag", "lp", "llc", "trust", "class", "a", "b", "common", "stock", "new", "de", "of", "and"}
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def name_similarity(a: Optional[str], b: Optional[str]) -> float:
    """Return the token overlap (Jaccard) of two company names after dropping legal-form words; 0 if either is empty."""
    left = {t for t in _TOKEN_RE.findall((a or "").lower()) if t not in _NAME_NOISE}
    right = {t for t in _TOKEN_RE.findall((b or "").lower()) if t not in _NAME_NOISE}
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def map_to_fmp(securities: pd.DataFrame, fmp_securities: pd.DataFrame) -> pd.Series:
    """
    Find the FMP security of each EODHD security.

    Candidates share the symbol (``.`` and ``-`` class separators treated alike) and overlap in price history. A
    candidate is accepted when the names agree, or when it is the security's only candidate and overlaps for at
    least ``MIN_OVERLAP_DAYS_WITHOUT_NAME`` days. Pairs are assigned greedily (name match, then overlap), each side
    at most once.

    Args:
        securities: ``security_id``, ``symbol``, ``name``, ``first_price_date``, ``last_price_date``.
        fmp_securities: FMP vintage securities (same columns plus ``instrument_type``).

    Returns:
        ``Int64`` series indexed like ``securities``: the FMP ``security_id`` or ``<NA>``.
    """
    columns = ["security_id", "symbol", "name", "first_price_date", "last_price_date"]
    left = securities[columns].copy()
    right = fmp_securities[fmp_securities["instrument_type"].isin(_MAPPABLE_FMP)][columns].copy()
    for frame in (left, right):
        frame["symbol"] = frame["symbol"].astype(str).str.upper().str.replace(".", "-", regex=False)
        frame["first_price_date"] = pd.to_datetime(frame["first_price_date"], errors="coerce")
        frame["last_price_date"] = pd.to_datetime(frame["last_price_date"], errors="coerce")
    left = left.dropna(subset=["first_price_date", "last_price_date"])
    right = right.dropna(subset=["first_price_date", "last_price_date"])
    pairs = left.merge(right, on="symbol", suffixes=("", "_fmp"))
    if pairs.empty:
        return pd.Series(pd.array([pd.NA] * len(securities), dtype="Int64"), index=securities.index)
    start = pairs[["first_price_date", "first_price_date_fmp"]].max(axis=1)
    end = pairs[["last_price_date", "last_price_date_fmp"]].min(axis=1)
    pairs["overlap"] = (end - start).dt.days
    pairs = pairs[pairs["overlap"] >= 0].copy()
    pairs["similarity"] = [name_similarity(a, b) for a, b in zip(pairs["name"], pairs["name_fmp"])]
    pairs["name_ok"] = pairs["similarity"] >= MIN_NAME_SIMILARITY
    candidates = pairs.groupby("security_id")["security_id_fmp"].transform("count")
    pairs = pairs[pairs["name_ok"] | ((candidates == 1) & (pairs["overlap"] >= MIN_OVERLAP_DAYS_WITHOUT_NAME))]
    pairs = pairs.sort_values(["name_ok", "overlap", "security_id", "security_id_fmp"], ascending=[False, False, True, True])
    used_left: set[int] = set()
    used_right: set[int] = set()
    chosen: dict[int, int] = {}
    for sid, fmp_sid in zip(pairs["security_id"].to_numpy(), pairs["security_id_fmp"].to_numpy()):
        if int(sid) in used_left or int(fmp_sid) in used_right:
            continue
        used_left.add(int(sid))
        used_right.add(int(fmp_sid))
        chosen[int(sid)] = int(fmp_sid)
    return securities["security_id"].map(chosen).astype("Int64")
