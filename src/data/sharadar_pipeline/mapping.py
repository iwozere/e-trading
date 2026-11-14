"""
Map Sharadar securities to FMP vintage securities by CIK and overlapping price history.

Used for cross-checking the two vendors and for joining FMP-only data (earnings events) to Sharadar securities.
Sharadar ids stay ``permaticker``; the FMP id is an extra column, never a replacement.
"""

from __future__ import annotations

import pandas as pd

_MAPPABLE = {"stock", "adr"}
_MAPPABLE_SHARADAR = _MAPPABLE | {"stock_secondary"}


def map_to_fmp(securities: pd.DataFrame, fmp_securities: pd.DataFrame) -> pd.Series:
    """
    Find the FMP security of each Sharadar security.

    A pair is a candidate when both have the same CIK, a mappable instrument type and overlapping price ranges.
    Pairs are assigned greedily, same symbol first and then by overlap length, each side at most once, so the
    result is one-to-one and share classes of one issuer (GOOGL / GOOG) land on their own listing.

    Args:
        securities: ``securities_frame`` output (``security_id``, ``symbol``, ``cik``, ``instrument_type``, price dates).
        fmp_securities: FMP vintage securities (same columns).

    Returns:
        ``Int64`` series indexed like ``securities``: the FMP ``security_id`` or ``<NA>``.
    """
    columns = ["security_id", "symbol", "cik", "first_price_date", "last_price_date"]
    left = securities[securities["instrument_type"].isin(_MAPPABLE_SHARADAR)][columns].copy()
    right = fmp_securities[fmp_securities["instrument_type"].isin(_MAPPABLE)][columns].copy()
    for frame in (left, right):
        frame["symbol"] = frame["symbol"].astype(str).str.replace(".", "-", regex=False)
        frame["cik"] = pd.to_numeric(frame["cik"], errors="coerce")
        frame["first_price_date"] = pd.to_datetime(frame["first_price_date"], errors="coerce")
        frame["last_price_date"] = pd.to_datetime(frame["last_price_date"], errors="coerce")
    left = left.dropna(subset=["cik", "first_price_date", "last_price_date"])
    right = right.dropna(subset=["cik", "first_price_date", "last_price_date"])
    pairs = left.merge(right, on="cik", suffixes=("", "_fmp"))
    start = pairs[["first_price_date", "first_price_date_fmp"]].max(axis=1)
    end = pairs[["last_price_date", "last_price_date_fmp"]].min(axis=1)
    pairs["overlap"] = (end - start).dt.days
    pairs["same_symbol"] = pairs["symbol"] == pairs["symbol_fmp"]
    pairs = pairs[pairs["overlap"] >= 0].sort_values(["same_symbol", "overlap", "security_id", "security_id_fmp"],
                                                     ascending=[False, False, True, True])
    used_left: set[int] = set()
    used_right: set[int] = set()
    chosen: dict[int, int] = {}
    for sid, fmp_sid in zip(pairs["security_id"].to_numpy(), pairs["security_id_fmp"].to_numpy()):
        if sid in used_left or fmp_sid in used_right:
            continue
        used_left.add(int(sid))
        used_right.add(int(fmp_sid))
        chosen[int(sid)] = int(fmp_sid)
    return securities["security_id"].map(chosen).astype("Int64")
