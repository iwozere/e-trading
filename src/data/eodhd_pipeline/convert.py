"""
Pure conversions from EODHD payloads to the raw-basis layout the project's readers expect. No I/O.

EODHD facts used here (docs/Design.md):

- ``/eod``: ``open``/``high``/``low``/``close`` are as traded; ``volume`` is split-adjusted; ``adjusted_close`` is
  adjusted for splits and dividends.
- ``/splits``: ``split`` is ``"new/old"`` (``"2.000000/1.000000"`` for 2-for-1), dated on the ex-date; ratios need
  not be integers (``"104.000000/100.000000"``).
- ``/div``: ``date`` is the ex-date; ``value`` is split-adjusted to today's shares, ``unadjustedValue`` is the
  amount paid in the shares of the day.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Optional

import numpy as np
import pandas as pd

_SPLIT_COLUMNS = ["date", "numerator", "denominator"]
_DIVIDEND_COLUMNS = ["date", "amount", "kind"]
BAR_COLUMNS = ["date", "open", "high", "low", "close", "volume", "closeadj_vendor"]


def _to_date(value: Any) -> Optional[date]:
    """Return an ISO date string (or date) as ``datetime.date``, or None."""
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _to_float(value: Any) -> float:
    """Return a number, or NaN when the value is missing or not numeric."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def parse_split(text: Any) -> Optional[tuple[float, float]]:
    """Return ``(new, old)`` shares of a ``"new/old"`` split string, or None if it is not a valid positive ratio."""
    if not isinstance(text, str) or "/" not in text:
        return None
    left, _, right = text.partition("/")
    try:
        numerator, denominator = float(left), float(right)
    except ValueError:
        return None
    if not (np.isfinite(numerator) and np.isfinite(denominator)) or numerator <= 0 or denominator <= 0 or numerator == denominator:
        return None
    return numerator, denominator


def splits_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Convert EODHD split rows to ``adjust()`` form.

    Returns:
        ``date`` (``datetime.date``), ``numerator``, ``denominator``; invalid rows dropped, one row per date.
    """
    records = []
    for row in rows:
        ratio = parse_split(row.get("split"))
        when = _to_date(row.get("date"))
        if ratio is not None and when is not None:
            records.append({"date": when, "numerator": ratio[0], "denominator": ratio[1]})
    frame = pd.DataFrame(records, columns=_SPLIT_COLUMNS)
    if frame.empty:
        return frame.astype({"numerator": float, "denominator": float})
    return frame.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def later_split_factor(dates: pd.Series, splits: pd.DataFrame) -> np.ndarray:
    """
    Return, for each date, the product of the ratios of all splits strictly after it.

    Multiplying a raw share count by this factor gives today's share terms; dividing a split-adjusted volume or
    multiplying a split-adjusted dividend by it gives the as-traded value.
    """
    factor = np.ones(len(dates))
    if splits.empty:
        return factor
    values = pd.to_datetime(dates).to_numpy()
    for when, numerator, denominator in zip(pd.to_datetime(splits["date"]).to_numpy(), splits["numerator"], splits["denominator"]):
        factor[values < when] *= float(numerator) / float(denominator)
    return factor


def dividends_frame(rows: list[dict[str, Any]], splits: pd.DataFrame, currency: str = "USD") -> pd.DataFrame:
    """
    Convert EODHD dividend rows to raw-basis cash dividends.

    ``unadjustedValue`` is used when present and positive; otherwise ``value`` is brought back to the shares of the
    ex-date with the later splits. Rows in another currency are dropped (their amount is not in the price currency).

    Returns:
        ``date``, ``amount`` (cash per share as traded on the ex-date), ``kind`` (``cash``); one row per date.
    """
    records = []
    for row in rows:
        when = _to_date(row.get("date"))
        row_currency = row.get("currency")
        if when is None or (row_currency and str(row_currency).upper() != currency):
            continue
        records.append({"date": when, "raw": _to_float(row.get("unadjustedValue")), "adjusted": _to_float(row.get("value"))})
    frame = pd.DataFrame(records, columns=["date", "raw", "adjusted"])
    if frame.empty:
        return pd.DataFrame(columns=_DIVIDEND_COLUMNS).astype({"amount": float})
    factor = later_split_factor(frame["date"], splits)
    raw = frame["raw"].astype(float)
    frame["amount"] = raw.where(raw > 0, frame["adjusted"].astype(float) * factor)
    frame = frame[frame["amount"] > 0]
    out = frame.groupby("date", as_index=False).agg(amount=("amount", "sum")).sort_values(by="date").reset_index(drop=True)
    out["kind"] = "cash"
    return out.loc[:, _DIVIDEND_COLUMNS]


def bars_frame(rows: list[dict[str, Any]], splits: pd.DataFrame, volume_split_adjusted: bool = True) -> pd.DataFrame:
    """
    Convert EODHD daily bars to the raw (as-traded) basis.

    Args:
        rows: ``/eod`` rows.
        splits: ``splits_frame`` output of the same symbol.
        volume_split_adjusted: EODHD's volume is split-adjusted; convert it back to shares as traded.

    Returns:
        ``BAR_COLUMNS``: raw OHLC and volume, ``closeadj_vendor`` (EODHD's adjusted close, kept only to check our
        own adjustment); rows without a positive close dropped, one row per date, sorted.
    """
    frame = pd.DataFrame(rows)
    if frame.empty or "date" not in frame.columns:
        return pd.DataFrame(columns=BAR_COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"])
    out = pd.DataFrame({"date": frame["date"].dt.date})
    for column in ("open", "high", "low", "close", "volume"):
        out[column] = pd.to_numeric(frame[column], errors="coerce") if column in frame.columns else np.nan
    out["closeadj_vendor"] = pd.to_numeric(frame["adjusted_close"], errors="coerce") if "adjusted_close" in frame.columns else np.nan
    out = out[out["close"] > 0]
    out = out.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    if volume_split_adjusted and len(out):
        out["volume"] = out["volume"].to_numpy(dtype=float) / later_split_factor(out["date"], splits)
    return out[BAR_COLUMNS].astype({column: float for column in BAR_COLUMNS[1:]})
