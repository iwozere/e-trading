"""
P24 Falling Knife — universe selection and point-in-time eligibility (spec §3.4).

The FMP vintage's reference table decides what counts as a plain US common stock. Eligibility on each bar uses only
past bars and the **raw** (as-traded) price: split-adjusted history of a stock that later reverse-split looks far
pricier than it traded, which would let sub-$1 bars through the price floor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.data.fmp_pipeline.builders.validate import SECURITIES
from src.ml.pipeline.p24_falling_knife.config import UniverseParams

_PLAIN_SYMBOL_RE = re.compile(r"^[A-Z]{1,5}$")
# Nasdaq fifth-letter suffixes for warrants, rights, units and preferreds (FMP types some preferreds as "stock").
_DERIVATIVE_SUFFIX_RE = re.compile(r"^[A-Z]{4}[WRUP]$")
_COLUMNS = ["security_id", "symbol", "name", "exchange", "instrument_type", "first_price_date", "last_price_date",
            "delisted_date", "is_actively_trading", "valid_to", "identity_uncertain"]


@dataclass(frozen=True)
class Security:
    """One vintage security as P24 needs it."""

    security_id: int
    symbol: str
    name: Optional[str]
    exchange: Optional[str]
    instrument_type: Optional[str]
    first_price_date: Optional[date]
    last_price_date: Optional[date]
    delisted_date: Optional[date]
    is_actively_trading: Optional[bool]
    valid_to: Optional[date]
    identity_uncertain: bool

    @property
    def delisted(self) -> bool:
        """The price series ends because the security stopped trading (not because the vintage ends)."""
        return self.delisted_date is not None or self.is_actively_trading is False or self.valid_to is not None


def _as_date(value: Any) -> Optional[date]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date()
    return value if isinstance(value, date) else None


def load_securities(vintage_dir: Path) -> List[Security]:
    """
    Read the vintage's security reference table.

    Args:
        vintage_dir: ``<data_root>/vintages/<id>`` (``FmpStore.dir``).

    Returns:
        One ``Security`` per row.
    """
    out: List[Security] = []
    for row in pq.read_table(vintage_dir / SECURITIES, columns=_COLUMNS).to_pylist():
        out.append(Security(
            security_id=int(row["security_id"]),
            symbol=row["symbol"] or "",
            name=row["name"],
            exchange=row["exchange"],
            instrument_type=row["instrument_type"],
            first_price_date=_as_date(row["first_price_date"]),
            last_price_date=_as_date(row["last_price_date"]),
            delisted_date=_as_date(row["delisted_date"]),
            is_actively_trading=row["is_actively_trading"],
            valid_to=_as_date(row["valid_to"]),
            identity_uncertain=bool(row["identity_uncertain"]),
        ))
    return out


def is_common_stock(security: Security, include_adr: bool = False) -> bool:
    """
    True for a plain common-stock listing with price history (ADRs only if ``include_adr``).

    Excludes ETFs/funds, warrants, units, rights, preferreds (also those FMP types as stock, by symbol suffix), other
    share-class symbols, and securities whose identity the FMP builder could not settle.

    Args:
        security: Vintage security.
        include_adr: Also accept ``adr``.

    Returns:
        True if the security belongs in the P24 universe.
    """
    if security.identity_uncertain or security.first_price_date is None:
        return False
    allowed = {"stock", "adr"} if include_adr else {"stock"}
    if security.instrument_type not in allowed:
        return False
    return bool(_PLAIN_SYMBOL_RE.match(security.symbol)) and not _DERIVATIVE_SUFFIX_RE.match(security.symbol)


def survivorship_table(securities: List[Security], first_year: int, last_year: int, reference_years: Tuple[int, int],
                       threshold: float) -> pd.DataFrame:
    """
    Delisted-name coverage per year (spec §7.4).

    The vintage cannot say how many delistings it is missing, so coverage is estimated from the delisting rate: the
    share of listed common stocks that delisted in a year, divided by the median rate of ``reference_years`` (years
    with good delisted coverage), capped at 1.

    Args:
        securities: Universe securities (already filtered to common stock).
        first_year: First year to report.
        last_year: Last year to report.
        reference_years: Inclusive year range whose median delisting rate counts as full coverage.
        threshold: Coverage below this marks the year ``upper_bound_only``.

    Returns:
        Frame with ``year, listed, delisted, delisting_rate, est_coverage, upper_bound_only``.
    """
    rows = []
    for year in range(first_year, last_year + 1):
        start, end = date(year, 1, 1), date(year, 12, 31)
        listed = [s for s in securities if s.first_price_date is not None and s.first_price_date <= end
                  and (s.last_price_date is None or s.last_price_date >= start)]
        delisted = [s for s in listed if s.delisted and s.last_price_date is not None and s.last_price_date.year == year]
        rows.append({"year": year, "listed": len(listed), "delisted": len(delisted),
                     "delisting_rate": len(delisted) / len(listed) if listed else np.nan})
    table = pd.DataFrame(rows)
    ref = table[(table["year"] >= reference_years[0]) & (table["year"] <= reference_years[1])]["delisting_rate"]
    reference_rate = float(ref.median()) if not ref.empty else np.nan
    table["est_coverage"] = (table["delisting_rate"] / reference_rate).clip(upper=1.0)
    table["upper_bound_only"] = table["est_coverage"] < threshold
    return table


def median_dollar_volume(daily: pd.DataFrame, params: UniverseParams) -> pd.Series:
    """
    Rolling median daily dollar volume on the as-traded price, zero-volume days left out.

    Args:
        daily: Daily bars of one security.
        params: Universe parameters (window).

    Returns:
        Series on ``daily.index``.
    """
    close = daily["close_raw"] if "close_raw" in daily.columns else daily["close"]
    volume = daily["volume_raw"] if "volume_raw" in daily.columns else daily["volume"]
    traded = volume.where(volume > 0)
    return (close * traded).rolling(params.dollar_volume_bars, min_periods=params.dollar_volume_bars // 2).median()


def eligibility_mask(daily: pd.DataFrame, params: UniverseParams) -> pd.Series:
    """
    Point-in-time universe filter per daily bar (spec §3.4).

    Uses ``close_raw`` / ``volume_raw`` when present (as traded), else ``close`` / ``volume``.

    Args:
        daily: Daily bars of one security.
        params: Universe filter parameters.

    Returns:
        Boolean series on ``daily.index``.
    """
    close = daily["close_raw"] if "close_raw" in daily.columns else daily["close"]
    volume = daily["volume_raw"] if "volume_raw" in daily.columns else daily["volume"]
    dollar = median_dollar_volume(daily, params)
    nonzero = (volume > 0).astype(float).rolling(params.nonzero_volume_bars, min_periods=1).mean()
    mask = (close >= params.min_price) & (dollar >= params.min_dollar_volume) & (nonzero >= params.min_nonzero_volume_frac)
    return mask.fillna(False).astype(bool)
