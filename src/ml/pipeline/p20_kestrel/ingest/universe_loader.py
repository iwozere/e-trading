"""
P20 Kestrel — Universe loader.

Reads the Nasdaq ticker CSV, enriches with fundamentals, and upserts
rows into k20_universe. Marks tickers absent from the CSV as 'delisted'.
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd

from src.common.fundamentals import get_fundamentals_unified
from src.ml.pipeline.p20_kestrel.config import NASDAQ_TICKERS_CSV, UNIVERSE_MIN_MCAP_USD
from src.data.db.services.kestrel_service import KestrelService as _KestrelService

_kestrel = _KestrelService()
get_active_tickers = _kestrel.get_active_tickers
mark_tickers_delisted = _kestrel.mark_tickers_delisted
upsert_universe_rows = _kestrel.upsert_universe_rows
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def _load_nasdaq_csv() -> pd.DataFrame:
    """
    Load the Nasdaq screener CSV and normalise column names.

    Returns:
        DataFrame with at minimum: ticker, exchange, sector, industry, mcap columns.

    Raises:
        FileNotFoundError: If the configured CSV path does not exist.
    """
    path = Path(str(NASDAQ_TICKERS_CSV))
    if not path.exists():
        raise FileNotFoundError(
            "Nasdaq ticker CSV not found at %s. "
            "Download from Nasdaq screener or configure NASDAQ_TICKERS_CSV." % path
        )
    df = pd.read_csv(path, dtype=str)
    rename = {
        "Symbol": "ticker",
        "Name": "company_name",
        "Market Cap": "mcap_raw",
        "Sector": "sector",
        "Industry": "industry",
        "Exchange": "exchange",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "ticker" not in df.columns:
        raise ValueError("CSV missing 'Symbol' column. Columns found: %s" % list(df.columns))
    df["ticker"] = df["ticker"].str.strip().str.upper()
    mask = df["ticker"].notna() & (df["ticker"] != "")
    result: pd.DataFrame = df[mask].reset_index(drop=True)  # type: ignore[assignment]
    return result


# Tickers that are pure uppercase alpha, 1–5 chars.
# Excludes: warrants (ABCDW), rights (ABCDR), units (ABCDU), slash-forms (BRK/B),
# preferred (ABC-A), test issues (ZZZ^), and anything with digits.
_STOCK_TICKER_RE = re.compile(r'^[A-Z]{1,5}$')
# Five-char tickers ending in W/R/U are almost always warrants, rights, or units.
_WARRANT_SUFFIX_RE = re.compile(r'^[A-Z]{4}[WRU]$')


def _is_stock_ticker(ticker: str) -> bool:
    """Return True if the ticker looks like an ordinary equity (not a warrant/right/unit)."""
    return bool(_STOCK_TICKER_RE.match(ticker)) and not bool(_WARRANT_SUFFIX_RE.match(ticker))


def _parse_mcap(raw: Any) -> Optional[float]:
    """Convert a market-cap string like '1.2B' or '450M' to a float in USD."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)) or raw == "":
        return None
    s = str(raw).strip().replace(",", "").replace("$", "")
    multipliers = {"T": 1e12, "B": 1e9, "M": 1e6, "K": 1e3}
    suffix = s[-1].upper() if s else ""
    if suffix in multipliers:
        try:
            return float(s[:-1]) * multipliers[suffix]
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


_FUNDAMENTALS_BATCH = 50


async def _fetch_fundamentals_for_ticker(ticker: str) -> Optional[Any]:
    """Fetch fundamentals for one ticker; return None on failure."""
    try:
        return await get_fundamentals_unified(ticker)
    except Exception:
        _logger.debug("No fundamentals for %s", ticker)
        return None


async def _fetch_all_fundamentals(tickers: List[str]) -> List[Optional[Any]]:
    """Fetch fundamentals for all tickers in concurrent batches to avoid rate limits."""
    results: List[Optional[Any]] = []
    for i in range(0, len(tickers), _FUNDAMENTALS_BATCH):
        batch = tickers[i: i + _FUNDAMENTALS_BATCH]
        batch_results = await asyncio.gather(
            *(_fetch_fundamentals_for_ticker(t) for t in batch),
            return_exceptions=True,
        )
        for r in batch_results:
            results.append(None if isinstance(r, BaseException) else r)
    return results


def run() -> Dict[str, Any]:
    """
    Refresh k20_universe from the Nasdaq CSV and fundamentals.

    Returns:
        Summary dict with tickers_upserted and tickers_delisted counts.
    """
    _logger.info("Loading Nasdaq ticker CSV from %s", NASDAQ_TICKERS_CSV)
    df = _load_nasdaq_csv()
    csv_tickers: set[str] = set(df["ticker"].tolist())
    _logger.info("CSV contains %d tickers", len(csv_tickers))

    # --- Pre-filter 1: ticker format (drop warrants, rights, units, test issues) ---
    before = len(df)
    df = df[df["ticker"].apply(_is_stock_ticker)]
    _logger.info("Ticker format filter: %d → %d (removed %d non-equity symbols)",
                 before, len(df), before - len(df))

    # --- Pre-filter 2: market cap floor when the CSV carries mcap data ---
    if "mcap_raw" in df.columns:
        mcap_vals = pd.Series(
            [_parse_mcap(v) for v in df["mcap_raw"]], index=df.index, dtype=object
        )
        below_floor = mcap_vals.notna() & (mcap_vals.astype(float) < UNIVERSE_MIN_MCAP_USD)
        if below_floor.any():
            _logger.info("Mcap filter (<$%.0f): removed %d tickers", UNIVERSE_MIN_MCAP_USD, int(below_floor.sum()))
            df = df[~below_floor].reset_index(drop=True)  # type: ignore[union-attr]

    tickers_list = df["ticker"].tolist()
    _logger.info(
        "Fetching fundamentals for %d tickers (batch size %d)",
        len(tickers_list), _FUNDAMENTALS_BATCH,
    )
    all_funds = asyncio.run(_fetch_all_fundamentals(tickers_list))

    rows: List[Dict[str, Any]] = []
    for (_, row), fund in zip(df.iterrows(), all_funds):
        ticker = str(row["ticker"])
        mcap = _parse_mcap(row.get("mcap_raw", ""))

        universe_row: Dict[str, Any] = {
            "ticker": ticker,
            "exchange": str(row.get("exchange") or ""),
            "sector": str(row.get("sector") or ""),
            "industry": str(row.get("industry") or ""),
            "mcap": mcap,
            "status": "active",
        }

        if fund is not None:
            universe_row.update({
                "revenue_yoy_growth": getattr(fund, "revenue_yoy_growth", None),
                "gross_margin": getattr(fund, "gross_margin", None),
                "net_debt_ebitda": getattr(fund, "net_debt_ebitda", None),
                "interest_coverage": getattr(fund, "interest_coverage", None),
                "mcap": getattr(fund, "market_cap", None) or mcap,
            })

        rows.append(universe_row)

    upserted = upsert_universe_rows(rows)
    _logger.info("Upserted %d universe rows", upserted)

    existing = set(get_active_tickers())
    to_delist = list(existing - csv_tickers)
    delisted = mark_tickers_delisted(to_delist) if to_delist else 0
    if delisted:
        _logger.info("Marked %d tickers as delisted", delisted)

    return {"tickers_upserted": upserted, "tickers_delisted": delisted}
