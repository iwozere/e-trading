"""
P20 Kestrel — Universe loader.

Reads the Nasdaq ticker CSV, enriches with fundamentals, and upserts
rows into k20_universe. Marks tickers absent from the CSV as 'delisted'.
"""

from __future__ import annotations

import asyncio
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.common.fundamentals import get_fundamentals_unified
from src.data.db.services.kestrel_service import KestrelService as _KestrelService
from src.ml.pipeline.p20_kestrel.config import NASDAQ_TICKERS_CSV, UNIVERSE_MIN_MCAP_USD

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
            "Nasdaq ticker CSV not found at %s. Download from Nasdaq screener or configure NASDAQ_TICKERS_CSV." % path
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
_STOCK_TICKER_RE = re.compile(r"^[A-Z]{1,5}$")
# Five-char tickers ending in W/R/U are almost always warrants, rights, or units.
_WARRANT_SUFFIX_RE = re.compile(r"^[A-Z]{4}[WRU]$")


def _is_stock_ticker(ticker: str) -> bool:
    """Return True if the ticker looks like an ordinary equity (not a warrant/right/unit)."""
    return bool(_STOCK_TICKER_RE.match(ticker)) and not bool(_WARRANT_SUFFIX_RE.match(ticker))


def _parse_mcap(raw: Any) -> float | None:
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


_FUNDAMENTALS_WORKERS = 1  # concurrent Yahoo fetch threads
_PROGRESS_LOG_EVERY = 500  # progress log interval (tickers)


def _fetch_fundamentals_for_ticker(ticker: str) -> Any | None:
    """Fetch fundamentals for one ticker using Yahoo only; return None on failure.

    Yahoo is used exclusively here: FMP/Polygon/TwelveData all hit free-tier
    rate limits or 403s on the profile endpoint, adding noise and latency
    without contributing data. The multi-provider merge is reserved for
    individual ticker lookups where data quality matters more than speed.
    """
    try:
        # get_fundamentals_unified is declared async but performs synchronous
        # I/O internally — run it in this worker thread's own event loop.
        return asyncio.run(get_fundamentals_unified(ticker, provider="yf"))
    except Exception:
        _logger.debug("No fundamentals for %s", ticker)
        return None


_RETRY_COOLDOWN_SECONDS = 300  # 5-minute global wait before the retry pass


def _fetch_all_fundamentals(tickers: List[str]) -> Iterator[Tuple[str, Any | None]]:
    """
    Fetch fundamentals for all tickers in two passes, yielding results as they
    complete so the caller can persist progress incrementally (a run killed by
    the scheduler timeout mid-fetch still keeps everything upserted so far).

    Pass 1 (parallel):
        Run with _FUNDAMENTALS_WORKERS threads.  On a 429, data_manager
        breaks immediately (no per-ticker waits) so the main pass moves fast.

    Pass 2 (single-threaded, after a global cooldown):
        Re-attempt every ticker that returned None.  By the time we reach
        this pass the IP block from Yahoo has typically expired.  A ticker
        that failed in pass 1 is yielded twice — once as None, once with the
        pass-2 result if recovered; the caller's upsert is last-write-wins.

    Yields:
        (ticker, fundamentals) pairs, fundamentals may be None on failure.
    """
    import time

    total = len(tickers)

    # ── Pass 1 ────────────────────────────────────────────────────────────
    failed: List[str] = []
    ok_count = 0
    with ThreadPoolExecutor(max_workers=_FUNDAMENTALS_WORKERS) as pool:
        for i, (ticker, fund) in enumerate(zip(tickers, pool.map(_fetch_fundamentals_for_ticker, tickers))):
            if fund is None:
                failed.append(ticker)
            else:
                ok_count += 1
            yield ticker, fund
            done = i + 1
            if done % _PROGRESS_LOG_EVERY == 0 or done == total:
                _logger.info("Pass 1 progress: %d/%d tickers", done, total)

    _logger.info(
        "Pass 1 complete: %d/%d succeeded, %d to retry",
        ok_count,
        total,
        len(failed),
    )

    if not failed:
        return

    # ── Cooldown ──────────────────────────────────────────────────────────
    _logger.info(
        "Waiting %d seconds for Yahoo rate-limit to reset before retry pass ...",
        _RETRY_COOLDOWN_SECONDS,
    )
    time.sleep(_RETRY_COOLDOWN_SECONDS)

    # ── Pass 2 (single-threaded) ──────────────────────────────────────────
    _logger.info("Pass 2: retrying %d failed tickers (single-threaded)", len(failed))
    recovered = 0
    for ticker in failed:
        result = _fetch_fundamentals_for_ticker(ticker)
        if result is not None:
            recovered += 1
            _logger.debug("Pass 2 recovered: %s", ticker)
        else:
            _logger.warning("Pass 2 still failed: %s", ticker)
        yield ticker, result

    _logger.info(
        "Pass 2 complete: recovered %d/%d previously-failed tickers",
        recovered,
        len(failed),
    )


_UPSERT_CHUNK_SIZE = 200  # persist progress every N tickers so a scheduler
# timeout mid-fetch doesn't discard everything already fetched


def run() -> Dict[str, Any]:
    """
    Refresh k20_universe from the Nasdaq CSV and fundamentals.

    Universe rows are upserted in chunks as fundamentals come in (rather than
    once at the end) so a run killed by the scheduler timeout still leaves
    k20_universe reflecting everything fetched up to that point.

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
    _logger.info("Ticker format filter: %d → %d (removed %d non-equity symbols)", before, len(df), before - len(df))

    # --- Pre-filter 2: market cap floor when the CSV carries mcap data ---
    if "mcap_raw" in df.columns:
        mcap_vals = pd.Series([_parse_mcap(v) for v in df["mcap_raw"]], index=df.index, dtype=object)
        below_floor = mcap_vals.notna() & (mcap_vals.astype(float) < UNIVERSE_MIN_MCAP_USD)
        if below_floor.any():
            _logger.info("Mcap filter (<$%.0f): removed %d tickers", UNIVERSE_MIN_MCAP_USD, below_floor.sum())
            df = df[~below_floor].reset_index(drop=True)  # type: ignore[union-attr]

    rows_by_ticker: Dict[str, Any] = {str(row["ticker"]): row for _, row in df.iterrows()}
    tickers_list = df["ticker"].tolist()
    _logger.info(
        "Fetching fundamentals for %d tickers (%d worker threads)",
        len(tickers_list),
        _FUNDAMENTALS_WORKERS,
    )

    upserted = 0
    buffer: List[Dict[str, Any]] = []

    def _flush() -> None:
        nonlocal upserted
        if not buffer:
            return
        count = upsert_universe_rows(buffer)
        upserted += count
        _logger.info("Upserted %d universe rows (%d total so far)", count, upserted)
        buffer.clear()

    for ticker, fund in _fetch_all_fundamentals(tickers_list):
        row = rows_by_ticker[ticker]
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
            universe_row.update(
                {
                    "revenue_yoy_growth": getattr(fund, "revenue_yoy_growth", None),
                    "gross_margin": getattr(fund, "gross_margin", None),
                    "net_debt_ebitda": getattr(fund, "net_debt_ebitda", None),
                    "interest_coverage": getattr(fund, "interest_coverage", None),
                    "mcap": getattr(fund, "market_cap", None) or mcap,
                }
            )

        buffer.append(universe_row)
        if len(buffer) >= _UPSERT_CHUNK_SIZE:
            _flush()

    _flush()

    existing = set(get_active_tickers())
    to_delist = list(existing - csv_tickers)
    delisted = mark_tickers_delisted(to_delist) if to_delist else 0
    if delisted:
        _logger.info("Marked %d tickers as delisted", delisted)

    return {"tickers_upserted": upserted, "tickers_delisted": delisted}
