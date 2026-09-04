"""
Insider activity (Form 4).

Pulls the trailing N days of insider (Form 4) transactions for the caller's
currently held tickers out of the shared EDGAR daily cache maintained by
P18's daily scan (``edgar/13f/form4/{date}.csv.gz``) — no new EDGAR network
surface for the steady-state case, the same reuse pattern P19's structural
profiler uses (``structural/profiler.py``'s ``_load_form4_window``).
"""

import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from src.data.downloader.edgar_downloader import EdgarDownloader
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

# 30 calendar days is enough to cover a "what have insiders been doing lately"
# view without digging up ancient, no-longer-actionable transactions.
DEFAULT_LOOKBACK_DAYS = 30

# Bounds how long a cold/gappy cache is allowed to self-heal inline during a
# single run (same hazard P19's structural profiler guards against — see
# _WINDOW_WARMUP_BUDGET_SECONDS there). In steady state P18's own daily scan
# keeps this window fully cached, so this only matters after an outage.
_WINDOW_WARMUP_BUDGET_SECONDS = 60.0


@dataclass(frozen=True)
class InsiderTransaction:
    """
    One Form 4 non-derivative transaction for a currently held ticker.

    Attributes:
        ticker: Issuer ticker symbol (uppercase).
        insider_name: Reporting owner's name.
        role: Human-readable role summary, e.g. "Director", "Officer (CFO)",
            "10% Owner", or a " / "-joined combination. "Insider" if the
            filing carries none of the three role flags.
        transaction_code: SEC transaction code (P=open-market purchase,
            S=sale, A=grant/award, M=option exercise, F=tax withholding, ...).
        acquired_disposed_code: "A" (acquired) or "D" (disposed).
        shares: Number of shares transacted.
        price_per_share: Execution price (0.0 for codes with no cash price,
            e.g. grants).
        total_value_usd: `shares * price_per_share`.
        transaction_date: Actual trade date, ``"YYYY-MM-DD"``.
        filed_date: SEC receipt date, ``"YYYY-MM-DD"`` (up to 2 business days
            after `transaction_date`).
        is_10b5_1_plan: True if the filing flags this as a pre-scheduled Rule
            10b5-1 plan trade (non-discretionary — set up in advance, not a
            same-day decision).
    """

    ticker: str
    insider_name: str
    role: str
    transaction_code: str
    acquired_disposed_code: str
    shares: int
    price_per_share: float
    total_value_usd: float
    transaction_date: str
    filed_date: str
    is_10b5_1_plan: bool


def _describe_role(row: "pd.Series[Any]") -> str:
    """Build a human-readable role summary from the boolean role flags."""
    roles: List[str] = []
    if bool(row.get("is_director")):
        roles.append("Director")
    if bool(row.get("is_officer")):
        title = str(row.get("officer_title") or "").strip()
        roles.append(f"Officer ({title})" if title else "Officer")
    if bool(row.get("is_ten_percent_owner")):
        roles.append("10% Owner")
    return " / ".join(roles) if roles else "Insider"


def _load_form4_window(
    edgar: EdgarDownloader,
    today_utc: date,
    lookback_days: int,
) -> pd.DataFrame:
    """
    Concatenate the shared Form 4 daily cache over the trailing window.

    Never reads *today*'s date: a day's Form 4 filings aren't complete until
    the day has closed, and fetching it here would cache a partial same-day
    snapshot as if it were final (``download_form4_filings`` never re-fetches
    once the file exists), poisoning every later reader for the rest of that
    date's lifetime — see ``structural/profiler.py``'s ``_load_form4_window``
    docstring for the incident (2026-08-19) this mirrors.
    """
    start = today_utc - timedelta(days=lookback_days)
    window_start = time.monotonic()
    frames: List[pd.DataFrame] = []

    d = today_utc - timedelta(days=1)
    while d >= start:
        if d.weekday() < 5:
            if time.monotonic() - window_start > _WINDOW_WARMUP_BUDGET_SECONDS:
                _logger.warning(
                    "Form 4 window load exceeded its %.0fs budget — stopping early at %s "
                    "(window start %s); older days omitted from this run's insider activity",
                    _WINDOW_WARMUP_BUDGET_SECONDS,
                    d,
                    start,
                )
                break
            try:
                # force=False reads the on-disk cache written by P18's daily
                # scan; only a genuinely missing day triggers a (self-healing)
                # live EDGAR call.
                day_df = edgar.download_form4_filings(as_of_date=d, force=False)
                if day_df is not None and not day_df.empty:
                    frames.append(day_df)
            except Exception:
                _logger.debug("Form 4 cache read failed for %s", d)
        d -= timedelta(days=1)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_insider_activity(
    tickers: Sequence[str],
    edgar: Optional[EdgarDownloader] = None,
    as_of: Optional[date] = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> Dict[str, List[InsiderTransaction]]:
    """
    Load the trailing `lookback_days` of Form 4 transactions for `tickers`.

    Args:
        tickers: Ticker symbols to filter to (case-insensitive). Typically
            the caller's currently held symbols only — this is not a
            market-wide scan.
        edgar: Optional shared `EdgarDownloader` (created fresh if None).
        as_of: Reference "today" (UTC). Defaults to now. The window covers
            the `lookback_days` calendar days strictly before this date.
        lookback_days: Size of the trailing window in calendar days.

    Returns:
        Mapping of uppercased ticker -> list of `InsiderTransaction`, each
        list sorted by `transaction_date` descending (most recent first).
        Tickers with no activity in the window are omitted from the mapping.
    """
    if not tickers:
        return {}

    wanted = {t.upper() for t in tickers}
    edgar = edgar or EdgarDownloader()
    today_utc = as_of or datetime.now(timezone.utc).date()

    combined = _load_form4_window(edgar, today_utc, lookback_days)
    if combined.empty or "ticker" not in combined.columns:
        return {}

    matched = combined[combined["ticker"].astype(str).str.upper().isin(wanted)]
    if matched.empty:
        return {}

    out: Dict[str, List[InsiderTransaction]] = {}
    for _, row in matched.iterrows():
        ticker = str(row["ticker"]).upper()
        out.setdefault(ticker, []).append(
            InsiderTransaction(
                ticker=ticker,
                insider_name=str(row.get("insider_name", "")),
                role=_describe_role(row),
                transaction_code=str(row.get("transaction_code", "")),
                acquired_disposed_code=str(row.get("acquired_disposed_code", "")),
                shares=int(row.get("shares", 0) or 0),
                price_per_share=float(row.get("price_per_share", 0.0) or 0.0),
                total_value_usd=float(row.get("total_value_usd", 0.0) or 0.0),
                transaction_date=str(row.get("transaction_date", "") or row.get("filed_date", "")),
                filed_date=str(row.get("filed_date", "")),
                is_10b5_1_plan=bool(row.get("is_10b5_1_plan", False)),
            )
        )

    for txns in out.values():
        txns.sort(key=lambda t: t.transaction_date, reverse=True)

    _logger.info(
        "Loaded insider activity for %d/%d holdings (%d total transaction(s), %d-day window)",
        len(out),
        len(wanted),
        sum(len(v) for v in out.values()),
        lookback_days,
    )
    return out
