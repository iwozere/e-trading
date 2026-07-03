"""
P20 Kestrel — Filings ingest.

Discovery via P15's edgar_8k_index cache (no duplicate download).
Fetches filing bodies for watchlist + positions tickers only.
Handles Form 4 insider buys, 8-K/PR LLM queue, and 13D/G activist signals.
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.data.downloader.edgar_downloader import EdgarDownloader
from src.ml.pipeline.p20_kestrel.config import ACTIVISTS_JSON, DATA_CACHE_PATH
from src.data.db.services.kestrel_service import KestrelService as _KestrelService

_kestrel = _KestrelService()
finish_job_run = _kestrel.finish_job_run
get_open_positions = _kestrel.get_open_positions
get_watchlist_tickers = _kestrel.get_watchlist_tickers
start_job_run = _kestrel.start_job_run
upsert_signals = _kestrel.upsert_signals
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_JOB_NAME = "ingest_filings"
# Paths written by P15 daily; P20 reads only — never downloads.
_P15_8K_INDEX_DIR = DATA_CACHE_PATH / "edgar" / "8k" / "index"
_P15_FORM4_DIR = DATA_CACHE_PATH / "edgar" / "13f" / "form4"
_P15_13DG_DIR = DATA_CACHE_PATH / "edgar" / "13f" / "13dg"
_FORM4_BUY_CODES = {"P", "A"}  # P = open market purchase, A = grant


def _get_target_tickers() -> Set[str]:
    """Return the union of watchlist + positions tickers — the filing discovery scope."""
    watchlist = set(get_watchlist_tickers())
    positions = {p["ticker"] for p in get_open_positions()}
    return watchlist | positions


def _build_cik_to_ticker() -> Dict[str, str]:
    """
    Build a CIK → ticker reverse map from EdgarDownloader's company_tickers.json cache.

    Returns:
        Dict mapping zero-stripped CIK strings to uppercase ticker symbols.
        Empty dict if the cache file is absent or malformed.
    """
    try:
        tickers_file = EdgarDownloader().download_company_tickers()
        with open(tickers_file, encoding="utf-8") as f:
            raw = json.load(f)
        return {
            str(int(v.get("cik_str", 0))): v.get("ticker", "").upper()
            for v in raw.values()
            if v.get("ticker") and v.get("cik_str")
        }
    except Exception:
        _logger.warning("Could not load company_tickers.json for CIK→ticker mapping")
        return {}


def _read_p15_8k_index(
    as_of_date: date, cik_map: Optional[Dict[str, str]] = None
) -> List[Dict[str, Any]]:
    """
    Read the 8-K filing index written by P15 for the given date.

    P15 saves DATA_CACHE_DIR/edgar/8k/index/YYYY-MM-DD.csv.gz with columns:
    cik, company, accession_number, items, description, filed_date, primary_document.
    This function adds a 'ticker' field resolved from the CIK→ticker map.

    Args:
        as_of_date: The date to read the index for.

    Returns:
        List of dicts enriched with a 'ticker' key (may be empty string if CIK unknown).
    """
    index_file = _P15_8K_INDEX_DIR / f"{as_of_date.isoformat()}.csv.gz"
    if not index_file.exists():
        _logger.debug("No P15 8-K index for %s at %s", as_of_date, index_file)
        return []

    try:
        df = pd.read_csv(index_file, compression="gzip", dtype=str)
        if cik_map is None:
            cik_map = _build_cik_to_ticker()
        records = []
        for _, row in df.iterrows():
            cik = str(row.get("cik", "") or "").lstrip("0")
            ticker = cik_map.get(cik, "")
            records.append({
                "ticker": ticker,
                "cik": row.get("cik", ""),
                "company": row.get("company", ""),
                "accession_number": row.get("accession_number", ""),
                "items": row.get("items", ""),
                "description": row.get("description", ""),
                "filed_date": row.get("filed_date", str(as_of_date)),
                "primary_document": row.get("primary_document", ""),
            })
        return records
    except Exception:
        _logger.exception("Failed to read P15 8-K index %s", index_file)
        return []


def _read_p15_13dg(as_of_date: date) -> List[Dict[str, Any]]:
    """
    Read the 13D/G filing cache written by P15 for the given date.

    P15 saves DATA_CACHE_DIR/edgar/13f/13dg/YYYY-MM-DD.csv.gz with columns:
    cik, entity_name, accession_number, filed_date, form_type.
    Note: these columns describe the FILER (activist investor), not the subject company.
    Ticker-based matching is therefore not available without parsing the full filing.

    Args:
        as_of_date: The date to read the cache for.

    Returns:
        List of dicts with the raw EDGAR form-index fields.
    """
    cache_file = _P15_13DG_DIR / f"{as_of_date.isoformat()}.csv.gz"
    if not cache_file.exists():
        _logger.debug("No P15 13D/G cache for %s", as_of_date)
        return []

    try:
        df = pd.read_csv(cache_file, compression="gzip", dtype=str)
        return df.to_dict("records")
    except Exception:
        _logger.exception("Failed to read P15 13D/G cache %s", cache_file)
        return []


def _process_form4(as_of_date: date, target_tickers: Set[str]) -> int:
    """
    Read Form 4 filings from the P15 cache, filter to buy transactions on target
    tickers, and record net insider-buy-value signals.

    P15 daily writes DATA_CACHE_DIR/edgar/13f/form4/YYYY-MM-DD.csv.gz.
    Columns: ticker, issuer_cik, insider_name, transaction_code,
             shares, price_per_share, total_value_usd, filed_date.

    Args:
        as_of_date: Date to process.
        target_tickers: Tickers to filter for.

    Returns:
        Number of buy transactions matched.
    """
    cache_file = _P15_FORM4_DIR / f"{as_of_date.isoformat()}.csv.gz"
    if not cache_file.exists():
        _logger.debug("No P15 Form 4 cache for %s at %s", as_of_date, cache_file)
        return 0

    try:
        df = pd.read_csv(cache_file, compression="gzip")
        if df.empty:
            return 0

        target_list = list(target_tickers)
        buy_codes_list = list(_FORM4_BUY_CODES)
        df_buys = df[
            df["transaction_code"].isin(buy_codes_list) &
            df["ticker"].str.upper().isin(target_list)
        ]
        if df_buys.empty:
            return 0

        agg = df_buys.groupby("ticker")["total_value_usd"].sum().reset_index()
        signal_rows = []
        for _, row in agg.iterrows():
            signal_rows.append({
                "ticker": str(row["ticker"]).upper(),
                "date": as_of_date,
                "signal_type": "insider_buy_value_90d",
                "value": float(row["total_value_usd"]),
            })

        if signal_rows:
            upsert_signals(signal_rows)
            _logger.info("Form 4: recorded insider buys for %d tickers", len(signal_rows))

        return int(df_buys.shape[0])
    except Exception:
        _logger.exception("Form 4 processing failed for %s", as_of_date)
        return 0


def _load_activist_aliases() -> List[str]:
    """
    Load lowercase activist name aliases from the curated ACTIVISTS_JSON list.

    Returns:
        Flat list of lowercase alias strings. Empty if file missing/malformed.
    """
    try:
        with open(ACTIVISTS_JSON, encoding="utf-8") as f:
            raw = json.load(f)
        return [
            alias.lower()
            for entry in raw.get("activists", [])
            for alias in entry.get("aliases", [])
        ]
    except Exception:
        _logger.warning("Could not load activists list from %s", ACTIVISTS_JSON)
        return []


def _process_13dg_activist(
    filings_13dg: List[Dict[str, Any]],
    target_tickers: Set[str],
    cik_map: Dict[str, str],
    as_of_date: date,
) -> int:
    """
    Match 13D/G filings to tickers and record activist signals.

    EDGAR's form index lists each SC 13D/G under BOTH the subject company and
    the filing person, sharing one accession number. The subject company is a
    listed issuer whose CIK resolves via company_tickers.json; the filer (a
    fund) normally does not. Grouping rows by accession therefore yields the
    subject ticker plus the filer name(s) for each filing.

    A signal is recorded when:
    - the subject ticker is in target_tickers (watchlist + positions), OR
    - any filer name matches the curated activist list (Sleeve B3 discovery).

    Args:
        filings_13dg: Raw rows from the P15 cache for one filing date.
        target_tickers: Watchlist + positions tickers.
        cik_map: Zero-stripped CIK → ticker map.
        as_of_date: Filing date being processed (used as the signal date).

    Returns:
        Number of filings recorded as signals.
    """
    activist_aliases = _load_activist_aliases()

    by_accession: Dict[str, List[Dict[str, Any]]] = {}
    for f in filings_13dg:
        acc = str(f.get("accession_number", ""))
        if acc:
            by_accession.setdefault(acc, []).append(f)

    matched = 0
    for acc, rows in by_accession.items():
        subject_ticker = ""
        filer_names: List[str] = []
        form_type = ""

        for r in rows:
            form_type = str(r.get("form_type", "")) or form_type
            cik_raw = str(r.get("cik", "") or "").strip().lstrip("0")
            ticker = cik_map.get(cik_raw, "")
            if ticker:
                subject_ticker = ticker.upper()
            else:
                name = str(r.get("entity_name", "")).strip()
                if name:
                    filer_names.append(name)

        if not subject_ticker:
            continue

        is_known_activist = any(
            alias in name.lower()
            for name in filer_names
            for alias in activist_aliases
        )

        if subject_ticker not in target_tickers and not is_known_activist:
            continue

        # 13D = active intent; 13G = passive stake
        is_13d = "13D" in form_type.upper()
        signal_type = "activist_13d" if is_13d else "activist_13g"
        try:
            upsert_signals([{
                "ticker": subject_ticker,
                "date": as_of_date,
                "signal_type": signal_type,
                "value": 1.0,
            }])
            matched += 1
            _logger.info(
                "13D/G matched: %s %s (filer: %s%s)",
                subject_ticker, form_type,
                filer_names[0] if filer_names else "unknown",
                ", known activist" if is_known_activist else "",
            )
        except Exception:
            _logger.exception("Failed to record 13D/G signal for %s", subject_ticker)

    return matched


def run(as_of_date: Optional[date] = None, lookback_days: int = 1) -> Dict[str, Any]:
    """
    Discover and ingest filings for the watchlist + positions universe.

    Args:
        as_of_date: Date to process (defaults to today).
        lookback_days: Number of days to look back in P15 cache (default 1).

    Returns:
        Summary dict.
    """
    target_date = as_of_date or date.today()
    _logger.info("Filings ingest for %s (lookback=%d)", target_date, lookback_days)
    start_job_run(_JOB_NAME, target_date)

    try:
        target_tickers = _get_target_tickers()
        _logger.info("Monitoring %d tickers for filings", len(target_tickers))

        form4_buys = 0
        activist_matches = 0
        llm_queue: List[Dict[str, Any]] = []

        dates_to_check = [
            target_date - timedelta(days=i) for i in range(lookback_days)
        ]

        cik_map = _build_cik_to_ticker()

        for check_date in dates_to_check:
            # 8-K discovery from P15 cache (CIK→ticker already resolved)
            filings_8k = _read_p15_8k_index(check_date, cik_map)
            for f in filings_8k:
                ticker = str(f.get("ticker", "")).upper()
                if ticker in target_tickers:
                    llm_queue.append(f)

            # 13D/G activist signals from P15 cache
            filings_13dg = _read_p15_13dg(check_date)
            activist_matches += _process_13dg_activist(
                filings_13dg, target_tickers, cik_map, check_date
            )

            # Form 4 insider buying from P15 cache
            form4_buys += _process_form4(check_date, target_tickers)

        # Save 8-K queue for LLM classifier
        queue_file = Path(str(PROJECT_ROOT)) / "results" / "p20_kestrel" / "llm_queue.json"
        queue_file.parent.mkdir(parents=True, exist_ok=True)
        with open(queue_file, "w", encoding="utf-8") as fh:
            json.dump(llm_queue, fh)
        _logger.info("Wrote %d 8-K filings to LLM queue", len(llm_queue))

        summary = {
            "target_tickers": len(target_tickers),
            "filings_8k_queued": len(llm_queue),
            "form4_buys": form4_buys,
            "activist_matches": activist_matches,
        }
        finish_job_run(_JOB_NAME, target_date, status="ok", rows_out=len(llm_queue))
        return summary

    except Exception as exc:
        _logger.exception("Filings ingest failed")
        finish_job_run(_JOB_NAME, target_date, status="failed", error=str(exc))
        raise
