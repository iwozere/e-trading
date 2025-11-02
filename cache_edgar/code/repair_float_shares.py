#!/usr/bin/env python3
"""
Standalone backfill script: fetch historical shares outstanding from
SEC CompanyFacts (EDGAR) and save to CSV.

Usage:
  python backfill_edgar_floats_csv.py --ticker-file tickers.txt --dates 2024-01-15,2024-02-15 --output floats.csv

Notes:
 - The script caches SEC mapping and companyfacts to ./cache/ for resuming.
 - Respect SEC: set SEC_USER_AGENT env var or edit SEC_USER_AGENT below.
 - Specify tickers in a text file (one per line) and settlement dates via command line.
 - For large ticker lists (e.g., 11k tickers), processing is done in batches.

# Basic usage (uses tickers.txt by default)
python backfill_edgar_floats_csv.py --dates 2024-01-15,2024-02-15

# Custom ticker file
python backfill_edgar_floats_csv.py --ticker-file my_tickers.txt --dates 2024-01-15

# Adjust batch size for faster/slower processing
python backfill_edgar_floats_csv.py --dates 2024-01-15 --batch-size 50

# With verbose logging
python backfill_edgar_floats_csv.py --dates 2024-01-15 --verbose
"""
from __future__ import annotations
import os
import sys
import time
import json
import csv
import logging
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Dict, Any, List, Tuple

import requests

# --- Configuration ----------------------------------------------------------
# SEC requires a descriptive User-Agent. Prefer exporting SEC_USER_AGENT env var.
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "MyOrg MyApp contact@example.com")
CACHE_DIR = Path("./cache_edgar")
CIK_MAP_FILE = CACHE_DIR / "company_tickers.json"
COMPANYFACTS_DIR = CACHE_DIR / "companyfacts"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Rate limiting / batch options
DEFAULT_SLEEP_SEC = float(os.getenv("EDGAR_SLEEP_SEC", "0.5"))  # seconds between SEC requests

# Candidates list (preference order)
FACT_CANDIDATES = [
    ("us-gaap", "CommonStockSharesOutstanding"),
    ("us-gaap", "CommonStockSharesOutstandingPeriod"),
    ("us-gaap", "WeightedAverageNumberOfShareOutstandingBasic"),
    ("us-gaap", "SharesOutstanding"),
    ("dei", "EntityCommonStockSharesOutstanding"),
]

# ---------------------------------------------------------------------------

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("backfill_edgar_floats")

# Prepare cache dirs
CACHE_DIR.mkdir(parents=True, exist_ok=True)
COMPANYFACTS_DIR.mkdir(parents=True, exist_ok=True)

SEC_HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept": "application/json",
}

TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:0>10}.json"

# --- Utilities --------------------------------------------------------------
def load_ticker_to_cik_map() -> Dict[str, int]:
    """
    Load mapping from ticker -> CIK. Cache on disk.
    """
    if CIK_MAP_FILE.exists():
        try:
            with open(CIK_MAP_FILE, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            # file structure is mapping idx -> { "cik_str": ..., "ticker": ... }
            m = {}
            for v in raw.values():
                ticker = v.get("ticker")
                cik_str = v.get("cik_str")
                if ticker and cik_str:
                    m[ticker.upper()] = int(cik_str)
            logger.info("Loaded ticker->CIK map from cache (%d entries)", len(m))
            return m
        except Exception:
            logger.exception("Failed to load cached CIK map; fetching fresh.")
    # fetch remote
    logger.info("Fetching ticker->CIK mapping from SEC...")
    r = requests.get(TICKER_CIK_URL, headers=SEC_HEADERS, timeout=30)
    r.raise_for_status()
    j = r.json()
    with open(CIK_MAP_FILE, "w", encoding="utf-8") as fh:
        json.dump(j, fh)
    # convert
    m = {}
    for v in j.values():
        t = v.get("ticker")
        cik = v.get("cik_str")
        if t and cik:
            m[t.upper()] = int(cik)
    logger.info("Fetched ticker->CIK map (%d entries)", len(m))
    return m

def fetch_companyfacts_json(cik: int, sleep_sec: float = DEFAULT_SLEEP_SEC) -> Optional[Dict[str, Any]]:
    """
    Fetch companyfacts JSON for a given CIK and cache on disk under COMPANYFACTS_DIR.
    """
    cache_file = COMPANYFACTS_DIR / f"{int(cik):0>10}.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            logger.exception("Failed to load cached companyfacts for %s; will refetch.", cik)
    url = COMPANYFACTS_URL.format(cik=int(cik))
    logger.debug("Fetching companyfacts for CIK %s -> %s", cik, url)
    r = requests.get(url, headers=SEC_HEADERS, timeout=30)
    if r.status_code == 404:
        logger.warning("Companyfacts not found (404) for CIK %s", cik)
        return None
    r.raise_for_status()
    data = r.json()
    with open(cache_file, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    # be polite
    time.sleep(sleep_sec)
    return data

def pick_best_shares_from_companyfacts(cf_json: Dict[str, Any], settlement_date: date) -> Optional[Dict[str, Any]]:
    """
    Given companyfacts JSON and a settlement_date, pick the best matching shares outstanding fact
    whose 'end' date is <= settlement_date and with the most recent end.
    Returns dict {value:int, effective_date:date, fact:str, unit:str} or None.
    """
    facts = cf_json.get("facts", {})
    best = None  # tuple(end_date:date, value:int, fact_name:str, unit_name:str)
    for space, fname in FACT_CANDIDATES:
        space_obj = facts.get(space, {})
        fact_obj = space_obj.get(fname)
        if not fact_obj:
            continue
        units = fact_obj.get("units", {})
        for unit_name, entries in units.items():
            if not isinstance(entries, list):
                continue
            for e in entries:
                end = e.get("end") or e.get("instant")
                if not end:
                    continue
                try:
                    end_date = datetime.strptime(end[:10], "%Y-%m-%d").date()
                except Exception:
                    continue
                if end_date <= settlement_date:
                    val = e.get("val")
                    if val is None:
                        continue
                    try:
                        val_int = int(float(val))
                    except Exception:
                        continue
                    if best is None or end_date > best[0]:
                        best = (end_date, val_int, f"{space}:{fname}", unit_name)
    if best:
        return {
            "value": best[1],
            "effective_date": best[0],
            "fact": best[2],
            "unit": best[3],
        }
    return None

# --- CSV helpers ------------------------------------------------------------
def write_to_csv(output_file: Path, rows: List[Dict[str, Any]]):
    """
    Write results to CSV file with columns: ticker, settlement_date, float_shares, effective_date, fact, source
    """
    if not rows:
        logger.warning("No rows to write to CSV")
        return

    fieldnames = ["ticker", "settlement_date", "float_shares", "effective_date", "fact", "source"]

    # Check if file exists to determine if we need to write header
    write_header = not output_file.exists()

    with open(output_file, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote %d rows to %s", len(rows), output_file)

# --- Main logic -------------------------------------------------------------
def process_batch(pairs: List[Tuple[str, date]], cik_map: Dict[str,int],
                  sleep_sec: float = DEFAULT_SLEEP_SEC, use_fallback_latest: bool = False) -> List[Dict[str, Any]]:
    """
    Process a list of (ticker, settlement_date) pairs.
    Returns list of result dictionaries ready for CSV output.
    """
    results = []

    # Group pair lists by ticker for efficiency
    by_ticker: Dict[str, List[date]] = {}
    for ticker, sdate in pairs:
        by_ticker.setdefault(ticker.upper(), []).append(sdate)

    for ticker, dates in by_ticker.items():
        logger.info("Processing ticker %s (%d dates)", ticker, len(dates))
        cik = cik_map.get(ticker)
        if not cik:
            logger.warning("No CIK mapping for ticker %s — skipping", ticker)
            continue
        cf = fetch_companyfacts_json(cik, sleep_sec=sleep_sec)
        if not cf:
            logger.warning("No companyfacts for CIK %s ticker %s — skipping", cik, ticker)
            continue

        # For each settlement date for this ticker, pick best fact and collect results
        for sdate in sorted(set(dates)):
            try:
                res = pick_best_shares_from_companyfacts(cf, sdate)
                if not res:
                    if use_fallback_latest:
                        # pick latest available (largest effective_date), even if > settlement_date
                        logger.debug("No prior fact <= %s for %s; trying fallback latest", sdate.isoformat(), ticker)
                        res2 = pick_best_shares_from_companyfacts(cf, date.today())
                        if res2:
                            res = res2
                    if not res:
                        logger.info("No shares fact found for %s @ %s", ticker, sdate.isoformat())
                        continue

                value = res["value"]
                eff_date = res["effective_date"]
                fact = res["fact"]

                results.append({
                    "ticker": ticker,
                    "settlement_date": sdate.isoformat(),
                    "float_shares": value,
                    "effective_date": eff_date.isoformat(),
                    "fact": fact,
                    "source": "sec-companyfacts"
                })

                logger.info("Found float: %s @ %s -> %s shares (effective: %s)",
                           ticker, sdate.isoformat(), value, eff_date.isoformat())

            except Exception:
                logger.exception("Error while processing %s@%s", ticker, sdate)

    return results

# --- CLI / Runner -----------------------------------------------------------
def main(argv: List[str] = None):
    import argparse
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser()
    p.add_argument("--ticker-file", type=str, default="tickers.txt", help="Path to text file with one ticker per line (default: tickers.txt)")
    p.add_argument("--dates", type=str, required=True, help="Comma-separated list of settlement dates in YYYY-MM-DD format")
    p.add_argument("--output", type=str, default="edgar_floats.csv", help="Output CSV filename (default: edgar_floats.csv)")
    p.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SEC, help="Seconds to sleep between SEC requests (default 0.5)")
    p.add_argument("--batch-size", type=int, default=100, help="Number of tickers to process in each batch (default 100)")
    p.add_argument("--use-fallback-latest", action="store_true", help="If no prior fact exists, fallback to latest available fact (less correct)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Read tickers from file
    ticker_file = Path(args.ticker_file)
    if not ticker_file.exists():
        logger.error("Ticker file not found: %s", ticker_file)
        sys.exit(1)

    with open(ticker_file, "r", encoding="utf-8") as f:
        tickers = [line.strip().upper() for line in f if line.strip()]

    if not tickers:
        logger.error("No tickers found in %s", ticker_file)
        sys.exit(1)

    logger.info("Loaded %d tickers from %s", len(tickers), ticker_file)

    # Parse dates
    dates = []
    for d in args.dates.split(","):
        d = d.strip()
        if not d:
            continue
        try:
            dates.append(datetime.strptime(d, "%Y-%m-%d").date())
        except ValueError:
            logger.error("Invalid date format: %s (expected YYYY-MM-DD)", d)
            sys.exit(1)

    if not dates:
        logger.error("No valid dates provided")
        sys.exit(1)

    logger.info("Processing %d tickers and %d dates = %d total pairs", len(tickers), len(dates), len(tickers) * len(dates))

    # Build mapping
    cik_map = load_ticker_to_cik_map()

    # Create pairs of all ticker/date combinations
    all_pairs = [(ticker, sdate) for ticker in tickers for sdate in dates]

    # Process in batches to avoid overwhelming memory with 11k tickers
    output_path = Path(args.output)
    total_results = 0

    for i in range(0, len(all_pairs), args.batch_size):
        batch_pairs = all_pairs[i:i + args.batch_size]
        batch_num = (i // args.batch_size) + 1
        total_batches = (len(all_pairs) + args.batch_size - 1) // args.batch_size

        logger.info("Processing batch %d/%d (%d pairs)", batch_num, total_batches, len(batch_pairs))

        # Process batch
        results = process_batch(batch_pairs, cik_map, sleep_sec=args.sleep, use_fallback_latest=args.use_fallback_latest)

        # Write to CSV (append mode)
        write_to_csv(output_path, results)
        total_results += len(results)

        logger.info("Batch %d/%d complete. Results so far: %d", batch_num, total_batches, total_results)

    logger.info("Done. Processed %d ticker/date pairs, wrote %d results to %s",
                len(all_pairs), total_results, output_path)

if __name__ == "__main__":
    main()