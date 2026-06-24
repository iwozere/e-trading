"""
P18 Institutional Flow Tracker — Full backfill via EDGAR bulk index files.

Bypasses the broken EFTS pagination by using EDGAR's static quarterly
form.gz index files directly.  Seeds the Q4 2025 and Q1 2026 index caches
so that rebuild_quarterly_consensus can download infotables and build the
consensus that the daily pipeline needs.

Expected runtime: 45 – 120 minutes (depends on cache state and EDGAR rate limits).
The script is fully resumable: already-cached infotables are skipped.

Run from the project root with the venv activated:
    python src/ml/pipeline/p18_institutional_flow_tracker/backfill_p18.py

Optional flags:
    --skip-q4-index   Skip re-downloading the Q4 2025 bulk index (use if already done)
    --skip-q1-index   Skip re-downloading the Q1 2026 bulk index (use if already done)
    --force           Force re-download of all infotables even if cached
"""

import argparse
import gzip
import io
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

import requests
import pandas as pd

from src.notification.logger import setup_logger
from src.data.downloader.edgar_downloader import EdgarDownloader
from src.ml.pipeline.p18_institutional_flow_tracker.config import P18Config
from src.ml.pipeline.p18_institutional_flow_tracker.pipeline import InstitutionalFlowPipeline

_logger = setup_logger(__name__)

# Q4 2025: filed Jan 1 – Feb 14 2026  → EDGAR QTR1 2026 bulk file
# Q1 2026: filed Apr 1 – May 15 2026  → EDGAR QTR2 2026 bulk file
_QUARTERS = [
    dict(label="Q4 2025", year=2025, quarter=4, edgar_year=2026, edgar_qtr=1,
         window_start="2026-01-01", window_end="2026-02-14"),
    dict(label="Q1 2026", year=2026, quarter=1, edgar_year=2026, edgar_qtr=2,
         window_start="2026-04-01", window_end="2026-05-15"),
]


def _download_bulk_index(edgar_year: int, edgar_qtr: int, session: requests.Session) -> str:
    """Download and decompress EDGAR quarterly form.gz index; return raw text."""
    url = f"https://www.sec.gov/Archives/edgar/full-index/{edgar_year}/QTR{edgar_qtr}/form.gz"
    _logger.info("Downloading EDGAR bulk index from %s", url)
    resp = session.get(url, timeout=120)
    resp.raise_for_status()
    with gzip.open(io.BytesIO(resp.content), "rt", encoding="latin-1") as fh:
        return fh.read()


def _parse_13f_records(raw_text: str, window_start: str, window_end: str) -> pd.DataFrame:
    """
    Extract 13F-HR records within the filing window from a raw form.gz text.

    The EDGAR form.gz uses fixed-width columns but company names contain spaces,
    so we match with a regex anchored on the form type and the trailing date/filename.
    """
    pattern = re.compile(
        r'^13F-HR\s+(.+?)\s{2,}(\d{6,12})\s+(\d{4}-\d{2}-\d{2})\s+'
        r'(edgar/data/\d+/[\w./-]+-index\.htm)',
        re.MULTILINE,
    )
    records = []
    for m in pattern.finditer(raw_text):
        company = m.group(1).strip()
        cik = str(int(m.group(2)))          # strip leading zeros
        filed_date = m.group(3)
        filename = m.group(4)

        # Accession number lives in the filename path: .../0001234567-26-000001-index.htm
        acc_m = re.search(r'(\d{10}-\d{2}-\d{6})-index', filename)
        if not acc_m:
            continue

        records.append({
            "cik": cik,
            "institution_name": company,
            "accession_number": acc_m.group(1),
            "filed_date": filed_date,
        })

    df = pd.DataFrame(records) if records else pd.DataFrame(
        columns=["cik", "institution_name", "accession_number", "filed_date"]
    )

    if df.empty:
        return df

    df["filed_date"] = pd.to_datetime(df["filed_date"])
    mask = (df["filed_date"] >= window_start) & (df["filed_date"] <= window_end)
    df = df[mask].copy()
    df["filed_date"] = df["filed_date"].dt.strftime("%Y-%m-%d")
    return df.reset_index(drop=True)


def _seed_index_cache(edgar: EdgarDownloader, index_df: pd.DataFrame, year: int, quarter: int) -> None:
    """Write index_df to the path that download_13f_index uses as its cache."""
    dest = edgar._13f_index_dir / f"{year}_Q{quarter}.csv.gz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    index_df.to_csv(dest, index=False, compression="gzip")
    _logger.info("Seeded index cache: %d filers → %s", len(index_df), dest)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P18 full backfill via EDGAR bulk index.")
    p.add_argument("--skip-q4-index", action="store_true", help="Skip downloading Q4 2025 bulk index")
    p.add_argument("--skip-q1-index", action="store_true", help="Skip downloading Q1 2026 bulk index")
    p.add_argument("--force", action="store_true", help="Force re-download of all infotables")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    config = P18Config.create_default()
    edgar = EdgarDownloader()

    session = requests.Session()
    session.headers.update({
        "User-Agent": "research alkotrader@gmail.com",
        "Accept-Encoding": "gzip, deflate",
    })

    skip_flags = {
        ("Q4 2025", 2025, 4): args.skip_q4_index,
        ("Q1 2026", 2026, 1): args.skip_q1_index,
    }

    # ------------------------------------------------------------------ #
    # Step 1: Seed the index cache for both quarters from bulk files.
    # rebuild_quarterly_consensus will use these cached indices instead of
    # calling EFTS when it fetches per-institution prior-quarter holdings.
    # ------------------------------------------------------------------ #
    for qinfo in _QUARTERS:
        label, year, quarter = qinfo["label"], qinfo["year"], qinfo["quarter"]
        dest = edgar._13f_index_dir / f"{year}_Q{quarter}.csv.gz"

        if skip_flags.get((label, year, quarter)) and dest.exists():
            _logger.info("%s index already cached at %s — skipping download", label, dest)
            continue

        raw = _download_bulk_index(qinfo["edgar_year"], qinfo["edgar_qtr"], session)
        index_df = _parse_13f_records(raw, qinfo["window_start"], qinfo["window_end"])

        if index_df.empty:
            _logger.error("%s: no 13F-HR records found in bulk index — aborting", label)
            return 1

        _logger.info("%s: found %d 13F-HR filers in window %s → %s",
                     label, len(index_df), qinfo["window_start"], qinfo["window_end"])
        _seed_index_cache(edgar, index_df, year, quarter)

    # ------------------------------------------------------------------ #
    # Step 2: Build Q1 2026 consensus.
    #
    # rebuild_quarterly_consensus will:
    #   - Load the cached Q1 2026 index (seeded above)
    #   - Download each Q1 2026 infotable from EDGAR (skips if cached)
    #   - For each institution call load_13f_holdings(cik, 2025, 4) which:
    #       * finds the cached Q4 2025 index (seeded above)
    #       * downloads the Q4 2025 infotable if not cached
    #   - Compute delta, exit screen, consensus
    #   - Write consensus to {edgar_dir}/13f/consensus/2026_Q1.csv.gz
    # ------------------------------------------------------------------ #
    _logger.info("Starting rebuild_quarterly_consensus for 2026 Q1 ...")
    pipeline = InstitutionalFlowPipeline(config)
    consensus_df = pipeline.rebuild_quarterly_consensus(2026, 1, force_download=args.force)

    if consensus_df.empty:
        _logger.error(
            "Consensus is empty after rebuild. Possible causes:\n"
            "  - No Q4 2025 infotables could be fetched (prior quarter missing)\n"
            "  - No institutions met the exit threshold (>= %.0f%% reduction)\n"
            "  - AUM filter ($%.0fB) too strict\n"
            "Check the log above for warnings from PositionDeltaCalculator / ExitScreener.",
            config.exit_threshold_pct * 100,
            config.min_aum_usd / 1e9,
        )
        return 1

    _logger.info(
        "Backfill complete: %d tickers in Q1 2026 consensus. "
        "The daily P18 pipeline will now produce signals on its next run.",
        len(consensus_df),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
