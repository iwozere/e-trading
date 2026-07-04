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

from __future__ import annotations

import argparse
import gzip
import io
import re
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import requests

from src.data.downloader.edgar_downloader import EdgarDownloader
from src.ml.pipeline.p18_institutional_flow_tracker.config import P18Config
from src.ml.pipeline.p18_institutional_flow_tracker.pipeline import InstitutionalFlowPipeline
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


@dataclass(frozen=True)
class QuarterSpec:
    label: str
    year: int  # reporting quarter year
    quarter: int  # reporting quarter 1-4
    edgar_year: int  # EDGAR bulk index year (filing calendar year)
    edgar_qtr: int  # EDGAR bulk index quarter (1-4, by filing date)
    window_start: str  # earliest 13F filing date for this quarter (YYYY-MM-DD)
    window_end: str  # latest 13F filing date for this quarter (YYYY-MM-DD)


# Q4 2025: 45-day window ends 2026-02-14  → filed in EDGAR QTR1 2026
# Q1 2026: 45-day window ends 2026-05-15  → filed in EDGAR QTR2 2026
_QUARTERS: list[QuarterSpec] = [
    QuarterSpec("Q4 2025", 2025, 4, 2026, 1, "2026-01-01", "2026-02-14"),
    QuarterSpec("Q1 2026", 2026, 1, 2026, 2, "2026-04-01", "2026-05-15"),
]

# form.gz line: fixed-width columns.
# Form Type  Company Name (variable)   CIK (right-justified)   Date Filed   File Name
# Filename format: edgar/data/{CIK}/{accno}.txt
# Accession number IS the filename stem, e.g. 0001356371-26-000004
_LINE_RE = re.compile(
    r"^13F-HR\s+(.+?)\s{2,}(\d{1,10})\s+(\d{4}-\d{2}-\d{2})\s+"
    r"edgar/data/\d+/(\d{10}-\d{2}-\d{6})\.txt",
    re.MULTILINE,
)


def _download_bulk_index(edgar_year: int, edgar_qtr: int, session: requests.Session) -> str:
    """Download and decompress an EDGAR quarterly form.gz index; return raw text."""
    url = f"https://www.sec.gov/Archives/edgar/full-index/{edgar_year}/QTR{edgar_qtr}/form.gz"
    _logger.info("Downloading EDGAR bulk index from %s", url)
    resp = session.get(url, timeout=120)
    resp.raise_for_status()
    with gzip.open(io.BytesIO(resp.content), "rt", encoding="latin-1") as fh:
        return fh.read()


def _parse_13f_records(raw_text: str, window_start: str, window_end: str) -> pd.DataFrame:
    """Extract 13F-HR records within the filing window from a raw form.gz text."""
    records = []
    for m in _LINE_RE.finditer(raw_text):
        records.append(
            {
                "cik": str(int(m.group(2))),  # strip leading zeros
                "institution_name": m.group(1).strip(),
                "accession_number": m.group(4),
                "filed_date": m.group(3),
            }
        )

    if not records:
        return pd.DataFrame(
            {
                "cik": pd.Series([], dtype=str),
                "institution_name": pd.Series([], dtype=str),
                "accession_number": pd.Series([], dtype=str),
                "filed_date": pd.Series([], dtype=str),
            }
        )

    df = pd.DataFrame(records)
    dates: pd.Series = pd.to_datetime(df["filed_date"])
    mask = (dates >= window_start) & (dates <= window_end)
    result: pd.DataFrame = df.loc[mask].reset_index(drop=True)
    return result


def _seed_index_cache(edgar: EdgarDownloader, index_df: pd.DataFrame, year: int, quarter: int) -> None:
    """Write index_df to the path that download_13f_index uses as its cache."""
    dest = edgar._13f_index_dir / f"{year}_Q{quarter}.csv.gz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    index_df.to_csv(dest, index=False, compression="gzip")
    _logger.info("Seeded index cache: %d filers → %s", len(index_df), dest)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P18 full backfill via EDGAR bulk index.")
    p.add_argument(
        "--skip-q4-index", action="store_true", help="Skip downloading Q4 2025 bulk index (use if cache already exists)"
    )
    p.add_argument(
        "--skip-q1-index", action="store_true", help="Skip downloading Q1 2026 bulk index (use if cache already exists)"
    )
    p.add_argument("--force", action="store_true", help="Force re-download of all infotables even if cached")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    config = P18Config.create_default()
    edgar = EdgarDownloader()

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "research alkotrader@gmail.com",
            "Accept-Encoding": "gzip, deflate",
        }
    )

    skip_by_label = {"Q4 2025": args.skip_q4_index, "Q1 2026": args.skip_q1_index}

    # ------------------------------------------------------------------ #
    # Step 1: Seed both index caches from EDGAR bulk files.
    # ------------------------------------------------------------------ #
    for spec in _QUARTERS:
        dest = edgar._13f_index_dir / f"{spec.year}_Q{spec.quarter}.csv.gz"

        if skip_by_label.get(spec.label) and dest.exists():
            _logger.info("%s index already cached — skipping download", spec.label)
            continue

        raw = _download_bulk_index(spec.edgar_year, spec.edgar_qtr, session)
        index_df = _parse_13f_records(raw, spec.window_start, spec.window_end)

        if index_df.empty:
            _logger.error("%s: no 13F-HR records found in bulk index — aborting", spec.label)
            return 1

        _logger.info(
            "%s: %d 13F-HR filers in window %s → %s", spec.label, len(index_df), spec.window_start, spec.window_end
        )
        _seed_index_cache(edgar, index_df, spec.year, spec.quarter)

    # ------------------------------------------------------------------ #
    # Step 2: Build Q1 2026 consensus.
    # rebuild_quarterly_consensus uses the cached indices seeded above.
    # ------------------------------------------------------------------ #
    _logger.info("Starting rebuild_quarterly_consensus for 2026 Q1 ...")
    pipeline = InstitutionalFlowPipeline(config)
    consensus_df = pipeline.rebuild_quarterly_consensus(2026, 1, force_download=args.force)

    if consensus_df.empty:
        _logger.error(
            "Consensus is empty after rebuild. Possible causes:\n"
            "  - No Q4 2025 infotables found (prior quarter not downloaded yet)\n"
            "  - No institutions met the exit threshold (>= %.0f%% reduction)\n"
            "  - AUM filter ($%.0fB) excluded all filers\n"
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
