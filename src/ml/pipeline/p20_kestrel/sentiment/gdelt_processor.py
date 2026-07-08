"""
P20 Kestrel — GDELT processor.

Reads the slim per-article orgs-slice files (``YYYYMMDD.gkg-orgs.csv.gz``)
from the shared GDELT cache, downloaded early each morning by
``p20_gdelt_download``. Matches organization mentions to k20_company_aliases,
aggregates per (ticker, date), computes rolling z-scores, and upserts into
k20_sentiment_daily.
"""

from __future__ import annotations

import gzip
import re
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from difflib import SequenceMatcher

import pandas as pd

from src.data.db.services.kestrel_service import KestrelService as _KestrelService
from src.ml.pipeline.p20_kestrel.config import DATA_CACHE_PATH

_kestrel = _KestrelService()
finish_job_run = _kestrel.finish_job_run
get_all_aliases = _kestrel.get_all_aliases
get_blocklist = _kestrel.get_blocklist
get_sentiment_history = _kestrel.get_sentiment_history
get_watchlist_tickers = _kestrel.get_watchlist_tickers
start_job_run = _kestrel.start_job_run
upsert_sentiment = _kestrel.upsert_sentiment
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_JOB_NAME = "gdelt_process"
_P15_GKG_DIR = DATA_CACHE_PATH / "gdelt" / "gkg"
_MIN_PERIODS = 15  # warm-up rule: z-score requires at least 15 days of history
_FUZZY_THRESHOLD = 0.93

# Slim orgs-slice format written by GdeltDownloader (YYYYMMDD.gkg-orgs.csv.gz):
# tab-separated with header row — date, source, themes, orgs, tone.
_ORGS_HEADER = ["date", "source", "themes", "orgs", "tone"]

_FINANCE_THEMES = re.compile(r"\b(ECON_|MARKET|INVEST|STOCK|FINANCE)\b")


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _fuzzy_score(a: str, b: str) -> float:
    """Quick ratio-based fuzzy similarity."""
    return SequenceMatcher(None, a, b).ratio()


def _parse_v2tone(tone_str: str) -> Tuple[float, float, float] | None:
    """
    Parse GDELT V2Tone CSV field.

    Args:
        tone_str: Comma-separated V2Tone string.

    Returns:
        Tuple of (avg_tone, pos_score, neg_score) or None on parse failure.
    """
    try:
        parts = tone_str.split(",")
        if len(parts) < 3:
            return None
        return float(parts[0]), float(parts[1]), abs(float(parts[2]))
    except (ValueError, IndexError):
        return None


class GdeltProcessor:
    """Processes one day's worth of GKG files against the alias table."""

    def __init__(self) -> None:
        self._aliases: Dict[str, List[Dict[str, Any]]] = {}  # normalized_alias → [alias_rows]
        self._blocklist: Dict[str, str] = {}  # alias → match_policy

    def load_alias_table(self) -> None:
        """Load aliases and blocklist from DB into memory."""
        raw_aliases = get_all_aliases()
        self._aliases = {}
        for row in raw_aliases:
            key = str(row.get("normalized_alias") or _normalize(row.get("alias", "")))
            self._aliases.setdefault(key, []).append(row)

        blocklist_rows = get_blocklist()
        self._blocklist = {str(r["alias"]): str(r["match_policy"]) for r in blocklist_rows}
        _logger.info(
            "Alias table loaded: %d normalized keys, %d blocklist entries",
            len(self._aliases),
            len(self._blocklist),
        )

    def _match_org(self, org: str, v2themes: str) -> str | None:
        """
        Attempt to match a GDELT organization mention to a ticker.

        Args:
            org: Raw organization name from V2Organizations.
            v2themes: V2Themes field for context.

        Returns:
            Matched ticker, or None.
        """
        norm = _normalize(org)
        if not norm:
            return None

        # Exact match first
        hits = self._aliases.get(norm)
        if hits:
            ticker = str(hits[0]["ticker"])
            policy = self._blocklist.get(norm)
            if policy == "never":
                return None
            if policy == "name_plus_context":
                if not _FINANCE_THEMES.search(v2themes):
                    return None
            return ticker

        # Fuzzy second pass
        best_score = 0.0
        best_ticker: str | None = None
        for key, alias_rows in self._aliases.items():
            if abs(len(key) - len(norm)) > 5:
                continue
            score = _fuzzy_score(norm, key)
            if score >= _FUZZY_THRESHOLD and score > best_score:
                best_score = score
                best_ticker = str(alias_rows[0]["ticker"])

        return best_ticker

    def process_gkg_file(self, gkg_path: Path) -> List[Dict[str, Any]]:
        """
        Process one GKG orgs-slice file and return per-article matched records.

        Expects the slim format written by GdeltDownloader
        (``YYYYMMDD.gkg-orgs.csv.gz``): tab-separated with a header row —
        date, source, themes, orgs, tone.

        Args:
            gkg_path: Path to a .gkg-orgs.csv or .gkg-orgs.csv.gz file.

        Returns:
            List of dicts with: ticker, date, avg_tone, pos_score, neg_score, source_domain.
        """
        records: List[Dict[str, Any]] = []
        opener = gzip.open if gkg_path.suffix == ".gz" else open
        try:
            with opener(gkg_path, "rt", encoding="utf-8", errors="replace") as f:  # type: ignore[call-overload]
                header = f.readline().rstrip("\n").split("\t")
                if header != _ORGS_HEADER:
                    _logger.warning("Unexpected orgs-slice header in %s: %s", gkg_path, header)
                    return records
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < len(_ORGS_HEADER):
                        continue

                    date_str, source, v2themes, orgs_raw, tone_raw = parts[:5]
                    try:
                        row_date = date.fromisoformat(date_str)
                    except ValueError:
                        continue

                    tone_parsed = _parse_v2tone(tone_raw)
                    if tone_parsed is None:
                        continue

                    avg_tone, pos_score, neg_score = tone_parsed
                    seen_tickers: set[str] = set()

                    for org in orgs_raw.split(";"):
                        org = org.split(",")[0].strip()  # strip count suffix
                        ticker = self._match_org(org, v2themes)
                        if ticker and ticker not in seen_tickers:
                            seen_tickers.add(ticker)
                            records.append(
                                {
                                    "ticker": ticker,
                                    "date": row_date,
                                    "avg_tone": avg_tone,
                                    "pos_score": pos_score,
                                    "neg_score": neg_score,
                                    "source_domain": source.split(",")[0].strip(),
                                }
                            )
        except Exception:
            _logger.exception("Error processing GKG file %s", gkg_path)
        return records


def _aggregate_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Aggregate per-article records into per-(ticker, date) sentiment rows.

    Args:
        records: Raw matched records from GKG files.

    Returns:
        Aggregated sentiment rows ready for k20_sentiment_daily upsert.
    """
    if not records:
        return []

    df = pd.DataFrame(records)
    out: List[Dict[str, Any]] = []

    for key, grp in df.groupby(["ticker", "date"]):
        ticker, row_date = key  # type: ignore[misc]
        top_domains = grp["source_domain"].value_counts().head(5).to_dict()
        out.append(
            {
                "ticker": str(ticker),
                "date": row_date,
                "source": "gdelt",
                "mentions": len(grp),
                "avg_tone": grp["avg_tone"].mean(),
                "tone_std": grp["avg_tone"].std() if len(grp) > 1 else 0.0,
                "pos_score": grp["pos_score"].mean(),
                "neg_score": grp["neg_score"].mean(),
                "top_domains": top_domains,
            }
        )

    return out


def _compute_zscores(
    agg_rows: List[Dict[str, Any]],
    as_of_date: date,
) -> List[Dict[str, Any]]:
    """
    Add rolling 20-day z-scores for mentions and avg_tone.

    Fetches 30 days of history per ticker and computes z-score for today's value.
    Stores NULL during warm-up (< MIN_PERIODS days of history).

    Args:
        agg_rows: Aggregated sentiment rows for as_of_date.
        as_of_date: The date being processed.

    Returns:
        Same rows with mention_z20 and tone_z20 populated.
    """
    lookback_start = as_of_date - timedelta(days=30)
    tickers = list({r["ticker"] for r in agg_rows})

    # Fetch historical rows per ticker
    history: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        hist_rows = get_sentiment_history(ticker, "gdelt", start=lookback_start, end=as_of_date)
        if hist_rows:
            history[ticker] = pd.DataFrame(hist_rows)

    for row in agg_rows:
        ticker = row["ticker"]
        hist_df = history.get(ticker)

        if hist_df is None or len(hist_df) < _MIN_PERIODS:
            row["mention_z20"] = None
            row["tone_z20"] = None
            continue

        mention_mean = hist_df["mentions"].mean()
        mention_std = hist_df["mentions"].std()
        tone_mean = hist_df["avg_tone"].mean()
        tone_std_hist = hist_df["avg_tone"].std()

        row["mention_z20"] = float((row["mentions"] - mention_mean) / mention_std) if mention_std > 0 else None
        row["tone_z20"] = float((row["avg_tone"] - tone_mean) / tone_std_hist) if tone_std_hist > 0 else None

    return agg_rows


def run(as_of_date: date | None = None) -> Dict[str, Any]:
    """
    Process all available GKG files for as_of_date and upsert sentiment.

    Args:
        as_of_date: Date to process. Defaults to yesterday.

    Returns:
        Summary dict with files_processed, articles_matched, rows_upserted.
    """
    target_date = as_of_date or (date.today() - timedelta(days=1))
    _logger.info("GDELT processor for %s", target_date)
    start_job_run(_JOB_NAME, target_date)

    try:
        processor = GdeltProcessor()
        processor.load_alias_table()

        date_prefix = target_date.strftime("%Y%m%d")
        gkg_files = sorted(_P15_GKG_DIR.glob(f"{date_prefix}*.gkg-orgs.csv*"))
        _logger.info("Found %d GKG orgs-slice files for %s", len(gkg_files), target_date)

        all_records: List[Dict[str, Any]] = []
        files_processed = 0
        for gkg_file in gkg_files:
            records = processor.process_gkg_file(gkg_file)
            all_records.extend(records)
            files_processed += 1

        articles_matched = len(all_records)
        agg_rows = _aggregate_records(all_records)
        scored_rows = _compute_zscores(agg_rows, target_date)
        rows_upserted = upsert_sentiment(scored_rows)

        _logger.info(
            "GDELT done: %d files, %d matches, %d tickers upserted",
            files_processed,
            articles_matched,
            rows_upserted,
        )
        finish_job_run(_JOB_NAME, target_date, status="ok", rows_out=rows_upserted)
        return {
            "files_processed": files_processed,
            "articles_matched": articles_matched,
            "rows_upserted": rows_upserted,
        }

    except Exception as exc:
        _logger.exception("GDELT processor failed")
        finish_job_run(_JOB_NAME, target_date, status="failed", error=str(exc))
        raise


def run_backfill(
    start_date: date,
    end_date: date | None = None,
) -> Dict[str, Any]:
    """
    Process GKG files for a date range (backfill mode).

    Calls run() for each weekday in [start_date, end_date] inclusive,
    logging per-date results. Continues on per-date errors.

    Args:
        start_date: First date to process (inclusive).
        end_date: Last date to process (inclusive). Defaults to yesterday.

    Returns:
        Aggregate summary with per_date breakdown.
    """
    end = end_date or (date.today() - timedelta(days=1))
    _logger.info("GDELT backfill %s → %s", start_date, end)

    per_date: List[Dict[str, Any]] = []
    dates_processed = 0
    dates_skipped = 0
    current = start_date

    while current <= end:
        if current.weekday() >= 5:  # skip weekends
            current += timedelta(days=1)
            continue
        try:
            result = run(current)
            per_date.append({"date": str(current), **result})
            dates_processed += 1
        except Exception:
            _logger.exception("Backfill failed for %s; continuing", current)
            per_date.append({"date": str(current), "error": True})
            dates_skipped += 1
        current += timedelta(days=1)

    total_files = sum(r.get("files_processed", 0) for r in per_date)
    total_rows = sum(r.get("rows_upserted", 0) for r in per_date)
    _logger.info(
        "GDELT backfill done: %d processed, %d skipped, %d files, %d rows",
        dates_processed,
        dates_skipped,
        total_files,
        total_rows,
    )
    return {
        "dates_processed": dates_processed,
        "dates_skipped": dates_skipped,
        "total_files": total_files,
        "total_rows_upserted": total_rows,
        "per_date": per_date,
    }
