"""
P22 — interim universe snapshot reader (M1 placeholder).

M2 (entity resolution) is what turns landed SEC DERA data into a real,
eligibility-filtered `p22_company` table (spec §2.0.2-2.0.3). Until that
exists, M1's per-company ingest jobs (SEC submissions/facts, CT.gov,
openFDA) need *some* CIK/name list to iterate — this module reads the most
recently landed raw-zone DERA partition directly and dedups by CIK, as a
bounded, explicitly-provisional stand-in.

**This is not the spec's universe construction.** No eligibility filters
(§2.0.3) are applied, no historical ticker resolution happens, and nothing
here writes to `p22_company`. Once M2 lands, every job that calls
`latest_universe_rows()` should switch to querying `p22_company` via
`P22Repo` instead, and this module can be retired.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.config import RAW_ZONE_ROOT
from src.ml.pipeline.p22_biotech_ma.ingest import raw_zone
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_SOURCE = "sec_dera_universe"


def latest_universe_rows(root: Path | None = None) -> List[Dict[str, Any]]:
    """
    Read every DERA row landed on the most recent date this source was
    fetched, deduped by CIK (keeping the row with the latest `filed` date
    per CIK, since a CIK can appear in multiple quarters that day if a
    backfill spans several quarters at once).

    Returns:
        List of raw DERA `sub.txt` row dicts, one per distinct CIK, possibly
        empty if nothing has been landed yet.
    """
    zone_root = root if root is not None else RAW_ZONE_ROOT
    source_dir = zone_root / _SOURCE
    if not source_dir.is_dir():
        _logger.warning("No SEC DERA universe data landed yet at %s", source_dir)
        return []

    date_dirs = sorted((d for d in source_dir.iterdir() if d.is_dir()), reverse=True)
    if not date_dirs:
        return []
    latest_dir = date_dirs[0]

    by_cik: Dict[str, Dict[str, Any]] = {}
    for payload_file in latest_dir.glob("*.json.gz"):
        rows = raw_zone.read(payload_file)
        if not isinstance(rows, list):
            continue
        for row in rows:
            cik = row.get("cik")
            if not cik:
                continue
            existing = by_cik.get(cik)
            if existing is None or row.get("filed", "") > existing.get("filed", ""):
                by_cik[cik] = row

    _logger.info("Loaded %d distinct CIKs from latest DERA snapshot (%s)", len(by_cik), latest_dir.name)
    return list(by_cik.values())


def all_landed_quarters(root: Path | None = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Read every landed DERA quarter across *all* ingest-date partitions, keyed
    by quarter string (e.g. "2019q3"), for point-in-time backtest use (spec
    §2.0.1/§2.0.3: "walk every quarter... the per-quarter set is the eligible
    universe for that `as_of`").

    Unlike `latest_universe_rows`, which only reads the single most recent
    ingest-date directory (fine for "what's the universe right now"), this
    walks every date partition under the source, since `land_all_quarters`
    lands 15+ years of quarters in one run dated by *ingest* day, not by the
    quarter itself — restricting to the latest ingest date would silently
    drop every quarter except whatever was landed on the most recent run.

    If the same quarter was landed more than once (a re-run on a later date),
    the most recently ingested copy wins — DERA data for a closed quarter is
    immutable, so this only matters for picking up a corrected/backfilled
    re-ingest.

    Returns:
        `{quarter: [DERA sub.txt row dicts]}`, possibly empty if nothing has
        been landed yet.
    """
    zone_root = root if root is not None else RAW_ZONE_ROOT
    source_dir = zone_root / _SOURCE
    if not source_dir.is_dir():
        _logger.warning("No SEC DERA universe data landed yet at %s", source_dir)
        return {}

    by_quarter: Dict[str, List[Dict[str, Any]]] = {}
    for date_dir in sorted((d for d in source_dir.iterdir() if d.is_dir())):
        for manifest_path in date_dir.glob("*.manifest.json"):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            quarter = manifest.get("entity")
            if not quarter:
                continue
            payload_path = manifest_path.with_suffix("").with_suffix(".json.gz")
            if not payload_path.exists():
                continue
            rows = raw_zone.read(payload_path)
            if isinstance(rows, list):
                by_quarter[quarter] = rows  # later date_dir (ascending order) overwrites earlier

    _logger.info("Loaded %d distinct DERA quarters across all ingest dates", len(by_quarter))
    return by_quarter
