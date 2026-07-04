#!/usr/bin/env python3
"""
Audit cached fundamentals for missing market cap / average volume.

Scans ``{cache_dir}/fundamentals/*/`` and reads the latest JSON per symbol
(ignores TTL) to classify rows without hitting the network.

Example:
    python tests/tmp_audit_fundamentals_cache.py --limit 2000
    python tests/tmp_audit_fundamentals_cache.py --cache-dir /data-cache --csv-out /tmp/fund_audit.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cache.fundamentals_cache import get_fundamentals_cache  # noqa: E402
from src.data.cache.fundamentals_combiner import get_fundamentals_combiner  # noqa: E402

try:
    from config.donotshare.donotshare import DATA_CACHE_DIR  # noqa: E402
except ImportError:
    DATA_CACHE_DIR = "data-cache"


def _liquidity_snapshot(data: Dict[str, Any]) -> Tuple[Any | None, Any | None]:
    """Match FundamentalFilter-style resolution for cap and avg volume."""
    profile = data.get("profile", {}) or {}
    if not isinstance(profile, dict):
        profile = {}

    market_cap = data.get("market_cap")
    if not market_cap:
        market_cap = profile.get("marketCap")

    avg_vol = data.get("avg_volume")
    if avg_vol is None or avg_vol == 0:
        avg_vol = data.get("average_volume")
    if avg_vol is None or avg_vol == 0:
        avg_vol = (
            profile.get("averageVolume")
            or profile.get("averageVolume10days")
            or profile.get("avgVolume")
            or profile.get("volAvg")
            or profile.get("volume")
        )
    return market_cap, avg_vol


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit fundamentals JSON cache on disk.")
    parser.add_argument("--cache-dir", type=str, default=DATA_CACHE_DIR, help="Base cache directory")
    parser.add_argument("--limit", type=int, default=0, help="Max symbols to scan (0 = all)")
    parser.add_argument("--csv-out", type=str, default=None, help="Optional CSV path for per-symbol rows")
    args = parser.parse_args()

    combiner = get_fundamentals_combiner()
    cache = get_fundamentals_cache(args.cache_dir, combiner)

    symbols_dir = Path(args.cache_dir) / "fundamentals"
    if not symbols_dir.is_dir():
        print("No fundamentals directory: %s" % symbols_dir)
        return 1

    dirs = sorted(p for p in symbols_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
    if args.limit > 0:
        dirs = dirs[: args.limit]

    stats = {
        "symbols": 0,
        "no_json": 0,
        "cap_ok_vol_ok": 0,
        "cap_ok_vol_missing": 0,
        "cap_missing_vol_ok": 0,
        "both_missing": 0,
    }
    rows: List[Dict[str, Any]] = []

    for d in dirs:
        stats["symbols"] += 1
        sym = d.name.upper()
        meta = cache.find_latest_json(sym, data_type="general", max_age_days=36500)
        if not meta:
            stats["no_json"] += 1
            rows.append({"symbol": sym, "bucket": "no_json"})
            continue
        payload = cache.read_json(meta.file_path)
        if not payload:
            stats["no_json"] += 1
            rows.append({"symbol": sym, "bucket": "read_fail"})
            continue

        cap, vol = _liquidity_snapshot(payload)
        cap_ok = cap is not None and cap != 0
        vol_ok = vol is not None and vol != 0

        if cap_ok and vol_ok:
            stats["cap_ok_vol_ok"] += 1
            bucket = "ok"
        elif cap_ok and not vol_ok:
            stats["cap_ok_vol_missing"] += 1
            bucket = "cap_ok_vol_missing"
        elif not cap_ok and vol_ok:
            stats["cap_missing_vol_ok"] += 1
            bucket = "cap_missing_vol_ok"
        else:
            stats["both_missing"] += 1
            bucket = "both_missing"

        rows.append(
            {
                "symbol": sym,
                "bucket": bucket,
                "provider": meta.provider,
                "file": meta.file_path,
                "market_cap": cap,
                "avg_volume": vol,
            }
        )

    print(json.dumps(stats, indent=2))
    if args.csv_out:
        out_path = Path(args.csv_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["symbol", "bucket"])
            w.writeheader()
            w.writerows(rows)
        print("Wrote %s" % out_path.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
