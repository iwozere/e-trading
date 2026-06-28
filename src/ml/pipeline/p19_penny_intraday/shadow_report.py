"""
P19 shadow-data QA / health report.

Read-only summary of the shadow store (``results/p19_penny_intraday/shadow.sqlite``)
so the multi-week collection phase is observable — and so the same tool seeds the
Phase-4 threshold calibration. Reports per-day row counts and coverage, RVOL and
%-from-open distributions, EOD-fill rate, and **sanity flags** (notably the volume
lot-size check).

Usage:
    python -m src.ml.pipeline.p19_penny_intraday.shadow_report            # latest day
    python -m src.ml.pipeline.p19_penny_intraday.shadow_report --date 2026-06-29
    python -m src.ml.pipeline.p19_penny_intraday.shadow_report --all      # every day
"""

import argparse
import os
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.ml.pipeline.p19_penny_intraday.shadow_store import DEFAULT_DB_PATH

_logger = setup_logger(__name__)


def _percentiles(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"n": 0, "min": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    s = sorted(vals)
    n = len(s)
    def at(p: float) -> float:
        return s[min(n - 1, int(p * n))]
    return {"n": n, "min": round(s[0], 3), "median": round(at(0.5), 3),
            "p90": round(at(0.9), 3), "max": round(s[-1], 3)}


def report(db_path: str = DEFAULT_DB_PATH, date: Optional[str] = None) -> Dict[str, Any]:
    """Build a stats dict for one trading date (default: the latest present)."""
    if not os.path.exists(db_path):
        return {"db": db_path, "error": "no shadow store yet"}
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        dates = [r[0] for r in conn.execute("SELECT DISTINCT date FROM shadow_log ORDER BY date")]
        total = conn.execute("SELECT COUNT(*) FROM shadow_log").fetchone()[0]
        target = date or (dates[-1] if dates else None)
        out: Dict[str, Any] = {"db": db_path, "total_rows": total,
                               "days_collected": len(dates), "date": target}
        if not target:
            return out

        rows = conn.execute(
            "SELECT ticker, source, pct_from_open, rvol_so_far, day_volume, "
            "avg_volume_30d, eod_close FROM shadow_log WHERE date = ?", (target,)
        ).fetchall()
        out["rows"] = len(rows)
        if not rows:
            return out

        tickers = {r[0] for r in rows}
        out["distinct_tickers"] = len(tickers)
        out["polls_est"] = round(len(rows) / max(1, len(tickers)), 1)
        out["by_source"] = dict(Counter(r[1] for r in rows))
        out["rvol_so_far"] = _percentiles([r[3] for r in rows if r[3] and r[3] > 0])
        out["pct_from_open"] = _percentiles([abs(r[2]) for r in rows if r[2] is not None])
        out["eod_fill_rate"] = round(sum(1 for r in rows if r[6] is not None) / len(rows), 3)

        # Volume lot-size sanity: day_volume / avg_volume_30d should sit in a sane
        # intraday band (~0.1–5×). ~100× or ~0.01× ⇒ a lot-vs-shares unit mismatch.
        ratios = [r[4] / r[5] for r in rows if r[4] and r[5] and r[5] > 0]
        out["vol_ratio_median"] = round(sorted(ratios)[len(ratios) // 2], 4) if ratios else None
        out["gappers_zero_rvol"] = sum(1 for r in rows if r[1] == "gapper" and (not r[3]))

        out["flags"] = _flags(out)
        return out
    finally:
        conn.close()


def _flags(s: Dict[str, Any]) -> List[str]:
    flags: List[str] = []
    vr = s.get("vol_ratio_median")
    if vr is not None:
        if vr > 20:
            flags.append(f"day_volume looks ~100x HIGH (vol_ratio_median={vr}) — lower ibkr_volume_lot_size?")
        elif 0 < vr < 0.02:
            flags.append(f"day_volume looks ~100x LOW (vol_ratio_median={vr}) — raise ibkr_volume_lot_size?")
    if s.get("eod_fill_rate", 1.0) < 0.8 and s.get("rows", 0) > 0:
        flags.append(f"low EOD fill rate ({s['eod_fill_rate']}) — is eod-backfill running?")
    if s.get("gappers_zero_rvol", 0) > 0:
        flags.append(f"{s['gappers_zero_rvol']} gapper rows with RVOL=0 — baseline enrichment gap")
    if s.get("rvol_so_far", {}).get("n", 0) == 0 and s.get("rows", 0) > 0:
        flags.append("no positive RVOL anywhere — feed/volume issue or all baselines missing")
    return flags


def format_report(s: Dict[str, Any]) -> str:
    if s.get("error"):
        return f"P19 shadow report: {s['error']} ({s['db']})"
    lines = [
        f"=== P19 shadow report — {s.get('date')} ===",
        f"store: {s['db']}",
        f"total rows: {s['total_rows']}  over {s['days_collected']} day(s)",
    ]
    if not s.get("rows"):
        lines.append("(no rows for this date)")
        return "\n".join(lines)
    rv, pm = s["rvol_so_far"], s["pct_from_open"]
    lines += [
        f"rows: {s['rows']}  tickers: {s['distinct_tickers']}  ~polls: {s['polls_est']}  sources: {s['by_source']}",
        f"RVOL-so-far (n={rv['n']}): median {rv['median']}  p90 {rv['p90']}  max {rv['max']}",
        f"|%-from-open| (n={pm['n']}): median {pm['median']}  p90 {pm['p90']}  max {pm['max']}",
        f"EOD fill rate: {s['eod_fill_rate']}   vol_ratio_median: {s['vol_ratio_median']}",
    ]
    if s["flags"]:
        lines.append("FLAGS:")
        lines += [f"  ⚠ {f}" for f in s["flags"]]
    else:
        lines.append("flags: none — collection looks healthy")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="P19 shadow-data QA report")
    ap.add_argument("--db", default=DEFAULT_DB_PATH)
    ap.add_argument("--date", default=None, help="Trading date YYYY-MM-DD (default: latest)")
    ap.add_argument("--all", action="store_true", help="Report every collected day")
    args = ap.parse_args()

    if args.all:
        if not os.path.exists(args.db):
            print(f"P19 shadow report: no shadow store yet ({args.db})")
            return 0
        conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
        dates = [r[0] for r in conn.execute("SELECT DISTINCT date FROM shadow_log ORDER BY date")]
        conn.close()
        for d in dates:
            print(format_report(report(args.db, d)))
            print()
        return 0

    print(format_report(report(args.db, args.date)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
