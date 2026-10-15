"""
Drift between the yfinance increments and the next FMP vintage (spec §10.3), and retiring the covered increments.

When a new vintage arrives, the increments ingested on top of the **previous** vintage overlap it. Per security
(``security_id`` is stable across dumps, D46) the report compares, over the overlap:

- raw close relative difference (median, p99, max);
- bars present in one source but not the other (only inside the span both sources cover);
- split and dividend events present in only one source, and ratio/amount differences on shared dates.

``fmp build`` writes ``reports/<new dump>/drift_report.md``/``.json`` after switching to a new vintage.
``fmp archive-increments`` (writer host only, §10.4) then **moves** every increment bar and event dated on or
before the new vintage's last bar of that security to ``increments/archive/<old vintage>/`` (events: ``archive/before-<new vintage>/``) — never deletes —
so the new vintage becomes the base and the increments only hold what comes after it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.fmp_pipeline.datamanager_adapter.fmp_store import FmpStore, SecurityInfo
from src.data.fmp_pipeline.fsutil import atomic_write_bytes
from src.data.fmp_pipeline.increments.store import IncrementStore, utc_now
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

ARCHIVE_DIR = "archive"
RATIO_TOLERANCE = 0.01
AMOUNT_TOLERANCE = 0.005


@dataclass
class SecurityDrift:
    """Drift of one security's increments against the new vintage."""

    security_id: int
    symbol: str
    old_vintage: str
    overlap_bars: int = 0
    median_rel_diff: float = 0.0
    p99_rel_diff: float = 0.0
    max_rel_diff: float = 0.0
    missing_in_vintage: int = 0
    missing_in_increments: int = 0
    event_mismatches: list[str] = field(default_factory=list)


@dataclass
class DriftReport:
    """The whole report."""

    new_vintage: str
    securities: list[SecurityDrift] = field(default_factory=list)

    def to_json(self) -> str:
        """``drift_report.json``."""
        return json.dumps({"new_vintage": self.new_vintage, "securities": [asdict(s) for s in self.securities]},
                          indent=1)

    def to_markdown(self) -> str:
        """``drift_report.md``: summary, then the securities that drifted most."""
        rows = [s for s in self.securities if s.overlap_bars]
        lines = [f"# Drift report — increments vs vintage {self.new_vintage}", "",
                 f"{len(self.securities)} securities with increments from an earlier vintage; {len(rows)} overlap "
                 f"the new vintage ({sum(s.overlap_bars for s in rows):,} bars).", ""]
        if rows:
            medians = np.array([s.median_rel_diff for s in rows])
            lines += [f"Median of per-security median close differences: {np.median(medians):.4%}; worst "
                      f"{medians.max():.4%}.", "",
                      "| Symbol | From vintage | Overlap bars | Median | p99 | Max | Missing in vintage | "
                      "Missing in increments | Event mismatches |", "|---|---|---|---|---|---|---|---|---|"]
            for s in sorted(rows, key=lambda s: -s.max_rel_diff)[:100]:
                lines.append(f"| {s.symbol} | {s.old_vintage} | {s.overlap_bars} | {s.median_rel_diff:.4%} | "
                             f"{s.p99_rel_diff:.4%} | {s.max_rel_diff:.4%} | {s.missing_in_vintage} | "
                             f"{s.missing_in_increments} | {'; '.join(s.event_mismatches) or ''} |")
        return "\n".join(lines) + "\n"

    def write(self, report_dir: Path) -> tuple[Path, Path]:
        """Write both files; return their paths."""
        md, js = report_dir / "drift_report.md", report_dir / "drift_report.json"
        atomic_write_bytes(md, self.to_markdown().encode("utf-8"))
        atomic_write_bytes(js, self.to_json().encode("utf-8"))
        return md, js


def _info(vintage: FmpStore, security_id: int) -> Optional[SecurityInfo]:
    try:
        return vintage.security(security_id)
    except KeyError:
        return None


def _increment_ids(store: IncrementStore, kind: str) -> list[int]:
    folder = store.root / kind
    return sorted(int(p.stem) for p in folder.glob("*/*.parquet")) if folder.is_dir() else []


def _values(table: pa.Table, column: str) -> np.ndarray:
    return np.asarray(table.column(column).to_pylist(), dtype=object)


def _old_rows(table: pa.Table, new_vintage: str) -> pa.Table:
    """Rows ingested on any vintage other than ``new_vintage``."""
    return table.filter(pa.array(_values(table, "vintage") != new_vintage))


def _on_or_before(table: pa.Table, column: str, last: date) -> np.ndarray:
    """Boolean mask: rows whose ``column`` date is on or before ``last``."""
    return np.array([value is not None and value <= last for value in table.column(column).to_pylist()], dtype=bool)


def _event_mismatches(inc_events: list[dict[str, Any]], vintage: FmpStore, sid: int, lo: date, hi: date) -> list[str]:
    splits, dividends = vintage.events(sid)
    fmp = {("split", pd.Timestamp(d).date()): float(n) / float(den)
           for d, n, den in zip(splits["date"], splits["numerator"], splits["denominator"])}
    fmp.update({("dividend", pd.Timestamp(d).date()): float(a) for d, a in zip(dividends["date"], dividends["amount"])})
    ours: dict[tuple[str, date], float] = {}
    for row in inc_events:
        value = row["numerator"] / row["denominator"] if row["type"] == "split" else row["amount"]
        ours[(str(row["type"]), row["date"])] = float(value)
    out = []
    for key in sorted(set(fmp) | set(ours)):
        kind, when = key
        if not lo <= when <= hi:
            continue
        if key not in fmp:
            out.append(f"{kind} {when} only in increments")
        elif key not in ours:
            out.append(f"{kind} {when} only in vintage")
        else:
            tolerance = RATIO_TOLERANCE if kind == "split" else AMOUNT_TOLERANCE
            if abs(fmp[key] - ours[key]) > tolerance * max(abs(fmp[key]), 1e-9):
                out.append(f"{kind} {when}: vintage {fmp[key]:g}, increments {ours[key]:g}")
    return out


def drift_report(data_root: Path, vintage: Optional[FmpStore] = None) -> DriftReport:
    """
    Compare the daily increments ingested on earlier vintages with the active (new) vintage.

    Args:
        data_root: Pipeline data root.
        vintage: The new vintage (default: the one ``current.txt`` names).
    """
    vintage = vintage or FmpStore(data_root)
    store = IncrementStore(data_root)
    report = DriftReport(vintage.vintage)
    for sid in _increment_ids(store, "1d"):
        table = store.read_table("1d", sid)
        if table is None:
            continue
        old = _old_rows(table, vintage.vintage)
        if old.num_rows == 0:
            continue
        frame = old.to_pandas()
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame["date"]))
        info = _info(vintage, sid)
        drift = SecurityDrift(sid, info.symbol if info else str(sid), str(frame["vintage"].iloc[0]))
        report.securities.append(drift)
        prices = vintage.prices(sid, "1d") if info is not None else None
        if prices is None or prices.empty:
            continue
        lo = max(frame.index.min(), prices.index.min())
        hi = min(frame.index.max(), prices.index.max())
        if lo > hi:
            continue
        ours = frame.loc[(frame.index >= lo) & (frame.index <= hi), "close"].astype(float)
        theirs = prices.loc[(prices.index >= lo) & (prices.index <= hi), "close"].astype(float)
        joined = pd.concat([ours.rename("inc"), theirs.rename("fmp")], axis=1, join="inner").dropna()
        drift.missing_in_vintage = int(len(ours.index.difference(theirs.index)))
        drift.missing_in_increments = int(len(theirs.index.difference(ours.index)))
        if not joined.empty:
            diff = (joined["inc"] / joined["fmp"] - 1).abs()
            drift.overlap_bars = len(joined)
            drift.median_rel_diff = round(float(diff.median()), 6)
            drift.p99_rel_diff = round(float(diff.quantile(0.99)), 6)
            drift.max_rel_diff = round(float(diff.max()), 6)
        events = store.read_events(sid)
        if events is not None:
            drift.event_mismatches = _event_mismatches(events.to_pylist(), vintage, sid, lo.date(), hi.date())
    _logger.info("Drift report for %s: %d securities with earlier increments", vintage.vintage,
                 len(report.securities))
    return report


@dataclass
class ArchiveResult:
    """What ``archive_increments`` moved."""

    bars: int = 0
    events: int = 0
    files: int = 0


def _append(path: Path, table: pa.Table, store: IncrementStore) -> None:
    if path.exists():
        table = pa.concat_tables([pq.read_table(path).cast(table.schema), table])
    path.parent.mkdir(parents=True, exist_ok=True)
    store._write(path, table)


def archive_increments(data_root: Path, vintage: Optional[FmpStore] = None) -> ArchiveResult:
    """
    Move increments the new vintage covers to ``increments/archive/<old vintage>/`` (never delete).

    Caller holds the increment writer's lock and has checked the writer host.
    """
    vintage = vintage or FmpStore(data_root)
    store = IncrementStore(data_root)
    result = ArchiveResult()
    for kind, key in (("1d", "date"), ("1h", "date_et")):
        for sid in _increment_ids(store, kind):
            info = _info(vintage, sid)
            if info is None or info.last_price_date is None:
                continue  # the new vintage does not cover this security: its increments stay the extension
            table = store.read_table(kind, sid)
            assert table is not None
            covered = _on_or_before(table, key, info.last_price_date)
            moved = table.filter(pa.array(covered))
            if moved.num_rows == 0:
                continue
            for old_vintage in sorted(set(moved.column("vintage").to_pylist())):
                part = moved.filter(pa.array(_values(moved, "vintage") == old_vintage))
                rel = store.path(kind, sid).relative_to(store.root)
                _append(store.root / ARCHIVE_DIR / str(old_vintage) / rel, part, store)
            kept = table.filter(pa.array(~covered))
            if kept.num_rows:
                store.write_bars(kind, sid, kept)
            else:
                store.path(kind, sid).unlink()
            result.bars += moved.num_rows
            result.files += 1
    for sid in _increment_ids(store, "events"):
        info = _info(vintage, sid)
        events = store.read_events(sid)
        if info is None or info.last_price_date is None or events is None:
            continue
        covered = _on_or_before(events, "date", info.last_price_date)
        moved = events.filter(pa.array(covered))
        if moved.num_rows == 0:
            continue
        rel = store.path("events", sid).relative_to(store.root)
        _append(store.root / ARCHIVE_DIR / f"before-{vintage.vintage}" / rel, moved, store)  # events carry no vintage
        kept = events.filter(pa.array(~covered))
        if kept.num_rows:
            store.write_events(sid, kept)
        else:
            store.path("events", sid).unlink()
        result.events += moved.num_rows
    meta = store.meta()
    meta.update({"increment_seq": int(meta.get("increment_seq", 0)) + (1 if result.bars or result.events else 0),
                 "archived": {"at": utc_now().isoformat(timespec="seconds"), "new_vintage": vintage.vintage,
                              **asdict(result)}})
    store.write_meta(meta)
    _logger.info("Archived %d increment bars and %d events covered by vintage %s", result.bars, result.events,
                 vintage.vintage)
    return result


def has_earlier_increments(data_root: Path, new_vintage: str) -> bool:
    """True when any daily increment row was ingested on a vintage other than ``new_vintage``."""
    store = IncrementStore(data_root)
    for sid in _increment_ids(store, "1d"):
        table = store.read_table("1d", sid)
        if table is not None and _old_rows(table, new_vintage).num_rows:
            return True
    return False
