"""
Checks of the built store before any research uses it.

Two checks, both written to ``<data_root>/reports`` (never to the repository: the licence forbids publishing
evaluations of the data, and the repository is public):

- ``adjustment_check``: our total-return adjustment of the raw bars against Sharadar's own ``closeadj``. Proves the
  raw-price derivation and the dividend re-basing.
- ``vendor_check``: raw closes and splits against the FMP vintage for securities mapped by CIK.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np
import pandas as pd

from src.data.fmp_pipeline.adjust.adjust import adjust
from src.data.sharadar_pipeline import config
from src.data.sharadar_pipeline.store import SharadarStore
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

EARLIEST = date(1990, 1, 1)
LATEST = date(2999, 1, 1)
#: A day agrees when the two values differ by less than this fraction.
TOLERANCE = 0.01


class DailyStore(Protocol):
    """The part of ``FmpStore`` the vendor check reads."""

    def bulk_daily(self, security_ids: Any, start: date, end: date) -> dict[int, pd.DataFrame]:
        """Raw daily bars by security id."""
        ...

    def events_many(self, security_ids: Any) -> dict[int, tuple[pd.DataFrame, pd.DataFrame]]:
        """``(splits, dividends)`` by security id."""
        ...


def adjustment_check(store: SharadarStore, security_ids: Optional[list[int]] = None, batch: int = 500) -> pd.DataFrame:
    """
    Compare our total-return adjusted close with Sharadar's ``closeadj``.

    The two series may differ by a constant scale, so the ratio is normalised by its last value; a correct
    conversion gives a ratio of 1 on every day.

    Args:
        store: Built store.
        security_ids: Securities to check; default all.
        batch: Securities per read.

    Returns:
        One row per security: ``bars``, ``max_abs_dev`` (worst day), ``share_ok`` (days within ``TOLERANCE``),
        ``splits``, ``dividends``.
    """
    ids = security_ids if security_ids is not None else [s.security_id for s in store.securities()]
    rows = []
    for offset in range(0, len(ids), batch):
        part = ids[offset:offset + batch]
        bars = store.bulk_daily(part, EARLIEST, LATEST, vendor_adjusted=True)
        events = store.events_many(bars.keys())
        for sid, frame in bars.items():
            splits, dividends = events.get(sid, (None, None))
            vendor = frame["closeadj_vendor"].to_numpy(dtype=float)
            ours = adjust(frame[["open", "high", "low", "close", "volume"]].astype(float), splits, dividends, mode="total")["close"].to_numpy()
            ok = np.isfinite(vendor) & (vendor > 0) & np.isfinite(ours) & (ours > 0)
            if not ok.any():
                continue
            ratio = ours[ok] / vendor[ok]
            deviation = np.abs(ratio / ratio[-1] - 1.0)
            rows.append({"security_id": sid, "symbol": store.security(sid).symbol, "bars": int(ok.sum()),
                         "max_abs_dev": float(deviation.max()), "share_ok": float((deviation < TOLERANCE).mean()),
                         "splits": 0 if splits is None else len(splits), "dividends": 0 if dividends is None else len(dividends)})
    return pd.DataFrame(rows)


def vendor_check(store: SharadarStore, fmp: DailyStore, limit: Optional[int] = None, batch: int = 300) -> pd.DataFrame:
    """
    Compare raw closes, volumes and splits with the FMP vintage for mapped securities.

    Args:
        store: Built Sharadar store whose securities carry ``fmp_security_id``.
        fmp: Open FMP store.
        limit: Check only the first ``limit`` mapped securities.
        batch: Pairs per read.

    Returns:
        One row per pair: ``common_days``, ``median_close_ratio`` (Sharadar / FMP), ``share_close_ok``,
        ``median_volume_ratio``, ``splits_sharadar``, ``splits_fmp``, ``splits_matched`` (same date and ratio).
    """
    pairs = [(s.security_id, s.fmp_security_id) for s in store.securities() if s.fmp_security_id is not None]
    pairs = pairs[:limit] if limit else pairs
    rows = []
    for offset in range(0, len(pairs), batch):
        part = pairs[offset:offset + batch]
        ours = store.bulk_daily([p[0] for p in part], EARLIEST, LATEST)
        theirs = fmp.bulk_daily([p[1] for p in part], EARLIEST, LATEST)
        our_events = store.events_many([p[0] for p in part])
        their_events = fmp.events_many([p[1] for p in part])
        for sid, fmp_sid in part:
            a, b = ours.get(sid), theirs.get(fmp_sid)
            if a is None or b is None:
                continue
            joined = a[["close", "volume"]].merge(b[["close", "volume"]], left_index=True, right_index=True, suffixes=("", "_fmp")).dropna()
            joined = joined[(joined["close"] > 0) & (joined["close_fmp"] > 0)]
            if joined.empty:
                continue
            ratio = joined["close"] / joined["close_fmp"]
            volume = joined[(joined["volume"] > 0) & (joined["volume_fmp"] > 0)]
            start, end = joined.index[0].date(), joined.index[-1].date()
            mine = _split_keys(our_events.get(sid), start, end)
            other = _split_keys(their_events.get(fmp_sid), start, end)
            rows.append({"security_id": sid, "fmp_security_id": fmp_sid, "symbol": store.security(sid).symbol,
                         "common_days": len(joined), "median_close_ratio": float(ratio.median()),
                         "share_close_ok": float(((ratio - 1.0).abs() < TOLERANCE).mean()),
                         "median_volume_ratio": float((volume["volume"] / volume["volume_fmp"]).median()) if len(volume) else np.nan,
                         "splits_sharadar": len(mine), "splits_fmp": len(other), "splits_matched": len(mine & other)})
    return pd.DataFrame(rows)


def _split_keys(events: Optional[tuple[pd.DataFrame, pd.DataFrame]], start: date, end: date) -> set[tuple[date, float]]:
    if events is None or events[0].empty:
        return set()
    splits = events[0]
    out = set()
    for when, numerator, denominator in zip(pd.to_datetime(splits["date"]).dt.date, splits["numerator"], splits["denominator"]):
        if start < when <= end and denominator:
            out.add((when, round(float(numerator) / float(denominator), 4)))
    return out


def summarize(adjustment: pd.DataFrame, vendor: Optional[pd.DataFrame]) -> dict[str, Any]:
    """Reduce both checks to the numbers the runbook's pass conditions use."""
    adjustment_part: dict[str, Any] = {"securities": len(adjustment)}
    out: dict[str, Any] = {"adjustment": adjustment_part}
    if len(adjustment):
        adjustment_part["share_fully_ok"] = float((adjustment["max_abs_dev"] < TOLERANCE).mean())
        adjustment_part["median_max_abs_dev"] = float(adjustment["max_abs_dev"].median())
        adjustment_part["worst"] = adjustment.nlargest(10, "max_abs_dev")[["symbol", "max_abs_dev"]].to_dict("records")
    if vendor is not None:
        vendor_part: dict[str, Any] = {"pairs": len(vendor)}
        out["vendor"] = vendor_part
        if len(vendor):
            vendor_part["median_close_ratio"] = float(vendor["median_close_ratio"].median())
            vendor_part["share_pairs_close_ok"] = float((vendor["share_close_ok"] >= 0.99).mean())
            vendor_part["median_volume_ratio"] = float(vendor["median_volume_ratio"].median())
            for column in ("splits_sharadar", "splits_fmp", "splits_matched"):
                vendor_part[column] = int(vendor[column].sum())
    return out


def run_crosscheck(store: SharadarStore, fmp: Optional[DailyStore] = None, limit: Optional[int] = None) -> dict[str, Any]:
    """
    Run both checks and write ``adjustment_check.csv``, ``vendor_check.csv`` and ``crosscheck_summary.json`` to
    ``<data_root>/reports``.

    Args:
        store: Built Sharadar store.
        fmp: Open FMP store; without it only the adjustment check runs.
        limit: Limit the vendor check to this many pairs.

    Returns:
        The summary.
    """
    reports = store.data_root / config.REPORTS_DIR
    reports.mkdir(parents=True, exist_ok=True)
    adjustment = adjustment_check(store)
    adjustment.to_csv(reports / "adjustment_check.csv", index=False)
    vendor = vendor_check(store, fmp, limit) if fmp is not None else None
    if vendor is not None:
        vendor.to_csv(reports / "vendor_check.csv", index=False)
    summary = summarize(adjustment, vendor)
    Path(reports / "crosscheck_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    _logger.info("Cross-check written to %s", reports)
    return summary
