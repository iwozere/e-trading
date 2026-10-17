"""
``fmp calibrate-usage --dashboard-gb X`` (spec Appendix A Q2): is the 50 GB cap counted on wire or decoded bytes?

FMP serves gzip, so decoded bytes are about 7× the wire bytes (probe F10). The owner reads "bandwidth used" for
the trailing window from the FMP dashboard; this compares it with what this machine recorded in the same window
(every dump's ``bandwidth_log`` plus both probes' ledgers). The basis whose total is within a factor of the
dashboard, and at least twice as close as the other, is recorded as ``settings.bandwidth.basis`` in
``probe_results.json`` together with ``scale`` (dashboard bytes per local byte), so ``fmp plan``/``run`` budget
against what FMP actually counts. Otherwise nothing changes.

Run it once enough traffic has accumulated for the dashboard's precision (a few GB, e.g. after the daily-price
block). Other programs using the same key (P22's client, ``FMPDataDownloader``) count on the dashboard but not
here, so the dashboard can only be higher; that is reported.
"""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

GB = 1024**3
#: The dashboard meter rounds and lags (it showed 4 MB while its own per-endpoint sizes summed to ~100 MB).
MIN_DASHBOARD_GB = 0.5
LEDGERS = ("probe-free/ledger.jsonl", "probe-paid/ledger.jsonl")


@dataclass(frozen=True)
class LocalUsage:
    """Bytes this machine recorded in the window."""

    wire: int
    decoded: int
    sources: tuple[str, ...]
    ledger_lines_without_decoded: int = 0


@dataclass(frozen=True)
class Calibration:
    """The verdict."""

    dashboard_gb: float
    wire_gb: float
    decoded_gb: float
    wire_error: float
    decoded_error: float
    basis: Optional[str]
    message: str
    scale: Optional[float] = None


def local_usage(data_root: Path, window_days: int = 30, now: Optional[datetime] = None) -> LocalUsage:
    """
    Sum the bytes recorded in the trailing window: every manifest's ``bandwidth_log`` and the probe ledgers.

    Args:
        data_root: Pipeline data root.
        window_days: Trailing window (the cap's).
        now: Window end (default: now, UTC).

    Returns:
        Wire and decoded totals and the files they came from.
    """
    cutoff = ((now or datetime.now(timezone.utc)) - timedelta(days=window_days)).isoformat()
    wire = decoded = missing = 0
    sources = []
    for manifest in sorted((data_root / "manifests").glob("*.sqlite")) if (data_root / "manifests").is_dir() else []:
        # closing(): a sqlite3 connection used as a context manager only commits, it is not closed (an open
        # handle keeps the file locked on Windows).
        with closing(sqlite3.connect(f"file:{manifest.as_posix()}?mode=ro", uri=True)) as db:
            row = db.execute("SELECT COALESCE(SUM(bytes_wire), 0), COALESCE(SUM(bytes_decoded), 0) "
                             "FROM bandwidth_log WHERE ts >= ?", (cutoff,)).fetchone()
        wire, decoded = wire + int(row[0]), decoded + int(row[1])
        sources.append(manifest.name)
    for rel in LEDGERS:
        path = data_root / rel
        if not path.exists():
            continue
        sources.append(rel)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if str(entry.get("at", "")) < cutoff:
                continue
            wire += int(entry.get("bytes_wire", 0))
            if "bytes_decoded" in entry:
                decoded += int(entry["bytes_decoded"])
            else:
                missing += 1
    return LocalUsage(wire, decoded, tuple(sources), missing)


def _distance(dashboard_gb: float, local_gb: float) -> float:
    """``|ln(dashboard / local)|``: how many times apart the two figures are, symmetric in both directions."""
    return abs(math.log(dashboard_gb / local_gb)) if local_gb > 0 else math.inf


def calibrate(dashboard_gb: float, usage: LocalUsage, tolerance: float = 0.5,
              min_dashboard_gb: float = MIN_DASHBOARD_GB) -> Calibration:
    """
    Decide the counting basis and the dashboard-to-local ratio (``scale``) from the dashboard value.

    The basis whose total is within a factor ``1 + tolerance`` of the dashboard, and at least twice as close (in
    log terms) as the other, wins; ``scale`` is then ``dashboard / local`` for it. FMP counts about 0.75 decoded
    bytes per decoded byte recorded here (probably JSON without the indentation it sends, D77), far from wire.

    Args:
        dashboard_gb: "Bandwidth used" from the FMP dashboard for the same window, in GB.
        usage: This machine's totals.
        tolerance: Largest relative distance accepted for the winning basis (0.5 = within 1.5×).
        min_dashboard_gb: Smallest dashboard figure trusted (the meter rounds and lags).

    Returns:
        The verdict; ``basis`` and ``scale`` are ``None`` when inconclusive.

    Raises:
        ValueError: A non-positive dashboard value.
    """
    if dashboard_gb <= 0:
        raise ValueError("--dashboard-gb must be positive")
    wire_gb, decoded_gb = usage.wire / GB, usage.decoded / GB
    wire_error, decoded_error = _distance(dashboard_gb, wire_gb), _distance(dashboard_gb, decoded_gb)
    best, best_gb, best_error, other_error = (("wire", wire_gb, wire_error, decoded_error) if wire_error <= decoded_error
                                              else ("decoded", decoded_gb, decoded_error, wire_error))
    basis: Optional[str] = None
    scale: Optional[float] = None
    recorded = f"{dashboard_gb:.2f} GB on the dashboard vs {wire_gb:.2f} GB wire / {decoded_gb:.2f} GB decoded recorded here"
    if usage.wire == 0:
        message = "nothing recorded locally in the window: run a download first"
    elif dashboard_gb < min_dashboard_gb:
        message = f"inconclusive: {recorded}; wait until the dashboard shows at least {min_dashboard_gb:g} GB"
    elif best_error <= math.log1p(tolerance) and other_error >= 2 * best_error:
        basis, scale = best, round(dashboard_gb / best_gb, 3)
        message = f"the dashboard counts {best} bytes, {scale:g} per byte recorded here ({recorded})"
    else:
        message = f"inconclusive: {recorded}; download more and re-run"
    if dashboard_gb > max(wire_gb, decoded_gb) * (1 + tolerance):
        message += ("; the dashboard is higher than both, so other programs probably use the same key "
                    "(P22, FMPDataDownloader)")
    return Calibration(dashboard_gb, round(wire_gb, 4), round(decoded_gb, 4), round(wire_error, 4),
                       round(decoded_error, 4), basis, message, scale)
