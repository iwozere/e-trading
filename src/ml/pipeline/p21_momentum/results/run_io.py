"""
P21 Momentum — Dated-run-folder I/O (docs/pipeline-specification.md §3).

One place to get the dated-folder-vs-``_state/`` split right: a job script
should never touch ``json.dump``/``Path`` directly for its primary outputs —
it calls the typed helper here instead.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

from src.ml.pipeline.p21_momentum.config import RESULTS_DIR
from src.ml.pipeline.p21_momentum.schemas import DailyMarkSnapshot, Position, SignalRow, TargetPosition
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def run_dir_for(run_date: date, results_dir: Path = RESULTS_DIR) -> Path:
    """Return results/p21_momentum/YYYY-MM-DD/, creating it if absent."""
    d = results_dir / run_date.isoformat()
    d.mkdir(parents=True, exist_ok=True)
    return d


def already_processed(run_date: date, primary_output_filename: str, results_dir: Path = RESULTS_DIR) -> bool:
    """
    Idempotency check every job runs first (spec §3 "Idempotency").

    Args:
        run_date: This run's date.
        primary_output_filename: e.g. "targets.json" for monthly_rebalance,
            "positions.json" for monthly_execute, "daily_mark.json" for daily_mark.
        results_dir: Root results dir (overridable for tests).

    Returns:
        True if results/p21_momentum/<run_date>/<primary_output_filename>
        already exists. Callers should no-op with "SKIP: already processed"
        unless --force is passed.
    """
    return (results_dir / run_date.isoformat() / primary_output_filename).exists()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    _logger.info("Wrote %s", path)


def write_universe(run_date: date, payload: Dict[str, Any], results_dir: Path = RESULTS_DIR) -> None:
    """Write universe.json (already-built payload, see data/universe.py's universe_to_json())."""
    _write_json(run_dir_for(run_date, results_dir) / "universe.json", payload)


def write_signals(run_date: date, rows: List[SignalRow], results_dir: Path = RESULTS_DIR) -> None:
    """Write signals.json — the full ranked signal table, all survivors and non-survivors alike."""
    payload = {"as_of": run_date.isoformat(), "signals": [r.to_dict() for r in rows]}
    _write_json(run_dir_for(run_date, results_dir) / "signals.json", payload)


def write_targets(run_date: date, targets: List[TargetPosition], results_dir: Path = RESULTS_DIR) -> None:
    """Write targets.json — the pre-execution target portfolio."""
    payload = {"as_of": run_date.isoformat(), "targets": [t.to_dict() for t in targets]}
    _write_json(run_dir_for(run_date, results_dir) / "targets.json", payload)


def read_targets(run_date: date, results_dir: Path = RESULTS_DIR) -> List[TargetPosition]:
    """Read targets.json written by a prior monthly_rebalance run."""
    path = run_dir_for(run_date, results_dir) / "targets.json"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return [TargetPosition.from_dict(t) for t in payload.get("targets", [])]


def write_positions(run_date: date, positions: List[Position], results_dir: Path = RESULTS_DIR) -> None:
    """Write positions.json — post-execution snapshot."""
    payload = {"as_of": run_date.isoformat(), "positions": [p.to_dict() for p in positions]}
    _write_json(run_dir_for(run_date, results_dir) / "positions.json", payload)


def write_daily_mark(run_date: date, snapshot: DailyMarkSnapshot, results_dir: Path = RESULTS_DIR) -> None:
    """Write daily_mark.json — today's NAV/high-water/anomaly snapshot."""
    _write_json(run_dir_for(run_date, results_dir) / "daily_mark.json", snapshot.to_dict())


def write_report(run_date: date, markdown: str, results_dir: Path = RESULTS_DIR) -> Path:
    """Write report.md — the monthly report (spec §12). Returns the path written."""
    path = run_dir_for(run_date, results_dir) / "report.md"
    path.write_text(markdown, encoding="utf-8")
    _logger.info("Wrote %s", path)
    return path


def append_regime_history(entry: Dict[str, Any], path: Path) -> None:
    """
    Append one regime entry to _state/regime_history.json (a JSON array, not JSONL).

    Args:
        entry: dataclasses.asdict()-compatible dict (e.g. RegimeResult via asdict()).
        path: Path to _state/regime_history.json.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            history = json.load(f)
    history.append(entry if isinstance(entry, dict) else asdict(entry))
    with path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, sort_keys=True)
    _logger.info("Appended regime history entry to %s (%d total)", path, len(history))


def read_regime_history(path: Path) -> List[Dict[str, Any]]:
    """Read the full _state/regime_history.json array. Empty list if absent."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def append_nav_row(row: Dict[str, Any], path: Path) -> None:
    """
    Append one row to _state/nav_daily.csv, writing the header on first write.

    Args:
        row: {"date": ..., "nav_a": ..., "nav_b": ..., "nav_c": ..., "nav_d": ..., "nav_e": ...}.
        path: Path to _state/nav_daily.csv.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    fieldnames = ["date", "nav_a", "nav_b", "nav_c", "nav_d", "nav_e"]
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
